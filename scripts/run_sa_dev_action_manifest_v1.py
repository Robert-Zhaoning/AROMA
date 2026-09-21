import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

sys.path.insert(0, "scripts")

from run_l18h13_gain_all300 import (
    MODEL_ID,
    prepare_inputs,
    move_inputs,
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)

from run_proc_count_causal_v3_final import (
    state_to_features,
    controller_decision,
)


# ============================================================
# FROZEN CONFIG
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

DATA_META = Path(
    "data/proc_count_sa_dev_v1/metadata.jsonl"
)

PROTOCOL = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_vs_mn_protocol_v1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "39e61d47edbe9625e93e5f18b2c6f5a9"
    "fd9b25e1b19a731b50908d8eb2c61869"
)

CONTROLLER_BUNDLE = Path(
    "outputs/proc_count_causal_v2/"
    "controller/final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

CONTROLLER_MANIFEST = Path(
    "configs/aroma_cardinality_controller_frozen.json"
)

CONTROLLER_FREEZE_RECORD = Path(
    "configs/aroma_controller_freeze_record.json"
)

GRADIENT_IDS = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_gradient_collection_v1/"
    "sample_ids.npy"
)

OUT_DIR = Path(
    "outputs/proc_count_sa_dev_v1/"
    "action_manifest_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = OUT_DIR / "partial_actions.jsonl"

FINAL_CSV = (
    OUT_DIR
    / "sa_vs_mn_action_manifest_v1.csv"
)

METADATA_JSON = (
    OUT_DIR
    / "sa_vs_mn_action_manifest_v1_metadata.json"
)

EXPECTED_N = 500

EXPECTED_ACTIONS = {
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
}

DUMMY_GT = 1


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_jsonl(path):
    path = Path(path)

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def append_jsonl(path, record):
    with Path(path).open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(record)
            + "\n"
        )
        f.flush()


# ============================================================
# FROZEN ARTIFACT AUDIT
# ============================================================

if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError(
        "Frozen SA-vs-MN protocol hash mismatch."
    )

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

if sha256(DATA_META) != protocol[
    "dataset_metadata_sha256"
]:
    raise RuntimeError(
        "Dataset metadata hash mismatch."
    )

for p in [
    CONTROLLER_BUNDLE,
    CONTROLLER_MANIFEST,
    CONTROLLER_FREEZE_RECORD,
    GRADIENT_IDS,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD DATA
# ============================================================

samples = load_jsonl(
    DATA_META
)

if len(samples) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} samples, "
        f"got {len(samples)}"
    )

samples = sorted(
    samples,
    key=lambda x: str(x["sample_id"]),
)

sample_ids = np.array(
    [
        str(x["sample_id"])
        for x in samples
    ],
    dtype=str,
)

gradient_ids = (
    np.load(
        GRADIENT_IDS,
        allow_pickle=True,
    )
    .astype(str)
)

if not np.array_equal(
    sample_ids,
    gradient_ids,
):
    raise RuntimeError(
        "Action-manifest sample order does not "
        "match archived SA gradient order."
    )

print(
    "PASS: N=500 sample IDs exactly match "
    "gradient collection"
)


# ============================================================
# LOAD FROZEN CONTROLLER
# ============================================================

bundle = joblib.load(
    CONTROLLER_BUNDLE
)

if not isinstance(bundle, dict):
    raise RuntimeError(
        "Controller bundle is not a dict."
    )

if "feature_names" not in bundle:
    raise RuntimeError(
        "Controller bundle missing feature_names."
    )

feature_names = list(
    bundle["feature_names"]
)

if len(feature_names) != 39:
    raise RuntimeError(
        f"Expected 39 features; "
        f"got {len(feature_names)}"
    )

print(
    "PASS: frozen controller loaded"
)

print(
    "Controller bundle SHA256:",
    sha256(CONTROLLER_BUNDLE),
)

print(
    "Feature count:",
    len(feature_names),
)


# ============================================================
# RESUME AUDIT
# ============================================================

records = load_jsonl(
    PARTIAL
)

if records:
    partial_ids = [
        str(r["sample_id"])
        for r in records
    ]

    expected_prefix = (
        sample_ids[
            :len(records)
        ].tolist()
    )

    if partial_ids != expected_prefix:
        raise RuntimeError(
            "Partial action manifest is not an "
            "exact prefix of current dataset."
        )

    print(
        f"Resuming from {len(records)}/500"
    )


# ============================================================
# MODEL
# ============================================================

print(
    "Loading processor..."
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
)

numeral_ids = numeral_token_ids(
    processor
)

print(
    "Loading model..."
)

model = (
    MllamaForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
)

model.eval()

print(
    "MODEL LOADED"
)


# ============================================================
# ACTION SELECTION
# ============================================================

start_idx = len(
    records
)

for idx in tqdm(
    range(
        start_idx,
        EXPECTED_N,
    ),
    initial=start_idx,
    total=EXPECTED_N,
    desc="Frozen router actions",
):

    sample = samples[idx]

    sid = str(
        sample["sample_id"]
    )

    image = (
        Image.open(
            sample["image_path"]
        )
        .convert("RGB")
    )

    inputs = prepare_inputs(
        processor,
        image,
    )

    inputs = move_inputs(
        inputs,
        model,
    )

    # --------------------------------------------------------
    # Baseline forward.
    #
    # No ground truth is used for feature extraction or
    # controller action selection.
    # --------------------------------------------------------

    with torch.inference_mode():

        baseline_state = score_state(
            model,
            inputs,
            numeral_ids,
            DUMMY_GT,
        )

    (
        X,
        feature_values,
    ) = state_to_features(
        baseline_state,
        feature_names,
    )

    if np.asarray(X).shape[-1] != 39:
        raise RuntimeError(
            f"{sid}: bad controller feature shape "
            f"{np.asarray(X).shape}"
        )

    (
        selected_alpha,
        selected_score,
        score_map,
    ) = controller_decision(
        bundle,
        X,
    )

    selected_alpha = float(
        selected_alpha
    )

    selected_score = float(
        selected_score
    )

    if selected_alpha not in EXPECTED_ACTIONS:
        raise RuntimeError(
            f"{sid}: unexpected action "
            f"{selected_alpha}"
        )

    baseline_prediction = int(
        baseline_state[
            "best_numeral"
        ]
    )

    # Ground truth is read only AFTER frozen controller action.
    ground_truth = int(
        sample["ground_truth"]
    )

    normalized_score_map = {
        str(float(k)): float(v)
        for k, v in score_map.items()
    }

    rec = {
        "sample_id":
            sid,

        "ground_truth":
            ground_truth,

        "baseline_prediction":
            baseline_prediction,

        "selected_alpha":
            selected_alpha,

        "selected_score":
            selected_score,

        "score_map":
            normalized_score_map,
    }

    append_jsonl(
        PARTIAL,
        rec,
    )

    records.append(
        rec
    )


# ============================================================
# FINALIZE
# ============================================================

if len(records) != EXPECTED_N:
    raise RuntimeError(
        f"Final N mismatch: {len(records)}"
    )

result = pd.DataFrame(
    [
        {
            "sample_id":
                r["sample_id"],

            "ground_truth":
                r["ground_truth"],

            "baseline_prediction":
                r["baseline_prediction"],

            "selected_alpha":
                r["selected_alpha"],

            "selected_score":
                r["selected_score"],
        }
        for r in records
    ]
)

if not np.array_equal(
    result[
        "sample_id"
    ].astype(str).to_numpy(),
    sample_ids,
):
    raise RuntimeError(
        "Final action-manifest ordering mismatch."
    )

result.to_csv(
    FINAL_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

action_counts = Counter(
    result[
        "selected_alpha"
    ].astype(float)
)

baseline_correct = (
    result[
        "baseline_prediction"
    ].astype(int)
    ==
    result[
        "ground_truth"
    ].astype(int)
)

action_distribution = {
    str(float(a)):
        int(
            action_counts.get(
                a,
                0,
            )
        )
    for a in sorted(
        EXPECTED_ACTIONS
    )
}

metadata = {
    "experiment":
        "sa_vs_mn_action_manifest_v1",

    "n":
        EXPECTED_N,

    "model_id":
        MODEL_ID,

    "model_revision":
        MODEL_REVISION,

    "protocol_sha256":
        sha256(PROTOCOL),

    "dataset_metadata_sha256":
        sha256(DATA_META),

    "controller_bundle":
        str(CONTROLLER_BUNDLE),

    "controller_bundle_sha256":
        sha256(CONTROLLER_BUNDLE),

    "controller_manifest_sha256":
        sha256(CONTROLLER_MANIFEST),

    "controller_freeze_record_sha256":
        sha256(CONTROLLER_FREEZE_RECORD),

    "feature_count":
        39,

    "expected_actions":
        sorted(
            EXPECTED_ACTIONS
        ),

    "action_distribution":
        action_distribution,

    "baseline_correct":
        int(
            baseline_correct.sum()
        ),

    "baseline_accuracy":
        float(
            baseline_correct.mean()
        ),

    "action_manifest_sha256":
        sha256(FINAL_CSV),
}

METADATA_JSON.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 88)
print("SA/MN ACTION MANIFEST COMPLETE")
print("=" * 88)

print(
    "N:",
    len(result),
)

print(
    "Baseline:",
    f"{int(baseline_correct.sum())}/500",
    f"({baseline_correct.mean():.6f})",
)

print()
print(
    "Action distribution:"
)

for alpha in sorted(
    EXPECTED_ACTIONS
):
    print(
        f"  alpha={alpha:g}: "
        f"{action_counts.get(alpha, 0)}"
    )

print()
print(
    "Manifest:",
    FINAL_CSV,
)

print(
    "Manifest SHA256:",
    sha256(FINAL_CSV),
)

print(
    "Metadata:",
    METADATA_JSON,
)

print("=" * 88)
