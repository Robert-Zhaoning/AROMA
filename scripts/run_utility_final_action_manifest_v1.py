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
    "data/proc_count_utility_final_v1/"
    "metadata.jsonl"
)

EXPECTED_DATA_META_SHA256 = (
    "700b20fe99944c9d25cb4b80873156b796b4bba90f4af"
    "743427a4195bb040231"
)

COHORT_FREEZE = Path(
    "outputs/proc_count_utility_final_v1/"
    "freeze_v1/"
    "utility_final_cohort_freeze_v1.json"
)

EXPECTED_COHORT_FREEZE_SHA256 = (
    "719915ee7fc139e70b936a9be41aaa5b3e2541d93d52d"
    "fff9dd0a7224006e79b"
)

FINAL_PROTOCOL = Path(
    "outputs/aroma2/"
    "utility_router_final_confirmation_v1/"
    "final_confirmation_protocol.json"
)

EXPECTED_FINAL_PROTOCOL_SHA256 = (
    "7a37d80f53e86944631571799d55b8eac8d4873ce74cf"
    "1f9a84d11c118654922"
)

NAMESPACE_CSV = Path(
    "outputs/proc_count_utility_final_v1/"
    "namespace_v1/"
    "utility_final_namespace_v1.csv"
)

EXPECTED_NAMESPACE_SHA256 = (
    "a5a6af6dd3f5103f6586de1e79b7f76570222b202bb14"
    "c98174f9283f3017937"
)

NAMESPACE_MANIFEST = Path(
    "outputs/proc_count_utility_final_v1/"
    "namespace_v1/"
    "utility_final_namespace_v1.json"
)

EXPECTED_NAMESPACE_MANIFEST_SHA256 = (
    "690fca1cf848d2d5d99ab34e9d05723b4b451294f6ecb"
    "5564ad17090093c1f3a"
)

CONTROLLER_BUNDLE = Path(
    "outputs/proc_count_causal_v2/"
    "controller/final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

CONTROLLER_MANIFEST = Path(
    "configs/"
    "aroma_cardinality_controller_frozen.json"
)

CONTROLLER_FREEZE_RECORD = Path(
    "configs/"
    "aroma_controller_freeze_record.json"
)

OUT_DIR = Path(
    "outputs/proc_count_utility_final_v1/"
    "action_manifest_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = (
    OUT_DIR
    / "partial_actions.jsonl"
)

FINAL_CSV = (
    OUT_DIR
    / "utility_final_action_manifest_v1.csv"
)

METADATA_JSON = (
    OUT_DIR
    / "utility_final_action_manifest_v1_metadata.json"
)

EXPECTED_N = 1000

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
# HARD ARTIFACT AUDIT
# ============================================================

checks = [
    (
        DATA_META,
        EXPECTED_DATA_META_SHA256,
        "dataset metadata",
    ),
    (
        COHORT_FREEZE,
        EXPECTED_COHORT_FREEZE_SHA256,
        "cohort freeze",
    ),
    (
        FINAL_PROTOCOL,
        EXPECTED_FINAL_PROTOCOL_SHA256,
        "final protocol",
    ),
    (
        NAMESPACE_CSV,
        EXPECTED_NAMESPACE_SHA256,
        "namespace CSV",
    ),
    (
        NAMESPACE_MANIFEST,
        EXPECTED_NAMESPACE_MANIFEST_SHA256,
        "namespace manifest",
    ),
]

for path, expected, name in checks:

    actual = sha256(path)

    if actual != expected:
        raise RuntimeError(
            f"{name} SHA256 mismatch:\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


for path in [
    CONTROLLER_BUNDLE,
    CONTROLLER_MANIFEST,
    CONTROLLER_FREEZE_RECORD,
]:
    if not path.exists():
        raise FileNotFoundError(path)


print(
    "PASS: final frozen artifacts verified"
)


# ============================================================
# DATASET + NAMESPACE
# ============================================================

samples = load_jsonl(
    DATA_META
)

if len(samples) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} samples; "
        f"got {len(samples)}"
    )


samples = sorted(
    samples,
    key=lambda r: str(
        r["sample_id"]
    ),
)


raw_ids = np.array(
    [
        str(r["sample_id"])
        for r in samples
    ],
    dtype=str,
)


namespace = pd.read_csv(
    NAMESPACE_CSV
)

namespace[
    "raw_sample_id"
] = (
    namespace[
        "raw_sample_id"
    ].astype(str)
)

namespace[
    "cohort_uid"
] = (
    namespace[
        "cohort_uid"
    ].astype(str)
)

namespace = (
    namespace
    .sort_values(
        "raw_sample_id"
    )
    .reset_index(
        drop=True
    )
)


if len(namespace) != EXPECTED_N:
    raise RuntimeError(
        "Namespace N mismatch."
    )


if not np.array_equal(
    raw_ids,
    namespace[
        "raw_sample_id"
    ].to_numpy(),
):
    raise RuntimeError(
        "Dataset / namespace ordering mismatch."
    )


uid_map = dict(
    zip(
        namespace[
            "raw_sample_id"
        ],
        namespace[
            "cohort_uid"
        ],
    )
)


print(
    "PASS: dataset and namespace match exactly"
)


# ============================================================
# LOAD FROZEN PROCEDURAL CONTROLLER
# ============================================================

bundle = joblib.load(
    CONTROLLER_BUNDLE
)

if not isinstance(
    bundle,
    dict,
):
    raise RuntimeError(
        "Controller bundle is not a dict."
    )


if "feature_names" not in bundle:
    raise RuntimeError(
        "Controller bundle missing feature_names."
    )


feature_names = list(
    bundle[
        "feature_names"
    ]
)


if len(feature_names) != 39:
    raise RuntimeError(
        f"Expected 39 controller features; "
        f"got {len(feature_names)}"
    )


controller_bundle_sha = sha256(
    CONTROLLER_BUNDLE
)


print(
    "PASS: frozen procedural controller loaded"
)

print(
    "Controller bundle SHA256:",
    controller_bundle_sha,
)

print(
    "Feature count:",
    len(feature_names),
)


# ============================================================
# RESUME
# ============================================================

records = load_jsonl(
    PARTIAL
)


if records:

    partial_ids = [
        str(
            r[
                "raw_sample_id"
            ]
        )
        for r in records
    ]

    expected_prefix = (
        raw_ids[
            :len(records)
        ].tolist()
    )

    if partial_ids != expected_prefix:
        raise RuntimeError(
            "Partial manifest is not "
            "an exact final-cohort prefix."
        )

    for r in records:

        sid = str(
            r[
                "raw_sample_id"
            ]
        )

        if (
            str(
                r[
                    "cohort_uid"
                ]
            )
            !=
            uid_map[sid]
        ):
            raise RuntimeError(
                "Partial cohort_uid mismatch."
            )

    print(
        f"Resuming from {len(records)}/1000"
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
# FINAL FROZEN PROCEDURAL ROUTER
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
    desc="Utility-final frozen procedural router",
):

    sample = samples[
        idx
    ]

    sid = str(
        sample[
            "sample_id"
        ]
    )

    cohort_uid = uid_map[
        sid
    ]


    image = (
        Image.open(
            sample[
                "image_path"
            ]
        )
        .convert(
            "RGB"
        )
    )


    inputs = prepare_inputs(
        processor,
        image,
    )

    inputs = move_inputs(
        inputs,
        model,
    )


    # ========================================================
    # BASELINE / GT-FREE ROUTER INPUT
    # ========================================================

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


    if X.shape != (
        1,
        39,
    ):
        raise RuntimeError(
            f"{sid}: bad feature shape "
            f"{X.shape}"
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


    # ========================================================
    # GT READ ONLY AFTER ROUTER DECISION
    # ========================================================

    ground_truth = int(
        sample[
            "ground_truth"
        ]
    )


    rec = {
        "raw_sample_id":
            sid,

        "cohort_uid":
            cohort_uid,

        "ground_truth":
            ground_truth,

        "baseline_prediction":
            baseline_prediction,

        "selected_alpha":
            selected_alpha,

        "selected_score":
            selected_score,
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
        f"Final N mismatch: "
        f"{len(records)}"
    )


result = pd.DataFrame(
    records
)


if not np.array_equal(
    result[
        "raw_sample_id"
    ].astype(str).to_numpy(),
    raw_ids,
):
    raise RuntimeError(
        "Final sample ordering mismatch."
    )


if (
    result[
        "cohort_uid"
    ].nunique()
    != EXPECTED_N
):
    raise RuntimeError(
        "cohort_uid uniqueness failure."
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
    str(float(alpha)):
        int(
            action_counts.get(
                alpha,
                0,
            )
        )
    for alpha in sorted(
        EXPECTED_ACTIONS
    )
}


metadata = {
    "experiment":
        "utility_final_action_manifest_v1",

    "n":
        EXPECTED_N,

    "model_id":
        MODEL_ID,

    "model_revision":
        MODEL_REVISION,

    "dataset_metadata_sha256":
        sha256(
            DATA_META
        ),

    "cohort_freeze_sha256":
        sha256(
            COHORT_FREEZE
        ),

    "final_protocol_sha256":
        sha256(
            FINAL_PROTOCOL
        ),

    "namespace_csv_sha256":
        sha256(
            NAMESPACE_CSV
        ),

    "namespace_manifest_sha256":
        sha256(
            NAMESPACE_MANIFEST
        ),

    "controller_bundle":
        str(
            CONTROLLER_BUNDLE
        ),

    "controller_bundle_sha256":
        controller_bundle_sha,

    "controller_manifest_sha256":
        sha256(
            CONTROLLER_MANIFEST
        ),

    "controller_freeze_record_sha256":
        sha256(
            CONTROLLER_FREEZE_RECORD
        ),

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
        sha256(
            FINAL_CSV
        ),
}


METADATA_JSON.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 96)
print(
    "UTILITY FINAL ACTION MANIFEST COMPLETE"
)
print("=" * 96)

print(
    "N:",
    len(result),
)

print(
    "Baseline:",
    f"{int(baseline_correct.sum())}/1000",
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
    "Upward samples:",
    int(
        (
            result[
                "selected_alpha"
            ].astype(float)
            > 1.0
        ).sum()
    ),
)

print()
print(
    "Manifest:",
    FINAL_CSV,
)

print(
    "Manifest SHA256:",
    sha256(
        FINAL_CSV
    ),
)

print(
    "Metadata:",
    METADATA_JSON,
)

print(
    "Metadata SHA256:",
    sha256(
        METADATA_JSON
    ),
)

print("=" * 96)
