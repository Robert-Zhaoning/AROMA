import hashlib
import json
import sys
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

META = Path(
    "data/proc_count_tsg_prospective_v1/metadata.jsonl"
)

ACTION = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "action_manifest_v1/"
    "tsg_prospective_action_manifest_v1.csv"
)

EXPECTED_ACTION_SHA = (
    "0c4f3cc02e5cde5661612e7f6ee1dd5367434b0e1"
    "cd9bdfab99b5b27aae11b04"
)

NAMESPACE = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "namespace_v1/"
    "tsg_prospective_namespace_v1.csv"
)

EXPECTED_NAMESPACE_SHA = (
    "295a79870c0d1762da1c8fe36774b9122edcbbb8ca"
    "23503a32cd3cef6d29b278"
)

GRAD_ROOT = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "gradient_collection_v1"
)

EXPECTED_GRAD_SHA = (
    "e5f932a1266860549243be49e142af6ae6f83b31cd8"
    "be56570c932d567559f5b"
)

EXPECTED_H_SHA = (
    "c79f596de6fa348d0ce3065a31397e82790cd5afbc6"
    "194067e3aa3bb76e9c970"
)

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/U4_primary.npy"
)

CONTROLLER_BUNDLE = Path(
    "outputs/proc_count_causal_v2/"
    "controller/final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

UTILITY_ROUTER = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_final_v1/"
    "aroma_utility_router_v1.joblib"
)

EXPECTED_UTILITY_ROUTER_SHA = (
    "4192129c525822fed7ab9d013dc3d564981f48faf81"
    "f28b3011a4be1a2168563"
)

OUT = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "utility_features_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = OUT / "partial_features.jsonl"

FINAL = (
    OUT / "tsg_utility_features_v1.csv"
)

METADATA = (
    OUT / "metadata.json"
)

EXPECTED_N = 500
DUMMY_GT = 1


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

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


def append_jsonl(path, rec):
    with Path(path).open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(rec)
            + "\n"
        )

        f.flush()


# ============================================================
# HARD ARTIFACT AUDIT
# ============================================================

checks = [
    (
        ACTION,
        EXPECTED_ACTION_SHA,
        "action manifest",
    ),
    (
        NAMESPACE,
        EXPECTED_NAMESPACE_SHA,
        "namespace",
    ),
    (
        GRAD_ROOT / "gradient_matrix.npy",
        EXPECTED_GRAD_SHA,
        "gradient matrix",
    ),
    (
        GRAD_ROOT / "pooled_h_matrix.npy",
        EXPECTED_H_SHA,
        "pooled-h matrix",
    ),
    (
        UTILITY_ROUTER,
        EXPECTED_UTILITY_ROUTER_SHA,
        "frozen utility router",
    ),
]

for path, expected, name in checks:

    if sha256(path) != expected:
        raise RuntimeError(
            f"{name} SHA256 mismatch."
        )

print(
    "PASS: frozen external-validation artifacts verified"
)


# ============================================================
# DATA / IDS
# ============================================================

samples = load_jsonl(
    META
)

samples = sorted(
    samples,
    key=lambda r: str(
        r["sample_id"]
    ),
)

if len(samples) != EXPECTED_N:
    raise RuntimeError(
        f"Expected N=500, got {len(samples)}"
    )

raw_ids = np.array(
    [
        str(r["sample_id"])
        for r in samples
    ],
    dtype=str,
)

sample_map = {
    str(r["sample_id"]):
        r
    for r in samples
}


actions = pd.read_csv(
    ACTION
).sort_values(
    "raw_sample_id"
).reset_index(
    drop=True
)

actions[
    "raw_sample_id"
] = (
    actions[
        "raw_sample_id"
    ].astype(str)
)

actions[
    "cohort_uid"
] = (
    actions[
        "cohort_uid"
    ].astype(str)
)

if not np.array_equal(
    actions[
        "raw_sample_id"
    ].to_numpy(),
    raw_ids,
):
    raise RuntimeError(
        "Action / metadata ordering mismatch."
    )


namespace = pd.read_csv(
    NAMESPACE
).sort_values(
    "raw_sample_id"
).reset_index(
    drop=True
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

if not np.array_equal(
    namespace[
        "raw_sample_id"
    ].to_numpy(),
    raw_ids,
):
    raise RuntimeError(
        "Namespace / metadata ordering mismatch."
    )

if not np.array_equal(
    namespace[
        "cohort_uid"
    ].to_numpy(),
    actions[
        "cohort_uid"
    ].to_numpy(),
):
    raise RuntimeError(
        "Namespace / action UID mismatch."
    )


# ============================================================
# MECHANISTIC FEATURES
# ============================================================

G = np.load(
    GRAD_ROOT / "gradient_matrix.npy"
).astype(np.float64)

H = np.load(
    GRAD_ROOT / "pooled_h_matrix.npy"
).astype(np.float64)

G_IDS = (
    np.load(
        GRAD_ROOT / "sample_ids.npy",
        allow_pickle=True,
    )
    .astype(str)
)

U4 = np.load(
    U4_PATH
).astype(np.float64)

if G.shape != (500, 128):
    raise RuntimeError(
        f"Bad G shape: {G.shape}"
    )

if H.shape != (500, 128):
    raise RuntimeError(
        f"Bad H shape: {H.shape}"
    )

if not np.array_equal(
    G_IDS,
    raw_ids,
):
    raise RuntimeError(
        "Gradient ID mismatch."
    )

P = U4 @ U4.T

PG = G @ P
PH = H @ P

g_norm = np.linalg.norm(
    G,
    axis=1,
)

h_norm = np.linalg.norm(
    H,
    axis=1,
)

pg_norm = np.linalg.norm(
    PG,
    axis=1,
)

ph_norm = np.linalg.norm(
    PH,
    axis=1,
)

den = (
    pg_norm
    *
    ph_norm
)

cos_pug_puh = np.divide(
    np.sum(
        PG * PH,
        axis=1,
    ),
    den,
    out=np.zeros(
        EXPECTED_N,
        dtype=np.float64,
    ),
    where=den > 1e-30,
)

gradient_capture = np.divide(
    pg_norm,
    g_norm,
    out=np.zeros_like(pg_norm),
    where=g_norm > 1e-30,
)

activation_capture = np.divide(
    ph_norm,
    h_norm,
    out=np.zeros_like(ph_norm),
    where=h_norm > 1e-30,
)

sa_sensitivity = (
    h_norm
    *
    pg_norm
)


# ============================================================
# FROZEN CONTROLLER SCHEMA
# ============================================================

bundle = joblib.load(
    CONTROLLER_BUNDLE
)

feature_names = list(
    bundle[
        "feature_names"
    ]
)

if len(feature_names) != 39:
    raise RuntimeError(
        "Expected 39 frozen controller features."
    )


# ============================================================
# RESUME
# ============================================================

records = load_jsonl(
    PARTIAL
)

if records:

    partial_ids = [
        str(r["raw_sample_id"])
        for r in records
    ]

    if partial_ids != raw_ids[
        :len(records)
    ].tolist():

        raise RuntimeError(
            "Partial archive is not exact prefix."
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
# ARCHIVE
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
    desc="TSG external utility features",
):

    sid = str(
        raw_ids[idx]
    )

    sample = sample_map[
        sid
    ]

    image = (
        Image.open(
            sample[
                "image_path"
            ]
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

    with torch.inference_mode():

        state = score_state(
            model,
            inputs,
            numeral_ids,
            DUMMY_GT,
        )

    (
        X,
        feature_values,
    ) = state_to_features(
        state,
        feature_names,
    )

    (
        selected_alpha,
        selected_score,
        _
    ) = controller_decision(
        bundle,
        X,
    )

    archived = actions.iloc[
        idx
    ]

    baseline_prediction = int(
        state[
            "best_numeral"
        ]
    )

    if (
        baseline_prediction
        != int(
            archived[
                "baseline_prediction"
            ]
        )
    ):
        raise RuntimeError(
            f"{sid}: baseline mismatch."
        )

    if not np.isclose(
        float(selected_alpha),
        float(
            archived[
                "selected_alpha"
            ]
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{sid}: alpha mismatch."
        )

    if not np.isclose(
        float(selected_score),
        float(
            archived[
                "selected_score"
            ]
        ),
        atol=1e-8,
        rtol=1e-6,
    ):
        raise RuntimeError(
            f"{sid}: score mismatch."
        )

    gt = int(
        sample[
            "ground_truth"
        ]
    )

    rec = {
        "cohort":
            "tsg_prospective_v1",

        "raw_sample_id":
            sid,

        "cohort_uid":
            str(
                archived[
                    "cohort_uid"
                ]
            ),

        "ground_truth":
            gt,

        "baseline_prediction":
            baseline_prediction,

        "baseline_correct":
            int(
                baseline_prediction
                == gt
            ),

        "selected_alpha":
            float(
                selected_alpha
            ),

        "selected_score":
            float(
                selected_score
            ),

        "pg_norm":
            float(
                pg_norm[idx]
            ),

        "h_norm":
            float(
                h_norm[idx]
            ),

        "ph_norm":
            float(
                ph_norm[idx]
            ),

        "g_norm":
            float(
                g_norm[idx]
            ),

        "sa_sensitivity":
            float(
                sa_sensitivity[idx]
            ),

        "cos_pug_puh":
            float(
                cos_pug_puh[idx]
            ),

        "gradient_capture":
            float(
                gradient_capture[idx]
            ),

        "activation_capture":
            float(
                activation_capture[idx]
            ),

        "controller_features":
            {
                name:
                    float(
                        feature_values[
                            name
                        ]
                    )
                for name in feature_names
            },
    }

    append_jsonl(
        PARTIAL,
        rec,
    )

    records.append(
        rec
    )


# ============================================================
# FLATTEN
# ============================================================

if len(records) != 500:
    raise RuntimeError(
        "Final N mismatch."
    )

flat = []

for rec in records:

    row = {
        k: v
        for k, v in rec.items()
        if k != "controller_features"
    }

    for name in feature_names:

        row[
            f"ctrl__{name}"
        ] = (
            rec[
                "controller_features"
            ][
                name
            ]
        )

    flat.append(
        row
    )

df = pd.DataFrame(
    flat
)

ctrl_cols = [
    f"ctrl__{x}"
    for x in feature_names
]

if len(ctrl_cols) != 39:
    raise RuntimeError(
        "Controller feature count mismatch."
    )

if not np.isfinite(
    df[
        ctrl_cols
    ].to_numpy(
        dtype=np.float64
    )
).all():
    raise RuntimeError(
        "Non-finite controller feature."
    )

df.to_csv(
    FINAL,
    index=False,
)


metadata = {
    "experiment":
        "tsg_external_utility_features_v1",

    "n":
        500,

    "controller_feature_count":
        39,

    "utility_router_sha256":
        sha256(
            UTILITY_ROUTER
        ),

    "action_manifest_sha256":
        sha256(
            ACTION
        ),

    "gradient_matrix_sha256":
        sha256(
            GRAD_ROOT
            / "gradient_matrix.npy"
        ),

    "pooled_h_matrix_sha256":
        sha256(
            GRAD_ROOT
            / "pooled_h_matrix.npy"
        ),

    "feature_csv_sha256":
        sha256(
            FINAL
        ),
}

METADATA.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 92)
print(
    "TSG EXTERNAL UTILITY FEATURE ARCHIVE COMPLETE"
)
print("=" * 92)

print(
    "N:",
    len(df),
)

print(
    "Controller features:",
    len(ctrl_cols),
)

print(
    "Baseline:",
    f"{df['baseline_correct'].sum()}/500",
    f"({df['baseline_correct'].mean():.6f})",
)

print(
    "Upward samples:",
    int(
        (
            df[
                "selected_alpha"
            ]
            > 1.0
        ).sum()
    ),
)

print()
print(
    "Feature CSV SHA256:",
    sha256(FINAL),
)

print(
    "Metadata SHA256:",
    sha256(METADATA),
)

print("=" * 92)
