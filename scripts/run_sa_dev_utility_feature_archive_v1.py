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

ACTION = Path(
    "outputs/proc_count_sa_dev_v1/"
    "action_manifest_v1/"
    "sa_vs_mn_action_manifest_v1.csv"
)

EXPECTED_ACTION_SHA256 = (
    "909577217bc38825eab33f03f1ce5633"
    "dab61815bbd83468a30dc4bac6e52021"
)

GRAD_ROOT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_gradient_collection_v1"
)

U4_PATH = Path(
    "outputs/aroma2/"
    "csa_gate_a_v3/"
    "U4_primary.npy"
)

CONTROLLER_BUNDLE = Path(
    "outputs/proc_count_causal_v2/"
    "controller/final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

OUT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_features_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = OUT / "partial_features.jsonl"

FINAL_CSV = (
    OUT / "sa_dev_utility_features_v1.csv"
)

FEATURE_NAMES_JSON = (
    OUT / "frozen_feature_names.json"
)

METADATA_JSON = (
    OUT / "metadata.json"
)

EXPECTED_N = 500

DUMMY_GT = 1

COHORT = "sa_dev_v1"


# ============================================================
# HELPERS
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

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


def append_jsonl(
    path,
    obj,
):

    with Path(path).open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(obj)
            + "\n"
        )

        f.flush()


# ============================================================
# FROZEN ARTIFACT AUDIT
# ============================================================

if (
    sha256(PROTOCOL)
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Protocol SHA256 mismatch."
    )

if (
    sha256(ACTION)
    != EXPECTED_ACTION_SHA256
):
    raise RuntimeError(
        "Action manifest SHA256 mismatch."
    )

for p in [
    META,
    GRAD_ROOT / "gradient_matrix.npy",
    GRAD_ROOT / "pooled_h_matrix.npy",
    GRAD_ROOT / "sample_ids.npy",
    U4_PATH,
    CONTROLLER_BUNDLE,
]:
    if not p.exists():
        raise FileNotFoundError(p)

print(
    "PASS: frozen SA-dev artifacts verified"
)


# ============================================================
# DATA
# ============================================================

samples = load_jsonl(
    META
)

if len(samples) != EXPECTED_N:
    raise RuntimeError(
        f"Expected N=500; got {len(samples)}"
    )

samples = sorted(
    samples,
    key=lambda r: str(
        r["sample_id"]
    ),
)

sample_ids = np.array(
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
)

actions[
    "sample_id"
] = (
    actions[
        "sample_id"
    ].astype(str)
)

if len(actions) != 500:
    raise RuntimeError(
        "Action manifest N mismatch."
    )

if not np.array_equal(
    actions[
        "sample_id"
    ].to_numpy(),
    sample_ids,
):
    raise RuntimeError(
        "Action / dataset ordering mismatch."
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

if U4.shape != (128, 4):
    raise RuntimeError(
        f"Bad U4 shape: {U4.shape}"
    )

if not np.array_equal(
    G_IDS,
    sample_ids,
):
    raise RuntimeError(
        "Gradient / dataset ordering mismatch."
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
    out=np.zeros_like(
        pg_norm
    ),
    where=g_norm > 1e-30,
)

activation_capture = np.divide(
    ph_norm,
    h_norm,
    out=np.zeros_like(
        ph_norm
    ),
    where=h_norm > 1e-30,
)

sa_sensitivity = (
    h_norm
    *
    pg_norm
)


# ============================================================
# FROZEN CONTROLLER
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

feature_names = list(
    bundle[
        "feature_names"
    ]
)

if len(feature_names) != 39:
    raise RuntimeError(
        f"Expected 39 feature names; "
        f"got {len(feature_names)}"
    )


FEATURE_NAMES_JSON.write_text(
    json.dumps(
        {
            "n_features":
                39,

            "feature_names":
                feature_names,
        },
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "PASS: frozen 39-feature schema loaded"
)


# ============================================================
# RESUME
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
            "Partial feature archive is not "
            "an exact dataset prefix."
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
# FEATURE ARCHIVE
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
    desc="SA-dev utility features",
):

    sid = str(
        sample_ids[idx]
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
    # GT-FREE BASELINE STATE
    # ========================================================

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


    if X.shape != (1, 39):
        raise RuntimeError(
            f"{sid}: bad feature shape "
            f"{X.shape}"
        )


    # ========================================================
    # RECONSTRUCT FROZEN ROUTER DECISION
    # ========================================================

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
            f"{sid}: baseline prediction "
            "does not reproduce action manifest."
        )


    if not np.isclose(
        selected_alpha,
        float(
            archived[
                "selected_alpha"
            ]
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{sid}: selected_alpha "
            "does not reproduce action manifest."
        )


    if not np.isclose(
        selected_score,
        float(
            archived[
                "selected_score"
            ]
        ),
        atol=1e-8,
        rtol=1e-6,
    ):
        raise RuntimeError(
            f"{sid}: selected_score "
            "does not reproduce action manifest."
        )


    # Ground truth is used only as downstream supervision,
    # never inside feature construction.
    ground_truth = int(
        sample[
            "ground_truth"
        ]
    )


    rec = {
        "cohort":
            COHORT,

        "sample_id":
            sid,

        "cohort_uid":
            f"{COHORT}::{sid}",

        "ground_truth":
            ground_truth,

        "baseline_prediction":
            baseline_prediction,

        "baseline_correct":
            int(
                baseline_prediction
                == ground_truth
            ),

        "selected_alpha":
            selected_alpha,

        "selected_score":
            selected_score,

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
# FINALIZE FLAT TABLE
# ============================================================

if len(records) != 500:
    raise RuntimeError(
        f"Final N mismatch: "
        f"{len(records)}"
    )


flat_rows = []

for rec in records:

    row = {
        "cohort":
            rec[
                "cohort"
            ],

        "sample_id":
            rec[
                "sample_id"
            ],

        "cohort_uid":
            rec[
                "cohort_uid"
            ],

        "ground_truth":
            rec[
                "ground_truth"
            ],

        "baseline_prediction":
            rec[
                "baseline_prediction"
            ],

        "baseline_correct":
            rec[
                "baseline_correct"
            ],

        "selected_alpha":
            rec[
                "selected_alpha"
            ],

        "selected_score":
            rec[
                "selected_score"
            ],

        "pg_norm":
            rec[
                "pg_norm"
            ],

        "h_norm":
            rec[
                "h_norm"
            ],

        "ph_norm":
            rec[
                "ph_norm"
            ],

        "g_norm":
            rec[
                "g_norm"
            ],

        "sa_sensitivity":
            rec[
                "sa_sensitivity"
            ],

        "cos_pug_puh":
            rec[
                "cos_pug_puh"
            ],

        "gradient_capture":
            rec[
                "gradient_capture"
            ],

        "activation_capture":
            rec[
                "activation_capture"
            ],
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


    flat_rows.append(
        row
    )


df = pd.DataFrame(
    flat_rows
)


if len(df) != 500:
    raise RuntimeError(
        "Final feature-table N mismatch."
    )


if df[
    "cohort_uid"
].nunique() != 500:
    raise RuntimeError(
        "cohort_uid uniqueness failure."
    )


controller_cols = [
    f"ctrl__{name}"
    for name in feature_names
]


if len(
    controller_cols
) != 39:

    raise RuntimeError(
        "Controller feature-column count mismatch."
    )


if not np.isfinite(
    df[
        controller_cols
    ].to_numpy(
        dtype=np.float64
    )
).all():

    raise RuntimeError(
        "Non-finite archived controller features."
    )


df.to_csv(
    FINAL_CSV,
    index=False,
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "experiment":
        "sa_dev_utility_feature_archive_v1",

    "n":
        500,

    "cohort":
        COHORT,

    "model_id":
        MODEL_ID,

    "model_revision":
        MODEL_REVISION,

    "controller_feature_count":
        39,

    "controller_feature_names":
        feature_names,

    "feature_policy":
        "39 frozen GT-free controller features "
        "plus archived mechanistic features",

    "supervision_policy":
        "ground truth retained only as downstream "
        "training/evaluation label",

    "protocol_sha256":
        sha256(
            PROTOCOL
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

    "U4_sha256":
        sha256(
            U4_PATH
        ),

    "feature_names_sha256":
        sha256(
            FEATURE_NAMES_JSON
        ),

    "utility_feature_csv_sha256":
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
    "SA-DEV UTILITY FEATURE ARCHIVE COMPLETE"
)
print("=" * 96)

print(
    "N:",
    len(df),
)

print(
    "Controller features:",
    len(
        controller_cols
    ),
)

print(
    "Baseline accuracy:",
    f"{df['baseline_correct'].sum()}/500",
    f"({df['baseline_correct'].mean():.6f})",
)

print(
    "Unique cohort UIDs:",
    df[
        "cohort_uid"
    ].nunique(),
)

print()
print(
    "Feature CSV:",
    FINAL_CSV,
)

print(
    "Feature CSV SHA256:",
    sha256(
        FINAL_CSV
    ),
)

print(
    "Metadata SHA256:",
    sha256(
        METADATA_JSON
    ),
)

print("=" * 96)
