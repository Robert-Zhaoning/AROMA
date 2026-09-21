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

sys.path.insert(
    0,
    "scripts",
)

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
    "data/proc_count_utility_final_v1/"
    "metadata.jsonl"
)

EXPECTED_META_SHA = (
    "700b20fe99944c9d25cb4b80873156b796b4bba90f4af"
    "743427a4195bb040231"
)

ACTION = Path(
    "outputs/proc_count_utility_final_v1/"
    "action_manifest_v1/"
    "utility_final_action_manifest_v1.csv"
)

EXPECTED_ACTION_SHA = (
    "3123d61cfca130376a2b911707ea20927872b1a270ac1"
    "fbd5a852d3e82fc3040"
)

ACTION_FREEZE = Path(
    "outputs/proc_count_utility_final_v1/"
    "action_manifest_v1/freeze/"
    "utility_final_action_manifest_freeze_v1.json"
)

EXPECTED_ACTION_FREEZE_SHA = (
    "4fb05b5d1c91ab3613432ee580126dcd504dd59abf875"
    "eb5a56ec79e4745edef"
)

GRAD_ROOT = Path(
    "outputs/proc_count_utility_final_v1/"
    "gradient_collection_v1"
)

GRAD_FREEZE = (
    GRAD_ROOT
    / "freeze"
    / "utility_final_gradient_freeze_v1.json"
)

EXPECTED_GRAD_FREEZE_SHA = (
    "086422c7a0b24023ee7e3cc4bc2774654cfb539d4ebd9"
    "a7b2ba844f853416ab9"
)

EXPECTED_G_SHA = (
    "ac771aadb34ef1682065435e84a88dd7f7fbc108e3fa"
    "44879a5ae2d536699b3b"
)

EXPECTED_H_SHA = (
    "07767f42678a330d8298acd2f63d9438b8bc8c30af61"
    "a35efc61cb8a019994c1"
)

PROTOCOL = Path(
    "outputs/aroma2/"
    "utility_router_final_confirmation_v1/"
    "final_confirmation_protocol.json"
)

EXPECTED_PROTOCOL_SHA = (
    "7a37d80f53e86944631571799d55b8eac8d4873ce74cf"
    "1f9a84d11c118654922"
)

UTILITY_ROUTER = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_final_v1/"
    "aroma_utility_router_v1.joblib"
)

EXPECTED_UTILITY_ROUTER_SHA = (
    "4192129c525822fed7ab9d013dc3d564981f48faf81f28"
    "b3011a4be1a2168563"
)

CONTROLLER_BUNDLE = Path(
    "outputs/proc_count_causal_v2/"
    "controller/final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

U4_PATH = Path(
    "outputs/aroma2/"
    "csa_gate_a_v3/"
    "U4_primary.npy"
)

DEV_UTILITY_METADATA = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_features_v1/"
    "metadata.json"
)

OUT = Path(
    "outputs/proc_count_utility_final_v1/"
    "utility_features_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = (
    OUT
    / "partial_features.jsonl"
)

FINAL = (
    OUT
    / "utility_final_features_v1.csv"
)

METADATA_OUT = (
    OUT
    / "metadata.json"
)

EXPECTED_N = 294
DUMMY_GT = 1


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
    rec,
):

    with Path(path).open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(rec)
            +
            "\n"
        )

        f.flush()


# ============================================================
# HARD ARTIFACT AUDIT
# ============================================================

checks = [
    (
        META,
        EXPECTED_META_SHA,
        "dataset metadata",
    ),
    (
        ACTION,
        EXPECTED_ACTION_SHA,
        "action manifest",
    ),
    (
        ACTION_FREEZE,
        EXPECTED_ACTION_FREEZE_SHA,
        "action freeze",
    ),
    (
        GRAD_FREEZE,
        EXPECTED_GRAD_FREEZE_SHA,
        "gradient freeze",
    ),
    (
        GRAD_ROOT / "gradient_matrix.npy",
        EXPECTED_G_SHA,
        "gradient matrix",
    ),
    (
        GRAD_ROOT / "pooled_h_matrix.npy",
        EXPECTED_H_SHA,
        "pooled-h matrix",
    ),
    (
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA,
        "final protocol",
    ),
    (
        UTILITY_ROUTER,
        EXPECTED_UTILITY_ROUTER_SHA,
        "utility router",
    ),
]

for path, expected, name in checks:

    actual = sha256(
        path
    )

    if actual != expected:

        raise RuntimeError(
            f"{name} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


dev_meta = json.loads(
    DEV_UTILITY_METADATA.read_text(
        encoding="utf-8"
    )
)

expected_u4_sha = dev_meta[
    "U4_sha256"
]

if sha256(
    U4_PATH
) != expected_u4_sha:

    raise RuntimeError(
        "U4 differs from the one used "
        "for development utility features."
    )


print(
    "PASS: frozen final artifacts verified"
)


# ============================================================
# LOAD FINAL UPWARD POPULATION
# ============================================================

all_records = load_jsonl(
    META
)

record_map = {
    str(
        r[
            "sample_id"
        ]
    ):
        r
    for r in all_records
}


actions = pd.read_csv(
    ACTION
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


up = actions[
    actions[
        "selected_alpha"
    ].astype(float)
    > 1.0
].copy()

up = (
    up
    .sort_values(
        "raw_sample_id"
    )
    .reset_index(
        drop=True
    )
)


if len(
    up
) != EXPECTED_N:

    raise RuntimeError(
        f"Expected upward N={EXPECTED_N}, "
        f"got {len(up)}"
    )


sample_ids = (
    up[
        "raw_sample_id"
    ]
    .astype(str)
    .to_numpy(
        dtype=object
    )
)

cohort_uids = (
    up[
        "cohort_uid"
    ]
    .astype(str)
    .to_numpy(
        dtype=object
    )
)


# ============================================================
# GRADIENT / ACTIVATION FEATURES
# ============================================================

G = np.load(
    GRAD_ROOT
    / "gradient_matrix.npy"
).astype(
    np.float64
)

H = np.load(
    GRAD_ROOT
    / "pooled_h_matrix.npy"
).astype(
    np.float64
)

G_IDS = (
    np.load(
        GRAD_ROOT
        / "sample_ids.npy",
        allow_pickle=True,
    )
    .astype(str)
)

G_UIDS = (
    np.load(
        GRAD_ROOT
        / "cohort_uids.npy",
        allow_pickle=True,
    )
    .astype(str)
)


if G.shape != (
    EXPECTED_N,
    128,
):

    raise RuntimeError(
        f"Bad G shape: "
        f"{G.shape}"
    )


if H.shape != (
    EXPECTED_N,
    128,
):

    raise RuntimeError(
        f"Bad H shape: "
        f"{H.shape}"
    )


if not np.array_equal(
    G_IDS,
    sample_ids.astype(str),
):

    raise RuntimeError(
        "Gradient/sample ordering mismatch."
    )


if not np.array_equal(
    G_UIDS,
    cohort_uids.astype(str),
):

    raise RuntimeError(
        "Gradient/cohort UID ordering mismatch."
    )


U4 = np.load(
    U4_PATH
).astype(
    np.float64
)


if U4.shape != (
    128,
    4,
):

    raise RuntimeError(
        f"Bad U4 shape: "
        f"{U4.shape}"
    )


P = (
    U4
    @
    U4.T
)

PG = (
    G
    @
    P
)

PH = (
    H
    @
    P
)


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
        PG
        *
        PH,
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
# FROZEN PROCEDURAL CONTROLLER SCHEMA
# ============================================================

controller = joblib.load(
    CONTROLLER_BUNDLE
)

feature_names = list(
    controller[
        "feature_names"
    ]
)


if len(
    feature_names
) != 39:

    raise RuntimeError(
        f"Expected 39 features; "
        f"got {len(feature_names)}"
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
        sample_ids[
            :len(
                records
            )
        ]
        .astype(str)
        .tolist()
    )


    if partial_ids != expected_prefix:

        raise RuntimeError(
            "Partial feature archive is not "
            "the exact frozen upward prefix."
        )


    print(
        f"Resuming from "
        f"{len(records)}/{EXPECTED_N}"
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
# GT-FREE FEATURE ARCHIVE
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
    desc="Utility-final GT-free features",
):

    sid = str(
        sample_ids[
            idx
        ]
    )

    uid = str(
        cohort_uids[
            idx
        ]
    )


    sample = record_map[
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
    # GT-FREE FORWARD
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


    (
        selected_alpha,
        selected_score,
        _
    ) = controller_decision(
        controller,
        X,
    )


    archived = up.iloc[
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
            "does not reproduce frozen manifest."
        )


    if not np.isclose(
        float(
            selected_alpha
        ),
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
            "does not reproduce frozen manifest."
        )


    if not np.isclose(
        float(
            selected_score
        ),
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
            "does not reproduce frozen manifest."
        )


    # Ground truth is copied only AFTER feature construction.
    # It is not part of the model input.
    gt = int(
        archived[
            "ground_truth"
        ]
    )


    rec = {
        "raw_sample_id":
            sid,

        "cohort_uid":
            uid,

        "ground_truth":
            gt,

        "baseline_prediction":
            baseline_prediction,

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
                pg_norm[
                    idx
                ]
            ),

        "h_norm":
            float(
                h_norm[
                    idx
                ]
            ),

        "ph_norm":
            float(
                ph_norm[
                    idx
                ]
            ),

        "g_norm":
            float(
                g_norm[
                    idx
                ]
            ),

        "sa_sensitivity":
            float(
                sa_sensitivity[
                    idx
                ]
            ),

        "cos_pug_puh":
            float(
                cos_pug_puh[
                    idx
                ]
            ),

        "gradient_capture":
            float(
                gradient_capture[
                    idx
                ]
            ),

        "activation_capture":
            float(
                activation_capture[
                    idx
                ]
            ),

        "controller_features": {
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

if len(
    records
) != EXPECTED_N:

    raise RuntimeError(
        "Final feature N mismatch."
    )


flat_rows = []


for rec in records:

    row = {
        key:
            value
        for key, value
        in rec.items()
        if key
        !=
        "controller_features"
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


ctrl_cols = [
    f"ctrl__{name}"
    for name in feature_names
]


if len(
    ctrl_cols
) != 39:

    raise RuntimeError(
        "Controller-column count mismatch."
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


if df[
    "cohort_uid"
].nunique() != EXPECTED_N:

    raise RuntimeError(
        "cohort_uid uniqueness failure."
    )


df.to_csv(
    FINAL,
    index=False,
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "experiment":
        "utility_final_features_v1",

    "population":
        "frozen final selected_alpha > 1",

    "n":
        EXPECTED_N,

    "controller_feature_count":
        39,

    "mechanistic_features": [
        "pg_norm",
        "h_norm",
        "ph_norm",
        "g_norm",
        "sa_sensitivity",
        "cos_pug_puh",
        "gradient_capture",
        "activation_capture",
    ],

    "model_input_policy":
        "ground truth is never used in "
        "feature construction",

    "dataset_metadata_sha256":
        sha256(
            META
        ),

    "action_manifest_sha256":
        sha256(
            ACTION
        ),

    "action_freeze_sha256":
        sha256(
            ACTION_FREEZE
        ),

    "gradient_freeze_sha256":
        sha256(
            GRAD_FREEZE
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

    "utility_router_sha256":
        sha256(
            UTILITY_ROUTER
        ),

    "final_protocol_sha256":
        sha256(
            PROTOCOL
        ),

    "feature_csv_sha256":
        sha256(
            FINAL
        ),
}


METADATA_OUT.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 96)
print(
    "UTILITY FINAL FEATURE ARCHIVE COMPLETE"
)
print("=" * 96)

print(
    "N:",
    len(
        df
    ),
)

print(
    "Controller features:",
    len(
        ctrl_cols
    ),
)

print(
    "Mechanistic features:",
    8,
)

print()
print(
    "Feature CSV SHA256:",
    sha256(
        FINAL
    ),
)

print(
    "Metadata SHA256:",
    sha256(
        METADATA_OUT
    ),
)

print("=" * 96)
