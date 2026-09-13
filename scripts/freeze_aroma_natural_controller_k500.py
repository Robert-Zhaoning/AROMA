#!/usr/bin/env python3

"""
Freeze AROMA Natural-Adapted Utility Controller (K=500)
======================================================

Development decision:
K=500 was selected from the previously completed post-hoc
sample-efficiency curve as a near-plateau, sample-efficient
operating point.

Calibration subset:
250 Simple + 250 Complex examples, selected deterministically
by SHA256(question_id + fixed salt). Selection does NOT use
ground-truth correctness, utility, repairability, or model score.

Frozen specification:
- feature family: full_geometry (39)
- model: StandardScaler -> Ridge
- ridge alpha: 0.01
- threshold: 0.1
- actions: [0, 1, 1.5, 2, 4]

This script creates the controller to be used on a NEW,
UNTOUCHED natural confirmation set.
"""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/workspace/AromaExperiments")

OLD_BUNDLE_PATH = (
    ROOT
    / "outputs/proc_count_causal_v2/controller/"
    "final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

TALLY_PATH = (
    ROOT
    / "outputs/tallyqa_natural_ood_v1/"
    "final_frozen_controller/"
    "tallyqa_final_results.csv"
)

MATRIX_PATH = (
    ROOT
    / "outputs/tallyqa_natural_ood_v1/"
    "forensics/action_oracle/"
    "action_matrix.csv"
)

OUT_DIR = (
    ROOT
    / "outputs/phase2_natural_controller_k500/"
    "final_frozen_controller"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BUNDLE_OUT = (
    OUT_DIR
    / "aroma_natural_utility_controller_k500.joblib"
)

CALIBRATION_MANIFEST_OUT = (
    OUT_DIR
    / "calibration_manifest.csv"
)

TRAINING_SUMMARY_OUT = (
    OUT_DIR
    / "training_summary.csv"
)

METADATA_OUT = (
    OUT_DIR
    / "frozen_controller_metadata.json"
)


K_PER_SUBSET = 250
SALT = "AROMA_NATURAL_K500_FREEZE_V1"

ALL_ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]

NONNOOP = [
    0.0,
    1.5,
    2.0,
    4.0,
]


def as_bool(s):

    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        return (
            pd.to_numeric(s)
            .fillna(0)
            .ne(0)
        )

    t = (
        s.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    bad = set(t.unique()) - set(mapping)

    if bad:
        raise RuntimeError(
            f"Unknown bool values: {bad}"
        )

    return t.map(mapping).astype(bool)


def selection_hash(qid):

    text = (
        f"{SALT}|{int(qid)}"
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


old_bundle = joblib.load(
    OLD_BUNDLE_PATH
)

tally = pd.read_csv(
    TALLY_PATH
)

matrix = pd.read_csv(
    MATRIX_PATH
)


feature_names = list(
    old_bundle[
        "feature_names"
    ]
)

feature_cols = [
    f"feature__{f}"
    for f in feature_names
]

ridge_alpha = float(
    old_bundle[
        "ridge_alpha"
    ]
)

threshold = float(
    old_bundle[
        "threshold"
    ]
)


if (
    len(feature_names) != 39
    or ridge_alpha != 0.01
    or threshold != 0.1
):
    raise RuntimeError(
        "Frozen specification mismatch."
    )


# ============================================================
# Deterministic calibration selection
# ============================================================

selection = tally[
    [
        "question_id",
        "subset",
        "image",
        "image_id",
    ]
].copy()

selection[
    "selection_hash"
] = (
    selection[
        "question_id"
    ]
    .apply(
        selection_hash
    )
)

parts = []

for subset_name in [
    "simple",
    "complex",
]:

    g = (
        selection[
            selection[
                "subset"
            ]
            ==
            subset_name
        ]
        .sort_values(
            [
                "selection_hash",
                "question_id",
            ]
        )
        .head(
            K_PER_SUBSET
        )
        .copy()
    )

    if len(g) != K_PER_SUBSET:
        raise RuntimeError(
            f"Could not select "
            f"{K_PER_SUBSET} "
            f"{subset_name} rows."
        )

    parts.append(g)


calibration_manifest = (
    pd.concat(
        parts,
        ignore_index=True,
    )
    .sort_values(
        [
            "subset",
            "selection_hash",
        ]
    )
    .reset_index(
        drop=True
    )
)


if len(calibration_manifest) != 500:
    raise RuntimeError(
        "Expected 500 calibration rows."
    )

if (
    calibration_manifest[
        "question_id"
    ]
    .nunique()
    !=
    500
):
    raise RuntimeError(
        "Duplicate calibration question."
    )

if (
    calibration_manifest[
        "image"
    ]
    .nunique()
    !=
    500
):
    raise RuntimeError(
        "Calibration set is not "
        "one-question-per-image."
    )


calibration_manifest.to_csv(
    CALIBRATION_MANIFEST_OUT,
    index=False,
)


# ============================================================
# Build X
# ============================================================

calibration_ids = (
    calibration_manifest[
        "question_id"
    ]
    .astype(int)
    .tolist()
)

base = (
    tally
    .set_index(
        "question_id"
    )
    .loc[
        calibration_ids
    ]
    .reset_index()
)

X = (
    base[
        feature_cols
    ]
    .to_numpy(
        dtype=float
    )
)

if X.shape != (500, 39):
    raise RuntimeError(
        f"Unexpected X shape: {X.shape}"
    )

if not np.isfinite(
    X
).all():
    raise RuntimeError(
        "Non-finite calibration feature."
    )


# ============================================================
# Utility labels
# ============================================================

matrix[
    "alpha"
] = pd.to_numeric(
    matrix[
        "alpha"
    ]
).astype(float)

matrix[
    "baseline_correct_bool"
] = as_bool(
    matrix[
        "baseline_correct"
    ]
)

matrix[
    "action_correct_bool"
] = as_bool(
    matrix[
        "action_correct"
    ]
)

matrix[
    "utility"
] = (
    matrix[
        "action_correct_bool"
    ].astype(int)
    -
    matrix[
        "baseline_correct_bool"
    ].astype(int)
)

utility = (
    matrix.pivot(
        index="question_id",
        columns="alpha",
        values="utility",
    )
    .reindex(
        calibration_ids
    )
)


for action in ALL_ACTIONS:

    if action not in utility.columns:
        raise RuntimeError(
            f"Missing utility for "
            f"action {action}"
        )


# ============================================================
# Fit fixed model family
# ============================================================

models = {}

summary_rows = []

for action in NONNOOP:

    y = (
        utility[
            action
        ]
        .to_numpy(
            dtype=float
        )
    )

    model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=ridge_alpha,
                ),
            ),
        ]
    )

    model.fit(
        X,
        y,
    )

    models[
        action
    ] = model

    summary_rows.append(
        {
            "alpha":
                action,

            "n":
                len(y),

            "utility_mean":
                float(
                    y.mean()
                ),

            "utility_positive":
                int(
                    (
                        y > 0
                    ).sum()
                ),

            "utility_negative":
                int(
                    (
                        y < 0
                    ).sum()
                ),

            "utility_zero":
                int(
                    (
                        y == 0
                    ).sum()
                ),
        }
    )


training_summary = pd.DataFrame(
    summary_rows
)

training_summary.to_csv(
    TRAINING_SUMMARY_OUT,
    index=False,
)


# ============================================================
# Frozen bundle
# ============================================================

bundle = {
    "bundle_version":
        "AROMA-natural-k500-v1",

    "parent_bundle_version":
        old_bundle.get(
            "bundle_version"
        ),

    "model_id":
        old_bundle[
            "model_id"
        ],

    "attention_implementation":
        old_bundle[
            "attention_implementation"
        ],

    "head_layer":
        old_bundle[
            "head_layer"
        ],

    "head_index":
        old_bundle[
            "head_index"
        ],

    "head_name":
        old_bundle[
            "head_name"
        ],

    "actions":
        ALL_ACTIONS,

    "nonnoop_actions":
        NONNOOP,

    "feature_family":
        "full_geometry",

    "feature_names":
        feature_names,

    "ridge_alpha":
        ridge_alpha,

    "threshold":
        threshold,

    "natural_calibration_n":
        500,

    "natural_calibration_simple":
        250,

    "natural_calibration_complex":
        250,

    "calibration_selection":
        (
            "SHA256 rank within "
            "Simple/Complex using fixed salt"
        ),

    "calibration_selection_salt":
        SALT,

    "models":
        models,
}


joblib.dump(
    bundle,
    BUNDLE_OUT,
)


metadata = {
    "status":
        "FROZEN_BEFORE_NEW_NATURAL_CONFIRMATION",

    "development_dataset":
        "TallyQA natural-OOD v1 4000-sample development set",

    "calibration_n":
        500,

    "selection_rule":
        (
            "250 Simple + 250 Complex; "
            "lowest SHA256 ranks under fixed salt; "
            "selection independent of outcomes"
        ),

    "ridge_alpha":
        ridge_alpha,

    "threshold":
        threshold,

    "actions":
        ALL_ACTIONS,

    "feature_count":
        39,

    "bundle_sha256":
        sha256_file(
            BUNDLE_OUT
        ),

    "calibration_manifest_sha256":
        sha256_file(
            CALIBRATION_MANIFEST_OUT
        ),

    "source_tallyqa_results_sha256":
        sha256_file(
            TALLY_PATH
        ),

    "source_action_matrix_sha256":
        sha256_file(
            MATRIX_PATH
        ),

    "independent_confirmation_results_seen":
        False,
}


with open(
    METADATA_OUT,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        sort_keys=True,
    )


print("=" * 112)
print("AROMA NATURAL CONTROLLER K=500 — FREEZE COMPLETE")
print("=" * 112)

print("Calibration samples:", 500)
print("Simple:", 250)
print("Complex:", 250)
print("Features:", 39)
print("Ridge alpha:", ridge_alpha)
print("Threshold:", threshold)
print("Actions:", ALL_ACTIONS)

print("\nUTILITY LABEL SUMMARY")
print(
    training_summary.to_string(
        index=False
    )
)

print("\nBundle:")
print(BUNDLE_OUT)

print("\nBundle SHA256:")
print(
    sha256_file(
        BUNDLE_OUT
    )
)

print("\nCalibration manifest SHA256:")
print(
    sha256_file(
        CALIBRATION_MANIFEST_OUT
    )
)

print("\nSTATUS:")
print(
    "FROZEN BEFORE NEW NATURAL CONFIRMATION"
)

