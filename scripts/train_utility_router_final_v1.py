import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# FROZEN INPUTS
# ============================================================

PROTOCOL = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_protocol_v1/"
    "utility_router_protocol_v1.json"
)

EXPECTED_PROTOCOL_SHA = (
    "2bf3ba6504c8915c846c810adf2a6652e31caba9658"
    "ba4993fd1de4d4959c5c8"
)

DATA = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_candidate_dataset_v1/"
    "utility_candidate_dataset_v1.csv"
)

EXPECTED_DATA_SHA = (
    "e615029ef04e44d923134a8aa355918af5f6b7a5fd49"
    "055a860ee90efb1c88cb"
)

CV_EXECUTION_SPEC = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_cv_v1/"
    "execution_spec.json"
)

EXPECTED_CV_EXECUTION_SHA = (
    "2937c2dfdb375e1e7a2a73f51b1e8e0fc5782cc7b83f"
    "36263fdf65f2e1e300b9"
)

CV_SUMMARY = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_cv_v1/"
    "summary.json"
)

OUT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_final_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FREEZE = (
    OUT / "final_training_freeze.json"
)

BUNDLE = (
    OUT / "aroma_utility_router_v1.joblib"
)

MANIFEST = (
    OUT / "manifest.json"
)

C_VALUE = 1.0


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


# ============================================================
# FROZEN ARTIFACT CHECKS
# ============================================================

if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
    raise RuntimeError(
        "Protocol SHA mismatch."
    )

if sha256(DATA) != EXPECTED_DATA_SHA:
    raise RuntimeError(
        "Candidate dataset SHA mismatch."
    )

if (
    sha256(CV_EXECUTION_SPEC)
    != EXPECTED_CV_EXECUTION_SHA
):
    raise RuntimeError(
        "CV execution-spec SHA mismatch."
    )

if not CV_SUMMARY.exists():
    raise FileNotFoundError(
        CV_SUMMARY
    )


cv = json.loads(
    CV_SUMMARY.read_text(
        encoding="utf-8"
    )
)

if int(cv["router_correct"]) != 108:
    raise RuntimeError(
        "Unexpected frozen OOF result."
    )

if int(cv["global_correct"]) != 100:
    raise RuntimeError(
        "Unexpected frozen global comparator."
    )


# ============================================================
# DATA
# ============================================================

df = pd.read_csv(
    DATA
)

if len(df) != 2400:
    raise RuntimeError(
        f"Expected 2400 rows; got {len(df)}"
    )

if df["sample_id"].nunique() != 160:
    raise RuntimeError(
        "Expected 160 independent samples."
    )


# ============================================================
# EXACT SAME FEATURE MAP AS FROZEN CV
# ============================================================

ctrl_cols = sorted(
    [
        c
        for c in df.columns
        if c.startswith("ctrl__")
    ]
)

if len(ctrl_cols) != 39:
    raise RuntimeError(
        f"Expected 39 controller features; "
        f"got {len(ctrl_cols)}"
    )

sample_cols = (
    ctrl_cols
    +
    [
        "selected_alpha",
        "selected_score",
        "mech__log_pg_norm",
        "mech__log_h_norm",
        "mech__log_sa_sensitivity",
        "mech__cos_pug_puh",
        "mech__gradient_capture",
        "mech__activation_capture",
    ]
)

candidate_channels = [
    "cand__mn",
    "cand__sa",
    "cand__beta",
    "cand__beta_sq",
    "cand__mn_beta",
    "cand__sa_beta",
    "cand__mn_beta_sq",
    "cand__sa_beta_sq",
]


def build_design(frame):

    Xs = (
        frame[sample_cols]
        .to_numpy(
            dtype=np.float64
        )
    )

    is_mn = (
        frame["candidate_is_mn"]
        .to_numpy(
            dtype=np.float64
        )
    )

    is_sa = (
        frame["candidate_is_sa"]
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        frame["candidate_beta"]
        .to_numpy(
            dtype=np.float64
        )
    )

    beta_sq = (
        frame["candidate_beta_sq"]
        .to_numpy(
            dtype=np.float64
        )
    )

    C = np.column_stack(
        [
            is_mn,
            is_sa,
            beta,
            beta_sq,
            is_mn * beta,
            is_sa * beta,
            is_mn * beta_sq,
            is_sa * beta_sq,
        ]
    )

    interactions = np.concatenate(
        [
            Xs * C[:, j:j+1]
            for j in range(
                C.shape[1]
            )
        ],
        axis=1,
    )

    X = np.concatenate(
        [
            Xs,
            C,
            interactions,
        ],
        axis=1,
    )

    names = (
        sample_cols
        +
        candidate_channels
        +
        [
            f"{s}__X__{c}"
            for c in candidate_channels
            for s in sample_cols
        ]
    )

    if X.shape[1] != len(names):
        raise RuntimeError(
            "Feature/name mismatch."
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Non-finite design matrix."
        )

    return X, names


X, feature_names = build_design(
    df
)

y = (
    df["candidate_correct"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# FREEZE FINAL TRAINING BEFORE FIT
# ============================================================

freeze = {
    "experiment":
        "aroma_utility_router_final_v1",

    "status":
        "final development router frozen "
        "before external evaluation",

    "training_samples":
        160,

    "training_candidate_rows":
        2400,

    "candidate_set":
        "identity + MN/SA x "
        "{0.5,0.75,1,1.25,1.5,1.75,2}",

    "model":
        "StandardScaler + L2 logistic regression",

    "C":
        C_VALUE,

    "solver":
        "lbfgs",

    "max_iter":
        10000,

    "feature_count":
        len(feature_names),

    "feature_names":
        feature_names,

    "selection_rule":
        "argmax predicted candidate correctness; "
        "tie -> identity -> smaller beta -> MN -> SA",

    "no_further_hyperparameter_tuning":
        True,

    "development_oof_result": {
        "router_correct":
            int(cv["router_correct"]),

        "global_correct":
            int(cv["global_correct"]),

        "router_minus_global_pp":
            float(
                cv[
                    "router_minus_global_pp"
                ]
            ),

        "mcnemar_exact_p":
            float(
                cv[
                    "mcnemar_exact_p"
                ]
            ),
    },

    "protocol_sha256":
        sha256(PROTOCOL),

    "candidate_dataset_sha256":
        sha256(DATA),

    "cv_execution_spec_sha256":
        sha256(
            CV_EXECUTION_SPEC
        ),

    "cv_summary_sha256":
        sha256(
            CV_SUMMARY
        ),
}


if FREEZE.exists():

    old = json.loads(
        FREEZE.read_text(
            encoding="utf-8"
        )
    )

    if old != freeze:
        raise RuntimeError(
            "Existing final freeze differs."
        )

else:

    FREEZE.write_text(
        json.dumps(
            freeze,
            indent=2,
        ),
        encoding="utf-8",
    )


print(
    "Final training freeze SHA256:",
    sha256(FREEZE),
)


# ============================================================
# FIT FINAL DEVELOPMENT ROUTER
# ============================================================

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logistic",
            LogisticRegression(
                penalty="l2",
                C=C_VALUE,
                solver="lbfgs",
                max_iter=10000,
            ),
        ),
    ]
)

model.fit(
    X,
    y,
)


bundle = {
    "version":
        "AROMA_UTILITY_ROUTER_V1",

    "model":
        model,

    "feature_names":
        feature_names,

    "sample_cols":
        sample_cols,

    "candidate_channels":
        candidate_channels,

    "candidate_beta_grid": [
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
    ],

    "candidate_methods": [
        "IDENTITY",
        "MN",
        "SA",
    ],

    "C":
        C_VALUE,

    "training_samples":
        160,

    "training_rows":
        2400,

    "training_freeze_sha256":
        sha256(FREEZE),
}


joblib.dump(
    bundle,
    BUNDLE,
)


manifest = {
    "version":
        "AROMA_UTILITY_ROUTER_V1",

    "bundle":
        str(BUNDLE),

    "bundle_sha256":
        sha256(BUNDLE),

    "training_freeze":
        str(FREEZE),

    "training_freeze_sha256":
        sha256(FREEZE),

    "protocol_sha256":
        sha256(PROTOCOL),

    "candidate_dataset_sha256":
        sha256(DATA),

    "feature_count":
        len(feature_names),

    "training_samples":
        160,

    "training_rows":
        2400,

    "no_further_tuning":
        True,
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 92)
print(
    "FINAL UTILITY ROUTER V1 TRAINED AND FROZEN"
)
print("=" * 92)

print(
    "Training samples:",
    160,
)

print(
    "Training rows:",
    2400,
)

print(
    "Design features:",
    len(feature_names),
)

print(
    "Bundle:",
    BUNDLE,
)

print(
    "Bundle SHA256:",
    sha256(BUNDLE),
)

print(
    "Training freeze SHA256:",
    sha256(FREEZE),
)

print(
    "Manifest SHA256:",
    sha256(MANIFEST),
)

print("=" * 92)
