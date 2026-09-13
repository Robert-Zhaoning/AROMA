#!/usr/bin/env python3

"""
AROMA Phase II-B:
Natural-Domain Conditional-Utility Learnability Probe
=====================================================

Question
--------
Does the existing 39-dimensional GT-free numeral-geometry
representation contain enough information to predict intervention
utility within the natural TallyQA domain?

Protocol
--------
- Dataset: already-used TallyQA-4000 development/forensic set.
- Evaluation: 5-fold out-of-fold only.
- Split: fixed StratifiedKFold by Simple/Complex subset.
- Model family: EXACT same StandardScaler -> Ridge.
- Ridge alpha: EXACT frozen value 0.01.
- Decision threshold: EXACT frozen value 0.1.
- Actions: EXACT frozen {0, 1, 1.5, 2, 4}.
- NO hyperparameter search.
- NO model/VLM inference.

This is a post-hoc learnability probe, NOT an independent
natural-domain confirmation experiment.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


ROOT = Path("/workspace/AromaExperiments")

BUNDLE_PATH = (
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
    / "outputs/phase2_natural_utility_learnability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

N_SPLITS = 5
RANDOM_STATE = 20260912

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


def safe_spearman(x, y):

    if not HAVE_SCIPY:
        return np.nan

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if (
        len(x) < 2
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return np.nan

    return float(
        spearmanr(
            x,
            y,
        ).statistic
    )


# ============================================================
# Load
# ============================================================

bundle = joblib.load(
    BUNDLE_PATH
)

tally = pd.read_csv(
    TALLY_PATH
)

matrix = pd.read_csv(
    MATRIX_PATH
)

feature_names = list(
    bundle["feature_names"]
)

feature_cols = [
    f"feature__{f}"
    for f in feature_names
]

ridge_alpha = float(
    bundle["ridge_alpha"]
)

threshold = float(
    bundle["threshold"]
)

if ridge_alpha != 0.01:
    raise RuntimeError(
        f"Unexpected ridge alpha: {ridge_alpha}"
    )

if threshold != 0.1:
    raise RuntimeError(
        f"Unexpected threshold: {threshold}"
    )

if len(tally) != 4000:
    raise RuntimeError(
        f"Expected 4000 TallyQA rows, "
        f"found {len(tally)}"
    )

if len(matrix) != 20000:
    raise RuntimeError(
        f"Expected 20000 action rows, "
        f"found {len(matrix)}"
    )

if tally["question_id"].nunique() != 4000:
    raise RuntimeError(
        "Expected unique question_id per row."
    )


# ============================================================
# X and baseline state
# ============================================================

X = (
    tally[
        feature_cols
    ]
    .to_numpy(
        dtype=float
    )
)

if X.shape != (4000, 39):
    raise RuntimeError(
        f"Unexpected X shape: {X.shape}"
    )

if not np.isfinite(X).all():
    raise RuntimeError(
        "Non-finite feature value."
    )

question_ids = (
    tally["question_id"]
    .astype(int)
    .to_numpy()
)

subset = (
    tally["subset"]
    .astype(str)
    .to_numpy()
)

baseline_correct = (
    as_bool(
        tally["baseline_correct"]
    )
    .astype(int)
    .to_numpy()
)

baseline_accuracy = float(
    baseline_correct.mean()
)


# ============================================================
# Utility matrix from completed frozen action sweep
# ============================================================

matrix["alpha"] = pd.to_numeric(
    matrix["alpha"]
).astype(float)

matrix["baseline_correct_bool"] = (
    as_bool(
        matrix["baseline_correct"]
    )
)

matrix["action_correct_bool"] = (
    as_bool(
        matrix["action_correct"]
    )
)

matrix["utility"] = (
    matrix["action_correct_bool"].astype(int)
    -
    matrix["baseline_correct_bool"].astype(int)
)

utility_df = (
    matrix.pivot(
        index="question_id",
        columns="alpha",
        values="utility",
    )
    .reindex(
        question_ids
    )
)

for action in ALL_ACTIONS:

    if action not in utility_df.columns:
        raise RuntimeError(
            f"Missing action utility {action}"
        )

utility_df = utility_df[
    ALL_ACTIONS
].copy()

utility_df.columns = [
    float(c)
    for c in utility_df.columns
]

U = {
    action:
        utility_df[action]
        .to_numpy(
            dtype=int
        )
    for action in ALL_ACTIONS
}

if not np.array_equal(
    U[1.0],
    np.zeros(
        len(tally),
        dtype=int,
    ),
):
    raise RuntimeError(
        "NOOP utility is not identically zero."
    )


# ============================================================
# OOF storage
# ============================================================

oof_scores = {
    action:
        np.full(
            len(tally),
            np.nan,
            dtype=float,
        )
    for action in NONNOOP
}

oof_fold = np.full(
    len(tally),
    -1,
    dtype=int,
)


# ============================================================
# Fixed 5-fold stratification by Simple / Complex only
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_rows = []

for fold, (
    train_idx,
    test_idx,
) in enumerate(
    cv.split(
        X,
        subset,
    )
):

    oof_fold[
        test_idx
    ] = fold

    if set(
        train_idx
    ) & set(
        test_idx
    ):
        raise RuntimeError(
            "Train/test overlap."
        )

    for action in NONNOOP:

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
            X[
                train_idx
            ],
            U[
                action
            ][
                train_idx
            ],
        )

        oof_scores[
            action
        ][
            test_idx
        ] = (
            model.predict(
                X[
                    test_idx
                ]
            )
        )

    fold_rows.append(
        {
            "fold":
                fold,

            "train_n":
                len(train_idx),

            "test_n":
                len(test_idx),

            "test_simple":
                int(
                    (
                        subset[
                            test_idx
                        ]
                        ==
                        "simple"
                    ).sum()
                ),

            "test_complex":
                int(
                    (
                        subset[
                            test_idx
                        ]
                        ==
                        "complex"
                    ).sum()
                ),

            "test_baseline_accuracy":
                float(
                    baseline_correct[
                        test_idx
                    ].mean()
                ),
        }
    )


if (
    oof_fold
    <
    0
).any():
    raise RuntimeError(
        "Incomplete OOF assignment."
    )

for action in NONNOOP:

    if not np.isfinite(
        oof_scores[
            action
        ]
    ).all():
        raise RuntimeError(
            f"Missing OOF score for {action}."
        )


# ============================================================
# OOF controller decision
# ============================================================

score_matrix = np.column_stack(
    [
        oof_scores[a]
        for a in NONNOOP
    ]
)

best_idx = np.argmax(
    score_matrix,
    axis=1,
)

best_score = score_matrix[
    np.arange(
        len(tally)
    ),
    best_idx,
]

best_action = np.array(
    [
        NONNOOP[i]
        for i in best_idx
    ],
    dtype=float,
)

selected_alpha = np.where(
    best_score
    >
    threshold,
    best_action,
    1.0,
)

selected_utility = np.zeros(
    len(tally),
    dtype=int,
)

for action in NONNOOP:

    mask = np.isclose(
        selected_alpha,
        action,
    )

    selected_utility[
        mask
    ] = U[
        action
    ][
        mask
    ]

post_correct = (
    baseline_correct
    +
    selected_utility
)

if not np.isin(
    post_correct,
    [0, 1],
).all():
    raise RuntimeError(
        "Invalid post correctness."
    )


# ============================================================
# Primary metrics
# ============================================================

post_accuracy = float(
    post_correct.mean()
)

repairs = int(
    (
        (baseline_correct == 0)
        &
        (post_correct == 1)
    ).sum()
)

breaks = int(
    (
        (baseline_correct == 1)
        &
        (post_correct == 0)
    ).sum()
)

intervention_rate = float(
    (
        selected_alpha
        !=
        1.0
    ).mean()
)


# ============================================================
# Per-action OOF calibration
# ============================================================

cal_rows = []

for action in NONNOOP:

    s = oof_scores[action]
    u = U[action]

    trigger = (
        s > threshold
    )

    triggered_u = (
        u[
            trigger
        ]
    )

    cal_rows.append(
        {
            "alpha":
                action,

            "score_mean":
                float(
                    s.mean()
                ),

            "score_median":
                float(
                    np.median(s)
                ),

            "score_utility_spearman":
                safe_spearman(
                    s,
                    u,
                ),

            "fraction_score_gt_threshold":
                float(
                    trigger.mean()
                ),

            "mean_utility_all":
                float(
                    u.mean()
                ),

            "mean_utility_triggered":
                (
                    float(
                        triggered_u.mean()
                    )
                    if len(
                        triggered_u
                    )
                    else 0.0
                ),

            "triggered_n":
                int(
                    trigger.sum()
                ),

            "triggered_positive_utility":
                int(
                    (
                        triggered_u
                        >
                        0
                    ).sum()
                ),

            "triggered_negative_utility":
                int(
                    (
                        triggered_u
                        <
                        0
                    ).sum()
                ),

            "triggered_net_utility":
                int(
                    triggered_u.sum()
                ),
        }
    )


calibration = pd.DataFrame(
    cal_rows
)

calibration.to_csv(
    OUT_DIR
    / "oof_per_action_calibration.csv",
    index=False,
)


# ============================================================
# Action distribution
# ============================================================

action_rows = []

for alpha in ALL_ACTIONS:

    count = int(
        np.isclose(
            selected_alpha,
            alpha,
        ).sum()
    )

    action_rows.append(
        {
            "alpha":
                alpha,

            "count":
                count,

            "fraction":
                count
                /
                len(tally),
        }
    )

action_distribution = pd.DataFrame(
    action_rows
)

action_distribution.to_csv(
    OUT_DIR
    / "oof_action_distribution.csv",
    index=False,
)


# ============================================================
# By subset
# ============================================================

subset_rows = []

for name in [
    "simple",
    "complex",
]:

    mask = (
        subset
        ==
        name
    )

    b = baseline_correct[
        mask
    ]

    p = post_correct[
        mask
    ]

    subset_rows.append(
        {
            "subset":
                name,

            "n":
                int(
                    mask.sum()
                ),

            "baseline_accuracy":
                float(
                    b.mean()
                ),

            "oof_post_accuracy":
                float(
                    p.mean()
                ),

            "accuracy_change":
                float(
                    p.mean()
                    -
                    b.mean()
                ),

            "repairs":
                int(
                    (
                        (b == 0)
                        &
                        (p == 1)
                    ).sum()
                ),

            "breaks":
                int(
                    (
                        (b == 1)
                        &
                        (p == 0)
                    ).sum()
                ),

            "intervention_rate":
                float(
                    (
                        selected_alpha[
                            mask
                        ]
                        !=
                        1.0
                    ).mean()
                ),
        }
    )

by_subset = pd.DataFrame(
    subset_rows
)

by_subset.to_csv(
    OUT_DIR
    / "oof_by_subset.csv",
    index=False,
)


# ============================================================
# Per-sample artifact
# ============================================================

per_sample = pd.DataFrame(
    {
        "question_id":
            question_ids,

        "subset":
            subset,

        "fold":
            oof_fold,

        "baseline_correct":
            baseline_correct,

        "selected_alpha":
            selected_alpha,

        "selected_score":
            best_score,

        "selected_utility":
            selected_utility,

        "post_correct":
            post_correct,
    }
)

for action in NONNOOP:

    tag = (
        str(action)
        .replace(
            ".",
            "p",
        )
    )

    per_sample[
        f"score_alpha_{tag}"
    ] = oof_scores[
        action
    ]

per_sample.to_csv(
    OUT_DIR
    / "oof_predictions.csv",
    index=False,
)


# ============================================================
# Summary + metadata
# ============================================================

summary = pd.DataFrame(
    [
        {
            "n":
                len(tally),

            "baseline_accuracy":
                baseline_accuracy,

            "oof_post_accuracy":
                post_accuracy,

            "accuracy_change":
                post_accuracy
                -
                baseline_accuracy,

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                repairs
                -
                breaks,

            "intervention_rate":
                intervention_rate,

            "ridge_alpha":
                ridge_alpha,

            "threshold":
                threshold,

            "n_splits":
                N_SPLITS,

            "random_state":
                RANDOM_STATE,
        }
    ]
)

summary.to_csv(
    OUT_DIR
    / "oof_summary.csv",
    index=False,
)

pd.DataFrame(
    fold_rows
).to_csv(
    OUT_DIR
    / "fold_summary.csv",
    index=False,
)


metadata = {
    "analysis_type":
        "post_hoc_natural_conditional_utility_learnability_probe",

    "independent_confirmation":
        False,

    "model_inference_performed":
        False,

    "feature_family":
        bundle[
            "feature_family"
        ],

    "feature_count":
        len(
            feature_names
        ),

    "ridge_alpha":
        ridge_alpha,

    "threshold":
        threshold,

    "actions":
        ALL_ACTIONS,

    "cv":
        (
            "5-fold StratifiedKFold by "
            "Simple/Complex subset"
        ),

    "random_state":
        RANDOM_STATE,

    "hyperparameter_search":
        False,
}

with open(
    OUT_DIR
    / "metadata.json",
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        sort_keys=True,
    )


# ============================================================
# Print
# ============================================================

print("=" * 118)
print("NATURAL UTILITY LEARNABILITY PROBE — OOF SUMMARY")
print("=" * 118)

print(
    summary.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("OOF PER-ACTION CALIBRATION")
print("=" * 118)

print(
    calibration.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("OOF ACTION DISTRIBUTION")
print("=" * 118)

print(
    action_distribution.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("OOF BY SUBSET")
print("=" * 118)

print(
    by_subset.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("LEARNABILITY INTERPRETATION")
print("=" * 118)

print(
    "Frozen synthetic controller change:",
    -0.02975,
)

print(
    "Natural-domain OOF refit change:",
    post_accuracy
    -
    baseline_accuracy,
)

print(
    "Oracle ceiling change:",
    0.06725,
)

print(
    "Fraction of oracle gain recovered:",
    (
        (
            post_accuracy
            -
            baseline_accuracy
        )
        /
        0.06725
    ),
)


print("\n" + "=" * 118)
print("PROBE COMPLETE")
print("=" * 118)

print(
    "This is POST-HOC DEVELOPMENT / "
    "LEARNABILITY ANALYSIS ONLY."
)

print("No VLM inference performed.")
print("No hyperparameter search performed.")

print("\nSaved to:")
print(OUT_DIR)

