#!/usr/bin/env python3

"""
AROMA Natural Utility Adaptation Sample-Efficiency Probe
========================================================

Goal:
Measure how much labeled natural-domain calibration data is needed
for the existing 39-dimensional GT-free geometry to learn useful
action utilities.

IMPORTANT:
- Post-hoc development experiment only.
- TallyQA-4000 is NOT an untouched confirmation set.
- No VLM inference.
- No hyperparameter search.
- Same feature family, Ridge alpha, threshold and action set.
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
    / "outputs/phase2_natural_adaptation_curve"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CALIBRATION_SIZES = [
    50,
    100,
    250,
    500,
    1000,
    2000,
    3200,
]

N_OUTER_FOLDS = 5
N_REPEATS = 5

OUTER_RANDOM_STATE = 20260912
SUBSAMPLE_BASE_SEED = 314159

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
        f"Unexpected ridge alpha {ridge_alpha}"
    )

if threshold != 0.1:
    raise RuntimeError(
        f"Unexpected threshold {threshold}"
    )


X = (
    tally[
        feature_cols
    ]
    .to_numpy(
        dtype=float
    )
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
# Utility labels from completed action sweep
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

utility = (
    matrix.pivot(
        index="question_id",
        columns="alpha",
        values="utility",
    )
    .reindex(
        question_ids
    )
)

utility = utility[
    ALL_ACTIONS
].copy()

U = {
    a:
        utility[a]
        .to_numpy(
            dtype=int
        )
    for a in ALL_ACTIONS
}


# ============================================================
# Frozen synthetic-controller test outcomes
# ============================================================

frozen_selected_alpha = (
    pd.to_numeric(
        tally["selected_alpha"]
    )
    .to_numpy(
        dtype=float
    )
)

frozen_post_correct = (
    as_bool(
        tally["post_correct"]
    )
    .astype(int)
    .to_numpy()
)


# ============================================================
# Outer folds
# ============================================================

outer = StratifiedKFold(
    n_splits=N_OUTER_FOLDS,
    shuffle=True,
    random_state=OUTER_RANDOM_STATE,
)

result_rows = []
per_fold_rows = []


for fold, (
    train_idx,
    test_idx,
) in enumerate(
    outer.split(
        X,
        subset,
    )
):

    train_idx = np.asarray(
        train_idx,
        dtype=int,
    )

    test_idx = np.asarray(
        test_idx,
        dtype=int,
    )

    if len(train_idx) != 3200:
        raise RuntimeError(
            "Expected outer training size 3200."
        )

    if len(test_idx) != 800:
        raise RuntimeError(
            "Expected outer test size 800."
        )

    test_base = baseline_correct[
        test_idx
    ]

    frozen_test = frozen_post_correct[
        test_idx
    ]

    frozen_gain = float(
        frozen_test.mean()
        -
        test_base.mean()
    )

    for k in CALIBRATION_SIZES:

        repeats = (
            1
            if k == 3200
            else N_REPEATS
        )

        for repeat in range(
            repeats
        ):

            if k == 3200:

                calibration_idx = (
                    train_idx.copy()
                )

            else:

                rng = np.random.default_rng(
                    SUBSAMPLE_BASE_SEED
                    +
                    10000 * fold
                    +
                    100 * k
                    +
                    repeat
                )

                # Keep Simple/Complex balanced in calibration set.
                train_simple = train_idx[
                    subset[
                        train_idx
                    ]
                    ==
                    "simple"
                ]

                train_complex = train_idx[
                    subset[
                        train_idx
                    ]
                    ==
                    "complex"
                ]

                k_simple = k // 2
                k_complex = (
                    k
                    -
                    k_simple
                )

                if (
                    k_simple
                    >
                    len(train_simple)
                    or
                    k_complex
                    >
                    len(train_complex)
                ):
                    raise RuntimeError(
                        "Calibration size too large."
                    )

                chosen_simple = rng.choice(
                    train_simple,
                    size=k_simple,
                    replace=False,
                )

                chosen_complex = rng.choice(
                    train_complex,
                    size=k_complex,
                    replace=False,
                )

                calibration_idx = np.concatenate(
                    [
                        chosen_simple,
                        chosen_complex,
                    ]
                )

                rng.shuffle(
                    calibration_idx
                )


            # ------------------------------------------------
            # Same model family, no tuning.
            # ------------------------------------------------

            scores = {}

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
                        calibration_idx
                    ],
                    U[
                        action
                    ][
                        calibration_idx
                    ],
                )

                scores[
                    action
                ] = np.asarray(
                    model.predict(
                        X[
                            test_idx
                        ]
                    ),
                    dtype=float,
                )


            score_matrix = np.column_stack(
                [
                    scores[a]
                    for a in NONNOOP
                ]
            )

            best_index = np.argmax(
                score_matrix,
                axis=1,
            )

            best_score = score_matrix[
                np.arange(
                    len(test_idx)
                ),
                best_index,
            ]

            best_action = np.array(
                [
                    NONNOOP[i]
                    for i in best_index
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
                len(test_idx),
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
                    test_idx[
                        mask
                    ]
                ]


            post = (
                test_base
                +
                selected_utility
            )

            if not np.isin(
                post,
                [0, 1],
            ).all():
                raise RuntimeError(
                    "Invalid post correctness."
                )


            repairs = int(
                (
                    (test_base == 0)
                    &
                    (post == 1)
                ).sum()
            )

            breaks = int(
                (
                    (test_base == 1)
                    &
                    (post == 0)
                ).sum()
            )

            gain = float(
                post.mean()
                -
                test_base.mean()
            )

            intervention_rate = float(
                (
                    selected_alpha
                    !=
                    1.0
                ).mean()
            )


            result_rows.append(
                {
                    "fold":
                        fold,

                    "calibration_size":
                        k,

                    "repeat":
                        repeat,

                    "test_n":
                        len(test_idx),

                    "baseline_accuracy":
                        float(
                            test_base.mean()
                        ),

                    "post_accuracy":
                        float(
                            post.mean()
                        ),

                    "accuracy_change":
                        gain,

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

                    "frozen_synthetic_gain_same_test":
                        frozen_gain,
                }
            )


results = pd.DataFrame(
    result_rows
)

results.to_csv(
    OUT_DIR
    / "all_runs.csv",
    index=False,
)


# ============================================================
# Aggregate learning curve
# ============================================================

summary_rows = []

for k in CALIBRATION_SIZES:

    g = results[
        results[
            "calibration_size"
        ]
        ==
        k
    ]

    summary_rows.append(
        {
            "calibration_size":
                k,

            "n_runs":
                len(g),

            "mean_accuracy_change":
                float(
                    g[
                        "accuracy_change"
                    ].mean()
                ),

            "std_accuracy_change":
                float(
                    g[
                        "accuracy_change"
                    ].std(ddof=1)
                )
                if len(g) > 1
                else 0.0,

            "min_accuracy_change":
                float(
                    g[
                        "accuracy_change"
                    ].min()
                ),

            "max_accuracy_change":
                float(
                    g[
                        "accuracy_change"
                    ].max()
                ),

            "mean_post_accuracy":
                float(
                    g[
                        "post_accuracy"
                    ].mean()
                ),

            "mean_repairs":
                float(
                    g[
                        "repairs"
                    ].mean()
                ),

            "mean_breaks":
                float(
                    g[
                        "breaks"
                    ].mean()
                ),

            "mean_net_repairs":
                float(
                    g[
                        "net_repairs"
                    ].mean()
                ),

            "mean_intervention_rate":
                float(
                    g[
                        "intervention_rate"
                    ].mean()
                ),

            "fraction_runs_positive_gain":
                float(
                    (
                        g[
                            "accuracy_change"
                        ]
                        >
                        0
                    ).mean()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_DIR
    / "adaptation_curve_summary.csv",
    index=False,
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "analysis_type":
        "post_hoc_natural_adaptation_sample_efficiency",

    "independent_confirmation":
        False,

    "model_inference_performed":
        False,

    "hyperparameter_search":
        False,

    "calibration_sizes":
        CALIBRATION_SIZES,

    "outer_folds":
        N_OUTER_FOLDS,

    "repeats_per_subsample_size":
        N_REPEATS,

    "ridge_alpha":
        ridge_alpha,

    "threshold":
        threshold,

    "feature_count":
        len(feature_names),

    "calibration_sampling":
        "balanced Simple/Complex random subsample of outer training fold",

    "outer_split":
        "5-fold StratifiedKFold by Simple/Complex",
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
print("NATURAL ADAPTATION SAMPLE-EFFICIENCY CURVE")
print("=" * 118)

print(
    summary.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("REFERENCE POINTS")
print("=" * 118)

print(
    "Baseline accuracy:",
    baseline_accuracy,
)

print(
    "Frozen synthetic controller overall gain:",
    float(
        frozen_post_correct.mean()
        -
        baseline_correct.mean()
    ),
)

full = summary[
    summary[
        "calibration_size"
    ]
    ==
    3200
].iloc[0]

print(
    "Full natural outer-train gain:",
    float(
        full[
            "mean_accuracy_change"
        ]
    ),
)

print(
    "Natural multi-action oracle ceiling:",
    0.06725,
)


print("\n" + "=" * 118)
print("SAMPLE EFFICIENCY")
print("=" * 118)

for _, row in summary.iterrows():

    print(
        f"K={int(row['calibration_size']):4d} | "
        f"gain={100*row['mean_accuracy_change']:+.3f} pp | "
        f"std={100*row['std_accuracy_change']:.3f} pp | "
        f"positive-runs="
        f"{100*row['fraction_runs_positive_gain']:.1f}% | "
        f"intervene="
        f"{100*row['mean_intervention_rate']:.2f}%"
    )


print("\n" + "=" * 118)
print("PROBE COMPLETE")
print("=" * 118)

print(
    "POST-HOC DEVELOPMENT ONLY."
)

print(
    "No VLM inference and no hyperparameter search."
)

print("\nSaved to:")
print(OUT_DIR)

