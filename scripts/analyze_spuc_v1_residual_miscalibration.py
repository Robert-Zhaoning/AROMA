#!/usr/bin/env python3

"""
AROMA Phase II-A
SPUC-v1 Residual Miscalibration Analysis

Purpose:
After empirical-support projection has removed catastrophic Ridge
extrapolation, determine why natural-domain action routing remains
incorrect.

This analysis measures:

1. original vs SPUC per-action score/utility calibration;
2. selected-policy calibration;
3. original -> SPUC action transitions;
4. outcome changes caused by those transitions;
5. performance inside vs outside v2 empirical feature support.

NO model inference.
NO controller fitting.
NO threshold tuning.
NO support tuning.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

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

SUPPORT_PATH = (
    ROOT
    / "outputs/proc_count_causal_v2/controller/"
    "training_support_reconstruction/"
    "v2_controller_training_support.csv"
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

SPUC_SAMPLE_PATH = (
    ROOT
    / "outputs/phase2_spuc_v1/"
    "tallyqa_spuc_v1_per_sample.csv"
)

OUT_DIR = (
    ROOT
    / "outputs/phase2_spuc_v1/"
    "residual_miscalibration"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

THRESHOLD = 0.1

ACTIONS = [
    0.0,
    1.5,
    2.0,
    4.0,
]


def as_bool(s):

    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s).fillna(0).ne(0)

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

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(
        spearmanr(
            x,
            y,
        ).statistic
    )


bundle = joblib.load(
    BUNDLE_PATH
)

support = pd.read_csv(
    SUPPORT_PATH
)

tally = pd.read_csv(
    TALLY_PATH
)

matrix = pd.read_csv(
    MATRIX_PATH
)

spuc_sample = pd.read_csv(
    SPUC_SAMPLE_PATH
)


# ============================================================
# Frozen feature specification
# ============================================================

feature_names = list(
    bundle["feature_names"]
)

feature_cols = [
    f"feature__{f}"
    for f in feature_names
]

X = (
    tally[
        feature_cols
    ]
    .to_numpy(
        dtype=float
    )
)

support = (
    support
    .set_index("feature")
    .reindex(feature_names)
)

z_lo = (
    support["z_min"]
    .to_numpy(dtype=float)
)

z_hi = (
    support["z_max"]
    .to_numpy(dtype=float)
)


# common scaler
scaler = (
    bundle["models"][0.0]
    .named_steps["scale"]
)

mu = np.asarray(
    scaler.mean_,
    dtype=float,
)

sigma = np.asarray(
    scaler.scale_,
    dtype=float,
)

Z = (
    X - mu
) / sigma

Zp = np.clip(
    Z,
    z_lo,
    z_hi,
)

clip_mask = (
    ~np.isclose(
        Z,
        Zp,
        rtol=0.0,
        atol=1e-12,
    )
)

any_clipped = (
    clip_mask.any(axis=1)
)


# ============================================================
# Original + SPUC per-action scores
# ============================================================

orig_scores = {}
spuc_scores = {}

for alpha in ACTIONS:

    pipe = bundle["models"][alpha]

    ridge = (
        pipe.named_steps["ridge"]
    )

    orig_scores[alpha] = np.asarray(
        pipe.predict(X),
        dtype=float,
    )

    spuc_scores[alpha] = (
        float(ridge.intercept_)
        +
        Zp.dot(
            np.asarray(
                ridge.coef_,
                dtype=float,
            )
        )
    )


# ============================================================
# Realized per-action utilities
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


# Align by question_id.
question_ids = (
    tally["question_id"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# Per-action score calibration
# ============================================================

action_rows = []

for alpha in ACTIONS:

    outcomes = (
        matrix[
            np.isclose(
                matrix["alpha"],
                alpha,
            )
        ][
            [
                "question_id",
                "utility",
                "baseline_correct_bool",
                "action_correct_bool",
            ]
        ]
        .copy()
    )

    frame = pd.DataFrame(
        {
            "question_id":
                question_ids,

            "original_score":
                orig_scores[alpha],

            "spuc_score":
                spuc_scores[alpha],
        }
    )

    frame = frame.merge(
        outcomes,
        on="question_id",
        how="inner",
    )

    if len(frame) != 4000:
        raise RuntimeError(
            f"alpha={alpha}: merge failed"
        )

    for policy in [
        "original",
        "spuc_v1",
    ]:

        score_col = (
            "original_score"
            if policy == "original"
            else "spuc_score"
        )

        score = frame[score_col]
        utility = frame["utility"]

        trigger = (
            score > THRESHOLD
        )

        g = frame[
            trigger
        ]

        repairs = int(
            (
                (~g["baseline_correct_bool"])
                &
                g["action_correct_bool"]
            ).sum()
        )

        breaks = int(
            (
                g["baseline_correct_bool"]
                &
                (~g["action_correct_bool"])
            ).sum()
        )

        action_rows.append(
            {
                "alpha":
                    alpha,

                "policy":
                    policy,

                "mean_score":
                    float(
                        score.mean()
                    ),

                "median_score":
                    float(
                        score.median()
                    ),

                "fraction_gt_threshold":
                    float(
                        trigger.mean()
                    ),

                "score_utility_spearman":
                    safe_spearman(
                        score,
                        utility,
                    ),

                "mean_utility_all":
                    float(
                        utility.mean()
                    ),

                "mean_utility_triggered":
                    (
                        float(
                            g["utility"].mean()
                        )
                        if len(g)
                        else 0.0
                    ),

                "triggered_n":
                    len(g),

                "triggered_repairs":
                    repairs,

                "triggered_breaks":
                    breaks,

                "triggered_net":
                    repairs - breaks,
            }
        )


action_calibration = pd.DataFrame(
    action_rows
)

action_calibration.to_csv(
    OUT_DIR
    / "per_action_original_vs_spuc_calibration.csv",
    index=False,
)


# ============================================================
# Selected-policy calibration
# ============================================================

final = (
    tally[
        [
            "question_id",
            "ground_truth",
            "baseline_prediction",
            "post_correct",
            "baseline_correct",
            "selected_alpha",
            "selected_score",
        ]
    ]
    .merge(
        spuc_sample[
            [
                "question_id",
                "spuc_selected_alpha",
                "spuc_selected_score",
                "spuc_correct",
            ]
        ],
        on="question_id",
        how="inner",
    )
)

if len(final) != 4000:
    raise RuntimeError(
        "Selected-policy merge failed."
    )


base_correct = as_bool(
    final["baseline_correct"]
)

orig_correct = as_bool(
    final["post_correct"]
)

spuc_correct = as_bool(
    final["spuc_correct"]
)

final["original_utility"] = (
    orig_correct.astype(int)
    -
    base_correct.astype(int)
)

final["spuc_utility"] = (
    spuc_correct.astype(int)
    -
    base_correct.astype(int)
)


BINS = [
    -np.inf,
    0.0,
    0.05,
    0.10,
    0.20,
    0.40,
    1.0,
    np.inf,
]

LABELS = [
    "<=0",
    "(0,.05]",
    "(.05,.10]",
    "(.10,.20]",
    "(.20,.40]",
    "(.40,1]",
    ">1",
]

selected_rows = []

for policy in [
    "original",
    "spuc_v1",
]:

    score_col = (
        "selected_score"
        if policy == "original"
        else "spuc_selected_score"
    )

    utility_col = (
        "original_utility"
        if policy == "original"
        else "spuc_utility"
    )

    tmp = final.copy()

    tmp["bin"] = pd.cut(
        tmp[score_col],
        bins=BINS,
        labels=LABELS,
        include_lowest=True,
    )

    for label in LABELS:

        g = tmp[
            tmp["bin"] == label
        ]

        if len(g) == 0:
            continue

        selected_rows.append(
            {
                "policy":
                    policy,

                "score_bin":
                    label,

                "n":
                    len(g),

                "mean_score":
                    float(
                        g[score_col].mean()
                    ),

                "mean_realized_utility":
                    float(
                        g[utility_col].mean()
                    ),

                "positive_utility":
                    int(
                        (
                            g[utility_col] > 0
                        ).sum()
                    ),

                "negative_utility":
                    int(
                        (
                            g[utility_col] < 0
                        ).sum()
                    ),

                "net_utility":
                    int(
                        g[utility_col].sum()
                    ),
            }
        )


selected_calibration = pd.DataFrame(
    selected_rows
)

selected_calibration.to_csv(
    OUT_DIR
    / "selected_policy_score_calibration.csv",
    index=False,
)


# ============================================================
# Original -> SPUC action transition matrix
# ============================================================

final[
    "selected_alpha"
] = pd.to_numeric(
    final[
        "selected_alpha"
    ]
).astype(float)

final[
    "spuc_selected_alpha"
] = pd.to_numeric(
    final[
        "spuc_selected_alpha"
    ]
).astype(float)


transition_rows = []

for old in [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]:

    for new in [
        0.0,
        1.0,
        1.5,
        2.0,
        4.0,
    ]:

        g = final[
            np.isclose(
                final[
                    "selected_alpha"
                ],
                old,
            )
            &
            np.isclose(
                final[
                    "spuc_selected_alpha"
                ],
                new,
            )
        ]

        if len(g) == 0:
            continue

        orig = as_bool(
            g[
                "post_correct"
            ]
        )

        newc = as_bool(
            g[
                "spuc_correct"
            ]
        )

        transition_rows.append(
            {
                "original_alpha":
                    old,

                "spuc_alpha":
                    new,

                "n":
                    len(g),

                "original_correct":
                    int(
                        orig.sum()
                    ),

                "spuc_correct":
                    int(
                        newc.sum()
                    ),

                "delta_correct":
                    int(
                        newc.sum()
                        -
                        orig.sum()
                    ),

                "improved_samples":
                    int(
                        (
                            (~orig)
                            &
                            newc
                        ).sum()
                    ),

                "worsened_samples":
                    int(
                        (
                            orig
                            &
                            (~newc)
                        ).sum()
                    ),
            }
        )


transitions = pd.DataFrame(
    transition_rows
)

transitions.to_csv(
    OUT_DIR
    / "original_to_spuc_action_transitions.csv",
    index=False,
)


# ============================================================
# In-support vs out-of-support strata
# ============================================================

clip_frame = pd.DataFrame(
    {
        "question_id":
            question_ids,

        "any_clipped":
            any_clipped,

        "n_clipped_features":
            clip_mask.sum(
                axis=1
            ),
    }
)

strata = (
    final.merge(
        clip_frame,
        on="question_id",
        how="inner",
    )
)

strata_rows = []

for label, mask in [
    (
        "fully_in_support",
        ~strata[
            "any_clipped"
        ],
    ),
    (
        "any_out_of_support",
        strata[
            "any_clipped"
        ],
    ),
]:

    g = strata[
        mask
    ]

    b = as_bool(
        g["baseline_correct"]
    )

    o = as_bool(
        g["post_correct"]
    )

    s = as_bool(
        g["spuc_correct"]
    )

    strata_rows.append(
        {
            "support_stratum":
                label,

            "n":
                len(g),

            "baseline_accuracy":
                float(
                    b.mean()
                ),

            "original_accuracy":
                float(
                    o.mean()
                ),

            "spuc_accuracy":
                float(
                    s.mean()
                ),

            "original_gain":
                float(
                    o.mean()
                    -
                    b.mean()
                ),

            "spuc_gain":
                float(
                    s.mean()
                    -
                    b.mean()
                ),

            "original_repairs":
                int(
                    (
                        (~b)
                        &
                        o
                    ).sum()
                ),

            "original_breaks":
                int(
                    b
                    &
                    (~o)
                .sum()
                )
                if False
                else int(
                    (
                        b
                        &
                        (~o)
                    ).sum()
                ),

            "spuc_repairs":
                int(
                    (
                        (~b)
                        &
                        s
                    ).sum()
                ),

            "spuc_breaks":
                int(
                    (
                        b
                        &
                        (~s)
                    ).sum()
                ),

            "policy_change_rate":
                float(
                    (
                        ~np.isclose(
                            g[
                                "selected_alpha"
                            ],
                            g[
                                "spuc_selected_alpha"
                            ],
                        )
                    ).mean()
                ),

            "mean_n_clipped_features":
                float(
                    g[
                        "n_clipped_features"
                    ].mean()
                ),
        }
    )


support_strata = pd.DataFrame(
    strata_rows
)

support_strata.to_csv(
    OUT_DIR
    / "support_stratified_outcomes.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

print("=" * 118)
print("PER-ACTION ORIGINAL VS SPUC CALIBRATION")
print("=" * 118)

print(
    action_calibration.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("SELECTED-POLICY SCORE CALIBRATION")
print("=" * 118)

print(
    selected_calibration.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("ORIGINAL -> SPUC ACTION TRANSITIONS")
print("=" * 118)

print(
    transitions.sort_values(
        "n",
        ascending=False,
    ).to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("IN-SUPPORT VS OUT-OF-SUPPORT OUTCOMES")
print("=" * 118)

print(
    support_strata.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("RESIDUAL MISCALIBRATION ANALYSIS COMPLETE")
print("=" * 118)

print("No model inference performed.")
print("No controller fitting performed.")
print("No hyperparameter tuning performed.")

print("\nSaved to:")
print(OUT_DIR)

