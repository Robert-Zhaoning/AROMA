#!/usr/bin/env python3

"""
AROMA Controller Domain-Shift Forensics
=======================================

Post-hoc diagnostic analysis of the frozen synthetic-trained
cardinality controller across:

    v2 development OOF
        ->
    v3 untouched procedural confirmation
        ->
    TallyQA natural-OOD evaluation

The analysis separates:

1. controller score calibration,
2. intervention-rate shift,
3. per-action score distribution shift,
4. realized natural-domain utility,
5. 39-feature distribution shift.

IMPORTANT
---------
This script:
- performs NO model inference,
- performs NO controller fitting,
- performs NO hyperparameter tuning,
- modifies NO frozen artifacts,
- creates post-hoc forensic summaries only.
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp, spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# Paths
# ============================================================

V2_PATH = Path(
    "outputs/proc_count_causal_v2/controller/"
    "final_frozen_controller/"
    "selected_hyperparameter_oof.csv"
)

V3_PATH = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller/"
    "v3_final_results.csv"
)

TALLY_PATH = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "final_frozen_controller/"
    "tallyqa_final_results.csv"
)

TALLY_MATRIX_PATH = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "forensics/action_oracle/"
    "action_matrix.csv"
)

OUT_DIR = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "forensics/controller_domain_shift"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

THRESHOLD = 0.1

ACTION_SCORE_COLS = {
    0.0: "score_alpha_0",
    1.5: "score_alpha_1p5",
    2.0: "score_alpha_2",
    4.0: "score_alpha_4",
}


# ============================================================
# Helpers
# ============================================================

def as_bool(series):
    """
    Robust conversion of CSV bool-like columns.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).ne(0)

    s = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    unknown = set(s.unique()) - set(mapping)

    if unknown:
        raise ValueError(
            f"Cannot safely convert to bool. "
            f"Unknown values: {sorted(unknown)}"
        )

    return s.map(mapping).astype(bool)


def utility_from_correctness(
    baseline_correct,
    post_correct,
):
    b = as_bool(
        baseline_correct
    ).astype(int)

    p = as_bool(
        post_correct
    ).astype(int)

    return (
        p - b
    ).astype(int)


def safe_spearman(x, y):
    if not SCIPY_AVAILABLE:
        return np.nan

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(x) < 2
        or
        np.nanstd(x) == 0
        or
        np.nanstd(y) == 0
    ):
        return np.nan

    r = spearmanr(
        x,
        y,
        nan_policy="omit",
    )

    return float(
        r.statistic
    )


def quantile_summary(x):
    x = pd.Series(
        x,
        dtype=float,
    ).dropna()

    return {
        "mean":
            float(
                x.mean()
            ),

        "std":
            float(
                x.std(ddof=1)
            ),

        "q05":
            float(
                x.quantile(0.05)
            ),

        "q10":
            float(
                x.quantile(0.10)
            ),

        "q25":
            float(
                x.quantile(0.25)
            ),

        "median":
            float(
                x.quantile(0.50)
            ),

        "q75":
            float(
                x.quantile(0.75)
            ),

        "q90":
            float(
                x.quantile(0.90)
            ),

        "q95":
            float(
                x.quantile(0.95)
            ),
    }


# ============================================================
# Load data
# ============================================================

for p in [
    V2_PATH,
    V3_PATH,
    TALLY_PATH,
    TALLY_MATRIX_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(
            p
        )

v2 = pd.read_csv(
    V2_PATH
)

v3 = pd.read_csv(
    V3_PATH
)

tally = pd.read_csv(
    TALLY_PATH
)

matrix = pd.read_csv(
    TALLY_MATRIX_PATH
)


# ============================================================
# Basic schema audits
# ============================================================

v3_features = sorted(
    c
    for c in v3.columns
    if c.startswith(
        "feature__"
    )
)

tally_features = sorted(
    c
    for c in tally.columns
    if c.startswith(
        "feature__"
    )
)

if len(v3_features) != 39:
    raise RuntimeError(
        f"Expected 39 v3 features; "
        f"found {len(v3_features)}."
    )

if len(tally_features) != 39:
    raise RuntimeError(
        f"Expected 39 TallyQA features; "
        f"found {len(tally_features)}."
    )

if v3_features != tally_features:
    raise RuntimeError(
        "v3/TallyQA feature sets differ."
    )

for df_name, df in [
    ("v3", v3),
    ("TallyQA", tally),
]:
    for col in ACTION_SCORE_COLS.values():
        if col not in df.columns:
            raise RuntimeError(
                f"{df_name} missing {col}"
            )

for df_name, df in [
    ("v2", v2),
    ("v3", v3),
    ("TallyQA", tally),
]:
    for col in [
        "selected_alpha",
        "selected_score",
        "baseline_correct",
    ]:
        if col not in df.columns:
            raise RuntimeError(
                f"{df_name} missing {col}"
            )


# ============================================================
# Prepare selected-action utility
# ============================================================

v2["selected_alpha"] = pd.to_numeric(
    v2["selected_alpha"]
)

v3["selected_alpha"] = pd.to_numeric(
    v3["selected_alpha"]
)

tally["selected_alpha"] = pd.to_numeric(
    tally["selected_alpha"]
)

if "selected_utility" in v2.columns:
    v2["realized_selected_utility"] = (
        pd.to_numeric(
            v2["selected_utility"]
        )
        .astype(int)
    )
else:
    raise RuntimeError(
        "v2 final OOF artifact lacks "
        "selected_utility."
    )


# v2 post correctness can be reconstructed exactly
# from baseline correctness + {-1,0,+1} utility.
v2_base_int = (
    as_bool(
        v2["baseline_correct"]
    )
    .astype(int)
)

v2["post_correct_reconstructed"] = (
    v2_base_int
    +
    v2[
        "realized_selected_utility"
    ]
)

if not v2[
    "post_correct_reconstructed"
].isin(
    [0, 1]
).all():
    raise RuntimeError(
        "Invalid reconstructed v2 post correctness."
    )


for df in [
    v3,
    tally,
]:
    if "post_correct" not in df.columns:
        raise RuntimeError(
            "Final result artifact missing "
            "post_correct."
        )

    df[
        "realized_selected_utility"
    ] = utility_from_correctness(
        df[
            "baseline_correct"
        ],
        df[
            "post_correct"
        ],
    )


# ============================================================
# 1. Domain-level selected-controller behavior
# ============================================================

domain_rows = []

for domain, df in [
    ("v2_oof", v2),
    ("v3_confirmation", v3),
    ("tallyqa_natural", tally),
]:

    baseline_correct = as_bool(
        df[
            "baseline_correct"
        ]
    )

    if domain == "v2_oof":
        post_correct = (
            df[
                "post_correct_reconstructed"
            ]
            .astype(bool)
        )
    else:
        post_correct = as_bool(
            df[
                "post_correct"
            ]
        )

    selected_score = pd.to_numeric(
        df[
            "selected_score"
        ],
        errors="coerce",
    )

    utility = pd.to_numeric(
        df[
            "realized_selected_utility"
        ],
        errors="coerce",
    )

    intervene = (
        df[
            "selected_alpha"
        ]
        .astype(float)
        .ne(1.0)
    )

    repair = (
        (~baseline_correct)
        &
        post_correct
    )

    break_case = (
        baseline_correct
        &
        (~post_correct)
    )

    score_q = quantile_summary(
        selected_score
    )

    intervened_utility = (
        utility[
            intervene
        ]
    )

    domain_rows.append(
        {
            "domain":
                domain,

            "n":
                len(df),

            "baseline_accuracy":
                float(
                    baseline_correct.mean()
                ),

            "post_accuracy":
                float(
                    post_correct.mean()
                ),

            "accuracy_change":
                float(
                    post_correct.mean()
                    -
                    baseline_correct.mean()
                ),

            "intervention_rate":
                float(
                    intervene.mean()
                ),

            "repairs":
                int(
                    repair.sum()
                ),

            "breaks":
                int(
                    break_case.sum()
                ),

            "net_repairs":
                int(
                    repair.sum()
                    -
                    break_case.sum()
                ),

            "selected_score_mean":
                score_q["mean"],

            "selected_score_median":
                score_q["median"],

            "selected_score_q10":
                score_q["q10"],

            "selected_score_q90":
                score_q["q90"],

            "fraction_selected_score_gt_threshold":
                float(
                    (
                        selected_score
                        >
                        THRESHOLD
                    ).mean()
                ),

            "mean_realized_selected_utility":
                float(
                    utility.mean()
                ),

            "mean_realized_utility_when_intervened":
                (
                    float(
                        intervened_utility.mean()
                    )
                    if len(
                        intervened_utility
                    )
                    else 0.0
                ),

            "score_utility_spearman":
                safe_spearman(
                    selected_score,
                    utility,
                ),
        }
    )

domain_summary = pd.DataFrame(
    domain_rows
)

domain_summary.to_csv(
    OUT_DIR
    / "domain_selected_controller_summary.csv",
    index=False,
)


# ============================================================
# 2. Selected-score calibration by fixed score bins
# ============================================================

SCORE_BINS = [
    -np.inf,
    0.0,
    0.05,
    0.10,
    0.20,
    0.40,
    np.inf,
]

SCORE_LABELS = [
    "<=0",
    "(0,0.05]",
    "(0.05,0.10]",
    "(0.10,0.20]",
    "(0.20,0.40]",
    ">0.40",
]

calibration_rows = []

for domain, df in [
    ("v2_oof", v2),
    ("v3_confirmation", v3),
    ("tallyqa_natural", tally),
]:

    tmp = df.copy()

    tmp["score_bin"] = pd.cut(
        pd.to_numeric(
            tmp[
                "selected_score"
            ],
            errors="coerce",
        ),
        bins=SCORE_BINS,
        labels=SCORE_LABELS,
        include_lowest=True,
        right=True,
    )

    baseline_correct = as_bool(
        tmp[
            "baseline_correct"
        ]
    )

    if domain == "v2_oof":
        post_correct = (
            tmp[
                "post_correct_reconstructed"
            ]
            .astype(bool)
        )
    else:
        post_correct = as_bool(
            tmp[
                "post_correct"
            ]
        )

    tmp[
        "_base"
    ] = baseline_correct.values

    tmp[
        "_post"
    ] = post_correct.values

    tmp[
        "_utility"
    ] = (
        tmp[
            "realized_selected_utility"
        ]
        .astype(float)
    )

    for label in SCORE_LABELS:

        g = tmp[
            tmp[
                "score_bin"
            ]
            ==
            label
        ]

        if len(g) == 0:
            continue

        repairs = int(
            (
                (~g["_base"])
                &
                g["_post"]
            ).sum()
        )

        breaks = int(
            (
                g["_base"]
                &
                (~g["_post"])
            ).sum()
        )

        calibration_rows.append(
            {
                "domain":
                    domain,

                "score_bin":
                    label,

                "n":
                    len(g),

                "mean_score":
                    float(
                        g[
                            "selected_score"
                        ].mean()
                    ),

                "mean_realized_utility":
                    float(
                        g[
                            "_utility"
                        ].mean()
                    ),

                "repairs":
                    repairs,

                "breaks":
                    breaks,

                "net_repairs":
                    repairs
                    -
                    breaks,

                "baseline_accuracy":
                    float(
                        g[
                            "_base"
                        ].mean()
                    ),

                "post_accuracy":
                    float(
                        g[
                            "_post"
                        ].mean()
                    ),
            }
        )

selected_calibration = pd.DataFrame(
    calibration_rows
)

selected_calibration.to_csv(
    OUT_DIR
    / "selected_score_calibration_by_domain.csv",
    index=False,
)


# ============================================================
# 3. Per-action score distribution: v3 vs TallyQA
# ============================================================

score_shift_rows = []

for alpha, score_col in ACTION_SCORE_COLS.items():

    v3_score = pd.to_numeric(
        v3[
            score_col
        ],
        errors="coerce",
    )

    tq_score = pd.to_numeric(
        tally[
            score_col
        ],
        errors="coerce",
    )

    v3_q = quantile_summary(
        v3_score
    )

    tq_q = quantile_summary(
        tq_score
    )

    if SCIPY_AVAILABLE:
        ks = ks_2samp(
            v3_score.dropna(),
            tq_score.dropna(),
        )

        ks_stat = float(
            ks.statistic
        )

        ks_p = float(
            ks.pvalue
        )
    else:
        ks_stat = np.nan
        ks_p = np.nan

    score_shift_rows.append(
        {
            "alpha":
                alpha,

            "score_column":
                score_col,

            "v3_mean":
                v3_q["mean"],

            "tally_mean":
                tq_q["mean"],

            "delta_mean_tally_minus_v3":
                tq_q["mean"]
                -
                v3_q["mean"],

            "v3_median":
                v3_q["median"],

            "tally_median":
                tq_q["median"],

            "delta_median_tally_minus_v3":
                tq_q["median"]
                -
                v3_q["median"],

            "v3_q90":
                v3_q["q90"],

            "tally_q90":
                tq_q["q90"],

            "v3_fraction_gt_threshold":
                float(
                    (
                        v3_score
                        >
                        THRESHOLD
                    ).mean()
                ),

            "tally_fraction_gt_threshold":
                float(
                    (
                        tq_score
                        >
                        THRESHOLD
                    ).mean()
                ),

            "delta_fraction_gt_threshold":
                float(
                    (
                        tq_score
                        >
                        THRESHOLD
                    ).mean()
                    -
                    (
                        v3_score
                        >
                        THRESHOLD
                    ).mean()
                ),

            "ks_statistic":
                ks_stat,

            "ks_pvalue":
                ks_p,
        }
    )

action_score_shift = pd.DataFrame(
    score_shift_rows
)

action_score_shift.to_csv(
    OUT_DIR
    / "per_action_score_shift_v3_vs_tallyqa.csv",
    index=False,
)


# ============================================================
# 4. TallyQA realized utility of each action
#    versus controller-predicted score
# ============================================================

# Prefer question_id because it exists in the frozen
# action matrix and final result artifact.
if (
    "question_id"
    not in tally.columns
    or
    "question_id"
    not in matrix.columns
):
    raise RuntimeError(
        "question_id required for TallyQA "
        "score/action-matrix merge."
    )

matrix[
    "alpha"
] = pd.to_numeric(
    matrix[
        "alpha"
    ]
)

matrix_base = as_bool(
    matrix[
        "baseline_correct"
    ]
)

matrix_action = as_bool(
    matrix[
        "action_correct"
    ]
)

matrix[
    "_utility"
] = (
    matrix_action.astype(int)
    -
    matrix_base.astype(int)
)

tally_action_rows = []

for alpha, score_col in ACTION_SCORE_COLS.items():

    scores = tally[
        [
            "question_id",
            score_col,
        ]
    ].copy()

    scores = scores.rename(
        columns={
            score_col:
                "predicted_score"
        }
    )

    outcomes = matrix[
        np.isclose(
            matrix[
                "alpha"
            ],
            alpha,
        )
    ][
        [
            "question_id",
            "_utility",
            "baseline_correct",
            "action_correct",
        ]
    ].copy()

    merged = scores.merge(
        outcomes,
        on="question_id",
        how="inner",
    )

    if len(merged) != len(tally):
        raise RuntimeError(
            f"alpha={alpha}: expected "
            f"{len(tally)} merged rows; "
            f"found {len(merged)}."
        )

    predicted_score = pd.to_numeric(
        merged[
            "predicted_score"
        ],
        errors="coerce",
    )

    utility = pd.to_numeric(
        merged[
            "_utility"
        ],
        errors="coerce",
    )

    trigger = (
        predicted_score
        >
        THRESHOLD
    )

    triggered = merged[
        trigger
    ].copy()

    if len(triggered):
        trig_base = as_bool(
            triggered[
                "baseline_correct"
            ]
        )

        trig_action = as_bool(
            triggered[
                "action_correct"
            ]
        )

        trig_repairs = int(
            (
                (~trig_base)
                &
                trig_action
            ).sum()
        )

        trig_breaks = int(
            (
                trig_base
                &
                (~trig_action)
            ).sum()
        )
    else:
        trig_repairs = 0
        trig_breaks = 0

    tally_action_rows.append(
        {
            "alpha":
                alpha,

            "n":
                len(merged),

            "mean_predicted_score":
                float(
                    predicted_score.mean()
                ),

            "median_predicted_score":
                float(
                    predicted_score.median()
                ),

            "fraction_score_gt_threshold":
                float(
                    trigger.mean()
                ),

            "mean_realized_utility_all":
                float(
                    utility.mean()
                ),

            "mean_realized_utility_when_score_gt_threshold":
                (
                    float(
                        utility[
                            trigger
                        ].mean()
                    )
                    if trigger.sum()
                    else 0.0
                ),

            "score_utility_spearman":
                safe_spearman(
                    predicted_score,
                    utility,
                ),

            "triggered_n":
                int(
                    trigger.sum()
                ),

            "triggered_repairs":
                trig_repairs,

            "triggered_breaks":
                trig_breaks,

            "triggered_net_repairs":
                trig_repairs
                -
                trig_breaks,

            "triggered_repair_fraction":
                (
                    trig_repairs
                    /
                    trigger.sum()
                    if trigger.sum()
                    else 0.0
                ),

            "triggered_break_fraction":
                (
                    trig_breaks
                    /
                    trigger.sum()
                    if trigger.sum()
                    else 0.0
                ),
        }
    )

tally_action_calibration = pd.DataFrame(
    tally_action_rows
)

tally_action_calibration.to_csv(
    OUT_DIR
    / "tallyqa_per_action_score_utility_calibration.csv",
    index=False,
)


# ============================================================
# 5. v3 vs TallyQA 39-feature distribution shift
# ============================================================

def feature_shift_table(
    v3_sub,
    tally_sub,
    group_name,
):

    rows = []

    for col in v3_features:

        a = pd.to_numeric(
            v3_sub[
                col
            ],
            errors="coerce",
        ).dropna()

        b = pd.to_numeric(
            tally_sub[
                col
            ],
            errors="coerce",
        ).dropna()

        mean_a = float(
            a.mean()
        )

        mean_b = float(
            b.mean()
        )

        std_a = float(
            a.std(ddof=1)
        )

        std_b = float(
            b.std(ddof=1)
        )

        pooled_sd = math.sqrt(
            (
                std_a ** 2
                +
                std_b ** 2
            )
            /
            2.0
        )

        if pooled_sd > 0:
            smd = (
                mean_b
                -
                mean_a
            ) / pooled_sd
        else:
            smd = (
                0.0
                if mean_a == mean_b
                else np.nan
            )

        median_a = float(
            a.median()
        )

        median_b = float(
            b.median()
        )

        iqr_a = float(
            a.quantile(0.75)
            -
            a.quantile(0.25)
        )

        iqr_b = float(
            b.quantile(0.75)
            -
            b.quantile(0.25)
        )

        robust_scale = (
            (
                iqr_a
                +
                iqr_b
            )
            /
            2.0
        )

        robust_median_shift = (
            (
                median_b
                -
                median_a
            )
            /
            robust_scale
            if robust_scale > 0
            else (
                0.0
                if median_a
                ==
                median_b
                else np.nan
            )
        )

        if SCIPY_AVAILABLE:
            ks = ks_2samp(
                a,
                b,
            )

            ks_stat = float(
                ks.statistic
            )

            ks_p = float(
                ks.pvalue
            )
        else:
            ks_stat = np.nan
            ks_p = np.nan

        rows.append(
            {
                "group":
                    group_name,

                "feature":
                    col.replace(
                        "feature__",
                        "",
                    ),

                "v3_mean":
                    mean_a,

                "tally_mean":
                    mean_b,

                "mean_difference":
                    mean_b
                    -
                    mean_a,

                "v3_std":
                    std_a,

                "tally_std":
                    std_b,

                "smd_tally_minus_v3":
                    smd,

                "abs_smd":
                    abs(smd)
                    if np.isfinite(smd)
                    else np.nan,

                "v3_median":
                    median_a,

                "tally_median":
                    median_b,

                "robust_median_shift":
                    robust_median_shift,

                "ks_statistic":
                    ks_stat,

                "ks_pvalue":
                    ks_p,
            }
        )

    out = pd.DataFrame(
        rows
    )

    out = out.sort_values(
        [
            "abs_smd",
            "ks_statistic",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return out


v3_base = as_bool(
    v3[
        "baseline_correct"
    ]
)

tq_base = as_bool(
    tally[
        "baseline_correct"
    ]
)

feature_shift_overall = (
    feature_shift_table(
        v3,
        tally,
        "overall",
    )
)

feature_shift_wrong = (
    feature_shift_table(
        v3[
            ~v3_base
        ],
        tally[
            ~tq_base
        ],
        "baseline_wrong",
    )
)

feature_shift_correct = (
    feature_shift_table(
        v3[
            v3_base
        ],
        tally[
            tq_base
        ],
        "baseline_correct",
    )
)

feature_shift_overall.to_csv(
    OUT_DIR
    / "feature_shift_v3_vs_tallyqa_overall.csv",
    index=False,
)

feature_shift_wrong.to_csv(
    OUT_DIR
    / "feature_shift_v3_vs_tallyqa_baseline_wrong.csv",
    index=False,
)

feature_shift_correct.to_csv(
    OUT_DIR
    / "feature_shift_v3_vs_tallyqa_baseline_correct.csv",
    index=False,
)


# ============================================================
# 6. Selected action distribution by domain
# ============================================================

action_dist_rows = []

for domain, df in [
    ("v2_oof", v2),
    ("v3_confirmation", v3),
    ("tallyqa_natural", tally),
]:

    total = len(df)

    for alpha in [
        0.0,
        1.0,
        1.5,
        2.0,
        4.0,
    ]:

        count = int(
            np.isclose(
                df[
                    "selected_alpha"
                ].astype(float),
                alpha,
            ).sum()
        )

        action_dist_rows.append(
            {
                "domain":
                    domain,

                "alpha":
                    alpha,

                "count":
                    count,

                "fraction":
                    count
                    /
                    total,
            }
        )

action_distribution = pd.DataFrame(
    action_dist_rows
)

action_distribution.to_csv(
    OUT_DIR
    / "selected_action_distribution_by_domain.csv",
    index=False,
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "analysis_type":
        "post_hoc_controller_domain_shift_forensics",

    "model_inference_performed":
        False,

    "controller_fitting_performed":
        False,

    "hyperparameter_tuning_performed":
        False,

    "frozen_artifacts_modified":
        False,

    "threshold":
        THRESHOLD,

    "v2_rows":
        int(
            len(v2)
        ),

    "v3_rows":
        int(
            len(v3)
        ),

    "tallyqa_rows":
        int(
            len(tally)
        ),

    "feature_count":
        int(
            len(v3_features)
        ),

    "scipy_available":
        SCIPY_AVAILABLE,
}

with open(
    OUT_DIR
    / "analysis_metadata.json",
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        sort_keys=True,
    )


# ============================================================
# Print primary outputs
# ============================================================

print(
    "=" * 118
)

print(
    "DOMAIN-LEVEL FROZEN CONTROLLER SUMMARY"
)

print(
    "=" * 118
)

print(
    domain_summary.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "SELECTED-SCORE CALIBRATION BY DOMAIN"
)

print(
    "=" * 118
)

print(
    selected_calibration.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "PER-ACTION SCORE SHIFT: V3 -> TALLYQA"
)

print(
    "=" * 118
)

print(
    action_score_shift.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "TALLYQA PER-ACTION SCORE / REALIZED-UTILITY CALIBRATION"
)

print(
    "=" * 118
)

print(
    tally_action_calibration.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "TOP 15 FEATURE SHIFTS: V3 -> TALLYQA (OVERALL)"
)

print(
    "=" * 118
)

print(
    feature_shift_overall[
        [
            "feature",
            "v3_mean",
            "tally_mean",
            "smd_tally_minus_v3",
            "ks_statistic",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "TOP 15 FEATURE SHIFTS AMONG BASELINE-WRONG SAMPLES"
)

print(
    "=" * 118
)

print(
    feature_shift_wrong[
        [
            "feature",
            "v3_mean",
            "tally_mean",
            "smd_tally_minus_v3",
            "ks_statistic",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "SELECTED ACTION DISTRIBUTION BY DOMAIN"
)

print(
    "=" * 118
)

print(
    action_distribution.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "CONTROLLER DOMAIN-SHIFT FORENSICS COMPLETE"
)

print(
    "=" * 118
)

print(
    "No model inference performed."
)

print(
    "No controller fitting performed."
)

print(
    "No hyperparameter tuning performed."
)

print(
    "No frozen artifact modified."
)

print(
    "\nSaved to:"
)

print(
    OUT_DIR
)

