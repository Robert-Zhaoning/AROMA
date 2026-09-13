#!/usr/bin/env python3

"""
Post-hoc Natural-OOD Error-Direction Forensics for AROMA.

Purpose
-------
Characterize the signed baseline counting errors on the frozen
TallyQA natural-OOD evaluation and relate them to:

1. fixed L18H13 interventions,
2. multi-action oracle repairability,
3. frozen-controller action selection.

This script:
- performs NO model inference,
- performs NO controller fitting,
- performs NO hyperparameter tuning,
- modifies NO frozen artifacts.

All analyses are explicitly post-hoc forensic analyses.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "outputs/tallyqa_natural_ood_v1"
)

MATRIX_PATH = (
    ROOT
    / "forensics"
    / "action_oracle"
    / "action_matrix.csv"
)

ARCHIVED_PATH = (
    ROOT
    / "final_frozen_controller"
    / "tallyqa_final_results.csv"
)

OUT_DIR = (
    ROOT
    / "forensics"
    / "error_direction"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]


# ============================================================
# Load frozen artifacts
# ============================================================

matrix = pd.read_csv(
    MATRIX_PATH
)

archived = pd.read_csv(
    ARCHIVED_PATH
)

matrix["alpha"] = (
    matrix["alpha"]
    .astype(float)
)

archived["selected_alpha"] = (
    archived["selected_alpha"]
    .astype(float)
)


# ============================================================
# Basic integrity
# ============================================================

assert len(matrix) == 20000
assert matrix["question_id"].nunique() == 4000

for alpha in ACTIONS:
    assert (
        np.isclose(
            matrix["alpha"],
            alpha,
        ).sum()
        ==
        4000
    )

baseline = (
    matrix[
        np.isclose(
            matrix["alpha"],
            1.0,
        )
    ]
    .copy()
)

assert len(baseline) == 4000


# Recompute correctness directly from predictions.
baseline["baseline_correct_check"] = (
    baseline["baseline_prediction"]
    ==
    baseline["ground_truth"]
)

baseline["signed_error"] = (
    baseline["baseline_prediction"]
    -
    baseline["ground_truth"]
)

baseline["abs_error"] = (
    baseline["signed_error"]
    .abs()
)


def direction_from_error(e):
    if e < 0:
        return "undercount"
    if e > 0:
        return "overcount"
    return "correct"


baseline["error_direction"] = (
    baseline["signed_error"]
    .apply(
        direction_from_error
    )
)


def magnitude_bucket(e):
    e = abs(int(e))

    if e == 0:
        return "0"

    if e == 1:
        return "1"

    if e == 2:
        return "2"

    return "3+"


baseline["error_magnitude_bucket"] = (
    baseline["signed_error"]
    .apply(
        magnitude_bucket
    )
)


# ============================================================
# Archived/fresh reproducibility audit
# ============================================================

audit = (
    baseline[
        [
            "question_id",
            "baseline_prediction",
            "ground_truth",
        ]
    ]
    .merge(
        archived[
            [
                "question_id",
                "baseline_prediction",
                "ground_truth",
            ]
        ],
        on="question_id",
        how="inner",
        suffixes=(
            "_fresh",
            "_archived",
        ),
    )
)

assert len(audit) == 4000

pred_mismatch = int(
    (
        audit["baseline_prediction_fresh"]
        !=
        audit["baseline_prediction_archived"]
    ).sum()
)

gt_mismatch = int(
    (
        audit["ground_truth_fresh"]
        !=
        audit["ground_truth_archived"]
    ).sum()
)

assert pred_mismatch == 0
assert gt_mismatch == 0


# ============================================================
# 1. Baseline error-direction summary
# ============================================================

summary_rows = []

for subset_name, data in [
    ("overall", baseline),
    (
        "simple",
        baseline[
            baseline["subset"]
            ==
            "simple"
        ],
    ),
    (
        "complex",
        baseline[
            baseline["subset"]
            ==
            "complex"
        ],
    ),
]:

    wrong = (
        data[
            data["signed_error"]
            !=
            0
        ]
        .copy()
    )

    under = int(
        (
            wrong["signed_error"]
            <
            0
        ).sum()
    )

    over = int(
        (
            wrong["signed_error"]
            >
            0
        ).sum()
    )

    abs1 = int(
        (
            wrong["abs_error"]
            ==
            1
        ).sum()
    )

    abs2 = int(
        (
            wrong["abs_error"]
            ==
            2
        ).sum()
    )

    abs3plus = int(
        (
            wrong["abs_error"]
            >=
            3
        ).sum()
    )

    summary_rows.append(
        {
            "subset":
                subset_name,

            "n":
                len(data),

            "baseline_correct":
                int(
                    (
                        data["signed_error"]
                        ==
                        0
                    ).sum()
                ),

            "baseline_wrong":
                len(wrong),

            "undercount":
                under,

            "overcount":
                over,

            "undercount_fraction_wrong":
                (
                    under / len(wrong)
                    if len(wrong)
                    else 0.0
                ),

            "overcount_fraction_wrong":
                (
                    over / len(wrong)
                    if len(wrong)
                    else 0.0
                ),

            "abs_error_1":
                abs1,

            "abs_error_1_fraction_wrong":
                (
                    abs1 / len(wrong)
                    if len(wrong)
                    else 0.0
                ),

            "abs_error_2":
                abs2,

            "abs_error_3plus":
                abs3plus,

            "mean_signed_error_wrong":
                float(
                    wrong[
                        "signed_error"
                    ].mean()
                )
                if len(wrong)
                else 0.0,

            "median_signed_error_wrong":
                float(
                    wrong[
                        "signed_error"
                    ].median()
                )
                if len(wrong)
                else 0.0,

            "mean_abs_error_wrong":
                float(
                    wrong[
                        "abs_error"
                    ].mean()
                )
                if len(wrong)
                else 0.0,
        }
    )

direction_summary = pd.DataFrame(
    summary_rows
)

direction_summary.to_csv(
    OUT_DIR
    / "baseline_error_direction_summary.csv",
    index=False,
)


# ============================================================
# 2. Full signed-error spectrum
# ============================================================

spectrum_rows = []

for subset_name, data in [
    ("overall", baseline),
    (
        "simple",
        baseline[
            baseline["subset"]
            ==
            "simple"
        ],
    ),
    (
        "complex",
        baseline[
            baseline["subset"]
            ==
            "complex"
        ],
    ),
]:

    counts = (
        data[
            data["signed_error"]
            !=
            0
        ]["signed_error"]
        .value_counts()
        .sort_index()
    )

    total_wrong = int(
        counts.sum()
    )

    for error, count in counts.items():

        spectrum_rows.append(
            {
                "subset":
                    subset_name,

                "signed_error":
                    int(error),

                "count":
                    int(count),

                "fraction_of_wrong":
                    (
                        float(count)
                        /
                        total_wrong
                    ),
            }
        )

error_spectrum = pd.DataFrame(
    spectrum_rows
)

error_spectrum.to_csv(
    OUT_DIR
    / "baseline_error_spectrum.csv",
    index=False,
)


# ============================================================
# 3. Fixed-action repairability by error direction
# ============================================================

matrix["signed_error"] = (
    matrix["baseline_prediction"]
    -
    matrix["ground_truth"]
)

matrix["error_direction"] = (
    matrix["signed_error"]
    .apply(
        direction_from_error
    )
)

matrix["action_correct_check"] = (
    matrix["action_prediction"]
    ==
    matrix["ground_truth"]
)

fixed_rows = []

for subset_name in [
    "overall",
    "simple",
    "complex",
]:

    if subset_name == "overall":
        d0 = matrix
    else:
        d0 = matrix[
            matrix["subset"]
            ==
            subset_name
        ]

    for alpha in ACTIONS:

        g = (
            d0[
                np.isclose(
                    d0["alpha"],
                    alpha,
                )
            ]
            .copy()
        )

        for direction in [
            "undercount",
            "overcount",
        ]:

            w = (
                g[
                    g["error_direction"]
                    ==
                    direction
                ]
                .copy()
            )

            repaired = int(
                w[
                    "action_correct_check"
                ]
                .sum()
            )

            fixed_rows.append(
                {
                    "subset":
                        subset_name,

                    "alpha":
                        alpha,

                    "error_direction":
                        direction,

                    "baseline_wrong_n":
                        len(w),

                    "repaired":
                        repaired,

                    "repair_rate":
                        (
                            repaired
                            /
                            len(w)
                            if len(w)
                            else 0.0
                        ),

                    "mean_expected_numeral_shift":
                        (
                            float(
                                w[
                                    "expected_numeral_shift"
                                ].mean()
                            )
                            if len(w)
                            else 0.0
                        ),
                }
            )

fixed_direction = pd.DataFrame(
    fixed_rows
)

fixed_direction.to_csv(
    OUT_DIR
    / "fixed_action_repairs_by_error_direction.csv",
    index=False,
)


# ============================================================
# 4. Oracle repairability by error direction and magnitude
# ============================================================

successful_nonnoop = (
    matrix[
        (
            ~np.isclose(
                matrix["alpha"],
                1.0,
            )
        )
        &
        matrix[
            "action_correct_check"
        ]
    ]
    .copy()
)

repairable_ids = set(
    successful_nonnoop[
        "question_id"
    ]
    .astype(int)
    .tolist()
)

baseline["oracle_repairable"] = (
    baseline["question_id"]
    .astype(int)
    .isin(
        repairable_ids
    )
    &
    (
        baseline["signed_error"]
        !=
        0
    )
)

oracle_rows = []

for subset_name, data in [
    ("overall", baseline),
    (
        "simple",
        baseline[
            baseline["subset"]
            ==
            "simple"
        ],
    ),
    (
        "complex",
        baseline[
            baseline["subset"]
            ==
            "complex"
        ],
    ),
]:

    wrong = data[
        data["signed_error"]
        !=
        0
    ]

    for direction in [
        "undercount",
        "overcount",
    ]:

        d = wrong[
            wrong["error_direction"]
            ==
            direction
        ]

        repairable = int(
            d[
                "oracle_repairable"
            ]
            .sum()
        )

        oracle_rows.append(
            {
                "subset":
                    subset_name,

                "error_direction":
                    direction,

                "baseline_wrong_n":
                    len(d),

                "oracle_repairable":
                    repairable,

                "repairable_fraction":
                    (
                        repairable
                        /
                        len(d)
                        if len(d)
                        else 0.0
                    ),
            }
        )

oracle_direction = pd.DataFrame(
    oracle_rows
)

oracle_direction.to_csv(
    OUT_DIR
    / "oracle_repairability_by_error_direction.csv",
    index=False,
)


# Oracle by exact signed error.
oracle_error_rows = []

wrong_all = baseline[
    baseline["signed_error"]
    !=
    0
]

for error, g in (
    wrong_all.groupby(
        "signed_error"
    )
):

    repairable = int(
        g[
            "oracle_repairable"
        ]
        .sum()
    )

    oracle_error_rows.append(
        {
            "signed_error":
                int(error),

            "n":
                len(g),

            "oracle_repairable":
                repairable,

            "repairable_fraction":
                (
                    repairable
                    /
                    len(g)
                ),
        }
    )

oracle_by_error = pd.DataFrame(
    oracle_error_rows
).sort_values(
    "signed_error"
)

oracle_by_error.to_csv(
    OUT_DIR
    / "oracle_repairability_by_signed_error.csv",
    index=False,
)


# ============================================================
# 5. Frozen controller action by baseline error direction
# ============================================================

controller = (
    archived[
        [
            "question_id",
            "selected_alpha",
            "post_prediction",
        ]
    ]
    .merge(
        baseline[
            [
                "question_id",
                "ground_truth",
                "baseline_prediction",
                "signed_error",
                "error_direction",
                "oracle_repairable",
                "subset",
            ]
        ],
        on="question_id",
        how="inner",
    )
)

assert len(controller) == 4000

controller["post_correct_check"] = (
    controller["post_prediction"]
    ==
    controller["ground_truth"]
)

controller_rows = []

for status in [
    "correct",
    "undercount",
    "overcount",
]:

    g = controller[
        controller["error_direction"]
        ==
        status
    ]

    for alpha in ACTIONS:

        a = g[
            np.isclose(
                g["selected_alpha"],
                alpha,
            )
        ]

        controller_rows.append(
            {
                "baseline_status":
                    status,

                "alpha":
                    alpha,

                "count":
                    len(a),

                "fraction_within_status":
                    (
                        len(a) / len(g)
                        if len(g)
                        else 0.0
                    ),

                "post_correct":
                    int(
                        a[
                            "post_correct_check"
                        ]
                        .sum()
                    ),

                "post_accuracy":
                    (
                        float(
                            a[
                                "post_correct_check"
                            ].mean()
                        )
                        if len(a)
                        else 0.0
                    ),
            }
        )

controller_direction = pd.DataFrame(
    controller_rows
)

controller_direction.to_csv(
    OUT_DIR
    / "controller_action_by_error_direction.csv",
    index=False,
)


# ============================================================
# 6. Controller capture among oracle-repairable errors
# ============================================================

repairable_wrong = controller[
    (
        controller[
            "error_direction"
        ]
        !=
        "correct"
    )
    &
    controller[
        "oracle_repairable"
    ]
].copy()

capture_rows = []

for direction in [
    "undercount",
    "overcount",
]:

    g = repairable_wrong[
        repairable_wrong[
            "error_direction"
        ]
        ==
        direction
    ]

    captured = int(
        g[
            "post_correct_check"
        ]
        .sum()
    )

    capture_rows.append(
        {
            "error_direction":
                direction,

            "oracle_repairable":
                len(g),

            "controller_captured":
                captured,

            "capture_fraction":
                (
                    captured
                    /
                    len(g)
                    if len(g)
                    else 0.0
                ),
        }
    )

capture_df = pd.DataFrame(
    capture_rows
)

capture_df.to_csv(
    OUT_DIR
    / "controller_capture_by_error_direction.csv",
    index=False,
)


# ============================================================
# Print principal results
# ============================================================

print(
    "=" * 112
)

print(
    "NATURAL-OOD BASELINE ERROR DIRECTION"
)

print(
    "=" * 112
)

print(
    direction_summary.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "OVERALL BASELINE SIGNED-ERROR SPECTRUM"
)

print(
    "=" * 112
)

print(
    error_spectrum[
        error_spectrum["subset"]
        ==
        "overall"
    ].to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "FIXED ACTION REPAIRS BY ERROR DIRECTION — OVERALL"
)

print(
    "=" * 112
)

print(
    fixed_direction[
        fixed_direction["subset"]
        ==
        "overall"
    ].to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "ORACLE REPAIRABILITY BY ERROR DIRECTION"
)

print(
    "=" * 112
)

print(
    oracle_direction.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "ORACLE REPAIRABILITY BY EXACT SIGNED ERROR"
)

print(
    "=" * 112
)

print(
    oracle_by_error.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "FROZEN CONTROLLER ACTIONS BY BASELINE STATUS"
)

print(
    "=" * 112
)

print(
    controller_direction.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "CONTROLLER CAPTURE OF ORACLE-REPAIRABLE ERRORS"
)

print(
    "=" * 112
)

print(
    capture_df.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "POST-HOC ERROR-DIRECTION FORENSICS COMPLETE"
)

print(
    "=" * 112
)

print(
    "No model inference performed."
)

print(
    "No controller fitting performed."
)

print(
    "No frozen artifact modified."
)

print(
    "\nSaved to:",
    OUT_DIR,
)

