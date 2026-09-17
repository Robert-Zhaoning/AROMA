#!/usr/bin/env python3

"""
Post-hoc HoloCount router-vs-actuator forensics.

No VLM inference is performed.

Inputs:
- frozen full HoloCount controller result
- post-hoc action-oracle gate result

Questions:
1. How much oracle repair capacity did the frozen router capture?
2. Were opportunities missed by NOOP gating or wrong-action routing?
3. Do frozen per-action utility scores still rank true repairs?
4. How does repairability depend on baseline error direction?
5. How heterogeneous is oracle headroom across HoloCount subsets?
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FULL_PATH = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/full_raw_results.csv"
)

ORACLE_PATH = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/oracle_gate_v1/"
    "wrong_gtle15_action_results.csv"
)

OUT_DIR = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/forensics_v1"
)

EXPECTED_FULL_SHA = (
    "82929de79583d0c0641a80e13c8978d4ee4c00cebe7deb8f132697d3b6295466"
)

EXPECTED_ORACLE_SHA = (
    "1c5f367c43a0af43e88432cc7c577c23e349f8d4ce16d754bda32cf7e44aa6ce"
)

ACTIONS = [
    0.0,
    1.5,
    2.0,
    4.0,
]

ACTION_SUFFIX = {
    0.0: "0",
    1.5: "1p5",
    2.0: "2",
    4.0: "4",
}


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .astype(bool)
    )


# ============================================================
# Integrity
# ============================================================

if sha256_file(FULL_PATH) != EXPECTED_FULL_SHA:
    raise RuntimeError(
        "Full-result SHA mismatch."
    )

if sha256_file(ORACLE_PATH) != EXPECTED_ORACLE_SHA:
    raise RuntimeError(
        "Oracle-result SHA mismatch."
    )

full = pd.read_csv(FULL_PATH)
oracle = pd.read_csv(ORACLE_PATH)

if len(full) != 2480:
    raise RuntimeError(
        f"Expected 2480 full rows, got {len(full)}."
    )

if len(oracle) != 1134:
    raise RuntimeError(
        f"Expected 1134 oracle rows, got {len(oracle)}."
    )


# ============================================================
# Reconstruct primary bookkeeping from raw predictions.
# ============================================================

full["ground_truth"] = (
    full["ground_truth"].astype(int)
)

full["baseline_prediction"] = (
    full["baseline_prediction"].astype(int)
)

full["post_prediction"] = (
    full["post_prediction"].astype(int)
)

full["selected_alpha"] = (
    full["selected_alpha"].astype(float)
)

full["baseline_correct_calc"] = (
    full["baseline_prediction"]
    ==
    full["ground_truth"]
)

full["post_correct_calc"] = (
    full["post_prediction"]
    ==
    full["ground_truth"]
)

baseline_correct_n = int(
    full["baseline_correct_calc"].sum()
)

post_correct_n = int(
    full["post_correct_calc"].sum()
)

assert baseline_correct_n == 1158
assert post_correct_n == 1157


# ============================================================
# Join oracle population.
# ============================================================

oracle_cols = [
    "sample_id",
    "oracle_repairable",
    "repair_alpha_0",
    "repair_alpha_1p5",
    "repair_alpha_2",
    "repair_alpha_4",
]

for c in oracle_cols[1:]:
    oracle[c] = as_bool(
        oracle[c]
    )

target = full.merge(
    oracle[oracle_cols],
    on="sample_id",
    how="inner",
    validate="one_to_one",
)

if len(target) != 1134:
    raise RuntimeError(
        f"Joined oracle target has {len(target)} rows."
    )

# These must all be supported baseline-wrong examples.
assert (
    target["ground_truth"]
    <= 15
).all()

assert (
    ~target["baseline_correct_calc"]
).all()


# ============================================================
# Determine whether frozen selected action was actually a repair.
# ============================================================

def selected_action_repairs(row):

    alpha = float(
        row["selected_alpha"]
    )

    if np.isclose(
        alpha,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        return False

    suffix = ACTION_SUFFIX[
        alpha
    ]

    return bool(
        row[
            f"repair_alpha_{suffix}"
        ]
    )


target[
    "selected_action_repairs"
] = target.apply(
    selected_action_repairs,
    axis=1,
)

# Cross-check against actual post prediction.
if not np.array_equal(
    target[
        "selected_action_repairs"
    ].to_numpy(dtype=bool),
    target[
        "post_correct_calc"
    ].to_numpy(dtype=bool),
):
    raise RuntimeError(
        "Selected-action oracle labels disagree "
        "with frozen controller post predictions."
    )


# ============================================================
# 1. Opportunity capture decomposition.
# ============================================================

repairable = (
    target[
        "oracle_repairable"
    ].astype(bool)
)

controller_repaired = (
    target[
        "selected_action_repairs"
    ].astype(bool)
)

noop = np.isclose(
    target[
        "selected_alpha"
    ].to_numpy(dtype=float),
    1.0,
    rtol=0.0,
    atol=1e-12,
)

repairable_n = int(
    repairable.sum()
)

captured_n = int(
    (
        repairable
        &
        controller_repaired
    ).sum()
)

missed_n = int(
    repairable_n
    -
    captured_n
)

missed_noop_n = int(
    (
        repairable.to_numpy()
        &
        noop
    ).sum()
)

missed_wrong_action_n = int(
    (
        repairable.to_numpy()
        &
        (~noop)
        &
        (~controller_repaired.to_numpy())
    ).sum()
)

supported_interventions = int(
    (~noop).sum()
)

supported_intervention_repairs = int(
    controller_repaired.sum()
)

supported_intervention_precision = (
    supported_intervention_repairs
    /
    supported_interventions
    if supported_interventions
    else float("nan")
)

capture_summary = {
    "oracle_repairable_supported_wrong":
        repairable_n,

    "controller_captured_repairs":
        captured_n,

    "controller_opportunity_capture_rate":
        captured_n / repairable_n,

    "missed_repairable_total":
        missed_n,

    "missed_due_to_noop":
        missed_noop_n,

    "missed_due_to_wrong_nonnoop_action":
        missed_wrong_action_n,

    "supported_wrong_interventions":
        supported_interventions,

    "supported_wrong_intervention_repairs":
        supported_intervention_repairs,

    "supported_wrong_intervention_precision":
        supported_intervention_precision,

    "full_controller_repairs":
        int(
            (
                (~full["baseline_correct_calc"])
                &
                full["post_correct_calc"]
            ).sum()
        ),

    "full_controller_breaks":
        int(
            (
                full["baseline_correct_calc"]
                &
                (~full["post_correct_calc"])
            ).sum()
        ),
}


# ============================================================
# 2. Action-specific score vs true repair utility.
# ============================================================

action_rows = []

for alpha in ACTIONS:

    suffix = ACTION_SUFFIX[
        alpha
    ]

    score_col = (
        "score_alpha_"
        +
        suffix
    )

    repair_col = (
        "repair_alpha_"
        +
        suffix
    )

    scores = (
        target[
            score_col
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    repairs = (
        target[
            repair_col
        ]
        .astype(bool)
        .to_numpy()
    )

    rho, p = spearmanr(
        scores,
        repairs.astype(int),
    )

    selected_mask = np.isclose(
        target[
            "selected_alpha"
        ].to_numpy(
            dtype=float
        ),
        alpha,
        rtol=0.0,
        atol=1e-12,
    )

    selected_count = int(
        selected_mask.sum()
    )

    selected_repairs = int(
        (
            selected_mask
            &
            repairs
        ).sum()
    )

    repair_scores = (
        scores[
            repairs
        ]
    )

    nonrepair_scores = (
        scores[
            ~repairs
        ]
    )

    action_rows.append({
        "alpha":
            alpha,

        "repair_count":
            int(
                repairs.sum()
            ),

        "repair_rate_on_supported_wrong":
            float(
                repairs.mean()
            ),

        "router_selected_count_on_supported_wrong":
            selected_count,

        "router_selected_repairs":
            selected_repairs,

        "router_selection_precision":
            (
                selected_repairs
                /
                selected_count
                if selected_count
                else np.nan
            ),

        "mean_score_all":
            float(
                scores.mean()
            ),

        "median_score_all":
            float(
                np.median(
                    scores
                )
            ),

        "mean_score_when_repair":
            (
                float(
                    repair_scores.mean()
                )
                if len(repair_scores)
                else np.nan
            ),

        "mean_score_when_no_repair":
            (
                float(
                    nonrepair_scores.mean()
                )
                if len(nonrepair_scores)
                else np.nan
            ),

        "score_repair_spearman_rho":
            float(
                rho
            ),

        "score_repair_spearman_p":
            float(
                p
            ),
    })

action_df = pd.DataFrame(
    action_rows
)


# ============================================================
# 3. Baseline error direction.
# ============================================================

target[
    "error_direction"
] = np.where(
    target[
        "baseline_prediction"
    ]
    <
    target[
        "ground_truth"
    ],
    "UNDERCOUNT",
    "OVERCOUNT",
)

direction_rows = []

for direction, g in target.groupby(
    "error_direction",
    sort=True,
):

    oracle_rep = (
        g[
            "oracle_repairable"
        ].astype(bool)
    )

    controller_rep = (
        g[
            "selected_action_repairs"
        ].astype(bool)
    )

    interventions = ~np.isclose(
        g[
            "selected_alpha"
        ].to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    direction_rows.append({
        "error_direction":
            direction,

        "n":
            len(g),

        "fraction_supported_wrong":
            len(g)
            /
            len(target),

        "oracle_repairable":
            int(
                oracle_rep.sum()
            ),

        "oracle_repairable_rate":
            float(
                oracle_rep.mean()
            ),

        "controller_repairs":
            int(
                controller_rep.sum()
            ),

        "controller_capture_fraction_of_oracle":
            (
                int(
                    controller_rep.sum()
                )
                /
                int(
                    oracle_rep.sum()
                )
                if int(
                    oracle_rep.sum()
                )
                else np.nan
            ),

        "controller_interventions":
            int(
                interventions.sum()
            ),

        "controller_intervention_rate":
            float(
                interventions.mean()
            ),
    })

direction_df = pd.DataFrame(
    direction_rows
)


# ============================================================
# 4. Subset oracle headroom and router capture.
# ============================================================

repairable_by_sample = (
    oracle[
        [
            "sample_id",
            "oracle_repairable",
        ]
    ]
    .set_index(
        "sample_id"
    )[
        "oracle_repairable"
    ]
    .to_dict()
)

subset_rows = []

for subset, g in full.groupby(
    "split",
    sort=True,
):

    n = len(g)

    baseline_correct = int(
        g[
            "baseline_correct_calc"
        ].sum()
    )

    post_correct = int(
        g[
            "post_correct_calc"
        ].sum()
    )

    supported_wrong = g[
        (
            ~g[
                "baseline_correct_calc"
            ]
        )
        &
        (
            g[
                "ground_truth"
            ]
            <= 15
        )
    ]

    oracle_repairable_n = int(
        sum(
            bool(
                repairable_by_sample[
                    sid
                ]
            )
            for sid
            in supported_wrong[
                "sample_id"
            ]
            if sid
            in repairable_by_sample
        )
    )

    controller_repairs = int(
        (
            (~g["baseline_correct_calc"])
            &
            g["post_correct_calc"]
        ).sum()
    )

    oracle_correct = (
        baseline_correct
        +
        oracle_repairable_n
    )

    subset_rows.append({
        "subset":
            subset,

        "n":
            n,

        "baseline_correct":
            baseline_correct,

        "baseline_accuracy":
            baseline_correct / n,

        "frozen_post_correct":
            post_correct,

        "frozen_post_accuracy":
            post_correct / n,

        "frozen_gain_pp":
            100.0
            *
            (
                post_correct
                -
                baseline_correct
            )
            /
            n,

        "supported_baseline_wrong":
            len(
                supported_wrong
            ),

        "gt_gt_15":
            int(
                (
                    g[
                        "ground_truth"
                    ]
                    > 15
                ).sum()
            ),

        "oracle_repairable":
            oracle_repairable_n,

        "oracle_accuracy":
            oracle_correct / n,

        "oracle_gain_pp":
            100.0
            *
            oracle_repairable_n
            /
            n,

        "controller_repairs":
            controller_repairs,

        "controller_capture_fraction_of_oracle":
            (
                controller_repairs
                /
                oracle_repairable_n
                if oracle_repairable_n
                else np.nan
            ),
    })

subset_df = (
    pd.DataFrame(
        subset_rows
    )
    .sort_values(
        "oracle_gain_pp",
        ascending=False,
    )
)


# ============================================================
# 5. Action repair overlap.
# ============================================================

repair_matrix = np.zeros(
    (
        len(ACTIONS),
        len(ACTIONS),
    ),
    dtype=int,
)

repair_bool = {}

for alpha in ACTIONS:
    suffix = ACTION_SUFFIX[
        alpha
    ]

    repair_bool[
        alpha
    ] = (
        target[
            f"repair_alpha_{suffix}"
        ]
        .astype(bool)
        .to_numpy()
    )


for i, a in enumerate(
    ACTIONS
):
    for j, b in enumerate(
        ACTIONS
    ):
        repair_matrix[
            i,
            j
        ] = int(
            (
                repair_bool[a]
                &
                repair_bool[b]
            ).sum()
        )


overlap_df = pd.DataFrame(
    repair_matrix,
    index=[
        f"alpha_{a}"
        for a in ACTIONS
    ],
    columns=[
        f"alpha_{a}"
        for a in ACTIONS
    ],
)

repair_action_count = np.zeros(
    len(target),
    dtype=int,
)

for alpha in ACTIONS:
    repair_action_count += (
        repair_bool[
            alpha
        ].astype(int)
    )

repair_count_dist = (
    pd.Series(
        repair_action_count,
        name="num_repair_actions",
    )
    .value_counts()
    .sort_index()
    .rename_axis(
        "num_repair_actions"
    )
    .reset_index(
        name="sample_count"
    )
)

repair_count_dist[
    "fraction_supported_wrong"
] = (
    repair_count_dist[
        "sample_count"
    ]
    /
    len(target)
)


# ============================================================
# Write artifacts.
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

action_df.to_csv(
    OUT_DIR
    /
    "action_score_utility.csv",
    index=False,
)

direction_df.to_csv(
    OUT_DIR
    /
    "by_error_direction.csv",
    index=False,
)

subset_df.to_csv(
    OUT_DIR
    /
    "by_subset_oracle.csv",
    index=False,
)

overlap_df.to_csv(
    OUT_DIR
    /
    "repair_overlap_matrix.csv",
)

repair_count_dist.to_csv(
    OUT_DIR
    /
    "repair_action_count_distribution.csv",
    index=False,
)

summary = {
    "status":
        "POST_HOC_FORENSICS",

    "full_result_sha256":
        EXPECTED_FULL_SHA,

    "oracle_result_sha256":
        EXPECTED_ORACLE_SHA,

    "baseline_accuracy":
        baseline_correct_n
        /
        2480,

    "frozen_post_accuracy":
        post_correct_n
        /
        2480,

    "oracle_accuracy":
        (
            baseline_correct_n
            +
            repairable_n
        )
        /
        2480,

    "oracle_gain_pp":
        100.0
        *
        repairable_n
        /
        2480,

    "capture":
        capture_summary,
}

(
    OUT_DIR
    /
    "forensics_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


# ============================================================
# Console summary.
# ============================================================

print()
print("=" * 72)
print(
    "HOLOCOUNT ROUTER-vs-ACTUATOR FORENSICS"
)
print("=" * 72)

print()
print("===== OPPORTUNITY CAPTURE =====")

for k, v in capture_summary.items():
    print(
        f"{k:42s}: {v}"
    )

print()
print("===== ACTION SCORE vs TRUE REPAIR =====")

print(
    action_df[
        [
            "alpha",
            "repair_count",
            "router_selected_count_on_supported_wrong",
            "router_selected_repairs",
            "router_selection_precision",
            "mean_score_when_repair",
            "mean_score_when_no_repair",
            "score_repair_spearman_rho",
        ]
    ]
    .to_string(
        index=False
    )
)

print()
print("===== ERROR DIRECTION =====")

print(
    direction_df
    .to_string(
        index=False
    )
)

print()
print("===== TOP 10 SUBSETS BY ORACLE HEADROOM =====")

print(
    subset_df[
        [
            "subset",
            "n",
            "baseline_accuracy",
            "frozen_gain_pp",
            "oracle_repairable",
            "oracle_gain_pp",
            "controller_capture_fraction_of_oracle",
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)

print()
print("===== REPAIR ACTION COUNT =====")

print(
    repair_count_dist
    .to_string(
        index=False
    )
)

print()
print(
    "Artifacts written to:",
    OUT_DIR,
)
