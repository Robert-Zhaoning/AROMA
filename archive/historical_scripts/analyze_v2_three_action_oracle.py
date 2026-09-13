from pathlib import Path

import numpy as np
import pandas as pd


DOWN_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/"
    "down_alpha0p5_expanded/"
    "down_alpha0p5_results.csv"
)

UP_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5_expanded_0_15/"
    "confirmation_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/"
    "three_action_oracle"
)

PER_SAMPLE_PATH = (
    OUT_DIR
    / "three_action_oracle_per_sample.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "three_action_oracle_summary.csv"
)

CONDITION_PATH = (
    OUT_DIR
    / "three_action_oracle_by_condition.csv"
)


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print(
        "AROMA V2 THREE-ACTION SIGNED-STEERING "
        "ORACLE CEILING"
    )
    print("=" * 112)

    # ========================================================
    # Load downward alpha=.5
    # ========================================================

    down = pd.read_csv(
        DOWN_PATH
    )

    if len(down) != 1000:
        raise RuntimeError(
            f"Expected 1000 downward rows, "
            f"found {len(down)}."
        )

    if (
        down["sample_id"].nunique()
        != 1000
    ):
        raise RuntimeError(
            "Downward sample IDs are not unique."
        )

    # ========================================================
    # Load alpha=1 and alpha=1.5 from expanded confirmation
    # ========================================================

    up_all = pd.read_csv(
        UP_PATH
    )

    noop = up_all[
        np.isclose(
            up_all["alpha"],
            1.0,
        )
    ].copy()

    up = up_all[
        np.isclose(
            up_all["alpha"],
            1.5,
        )
    ].copy()

    if len(noop) != 1000:
        raise RuntimeError(
            f"Expected 1000 alpha=1 rows, "
            f"found {len(noop)}."
        )

    if len(up) != 1000:
        raise RuntimeError(
            f"Expected 1000 alpha=1.5 rows, "
            f"found {len(up)}."
        )

    # ========================================================
    # Keep only necessary columns and merge
    # ========================================================

    down_small = down[
        [
            "sample_id",
            "ground_truth",
            "condition",
            "baseline_best_numeral",
            "modulated_best_numeral",
            "baseline_correct",
            "modulated_correct",
            "baseline_expected_numeral",
            "modulated_expected_numeral",
        ]
    ].copy()

    down_small = down_small.rename(
        columns={
            "baseline_best_numeral":
                "baseline_pred_downfile",

            "modulated_best_numeral":
                "down_pred",

            "baseline_correct":
                "baseline_correct_downfile",

            "modulated_correct":
                "down_correct",

            "baseline_expected_numeral":
                "baseline_expected_downfile",

            "modulated_expected_numeral":
                "down_expected",
        }
    )

    noop_small = noop[
        [
            "sample_id",
            "ground_truth",
            "condition",
            "baseline_best_numeral",
            "modulated_best_numeral",
            "baseline_correct",
            "modulated_correct",
            "baseline_expected_numeral",
            "modulated_expected_numeral",
        ]
    ].copy()

    noop_small = noop_small.rename(
        columns={
            "baseline_best_numeral":
                "baseline_pred_noopfile",

            "modulated_best_numeral":
                "noop_pred",

            "baseline_correct":
                "baseline_correct_noopfile",

            "modulated_correct":
                "noop_correct",

            "baseline_expected_numeral":
                "baseline_expected_noopfile",

            "modulated_expected_numeral":
                "noop_expected",
        }
    )

    up_small = up[
        [
            "sample_id",
            "ground_truth",
            "condition",
            "modulated_best_numeral",
            "modulated_correct",
            "modulated_expected_numeral",
        ]
    ].copy()

    up_small = up_small.rename(
        columns={
            "modulated_best_numeral":
                "up_pred",

            "modulated_correct":
                "up_correct",

            "modulated_expected_numeral":
                "up_expected",
        }
    )

    merged = down_small.merge(
        noop_small,
        on="sample_id",
        suffixes=("_down", "_noop"),
        validate="one_to_one",
    )

    merged = merged.merge(
        up_small,
        on="sample_id",
        validate="one_to_one",
    )

    if len(merged) != 1000:
        raise RuntimeError(
            f"Merged row count is "
            f"{len(merged)}, expected 1000."
        )

    # ========================================================
    # Integrity audits
    # ========================================================

    gt_cols = [
        "ground_truth_down",
        "ground_truth_noop",
        "ground_truth",
    ]

    condition_cols = [
        "condition_down",
        "condition_noop",
        "condition",
    ]

    if not (
        merged[
            gt_cols
        ].nunique(
            axis=1
        )
        == 1
    ).all():
        raise RuntimeError(
            "Ground-truth mismatch across files."
        )

    if not (
        merged[
            condition_cols
        ].nunique(
            axis=1
        )
        == 1
    ).all():
        raise RuntimeError(
            "Condition mismatch across files."
        )

    baseline_pred_mismatch = (
        merged[
            "baseline_pred_downfile"
        ].astype(int)
        !=
        merged[
            "baseline_pred_noopfile"
        ].astype(int)
    )

    if baseline_pred_mismatch.any():
        raise RuntimeError(
            "Baseline predictions disagree "
            "between downward and noop files."
        )

    identity_mismatch = (
        merged[
            "baseline_pred_noopfile"
        ].astype(int)
        !=
        merged[
            "noop_pred"
        ].astype(int)
    )

    if identity_mismatch.any():
        raise RuntimeError(
            "Alpha=1 identity prediction audit failed."
        )

    max_expected_identity_error = float(
        (
            merged[
                "baseline_expected_noopfile"
            ]
            -
            merged[
                "noop_expected"
            ]
        )
        .abs()
        .max()
    )

    if max_expected_identity_error > 1e-10:
        raise RuntimeError(
            "Alpha=1 expected-numeral identity "
            f"failed: {max_expected_identity_error}"
        )

    print(
        "Rows                :",
        len(merged),
    )

    print(
        "Sample integrity    : PASS"
    )

    print(
        "Baseline invariance : PASS"
    )

    print(
        "Alpha=1 identity    : PASS"
    )

    # ========================================================
    # Canonical fields
    # ========================================================

    merged[
        "ground_truth"
    ] = merged[
        "ground_truth"
    ].astype(int)

    merged[
        "condition"
    ] = merged[
        "condition"
    ].astype(str)

    merged[
        "baseline_pred"
    ] = merged[
        "noop_pred"
    ].astype(int)

    merged[
        "down_pred"
    ] = merged[
        "down_pred"
    ].astype(int)

    merged[
        "up_pred"
    ] = merged[
        "up_pred"
    ].astype(int)

    merged[
        "baseline_correct"
    ] = (
        merged[
            "baseline_pred"
        ]
        ==
        merged[
            "ground_truth"
        ]
    )

    merged[
        "down_correct"
    ] = (
        merged[
            "down_pred"
        ]
        ==
        merged[
            "ground_truth"
        ]
    )

    merged[
        "up_correct"
    ] = (
        merged[
            "up_pred"
        ]
        ==
        merged[
            "ground_truth"
        ]
    )

    merged[
        "baseline_error"
    ] = (
        merged[
            "baseline_pred"
        ]
        -
        merged[
            "ground_truth"
        ]
    )

    merged[
        "down_shift"
    ] = (
        merged[
            "down_pred"
        ]
        -
        merged[
            "baseline_pred"
        ]
    )

    merged[
        "up_shift"
    ] = (
        merged[
            "up_pred"
        ]
        -
        merged[
            "baseline_pred"
        ]
    )

    # ========================================================
    # Oracle correctness
    #
    # Tie-breaking policy:
    #   1. Keep NOOP whenever baseline is already correct.
    #   2. Otherwise use DOWN if it repairs.
    #   3. Otherwise use UP if it repairs.
    #   4. Otherwise NOOP.
    #
    # This makes the oracle conservative on correct samples.
    # ========================================================

    oracle_actions = []
    oracle_preds = []

    successful_action_counts = []

    for _, row in merged.iterrows():

        successful = []

        if bool(
            row[
                "baseline_correct"
            ]
        ):
            successful.append(
                "noop"
            )

        if bool(
            row[
                "down_correct"
            ]
        ):
            successful.append(
                "down"
            )

        if bool(
            row[
                "up_correct"
            ]
        ):
            successful.append(
                "up"
            )

        successful_action_counts.append(
            len(
                successful
            )
        )

        if bool(
            row[
                "baseline_correct"
            ]
        ):

            action = "noop"

            pred = int(
                row[
                    "baseline_pred"
                ]
            )

        elif bool(
            row[
                "down_correct"
            ]
        ):

            action = "down"

            pred = int(
                row[
                    "down_pred"
                ]
            )

        elif bool(
            row[
                "up_correct"
            ]
        ):

            action = "up"

            pred = int(
                row[
                    "up_pred"
                ]
            )

        else:

            action = "noop"

            pred = int(
                row[
                    "baseline_pred"
                ]
            )

        oracle_actions.append(
            action
        )

        oracle_preds.append(
            pred
        )

    merged[
        "successful_action_count"
    ] = successful_action_counts

    merged[
        "oracle_action"
    ] = oracle_actions

    merged[
        "oracle_pred"
    ] = oracle_preds

    merged[
        "oracle_correct"
    ] = (
        merged[
            "oracle_pred"
        ]
        ==
        merged[
            "ground_truth"
        ]
    )

    # ========================================================
    # Repair opportunity overlap
    # ========================================================

    baseline_wrong = (
        ~merged[
            "baseline_correct"
        ]
    )

    down_repairs = (
        baseline_wrong
        &
        merged[
            "down_correct"
        ]
    )

    up_repairs = (
        baseline_wrong
        &
        merged[
            "up_correct"
        ]
    )

    both_repairs = (
        down_repairs
        &
        up_repairs
    )

    down_only = (
        down_repairs
        &
        ~up_repairs
    )

    up_only = (
        up_repairs
        &
        ~down_repairs
    )

    any_repair = (
        down_repairs
        |
        up_repairs
    )

    # ========================================================
    # Summary
    # ========================================================

    n = len(
        merged
    )

    baseline_correct_n = int(
        merged[
            "baseline_correct"
        ].sum()
    )

    baseline_wrong_n = (
        n
        -
        baseline_correct_n
    )

    oracle_correct_n = int(
        merged[
            "oracle_correct"
        ].sum()
    )

    oracle_repairs = (
        oracle_correct_n
        -
        baseline_correct_n
    )

    summary = pd.DataFrame([
        {
            "n":
                n,

            "baseline_correct":
                baseline_correct_n,

            "baseline_wrong":
                baseline_wrong_n,

            "baseline_accuracy":
                baseline_correct_n
                / n,

            "down_repairs":
                int(
                    down_repairs.sum()
                ),

            "up_repairs":
                int(
                    up_repairs.sum()
                ),

            "down_only_repairs":
                int(
                    down_only.sum()
                ),

            "up_only_repairs":
                int(
                    up_only.sum()
                ),

            "both_down_up_repair":
                int(
                    both_repairs.sum()
                ),

            "unique_repairable_wrong":
                int(
                    any_repair.sum()
                ),

            "repairable_fraction_of_wrong":
                float(
                    any_repair.sum()
                    /
                    baseline_wrong_n
                ),

            "oracle_correct":
                oracle_correct_n,

            "oracle_accuracy":
                oracle_correct_n
                / n,

            "oracle_accuracy_gain":
                (
                    oracle_correct_n
                    -
                    baseline_correct_n
                )
                / n,

            "oracle_repairs":
                oracle_repairs,

            "oracle_breaks":
                0,

            "oracle_net_repairs":
                oracle_repairs,
        }
    ])

    # ========================================================
    # By condition
    # ========================================================

    condition_rows = []

    for condition, g in merged.groupby(
        "condition"
    ):

        base_correct = int(
            g[
                "baseline_correct"
            ].sum()
        )

        oracle_correct = int(
            g[
                "oracle_correct"
            ].sum()
        )

        wrong = (
            ~g[
                "baseline_correct"
            ]
        )

        d_rep = (
            wrong
            &
            g[
                "down_correct"
            ]
        )

        u_rep = (
            wrong
            &
            g[
                "up_correct"
            ]
        )

        repairable = (
            d_rep
            |
            u_rep
        )

        condition_rows.append({
            "condition":
                condition,

            "n":
                len(g),

            "baseline_correct":
                base_correct,

            "baseline_accuracy":
                base_correct
                / len(g),

            "baseline_wrong":
                int(
                    wrong.sum()
                ),

            "down_repairs":
                int(
                    d_rep.sum()
                ),

            "up_repairs":
                int(
                    u_rep.sum()
                ),

            "unique_repairable_wrong":
                int(
                    repairable.sum()
                ),

            "repairable_fraction_wrong":
                (
                    float(
                        repairable.sum()
                        /
                        wrong.sum()
                    )
                    if wrong.sum()
                    else 0.0
                ),

            "oracle_correct":
                oracle_correct,

            "oracle_accuracy":
                oracle_correct
                / len(g),

            "oracle_gain":
                (
                    oracle_correct
                    -
                    base_correct
                )
                /
                len(g),
        })

    by_condition = (
        pd.DataFrame(
            condition_rows
        )
        .sort_values(
            "condition"
        )
    )

    # ========================================================
    # Save clean per-sample table
    # ========================================================

    save_cols = [
        "sample_id",
        "ground_truth",
        "condition",
        "baseline_pred",
        "baseline_error",
        "baseline_correct",
        "down_pred",
        "down_shift",
        "down_correct",
        "up_pred",
        "up_shift",
        "up_correct",
        "successful_action_count",
        "oracle_action",
        "oracle_pred",
        "oracle_correct",
    ]

    merged[
        save_cols
    ].to_csv(
        PER_SAMPLE_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    by_condition.to_csv(
        CONDITION_PATH,
        index=False,
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "\n" + "=" * 112
    )

    print(
        "THREE-ACTION ORACLE CEILING"
    )

    print(
        "=" * 112
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "REPAIR OPPORTUNITY OVERLAP"
    )

    print(
        "=" * 112
    )

    print(
        "DOWN repairs:",
        int(
            down_repairs.sum()
        ),
    )

    print(
        "UP repairs:",
        int(
            up_repairs.sum()
        ),
    )

    print(
        "DOWN only:",
        int(
            down_only.sum()
        ),
    )

    print(
        "UP only:",
        int(
            up_only.sum()
        ),
    )

    print(
        "BOTH:",
        int(
            both_repairs.sum()
        ),
    )

    print(
        "Unique repairable wrong:",
        int(
            any_repair.sum()
        ),
        "/",
        baseline_wrong_n,
        "=",
        float(
            any_repair.sum()
            /
            baseline_wrong_n
        ),
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "ORACLE ACTION DISTRIBUTION"
    )

    print(
        "=" * 112
    )

    print(
        merged[
            "oracle_action"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "REPAIR TYPE BY BASELINE ERROR"
    )

    print(
        "=" * 112
    )

    repaired = merged[
        baseline_wrong
        &
        merged[
            "oracle_correct"
        ]
    ]

    if len(
        repaired
    ):

        print(
            repaired.groupby(
                [
                    "baseline_error",
                    "oracle_action",
                ]
            )
            .size()
            .to_string()
        )

    else:

        print(
            "NO ORACLE REPAIRS"
        )

    print(
        "\n" + "=" * 112
    )

    print(
        "ORACLE BY CONDITION"
    )

    print(
        "=" * 112
    )

    print(
        by_condition.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nSaved:"
    )

    print(
        PER_SAMPLE_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        CONDITION_PATH
    )

    print(
        "\nTHREE-ACTION ORACLE ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
