from pathlib import Path

import numpy as np
import pandas as pd


NEW_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/"
    "multistrength_wrong_sweep/"
    "multistrength_wrong_results.csv"
)

DOWN_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/"
    "down_alpha0p5_expanded/"
    "down_alpha0p5_results.csv"
)

CONF_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5_expanded_0_15/"
    "confirmation_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/"
    "multistrength_oracle"
)

PER_SAMPLE_OUT = (
    OUT_DIR /
    "multistrength_oracle_per_sample.csv"
)

SUMMARY_OUT = (
    OUT_DIR /
    "multistrength_oracle_summary.csv"
)

BY_CONDITION_OUT = (
    OUT_DIR /
    "multistrength_oracle_by_condition.csv"
)

BY_ERROR_OUT = (
    OUT_DIR /
    "multistrength_oracle_by_error.csv"
)


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print(
        "AROMA V2 MULTI-STRENGTH "
        "L18H13 ORACLE CEILING"
    )
    print("=" * 112)

    # ========================================================
    # New sweep:
    # 0, .25, .75, 1.25, 2, 3, 4
    # ========================================================

    new = pd.read_csv(
        NEW_PATH
    )

    expected_alphas = {
        0.0,
        0.25,
        0.75,
        1.25,
        2.0,
        3.0,
        4.0,
    }

    found_alphas = set(
        np.round(
            new["alpha"].astype(float),
            8,
        )
    )

    if found_alphas != expected_alphas:
        raise RuntimeError(
            "Unexpected new-sweep alpha set: "
            f"{sorted(found_alphas)}"
        )

    if len(new) != 3479:
        raise RuntimeError(
            f"Expected 3479 new rows, "
            f"found {len(new)}."
        )

    # ========================================================
    # Existing alpha=.5
    # ========================================================

    down = pd.read_csv(
        DOWN_PATH
    )

    if len(down) != 1000:
        raise RuntimeError(
            "Expected 1000 alpha=.5 rows."
        )

    down = down[
        ~down[
            "baseline_correct"
        ].astype(bool)
    ].copy()

    if len(down) != 497:
        raise RuntimeError(
            f"Expected 497 baseline-wrong "
            f"alpha=.5 rows, found {len(down)}."
        )

    half = pd.DataFrame({
        "sample_id":
            down["sample_id"].astype(str),

        "ground_truth":
            down["ground_truth"].astype(int),

        "condition":
            down["condition"].astype(str),

        "baseline_pred":
            down[
                "baseline_best_numeral"
            ].astype(int),

        "baseline_error":
            (
                down[
                    "baseline_best_numeral"
                ].astype(int)
                -
                down[
                    "ground_truth"
                ].astype(int)
            ),

        "alpha":
            0.5,

        "modulated_pred":
            down[
                "modulated_best_numeral"
            ].astype(int),

        "correct":
            down[
                "modulated_correct"
            ].astype(bool),
    })

    # ========================================================
    # Existing alpha=1 and 1.5
    # ========================================================

    conf = pd.read_csv(
        CONF_PATH
    )

    existing_rows = []

    for alpha in [
        1.0,
        1.5,
    ]:

        g = conf[
            np.isclose(
                conf["alpha"],
                alpha,
            )
        ].copy()

        if len(g) != 1000:
            raise RuntimeError(
                f"Expected 1000 rows for "
                f"alpha={alpha}, found {len(g)}."
            )

        g = g[
            ~g[
                "baseline_correct"
            ].astype(bool)
        ].copy()

        if len(g) != 497:
            raise RuntimeError(
                f"Expected 497 wrong rows "
                f"for alpha={alpha}, "
                f"found {len(g)}."
            )

        tmp = pd.DataFrame({
            "sample_id":
                g["sample_id"].astype(str),

            "ground_truth":
                g[
                    "ground_truth"
                ].astype(int),

            "condition":
                g[
                    "condition"
                ].astype(str),

            "baseline_pred":
                g[
                    "baseline_best_numeral"
                ].astype(int),

            "baseline_error":
                (
                    g[
                        "baseline_best_numeral"
                    ].astype(int)
                    -
                    g[
                        "ground_truth"
                    ].astype(int)
                ),

            "alpha":
                float(alpha),

            "modulated_pred":
                g[
                    "modulated_best_numeral"
                ].astype(int),

            "correct":
                g[
                    "modulated_correct"
                ].astype(bool),
        })

        existing_rows.append(
            tmp
        )

    # ========================================================
    # Normalize new sweep columns
    # ========================================================

    new_small = new[
        [
            "sample_id",
            "ground_truth",
            "condition",
            "baseline_pred",
            "baseline_error",
            "alpha",
            "modulated_pred",
            "correct",
        ]
    ].copy()

    # ========================================================
    # Full 10-action table
    # ========================================================

    actions = pd.concat(
        [
            new_small,
            half,
            *existing_rows,
        ],
        ignore_index=True,
    )

    actions[
        "sample_id"
    ] = actions[
        "sample_id"
    ].astype(str)

    actions[
        "alpha"
    ] = actions[
        "alpha"
    ].astype(float)

    actions[
        "correct"
    ] = actions[
        "correct"
    ].astype(bool)

    actions = (
        actions
        .drop_duplicates(
            [
                "sample_id",
                "alpha",
            ],
            keep="last",
        )
        .sort_values(
            [
                "sample_id",
                "alpha",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    full_alphas = sorted(
        actions[
            "alpha"
        ]
        .unique()
        .tolist()
    )

    expected_full = [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
        4.0,
    ]

    if not np.allclose(
        full_alphas,
        expected_full,
    ):
        raise RuntimeError(
            "Full alpha set mismatch: "
            f"{full_alphas}"
        )

    if len(actions) != (
        497
        *
        len(expected_full)
    ):
        raise RuntimeError(
            f"Expected {497 * len(expected_full)} "
            f"combined rows, found {len(actions)}."
        )

    counts = (
        actions.groupby(
            "sample_id"
        )[
            "alpha"
        ]
        .nunique()
    )

    if not (
        counts
        ==
        len(expected_full)
    ).all():
        raise RuntimeError(
            "Incomplete action coverage."
        )

    print(
        "Wrong samples       :",
        actions[
            "sample_id"
        ].nunique(),
    )

    print(
        "Action count        :",
        len(expected_full),
    )

    print(
        "Action rows         :",
        len(actions),
    )

    print(
        "Alpha values        :",
        expected_full,
    )

    print(
        "Coverage audit      : PASS"
    )

    # ========================================================
    # Per-alpha repair counts
    # ========================================================

    repair_by_alpha = (
        actions.groupby(
            "alpha"
        )[
            "correct"
        ]
        .sum()
        .astype(int)
        .reset_index(
            name="repairs"
        )
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "REPAIRS BY ALPHA"
    )

    print(
        "=" * 112
    )

    print(
        repair_by_alpha.to_string(
            index=False
        )
    )

    # ========================================================
    # Per-sample oracle
    # ========================================================

    per_sample_rows = []

    for sid, g in actions.groupby(
        "sample_id"
    ):

        first = g.iloc[0]

        successful = (
            g[
                g[
                    "correct"
                ]
            ]
            .copy()
        )

        repairable = (
            len(successful)
            > 0
        )

        successful_alphas = (
            successful[
                "alpha"
            ]
            .astype(float)
            .tolist()
        )

        # Conservative oracle tie-break:
        # among successful actions,
        # choose alpha closest to identity (=1).
        # If tied, choose smaller absolute gain magnitude;
        # then lower alpha for determinism.
        if repairable:

            successful[
                "distance_from_identity"
            ] = (
                successful[
                    "alpha"
                ]
                -
                1.0
            ).abs()

            best = (
                successful
                .sort_values(
                    [
                        "distance_from_identity",
                        "alpha",
                    ]
                )
                .iloc[0]
            )

            oracle_alpha = float(
                best[
                    "alpha"
                ]
            )

            oracle_pred = int(
                best[
                    "modulated_pred"
                ]
            )

        else:

            oracle_alpha = 1.0

            oracle_pred = int(
                first[
                    "baseline_pred"
                ]
            )

        per_sample_rows.append({
            "sample_id":
                sid,

            "ground_truth":
                int(
                    first[
                        "ground_truth"
                    ]
                ),

            "condition":
                str(
                    first[
                        "condition"
                    ]
                ),

            "baseline_pred":
                int(
                    first[
                        "baseline_pred"
                    ]
                ),

            "baseline_error":
                int(
                    first[
                        "baseline_error"
                    ]
                ),

            "repairable":
                bool(
                    repairable
                ),

            "n_successful_alphas":
                len(
                    successful_alphas
                ),

            "successful_alphas":
                ",".join(
                    str(x)
                    for x
                    in successful_alphas
                ),

            "oracle_alpha":
                oracle_alpha,

            "oracle_pred":
                oracle_pred,

            "oracle_correct":
                bool(
                    oracle_pred
                    ==
                    int(
                        first[
                            "ground_truth"
                        ]
                    )
                ),
        })

    per_sample = pd.DataFrame(
        per_sample_rows
    )

    unique_repairs = int(
        per_sample[
            "repairable"
        ].sum()
    )

    BASELINE_CORRECT = 503
    TOTAL = 1000
    BASELINE_WRONG = 497

    oracle_correct = (
        BASELINE_CORRECT
        +
        unique_repairs
    )

    oracle_accuracy = (
        oracle_correct
        /
        TOTAL
    )

    oracle_gain = (
        unique_repairs
        /
        TOTAL
    )

    repair_fraction_wrong = (
        unique_repairs
        /
        BASELINE_WRONG
    )

    summary = pd.DataFrame([
        {
            "n":
                TOTAL,

            "baseline_correct":
                BASELINE_CORRECT,

            "baseline_wrong":
                BASELINE_WRONG,

            "baseline_accuracy":
                BASELINE_CORRECT
                /
                TOTAL,

            "unique_repaired_wrong":
                unique_repairs,

            "repair_fraction_wrong":
                repair_fraction_wrong,

            "oracle_correct":
                oracle_correct,

            "oracle_accuracy":
                oracle_accuracy,

            "oracle_accuracy_gain":
                oracle_gain,
        }
    ])

    # ========================================================
    # By baseline error
    # ========================================================

    error_rows = []

    for error, g in per_sample.groupby(
        "baseline_error"
    ):

        repaired = int(
            g[
                "repairable"
            ].sum()
        )

        error_rows.append({
            "baseline_error":
                int(error),

            "n":
                len(g),

            "repaired":
                repaired,

            "repair_rate":
                repaired
                /
                len(g),
        })

    by_error = (
        pd.DataFrame(
            error_rows
        )
        .sort_values(
            "baseline_error"
        )
    )

    # ========================================================
    # By condition
    # ========================================================

    condition_rows = []

    baseline_correct_by_condition = {
        "dense": 89,
        "distractors": 122,
        "grid": 93,
        "random_sparse": 115,
        "row": 84,
    }

    for condition, g in per_sample.groupby(
        "condition"
    ):

        repaired = int(
            g[
                "repairable"
            ].sum()
        )

        wrong_n = len(g)

        base_correct = int(
            baseline_correct_by_condition[
                condition
            ]
        )

        oracle_correct_condition = (
            base_correct
            +
            repaired
        )

        condition_rows.append({
            "condition":
                condition,

            "baseline_correct":
                base_correct,

            "baseline_wrong":
                wrong_n,

            "baseline_accuracy":
                base_correct
                /
                200,

            "repaired_wrong":
                repaired,

            "repair_rate_wrong":
                repaired
                /
                wrong_n,

            "oracle_correct":
                oracle_correct_condition,

            "oracle_accuracy":
                oracle_correct_condition
                /
                200,

            "oracle_gain":
                repaired
                /
                200,
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
    # Alpha distribution
    # ========================================================

    repaired = per_sample[
        per_sample[
            "repairable"
        ]
    ].copy()

    alpha_distribution = (
        repaired[
            "oracle_alpha"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # Save
    # ========================================================

    per_sample.to_csv(
        PER_SAMPLE_OUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    by_condition.to_csv(
        BY_CONDITION_OUT,
        index=False,
    )

    by_error.to_csv(
        BY_ERROR_OUT,
        index=False,
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "\n" + "=" * 112
    )

    print(
        "MULTI-STRENGTH ORACLE CEILING"
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
        "\nOracle selected-alpha distribution "
        "(closest-to-identity successful action):"
    )

    print(
        alpha_distribution.to_string()
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "ORACLE BY BASELINE ERROR"
    )

    print(
        "=" * 112
    )

    print(
        by_error.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
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
        PER_SAMPLE_OUT
    )

    print(
        SUMMARY_OUT
    )

    print(
        BY_CONDITION_OUT
    )

    print(
        BY_ERROR_OUT
    )

    print(
        "\nMULTI-STRENGTH ORACLE ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
