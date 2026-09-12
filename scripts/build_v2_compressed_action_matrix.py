from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


WRONG_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/multistrength_wrong_sweep/"
    "multistrength_wrong_results.csv"
)

CORRECT_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/compressed_actions_correct/"
    "compressed_actions_correct_results.csv"
)

CONF_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5_expanded_0_15/"
    "confirmation_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/compressed_action_matrix"
)

MATRIX_PATH = (
    OUT_DIR /
    "compressed_action_matrix.csv"
)

SUMMARY_PATH = (
    OUT_DIR /
    "compressed_action_summary.csv"
)

CONDITION_PATH = (
    OUT_DIR /
    "compressed_action_by_condition.csv"
)

ORACLE_PATH = (
    OUT_DIR /
    "compressed_oracle_per_sample.csv"
)


ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]


def bool_series(s):

    if s.dtype == bool:
        return s

    return (
        s.astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .astype(bool)
    )


def exact_mcnemar(
    repairs,
    breaks,
):

    repairs = int(repairs)
    breaks = int(breaks)

    n = (
        repairs
        +
        breaks
    )

    if n == 0:
        return 1.0

    k = min(
        repairs,
        breaks,
    )

    tail = sum(
        comb(n, i)
        for i in range(
            k + 1
        )
    )

    p = (
        2.0
        *
        tail
        /
        (2 ** n)
    )

    return min(
        1.0,
        float(p),
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print(
        "AROMA V2 COMPRESSED "
        "COUNTERFACTUAL ACTION MATRIX"
    )
    print("=" * 112)

    wrong = pd.read_csv(
        WRONG_PATH
    )

    correct = pd.read_csv(
        CORRECT_PATH
    )

    conf = pd.read_csv(
        CONF_PATH
    )

    # ========================================================
    # alpha = 0, 2, 4
    # baseline-wrong rows
    # ========================================================

    wrong = wrong[
        wrong["alpha"].isin(
            [
                0.0,
                2.0,
                4.0,
            ]
        )
    ].copy()

    if len(wrong) != (
        497 * 3
    ):
        raise RuntimeError(
            f"Wrong-action rows: "
            f"{len(wrong)}, expected 1491."
        )

    wrong_table = pd.DataFrame({
        "sample_id":
            wrong[
                "sample_id"
            ].astype(str),

        "ground_truth":
            wrong[
                "ground_truth"
            ].astype(int),

        "condition":
            wrong[
                "condition"
            ].astype(str),

        "alpha":
            wrong[
                "alpha"
            ].astype(float),

        "baseline_pred":
            wrong[
                "baseline_pred"
            ].astype(int),

        "action_pred":
            wrong[
                "modulated_pred"
            ].astype(int),

        "baseline_correct":
            False,

        "action_correct":
            bool_series(
                wrong[
                    "correct"
                ]
            ),

        "baseline_expected_numeral":
            wrong[
                "baseline_expected_numeral"
            ].astype(float),

        "action_expected_numeral":
            wrong[
                "modulated_expected_numeral"
            ].astype(float),

        "expected_shift":
            wrong[
                "expected_shift"
            ].astype(float),
    })

    # ========================================================
    # alpha = 0, 2, 4
    # baseline-correct rows
    # ========================================================

    if len(correct) != (
        503 * 3
    ):
        raise RuntimeError(
            f"Correct-action rows: "
            f"{len(correct)}, expected 1509."
        )

    correct_table = pd.DataFrame({
        "sample_id":
            correct[
                "sample_id"
            ].astype(str),

        "ground_truth":
            correct[
                "ground_truth"
            ].astype(int),

        "condition":
            correct[
                "condition"
            ].astype(str),

        "alpha":
            correct[
                "alpha"
            ].astype(float),

        "baseline_pred":
            correct[
                "baseline_pred"
            ].astype(int),

        "action_pred":
            correct[
                "modulated_pred"
            ].astype(int),

        "baseline_correct":
            True,

        "action_correct":
            bool_series(
                correct[
                    "modulated_correct"
                ]
            ),

        "baseline_expected_numeral":
            correct[
                "baseline_expected_numeral"
            ].astype(float),

        "action_expected_numeral":
            correct[
                "modulated_expected_numeral"
            ].astype(float),

        "expected_shift":
            correct[
                "expected_shift"
            ].astype(float),
    })

    # ========================================================
    # alpha = 1, 1.5
    # full 1000
    # ========================================================

    conf_rows = []

    for alpha in [
        1.0,
        1.5,
    ]:

        g = conf[
            np.isclose(
                conf[
                    "alpha"
                ].astype(float),
                alpha,
            )
        ].copy()

        if len(g) != 1000:
            raise RuntimeError(
                f"alpha={alpha}: "
                f"{len(g)} rows, expected 1000."
            )

        tmp = pd.DataFrame({
            "sample_id":
                g[
                    "sample_id"
                ].astype(str),

            "ground_truth":
                g[
                    "ground_truth"
                ].astype(int),

            "condition":
                g[
                    "condition"
                ].astype(str),

            "alpha":
                float(alpha),

            "baseline_pred":
                g[
                    "baseline_best_numeral"
                ].astype(int),

            "action_pred":
                g[
                    "modulated_best_numeral"
                ].astype(int),

            "baseline_correct":
                bool_series(
                    g[
                        "baseline_correct"
                    ]
                ),

            "action_correct":
                bool_series(
                    g[
                        "modulated_correct"
                    ]
                ),

            "baseline_expected_numeral":
                g[
                    "baseline_expected_numeral"
                ].astype(float),

            "action_expected_numeral":
                g[
                    "modulated_expected_numeral"
                ].astype(float),
        })

        tmp[
            "expected_shift"
        ] = (
            tmp[
                "action_expected_numeral"
            ]
            -
            tmp[
                "baseline_expected_numeral"
            ]
        )

        conf_rows.append(
            tmp
        )

    # ========================================================
    # Assemble full 1000 x 5 table
    # ========================================================

    table = pd.concat(
        [
            wrong_table,
            correct_table,
            *conf_rows,
        ],
        ignore_index=True,
    )

    table = (
        table
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

    if len(table) != 5000:
        raise RuntimeError(
            f"Matrix rows={len(table)}, "
            f"expected 5000."
        )

    if (
        table[
            "sample_id"
        ].nunique()
        != 1000
    ):
        raise RuntimeError(
            "Expected 1000 unique samples."
        )

    counts = (
        table.groupby(
            "sample_id"
        )[
            "alpha"
        ]
        .nunique()
    )

    if not (
        counts == 5
    ).all():
        raise RuntimeError(
            "Not every sample has 5 actions."
        )

    alpha_set = sorted(
        table[
            "alpha"
        ]
        .unique()
        .tolist()
    )

    if not np.allclose(
        alpha_set,
        ACTIONS,
    ):
        raise RuntimeError(
            f"Action set mismatch: {alpha_set}"
        )

    # ========================================================
    # Baseline invariance audit
    #
    # Discrete fields must match exactly.
    # Floating-point fields are checked with tolerance.
    # ========================================================

    for col in [
        "ground_truth",
        "condition",
        "baseline_pred",
        "baseline_correct",
    ]:

        invariant = (
            table.groupby(
                "sample_id"
            )[col]
            .nunique()
        )

        if not (
            invariant == 1
        ).all():

            raise RuntimeError(
                f"Baseline invariance "
                f"failed for {col}."
            )

    expected_ranges = (
        table.groupby(
            "sample_id"
        )[
            "baseline_expected_numeral"
        ]
        .agg(
            lambda x:
                float(
                    x.max()
                    -
                    x.min()
                )
        )
    )

    max_expected_range = float(
        expected_ranges.max()
    )

    print(
        "Max baseline expected-numeral "
        "range:",
        max_expected_range,
    )

    FLOAT_TOL = 1e-8

    if max_expected_range > FLOAT_TOL:

        bad = expected_ranges[
            expected_ranges
            >
            FLOAT_TOL
        ]

        raise RuntimeError(
            "Baseline expected-numeral "
            "invariance failed beyond "
            f"tolerance={FLOAT_TOL}: "
            f"max_range={max_expected_range}, "
            f"bad_samples={len(bad)}"
        )

    # ========================================================
    # Utility labels
    #
    # +1 = repairs a baseline error
    # -1 = breaks a baseline-correct answer
    #  0 = no correctness change
    # ========================================================

    table[
        "repair"
    ] = (
        (~table[
            "baseline_correct"
        ])
        &
        table[
            "action_correct"
        ]
    )

    table[
        "break_case"
    ] = (
        table[
            "baseline_correct"
        ]
        &
        (~table[
            "action_correct"
        ])
    )

    table[
        "utility"
    ] = 0

    table.loc[
        table[
            "repair"
        ],
        "utility"
    ] = 1

    table.loc[
        table[
            "break_case"
        ],
        "utility"
    ] = -1

    table[
        "prediction_shift"
    ] = (
        table[
            "action_pred"
        ]
        -
        table[
            "baseline_pred"
        ]
    )

    table.to_csv(
        MATRIX_PATH,
        index=False,
    )

    # ========================================================
    # Fixed-action summary
    # ========================================================

    summary_rows = []

    for alpha, g in table.groupby(
        "alpha"
    ):

        repairs = int(
            g[
                "repair"
            ].sum()
        )

        breaks = int(
            g[
                "break_case"
            ].sum()
        )

        net = (
            repairs
            -
            breaks
        )

        post_correct = (
            503
            +
            net
        )

        summary_rows.append({
            "alpha":
                float(alpha),

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                net,

            "post_correct":
                post_correct,

            "post_accuracy":
                post_correct / 1000,

            "changed_predictions":
                int(
                    (
                        g[
                            "prediction_shift"
                        ]
                        != 0
                    ).sum()
                ),

            "mean_expected_shift":
                float(
                    g[
                        "expected_shift"
                    ].mean()
                ),

            "mcnemar_exact_p":
                exact_mcnemar(
                    repairs,
                    breaks,
                ),
        })

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "alpha"
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # By condition
    # ========================================================

    condition_rows = []

    for (
        condition,
        alpha
    ), g in table.groupby(
        [
            "condition",
            "alpha",
        ]
    ):

        baseline_correct_n = int(
            g[
                "baseline_correct"
            ].sum()
        )

        repairs = int(
            g[
                "repair"
            ].sum()
        )

        breaks = int(
            g[
                "break_case"
            ].sum()
        )

        post_correct = (
            baseline_correct_n
            +
            repairs
            -
            breaks
        )

        condition_rows.append({
            "condition":
                condition,

            "alpha":
                float(alpha),

            "n":
                len(g),

            "baseline_correct":
                baseline_correct_n,

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                repairs - breaks,

            "post_accuracy":
                post_correct / len(g),

            "mean_expected_shift":
                float(
                    g[
                        "expected_shift"
                    ].mean()
                ),
        })

    by_condition = (
        pd.DataFrame(
            condition_rows
        )
        .sort_values(
            [
                "condition",
                "alpha",
            ]
        )
    )

    by_condition.to_csv(
        CONDITION_PATH,
        index=False,
    )

    # ========================================================
    # Compressed oracle
    # ========================================================

    oracle_rows = []

    for sid, g in table.groupby(
        "sample_id"
    ):

        base = g.iloc[0]

        if bool(
            base[
                "baseline_correct"
            ]
        ):

            chosen = g[
                np.isclose(
                    g[
                        "alpha"
                    ],
                    1.0,
                )
            ].iloc[0]

        else:

            successful = g[
                g[
                    "repair"
                ]
            ].copy()

            if len(successful):

                successful[
                    "distance_to_identity"
                ] = (
                    successful[
                        "alpha"
                    ]
                    -
                    1.0
                ).abs()

                chosen = (
                    successful
                    .sort_values(
                        [
                            "distance_to_identity",
                            "alpha",
                        ]
                    )
                    .iloc[0]
                )

            else:

                chosen = g[
                    np.isclose(
                        g[
                            "alpha"
                        ],
                        1.0,
                    )
                ].iloc[0]

        oracle_rows.append({
            "sample_id":
                sid,

            "ground_truth":
                int(
                    base[
                        "ground_truth"
                    ]
                ),

            "condition":
                str(
                    base[
                        "condition"
                    ]
                ),

            "baseline_pred":
                int(
                    base[
                        "baseline_pred"
                    ]
                ),

            "baseline_correct":
                bool(
                    base[
                        "baseline_correct"
                    ]
                ),

            "oracle_alpha":
                float(
                    chosen[
                        "alpha"
                    ]
                ),

            "oracle_pred":
                int(
                    chosen[
                        "action_pred"
                    ]
                ),

            "oracle_correct":
                bool(
                    chosen[
                        "action_correct"
                    ]
                ),

            "oracle_utility":
                int(
                    chosen[
                        "utility"
                    ]
                ),
        })

    oracle = pd.DataFrame(
        oracle_rows
    )

    oracle.to_csv(
        ORACLE_PATH,
        index=False,
    )

    oracle_correct = int(
        oracle[
            "oracle_correct"
        ].sum()
    )

    oracle_repairs = int(
        (
            (~oracle[
                "baseline_correct"
            ])
            &
            oracle[
                "oracle_correct"
            ]
        ).sum()
    )

    # ========================================================
    # Correct-sample fragility
    # ========================================================

    correct_nonnoop = table[
        table[
            "baseline_correct"
        ]
        &
        (~np.isclose(
            table[
                "alpha"
            ],
            1.0,
        ))
    ]

    fragile_correct = (
        correct_nonnoop
        .groupby(
            "sample_id"
        )[
            "break_case"
        ]
        .any()
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "Rows              :",
        len(table),
    )

    print(
        "Samples           :",
        table[
            "sample_id"
        ].nunique(),
    )

    print(
        "Actions/sample    :",
        5,
    )

    print(
        "Integrity audit   : PASS"
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "FIXED ACTION RESULTS"
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
        "COMPRESSED ORACLE"
    )

    print(
        "=" * 112
    )

    print(
        "Baseline accuracy :",
        503 / 1000,
    )

    print(
        "Oracle repairs    :",
        oracle_repairs,
    )

    print(
        "Oracle correct    :",
        oracle_correct,
    )

    print(
        "Oracle accuracy   :",
        oracle_correct / 1000,
    )

    print(
        "Oracle gain       :",
        (
            oracle_correct
            -
            503
        ) / 1000,
    )

    print(
        "\nOracle alpha distribution:"
    )

    print(
        oracle[
            "oracle_alpha"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "CORRECT-SAMPLE FRAGILITY"
    )

    print(
        "=" * 112
    )

    print(
        "Correct samples broken by "
        "AT LEAST one non-NOOP action:",
        int(
            fragile_correct.sum()
        ),
        "/ 503",
    )

    print(
        "\nSaved:"
    )

    print(
        MATRIX_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        CONDITION_PATH
    )

    print(
        ORACLE_PATH
    )

    print(
        "\nCOMPRESSED ACTION MATRIX COMPLETE"
    )


if __name__ == "__main__":
    main()
