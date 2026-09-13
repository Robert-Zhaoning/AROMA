from pathlib import Path

import numpy as np
import pandas as pd


RESULT_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "gain_dose_response/"
    "gain_dose_response_results.csv"
)

PAIR_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "exact_matched_pairs.csv"
)

OUTPUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "gain_dose_response"
)


BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260910


def bootstrap_mean_ci(
    values,
    seed,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
        )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    n = len(values)

    means = np.empty(
        BOOTSTRAP_N,
        dtype=float,
    )

    for i in range(
        BOOTSTRAP_N
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        means[i] = (
            values[
                indices
            ].mean()
        )

    low, high = (
        np.percentile(
            means,
            [
                2.5,
                97.5,
            ],
        )
    )

    return (
        float(low),
        float(high),
    )


def main():

    df = pd.read_csv(
        RESULT_PATH
    )

    pairs = pd.read_csv(
        PAIR_PATH
    )

    print("=" * 96)
    print(
        "Proc-Count-Causal v1 "
        "Gain Dose-Response Analysis"
    )
    print("=" * 96)

    print(
        "\nRows:",
        len(df),
    )

    print(
        "Samples:",
        df[
            "sample_id"
        ].nunique(),
    )

    print(
        "Heads:",
        df[
            [
                "head_type",
                "layer",
                "head",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "Alphas:",
        sorted(
            df[
                "alpha"
            ].unique()
        ),
    )

    # =========================================================
    # 1. Raw wrong/correct response by head type + alpha
    # =========================================================

    type_role = (
        df
        .groupby(
            [
                "head_type",
                "role",
                "alpha",
            ]
        )
        .agg(
            n_samples=(
                "sample_id",
                "nunique",
            ),

            mean_delta_logp=(
                "delta_gt_logp",
                "mean",
            ),

            median_delta_logp=(
                "delta_gt_logp",
                "median",
            ),

            mean_delta_margin=(
                "delta_margin",
                "mean",
            ),

            numeral_change_rate=(
                "numeral_changed",
                "mean",
            ),

            numeral_accuracy=(
                "modulated_numeral_correct",
                "mean",
            ),
        )
        .reset_index()
    )

    type_role.to_csv(
        OUTPUT_DIR
        / "dose_response_by_type_role.csv",
        index=False,
    )

    print(
        "\nDOSE RESPONSE BY HEAD TYPE / ROLE"
    )

    print(
        type_role.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # =========================================================
    # 2. Exact paired interaction for every head + alpha
    # =========================================================

    paired_rows = []

    for _, pair in (
        pairs.iterrows()
    ):

        pair_id = int(
            pair[
                "pair_id"
            ]
        )

        wrong_id = (
            pair[
                "wrong_sample_id"
            ]
        )

        correct_id = (
            pair[
                "correct_sample_id"
            ]
        )

        wrong = df[
            df[
                "sample_id"
            ]
            == wrong_id
        ].copy()

        correct = df[
            df[
                "sample_id"
            ]
            == correct_id
        ].copy()

        merged = wrong.merge(
            correct,
            on=[
                "head_type",
                "layer",
                "head",
                "alpha",
            ],
            suffixes=(
                "_wrong",
                "_correct",
            ),
        )

        for _, row in (
            merged.iterrows()
        ):

            paired_rows.append(
                {
                    "pair_id":
                        pair_id,

                    "ground_truth":
                        int(
                            pair[
                                "ground_truth"
                            ]
                        ),

                    "condition":
                        pair[
                            "condition"
                        ],

                    "head_type":
                        row[
                            "head_type"
                        ],

                    "layer":
                        int(
                            row[
                                "layer"
                            ]
                        ),

                    "head":
                        int(
                            row[
                                "head"
                            ]
                        ),

                    "alpha":
                        float(
                            row[
                                "alpha"
                            ]
                        ),

                    "effect_wrong":
                        row[
                            "delta_gt_logp_wrong"
                        ],

                    "effect_correct":
                        row[
                            "delta_gt_logp_correct"
                        ],

                    "interaction_logp":
                        (
                            row[
                                "delta_gt_logp_wrong"
                            ]
                            - row[
                                "delta_gt_logp_correct"
                            ]
                        ),

                    "wrong_margin_effect":
                        row[
                            "delta_margin_wrong"
                        ],

                    "correct_margin_effect":
                        row[
                            "delta_margin_correct"
                        ],

                    "interaction_margin":
                        (
                            row[
                                "delta_margin_wrong"
                            ]
                            - row[
                                "delta_margin_correct"
                            ]
                        ),

                    "wrong_repaired":
                        (
                            (
                                not bool(
                                    row[
                                        "baseline_numeral_correct_wrong"
                                    ]
                                )
                            )
                            and bool(
                                row[
                                    "modulated_numeral_correct_wrong"
                                ]
                            )
                        ),

                    "correct_broken":
                        (
                            bool(
                                row[
                                    "baseline_numeral_correct_correct"
                                ]
                            )
                            and (
                                not bool(
                                    row[
                                        "modulated_numeral_correct_correct"
                                    ]
                                )
                            )
                        ),
                }
            )

    paired = pd.DataFrame(
        paired_rows
    )

    paired.to_csv(
        OUTPUT_DIR
        / "paired_gain_interactions.csv",
        index=False,
    )

    # =========================================================
    # 3. Candidate set vs random control set by alpha
    # =========================================================

    pair_type = (
        paired
        .groupby(
            [
                "pair_id",
                "head_type",
                "alpha",
            ]
        )
        .agg(
            mean_effect_wrong=(
                "effect_wrong",
                "mean",
            ),

            mean_effect_correct=(
                "effect_correct",
                "mean",
            ),

            mean_interaction=(
                "interaction_logp",
                "mean",
            ),

            repair_rate=(
                "wrong_repaired",
                "mean",
            ),

            break_rate=(
                "correct_broken",
                "mean",
            ),
        )
        .reset_index()
    )

    type_alpha_rows = []

    for (
        head_type,
        alpha,
    ), group in (
        pair_type.groupby(
            [
                "head_type",
                "alpha",
            ]
        )
    ):

        wrong_values = (
            group[
                "mean_effect_wrong"
            ]
            .to_numpy(
                dtype=float
            )
        )

        interaction_values = (
            group[
                "mean_interaction"
            ]
            .to_numpy(
                dtype=float
            )
        )

        wrong_low, wrong_high = (
            bootstrap_mean_ci(
                wrong_values,
                seed=(
                    BOOTSTRAP_SEED
                    + int(
                        alpha
                        * 100
                    )
                ),
            )
        )

        int_low, int_high = (
            bootstrap_mean_ci(
                interaction_values,
                seed=(
                    BOOTSTRAP_SEED
                    + 1000
                    + int(
                        alpha
                        * 100
                    )
                ),
            )
        )

        type_alpha_rows.append(
            {
                "head_type":
                    head_type,

                "alpha":
                    alpha,

                "n_pairs":
                    group[
                        "pair_id"
                    ].nunique(),

                "mean_effect_wrong":
                    wrong_values.mean(),

                "wrong_ci_low":
                    wrong_low,

                "wrong_ci_high":
                    wrong_high,

                "mean_effect_correct":
                    group[
                        "mean_effect_correct"
                    ].mean(),

                "mean_interaction":
                    interaction_values.mean(),

                "interaction_ci_low":
                    int_low,

                "interaction_ci_high":
                    int_high,

                "mean_repair_rate":
                    group[
                        "repair_rate"
                    ].mean(),

                "mean_break_rate":
                    group[
                        "break_rate"
                    ].mean(),
            }
        )

    type_alpha = (
        pd.DataFrame(
            type_alpha_rows
        )
        .sort_values(
            [
                "head_type",
                "alpha",
            ]
        )
    )

    type_alpha.to_csv(
        OUTPUT_DIR
        / "dose_response_set_summary.csv",
        index=False,
    )

    print(
        "\nSET-LEVEL DOSE RESPONSE"
    )

    print(
        type_alpha.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # =========================================================
    # 4. Individual candidate-head response
    # =========================================================

    head_alpha = (
        paired
        .groupby(
            [
                "head_type",
                "layer",
                "head",
                "alpha",
            ]
        )
        .agg(
            n_pairs=(
                "pair_id",
                "nunique",
            ),

            mean_effect_wrong=(
                "effect_wrong",
                "mean",
            ),

            mean_effect_correct=(
                "effect_correct",
                "mean",
            ),

            mean_interaction=(
                "interaction_logp",
                "mean",
            ),

            positive_wrong_fraction=(
                "effect_wrong",
                lambda x:
                    np.mean(
                        np.asarray(
                            x
                        )
                        > 0
                    ),
            ),

            repair_fraction=(
                "wrong_repaired",
                "mean",
            ),

            break_fraction=(
                "correct_broken",
                "mean",
            ),
        )
        .reset_index()
    )

    head_alpha.to_csv(
        OUTPUT_DIR
        / "dose_response_by_head.csv",
        index=False,
    )

    candidate_gain = (
        head_alpha[
            (
                head_alpha[
                    "head_type"
                ]
                == "candidate"
            )
            &
            (
                head_alpha[
                    "alpha"
                ]
                > 1.0
            )
        ]
        .sort_values(
            [
                "alpha",
                "mean_effect_wrong",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        "\nINDIVIDUAL CANDIDATE HEADS "
        "FOR ALPHA > 1"
    )

    print(
        candidate_gain.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # =========================================================
    # 5. Condition sensitivity for candidate set
    # =========================================================

    candidate = paired[
        paired[
            "head_type"
        ]
        == "candidate"
    ].copy()

    candidate_condition = (
        candidate
        .groupby(
            [
                "condition",
                "alpha",
            ]
        )
        .agg(
            n_pairs=(
                "pair_id",
                "nunique",
            ),

            mean_wrong_effect=(
                "effect_wrong",
                "mean",
            ),

            mean_correct_effect=(
                "effect_correct",
                "mean",
            ),

            mean_interaction=(
                "interaction_logp",
                "mean",
            ),
        )
        .reset_index()
    )

    candidate_condition.to_csv(
        OUTPUT_DIR
        / "candidate_dose_by_condition.csv",
        index=False,
    )

    print(
        "\nCANDIDATE DOSE RESPONSE BY CONDITION"
    )

    print(
        candidate_condition.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "dose_response_by_type_role.csv"
    )

    print(
        OUTPUT_DIR
        / "paired_gain_interactions.csv"
    )

    print(
        OUTPUT_DIR
        / "dose_response_set_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "dose_response_by_head.csv"
    )

    print(
        OUTPUT_DIR
        / "candidate_dose_by_condition.csv"
    )


if __name__ == "__main__":
    main()
