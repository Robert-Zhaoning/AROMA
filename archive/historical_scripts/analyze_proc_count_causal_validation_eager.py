from pathlib import Path

import numpy as np
import pandas as pd


RESULT_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "causal_validation/"
    "causal_validation_results.csv"
)

PAIR_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "exact_matched_pairs_eager.csv"
)

OUTPUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "causal_validation"
)

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260910


def bootstrap_ci(
    values,
    n_boot=BOOTSTRAP_N,
    seed=BOOTSTRAP_SEED,
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

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        n_boot,
        dtype=float,
    )

    n = len(values)

    for i in range(
        n_boot
    ):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means[i] = (
            sample.mean()
        )

    low, high = np.percentile(
        means,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(low),
        float(high),
    )


def main():

    results = pd.read_csv(
        RESULT_PATH
    )

    pairs = pd.read_csv(
        PAIR_PATH
    )

    pair_rows = []

    for _, pair in (
        pairs.iterrows()
    ):

        pair_id = (
            pair["pair_id"]
        )

        correct_id = (
            pair[
                "correct_sample_id"
            ]
        )

        wrong_id = (
            pair[
                "wrong_sample_id"
            ]
        )

        correct = results[
            results["sample_id"]
            == correct_id
        ].copy()

        wrong = results[
            results["sample_id"]
            == wrong_id
        ].copy()

        merged = wrong.merge(
            correct,
            on=[
                "head_type",
                "layer",
                "head",
            ],
            suffixes=(
                "_wrong",
                "_correct",
            ),
        )

        for _, row in (
            merged.iterrows()
        ):

            pair_rows.append(
                {
                    "pair_id":
                        pair_id,

                    "ground_truth":
                        pair[
                            "ground_truth"
                        ],

                    "condition":
                        pair[
                            "condition"
                        ],

                    "wrong_sample_id":
                        wrong_id,

                    "correct_sample_id":
                        correct_id,

                    "head_type":
                        row[
                            "head_type"
                        ],

                    "layer":
                        row[
                            "layer"
                        ],

                    "head":
                        row[
                            "head"
                        ],

                    "effect_wrong_logp":
                        row[
                            "delta_gt_logp_wrong"
                        ],

                    "effect_correct_logp":
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

                    "effect_wrong_margin":
                        row[
                            "delta_margin_wrong"
                        ],

                    "effect_correct_margin":
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
                }
            )

    paired = pd.DataFrame(
        pair_rows
    )

    paired_path = (
        OUTPUT_DIR
        / "paired_validation_interactions.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Head-level validation
    # ---------------------------------------------------------

    summaries = []

    grouped = paired.groupby(
        [
            "head_type",
            "layer",
            "head",
        ]
    )

    for (
        head_type,
        layer,
        head,
    ), group in grouped:

        interactions = (
            group[
                "interaction_logp"
            ]
            .to_numpy(
                dtype=float
            )
        )

        ci_low, ci_high = (
            bootstrap_ci(
                interactions
            )
        )

        summaries.append(
            {
                "head_type":
                    head_type,

                "layer":
                    layer,

                "head":
                    head,

                "n_pairs":
                    len(group),

                "mean_effect_wrong":
                    group[
                        "effect_wrong_logp"
                    ].mean(),

                "mean_effect_correct":
                    group[
                        "effect_correct_logp"
                    ].mean(),

                "mean_interaction_logp":
                    interactions.mean(),

                "median_interaction_logp":
                    np.median(
                        interactions
                    ),

                "interaction_ci_low":
                    ci_low,

                "interaction_ci_high":
                    ci_high,

                "negative_fraction":
                    np.mean(
                        interactions
                        < 0
                    ),

                "mean_abs_interaction":
                    np.mean(
                        np.abs(
                            interactions
                        )
                    ),

                "mean_interaction_margin":
                    group[
                        "interaction_margin"
                    ].mean(),
            }
        )

    summary = pd.DataFrame(
        summaries
    )

    # Candidate ranking is now purely reporting.
    # DO NOT select/redefine candidates from this validation set.
    summary = summary.sort_values(
        [
            "head_type",
            "mean_interaction_logp",
        ]
    )

    summary_path = (
        OUTPUT_DIR
        / "frozen_head_validation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Candidate-vs-random as two pre-frozen sets
    # ---------------------------------------------------------

    per_pair_type = (
        paired
        .groupby(
            [
                "pair_id",
                "head_type",
            ]
        )
        .agg(
            mean_interaction_logp=(
                "interaction_logp",
                "mean",
            )
        )
        .reset_index()
    )

    pivot = (
        per_pair_type
        .pivot(
            index="pair_id",
            columns="head_type",
            values="mean_interaction_logp",
        )
        .dropna()
    )

    pivot[
        "candidate_minus_random"
    ] = (
        pivot[
            "candidate"
        ]
        - pivot[
            "random"
        ]
    )

    contrast_values = (
        pivot[
            "candidate_minus_random"
        ]
        .to_numpy(
            dtype=float
        )
    )

    contrast_low, contrast_high = (
        bootstrap_ci(
            contrast_values,
            seed=BOOTSTRAP_SEED + 1,
        )
    )

    contrast_mean = (
        contrast_values.mean()
    )

    # ---------------------------------------------------------
    # Stratified summaries
    # ---------------------------------------------------------

    by_condition = (
        paired
        .groupby(
            [
                "head_type",
                "condition",
            ]
        )
        .agg(
            n_pairs=(
                "pair_id",
                "nunique",
            ),
            mean_interaction_logp=(
                "interaction_logp",
                "mean",
            ),
        )
        .reset_index()
    )

    by_count = (
        paired
        .groupby(
            [
                "head_type",
                "ground_truth",
            ]
        )
        .agg(
            n_pairs=(
                "pair_id",
                "nunique",
            ),
            mean_interaction_logp=(
                "interaction_logp",
                "mean",
            ),
        )
        .reset_index()
    )

    by_condition.to_csv(
        OUTPUT_DIR
        / "validation_by_condition.csv",
        index=False,
    )

    by_count.to_csv(
        OUTPUT_DIR
        / "validation_by_count.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("=" * 90)
    print(
        "Proc-Count-Causal v1 "
        "Held-out Causal Validation Analysis"
    )
    print("=" * 90)

    print(
        "\nExact pairs analyzed:",
        paired[
            "pair_id"
        ].nunique(),
    )

    print(
        "\nFROZEN HEAD VALIDATION"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    print(
        "\nCANDIDATE SET vs RANDOM CONTROL SET"
    )

    print(
        "Pairs:",
        len(
            contrast_values
        ),
    )

    print(
        "Mean candidate-random interaction:",
        f"{contrast_mean:+.6f}",
    )

    print(
        "95% bootstrap CI:",
        f"[{contrast_low:+.6f}, "
        f"{contrast_high:+.6f}]",
    )

    print(
        "\nInterpretation of sign:"
    )

    print(
        "  More negative = ablation harms "
        "wrong samples more than matched correct samples."
    )

    print(
        "  candidate_minus_random < 0 therefore "
        "supports stronger failure-state dependence "
        "for frozen candidates."
    )

    print(
        "\nBY CONDITION"
    )

    print(
        by_condition.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    print(
        "\nBY COUNT"
    )

    print(
        by_count.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    print(
        "\nSaved:"
    )

    print(
        paired_path
    )

    print(
        summary_path
    )

    print(
        OUTPUT_DIR
        / "validation_by_condition.csv"
    )

    print(
        OUTPUT_DIR
        / "validation_by_count.csv"
    )


if __name__ == "__main__":
    main()
