from pathlib import Path

import numpy as np
import pandas as pd


RESULT_PATH = Path(
    "outputs/calibration50/exhaustive_causal/"
    "exhaustive_causal_results.csv"
)

PAIR_PATH = Path(
    "outputs/calibration50/attention_pilot/"
    "clean_matched_pairs.csv"
)

OUTPUT_DIR = Path(
    "outputs/calibration50/exhaustive_causal"
)


def main():

    results = pd.read_csv(
        RESULT_PATH
    )

    pairs = pd.read_csv(
        PAIR_PATH
    )

    paired_rows = []

    # ----------------------------------------------------------
    # Build causal effect difference:
    #
    # intervention effect on wrong
    # minus intervention effect on correct
    # ----------------------------------------------------------

    for pair_id, pair in (
        pairs.reset_index()
        .iterrows()
    ):

        wrong = results[
            results["sample_id"]
            == pair["wrong_sample_id"]
        ].copy()

        correct = results[
            results["sample_id"]
            == pair["correct_sample_id"]
        ].copy()

        merged = wrong.merge(
            correct,
            on=[
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

            paired_rows.append(
                {
                    "pair_id":
                        pair_id,

                    "wrong_label":
                        pair[
                            "wrong_label"
                        ],

                    "wrong_sample_id":
                        pair[
                            "wrong_sample_id"
                        ],

                    "correct_sample_id":
                        pair[
                            "correct_sample_id"
                        ],

                    "layer":
                        row[
                            "layer"
                        ],

                    "head":
                        row[
                            "head"
                        ],

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
        paired_rows
    )

    paired.to_csv(
        OUTPUT_DIR
        / "paired_causal_interactions.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # All-pair head summary
    # ----------------------------------------------------------

    summary = (
        paired
        .groupby(
            [
                "layer",
                "head",
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

            mean_interaction_logp=(
                "interaction_logp",
                "mean",
            ),

            mean_abs_interaction_logp=(
                "interaction_logp",
                lambda x:
                    np.abs(x).mean(),
            ),

            negative_interaction_fraction=(
                "interaction_logp",
                lambda x:
                    np.mean(
                        np.asarray(x)
                        < 0
                    ),
            ),

            positive_interaction_fraction=(
                "interaction_logp",
                lambda x:
                    np.mean(
                        np.asarray(x)
                        > 0
                    ),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "causal_head_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Stable-only
    # ----------------------------------------------------------

    stable = paired[
        paired["wrong_label"]
        == "stable_wrong"
    ]

    stable_summary = (
        stable
        .groupby(
            [
                "layer",
                "head",
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

            mean_interaction_logp=(
                "interaction_logp",
                "mean",
            ),

            median_interaction_logp=(
                "interaction_logp",
                "median",
            ),

            negative_fraction=(
                "interaction_logp",
                lambda x:
                    np.mean(
                        np.asarray(x)
                        < 0
                    ),
            ),

            mean_abs_interaction=(
                "interaction_logp",
                lambda x:
                    np.abs(x).mean(),
            ),
        )
        .reset_index()
    )

    stable_summary.to_csv(
        OUTPUT_DIR
        / "stable_causal_head_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Ranking:
    #
    # negative interaction means ablation hurts GT logp
    # more in stable-wrong than in matched correct.
    # ----------------------------------------------------------

    ranked = stable_summary.sort_values(
        [
            "negative_fraction",
            "mean_interaction_logp",
        ],
        ascending=[
            False,
            True,
        ],
    )

    print("=" * 80)
    print(
        "AROMA Exhaustive Causal Screen Analysis"
    )
    print("=" * 80)

    print(
        "\nTotal heads:",
        len(summary),
    )

    print(
        "Stable pairs:",
        stable["pair_id"].nunique(),
    )

    print(
        "\nTop stable-wrong "
        "failure-state-dependent heads:"
    )

    print(
        ranked.head(30)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # ----------------------------------------------------------
    # Locate our previous candidates
    # ----------------------------------------------------------

    old_candidates = [
        (33, 21),
        (33, 23),
        (13, 11),
        (13, 10),
    ]

    print(
        "\nPrevious observational candidates:"
    )

    for layer, head in old_candidates:

        row = stable_summary[
            (
                stable_summary["layer"]
                == layer
            )
            & (
                stable_summary["head"]
                == head
            )
        ]

        if len(row) == 1:

            rank = (
                ranked
                .reset_index(drop=True)
            )

            rank_match = rank[
                (
                    rank["layer"]
                    == layer
                )
                & (
                    rank["head"]
                    == head
                )
            ]

            rank_number = (
                int(
                    rank_match.index[0]
                )
                + 1
                if len(rank_match)
                else None
            )

            r = row.iloc[0]

            print(
                f"  L{layer}H{head}: "
                f"rank={rank_number}, "
                f"interaction="
                f"{r['mean_interaction_logp']:+.5f}, "
                f"negative_fraction="
                f"{r['negative_fraction']:.3f}"
            )

    print(
        "\nSaved:"
    )

    print(
        "paired_causal_interactions.csv"
    )

    print(
        "causal_head_summary.csv"
    )

    print(
        "stable_causal_head_summary.csv"
    )


if __name__ == "__main__":
    main()
