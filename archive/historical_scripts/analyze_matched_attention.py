from pathlib import Path

import numpy as np
import pandas as pd


PAIR_PATH = Path(
    "outputs/calibration50/attention_pilot/"
    "clean_matched_pairs.csv"
)

METRIC_PATH = Path(
    "outputs/calibration50/matched_attention/"
    "matched_head_metrics.csv"
)

OUTPUT = Path(
    "outputs/calibration50/matched_attention/"
    "paired_head_contrasts.csv"
)


def main():

    pairs = pd.read_csv(
        PAIR_PATH
    )

    metrics = pd.read_csv(
        METRIC_PATH
    )

    rows = []

    for pair_id, pair in (
        pairs.reset_index()
        .iterrows()
    ):

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

        wrong = metrics[
            metrics["sample_id"]
            == wrong_id
        ].copy()

        correct = metrics[
            metrics["sample_id"]
            == correct_id
        ].copy()

        merged = wrong.merge(
            correct,
            on=[
                "query_group",
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

            rows.append(
                {
                    "pair_id":
                        pair_id,

                    "wrong_sample_id":
                        wrong_id,

                    "correct_sample_id":
                        correct_id,

                    "wrong_label":
                        pair[
                            "wrong_label"
                        ],

                    "condition":
                        pair[
                            "wrong_condition"
                        ],

                    "wrong_gt":
                        pair[
                            "wrong_gt"
                        ],

                    "correct_gt":
                        pair[
                            "correct_gt"
                        ],

                    "query_group":
                        row[
                            "query_group"
                        ],

                    "layer":
                        row[
                            "layer"
                        ],

                    "head":
                        row[
                            "head"
                        ],

                    "delta_cv":
                        row[
                            "cv_wrong"
                        ]
                        - row[
                            "cv_correct"
                        ],

                    "delta_min":
                        row[
                            "min_enrichment_wrong"
                        ]
                        - row[
                            "min_enrichment_correct"
                        ],

                    "delta_mean":
                        row[
                            "mean_enrichment_wrong"
                        ]
                        - row[
                            "mean_enrichment_correct"
                        ],
                }
            )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    # ----------------------------------------------------------
    # Aggregate over pairs
    # ----------------------------------------------------------

    summary = (
        out
        .groupby(
            [
                "query_group",
                "layer",
                "head",
            ]
        )
        .agg(
            n_pairs=(
                "pair_id",
                "nunique",
            ),

            mean_delta_cv=(
                "delta_cv",
                "mean",
            ),

            median_delta_cv=(
                "delta_cv",
                "median",
            ),

            positive_cv_fraction=(
                "delta_cv",
                lambda x:
                    np.mean(
                        np.asarray(x)
                        > 0
                    ),
            ),

            mean_delta_min=(
                "delta_min",
                "mean",
            ),

            mean_delta_mean=(
                "delta_mean",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        OUTPUT.parent
        / "paired_head_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("=" * 80)
    print("AROMA Clean Matched Attention Analysis")
    print("=" * 80)

    print(
        "\nPairs analyzed:",
        out["pair_id"].nunique(),
    )

    q_many = (
        summary[
            summary[
                "query_group"
            ]
            == "q_many"
        ]
        .sort_values(
            [
                "positive_cv_fraction",
                "mean_delta_cv",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(20)
    )

    print(
        "\nTop q_many heads by paired consistency:"
    )

    print(
        q_many.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print("\nSaved:")
    print(OUTPUT)
    print(summary_path)


if __name__ == "__main__":
    main()
