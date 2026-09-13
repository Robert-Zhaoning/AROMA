from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/calibration50/matched_attention/"
    "paired_head_contrasts.csv"
)

OUTPUT_DIR = Path(
    "outputs/calibration50/matched_attention"
)


def summarize_group(df, label):
    sub = df[
        df["wrong_label"] == label
    ].copy()

    summary = (
        sub
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
                        np.asarray(x) > 0
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

    return summary


def add_routing_score(df):
    """
    Exploratory ranking only.

    Desired routing-like pattern:
      1. wrong has larger object imbalance:
            delta_cv > 0

      2. weakest object gets less attention:
            delta_min < 0

      3. overall target attention does not change too much:
            abs(delta_mean) small

    This is NOT a mechanistic label.
    """

    out = df.copy()

    out["routing_score"] = (
        out["mean_delta_cv"]
        + (-out["mean_delta_min"]).clip(lower=0)
        - out["mean_delta_mean"].abs()
    )

    return out


def print_top(summary, title, n=20):
    q = summary[
        summary["query_group"] == "q_many"
    ].copy()

    q = add_routing_score(q)

    q = q.sort_values(
        [
            "positive_cv_fraction",
            "routing_score",
            "mean_delta_cv",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    cols = [
        "query_group",
        "layer",
        "head",
        "n_pairs",
        "mean_delta_cv",
        "median_delta_cv",
        "positive_cv_fraction",
        "mean_delta_min",
        "mean_delta_mean",
        "routing_score",
    ]

    print(
        q[cols]
        .head(n)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return q


def main():
    df = pd.read_csv(INPUT)

    print("=" * 80)
    print("AROMA Matched Attention — Label-Stratified Analysis")
    print("=" * 80)

    print("\nPairs by wrong label:")

    print(
        df[
            [
                "pair_id",
                "wrong_label",
            ]
        ]
        .drop_duplicates()
        ["wrong_label"]
        .value_counts()
        .to_string()
    )

    stable = summarize_group(
        df,
        "stable_wrong",
    )

    unstable = summarize_group(
        df,
        "unstable_wrong",
    )

    stable = add_routing_score(
        stable
    )

    unstable = add_routing_score(
        unstable
    )

    stable.to_csv(
        OUTPUT_DIR
        / "stable_wrong_head_summary.csv",
        index=False,
    )

    unstable.to_csv(
        OUTPUT_DIR
        / "unstable_wrong_head_summary.csv",
        index=False,
    )

    stable_ranked = print_top(
        stable,
        "STABLE-WRONG: top q_many routing-like candidates",
    )

    unstable_ranked = print_top(
        unstable,
        "UNSTABLE-WRONG: top q_many routing-like candidates",
    )

    # ------------------------------------------------------------
    # Compare heads that appear consistently in both groups
    # ------------------------------------------------------------

    merged = stable.merge(
        unstable,
        on=[
            "query_group",
            "layer",
            "head",
        ],
        suffixes=(
            "_stable",
            "_unstable",
        ),
    )

    merged["min_positive_fraction"] = np.minimum(
        merged["positive_cv_fraction_stable"],
        merged["positive_cv_fraction_unstable"],
    )

    merged["mean_cv_across_groups"] = (
        merged["mean_delta_cv_stable"]
        + merged["mean_delta_cv_unstable"]
    ) / 2

    merged["stable_specificity"] = (
        merged["mean_delta_cv_stable"]
        - merged["mean_delta_cv_unstable"]
    )

    merged.to_csv(
        OUTPUT_DIR
        / "stable_vs_unstable_head_comparison.csv",
        index=False,
    )

    q = merged[
        merged["query_group"] == "q_many"
    ].copy()

    q = q.sort_values(
        [
            "positive_cv_fraction_stable",
            "mean_delta_cv_stable",
        ],
        ascending=[
            False,
            False,
        ],
    )

    print("\n" + "=" * 80)
    print("STABLE vs UNSTABLE q_many comparison")
    print("=" * 80)

    cols = [
        "layer",
        "head",
        "n_pairs_stable",
        "mean_delta_cv_stable",
        "positive_cv_fraction_stable",
        "mean_delta_min_stable",
        "mean_delta_mean_stable",
        "n_pairs_unstable",
        "mean_delta_cv_unstable",
        "positive_cv_fraction_unstable",
        "stable_specificity",
    ]

    print(
        q[cols]
        .head(25)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved:")
    print("stable_wrong_head_summary.csv")
    print("unstable_wrong_head_summary.csv")
    print("stable_vs_unstable_head_comparison.csv")


if __name__ == "__main__":
    main()
