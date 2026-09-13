from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/calibration50/attention_pilot/"
    "pilot_head_metrics.csv"
)

OUTPUT_DIR = Path(
    "outputs/calibration50/attention_pilot"
)


def main():
    df = pd.read_csv(
        INPUT
    )

    print("=" * 80)
    print("AROMA Attention Pilot Analysis")
    print("=" * 80)

    print("\nSamples per group:")

    counts = (
        df[
            [
                "sample_id",
                "behavioral_label",
            ]
        ]
        .drop_duplicates()
        ["behavioral_label"]
        .value_counts()
    )

    print(
        counts.to_string()
    )

    # ----------------------------------------------------------
    # Sample-level averaging
    # ----------------------------------------------------------

    sample_level = (
        df
        .groupby(
            [
                "sample_id",
                "behavioral_label",
                "query_group",
            ]
        )
        .agg(
            mean_head_enrichment=(
                "head_mean_enrichment",
                "mean",
            ),
            mean_head_cv=(
                "head_cv",
                "mean",
            ),
            mean_head_min=(
                "head_min_enrichment",
                "mean",
            ),
        )
        .reset_index()
    )

    sample_level.to_csv(
        OUTPUT_DIR
        / "pilot_sample_level.csv",
        index=False,
    )

    print("\nGroup means:")

    group_summary = (
        sample_level
        .groupby(
            [
                "behavioral_label",
                "query_group",
            ]
        )
        .agg(
            n=("sample_id", "nunique"),
            mean_enrichment=(
                "mean_head_enrichment",
                "mean",
            ),
            mean_cv=(
                "mean_head_cv",
                "mean",
            ),
            mean_min_enrichment=(
                "mean_head_min",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        group_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    group_summary.to_csv(
        OUTPUT_DIR
        / "pilot_group_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Head-wise group contrast
    # ----------------------------------------------------------

    head_level = (
        df
        .groupby(
            [
                "sample_id",
                "behavioral_label",
                "query_group",
                "layer",
                "head",
            ]
        )
        .agg(
            head_mean_enrichment=(
                "head_mean_enrichment",
                "first",
            ),
            head_cv=(
                "head_cv",
                "first",
            ),
            head_min_enrichment=(
                "head_min_enrichment",
                "first",
            ),
        )
        .reset_index()
    )

    group_head = (
        head_level
        .groupby(
            [
                "behavioral_label",
                "query_group",
                "layer",
                "head",
            ]
        )
        .agg(
            mean_cv=("head_cv", "mean"),
            mean_enrichment=(
                "head_mean_enrichment",
                "mean",
            ),
            mean_min=(
                "head_min_enrichment",
                "mean",
            ),
        )
        .reset_index()
    )

    # Pivot stable_wrong against correct
    correct = group_head[
        group_head["behavioral_label"]
        == "correct"
    ].copy()

    stable = group_head[
        group_head["behavioral_label"]
        == "stable_wrong"
    ].copy()

    merged = stable.merge(
        correct,
        on=[
            "query_group",
            "layer",
            "head",
        ],
        suffixes=(
            "_stable",
            "_correct",
        ),
    )

    merged[
        "delta_cv"
    ] = (
        merged["mean_cv_stable"]
        - merged["mean_cv_correct"]
    )

    merged[
        "delta_min_enrichment"
    ] = (
        merged["mean_min_stable"]
        - merged["mean_min_correct"]
    )

    merged[
        "delta_mean_enrichment"
    ] = (
        merged[
            "mean_enrichment_stable"
        ]
        - merged[
            "mean_enrichment_correct"
        ]
    )

    merged.to_csv(
        OUTPUT_DIR
        / "stable_vs_correct_heads.csv",
        index=False,
    )

    print(
        "\nTop candidate heads where stable_wrong "
        "has larger object imbalance than correct:"
    )

    top = (
        merged[
            merged["query_group"]
            == "q_many"
        ]
        .sort_values(
            "delta_cv",
            ascending=False,
        )
        .head(20)
    )

    print(
        top[
            [
                "query_group",
                "layer",
                "head",
                "mean_cv_stable",
                "mean_cv_correct",
                "delta_cv",
                "mean_min_stable",
                "mean_min_correct",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print("\nSaved:")
    print("pilot_sample_level.csv")
    print("pilot_group_summary.csv")
    print("stable_vs_correct_heads.csv")


if __name__ == "__main__":
    main()
