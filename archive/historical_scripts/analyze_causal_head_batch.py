from pathlib import Path

import pandas as pd


INPUT = Path(
    "outputs/calibration50/causal_head_batch/"
    "causal_head_results.csv"
)


def main():
    df = pd.read_csv(INPUT)

    print("=" * 80)
    print("AROMA Causal Head Batch Analysis")
    print("=" * 80)

    print("\nHEAD-LEVEL EFFECTS")

    head_summary = (
        df
        .groupby(
            [
                "head_type",
                "layer",
                "head",
            ]
        )
        .agg(
            n=("sample_id", "nunique"),
            mean_abs_delta_logp=(
                "delta_gt_logp",
                lambda x: x.abs().mean(),
            ),
            mean_abs_delta_margin=(
                "delta_margin",
                lambda x: x.abs().mean(),
            ),
            mean_delta_logp=(
                "delta_gt_logp",
                "mean",
            ),
            mean_delta_margin=(
                "delta_margin",
                "mean",
            ),
            prediction_change_rate=(
                "prediction_changed",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_abs_delta_margin",
            ascending=False,
        )
    )

    print(
        head_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\nCANDIDATE VS RANDOM")

    type_summary = (
        df
        .groupby("head_type")
        .agg(
            mean_abs_delta_logp=(
                "delta_gt_logp",
                lambda x: x.abs().mean(),
            ),
            mean_abs_delta_margin=(
                "delta_margin",
                lambda x: x.abs().mean(),
            ),
            prediction_change_rate=(
                "prediction_changed",
                "mean",
            ),
        )
    )

    print(
        type_summary.to_string(
            float_format=lambda x: f"{x:.5f}"
        )
    )

    print("\nBY ROLE")

    by_role = (
        df
        .groupby(
            [
                "head_type",
                "role",
            ]
        )
        .agg(
            n=("sample_id", "nunique"),
            mean_delta_logp=(
                "delta_gt_logp",
                "mean",
            ),
            mean_abs_delta_logp=(
                "delta_gt_logp",
                lambda x: x.abs().mean(),
            ),
            mean_delta_margin=(
                "delta_margin",
                "mean",
            ),
            mean_abs_delta_margin=(
                "delta_margin",
                lambda x: x.abs().mean(),
            ),
        )
        .reset_index()
    )

    print(
        by_role.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\nINDIVIDUAL CANDIDATES BY ROLE")

    candidate = df[
        df["head_type"] == "candidate"
    ]

    cand_role = (
        candidate
        .groupby(
            [
                "layer",
                "head",
                "role",
            ]
        )
        .agg(
            n=("sample_id", "nunique"),
            mean_delta_logp=(
                "delta_gt_logp",
                "mean",
            ),
            mean_delta_margin=(
                "delta_margin",
                "mean",
            ),
            mean_abs_delta_margin=(
                "delta_margin",
                lambda x: x.abs().mean(),
            ),
        )
        .reset_index()
    )

    print(
        cand_role.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )


if __name__ == "__main__":
    main()
