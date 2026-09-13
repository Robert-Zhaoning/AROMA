from pathlib import Path

import numpy as np
import pandas as pd


RESULT_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "oracle_dam_v1_batch/"
    "oracle_dam_v1_batch_results.csv"
)

OUTPUT_DIR = RESULT_PATH.parent

BOOTSTRAP_N = 10000
SEED = 20260910


def bootstrap_ci(
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

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    means = np.empty(
        BOOTSTRAP_N,
        dtype=float,
    )

    for i in range(
        BOOTSTRAP_N
    ):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        means[i] = (
            values[
                idx
            ].mean()
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

    df = pd.read_csv(
        RESULT_PATH
    )

    print("=" * 100)
    print(
        "AROMA Oracle DAM-v1 Batch Analysis"
    )
    print("=" * 100)

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
                "layer",
                "head",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "Gammas:",
        sorted(
            df[
                "gamma"
            ].unique()
        ),
    )

    # ======================================================
    # Gamma=0 identity audit
    # ======================================================

    identity = df[
        df[
            "gamma"
        ]
        == 0.0
    ].copy()

    max_abs_logp = (
        identity[
            "delta_gt_logp"
        ]
        .abs()
        .max()
    )

    max_abs_margin = (
        identity[
            "delta_margin"
        ]
        .abs()
        .max()
    )

    pred_changes = (
        identity[
            "baseline_prediction"
        ]
        !=
        identity[
            "dam_prediction"
        ]
    ).sum()

    print(
        "\nIDENTITY AUDIT (gamma=0)"
    )

    print(
        "Max |delta logp|:",
        max_abs_logp,
    )

    print(
        "Max |delta margin|:",
        max_abs_margin,
    )

    print(
        "Prediction changes:",
        pred_changes,
    )

    # ======================================================
    # Overall by region type / role / gamma
    # ======================================================

    overall = (
        df
        .groupby(
            [
                "region_type",
                "role",
                "gamma",
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

            repairs=(
                "repaired",
                "sum",
            ),

            breaks=(
                "broken",
                "sum",
            ),

            mean_delta_object_cv=(
                "delta_object_cv",
                "mean",
            ),
        )
        .reset_index()
    )

    overall.to_csv(
        OUTPUT_DIR
        / "dam_overall.csv",
        index=False,
    )

    print(
        "\nOVERALL BY REGION / ROLE / GAMMA"
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # ======================================================
    # Collapse across frozen heads per sample.
    # This avoids treating 7 heads as independent samples.
    # ======================================================

    sample_level = (
        df
        .groupby(
            [
                "sample_id",
                "role",
                "condition",
                "ground_truth",
                "region_type",
                "gamma",
            ]
        )
        .agg(
            mean_delta_logp=(
                "delta_gt_logp",
                "mean",
            ),

            mean_delta_margin=(
                "delta_margin",
                "mean",
            ),

            mean_delta_object_cv=(
                "delta_object_cv",
                "mean",
            ),

            repair_fraction_heads=(
                "repaired",
                "mean",
            ),

            break_fraction_heads=(
                "broken",
                "mean",
            ),
        )
        .reset_index()
    )

    sample_level.to_csv(
        OUTPUT_DIR
        / "dam_sample_level.csv",
        index=False,
    )

    # ======================================================
    # Oracle-vs-random region contrast on wrong samples
    # ======================================================

    wrong = sample_level[
        sample_level[
            "role"
        ]
        == "wrong"
    ].copy()

    contrast_rows = []

    for gamma in sorted(
        wrong[
            "gamma"
        ].unique()
    ):

        if gamma == 0:
            continue

        g = wrong[
            wrong[
                "gamma"
            ]
            == gamma
        ]

        pivot = g.pivot(
            index="sample_id",
            columns="region_type",
            values="mean_delta_logp",
        ).dropna()

        if (
            "oracle_gt"
            not in pivot.columns
            or
            "random_region"
            not in pivot.columns
        ):
            continue

        values = (
            pivot[
                "oracle_gt"
            ]
            -
            pivot[
                "random_region"
            ]
        ).to_numpy(
            dtype=float
        )

        low, high = bootstrap_ci(
            values,
            seed=(
                SEED
                + int(
                    gamma
                    * 1000
                )
            ),
        )

        contrast_rows.append(
            {
                "gamma":
                    gamma,

                "n_samples":
                    len(values),

                "mean_oracle_minus_random":
                    values.mean(),

                "median_oracle_minus_random":
                    np.median(
                        values
                    ),

                "ci_low":
                    low,

                "ci_high":
                    high,

                "positive_fraction":
                    np.mean(
                        values
                        > 0
                    ),
            }
        )

    contrast = pd.DataFrame(
        contrast_rows
    )

    contrast.to_csv(
        OUTPUT_DIR
        / "dam_oracle_vs_random.csv",
        index=False,
    )

    print(
        "\nORACLE GT vs RANDOM REGION — WRONG SAMPLES"
    )

    if len(contrast):

        print(
            contrast.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.5f}",
            )
        )

    # ======================================================
    # Head-level
    # ======================================================

    head_summary = (
        df[
            df[
                "gamma"
            ]
            > 0
        ]
        .groupby(
            [
                "region_type",
                "layer",
                "head",
                "role",
                "gamma",
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

            mean_delta_margin=(
                "delta_margin",
                "mean",
            ),

            repair_rate=(
                "repaired",
                "mean",
            ),

            break_rate=(
                "broken",
                "mean",
            ),
        )
        .reset_index()
    )

    head_summary.to_csv(
        OUTPUT_DIR
        / "dam_by_head.csv",
        index=False,
    )

    print(
        "\nORACLE GT — WRONG SAMPLES BY HEAD"
    )

    oracle_wrong_heads = (
        head_summary[
            (
                head_summary[
                    "region_type"
                ]
                == "oracle_gt"
            )
            &
            (
                head_summary[
                    "role"
                ]
                == "wrong"
            )
        ]
        .sort_values(
            [
                "gamma",
                "mean_delta_logp",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        oracle_wrong_heads.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # ======================================================
    # Condition sensitivity
    # ======================================================

    by_condition = (
        sample_level[
            sample_level[
                "gamma"
            ]
            > 0
        ]
        .groupby(
            [
                "region_type",
                "role",
                "condition",
                "gamma",
            ]
        )
        .agg(
            n_samples=(
                "sample_id",
                "nunique",
            ),

            mean_delta_logp=(
                "mean_delta_logp",
                "mean",
            ),

            mean_delta_margin=(
                "mean_delta_margin",
                "mean",
            ),
        )
        .reset_index()
    )

    by_condition.to_csv(
        OUTPUT_DIR
        / "dam_by_condition.csv",
        index=False,
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

    # ======================================================
    # Object equalization diagnostic
    # ======================================================

    equalization = (
        sample_level[
            (
                sample_level[
                    "region_type"
                ]
                == "oracle_gt"
            )
            &
            (
                sample_level[
                    "gamma"
                ]
                > 0
            )
        ]
        .groupby(
            [
                "role",
                "gamma",
            ]
        )
        .agg(
            mean_delta_object_cv=(
                "mean_delta_object_cv",
                "mean",
            ),

            mean_delta_logp=(
                "mean_delta_logp",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        "\nOBJECT EQUALIZATION"
    )

    print(
        equalization.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    equalization.to_csv(
        OUTPUT_DIR
        / "dam_equalization_summary.csv",
        index=False,
    )

    print(
        "\nSaved outputs to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
