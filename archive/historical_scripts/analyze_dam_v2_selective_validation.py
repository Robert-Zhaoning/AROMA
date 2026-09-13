from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/proc_count_causal_v1/"
    "dam_v2_selective_validation/"
    "selective_validation_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "dam_v2_selective_validation"
)

SEED = 20260910
N_BOOT = 20000


def bootstrap_mean_ci(
    values,
    n_boot=N_BOOT,
    seed=SEED,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        len(values),
        size=(
            n_boot,
            len(values),
        ),
    )

    means = values[idx].mean(
        axis=1
    )

    return (
        float(values.mean()),
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def summarize_role_subset(df):
    rows = []

    for (
        role,
        subset,
    ), g in df.groupby(
        [
            "role",
            "subset",
        ]
    ):
        mean_logp, lo_logp, hi_logp = (
            bootstrap_mean_ci(
                g["delta_logp"]
            )
        )

        mean_margin, lo_margin, hi_margin = (
            bootstrap_mean_ci(
                g["delta_margin"]
            )
        )

        rows.append(
            {
                "role":
                    role,

                "subset":
                    subset,

                "n":
                    len(g),

                "mean_delta_logp":
                    mean_logp,

                "delta_logp_ci_low":
                    lo_logp,

                "delta_logp_ci_high":
                    hi_logp,

                "mean_delta_margin":
                    mean_margin,

                "delta_margin_ci_low":
                    lo_margin,

                "delta_margin_ci_high":
                    hi_margin,

                "positive_logp_fraction":
                    float(
                        (
                            g[
                                "delta_logp"
                            ]
                            > 0
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def paired_subset_comparison(
    df,
    subset_a,
    subset_b,
    role,
):
    sub = df[
        df["role"] == role
    ]

    a = (
        sub[
            sub["subset"]
            == subset_a
        ][
            [
                "pair_idx",
                "sample_id",
                "delta_logp",
                "delta_margin",
            ]
        ]
        .rename(
            columns={
                "delta_logp":
                    "delta_logp_a",

                "delta_margin":
                    "delta_margin_a",
            }
        )
    )

    b = (
        sub[
            sub["subset"]
            == subset_b
        ][
            [
                "pair_idx",
                "sample_id",
                "delta_logp",
                "delta_margin",
            ]
        ]
        .rename(
            columns={
                "delta_logp":
                    "delta_logp_b",

                "delta_margin":
                    "delta_margin_b",
            }
        )
    )

    merged = a.merge(
        b,
        on=[
            "pair_idx",
            "sample_id",
        ],
        how="inner",
    )

    merged[
        "diff_logp"
    ] = (
        merged[
            "delta_logp_a"
        ]
        -
        merged[
            "delta_logp_b"
        ]
    )

    merged[
        "diff_margin"
    ] = (
        merged[
            "delta_margin_a"
        ]
        -
        merged[
            "delta_margin_b"
        ]
    )

    mean_logp, lo_logp, hi_logp = (
        bootstrap_mean_ci(
            merged[
                "diff_logp"
            ]
        )
    )

    mean_margin, lo_margin, hi_margin = (
        bootstrap_mean_ci(
            merged[
                "diff_margin"
            ]
        )
    )

    return {
        "role":
            role,

        "subset_a":
            subset_a,

        "subset_b":
            subset_b,

        "n":
            len(merged),

        "mean_diff_logp":
            mean_logp,

        "diff_logp_ci_low":
            lo_logp,

        "diff_logp_ci_high":
            hi_logp,

        "mean_diff_margin":
            mean_margin,

        "diff_margin_ci_low":
            lo_margin,

        "diff_margin_ci_high":
            hi_margin,

        "a_beats_b_fraction":
            float(
                (
                    merged[
                        "diff_logp"
                    ]
                    > 0
                ).mean()
            ),
    }


def specificity_analysis(df):
    rows = []

    for subset, g in df.groupby(
        "subset"
    ):
        wrong = (
            g[
                g["role"]
                == "wrong"
            ][
                [
                    "pair_idx",
                    "delta_logp",
                    "delta_margin",
                ]
            ]
            .rename(
                columns={
                    "delta_logp":
                        "wrong_delta_logp",

                    "delta_margin":
                        "wrong_delta_margin",
                }
            )
        )

        correct = (
            g[
                g["role"]
                == "correct"
            ][
                [
                    "pair_idx",
                    "delta_logp",
                    "delta_margin",
                ]
            ]
            .rename(
                columns={
                    "delta_logp":
                        "correct_delta_logp",

                    "delta_margin":
                        "correct_delta_margin",
                }
            )
        )

        paired = wrong.merge(
            correct,
            on="pair_idx",
            how="inner",
        )

        paired[
            "specificity_logp"
        ] = (
            paired[
                "wrong_delta_logp"
            ]
            -
            paired[
                "correct_delta_logp"
            ]
        )

        paired[
            "specificity_margin"
        ] = (
            paired[
                "wrong_delta_margin"
            ]
            -
            paired[
                "correct_delta_margin"
            ]
        )

        mean_logp, lo_logp, hi_logp = (
            bootstrap_mean_ci(
                paired[
                    "specificity_logp"
                ]
            )
        )

        mean_margin, lo_margin, hi_margin = (
            bootstrap_mean_ci(
                paired[
                    "specificity_margin"
                ]
            )
        )

        rows.append(
            {
                "subset":
                    subset,

                "n_pairs":
                    len(paired),

                "mean_specificity_logp":
                    mean_logp,

                "specificity_logp_ci_low":
                    lo_logp,

                "specificity_logp_ci_high":
                    hi_logp,

                "mean_specificity_margin":
                    mean_margin,

                "specificity_margin_ci_low":
                    lo_margin,

                "specificity_margin_ci_high":
                    hi_margin,

                "wrong_benefits_more_fraction":
                    float(
                        (
                            paired[
                                "specificity_logp"
                            ]
                            > 0
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def condition_summary(df):
    return (
        df.groupby(
            [
                "subset",
                "role",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            n=(
                "sample_id",
                "count",
            ),

            mean_delta_logp=(
                "delta_logp",
                "mean",
            ),

            mean_delta_margin=(
                "delta_margin",
                "mean",
            ),

            positive_fraction=(
                "delta_logp",
                lambda x:
                    float(
                        (
                            x > 0
                        ).mean()
                    ),
            ),
        )
    )


def main():
    print("=" * 100)
    print(
        "AROMA DAM-v2 Selective "
        "Held-out Validation Analysis"
    )
    print("=" * 100)

    df = pd.read_csv(
        INPUT
    )

    print(
        "\nRows:",
        len(df),
    )

    print(
        "Pairs:",
        df[
            "pair_idx"
        ].nunique(),
    )

    print(
        "Samples:",
        df[
            "sample_id"
        ].nunique(),
    )

    print(
        "Subsets:",
        sorted(
            df[
                "subset"
            ].unique()
        ),
    )

    # ------------------------------------------------------
    # Basic summary
    # ------------------------------------------------------

    summary = summarize_role_subset(
        df
    )

    summary_path = (
        OUT_DIR
        /
        "heldout_subset_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "WRONG / CORRECT EFFECTS"
    )

    print("=" * 100)

    print(
        summary.sort_values(
            [
                "role",
                "mean_delta_logp",
            ],
            ascending=[
                True,
                False,
            ],
        ).to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Selective A against controls
    # ------------------------------------------------------

    comparisons = []

    controls = [
        "single_L3H4",
        "single_L8H30",
        "destructive_F",
        "all7",
    ]

    for role in [
        "wrong",
        "correct",
    ]:
        for control in controls:
            comparisons.append(
                paired_subset_comparison(
                    df=df,
                    subset_a="selective_A",
                    subset_b=control,
                    role=role,
                )
            )

    comp_df = pd.DataFrame(
        comparisons
    )

    comp_path = (
        OUT_DIR
        /
        "selective_A_comparisons.csv"
    )

    comp_df.to_csv(
        comp_path,
        index=False,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "SELECTIVE_A VS CONTROLS"
    )

    print("=" * 100)

    print(
        comp_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Failure-state specificity
    # ------------------------------------------------------

    specificity = (
        specificity_analysis(
            df
        )
    )

    specificity_path = (
        OUT_DIR
        /
        "failure_state_specificity.csv"
    )

    specificity.to_csv(
        specificity_path,
        index=False,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "FAILURE-STATE SPECIFICITY"
    )

    print("=" * 100)

    print(
        specificity.sort_values(
            "mean_specificity_logp",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Condition analysis
    # ------------------------------------------------------

    cond = condition_summary(
        df
    )

    cond_path = (
        OUT_DIR
        /
        "heldout_by_condition.csv"
    )

    cond.to_csv(
        cond_path,
        index=False,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "SELECTIVE_A BY CONDITION"
    )

    print("=" * 100)

    print(
        cond[
            cond["subset"]
            == "selective_A"
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    for path in [
        summary_path,
        comp_path,
        specificity_path,
        cond_path,
    ]:
        print(
            path
        )

    print(
        "\nANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
