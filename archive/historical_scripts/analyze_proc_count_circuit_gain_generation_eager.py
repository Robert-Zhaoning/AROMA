from pathlib import Path

import pandas as pd


PATH = Path(
    "outputs/proc_count_causal_v1/"
    "circuit_gain_generation_eager/"
    "circuit_gain_generation_results.csv"
)


def main():

    df = pd.read_csv(PATH)

    baseline = (
        df[
            df[
                "circuit_type"
            ]
            == "baseline"
        ]
        .set_index(
            "sample_id"
        )
    )

    test = df[
        df[
            "circuit_type"
        ]
        != "baseline"
    ].copy()

    print("=" * 90)
    print(
        "Distributed Circuit Gain "
        "Generation Analysis"
    )
    print("=" * 90)

    print(
        "\nBaseline accuracy:"
    )

    print(
        baseline[
            "correct"
        ].mean()
    )

    # -----------------------------------------------------
    # Overall
    # -----------------------------------------------------

    overall = (
        test
        .groupby(
            [
                "circuit_type",
                "alpha",
            ]
        )
        .agg(
            accuracy=(
                "correct",
                "mean",
            ),

            mean_delta_logp=(
                "delta_gt_logp",
                "mean",
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
        )
        .reset_index()
    )

    print(
        "\nOVERALL"
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # -----------------------------------------------------
    # Wrong samples only
    # -----------------------------------------------------

    wrong = test[
        test[
            "role"
        ]
        == "wrong"
    ]

    wrong_summary = (
        wrong
        .groupby(
            [
                "circuit_type",
                "alpha",
            ]
        )
        .agg(
            n=(
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

            repairs=(
                "repaired",
                "sum",
            ),

            resulting_accuracy=(
                "correct",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        "\nWRONG SAMPLES"
    )

    print(
        wrong_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # -----------------------------------------------------
    # Correct samples only
    # -----------------------------------------------------

    correct = test[
        test[
            "role"
        ]
        == "correct"
    ]

    correct_summary = (
        correct
        .groupby(
            [
                "circuit_type",
                "alpha",
            ]
        )
        .agg(
            n=(
                "sample_id",
                "nunique",
            ),

            mean_delta_logp=(
                "delta_gt_logp",
                "mean",
            ),

            breaks=(
                "broken",
                "sum",
            ),

            resulting_accuracy=(
                "correct",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        "\nCORRECT SAMPLES"
    )

    print(
        correct_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    # -----------------------------------------------------
    # Condition
    # -----------------------------------------------------

    condition = (
        test
        .groupby(
            [
                "circuit_type",
                "alpha",
                "role",
                "condition",
            ]
        )
        .agg(
            n=(
                "sample_id",
                "nunique",
            ),

            mean_delta_logp=(
                "delta_gt_logp",
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
        )
        .reset_index()
    )

    print(
        "\nBY CONDITION"
    )

    print(
        condition.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    condition.to_csv(
        PATH.parent
        / "circuit_gain_by_condition.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
