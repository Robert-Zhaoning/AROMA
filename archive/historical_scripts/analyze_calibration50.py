from pathlib import Path

import numpy as np
import pandas as pd


PATH = Path(
    "outputs/calibration50/samples.csv"
)


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    df = pd.read_csv(PATH)

    n = len(df)

    direct_acc = df[
        "direct_correct"
    ].mean()

    enum_acc = df[
        "enumeration_correct"
    ].mean()

    spatial_acc = df[
        "spatial_enumeration_correct"
    ].mean()

    direct_wrong = df[
        ~df["direct_correct"]
    ]

    if len(direct_wrong) > 0:

        enum_rescue = direct_wrong[
            "enumeration_correct"
        ].mean()

        spatial_rescue = direct_wrong[
            "spatial_enumeration_correct"
        ].mean()

    else:
        enum_rescue = np.nan
        spatial_rescue = np.nan

    enum_disagreement = (
        df["direct_pred"]
        != df["enumeration_pred"]
    ).mean()

    spatial_disagreement = (
        df["direct_pred"]
        != df[
            "spatial_enumeration_pred"
        ]
    ).mean()

    stable_wrong = (
        (
            df["prompt_stability"]
            == 1.0
        )
        &
        (
            ~df["direct_correct"]
        )
    ).mean()

    unstable = (
        df["prompt_stability"]
        < 1.0
    ).mean()

    print("=" * 72)
    print("AROMA Calibration Diagnostic Analysis")
    print("=" * 72)

    print("\nN =", n)

    print(
        "\nDirect accuracy:",
        pct(direct_acc),
    )

    print(
        "Enumeration accuracy:",
        pct(enum_acc),
    )

    print(
        "Spatial enumeration accuracy:",
        pct(spatial_acc),
    )

    print(
        "\nEnumeration disagreement with direct:",
        pct(enum_disagreement),
    )

    print(
        "Spatial enumeration disagreement with direct:",
        pct(spatial_disagreement),
    )

    if len(direct_wrong) > 0:

        print(
            "\nP(enum correct | direct wrong):",
            pct(enum_rescue),
        )

        print(
            "P(spatial enum correct | direct wrong):",
            pct(spatial_rescue),
        )

    print(
        "\nFully stable but wrong:",
        pct(stable_wrong),
    )

    print(
        "Prompt-unstable samples:",
        pct(unstable),
    )

    print("\nBy count:")

    cols = [
        "ground_truth",
        "direct_correct",
        "enumeration_correct",
        "spatial_enumeration_correct",
        "prompt_stability",
    ]

    grouped = (
        df[cols]
        .groupby("ground_truth")
        .mean()
    )

    print(
        grouped.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\nInterpretation guide:")

    print(
        """
A. Direct wrong + Enumeration correct:
   candidate post-direct / mapping-side failure.

B. Direct wrong + Enumeration wrong + high prompt stability:
   candidate stable perception/individuation failure.

C. Strong disagreement across prompts:
   native elicitation instability; do NOT assign P/R/M yet.

D. Enumeration and spatial enumeration both outperform direct:
   native evidence may be useful as a diagnostic representation.

E. All methods fail similarly as count increases:
   likely shared visual/individuation bottleneck rather than
   simple output-format failure.
"""
    )


if __name__ == "__main__":
    main()
