from pathlib import Path

import pandas as pd


INPUT = Path(
    "outputs/calibration50/attention_pilot/matched_pairs.csv"
)

OUTPUT = Path(
    "outputs/calibration50/attention_pilot/"
    "clean_matched_pairs.csv"
)


def main():
    df = pd.read_csv(INPUT)

    clean = df[
        (df["same_condition"] == True)
        & (df["count_distance"] <= 1)
    ].copy()

    clean = clean.sort_values(
        [
            "wrong_condition",
            "wrong_gt",
            "wrong_sample_id",
        ]
    )

    clean.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 80)
    print("AROMA Clean Matched Pairs")
    print("=" * 80)

    print(
        clean.to_string(
            index=False
        )
    )

    print("\nNumber of clean pairs:", len(clean))

    print(
        "Stable-wrong pairs:",
        (
            clean["wrong_label"]
            == "stable_wrong"
        ).sum(),
    )

    print(
        "Unstable-wrong pairs:",
        (
            clean["wrong_label"]
            == "unstable_wrong"
        ).sum(),
    )

    print(
        "Mean count distance:",
        clean["count_distance"].mean(),
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
