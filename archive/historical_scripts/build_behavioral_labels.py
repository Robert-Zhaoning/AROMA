from pathlib import Path

import pandas as pd


INPUT = Path("outputs/calibration50/samples_v2.csv")
OUTPUT = Path("outputs/calibration50/behavioral_labels.csv")


def main():
    df = pd.read_csv(INPUT)

    labels = []

    for _, row in df.iterrows():
        gt = row["ground_truth"]

        direct_correct = bool(row["direct_correct"])
        enum_correct = bool(row["enumeration_correct"])

        stability = float(row["prompt_stability"])
        coverage = float(row["prompt_coverage"])

        if direct_correct:
            label = "correct"

        elif enum_correct:
            label = "enumeration_rescue"

        elif stability == 1.0 and coverage == 1.0:
            label = "stable_wrong"

        elif stability < 0.75 or coverage < 1.0:
            label = "unstable_wrong"

        else:
            label = "other_wrong"

        labels.append(label)

    df["behavioral_label"] = labels

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 72)
    print("AROMA Behavioral Labels")
    print("=" * 72)

    print(
        df["behavioral_label"]
        .value_counts()
        .to_string()
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
