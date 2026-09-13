from pathlib import Path
import pandas as pd


LABELS = Path(
    "outputs/calibration50/behavioral_labels.csv"
)

OUTPUT = Path(
    "outputs/calibration50/attention_pilot/"
    "matched_pairs.csv"
)


def main():
    df = pd.read_csv(LABELS)

    correct = df[
        df["behavioral_label"] == "correct"
    ].copy()

    wrong = df[
        df["behavioral_label"].isin(
            [
                "stable_wrong",
                "unstable_wrong",
            ]
        )
    ].copy()

    pairs = []

    for _, w in wrong.iterrows():

        candidates = correct.copy()

        # Prefer same condition.
        same_condition = candidates[
            candidates["condition"]
            == w["condition"]
        ].copy()

        if len(same_condition) > 0:
            candidates = same_condition

        candidates[
            "count_distance"
        ] = (
            candidates["ground_truth"]
            - w["ground_truth"]
        ).abs()

        candidates = candidates.sort_values(
            [
                "count_distance",
                "ground_truth",
                "sample_id",
            ]
        )

        if len(candidates) == 0:
            continue

        c = candidates.iloc[0]

        pairs.append(
            {
                "wrong_sample_id":
                    w["sample_id"],

                "wrong_label":
                    w["behavioral_label"],

                "wrong_gt":
                    w["ground_truth"],

                "wrong_condition":
                    w["condition"],

                "correct_sample_id":
                    c["sample_id"],

                "correct_gt":
                    c["ground_truth"],

                "correct_condition":
                    c["condition"],

                "count_distance":
                    abs(
                        c["ground_truth"]
                        - w["ground_truth"]
                    ),

                "same_condition":
                    c["condition"]
                    == w["condition"],
            }
        )

    out = pd.DataFrame(pairs)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 80)
    print("AROMA Matched Pair Builder")
    print("=" * 80)

    print(
        out.to_string(
            index=False
        )
    )

    print("\nSummary:")

    print(
        "Pairs:",
        len(out),
    )

    print(
        "Same-condition fraction:",
        out["same_condition"].mean()
        if len(out) > 0
        else 0,
    )

    print(
        "Mean count distance:",
        out["count_distance"].mean()
        if len(out) > 0
        else 0,
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
