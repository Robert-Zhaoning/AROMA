import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "outputs/calibration50/results.jsonl"
)

OUTPUT_DIR = Path(
    "outputs/calibration50"
)

PROMPTS = [
    "direct",
    "count_json",
    "enumeration",
    "spatial_enumeration",
]


def safe_abs_error(pred, gt):
    if pred is None:
        return np.nan

    return abs(pred - gt)


def main():
    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            raw = json.loads(line)

            row = {
                "sample_id":
                    raw["sample_id"],

                "ground_truth":
                    raw["ground_truth"],

                "condition":
                    raw["condition"],

                "prompt_stability":
                    raw.get(
                        "prompt_stability"
                    ),
            }

            gt = raw[
                "ground_truth"
            ]

            for prompt in PROMPTS:
                out = raw[
                    "outputs"
                ][prompt]

                pred = out.get(
                    "parsed_count"
                )

                row[
                    f"{prompt}_pred"
                ] = pred

                row[
                    f"{prompt}_correct"
                ] = pred == gt

                row[
                    f"{prompt}_abs_error"
                ] = safe_abs_error(
                    pred,
                    gt,
                )

                row[
                    f"{prompt}_parse_success"
                ] = (
                    pred is not None
                )

                row[
                    f"{prompt}_elapsed"
                ] = out.get(
                    "elapsed_seconds"
                )

            records.append(row)

    df = pd.DataFrame(
        records
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples_path = (
        OUTPUT_DIR
        / "samples.csv"
    )

    df.to_csv(
        samples_path,
        index=False,
    )

    summary_rows = []

    for prompt in PROMPTS:
        accuracy = df[
            f"{prompt}_correct"
        ].mean()

        mae = df[
            f"{prompt}_abs_error"
        ].mean()

        parse_rate = df[
            f"{prompt}_parse_success"
        ].mean()

        mean_time = df[
            f"{prompt}_elapsed"
        ].mean()

        summary_rows.append(
            {
                "prompt": prompt,
                "accuracy": accuracy,
                "mae": mae,
                "parse_rate": parse_rate,
                "mean_seconds": mean_time,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    by_count = []

    for gt, group in df.groupby(
        "ground_truth"
    ):
        row = {
            "ground_truth": gt,
            "n": len(group),
        }

        for prompt in PROMPTS:
            row[
                f"{prompt}_accuracy"
            ] = group[
                f"{prompt}_correct"
            ].mean()

            row[
                f"{prompt}_mae"
            ] = group[
                f"{prompt}_abs_error"
            ].mean()

        by_count.append(row)

    by_count_df = pd.DataFrame(
        by_count
    )

    by_count_df.to_csv(
        OUTPUT_DIR
        / "by_count.csv",
        index=False,
    )

    by_condition = []

    for condition, group in df.groupby(
        "condition"
    ):
        row = {
            "condition": condition,
            "n": len(group),
        }

        for prompt in PROMPTS:
            row[
                f"{prompt}_accuracy"
            ] = group[
                f"{prompt}_correct"
            ].mean()

            row[
                f"{prompt}_mae"
            ] = group[
                f"{prompt}_abs_error"
            ].mean()

        by_condition.append(row)

    by_condition_df = pd.DataFrame(
        by_condition
    )

    by_condition_df.to_csv(
        OUTPUT_DIR
        / "by_condition.csv",
        index=False,
    )

    direct_wrong = df[
        ~df[
            "direct_correct"
        ]
    ]

    if len(direct_wrong) > 0:
        enum_rescue = (
            direct_wrong[
                "enumeration_correct"
            ].mean()
        )

        spatial_rescue = (
            direct_wrong[
                "spatial_enumeration_correct"
            ].mean()
        )
    else:
        enum_rescue = np.nan
        spatial_rescue = np.nan

    exact_agreement = (
        df[
            [
                f"{p}_pred"
                for p in PROMPTS
            ]
        ]
        .nunique(
            axis=1,
            dropna=False,
        )
        == 1
    ).mean()

    print("=" * 72)
    print("AROMA Calibration-50 Summary")
    print("=" * 72)

    print("\nOverall:")
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nMean prompt stability:",
        round(
            df[
                "prompt_stability"
            ].mean(),
            4,
        ),
    )

    print(
        "Exact four-prompt agreement:",
        round(
            exact_agreement,
            4,
        ),
    )

    print(
        "Enumeration correct | Direct wrong:",
        (
            "NA"
            if np.isnan(
                enum_rescue
            )
            else round(
                enum_rescue,
                4,
            )
        ),
    )

    print(
        "Spatial enumeration correct | Direct wrong:",
        (
            "NA"
            if np.isnan(
                spatial_rescue
            )
            else round(
                spatial_rescue,
                4,
            )
        ),
    )

    print("\nSaved:")
    print(
        OUTPUT_DIR
        / "samples.csv"
    )
    print(
        OUTPUT_DIR
        / "summary.csv"
    )
    print(
        OUTPUT_DIR
        / "by_count.csv"
    )
    print(
        OUTPUT_DIR
        / "by_condition.csv"
    )


if __name__ == "__main__":
    main()
