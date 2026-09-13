import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("outputs/calibration50/results.jsonl")
OUTPUT_DIR = Path("outputs/calibration50")

PROMPTS = [
    "direct",
    "count_json",
    "enumeration",
    "spatial_enumeration",
]


def is_valid_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def strict_format_compliance(prompt_name, raw_text):
    text = raw_text.strip()

    if prompt_name == "direct":
        try:
            int(text)
            return True
        except Exception:
            return False

    try:
        parsed = json.loads(text)
        return isinstance(parsed, dict)
    except Exception:
        return False


def prompt_metrics(predictions):
    total = len(predictions)

    valid = [
        x
        for x in predictions
        if is_valid_int(x)
    ]

    coverage = len(valid) / total if total else 0.0

    if not valid:
        return {
            "agreement": 0.0,
            "coverage": coverage,
            "stability": 0.0,
            "modal_prediction": None,
        }

    counts = Counter(valid)

    modal_prediction, modal_frequency = counts.most_common(1)[0]

    agreement = modal_frequency / len(valid)
    stability = modal_frequency / total

    return {
        "agreement": agreement,
        "coverage": coverage,
        "stability": stability,
        "modal_prediction": modal_prediction,
    }


def safe_abs_error(pred, gt):
    if not is_valid_int(pred):
        return np.nan

    return abs(pred - gt)


def main():
    raw_records = []

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                raw_records.append(json.loads(line))

    assert len(raw_records) == 50, (
        f"Expected 50 results, found {len(raw_records)}"
    )

    rows = []

    for r in raw_records:
        gt = r["ground_truth"]

        predictions = [
            r["outputs"][name].get("parsed_count")
            for name in PROMPTS
        ]

        pm = prompt_metrics(predictions)

        row = {
            "sample_id": r["sample_id"],
            "ground_truth": gt,
            "condition": r["condition"],
            "prompt_agreement": pm["agreement"],
            "prompt_coverage": pm["coverage"],
            "prompt_stability": pm["stability"],
            "modal_prediction": pm["modal_prediction"],
            "modal_correct": pm["modal_prediction"] == gt,
        }

        for name in PROMPTS:
            out = r["outputs"][name]

            pred = out.get("parsed_count")

            row[f"{name}_pred"] = pred
            row[f"{name}_correct"] = pred == gt
            row[f"{name}_abs_error"] = safe_abs_error(pred, gt)
            row[f"{name}_parse_success"] = is_valid_int(pred)

            row[f"{name}_strict_format"] = (
                strict_format_compliance(
                    name,
                    out.get("raw", ""),
                )
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_DIR / "samples_v2.csv",
        index=False,
    )

    print("=" * 78)
    print("AROMA Calibration-50 — Formal Diagnostic Analysis")
    print("=" * 78)

    print("\nOVERALL PERFORMANCE")

    summary = []

    for name in PROMPTS:
        summary.append(
            {
                "prompt": name,
                "accuracy": df[f"{name}_correct"].mean(),
                "mae": df[f"{name}_abs_error"].mean(),
                "parse_rate": df[f"{name}_parse_success"].mean(),
                "strict_format_rate": df[f"{name}_strict_format"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    summary_df.to_csv(
        OUTPUT_DIR / "summary_v2.csv",
        index=False,
    )

    print("\nPROMPT RELIABILITY")

    print(
        "Mean valid-prompt agreement :",
        f"{df['prompt_agreement'].mean():.3f}",
    )

    print(
        "Mean prompt coverage        :",
        f"{df['prompt_coverage'].mean():.3f}",
    )

    print(
        "Mean prompt stability       :",
        f"{df['prompt_stability'].mean():.3f}",
    )

    fully_stable = (
        df["prompt_stability"] == 1.0
    )

    fully_stable_wrong = (
        fully_stable
        & (~df["modal_correct"])
    )

    print(
        "Fully stable samples        :",
        f"{fully_stable.mean():.3f}",
    )

    print(
        "Stable-but-wrong samples    :",
        f"{fully_stable_wrong.mean():.3f}",
    )

    print("\nDIRECT VS ENUMERATION")

    direct_wrong = ~df["direct_correct"]

    if direct_wrong.sum() > 0:
        enum_rescue = df.loc[
            direct_wrong,
            "enumeration_correct",
        ].mean()

        spatial_rescue = df.loc[
            direct_wrong,
            "spatial_enumeration_correct",
        ].mean()

        print(
            "P(enum correct | direct wrong)         :",
            f"{enum_rescue:.3f}",
        )

        print(
            "P(spatial correct | direct wrong)      :",
            f"{spatial_rescue:.3f}",
        )

    disagreement = (
        df["direct_pred"]
        != df["enumeration_pred"]
    ).mean()

    print(
        "Direct/enumeration disagreement         :",
        f"{disagreement:.3f}",
    )

    print("\nBY GROUND-TRUTH COUNT")

    by_count = []

    for gt, group in df.groupby("ground_truth"):
        row = {
            "ground_truth": gt,
            "n": len(group),
            "direct_acc": group["direct_correct"].mean(),
            "enum_acc": group["enumeration_correct"].mean(),
            "spatial_acc": group["spatial_enumeration_correct"].mean(),
            "mean_stability": group["prompt_stability"].mean(),
        }

        by_count.append(row)

    by_count_df = pd.DataFrame(by_count)

    print(
        by_count_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    by_count_df.to_csv(
        OUTPUT_DIR / "by_count_v2.csv",
        index=False,
    )

    print("\nBY CONDITION")

    by_condition = []

    for condition, group in df.groupby("condition"):
        row = {
            "condition": condition,
            "n": len(group),
            "direct_acc": group["direct_correct"].mean(),
            "enum_acc": group["enumeration_correct"].mean(),
            "spatial_acc": group["spatial_enumeration_correct"].mean(),
            "mean_stability": group["prompt_stability"].mean(),
        }

        by_condition.append(row)

    by_condition_df = pd.DataFrame(by_condition)

    print(
        by_condition_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    by_condition_df.to_csv(
        OUTPUT_DIR / "by_condition_v2.csv",
        index=False,
    )

    print("\nCANDIDATE FAILURE GROUPS")

    stable_wrong = df[
        (df["prompt_stability"] == 1.0)
        & (~df["modal_correct"])
    ]

    enum_rescue_samples = df[
        (~df["direct_correct"])
        & (df["enumeration_correct"])
    ]

    unstable = df[
        df["prompt_stability"] < 0.75
    ]

    print(
        "Stable shared-error candidates :",
        len(stable_wrong),
    )

    print(
        "Enumeration-rescue candidates  :",
        len(enum_rescue_samples),
    )

    print(
        "Strong instability candidates  :",
        len(unstable),
    )

    stable_wrong[
        [
            "sample_id",
            "ground_truth",
            "condition",
            "direct_pred",
            "enumeration_pred",
            "prompt_stability",
        ]
    ].to_csv(
        OUTPUT_DIR / "candidate_stable_wrong.csv",
        index=False,
    )

    enum_rescue_samples[
        [
            "sample_id",
            "ground_truth",
            "condition",
            "direct_pred",
            "enumeration_pred",
            "prompt_stability",
        ]
    ].to_csv(
        OUTPUT_DIR / "candidate_enum_rescue.csv",
        index=False,
    )

    unstable[
        [
            "sample_id",
            "ground_truth",
            "condition",
            "direct_pred",
            "count_json_pred",
            "enumeration_pred",
            "spatial_enumeration_pred",
            "prompt_stability",
            "prompt_coverage",
        ]
    ].to_csv(
        OUTPUT_DIR / "candidate_unstable.csv",
        index=False,
    )

    print("\nSaved:")
    print("  samples_v2.csv")
    print("  summary_v2.csv")
    print("  by_count_v2.csv")
    print("  by_condition_v2.csv")
    print("  candidate_stable_wrong.csv")
    print("  candidate_enum_rescue.csv")
    print("  candidate_unstable.csv")


if __name__ == "__main__":
    main()
