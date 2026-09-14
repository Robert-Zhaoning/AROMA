#!/usr/bin/env python3

"""
HoloCount post-hoc baseline-correct action matrix.

Scientific purpose:
Complete the HoloCount action-utility labels required for:

1. fixed-action baselines;
2. full per-action utility matrix;
3. same-39-feature HoloCount utility learnability probe.

Already available:
- 1134 baseline-wrong GT<=15 examples × 4 actions;
- 188 GT>15 baseline-wrong examples have known utility 0.

Therefore only the 1158 baseline-correct examples require inference here.
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, "scripts")

from run_holocount_external_v1 import (
    DATA_ROOT,
    MODEL_REVISION,
    LAYER,
    HEAD,
    DUMMY_GT,
    load_model_stack,
    move_inputs,
    prepare_tallyqa_inputs,
    HeadGainModifier,
    score_state,
    sha256_file,
)


FULL_PATH = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/full_raw_results.csv"
)

EXPECTED_FULL_SHA = (
    "82929de79583d0c0641a80e13c8978d4ee4c00cebe7deb8f132697d3b6295466"
)

OUT_DIR = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/action_matrix_v1"
)

RESULT_PATH = (
    OUT_DIR / "baseline_correct_action_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR / "baseline_correct_action_summary.json"
)

META_PATH = (
    OUT_DIR / "run_metadata.json"
)

ACTIONS = [
    0.0,
    1.5,
    2.0,
    4.0,
]


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def evaluate_action(
    model,
    inputs,
    numeral_ids,
    alpha,
):

    modifier = HeadGainModifier(
        model=model,
        layer_idx=LAYER,
        head_idx=HEAD,
        alpha=alpha,
    )

    modifier.register()

    try:
        state = score_state(
            model,
            inputs,
            numeral_ids,
            DUMMY_GT,
        )

    finally:
        modifier.remove()

    if int(modifier.calls) <= 0:
        raise RuntimeError(
            f"Hook not called for alpha={alpha}"
        )

    return (
        int(state["best_numeral"]),
        int(modifier.calls),
    )


def load_completed():

    if not RESULT_PATH.exists():
        return set()

    df = pd.read_csv(
        RESULT_PATH
    )

    return set(
        df["sample_id"]
        .astype(str)
        .tolist()
    )


def main(resume=False):

    if sha256_file(
        FULL_PATH
    ) != EXPECTED_FULL_SHA:
        raise RuntimeError(
            "Frozen HoloCount raw-result SHA mismatch."
        )

    full = pd.read_csv(
        FULL_PATH
    )

    if len(full) != 2480:
        raise RuntimeError(
            f"Expected 2480 rows, got {len(full)}."
        )

    full["ground_truth"] = (
        full["ground_truth"]
        .astype(int)
    )

    full["baseline_prediction"] = (
        full["baseline_prediction"]
        .astype(int)
    )

    baseline_correct = (
        full["baseline_prediction"]
        ==
        full["ground_truth"]
    )

    target = full[
        baseline_correct
    ].copy()

    if len(target) != 1158:
        raise RuntimeError(
            f"Expected 1158 baseline-correct examples, "
            f"got {len(target)}."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        RESULT_PATH.exists()
        and
        not resume
    ):
        raise RuntimeError(
            f"{RESULT_PATH} already exists. "
            "Use --resume after interruption."
        )

    completed = (
        load_completed()
        if resume
        else set()
    )

    remaining = target[
        ~target[
            "sample_id"
        ]
        .astype(str)
        .isin(
            completed
        )
    ].copy()

    print("=" * 68)
    print(
        "HOLOCOUNT BASELINE-CORRECT ACTION MATRIX V1"
    )
    print("=" * 68)

    print(
        "Baseline-correct population :",
        len(target),
    )

    print(
        "Already completed           :",
        len(completed),
    )

    print(
        "Remaining                   :",
        len(remaining),
    )

    print(
        "Actions                     :",
        ACTIONS,
    )

    started = utc_now()

    (
        processor,
        model,
        numeral_ids,
    ) = load_model_stack()

    fieldnames = [
        "manifest_index",
        "sample_id",
        "split",
        "ground_truth",
        "baseline_prediction",

        "pred_alpha_0",
        "pred_alpha_1p5",
        "pred_alpha_2",
        "pred_alpha_4",

        "correct_alpha_0",
        "correct_alpha_1p5",
        "correct_alpha_2",
        "correct_alpha_4",

        "break_alpha_0",
        "break_alpha_1p5",
        "break_alpha_2",
        "break_alpha_4",

        "hook_calls_alpha_0",
        "hook_calls_alpha_1p5",
        "hook_calls_alpha_2",
        "hook_calls_alpha_4",
    ]

    mode = (
        "a"
        if RESULT_PATH.exists()
        and resume
        else
        "w"
    )

    with RESULT_PATH.open(
        mode,
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if mode == "w":
            writer.writeheader()

        for sample in tqdm(
            remaining.itertuples(
                index=False
            ),
            total=len(remaining),
            desc="HoloCount correct-action matrix",
        ):

            gt = int(
                sample.ground_truth
            )

            image_path = (
                DATA_ROOT
                /
                str(
                    sample.file_name
                )
            )

            image = (
                Image.open(
                    image_path
                )
                .convert(
                    "RGB"
                )
            )

            inputs = prepare_tallyqa_inputs(
                processor,
                image,
                str(
                    sample.question
                ),
            )

            inputs = move_inputs(
                inputs,
                model,
            )

            pred = {}
            correct = {}
            breaks = {}
            hooks = {}

            for alpha in ACTIONS:

                p, h = evaluate_action(
                    model,
                    inputs,
                    numeral_ids,
                    alpha,
                )

                pred[alpha] = p
                hooks[alpha] = h

                correct[alpha] = bool(
                    p == gt
                )

                breaks[alpha] = bool(
                    p != gt
                )

            row = {
                "manifest_index":
                    int(
                        sample.manifest_index
                    ),

                "sample_id":
                    str(
                        sample.sample_id
                    ),

                "split":
                    str(
                        sample.split
                    ),

                "ground_truth":
                    gt,

                "baseline_prediction":
                    int(
                        sample.baseline_prediction
                    ),

                "pred_alpha_0":
                    pred[0.0],

                "pred_alpha_1p5":
                    pred[1.5],

                "pred_alpha_2":
                    pred[2.0],

                "pred_alpha_4":
                    pred[4.0],

                "correct_alpha_0":
                    correct[0.0],

                "correct_alpha_1p5":
                    correct[1.5],

                "correct_alpha_2":
                    correct[2.0],

                "correct_alpha_4":
                    correct[4.0],

                "break_alpha_0":
                    breaks[0.0],

                "break_alpha_1p5":
                    breaks[1.5],

                "break_alpha_2":
                    breaks[2.0],

                "break_alpha_4":
                    breaks[4.0],

                "hook_calls_alpha_0":
                    hooks[0.0],

                "hook_calls_alpha_1p5":
                    hooks[1.5],

                "hook_calls_alpha_2":
                    hooks[2.0],

                "hook_calls_alpha_4":
                    hooks[4.0],
            }

            writer.writerow(
                row
            )

            f.flush()
            os.fsync(
                f.fileno()
            )

    result = pd.read_csv(
        RESULT_PATH
    )

    if len(result) != 1158:
        raise RuntimeError(
            f"Incomplete action matrix: "
            f"{len(result)}/1158."
        )

    repair_counts = {
        "0.0": 99,
        "1.5": 59,
        "2.0": 95,
        "4.0": 176,
    }

    summary = {}

    for alpha, suffix in [
        (0.0, "0"),
        (1.5, "1p5"),
        (2.0, "2"),
        (4.0, "4"),
    ]:

        breaks = (
            result[
                f"break_alpha_{suffix}"
            ]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        break_n = int(
            breaks.sum()
        )

        repairs = int(
            repair_counts[
                str(alpha)
            ]
        )

        post_correct = (
            1158
            -
            break_n
            +
            repairs
        )

        post_acc = (
            post_correct
            /
            2480
        )

        baseline_acc = (
            1158
            /
            2480
        )

        summary[
            str(alpha)
        ] = {
            "repairs":
                repairs,

            "breaks":
                break_n,

            "net_repairs":
                repairs
                -
                break_n,

            "post_correct":
                post_correct,

            "post_accuracy":
                post_acc,

            "gain_pp":
                100.0
                *
                (
                    post_acc
                    -
                    baseline_acc
                ),
        }

    payload = {
        "status":
            "POST_HOC_ACTION_MATRIX",

        "baseline_correct_population":
            1158,

        "actions":
            ACTIONS,

        "fixed_action_summary":
            summary,

        "result_sha256":
            sha256_file(
                RESULT_PATH
            ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    META_PATH.write_text(
        json.dumps({
            "started_utc":
                started,

            "completed_utc":
                utc_now(),

            "model_revision":
                MODEL_REVISION,

            "full_result_sha256":
                EXPECTED_FULL_SHA,

            "result_sha256":
                sha256_file(
                    RESULT_PATH
                ),
        }, indent=2, sort_keys=True)
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 68)
    print(
        "HOLOCOUNT BASELINE-CORRECT ACTION MATRIX COMPLETE"
    )
    print("=" * 68)

    for alpha in [
        "0.0",
        "1.5",
        "2.0",
        "4.0",
    ]:

        x = summary[
            alpha
        ]

        print(
            f"alpha={alpha:>3s}  "
            f"repairs={x['repairs']:3d}  "
            f"breaks={x['breaks']:3d}  "
            f"net={x['net_repairs']:+4d}  "
            f"acc={100.0*x['post_accuracy']:.4f}%  "
            f"gain={x['gain_pp']:+.4f} pp"
        )

    print()
    print(
        "Result SHA256:",
        sha256_file(
            RESULT_PATH
        ),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        resume=args.resume
    )
