#!/usr/bin/env python3

"""
Post-hoc HoloCount actuator-capacity oracle gate.

This is NOT a confirmatory evaluation.
The frozen cross-dataset controller evaluation has already failed.

Goal:
Determine whether L18H13 still has useful repair capacity on HoloCount.

Efficiency:
Only baseline-wrong examples with GT <= 15 are evaluated.

Why this is exact for oracle headroom:
- baseline-correct examples can retain alpha=1;
- GT > 15 cannot be correct under frozen 0..15 prediction support;
- therefore only baseline-wrong GT<=15 examples can improve oracle accuracy.
"""

import argparse
import csv
import json
import math
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


RAW_RESULT = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/full_raw_results.csv"
)

EXPECTED_RAW_SHA = (
    "82929de79583d0c0641a80e13c8978d4ee4c00cebe7deb8f132697d3b6295466"
)

OUT_DIR = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/oracle_gate_v1"
)

RESULT_PATH = (
    OUT_DIR / "wrong_gtle15_action_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR / "oracle_gate_summary.json"
)

META_PATH = (
    OUT_DIR / "oracle_gate_run_metadata.json"
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


def load_completed():
    if not RESULT_PATH.exists():
        return set()

    with RESULT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    return {
        r["sample_id"]
        for r in rows
    }


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
        int(
            state["best_numeral"]
        ),
        int(
            modifier.calls
        ),
    )


def main(resume=False):

    if sha256_file(
        RAW_RESULT
    ) != EXPECTED_RAW_SHA:
        raise RuntimeError(
            "Frozen HoloCount raw-result SHA mismatch."
        )

    full = pd.read_csv(
        RAW_RESULT
    )

    if len(full) != 2480:
        raise RuntimeError(
            f"Expected 2480 rows, got {len(full)}."
        )

    full[
        "ground_truth"
    ] = (
        full[
            "ground_truth"
        ]
        .astype(int)
    )

    full[
        "baseline_prediction"
    ] = (
        full[
            "baseline_prediction"
        ]
        .astype(int)
    )

    baseline_correct_mask = (
        full[
            "baseline_prediction"
        ]
        ==
        full[
            "ground_truth"
        ]
    )

    baseline_correct_n = int(
        baseline_correct_mask.sum()
    )

    baseline_wrong_n = int(
        len(full)
        -
        baseline_correct_n
    )

    target = full[
        (
            ~baseline_correct_mask
        )
        &
        (
            full[
                "ground_truth"
            ]
            <= 15
        )
    ].copy()

    unsupported_wrong_n = int(
        (
            (
                ~baseline_correct_mask
            )
            &
            (
                full[
                    "ground_truth"
                ]
                > 15
            )
        ).sum()
    )

    print(
        "============================================================"
    )
    print(
        "HOLOCOUNT POST-HOC ACTION ORACLE GATE V1"
    )
    print(
        "============================================================"
    )
    print(
        "Full N                     :",
        len(full),
    )
    print(
        "Baseline correct           :",
        baseline_correct_n,
    )
    print(
        "Baseline wrong             :",
        baseline_wrong_n,
    )
    print(
        "Wrong with GT<=15          :",
        len(target),
    )
    print(
        "Wrong with GT>15           :",
        unsupported_wrong_n,
    )

    if baseline_correct_n != 1158:
        raise RuntimeError(
            "Unexpected baseline-correct count."
        )

    if len(target) != 1134:
        raise RuntimeError(
            f"Expected 1134 oracle-gate examples, "
            f"got {len(target)}."
        )

    if unsupported_wrong_n != 188:
        raise RuntimeError(
            "Unexpected GT>15 wrong count."
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
            "Use --resume after an interrupted run."
        )

    completed = (
        load_completed()
        if resume
        else set()
    )

    remaining = target[
        ~target[
            "sample_id"
        ].astype(str)
        .isin(
            completed
        )
    ].copy()

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
    print(
        "Model revision              :",
        MODEL_REVISION,
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

        "repair_alpha_0",
        "repair_alpha_1p5",
        "repair_alpha_2",
        "repair_alpha_4",

        "hook_calls_alpha_0",
        "hook_calls_alpha_1p5",
        "hook_calls_alpha_2",
        "hook_calls_alpha_4",

        "oracle_repairable",
        "repair_actions",
        "first_repair_action",
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
            desc="HoloCount oracle gate",
        ):

            sample_id = str(
                sample.sample_id
            )

            gt = int(
                sample.ground_truth
            )

            baseline_prediction = int(
                sample.baseline_prediction
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

            inputs = (
                prepare_tallyqa_inputs(
                    processor,
                    image,
                    str(
                        sample.question
                    ),
                )
            )

            inputs = move_inputs(
                inputs,
                model,
            )

            predictions = {}
            hook_calls = {}
            repair = {}

            for alpha in ACTIONS:

                pred, calls = (
                    evaluate_action(
                        model,
                        inputs,
                        numeral_ids,
                        alpha,
                    )
                )

                predictions[
                    alpha
                ] = pred

                hook_calls[
                    alpha
                ] = calls

                repair[
                    alpha
                ] = bool(
                    pred == gt
                )

            repair_actions = [
                alpha
                for alpha in ACTIONS
                if repair[
                    alpha
                ]
            ]

            oracle_repairable = bool(
                repair_actions
            )

            row = {
                "manifest_index":
                    int(
                        sample.manifest_index
                    ),

                "sample_id":
                    sample_id,

                "split":
                    str(
                        sample.split
                    ),

                "ground_truth":
                    gt,

                "baseline_prediction":
                    baseline_prediction,

                "pred_alpha_0":
                    predictions[0.0],

                "pred_alpha_1p5":
                    predictions[1.5],

                "pred_alpha_2":
                    predictions[2.0],

                "pred_alpha_4":
                    predictions[4.0],

                "repair_alpha_0":
                    repair[0.0],

                "repair_alpha_1p5":
                    repair[1.5],

                "repair_alpha_2":
                    repair[2.0],

                "repair_alpha_4":
                    repair[4.0],

                "hook_calls_alpha_0":
                    hook_calls[0.0],

                "hook_calls_alpha_1p5":
                    hook_calls[1.5],

                "hook_calls_alpha_2":
                    hook_calls[2.0],

                "hook_calls_alpha_4":
                    hook_calls[4.0],

                "oracle_repairable":
                    oracle_repairable,

                "repair_actions":
                    ";".join(
                        str(x)
                        for x
                        in repair_actions
                    ),

                "first_repair_action":
                    (
                        repair_actions[0]
                        if repair_actions
                        else ""
                    ),
            }

            writer.writerow(
                row
            )

            f.flush()

    # --------------------------------------------------------
    # Exact oracle summary.
    # --------------------------------------------------------

    oracle_df = pd.read_csv(
        RESULT_PATH
    )

    if len(
        oracle_df
    ) != 1134:
        raise RuntimeError(
            f"Oracle gate incomplete: "
            f"{len(oracle_df)}/1134."
        )

    oracle_repairable = (
        oracle_df[
            "oracle_repairable"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "true"
        )
    )

    repairable_n = int(
        oracle_repairable.sum()
    )

    oracle_correct = (
        baseline_correct_n
        +
        repairable_n
    )

    baseline_acc = (
        baseline_correct_n
        /
        2480
    )

    oracle_acc = (
        oracle_correct
        /
        2480
    )

    gain_pp = (
        100.0
        *
        (
            oracle_acc
            -
            baseline_acc
        )
    )

    action_repairs = {}

    for col, name in [
        (
            "repair_alpha_0",
            "0.0",
        ),
        (
            "repair_alpha_1p5",
            "1.5",
        ),
        (
            "repair_alpha_2",
            "2.0",
        ),
        (
            "repair_alpha_4",
            "4.0",
        ),
    ]:

        action_repairs[
            name
        ] = int(
            oracle_df[
                col
            ]
            .astype(str)
            .str.lower()
            .eq(
                "true"
            )
            .sum()
        )

    summary = {
        "status":
            "POST_HOC_DIAGNOSTIC",

        "full_n":
            2480,

        "baseline_correct":
            baseline_correct_n,

        "baseline_accuracy":
            baseline_acc,

        "baseline_wrong":
            baseline_wrong_n,

        "oracle_gate_population":
            1134,

        "unsupported_gt_gt_15":
            188,

        "repairable_wrong":
            repairable_n,

        "repairable_fraction_all_wrong":
            repairable_n
            /
            baseline_wrong_n,

        "repairable_fraction_supported_wrong":
            repairable_n
            /
            1134,

        "oracle_correct":
            oracle_correct,

        "oracle_accuracy":
            oracle_acc,

        "oracle_gain_pp":
            gain_pp,

        "action_repair_counts_on_supported_wrong":
            action_repairs,

        "adaptation_gate_rule":
            "continue only if oracle_gain_pp > 1.0",

        "adaptation_gate_pass":
            bool(
                gain_pp
                >
                1.0
            ),

        "result_sha256":
            sha256_file(
                RESULT_PATH
            ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    metadata = {
        "started_utc":
            started,

        "completed_utc":
            utc_now(),

        "model_revision":
            MODEL_REVISION,

        "raw_primary_result_sha256":
            EXPECTED_RAW_SHA,

        "oracle_result_sha256":
            sha256_file(
                RESULT_PATH
            ),

        "actions":
            ACTIONS,

        "oracle_gate_population":
            1134,
    }

    META_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "============================================================"
    )
    print(
        "HOLOCOUNT ACTION ORACLE GATE COMPLETE"
    )
    print(
        "============================================================"
    )
    print(
        f"Baseline accuracy          : "
        f"{100.0 * baseline_acc:.4f}%"
    )
    print(
        f"Repairable wrong           : "
        f"{repairable_n} / {baseline_wrong_n}"
    )
    print(
        f"Repairable supported wrong : "
        f"{repairable_n} / 1134"
    )
    print(
        f"Oracle accuracy            : "
        f"{100.0 * oracle_acc:.4f}%"
    )
    print(
        f"Oracle headroom            : "
        f"{gain_pp:+.4f} pp"
    )
    print(
        "Action repair counts       :",
        action_repairs,
    )
    print(
        "Adaptation gate (>1 pp)    :",
        summary[
            "adaptation_gate_pass"
        ],
    )
    print(
        "Result SHA256              :",
        summary[
            "result_sha256"
        ],
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
