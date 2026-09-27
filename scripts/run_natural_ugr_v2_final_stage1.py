#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


# ============================================================
# EXACT EXISTING AROMA IMPLEMENTATIONS
# ============================================================

sys.path.insert(
    0,
    "scripts",
)

from run_l18h13_gain_all300 import (
    MODEL_ID,
    move_inputs,
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)

from run_proc_count_causal_v3_final import (
    state_to_features,
    controller_decision,
)

from run_tallyqa_natural_confirmation_v2_final import (
    prepare_tallyqa_inputs,
)


# ============================================================
# FROZEN PATHS
# ============================================================

ROOT = Path(
    "/workspace/AromaExperiments"
)

MANIFEST = (
    ROOT
    / "artifacts/natural_ugr_v1/final/cohort_v1/"
      "final_manifest_v1.jsonl"
)

COHORT_FREEZE = (
    ROOT
    / "artifacts/natural_ugr_v1/final/cohort_v1/"
      "final_cohort_freeze_v1.json"
)

IMAGE_ROOT = (
    ROOT
    / "data/natural_ugr_final_v1/images"
)

CONTROLLER = (
    ROOT
    / "outputs/phase2_natural_controller_k500/"
      "final_frozen_controller/"
      "aroma_natural_utility_controller_k500.joblib"
)

EXECUTION_FREEZE = (
    ROOT
    / "artifacts/natural_ugr_v2/final/stage1/"
      "stage1_execution_freeze_v1.json"
)

OUT = (
    ROOT
    / "artifacts/natural_ugr_v2/final/stage1/results"
)

PARTIAL = (
    OUT
    / "partial_stage1_v1.jsonl"
)

FINAL_CSV = (
    OUT
    / "stage1_action_manifest_v1.csv"
)

UPWARD_CSV = (
    OUT
    / "upward_population_v1.csv"
)

ACTION_COUNTS = (
    OUT
    / "stage1_action_distribution_v1.csv"
)

METADATA = (
    OUT
    / "stage1_metadata_v1.json"
)

RUN_STARTED = (
    OUT
    / "stage1_run_started_v1.json"
)


# ============================================================
# FROZEN CONSTANTS
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

EXPECTED_MANIFEST_SHA256 = (
    "cb9bd03edb1788494d27c2a8f29e1a8"
    "a06f19138ce03c43bf867d09233727c83"
)

EXPECTED_CONTROLLER_SHA256 = (
    "d9d8ba3a24814bf71b3f4039d1effa8"
    "64b57754a00539f1260a1d077c03449f5"
)

EXPECTED_N = 8000

EXPECTED_ACTIONS = {
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
}

DUMMY_GT = 1


# ============================================================
# HELPERS
# ============================================================

def sha256(path):

    path = Path(path)

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


def load_jsonl(path):

    path = Path(path)

    if not path.exists():

        return []

    return [
        json.loads(line)

        for line
        in path.read_text(
            encoding="utf-8"
        ).splitlines()

        if line.strip()
    ]


def append_jsonl(
    path,
    obj,
):

    path = Path(path)

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            +
            "\n"
        )

        f.flush()


def git_head():

    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        text=True,
    ).strip()


def git_is_ancestor(
    ancestor,
):

    r = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            "HEAD",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return (
        r.returncode
        ==
        0
    )


# ============================================================
# PRE-INFERENCE PROTOCOL AUDIT
# ============================================================

def protocol_audit():

    # --------------------------------------------------------
    # Final cohort identity
    # --------------------------------------------------------

    actual_manifest_sha = sha256(
        MANIFEST
    )

    if (
        actual_manifest_sha
        !=
        EXPECTED_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Final manifest SHA256 mismatch:\n"
            f"expected={EXPECTED_MANIFEST_SHA256}\n"
            f"actual={actual_manifest_sha}"
        )


    records = load_jsonl(
        MANIFEST
    )

    if (
        len(records)
        !=
        EXPECTED_N
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_N} final samples; "
            f"got {len(records)}"
        )


    indices = sorted(
        int(
            r[
                "manifest_index"
            ]
        )

        for r
        in records
    )

    if (
        indices
        !=
        list(
            range(
                EXPECTED_N
            )
        )
    ):

        raise RuntimeError(
            "manifest_index is not exactly 0..7999"
        )


    question_ids = [
        str(
            r[
                "question_id"
            ]
        )

        for r
        in records
    ]

    if (
        len(
            set(
                question_ids
            )
        )
        !=
        EXPECTED_N
    ):

        raise RuntimeError(
            "Final question_id values are not unique"
        )


    image_ids = [
        str(
            r[
                "image_id"
            ]
        )

        for r
        in records
    ]

    if (
        len(
            set(
                image_ids
            )
        )
        !=
        EXPECTED_N
    ):

        raise RuntimeError(
            "Final image_id values are not unique"
        )


    # --------------------------------------------------------
    # Cohort freeze
    # --------------------------------------------------------

    freeze = json.loads(
        COHORT_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    if (
        freeze[
            "status"
        ]
        !=
        "FROZEN BEFORE ANY FINAL MODEL INFERENCE"
    ):

        raise RuntimeError(
            "Unexpected final cohort freeze status"
        )


    frozen_manifest_sha = (
        freeze[
            "files"
        ][
            "manifest"
        ][
            "sha256"
        ]
    )

    if (
        frozen_manifest_sha
        !=
        EXPECTED_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Final cohort freeze does not point "
            "to the expected manifest"
        )


    # --------------------------------------------------------
    # Frozen K500 controller
    # --------------------------------------------------------

    actual_controller_sha = sha256(
        CONTROLLER
    )

    if (
        actual_controller_sha
        !=
        EXPECTED_CONTROLLER_SHA256
    ):

        raise RuntimeError(
            "Natural K500 controller SHA256 mismatch:\n"
            f"expected={EXPECTED_CONTROLLER_SHA256}\n"
            f"actual={actual_controller_sha}"
        )


    bundle = joblib.load(
        CONTROLLER
    )

    feature_names = list(
        bundle[
            "feature_names"
        ]
    )

    if (
        len(
            feature_names
        )
        !=
        39
    ):

        raise RuntimeError(
            f"Expected 39 controller features; "
            f"got {len(feature_names)}"
        )


    bundle_actions = {
        float(x)

        for x
        in bundle[
            "actions"
        ]
    }

    if (
        bundle_actions
        !=
        EXPECTED_ACTIONS
    ):

        raise RuntimeError(
            "Frozen Proposal Controller action "
            f"set mismatch: {bundle_actions}"
        )


    # --------------------------------------------------------
    # Stage-1 execution freeze
    # --------------------------------------------------------

    if not EXECUTION_FREEZE.exists():

        raise RuntimeError(
            "Stage-1 execution freeze missing"
        )


    execution = json.loads(
        EXECUTION_FREEZE.read_text(
            encoding="utf-8"
        )
    )


    expected_status = (
        "FROZEN BEFORE NATURAL UGR V2 "
        "FINAL STAGE-1 INFERENCE"
    )

    if (
        execution[
            "status"
        ]
        !=
        expected_status
    ):

        raise RuntimeError(
            "Unexpected Stage-1 execution freeze status"
        )


    frozen_runner_sha = (
        execution[
            "runner"
        ][
            "sha256"
        ]
    )

    current_runner_sha = sha256(
        Path(
            __file__
        ).resolve()
    )

    if (
        frozen_runner_sha
        !=
        current_runner_sha
    ):

        raise RuntimeError(
            "Stage-1 runner changed after freeze:\n"
            f"frozen={frozen_runner_sha}\n"
            f"current={current_runner_sha}"
        )


    source_head = execution[
        "source_git_head"
    ]

    if not git_is_ancestor(
        source_head
    ):

        raise RuntimeError(
            "Frozen source commit is not an ancestor "
            "of current HEAD"
        )


    # --------------------------------------------------------
    # Image availability
    # --------------------------------------------------------

    missing = []

    for r in records:

        image_path = (
            IMAGE_ROOT
            /
            str(
                r[
                    "image"
                ]
            )
        )

        if not image_path.exists():

            missing.append(
                str(
                    image_path
                )
            )

            if (
                len(
                    missing
                )
                >=
                10
            ):

                break


    if missing:

        raise RuntimeError(
            "Missing final image files:\n"
            +
            "\n".join(
                missing
            )
        )


    print(
        "PASS: frozen final manifest"
    )

    print(
        "PASS: final cohort N=8000"
    )

    print(
        "PASS: all final image paths available"
    )

    print(
        "PASS: frozen K500 controller"
    )

    print(
        "PASS: Stage-1 execution freeze"
    )

    print(
        "Manifest SHA256:",
        actual_manifest_sha,
    )

    print(
        "Controller SHA256:",
        actual_controller_sha,
    )

    print(
        "Runner SHA256:",
        current_runner_sha,
    )


    return (
        records,
        bundle,
        feature_names,
    )


# ============================================================
# RESUME SUPPORT
# ============================================================

def completed_qids():

    if not PARTIAL.exists():

        return set()

    rows = load_jsonl(
        PARTIAL
    )

    qids = [
        str(
            r[
                "question_id"
            ]
        )

        for r
        in rows
    ]

    if (
        len(
            qids
        )
        !=
        len(
            set(
                qids
            )
        )
    ):

        raise RuntimeError(
            "Duplicate question_id found "
            "inside partial Stage-1 output"
        )

    return set(
        qids
    )


# ============================================================
# MAIN
# ============================================================

def main(
    protocol_only=False,
    resume=False,
):

    (
        records,
        bundle,
        feature_names,
    ) = protocol_audit()


    if protocol_only:

        print()

        print(
            "NATURAL UGR V2 FINAL STAGE-1 "
            "PROTOCOL-ONLY: PASS"
        )

        print(
            "No model loaded."
        )

        print(
            "No final prediction produced."
        )

        return


    # --------------------------------------------------------
    # Runtime GPU checks
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available"
        )


    if not torch.cuda.is_bf16_supported():

        raise RuntimeError(
            "GPU does not report BF16 support"
        )


    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )


    if FINAL_CSV.exists():

        raise RuntimeError(
            "Final Stage-1 action manifest already exists. "
            "Refusing to rerun completed final Stage-1."
        )


    # --------------------------------------------------------
    # Start / resume state
    # --------------------------------------------------------

    if not resume:

        if RUN_STARTED.exists():

            raise RuntimeError(
                "Stage-1 run-start marker already exists. "
                "Use --resume only after a technical interruption."
            )


        if PARTIAL.exists():

            raise RuntimeError(
                "Stage-1 partial output already exists. "
                "Use --resume only after a technical interruption."
            )


        started = {
            "artifact":
                "Natural UGR v2 final Stage-1 formal run",

            "started_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "git_head":
                git_head(),

            "manifest_sha256":
                sha256(
                    MANIFEST
                ),

            "controller_sha256":
                sha256(
                    CONTROLLER
                ),

            "runner_sha256":
                sha256(
                    Path(
                        __file__
                    ).resolve()
                ),

            "model_id":
                MODEL_ID,

            "model_revision":
                MODEL_REVISION,

            "torch":
                torch.__version__,

            "torch_cuda":
                torch.version.cuda,

            "gpu":
                torch.cuda.get_device_name(
                    0
                ),

            "gpu_vram_bytes":
                int(
                    torch.cuda.get_device_properties(
                        0
                    ).total_memory
                ),

            "bf16_supported":
                bool(
                    torch.cuda.is_bf16_supported()
                ),

            "retuning_after_start":
                False,
        }


        RUN_STARTED.write_text(
            json.dumps(
                started,
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )


        completed = set()


    else:

        if not RUN_STARTED.exists():

            raise RuntimeError(
                "--resume requested but no "
                "Stage-1 run-start marker exists"
            )

        completed = completed_qids()


    ordered = sorted(
        records,
        key=lambda r:
            int(
                r[
                    "manifest_index"
                ]
            ),
    )


    remaining = [
        r

        for r
        in ordered

        if str(
            r[
                "question_id"
            ]
        )
        not in
        completed
    ]


    print()

    print(
        "Already completed:",
        len(
            completed
        ),
    )

    print(
        "Remaining:",
        len(
            remaining
        ),
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )


    # ========================================================
    # MODEL LOAD
    # ========================================================

    print()

    print(
        "Loading processor..."
    )


    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            revision=
                MODEL_REVISION,
        )
    )


    numeral_ids = numeral_token_ids(
        processor
    )


    if sorted(
        numeral_ids.keys()
    ) != list(
        range(
            16
        )
    ):

        raise RuntimeError(
            "Expanded numeral-token audit failed"
        )


    print(
        "Numeral-token audit: PASS"
    )


    print(
        "Loading model..."
    )


    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            revision=
                MODEL_REVISION,
            torch_dtype=
                torch.bfloat16,
            device_map=
                "auto",
            attn_implementation=
                "eager",
        )
    )


    model.eval()


    print(
        "Model loaded."
    )


    # ========================================================
    # FORMAL 8000-SAMPLE STAGE-1 LOOP
    # ========================================================

    for sample in tqdm(
        remaining,
        desc=(
            "Natural UGR v2 final Stage-1"
        ),
    ):

        qid = str(
            sample[
                "question_id"
            ]
        )


        image_path = (
            IMAGE_ROOT
            /
            str(
                sample[
                    "image"
                ]
            )
        )


        with Image.open(
            image_path
        ) as im:

            image = (
                im.convert(
                    "RGB"
                )
            )


            inputs = (
                prepare_tallyqa_inputs(
                    processor,
                    image,
                    str(
                        sample[
                            "question"
                        ]
                    ),
                )
            )


        inputs = move_inputs(
            inputs,
            model,
        )


        # ====================================================
        # IMPORTANT:
        #
        # sample["answer"] is intentionally NOT read before
        # Proposal Controller action selection.
        # ====================================================

        with torch.inference_mode():

            baseline_state = (
                score_state(
                    model,
                    inputs,
                    numeral_ids,
                    DUMMY_GT,
                )
            )


        (
            X,
            _,
        ) = (
            state_to_features(
                baseline_state,
                feature_names,
            )
        )


        (
            selected_alpha,
            selected_score,
            score_map,
        ) = (
            controller_decision(
                bundle,
                X,
            )
        )


        selected_alpha = float(
            selected_alpha
        )

        selected_score = float(
            selected_score
        )


        if (
            selected_alpha
            not in
            EXPECTED_ACTIONS
        ):

            raise RuntimeError(
                f"Unexpected selected_alpha="
                f"{selected_alpha} "
                f"for question_id={qid}"
            )


        baseline_prediction = int(
            baseline_state[
                "best_numeral"
            ]
        )


        x_values = (
            np.asarray(
                X,
                dtype=np.float64,
            )
            .reshape(
                -1
            )
        )


        if (
            len(
                x_values
            )
            !=
            39
        ):

            raise RuntimeError(
                f"Expected 39 Stage-1 features "
                f"for question_id={qid}; "
                f"got {len(x_values)}"
            )


        # ====================================================
        # Ground truth becomes readable only AFTER
        # action selection has finished.
        # ====================================================

        ground_truth = int(
            sample[
                "answer"
            ]
        )


        row = {
            "manifest_index":
                int(
                    sample[
                        "manifest_index"
                    ]
                ),

            "question_id":
                sample[
                    "question_id"
                ],

            "image":
                str(
                    sample[
                        "image"
                    ]
                ),

            "image_id":
                sample[
                    "image_id"
                ],

            "subset":
                str(
                    sample[
                        "subset"
                    ]
                ),

            "issimple":
                bool(
                    sample[
                        "issimple"
                    ]
                ),

            "data_source":
                str(
                    sample[
                        "data_source"
                    ]
                ),

            "question":
                str(
                    sample[
                        "question"
                    ]
                ),

            "ground_truth":
                ground_truth,

            "baseline_prediction":
                baseline_prediction,

            "baseline_correct":
                bool(
                    baseline_prediction
                    ==
                    ground_truth
                ),

            "selected_alpha":
                selected_alpha,

            "selected_score":
                selected_score,

            "score_map_json":
                json.dumps(
                    {
                        str(k):
                            float(v)

                        for k, v
                        in score_map.items()
                    },
                    sort_keys=True,
                ),
        }


        for (
            feature_name,
            feature_value,
        ) in zip(
            feature_names,
            x_values,
        ):

            row[
                "feature__"
                +
                str(
                    feature_name
                )
            ] = float(
                feature_value
            )


        append_jsonl(
            PARTIAL,
            row,
        )


    # ========================================================
    # FINALIZE STAGE-1
    # ========================================================

    rows = load_jsonl(
        PARTIAL
    )


    if (
        len(
            rows
        )
        !=
        EXPECTED_N
    ):

        raise RuntimeError(
            f"Expected 8000 completed Stage-1 rows; "
            f"got {len(rows)}"
        )


    qids = [
        str(
            r[
                "question_id"
            ]
        )

        for r
        in rows
    ]


    if (
        len(
            qids
        )
        !=
        len(
            set(
                qids
            )
        )
    ):

        raise RuntimeError(
            "Duplicate Stage-1 question IDs"
        )


    df = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "manifest_index"
        )
        .reset_index(
            drop=True
        )
    )


    df.to_csv(
        FINAL_CSV,
        index=False,
        lineterminator="\n",
    )


    upward = (
        df[
            df[
                "selected_alpha"
            ].astype(
                float
            )
            >
            1.0
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    upward.to_csv(
        UPWARD_CSV,
        index=False,
        lineterminator="\n",
    )


    counts = (
        df[
            "selected_alpha"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "selected_alpha"
        )
        .reset_index(
            name=
                "count"
        )
    )


    counts[
        "fraction"
    ] = (
        counts[
            "count"
        ]
        /
        len(
            df
        )
    )


    counts.to_csv(
        ACTION_COUNTS,
        index=False,
        lineterminator="\n",
    )


    metadata = {
        "artifact":
            "Natural UGR v2 final Stage-1 result",

        "status":
            "COMPLETE",

        "n":
            int(
                len(
                    df
                )
            ),

        "baseline_correct":
            int(
                df[
                    "baseline_correct"
                ].sum()
            ),

        "baseline_accuracy":
            float(
                df[
                    "baseline_correct"
                ].mean()
            ),

        "upward_definition":
            "selected_alpha > 1",

        "upward_n":
            int(
                len(
                    upward
                )
            ),

        "manifest_sha256":
            sha256(
                MANIFEST
            ),

        "controller_sha256":
            sha256(
                CONTROLLER
            ),

        "runner_sha256":
            sha256(
                Path(
                    __file__
                ).resolve()
            ),

        "model_id":
            MODEL_ID,

        "model_revision":
            MODEL_REVISION,

        "stage1_action_manifest_sha256":
            sha256(
                FINAL_CSV
            ),

        "upward_population_sha256":
            sha256(
                UPWARD_CSV
            ),

        "action_distribution_sha256":
            sha256(
                ACTION_COUNTS
            ),

        "no_stage2_router_executed":
            True,

        "no_candidate_sweep_executed":
            True,

        "development_retuned":
            False,
    }


    METADATA.write_text(
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
        "=" * 78
    )

    print(
        "NATURAL UGR V2 FINAL STAGE-1 COMPLETE"
    )

    print(
        "=" * 78
    )


    print(
        "N:",
        len(
            df
        ),
    )


    print(
        "Baseline:",
        f"{int(df['baseline_correct'].sum())}"
        f"/{len(df)}"
        f" = "
        f"{100 * df['baseline_correct'].mean():.3f}%"
    )


    print()

    print(
        "Action distribution:"
    )

    print(
        counts.to_string(
            index=False
        )
    )


    print()

    print(
        "Stage-2 upward N:",
        len(
            upward
        ),
    )


    print(
        "Stage-2 population rule: "
        "selected_alpha > 1"
    )


    print(
        "Candidate sweep executed: False"
    )


    print(
        "Stage-2 router executed: False"
    )


    print(
        "Development retuned: False"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--protocol-only",
        action="store_true",
    )


    parser.add_argument(
        "--resume",
        action="store_true",
    )


    args = parser.parse_args()


    main(
        protocol_only=
            args.protocol_only,

        resume=
            args.resume,
    )
