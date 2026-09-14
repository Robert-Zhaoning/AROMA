#!/usr/bin/env python3

"""
AROMA HoloCount external transfer v1.

Scientific role
---------------
This runner evaluates the already-frozen TallyQA-adapted AROMA controller
on the independently frozen HoloCount benchmark.

The computational stack is intentionally reused from the final AROMA
natural-confirmation implementation:

- exact Llama model family;
- exact L18H13 actuator;
- exact 0..15 next-token numeral proxy;
- exact 39-dimensional GT-free feature construction;
- exact frozen K=500 controller decision rule.

Modes
-----
--audit-only
    Perform provenance / contract / dataset integrity checks only.
    Does NOT load the processor or model.

--sanity
    Run the pre-frozen GT-free 10-example engineering sanity manifest.
    Does NOT read or report ground truth or accuracy.

--full
    Run all 2480 frozen HoloCount question records.

Ground-truth discipline
-----------------------
For --full, answer_int is not accessed until AFTER:

1. baseline next-token numeral scoring;
2. 39-feature construction;
3. frozen controller decision;
4. optional L18H13 intervention;
5. post-intervention next-token numeral scoring.

Thus ground truth cannot influence routing or intervention.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


sys.path.insert(
    0,
    "scripts",
)


# ============================================================
# EXACT frozen computational primitives.
# ============================================================

from run_l18h13_gain_all300 import (
    MODEL_ID,
    LAYER,
    HEAD,
    move_inputs,
    HeadGainModifier,
)

from expanded_numeral_metrics import (
    NUMERALS,
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
# Frozen identities.
# ============================================================

DATASET_REVISION = (
    "f44cfe591e8f7e64b63a2fb080b98bffa10b45aa"
)

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

EXPECTED_MANIFEST_SHA256 = (
    "07e9672295230dfdad9e356102165b0190117101f865418efd1d3a6be9d403cd"
)

EXPECTED_CONTROLLER_SHA256 = (
    "d9d8ba3a24814bf71b3f4039d1effa864b57754a00539f1260a1d077c03449f5"
)

EXPECTED_RUNTIME_CONTRACT_SHA256 = (
    "fecae12571c7aa5587a52371beef0494d9a6ba719be7c071fd35af947b6666cd"
)

EXPECTED_SANITY_MANIFEST_SHA256 = (
    "b3ebc25423203f0584327f23a1cabf6b388b5abe8a154a31c07d7def4d429044"
)

EXPECTED_RUNTIME_SOURCE_SHA256 = (
    "8a6582db481b1866a6ab6b086bd1699711a66a9f142927492bc7a8d63e67d68e"
)

EXPECTED_MODEL_AUDIT_SHA256 = (
    "a640968f1a3deaab111d2b782b7948e2ceb8995c92d69d3ff8ea85d5a4b6cdbd"
)

EXPECTED_ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]

EXPECTED_NONNOOP = [
    0.0,
    1.5,
    2.0,
    4.0,
]

EXPECTED_FEATURE_COUNT = 39
EXPECTED_THRESHOLD = 0.1
EXPECTED_RIDGE_ALPHA = 0.01

DUMMY_GT = 1


# ============================================================
# Paths.
# ============================================================

ROOT = Path(".")

RUNTIME_CONTRACT_PATH = Path(
    "configs/holocount_external_v1.json"
)

FULL_MANIFEST_PATH = Path(
    "manifests/generalization_extension/"
    "holocount/holocount_manifest.csv"
)

SANITY_MANIFEST_PATH = Path(
    "manifests/generalization_extension/"
    "holocount/sanity10_manifest.csv"
)

RUNTIME_SOURCE_PATH = Path(
    "manifests/generalization_extension/"
    "holocount/runtime_source_sha256.json"
)

MODEL_REVISION_AUDIT_PATH = Path(
    "manifests/generalization_extension/"
    "holocount/model_revision_audit.json"
)

CONTROLLER_BUNDLE_PATH = Path(
    "outputs/phase2_natural_controller_k500/"
    "final_frozen_controller/"
    "aroma_natural_utility_controller_k500.joblib"
)

DATA_ROOT = Path(
    "/workspace/datasets/"
    f"holocount_{DATASET_REVISION}/data"
)

OUT_ROOT = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1"
)

SANITY_RESULT_PATH = (
    OUT_ROOT / "sanity10_raw_results.csv"
)

SANITY_META_PATH = (
    OUT_ROOT / "sanity10_run_metadata.json"
)

FULL_RESULT_PATH = (
    OUT_ROOT / "full_raw_results.csv"
)

FULL_META_PATH = (
    OUT_ROOT / "full_run_metadata.json"
)

FULL_STARTED_PATH = (
    OUT_ROOT / "full_run_started.json"
)


# ============================================================
# Basic helpers.
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path,
    chunk_size=1024 * 1024,
):

    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            b = f.read(
                chunk_size
            )

            if not b:
                break

            h.update(
                b
            )

    return h.hexdigest()


def sha256_text(
    text,
):

    return hashlib.sha256(
        str(text)
        .encode("utf-8")
    ).hexdigest()


def git_head():

    return (
        subprocess
        .check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )


def load_json(
    path,
):

    return json.loads(
        Path(path)
        .read_text(
            encoding="utf-8"
        )
    )


def load_csv_records(
    path,
):

    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(
                f
            )
        )


# ============================================================
# Frozen source / runtime audits.
# ============================================================

def verify_file_hash(
    path,
    expected,
    label,
):

    actual = sha256_file(
        path
    )

    if actual != expected:

        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    return actual


def verify_source_hashes(
    contract,
):

    expected = dict(
        contract[
            "source_sha256"
        ]
    )

    for path_str, expected_hash in expected.items():

        path = Path(
            path_str
        )

        if not path.is_file():

            raise RuntimeError(
                f"Frozen source file missing: {path}"
            )

        actual = sha256_file(
            path
        )

        if actual != expected_hash:

            raise RuntimeError(
                "Frozen computational source changed:\n"
                f"{path}\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual}"
            )


def load_and_verify_bundle():

    verify_file_hash(
        CONTROLLER_BUNDLE_PATH,
        EXPECTED_CONTROLLER_SHA256,
        "Frozen K500 controller",
    )

    bundle = joblib.load(
        CONTROLLER_BUNDLE_PATH
    )

    if bundle["model_id"] != MODEL_ID:
        raise RuntimeError(
            "Controller model mismatch."
        )

    if (
        bundle[
            "attention_implementation"
        ]
        != "eager"
    ):
        raise RuntimeError(
            "Attention implementation mismatch."
        )

    if int(
        bundle["head_layer"]
    ) != LAYER:
        raise RuntimeError(
            "Controller layer mismatch."
        )

    if int(
        bundle["head_index"]
    ) != HEAD:
        raise RuntimeError(
            "Controller head mismatch."
        )

    if (
        bundle["head_name"]
        != "L18H13"
    ):
        raise RuntimeError(
            "Controller head-name mismatch."
        )

    if (
        bundle["bundle_version"]
        != "AROMA-natural-k500-v1"
    ):
        raise RuntimeError(
            "Controller version mismatch."
        )

    actions = [
        float(x)
        for x in bundle[
            "actions"
        ]
    ]

    if actions != EXPECTED_ACTIONS:
        raise RuntimeError(
            f"Unexpected actions: {actions}"
        )

    nonnoop = [
        float(x)
        for x in bundle[
            "nonnoop_actions"
        ]
    ]

    if nonnoop != EXPECTED_NONNOOP:
        raise RuntimeError(
            f"Unexpected non-NOOP actions: {nonnoop}"
        )

    feature_names = list(
        bundle[
            "feature_names"
        ]
    )

    if (
        len(feature_names)
        != EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Frozen feature-count mismatch."
        )

    if (
        bundle["feature_family"]
        != "full_geometry"
    ):
        raise RuntimeError(
            "Feature-family mismatch."
        )

    if not math.isclose(
        float(
            bundle[
                "ridge_alpha"
            ]
        ),
        EXPECTED_RIDGE_ALPHA,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Ridge alpha mismatch."
        )

    if not math.isclose(
        float(
            bundle[
                "threshold"
            ]
        ),
        EXPECTED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Controller threshold mismatch."
        )

    if int(
        bundle[
            "natural_calibration_n"
        ]
    ) != 500:
        raise RuntimeError(
            "Natural calibration size mismatch."
        )

    return (
        bundle,
        feature_names,
    )


def verify_runtime_contract():

    verify_file_hash(
        RUNTIME_CONTRACT_PATH,
        EXPECTED_RUNTIME_CONTRACT_SHA256,
        "HoloCount runtime contract",
    )

    verify_file_hash(
        RUNTIME_SOURCE_PATH,
        EXPECTED_RUNTIME_SOURCE_SHA256,
        "Frozen runtime-source manifest",
    )

    verify_file_hash(
        MODEL_REVISION_AUDIT_PATH,
        EXPECTED_MODEL_AUDIT_SHA256,
        "Model revision audit",
    )

    contract = load_json(
        RUNTIME_CONTRACT_PATH
    )

    if (
        contract["status"]
        !=
        "FROZEN_BEFORE_HOLOCOUNT_MODEL_INFERENCE"
    ):
        raise RuntimeError(
            "Unexpected runtime-contract status."
        )

    if (
        contract["dataset"]["revision"]
        != DATASET_REVISION
    ):
        raise RuntimeError(
            "HoloCount revision mismatch."
        )

    if (
        contract[
            "dataset"
        ][
            "manifest_sha256"
        ]
        != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Manifest contract mismatch."
        )

    if (
        contract[
            "model"
        ][
            "revision_sha"
        ]
        != MODEL_REVISION
    ):
        raise RuntimeError(
            "Model revision mismatch."
        )

    if (
        contract[
            "model"
        ][
            "processor_revision_sha"
        ]
        != MODEL_REVISION
    ):
        raise RuntimeError(
            "Processor revision mismatch."
        )

    if (
        contract[
            "controller"
        ][
            "bundle_sha256"
        ]
        != EXPECTED_CONTROLLER_SHA256
    ):
        raise RuntimeError(
            "Controller hash contract mismatch."
        )

    verify_source_hashes(
        contract
    )

    return contract


# ============================================================
# Manifest integrity.
# ============================================================

def validate_records(
    records,
    mode,
):

    if mode == "sanity":

        if len(records) != 10:

            raise RuntimeError(
                f"Expected 10 sanity records, "
                f"got {len(records)}."
            )

        forbidden = {
            "answer",
            "answer_int",
            "answer_text",
            "ground_truth",
        }

        present = (
            forbidden
            &
            set(
                records[0].keys()
            )
        )

        if present:

            raise RuntimeError(
                "GT field present in sanity manifest: "
                f"{sorted(present)}"
            )

    elif mode == "full":

        if len(records) != 2480:

            raise RuntimeError(
                f"Expected 2480 full records, "
                f"got {len(records)}."
            )

        required = {
            "answer_int",
        }

        missing = (
            required
            -
            set(
                records[0].keys()
            )
        )

        if missing:

            raise RuntimeError(
                f"Full manifest missing fields: "
                f"{sorted(missing)}"
            )

    else:

        raise ValueError(
            mode
        )

    sample_ids = [
        str(
            r[
                "sample_id"
            ]
        )
        for r in records
    ]

    if (
        len(
            set(
                sample_ids
            )
        )
        != len(sample_ids)
    ):
        raise RuntimeError(
            "Duplicate sample IDs in runner population."
        )

    required_common = {
        "manifest_index",
        "sample_id",
        "split",
        "question",
        "question_sha256",
        "file_name",
        "image_sha256",
        "dataset_revision",
    }

    for i, sample in enumerate(
        records
    ):

        missing = (
            required_common
            -
            set(
                sample.keys()
            )
        )

        if missing:

            raise RuntimeError(
                f"Record {i} missing fields: "
                f"{sorted(missing)}"
            )

        if (
            sample[
                "dataset_revision"
            ]
            != DATASET_REVISION
        ):

            raise RuntimeError(
                "Per-record dataset revision mismatch."
            )

        question = str(
            sample[
                "question"
            ]
        )

        if (
            sha256_text(
                question
            )
            !=
            sample[
                "question_sha256"
            ]
        ):

            raise RuntimeError(
                "Question SHA mismatch for "
                f"{sample['sample_id']}"
            )

        image_path = (
            DATA_ROOT
            /
            str(
                sample[
                    "file_name"
                ]
            )
        )

        if not image_path.is_file():

            raise RuntimeError(
                f"Missing image: {image_path}"
            )

        image_hash = sha256_file(
            image_path
        )

        if (
            image_hash
            !=
            sample[
                "image_sha256"
            ]
        ):

            raise RuntimeError(
                "Image SHA mismatch for "
                f"{sample['sample_id']}"
            )


# ============================================================
# Main protocol audit.
# ============================================================

def protocol_audit():

    contract = (
        verify_runtime_contract()
    )

    verify_file_hash(
        FULL_MANIFEST_PATH,
        EXPECTED_MANIFEST_SHA256,
        "Frozen HoloCount manifest",
    )

    verify_file_hash(
        SANITY_MANIFEST_PATH,
        EXPECTED_SANITY_MANIFEST_SHA256,
        "Frozen sanity-10 manifest",
    )

    if MODEL_ID != (
        "meta-llama/"
        "Llama-3.2-11B-Vision-Instruct"
    ):

        raise RuntimeError(
            "Canonical MODEL_ID changed."
        )

    if LAYER != 18:
        raise RuntimeError(
            "Canonical LAYER changed."
        )

    if HEAD != 13:
        raise RuntimeError(
            "Canonical HEAD changed."
        )

    if NUMERALS != list(
        range(
            16
        )
    ):

        raise RuntimeError(
            "Canonical numeral support changed."
        )

    (
        bundle,
        feature_names,
    ) = (
        load_and_verify_bundle()
    )

    sanity_records = (
        load_csv_records(
            SANITY_MANIFEST_PATH
        )
    )

    validate_records(
        sanity_records,
        "sanity",
    )

    full_records = (
        load_csv_records(
            FULL_MANIFEST_PATH
        )
    )

    validate_records(
        full_records,
        "full",
    )

    print(
        "Runtime contract          : PASS"
    )
    print(
        "Frozen source hashes      : PASS"
    )
    print(
        "Controller SHA            : PASS"
    )
    print(
        "HoloCount manifest SHA    : PASS"
    )
    print(
        "Sanity-10 manifest SHA    : PASS"
    )
    print(
        "All 2480 image SHA256     : PASS"
    )
    print(
        "All question SHA256       : PASS"
    )
    print(
        "Model ID                  :",
        MODEL_ID,
    )
    print(
        "Model revision            :",
        MODEL_REVISION,
    )
    print(
        "Head                      :",
        "L18H13",
    )
    print(
        "Numerals                  :",
        NUMERALS,
    )
    print(
        "Frozen feature count      :",
        len(feature_names),
    )
    print(
        "Frozen threshold          :",
        bundle[
            "threshold"
        ],
    )

    return (
        contract,
        bundle,
        feature_names,
    )


# ============================================================
# Exact model stack.
# ============================================================

def load_model_stack():

    print(
        "\nLoading processor at exact revision..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            revision=
                MODEL_REVISION,
        )
    )

    numeral_ids = (
        numeral_token_ids(
            processor
        )
    )

    if sorted(
        numeral_ids.keys()
    ) != list(
        range(
            16
        )
    ):

        raise RuntimeError(
            "Numeral-token audit failed."
        )

    print(
        "Numeral-token audit       : PASS"
    )

    print(
        "Loading model at exact revision..."
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
        "Model load                : PASS"
    )

    return (
        processor,
        model,
        numeral_ids,
    )


# ============================================================
# One frozen inference example.
# ============================================================

def evaluate_one(
    sample,
    processor,
    model,
    numeral_ids,
    bundle,
    feature_names,
    include_ground_truth,
):

    sample_id = str(
        sample[
            "sample_id"
        ]
    )

    image_path = (
        DATA_ROOT
        /
        str(
            sample[
                "file_name"
            ]
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

    question = str(
        sample[
            "question"
        ]
    )

    inputs = (
        prepare_tallyqa_inputs(
            processor,
            image,
            question,
        )
    )

    inputs = (
        move_inputs(
            inputs,
            model,
        )
    )

    # ========================================================
    # STRICT GT-FREE ROUTING PATH.
    #
    # No answer field is accessed above or below until after
    # post_prediction has been computed.
    # ========================================================

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
        feature_values,
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

    baseline_prediction = int(
        baseline_state[
            "best_numeral"
        ]
    )

    # --------------------------------------------------------
    # Exact historical NOOP semantics.
    # --------------------------------------------------------

    if math.isclose(
        selected_alpha,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        post_state = (
            baseline_state
        )

        hook_calls = 0

    else:

        modifier = (
            HeadGainModifier(
                model=model,
                layer_idx=LAYER,
                head_idx=HEAD,
                alpha=
                    selected_alpha,
            )
        )

        modifier.register()

        try:

            post_state = (
                score_state(
                    model,
                    inputs,
                    numeral_ids,
                    DUMMY_GT,
                )
            )

        finally:

            modifier.remove()

        hook_calls = int(
            modifier.calls
        )

        if hook_calls <= 0:

            raise RuntimeError(
                "L18H13 gain hook not called "
                f"for {sample_id}"
            )

    post_prediction = int(
        post_state[
            "best_numeral"
        ]
    )

    # ========================================================
    # Base row remains GT-free.
    # ========================================================

    row = {
        "manifest_index":
            int(
                sample[
                    "manifest_index"
                ]
            ),

        "sample_id":
            sample_id,

        "split":
            str(
                sample[
                    "split"
                ]
            ),

        "question":
            question,

        "file_name":
            str(
                sample[
                    "file_name"
                ]
            ),

        "image_sha256":
            str(
                sample[
                    "image_sha256"
                ]
            ),

        "baseline_prediction":
            baseline_prediction,

        "selected_alpha":
            float(
                selected_alpha
            ),

        "selected_score":
            float(
                selected_score
            ),

        "post_prediction":
            post_prediction,

        "hook_calls":
            int(
                hook_calls
            ),

        "score_alpha_0":
            float(
                score_map[
                    0.0
                ]
            ),

        "score_alpha_1p5":
            float(
                score_map[
                    1.5
                ]
            ),

        "score_alpha_2":
            float(
                score_map[
                    2.0
                ]
            ),

        "score_alpha_4":
            float(
                score_map[
                    4.0
                ]
            ),

        "baseline_expected_numeral":
            float(
                baseline_state[
                    "expected_numeral"
                ]
            ),

        "post_expected_numeral":
            float(
                post_state[
                    "expected_numeral"
                ]
            ),

        "baseline_entropy":
            float(
                baseline_state[
                    "entropy"
                ]
            ),

        "post_entropy":
            float(
                post_state[
                    "entropy"
                ]
            ),

        "baseline_top1_prob":
            float(
                baseline_state[
                    "top1_prob"
                ]
            ),

        "post_top1_prob":
            float(
                post_state[
                    "top1_prob"
                ]
            ),

        "baseline_conditional_margin":
            float(
                baseline_state[
                    "conditional_margin"
                ]
            ),

        "post_conditional_margin":
            float(
                post_state[
                    "conditional_margin"
                ]
            ),
    }

    for name in feature_names:

        row[
            f"feature__{name}"
        ] = float(
            feature_values[
                name
            ]
        )

    # ========================================================
    # ONLY NOW may ground truth be accessed.
    # ========================================================

    if include_ground_truth:

        gt = int(
            sample[
                "answer_int"
            ]
        )

        baseline_correct = bool(
            baseline_prediction
            ==
            gt
        )

        post_correct = bool(
            post_prediction
            ==
            gt
        )

        repair = bool(
            (
                not baseline_correct
            )
            and
            post_correct
        )

        break_case = bool(
            baseline_correct
            and
            (
                not post_correct
            )
        )

        row.update({
            "ground_truth":
                gt,

            "ground_truth_le_15":
                bool(
                    gt <= 15
                ),

            "baseline_correct":
                baseline_correct,

            "post_correct":
                post_correct,

            "repair":
                repair,

            "break_case":
                break_case,
        })

    return row


# ============================================================
# Durable CSV writer.
# ============================================================

def write_rows(
    records,
    result_path,
    processor,
    model,
    numeral_ids,
    bundle,
    feature_names,
    include_ground_truth,
    resume=False,
    desc="HoloCount",
):

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = set()

    if result_path.exists():

        if not resume:

            raise RuntimeError(
                f"Output already exists: {result_path}. "
                "Refusing to overwrite."
            )

        existing = (
            load_csv_records(
                result_path
            )
        )

        completed = {
            str(
                r[
                    "sample_id"
                ]
            )
            for r in existing
        }

    remaining = [
        r
        for r in records
        if str(
            r[
                "sample_id"
            ]
        )
        not in completed
    ]

    if not remaining:

        print(
            "No remaining examples."
        )

        return

    first_row = evaluate_one(
        remaining[0],
        processor,
        model,
        numeral_ids,
        bundle,
        feature_names,
        include_ground_truth,
    )

    fieldnames = list(
        first_row.keys()
    )

    mode = (
        "a"
        if result_path.exists()
        and resume
        else
        "w"
    )

    with result_path.open(
        mode,
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                fieldnames,
        )

        if mode == "w":
            writer.writeheader()

        writer.writerow(
            first_row
        )

        f.flush()
        os.fsync(
            f.fileno()
        )

        for sample in tqdm(
            remaining[1:],
            desc=desc,
        ):

            row = evaluate_one(
                sample,
                processor,
                model,
                numeral_ids,
                bundle,
                feature_names,
                include_ground_truth,
            )

            if list(
                row.keys()
            ) != fieldnames:

                raise RuntimeError(
                    "Result schema changed during run."
                )

            writer.writerow(
                row
            )

            f.flush()
            os.fsync(
                f.fileno()
            )


# ============================================================
# Run metadata.
# ============================================================

def write_run_metadata(
    path,
    mode,
    result_path,
    record_count,
    started_utc,
):

    payload = {
        "experiment":
            "AROMA HoloCount external transfer v1",

        "mode":
            mode,

        "started_utc":
            started_utc,

        "completed_utc":
            utc_now(),

        "git_head":
            git_head(),

        "dataset_revision":
            DATASET_REVISION,

        "manifest_sha256":
            EXPECTED_MANIFEST_SHA256,

        "model_id":
            MODEL_ID,

        "model_revision":
            MODEL_REVISION,

        "controller_sha256":
            EXPECTED_CONTROLLER_SHA256,

        "head":
            "L18H13",

        "numeral_candidates":
            NUMERALS,

        "result_path":
            str(
                result_path
            ),

        "result_sha256":
            sha256_file(
                result_path
            ),

        "record_count":
            int(
                record_count
            ),

        "ground_truth_used":
            bool(
                mode == "full"
            ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


# ============================================================
# Sanity mode.
# ============================================================

def run_sanity(
    bundle,
    feature_names,
):

    records = load_csv_records(
        SANITY_MANIFEST_PATH
    )

    validate_records(
        records,
        "sanity",
    )

    # Absolute safety assertion.
    for sample in records:

        for forbidden in [
            "answer",
            "answer_int",
            "answer_text",
            "ground_truth",
        ]:

            if forbidden in sample:

                raise RuntimeError(
                    "Ground-truth field entered sanity path."
                )

    started = utc_now()

    (
        processor,
        model,
        numeral_ids,
    ) = (
        load_model_stack()
    )

    write_rows(
        records=records,
        result_path=
            SANITY_RESULT_PATH,
        processor=processor,
        model=model,
        numeral_ids=numeral_ids,
        bundle=bundle,
        feature_names=
            feature_names,
        include_ground_truth=
            False,
        resume=False,
        desc="HoloCount sanity-10",
    )

    results = load_csv_records(
        SANITY_RESULT_PATH
    )

    if len(results) != 10:

        raise RuntimeError(
            "Sanity run did not produce exactly 10 rows."
        )

    # Engineering-only diagnostics.
    alphas = [
        float(
            x[
                "selected_alpha"
            ]
        )
        for x in results
    ]

    nonnoop_rows = [
        x
        for x in results
        if not math.isclose(
            float(
                x[
                    "selected_alpha"
                ]
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]

    for row in nonnoop_rows:

        if int(
            row[
                "hook_calls"
            ]
        ) <= 0:

            raise RuntimeError(
                "Non-NOOP sanity row "
                "has no hook calls."
            )

    write_run_metadata(
        SANITY_META_PATH,
        "sanity",
        SANITY_RESULT_PATH,
        10,
        started,
    )

    print()
    print(
        "SANITY ENGINEERING CHECK COMPLETE"
    )
    print(
        "Rows                      :",
        len(results),
    )
    print(
        "39-feature extraction     : PASS"
    )
    print(
        "GT-free sanity manifest   : PASS"
    )
    print(
        "Controller decisions      : PASS"
    )
    print(
        "Non-NOOP hook audit       : PASS"
    )
    print(
        "Selected alphas           :",
        alphas,
    )
    print(
        "Accuracy                  : NOT COMPUTED"
    )
    print(
        "Ground truth              : NOT ACCESSED"
    )


# ============================================================
# Full frozen evaluation.
# ============================================================

def run_full(
    bundle,
    feature_names,
    resume=False,
):

    records = load_csv_records(
        FULL_MANIFEST_PATH
    )

    validate_records(
        records,
        "full",
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        FULL_STARTED_PATH.exists()
        and
        not resume
    ):

        raise RuntimeError(
            "Full HoloCount evaluation has already "
            "been started. Use --resume if appropriate."
        )

    if not FULL_STARTED_PATH.exists():

        FULL_STARTED_PATH.write_text(
            json.dumps({
                "started_utc":
                    utc_now(),

                "git_head":
                    git_head(),

                "manifest_sha256":
                    EXPECTED_MANIFEST_SHA256,

                "controller_sha256":
                    EXPECTED_CONTROLLER_SHA256,

                "model_revision":
                    MODEL_REVISION,
            }, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    started = utc_now()

    (
        processor,
        model,
        numeral_ids,
    ) = (
        load_model_stack()
    )

    write_rows(
        records=records,
        result_path=
            FULL_RESULT_PATH,
        processor=processor,
        model=model,
        numeral_ids=numeral_ids,
        bundle=bundle,
        feature_names=
            feature_names,
        include_ground_truth=
            True,
        resume=resume,
        desc="HoloCount external full",
    )

    results = load_csv_records(
        FULL_RESULT_PATH
    )

    if len(results) != 2480:

        raise RuntimeError(
            "Full run incomplete: "
            f"{len(results)} / 2480 rows."
        )

    write_run_metadata(
        FULL_META_PATH,
        "full",
        FULL_RESULT_PATH,
        2480,
        started,
    )

    print()
    print(
        "FULL RAW INFERENCE COMPLETE"
    )
    print(
        "Rows                      :",
        len(results),
    )
    print(
        "Result path               :",
        FULL_RESULT_PATH,
    )
    print(
        "Result SHA256             :",
        sha256_file(
            FULL_RESULT_PATH
        ),
    )
    print()
    print(
        "NOTE: publication statistics are intentionally "
        "computed by the separately frozen analysis stage."
    )


# ============================================================
# CLI.
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Run provenance and integrity checks only. "
            "Does not load the VLM."
        ),
    )

    group.add_argument(
        "--sanity",
        action="store_true",
        help=(
            "Run the frozen GT-free sanity-10 set."
        ),
    )

    group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the frozen 2480-example HoloCount evaluation."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted --full run."
        ),
    )

    args = parser.parse_args()

    if (
        args.resume
        and
        not args.full
    ):

        raise RuntimeError(
            "--resume is valid only with --full."
        )

    print(
        "======================================================================"
    )
    print(
        "AROMA HOLOCOUNT EXTERNAL V1"
    )
    print(
        "======================================================================"
    )
    print(
        "Git HEAD                  :",
        git_head(),
    )

    (
        contract,
        bundle,
        feature_names,
    ) = (
        protocol_audit()
    )

    if args.audit_only:

        print()
        print(
            "AUDIT-ONLY PASS"
        )
        print(
            "NO PROCESSOR OR MODEL WAS LOADED"
        )
        print(
            "NO HOLOCOUNT MODEL INFERENCE OCCURRED"
        )

        return

    if args.sanity:

        run_sanity(
            bundle,
            feature_names,
        )

        return

    if args.full:

        run_full(
            bundle,
            feature_names,
            resume=args.resume,
        )

        return


if __name__ == "__main__":
    main()
