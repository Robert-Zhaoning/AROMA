#!/usr/bin/env python3

"""
AROMA C2 — Prospective off-target selectivity baseline population runner.

Purpose
-------
Construct the prospectively frozen N=1000 baseline-correct GQA population
for off_target_selectivity_v1.

This runner is BASELINE ONLY.

It contains:
    - no intervention hook;
    - no MN actuation;
    - no Whole-head actuation;
    - no proposal controller;
    - no candidate utility router.

The intervention stress test is intentionally a later stage and MUST NOT
run before the N=1000 baseline-correct population is frozen.

Modes
-----
preflight:
    CPU-only frozen-asset and semantic checks.

smoke:
    Run baseline free generation on the first N frozen candidates.
    Smoke outcomes are plumbing diagnostics only and are never used to
    alter the frozen protocol or select the formal population.

formal:
    Scan candidate_rank order from 1 upward and stop immediately when
    exactly 1000 baseline-correct examples have been accumulated.
    Formal runs are resumable from an atomically written scan checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import string
import subprocess
import sys
import time
import unicodedata

from pathlib import Path

import pandas as pd
from PIL import Image


# ============================================================
# Frozen paths / identifiers
# ============================================================

PROTOCOL_PATH = Path(
    "manifests/off_target_selectivity_v1/"
    "prebaseline_protocol_v1.json"
)

MANIFEST_PATH = Path(
    "manifests/off_target_selectivity_v1/"
    "prebaseline_candidate_manifest_v1.csv"
)

SOURCE_PATH = Path(
    "data/gqa_raw/"
    "val_balanced_questions.json"
)

IMAGE_ROOT = Path(
    "data/gqa_raw/images"
)

MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

MODEL_SNAPSHOT = Path(
    "/workspace/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-11B-Vision-Instruct/"
    "snapshots/"
    + MODEL_REVISION
)

EXPECTED_PROTOCOL_SHA256 = (
    "b3fde7262e025ac7c0bb5c11750c9ee218ea7e7ffdb2e0b3130d795c4cc0dad2"
)

EXPECTED_MANIFEST_SHA256 = (
    "129713fe4e746804078a9c574a39f515be40e97d61271b09d52d5def52edcdaf"
)

EXPECTED_SOURCE_SHA256 = (
    "2f675643c0cfffe0485dbae0c0c0dfc16b547443309a98544502bc1ae87c0779"
)

EXPECTED_CANDIDATES = 10234
EXPECTED_TARGET = 1000

INSTRUCTION = (
    "Answer the question about the image. "
    "Respond with only the short answer, using a single word or short phrase. "
    "Do not explain."
)

DO_SAMPLE = False
MAX_NEW_TOKENS = 12

SMOKE_DIR = Path(
    "outputs/off_target_selectivity_v1/"
    "baseline_smoke_v1"
)

FORMAL_DIR = Path(
    "outputs/off_target_selectivity_v1/"
    "baseline_population_v1"
)


# ============================================================
# Utilities
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write(text)
        f.flush()
        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def atomic_write_json(
    path: Path,
    obj,
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )


def atomic_write_jsonl(
    path: Path,
    rows,
) -> None:
    text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
        )
        + "\n"
        for row in rows
    )

    atomic_write_text(
        path,
        text,
    )


def load_jsonl(
    path: Path,
):
    if not path.exists():
        return []

    rows = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        rows.append(
            json.loads(line)
        )

    return rows


# ============================================================
# Frozen minimal cleanup
# ============================================================

def frozen_cleanup(
    text,
) -> str:
    """
    Frozen C2 cleanup:

        Unicode NFKC
        -> lowercase
        -> first non-empty generated line
        -> trim whitespace
        -> collapse internal whitespace
        -> strip surrounding punctuation only
    """

    if text is None:
        text = ""

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = text.lower()

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    text = (
        lines[0]
        if lines
        else ""
    )

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.strip(
        string.punctuation
    )

    text = text.strip()

    return text


# ============================================================
# Prompt / multimodal input
# ============================================================

def compose_prompt(
    question: str,
) -> str:
    question = str(
        question
    ).strip()

    if not question:
        raise RuntimeError(
            "Empty GQA question."
        )

    return (
        question
        + "\n\n"
        + INSTRUCTION
    )


def prepare_inputs(
    processor,
    image,
    question,
):
    prompt_text = (
        compose_prompt(
            question
        )
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]

    formatted = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    inputs = processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )

    return (
        inputs,
        prompt_text,
        formatted,
    )


# ============================================================
# Frozen asset audit
# ============================================================

def load_and_audit_frozen_assets():
    print(
        "===== FROZEN ASSET AUDIT ====="
    )

    required = [
        PROTOCOL_PATH,
        MANIFEST_PATH,
        SOURCE_PATH,
    ]

    for path in required:
        if not path.is_file():
            raise RuntimeError(
                f"Missing frozen file: {path}"
            )

    if not IMAGE_ROOT.is_dir():
        raise RuntimeError(
            f"Missing image root: {IMAGE_ROOT}"
        )

    protocol_sha = (
        sha256_file(
            PROTOCOL_PATH
        )
    )

    manifest_sha = (
        sha256_file(
            MANIFEST_PATH
        )
    )

    source_sha = (
        sha256_file(
            SOURCE_PATH
        )
    )

    print(
        "protocol SHA :",
        protocol_sha,
    )
    print(
        "manifest SHA :",
        manifest_sha,
    )
    print(
        "source SHA   :",
        source_sha,
    )

    if (
        protocol_sha
        != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "Frozen protocol SHA mismatch."
        )

    if (
        manifest_sha
        != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Frozen manifest SHA mismatch."
        )

    if (
        source_sha
        != EXPECTED_SOURCE_SHA256
    ):
        raise RuntimeError(
            "Frozen GQA source SHA mismatch."
        )

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        protocol["experiment"]
        != "off_target_selectivity_v1"
    ):
        raise RuntimeError(
            "Unexpected experiment ID."
        )

    if (
        protocol["stage"]
        != "prebaseline_freeze"
    ):
        raise RuntimeError(
            "Unexpected protocol stage."
        )

    if (
        protocol["candidate_sampling"][
            "candidate_manifest_sha256"
        ]
        != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Protocol-recorded manifest SHA mismatch."
        )

    if (
        protocol["dataset"][
            "source_sha256"
        ]
        != EXPECTED_SOURCE_SHA256
    ):
        raise RuntimeError(
            "Protocol-recorded source SHA mismatch."
        )

    if (
        protocol["model"]["id"]
        != MODEL_ID
    ):
        raise RuntimeError(
            "Frozen model ID mismatch."
        )

    if (
        protocol["model"]["revision"]
        != MODEL_REVISION
    ):
        raise RuntimeError(
            "Frozen model revision mismatch."
        )

    decoding = (
        protocol[
            "baseline_decoding"
        ]
    )

    if (
        decoding[
            "prompt_template"
        ]
        != INSTRUCTION
    ):
        raise RuntimeError(
            "Frozen prompt mismatch."
        )

    if (
        bool(
            decoding[
                "do_sample"
            ]
        )
        is not DO_SAMPLE
    ):
        raise RuntimeError(
            "Frozen do_sample mismatch."
        )

    if (
        int(
            decoding[
                "max_new_tokens"
            ]
        )
        != MAX_NEW_TOKENS
    ):
        raise RuntimeError(
            "Frozen max_new_tokens mismatch."
        )

    if (
        int(
            protocol[
                "baseline_population"
            ][
                "target_n"
            ]
        )
        != EXPECTED_TARGET
    ):
        raise RuntimeError(
            "Frozen target N mismatch."
        )

    if not MODEL_SNAPSHOT.is_dir():
        raise RuntimeError(
            "Exact frozen model snapshot "
            "is not available locally."
        )

    df = pd.read_csv(
        MANIFEST_PATH,
        dtype={
            "question_id":
                "string",
            "image_id":
                "string",
            "question":
                "string",
            "answer":
                "string",
            "answer_normalized":
                "string",
        },
    )

    if len(df) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES} "
            f"candidates, got {len(df)}."
        )

    df["candidate_rank"] = (
        pd.to_numeric(
            df[
                "candidate_rank"
            ],
            errors="raise",
        )
        .astype(int)
    )

    df = (
        df.sort_values(
            "candidate_rank"
        )
        .reset_index(
            drop=True
        )
    )

    expected_ranks = list(
        range(
            1,
            EXPECTED_CANDIDATES + 1,
        )
    )

    if (
        df[
            "candidate_rank"
        ].tolist()
        != expected_ranks
    ):
        raise RuntimeError(
            "candidate_rank is not exactly "
            "1..10234."
        )

    if (
        df["image_id"].nunique()
        != EXPECTED_CANDIDATES
    ):
        raise RuntimeError(
            "Frozen image IDs are not unique."
        )

    if (
        df["question_id"].nunique()
        != EXPECTED_CANDIDATES
    ):
        raise RuntimeError(
            "Frozen question IDs are not unique."
        )

    derived_gold = (
        df["answer"]
        .map(
            frozen_cleanup
        )
    )

    stored_gold = (
        df[
            "answer_normalized"
        ]
        .fillna("")
    )

    if not derived_gold.eq(
        stored_gold
    ).all():
        raise RuntimeError(
            "Frozen gold normalization "
            "consistency failed."
        )

    missing = []

    for row in df.itertuples(
        index=False
    ):
        path = (
            IMAGE_ROOT
            /
            f"{row.image_id}.jpg"
        )

        if not path.is_file():
            missing.append(
                (
                    row.candidate_rank,
                    row.image_id,
                )
            )

    if missing:
        raise RuntimeError(
            "Missing frozen images, first: "
            + repr(
                missing[:20]
            )
        )

    print(
        "candidate rows :",
        len(df),
    )
    print(
        "unique images  :",
        df[
            "image_id"
        ].nunique(),
    )
    print(
        "unique qids    :",
        df[
            "question_id"
        ].nunique(),
    )
    print(
        "gold cleanup   : PASS"
    )
    print(
        "image resolve  : "
        "10234 / 10234 PASS"
    )
    print(
        "model snapshot : PASS"
    )

    print()
    print(
        "First composed prompt:"
    )
    print(
        "-" * 72
    )
    print(
        compose_prompt(
            df.iloc[0][
                "question"
            ]
        )
    )
    print(
        "-" * 72
    )

    return (
        protocol,
        df,
    )


# ============================================================
# Model load
# ============================================================

def load_model():
    import torch

    from transformers import (
        AutoProcessor,
        MllamaForConditionalGeneration,
    )

    print()
    print(
        "===== MODEL LOAD ====="
    )
    print(
        "model    :",
        MODEL_ID,
    )
    print(
        "revision :",
        MODEL_REVISION,
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    )

    model.eval()

    input_device = (
        next(
            model.parameters()
        ).device
    )

    generation_config = (
        copy.deepcopy(
            model.generation_config
        )
    )

    # Explicitly override sampling defaults from
    # the model's generation_config.json.
    generation_config.do_sample = False
    generation_config.max_new_tokens = (
        MAX_NEW_TOKENS
    )

    # These sampling-only values MUST NOT silently
    # affect the frozen greedy decode.
    generation_config.temperature = None
    generation_config.top_p = None

    print(
        "input device :",
        input_device,
    )
    print(
        "dtype        :",
        next(
            model.parameters()
        ).dtype,
    )
    print(
        "do_sample    :",
        generation_config.do_sample,
    )
    print(
        "max_new_tokens:",
        generation_config.max_new_tokens,
    )

    return (
        processor,
        model,
        input_device,
        generation_config,
    )


# ============================================================
# Single baseline generation
# ============================================================

def run_one(
    *,
    row,
    processor,
    model,
    input_device,
    generation_config,
):
    import torch

    image_path = (
        IMAGE_ROOT
        /
        f"{row.image_id}.jpg"
    )

    if not image_path.is_file():
        raise RuntimeError(
            "Frozen image disappeared: "
            f"{image_path}"
        )

    with Image.open(
        image_path
    ) as im:
        image = (
            im.convert(
                "RGB"
            )
        )

    (
        inputs,
        prompt_text,
        formatted_prompt,
    ) = prepare_inputs(
        processor,
        image,
        row.question,
    )

    input_len = int(
        inputs[
            "input_ids"
        ].shape[1]
    )

    moved = {}

    for key, value in inputs.items():
        if hasattr(
            value,
            "to",
        ):
            moved[key] = (
                value.to(
                    input_device
                )
            )
        else:
            moved[key] = value

    start = time.time()

    with torch.inference_mode():
        output_ids = (
            model.generate(
                **moved,
                generation_config=
                    generation_config,
            )
        )

    elapsed = (
        time.time()
        - start
    )

    if (
        output_ids.ndim != 2
        or output_ids.shape[0] != 1
    ):
        raise RuntimeError(
            "Unexpected generation shape: "
            f"{tuple(output_ids.shape)}"
        )

    if (
        output_ids.shape[1]
        < input_len
    ):
        raise RuntimeError(
            "Generated sequence shorter "
            "than input sequence."
        )

    new_ids = (
        output_ids[
            0,
            input_len:
        ]
        .detach()
        .cpu()
    )

    # IMPORTANT:
    # decode ONLY newly generated tokens.
    raw_generation = (
        processor.decode(
            new_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )

    prediction = (
        frozen_cleanup(
            raw_generation
        )
    )

    gold = str(
        row.answer_normalized
    )

    correct = (
        prediction
        == gold
    )

    return {
        "candidate_rank":
            int(
                row.candidate_rank
            ),
        "selection_hash":
            str(
                row.selection_hash
            ),
        "question_id":
            str(
                row.question_id
            ),
        "image_id":
            str(
                row.image_id
            ),
        "question":
            str(
                row.question
            ),
        "answer":
            str(
                row.answer
            ),
        "answer_normalized":
            gold,
        "structural_type":
            str(
                row.structural_type
            ),
        "semantic_type":
            str(
                row.semantic_type
            ),
        "detailed_type":
            str(
                row.detailed_type
            ),
        "prompt_sha256":
            sha256_text(
                prompt_text
            ),
        "formatted_prompt_sha256":
            sha256_text(
                formatted_prompt
            ),
        "input_tokens":
            input_len,
        "generated_token_ids":
            [
                int(x)
                for x
                in new_ids.tolist()
            ],
        "generated_token_count":
            int(
                len(
                    new_ids
                )
            ),
        "raw_generation":
            raw_generation,
        "prediction_normalized":
            prediction,
        "baseline_correct":
            bool(
                correct
            ),
        "generation_seconds":
            float(
                elapsed
            ),
    }


# ============================================================
# Checkpoint validation
# ============================================================

def validate_existing_scan(
    rows,
    manifest_df,
):
    if not rows:
        return

    ranks = [
        int(
            r[
                "candidate_rank"
            ]
        )
        for r in rows
    ]

    expected = list(
        range(
            1,
            len(rows) + 1,
        )
    )

    if ranks != expected:
        raise RuntimeError(
            "Existing scan is not a "
            "contiguous prefix of "
            "candidate_rank."
        )

    for i, result in enumerate(
        rows
    ):
        manifest_row = (
            manifest_df.iloc[i]
        )

        if (
            str(
                result[
                    "question_id"
                ]
            )
            != str(
                manifest_row[
                    "question_id"
                ]
            )
        ):
            raise RuntimeError(
                "Existing checkpoint "
                "question_id mismatch at "
                f"rank {i + 1}."
            )

        if (
            str(
                result[
                    "image_id"
                ]
            )
            != str(
                manifest_row[
                    "image_id"
                ]
            )
        ):
            raise RuntimeError(
                "Existing checkpoint "
                "image_id mismatch at "
                f"rank {i + 1}."
            )


# ============================================================
# Final population freeze
# ============================================================

def freeze_population(
    out_dir,
    scan_rows,
):
    selected = [
        row
        for row in scan_rows
        if bool(
            row[
                "baseline_correct"
            ]
        )
    ]

    if (
        len(selected)
        != EXPECTED_TARGET
    ):
        raise RuntimeError(
            "Cannot freeze population: "
            f"expected exactly "
            f"{EXPECTED_TARGET} correct, "
            f"got {len(selected)}."
        )

    population_path = (
        out_dir
        /
        "baseline_correct_population_v1.csv"
    )

    tmp_path = Path(
        str(
            population_path
        )
        + ".tmp"
    )

    pd.DataFrame(
        selected
    ).to_csv(
        tmp_path,
        index=False,
        lineterminator="\n",
    )

    os.replace(
        tmp_path,
        population_path,
    )

    return population_path


# ============================================================
# Main run
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "preflight",
            "smoke",
            "formal",
        ],
        required=True,
    )

    parser.add_argument(
        "--smoke-n",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--target-correct",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--reset-smoke",
        action="store_true",
    )

    args = parser.parse_args()

    print(
        "=" * 72
    )
    print(
        "AROMA C2 BASELINE POPULATION RUNNER V1"
    )
    print(
        "=" * 72
    )
    print(
        "mode:",
        args.mode,
    )

    (
        protocol,
        df,
    ) = load_and_audit_frozen_assets()

    if args.mode == "preflight":
        print()
        print(
            "=" * 72
        )
        print(
            "C2 BASELINE PREFLIGHT: PASS"
        )
        print(
            "NO MODEL LOADED"
        )
        print(
            "NO GPU INFERENCE"
        )
        print(
            "=" * 72
        )
        return

    if args.mode == "formal":
        if (
            args.target_correct
            != EXPECTED_TARGET
        ):
            raise RuntimeError(
                "Formal target is frozen at "
                "N=1000 and cannot be changed."
            )

        out_dir = FORMAL_DIR
        target_correct = (
            EXPECTED_TARGET
        )
        max_candidates = len(
            df
        )

    else:
        if args.smoke_n <= 0:
            raise RuntimeError(
                "--smoke-n must be positive."
            )

        if (
            args.smoke_n
            > len(df)
        ):
            raise RuntimeError(
                "--smoke-n exceeds "
                "candidate count."
            )

        out_dir = SMOKE_DIR
        target_correct = None
        max_candidates = (
            args.smoke_n
        )

        if (
            args.reset_smoke
            and out_dir.exists()
        ):
            shutil.rmtree(
                out_dir
            )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scan_path = (
        out_dir
        /
        "baseline_scan_v1.jsonl"
    )

    progress_path = (
        out_dir
        /
        "progress_v1.json"
    )

    metadata_path = (
        out_dir
        /
        "metadata_v1.json"
    )

    existing_rows = (
        load_jsonl(
            scan_path
        )
    )

    validate_existing_scan(
        existing_rows,
        df,
    )

    if (
        args.mode == "smoke"
        and len(existing_rows)
        >= max_candidates
    ):
        print(
            "Smoke checkpoint already "
            "contains requested samples."
        )
        return

    correct_count = sum(
        bool(
            row[
                "baseline_correct"
            ]
        )
        for row in existing_rows
    )

    if (
        args.mode == "formal"
        and correct_count
        >= EXPECTED_TARGET
    ):
        if (
            correct_count
            != EXPECTED_TARGET
        ):
            raise RuntimeError(
                "Existing formal checkpoint "
                "contains more than the "
                "frozen target N."
            )

        population_path = (
            freeze_population(
                out_dir,
                existing_rows,
            )
        )

        print(
            "Formal population already "
            "complete:"
        )
        print(
            population_path
        )
        return

    metadata = {
        "experiment":
            "off_target_selectivity_v1",
        "stage":
            (
                "baseline_smoke_v1"
                if args.mode
                == "smoke"
                else
                "baseline_population_v1"
            ),
        "baseline_only":
            True,
        "intervention_enabled":
            False,
        "candidate_manifest":
            str(
                MANIFEST_PATH
            ),
        "candidate_manifest_sha256":
            EXPECTED_MANIFEST_SHA256,
        "protocol":
            str(
                PROTOCOL_PATH
            ),
        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,
        "gqa_source":
            str(
                SOURCE_PATH
            ),
        "gqa_source_sha256":
            EXPECTED_SOURCE_SHA256,
        "image_root":
            str(
                IMAGE_ROOT
            ),
        "model_id":
            MODEL_ID,
        "model_revision":
            MODEL_REVISION,
        "local_snapshot":
            str(
                MODEL_SNAPSHOT
            ),
        "prompt_instruction":
            INSTRUCTION,
        "prompt_composition":
            "question + '\\n\\n' + frozen instruction",
        "chat_template":
            (
                "user content = "
                "[image, text]; "
                "add_generation_prompt=True"
            ),
        "processor_add_special_tokens":
            False,
        "do_sample":
            False,
        "max_new_tokens":
            MAX_NEW_TOKENS,
        "decode_scope":
            "newly generated tokens only",
        "normalization": [
            "Unicode NFKC",
            "lowercase",
            "first non-empty generated line",
            "trim whitespace",
            "collapse internal whitespace",
            "strip surrounding ASCII punctuation only",
        ],
        "correctness":
            "exact normalized string match",
        "formal_target_n":
            EXPECTED_TARGET,
        "git_head_at_start":
            git_head(),
        "resume_rows_at_start":
            len(
                existing_rows
            ),
    }

    atomic_write_json(
        metadata_path,
        metadata,
    )

    (
        processor,
        model,
        input_device,
        generation_config,
    ) = load_model()

    scan_rows = list(
        existing_rows
    )

    start_idx = len(
        scan_rows
    )

    print()
    print(
        "===== BASELINE SCAN ====="
    )
    print(
        "resume rank :",
        start_idx + 1,
    )
    print(
        "existing correct:",
        correct_count,
    )

    for idx in range(
        start_idx,
        max_candidates,
    ):
        row = df.iloc[
            idx
        ]

        result = run_one(
            row=row,
            processor=processor,
            model=model,
            input_device=input_device,
            generation_config=
                generation_config,
        )

        scan_rows.append(
            result
        )

        if result[
            "baseline_correct"
        ]:
            correct_count += 1

        atomic_write_jsonl(
            scan_path,
            scan_rows,
        )

        progress = {
            "mode":
                args.mode,
            "processed":
                len(
                    scan_rows
                ),
            "last_candidate_rank":
                result[
                    "candidate_rank"
                ],
            "baseline_correct":
                correct_count,
            "formal_target":
                EXPECTED_TARGET,
            "last_question_id":
                result[
                    "question_id"
                ],
            "last_image_id":
                result[
                    "image_id"
                ],
        }

        atomic_write_json(
            progress_path,
            progress,
        )

        print(
            f"rank={result['candidate_rank']:5d} "
            f"qid={result['question_id']} "
            f"gold={result['answer_normalized']!r} "
            f"raw={result['raw_generation']!r} "
            f"pred={result['prediction_normalized']!r} "
            f"correct={result['baseline_correct']} "
            f"correct_total={correct_count} "
            f"gen_s={result['generation_seconds']:.3f}",
            flush=True,
        )

        if (
            args.mode == "formal"
            and correct_count
            == EXPECTED_TARGET
        ):
            break

    print()
    print(
        "===== SCAN COMPLETE ====="
    )
    print(
        "processed:",
        len(
            scan_rows
        ),
    )
    print(
        "correct  :",
        correct_count,
    )

    if args.mode == "smoke":
        if (
            len(scan_rows)
            != max_candidates
        ):
            raise RuntimeError(
                "Smoke did not process the "
                "requested candidate count."
            )

        print()
        print(
            "=" * 72
        )
        print(
            "C2 BASELINE SMOKE COMPLETE"
        )
        print(
            "NOTE: smoke accuracy is NOT "
            "a pass/fail criterion."
        )
        print(
            "=" * 72
        )
        return

    if (
        correct_count
        != EXPECTED_TARGET
    ):
        raise RuntimeError(
            "Candidate pool exhausted before "
            "reaching N=1000 "
            "baseline-correct examples."
        )

    population_path = (
        freeze_population(
            out_dir,
            scan_rows,
        )
    )

    final_summary = {
        "experiment":
            "off_target_selectivity_v1",
        "stage":
            "baseline_population_v1",
        "processed_candidates":
            len(
                scan_rows
            ),
        "baseline_correct_n":
            correct_count,
        "target_n":
            EXPECTED_TARGET,
        "last_candidate_rank":
            scan_rows[-1][
                "candidate_rank"
            ],
        "population_file":
            str(
                population_path
            ),
        "population_sha256":
            sha256_file(
                population_path
            ),
        "scan_file":
            str(
                scan_path
            ),
        "scan_sha256":
            sha256_file(
                scan_path
            ),
        "git_head_at_finish":
            git_head(),
    }

    atomic_write_json(
        out_dir
        /
        "summary_v1.json",
        final_summary,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "FORMAL N=1000 BASELINE POPULATION FROZEN"
    )
    print(
        "population:",
        population_path,
    )
    print(
        "population SHA:",
        final_summary[
            "population_sha256"
        ],
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
