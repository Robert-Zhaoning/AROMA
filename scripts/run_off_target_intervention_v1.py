#!/usr/bin/env python3

"""
AROMA C2 — Prospective off-target selectivity intervention runner.

Frozen population:
    N = 1000 baseline-correct GQA examples.

Frozen methods:
    WHOLE
    MN

Frozen lambda grid:
    [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

Total formal conditions:
    1000 * 7 * 2 = 14000

This runner:
    - reuses the frozen baseline prompt / generation / normalization path;
    - reuses the already-audited WHOLE and MN intervention implementations;
    - enforces displacement invariants on every generation;
    - is checkpoint/resume safe;
    - performs NO statistical endpoint analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_off_target_baseline_v1 import (
    frozen_cleanup,
    load_model,
)

from audit_off_target_matched_budget_v1 import (
    AuditedWholeModifier,
    AuditedMNModifier,
    build_inputs,
)


# ============================================================
# Frozen artifacts
# ============================================================

PROTOCOL_PATH = Path(
    "manifests/off_target_selectivity_v1/"
    "prebaseline_protocol_v1.json"
)

POPULATION_PATH = Path(
    "manifests/off_target_selectivity_v1/"
    "baseline_correct_population_v1.csv"
)

POPULATION_FREEZE_PATH = Path(
    "manifests/off_target_selectivity_v1/"
    "baseline_population_freeze_v1.json"
)

EXECUTION_FREEZE_PATH = Path(
    "manifests/off_target_selectivity_v1/"
    "intervention_execution_freeze_v1.json"
)

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

RUNNER_PATH = Path(
    "scripts/run_off_target_intervention_v1.py"
)

OUTPUT_DIR = Path(
    "outputs/off_target_selectivity_v1/"
    "intervention_stress_test_v1"
)

RESULTS_JSONL = (
    OUTPUT_DIR /
    "intervention_results_v1.jsonl"
)

RESULTS_CSV = (
    OUTPUT_DIR /
    "intervention_results_v1.csv"
)

PROGRESS_PATH = (
    OUTPUT_DIR /
    "progress_v1.json"
)

RUN_METADATA_PATH = (
    OUTPUT_DIR /
    "run_metadata_v1.json"
)

COMPLETION_PATH = (
    OUTPUT_DIR /
    "completion_v1.json"
)


# ============================================================
# Frozen hashes
# ============================================================

EXPECTED_PROTOCOL_SHA256 = (
    "b3fde7262e025ac7c0bb5c11750c9ee"
    "218ea7e7ffdb2e0b3130d795c4cc0dad2"
)

EXPECTED_POPULATION_SHA256 = (
    "76a409ea7ee097899cc091fb19ecddb8d"
    "95f2305d3167989730bf3f22abf204d"
)

EXPECTED_POPULATION_FREEZE_SHA256 = (
    "3a7de982ad4e9678c1b40ccb3ba1c02e"
    "c7b9cf29257242a56e4fdf9ec90eb2c4"
)

EXPECTED_U4_SHA256 = (
    "af50da2cf49268cc55dc33d44dad50c0"
    "cd5a4fa29cda6cad3c01cc0b4ae63f14"
)


# ============================================================
# Frozen execution semantics
# ============================================================

LAMBDAS = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
]

METHODS = [
    "WHOLE",
    "MN",
]

EXPECTED_N = 1000
EXPECTED_TOTAL_CONDITIONS = (
    EXPECTED_N
    *
    len(LAMBDAS)
    *
    len(METHODS)
)

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

# Frozen numeric implementation guardrails.
MAX_PLANNED_STRENGTH_ERROR = 1e-5
MAX_PLANNED_SUBSPACE_RESIDUAL = 1e-5

MAX_REALIZED_STRENGTH_ERROR = 0.03
MAX_REALIZED_MN_SUBSPACE_RESIDUAL = 0.05


# ============================================================
# Utilities
# ============================================================

def sha256_file(
    path: Path,
) -> str:

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
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path)
        +
        ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        f.write(
            text
        )

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
):

    atomic_write_text(
        path,
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
    )


def append_jsonl(
    path: Path,
    obj,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    line = (
        json.dumps(
            obj,
            ensure_ascii=False,
        )
        +
        "\n"
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as f:

        f.write(
            line
        )

        f.flush()

        os.fsync(
            f.fileno()
        )


def load_jsonl(
    path: Path,
):

    if not path.exists():
        return []

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            1,
        ):

            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(
                        line
                    )
                )

            except Exception as e:

                raise RuntimeError(
                    "Malformed JSONL at "
                    f"{path}:{line_no}"
                ) from e

    return rows


def close(
    a,
    b,
    *,
    rtol=1e-6,
    atol=1e-5,
):

    return math.isclose(
        float(a),
        float(b),
        rel_tol=rtol,
        abs_tol=atol,
    )


# ============================================================
# Frozen population / execution preflight
# ============================================================

def load_population():

    df = pd.read_csv(
        POPULATION_PATH,
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
            "prediction_normalized":
                "string",
            "structural_type":
                "string",
            "semantic_type":
                "string",
            "detailed_type":
                "string",
        },
    )

    if len(df) != EXPECTED_N:

        raise RuntimeError(
            f"Expected N={EXPECTED_N}, "
            f"got {len(df)}."
        )

    if (
        df[
            "image_id"
        ].nunique()
        != EXPECTED_N
    ):

        raise RuntimeError(
            "Frozen image IDs "
            "are not unique."
        )

    if (
        df[
            "question_id"
        ].nunique()
        != EXPECTED_N
    ):

        raise RuntimeError(
            "Frozen question IDs "
            "are not unique."
        )

    if "baseline_correct" in df.columns:

        vals = (
            df[
                "baseline_correct"
            ]
            .astype(str)
            .str.lower()
        )

        if not vals.isin(
            [
                "true",
                "1",
                "1.0",
            ]
        ).all():

            raise RuntimeError(
                "Frozen population contains "
                "a non-baseline-correct row."
            )

    if (
        df[
            "prediction_normalized"
        ]
        .fillna("")
        .tolist()
        !=
        df[
            "answer_normalized"
        ]
        .fillna("")
        .tolist()
    ):

        raise RuntimeError(
            "Frozen population is not "
            "exact-match baseline correct."
        )

    return df


def preflight():

    print(
        "===== C2 INTERVENTION PREFLIGHT ====="
    )

    required = [
        PROTOCOL_PATH,
        POPULATION_PATH,
        POPULATION_FREEZE_PATH,
        EXECUTION_FREEZE_PATH,
        U4_PATH,
        RUNNER_PATH,
    ]

    for path in required:

        if not path.is_file():

            raise RuntimeError(
                f"Missing required file: "
                f"{path}"
            )

    hashes = {
        "protocol":
            sha256_file(
                PROTOCOL_PATH
            ),

        "population":
            sha256_file(
                POPULATION_PATH
            ),

        "population_freeze":
            sha256_file(
                POPULATION_FREEZE_PATH
            ),

        "u4":
            sha256_file(
                U4_PATH
            ),

        "runner":
            sha256_file(
                RUNNER_PATH
            ),
    }

    print(
        "protocol SHA         :",
        hashes[
            "protocol"
        ],
    )

    print(
        "population SHA       :",
        hashes[
            "population"
        ],
    )

    print(
        "population freeze SHA:",
        hashes[
            "population_freeze"
        ],
    )

    print(
        "U4 SHA               :",
        hashes[
            "u4"
        ],
    )

    print(
        "runner SHA           :",
        hashes[
            "runner"
        ],
    )

    if (
        hashes[
            "protocol"
        ]
        !=
        EXPECTED_PROTOCOL_SHA256
    ):

        raise RuntimeError(
            "Protocol SHA mismatch."
        )

    if (
        hashes[
            "population"
        ]
        !=
        EXPECTED_POPULATION_SHA256
    ):

        raise RuntimeError(
            "Population SHA mismatch."
        )

    if (
        hashes[
            "population_freeze"
        ]
        !=
        EXPECTED_POPULATION_FREEZE_SHA256
    ):

        raise RuntimeError(
            "Population freeze SHA mismatch."
        )

    if (
        hashes[
            "u4"
        ]
        !=
        EXPECTED_U4_SHA256
    ):

        raise RuntimeError(
            "U4 SHA mismatch."
        )

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    freeze = json.loads(
        EXECUTION_FREEZE_PATH.read_text(
            encoding="utf-8"
        )
    )

    frozen_grid = [
        float(x)
        for x in
        protocol[
            "intervention_stress_test"
        ][
            "lambda_grid"
        ]
    ]

    if frozen_grid != LAMBDAS:

        raise RuntimeError(
            "Lambda grid differs "
            "from frozen protocol."
        )

    if (
        protocol[
            "intervention_stress_test"
        ][
            "proposal_controller_used"
        ]
        is not False
    ):

        raise RuntimeError(
            "Frozen protocol unexpectedly "
            "enables proposal controller."
        )

    if (
        protocol[
            "intervention_stress_test"
        ][
            "candidate_utility_router_used"
        ]
        is not False
    ):

        raise RuntimeError(
            "Frozen protocol unexpectedly "
            "enables utility router."
        )

    if (
        freeze[
            "runner_sha256"
        ]
        !=
        hashes[
            "runner"
        ]
    ):

        raise RuntimeError(
            "Execution freeze runner SHA "
            "does not match current runner."
        )

    if (
        freeze[
            "population_sha256"
        ]
        !=
        EXPECTED_POPULATION_SHA256
    ):

        raise RuntimeError(
            "Execution freeze population "
            "SHA mismatch."
        )

    if (
        freeze[
            "u4_sha256"
        ]
        !=
        EXPECTED_U4_SHA256
    ):

        raise RuntimeError(
            "Execution freeze U4 "
            "SHA mismatch."
        )

    if [
        float(x)
        for x in
        freeze[
            "lambda_grid"
        ]
    ] != LAMBDAS:

        raise RuntimeError(
            "Execution freeze lambda "
            "grid mismatch."
        )

    if (
        freeze[
            "method_order"
        ]
        != METHODS
    ):

        raise RuntimeError(
            "Execution freeze method "
            "order mismatch."
        )

    if int(
        freeze[
            "total_conditions"
        ]
    ) != EXPECTED_TOTAL_CONDITIONS:

        raise RuntimeError(
            "Execution freeze total "
            "condition count mismatch."
        )

    U = np.load(
        U4_PATH
    )

    if U.shape != (
        128,
        4,
    ):

        raise RuntimeError(
            f"Unexpected U4 shape: "
            f"{U.shape}"
        )

    population = (
        load_population()
    )

    print(
        "population N         :",
        len(
            population
        ),
    )

    print(
        "lambda grid          :",
        LAMBDAS,
    )

    print(
        "method order         :",
        METHODS,
    )

    print(
        "total conditions     :",
        EXPECTED_TOTAL_CONDITIONS,
    )

    print(
        "U4 shape             :",
        U.shape,
    )

    print(
        "C2 INTERVENTION PREFLIGHT: PASS"
    )

    return (
        population,
        U,
        hashes,
    )


# ============================================================
# Exact deterministic schedule
# ============================================================

def build_schedule(
    population,
):

    schedule = []

    condition_index = 0

    for population_index, row in (
        population.iterrows()
    ):

        for lam in LAMBDAS:

            for method in METHODS:

                condition_index += 1

                schedule.append(
                    {
                        "condition_index":
                            condition_index,

                        "population_index":
                            int(
                                population_index
                            ),

                        "candidate_rank":
                            int(
                                row[
                                    "candidate_rank"
                                ]
                            ),

                        "question_id":
                            str(
                                row[
                                    "question_id"
                                ]
                            ),

                        "lambda":
                            float(
                                lam
                            ),

                        "method":
                            method,
                    }
                )

    if (
        len(
            schedule
        )
        !=
        EXPECTED_TOTAL_CONDITIONS
    ):

        raise RuntimeError(
            "Schedule size mismatch."
        )

    return schedule


def validate_existing_prefix(
    existing,
    schedule,
):

    if len(
        existing
    ) > len(
        schedule
    ):

        raise RuntimeError(
            "Checkpoint longer than "
            "frozen schedule."
        )

    for i, result in enumerate(
        existing
    ):

        expected = schedule[
            i
        ]

        for key in [
            "condition_index",
            "population_index",
            "candidate_rank",
            "question_id",
            "method",
        ]:

            if (
                result[
                    key
                ]
                !=
                expected[
                    key
                ]
            ):

                raise RuntimeError(
                    "Checkpoint schedule "
                    f"mismatch at row {i+1}, "
                    f"field={key}."
                )

        if not close(
            result[
                "lambda"
            ],
            expected[
                "lambda"
            ],
            rtol=0.0,
            atol=1e-12,
        ):

            raise RuntimeError(
                "Checkpoint lambda "
                f"mismatch at row {i+1}."
            )


# ============================================================
# Per-generation invariant audit
# ============================================================

def validate_modifier_records(
    method,
    records,
):

    if not records:

        raise RuntimeError(
            f"{method}: hook "
            "was never called."
        )

    planned_strength = [
        float(
            r[
                "planned_strength_error"
            ]
        )
        for r in records
    ]

    realized_strength = [
        float(
            r[
                "realized_strength_error"
            ]
        )
        for r in records
    ]

    if max(
        planned_strength
    ) > MAX_PLANNED_STRENGTH_ERROR:

        raise RuntimeError(
            f"{method}: planned "
            "strength invariant failed."
        )

    if max(
        realized_strength
    ) > MAX_REALIZED_STRENGTH_ERROR:

        raise RuntimeError(
            f"{method}: realized "
            "strength invariant failed."
        )

    out = {
        "hook_calls":
            len(
                records
            ),

        "hook_token_lengths":
            [
                int(
                    r[
                        "T"
                    ]
                )
                for r in records
            ],

        "first_h_norm":
            float(
                records[
                    0
                ][
                    "h_norm"
                ]
            ),

        "first_target_shift":
            float(
                records[
                    0
                ][
                    "target_shift"
                ]
            ),

        "max_planned_strength_error":
            max(
                planned_strength
            ),

        "max_realized_strength_error":
            max(
                realized_strength
            ),

        "max_realized_vector_error":
            max(
                float(
                    r[
                        "realized_vector_error"
                    ]
                )
                for r in records
            ),
    }

    if method == "MN":

        planned_subspace = [
            float(
                r[
                    "planned_subspace_residual"
                ]
            )
            for r in records
        ]

        realized_subspace = [
            float(
                r[
                    "realized_subspace_residual"
                ]
            )
            for r in records
        ]

        if max(
            planned_subspace
        ) > MAX_PLANNED_SUBSPACE_RESIDUAL:

            raise RuntimeError(
                "MN planned subspace "
                "invariant failed."
            )

        if max(
            realized_subspace
        ) > MAX_REALIZED_MN_SUBSPACE_RESIDUAL:

            raise RuntimeError(
                "MN realized subspace "
                "invariant failed."
            )

        out[
            "max_planned_subspace_residual"
        ] = max(
            planned_subspace
        )

        out[
            "max_realized_subspace_residual"
        ] = max(
            realized_subspace
        )

        out[
            "first_rho"
        ] = float(
            records[
                0
            ][
                "rho"
            ]
        )

    else:

        out[
            "max_planned_subspace_residual"
        ] = None

        out[
            "max_realized_subspace_residual"
        ] = None

        out[
            "first_rho"
        ] = None

    return out


# ============================================================
# Run one frozen condition
# ============================================================

def run_condition(
    *,
    model,
    processor,
    inputs,
    generation_config,
    row,
    method,
    lam,
    U,
):

    input_len = int(
        inputs[
            "input_ids"
        ].shape[
            1
        ]
    )

    if method == "WHOLE":

        modifier = (
            AuditedWholeModifier(
                model=model,
                lam=lam,
            )
        )

    elif method == "MN":

        modifier = (
            AuditedMNModifier(
                model=model,
                lam=lam,
                U=U,
            )
        )

    else:

        raise RuntimeError(
            f"Unknown method: "
            f"{method}"
        )

    modifier.register()

    start = time.time()

    try:

        with torch.inference_mode():

            output_ids = (
                model.generate(
                    **inputs,
                    generation_config=
                        generation_config,
                )
            )

    finally:

        modifier.remove()

    elapsed = (
        time.time()
        -
        start
    )

    if (
        output_ids.ndim != 2
        or
        output_ids.shape[
            0
        ]
        != 1
    ):

        raise RuntimeError(
            "Unexpected generation "
            f"shape: "
            f"{tuple(output_ids.shape)}"
        )

    if (
        output_ids.shape[
            1
        ]
        <
        input_len
    ):

        raise RuntimeError(
            "Generated sequence shorter "
            "than prompt."
        )

    new_ids = (
        output_ids[
            0,
            input_len:
        ]
        .detach()
        .cpu()
    )

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
        row[
            "answer_normalized"
        ]
    )

    baseline_prediction = str(
        row[
            "prediction_normalized"
        ]
    )

    if (
        baseline_prediction
        != gold
    ):

        raise RuntimeError(
            "Frozen population row "
            "is not baseline-correct."
        )

    invariant = (
        validate_modifier_records(
            method,
            modifier.records,
        )
    )

    return {
        "question_id":
            str(
                row[
                    "question_id"
                ]
            ),

        "image_id":
            str(
                row[
                    "image_id"
                ]
            ),

        "candidate_rank":
            int(
                row[
                    "candidate_rank"
                ]
            ),

        "structural_type":
            str(
                row[
                    "structural_type"
                ]
            ),

        "semantic_type":
            str(
                row[
                    "semantic_type"
                ]
            ),

        "detailed_type":
            str(
                row[
                    "detailed_type"
                ]
            ),

        "method":
            method,

        "lambda":
            float(
                lam
            ),

        "gold_normalized":
            gold,

        "baseline_prediction_normalized":
            baseline_prediction,

        "raw_generation":
            raw_generation,

        "prediction_normalized":
            prediction,

        "intervention_correct":
            bool(
                prediction
                ==
                gold
            ),

        "break_from_baseline":
            bool(
                prediction
                !=
                gold
            ),

        "answer_flip":
            bool(
                prediction
                !=
                baseline_prediction
            ),

        "generated_token_ids":
            [
                int(x)
                for x in
                new_ids.tolist()
            ],

        "generated_token_count":
            int(
                len(
                    new_ids
                )
            ),

        "generation_seconds":
            float(
                elapsed
            ),

        **invariant,
    }


# ============================================================
# Final materialization
# ============================================================

def finalize(
    rows,
):

    if (
        len(
            rows
        )
        !=
        EXPECTED_TOTAL_CONDITIONS
    ):

        raise RuntimeError(
            "Cannot finalize incomplete "
            "condition set."
        )

    df = pd.DataFrame(
        rows
    )

    tmp = Path(
        str(
            RESULTS_CSV
        )
        +
        ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        lineterminator="\n",
    )

    os.replace(
        tmp,
        RESULTS_CSV,
    )

    completion = {
        "experiment":
            "off_target_selectivity_v1",

        "stage":
            "intervention_stress_test_v1",

        "complete":
            True,

        "population_n":
            EXPECTED_N,

        "lambda_grid":
            LAMBDAS,

        "methods":
            METHODS,

        "total_conditions":
            EXPECTED_TOTAL_CONDITIONS,

        "results_jsonl":
            str(
                RESULTS_JSONL
            ),

        "results_jsonl_sha256":
            sha256_file(
                RESULTS_JSONL
            ),

        "results_csv":
            str(
                RESULTS_CSV
            ),

        "results_csv_sha256":
            sha256_file(
                RESULTS_CSV
            ),

        "git_head_at_completion":
            git_head(),
    }

    atomic_write_json(
        COMPLETION_PATH,
        completion,
    )

    return completion


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "preflight",
            "formal",
        ],
        required=True,
    )

    args = parser.parse_args()

    print(
        "=" * 72
    )

    print(
        "AROMA C2 INTERVENTION STRESS TEST V1"
    )

    print(
        "=" * 72
    )

    (
        population,
        U,
        hashes,
    ) = preflight()

    schedule = (
        build_schedule(
            population
        )
    )

    if (
        args.mode
        ==
        "preflight"
    ):

        print()
        print(
            "Schedule first:"
        )

        print(
            schedule[
                0
            ]
        )

        print(
            "Schedule last :"
        )

        print(
            schedule[
                -1
            ]
        )

        print()

        print(
            "=" * 72
        )

        print(
            "C2 INTERVENTION RUNNER PREFLIGHT: PASS"
        )

        print(
            "NO MODEL LOADED"
        )

        print(
            "NO INTERVENTION OUTCOME GENERATED"
        )

        print(
            "=" * 72
        )

        return

    # --------------------------------------------------------
    # Formal-only repository integrity
    # --------------------------------------------------------

    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            str(
                RUNNER_PATH
            ),
            "manifests/"
            "off_target_selectivity_v1",
        ],
        text=True,
    )

    if status.strip():

        raise RuntimeError(
            "Tracked C2 runner/manifests "
            "are not clean:\n"
            +
            status
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = load_jsonl(
        RESULTS_JSONL
    )

    validate_existing_prefix(
        existing,
        schedule,
    )

    print()
    print(
        "===== RESUME STATUS ====="
    )

    print(
        "completed :",
        len(
            existing
        ),
        "/",
        EXPECTED_TOTAL_CONDITIONS,
    )

    if (
        len(
            existing
        )
        ==
        EXPECTED_TOTAL_CONDITIONS
    ):

        completion = finalize(
            existing
        )

        print(
            "Already complete."
        )

        print(
            json.dumps(
                completion,
                indent=2,
            )
        )

        return

    run_metadata = {
        "experiment":
            "off_target_selectivity_v1",

        "stage":
            "intervention_stress_test_v1",

        "runner_sha256":
            hashes[
                "runner"
            ],

        "execution_freeze_sha256":
            sha256_file(
                EXECUTION_FREEZE_PATH
            ),

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "population_sha256":
            EXPECTED_POPULATION_SHA256,

        "u4_sha256":
            EXPECTED_U4_SHA256,

        "model_revision":
            MODEL_REVISION,

        "population_n":
            EXPECTED_N,

        "lambda_grid":
            LAMBDAS,

        "method_order":
            METHODS,

        "schedule_order":
            (
                "population order -> "
                "lambda ascending -> "
                "WHOLE then MN"
            ),

        "total_conditions":
            EXPECTED_TOTAL_CONDITIONS,

        "numeric_guardrails": {
            "max_planned_strength_error":
                MAX_PLANNED_STRENGTH_ERROR,

            "max_planned_subspace_residual":
                MAX_PLANNED_SUBSPACE_RESIDUAL,

            "max_realized_strength_error":
                MAX_REALIZED_STRENGTH_ERROR,

            "max_realized_mn_subspace_residual":
                MAX_REALIZED_MN_SUBSPACE_RESIDUAL,
        },

        "git_head_at_start":
            git_head(),

        "resume_conditions_at_start":
            len(
                existing
            ),
    }

    atomic_write_json(
        RUN_METADATA_PATH,
        run_metadata,
    )

    # --------------------------------------------------------
    # Exact frozen model
    # --------------------------------------------------------

    (
        processor,
        model,
        input_device,
        generation_config,
    ) = load_model()

    start_idx = len(
        existing
    )

    # --------------------------------------------------------
    # Condition loop
    # --------------------------------------------------------

    current_population_index = None
    current_inputs = None

    for schedule_idx in range(
        start_idx,
        len(
            schedule
        ),
    ):

        spec = schedule[
            schedule_idx
        ]

        population_index = int(
            spec[
                "population_index"
            ]
        )

        row = population.iloc[
            population_index
        ]

        # Rebuild multimodal inputs only when
        # moving to a new frozen sample.
        if (
            current_population_index
            !=
            population_index
        ):

            current_inputs = (
                build_inputs(
                    processor,
                    row,
                    input_device,
                )
            )

            current_population_index = (
                population_index
            )

        result = run_condition(
            model=model,
            processor=processor,
            inputs=current_inputs,
            generation_config=
                generation_config,
            row=row,
            method=
                spec[
                    "method"
                ],
            lam=
                spec[
                    "lambda"
                ],
            U=U,
        )

        result[
            "condition_index"
        ] = int(
            spec[
                "condition_index"
            ]
        )

        result[
            "population_index"
        ] = population_index

        # ----------------------------------------------------
        # WHOLE ↔ MN first-call matched-budget check.
        #
        # Frozen schedule always runs WHOLE then MN
        # for the same sample/lambda.
        # ----------------------------------------------------

        if (
            result[
                "method"
            ]
            ==
            "MN"
        ):

            if existing:

                previous = (
                    existing[
                        -1
                    ]
                )

            else:

                previous = None

            if previous is None:

                raise RuntimeError(
                    "MN condition has no "
                    "preceding WHOLE record."
                )

            if (
                previous[
                    "method"
                ]
                !=
                "WHOLE"
                or
                int(
                    previous[
                        "candidate_rank"
                    ]
                )
                !=
                int(
                    result[
                        "candidate_rank"
                    ]
                )
                or
                not close(
                    previous[
                        "lambda"
                    ],
                    result[
                        "lambda"
                    ],
                    rtol=0.0,
                    atol=1e-12,
                )
            ):

                raise RuntimeError(
                    "MN preceding WHOLE pair "
                    "does not match."
                )

            if not close(
                previous[
                    "first_h_norm"
                ],
                result[
                    "first_h_norm"
                ],
            ):

                raise RuntimeError(
                    "WHOLE/MN first-call "
                    "h norm mismatch."
                )

            if not close(
                previous[
                    "first_target_shift"
                ],
                result[
                    "first_target_shift"
                ],
            ):

                raise RuntimeError(
                    "WHOLE/MN first-call "
                    "target displacement mismatch."
                )

            result[
                "paired_first_call_match"
            ] = True

        else:

            result[
                "paired_first_call_match"
            ] = None

        append_jsonl(
            RESULTS_JSONL,
            result,
        )

        existing.append(
            result
        )

        atomic_write_json(
            PROGRESS_PATH,
            {
                "completed_conditions":
                    len(
                        existing
                    ),

                "total_conditions":
                    EXPECTED_TOTAL_CONDITIONS,

                "population_index":
                    population_index,

                "candidate_rank":
                    result[
                        "candidate_rank"
                    ],

                "question_id":
                    result[
                        "question_id"
                    ],

                "lambda":
                    result[
                        "lambda"
                    ],

                "method":
                    result[
                        "method"
                    ],

                "hook_calls":
                    result[
                        "hook_calls"
                    ],

                "generation_seconds":
                    result[
                        "generation_seconds"
                    ],
            },
        )

        print(
            f"[{len(existing):5d}/"
            f"{EXPECTED_TOTAL_CONDITIONS}] "
            f"pop={population_index+1:4d}/"
            f"{EXPECTED_N} "
            f"rank={result['candidate_rank']:5d} "
            f"lambda={result['lambda']:.2f} "
            f"method={result['method']:5s} "
            f"hooks={result['hook_calls']:2d} "
            f"tokens="
            f"{result['generated_token_count']:2d} "
            f"gen_s="
            f"{result['generation_seconds']:.3f}",
            flush=True,
        )

    # --------------------------------------------------------
    # Final raw outcome freeze.
    # No endpoint statistics here.
    # --------------------------------------------------------

    completion = finalize(
        existing
    )

    print()
    print(
        "=" * 72
    )

    print(
        "C2 INTERVENTION RAW OUTCOMES COMPLETE"
    )

    print(
        "TOTAL CONDITIONS:",
        EXPECTED_TOTAL_CONDITIONS,
    )

    print(
        "JSONL SHA:",
        completion[
            "results_jsonl_sha256"
        ],
    )

    print(
        "CSV SHA  :",
        completion[
            "results_csv_sha256"
        ],
    )

    print(
        "NO STATISTICAL ENDPOINT ANALYSIS "
        "PERFORMED BY THIS RUNNER"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
