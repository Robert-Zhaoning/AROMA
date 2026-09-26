#!/usr/bin/env python3

"""
C2 observational audit of L18H13 cross-attention o_proj hook lifecycle
during autoregressive free generation.

IMPORTANT:
- baseline generation only
- observational pre-hook only
- no tensor modification
- no MN intervention
- no WHOLE intervention
- no lambda outcome
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from run_off_target_baseline_v1 import (
    load_model,
    run_one,
)


# ============================================================
# Frozen identifiers
# ============================================================

POPULATION_PATH = Path(
    "manifests/off_target_selectivity_v1/"
    "baseline_correct_population_v1.csv"
)

BASELINE_SCAN_PATH = Path(
    "outputs/off_target_selectivity_v1/"
    "baseline_population_v1/"
    "baseline_scan_v1.jsonl"
)

EXPECTED_POPULATION_SHA256 = (
    "76a409ea7ee097899cc091fb19ecddb8d95f2305d3167989730bf3f22abf204d"
)

EXPECTED_BASELINE_SCAN_SHA256 = (
    "426c3b8df67f308bf2c28acb2c64db1d2f32edacec3121911121d3c97594dafb"
)

LAYER = 18
HEAD = 13
HEAD_DIM = 128

START = HEAD * HEAD_DIM
END = START + HEAD_DIM

EXPECTED_START = 1664
EXPECTED_END = 1792


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


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


# ============================================================
# Frozen asset audit
# ============================================================

def audit_assets():
    print(
        "===== FROZEN ASSET AUDIT ====="
    )

    if not POPULATION_PATH.is_file():
        raise RuntimeError(
            f"Missing population: "
            f"{POPULATION_PATH}"
        )

    if not BASELINE_SCAN_PATH.is_file():
        raise RuntimeError(
            f"Missing baseline scan: "
            f"{BASELINE_SCAN_PATH}"
        )

    pop_sha = sha256_file(
        POPULATION_PATH
    )

    scan_sha = sha256_file(
        BASELINE_SCAN_PATH
    )

    print(
        "population SHA:",
        pop_sha,
    )

    print(
        "scan SHA      :",
        scan_sha,
    )

    if (
        pop_sha
        != EXPECTED_POPULATION_SHA256
    ):
        raise RuntimeError(
            "Frozen population SHA mismatch."
        )

    if (
        scan_sha
        != EXPECTED_BASELINE_SCAN_SHA256
    ):
        raise RuntimeError(
            "Frozen baseline scan SHA mismatch."
        )

    if START != EXPECTED_START:
        raise RuntimeError(
            f"Unexpected START={START}"
        )

    if END != EXPECTED_END:
        raise RuntimeError(
            f"Unexpected END={END}"
        )

    print(
        "L18H13 slice  :",
        f"[{START}:{END}]",
    )

    print(
        "asset audit   : PASS"
    )


# ============================================================
# Main
# ============================================================

def main():
    print(
        "=" * 72
    )
    print(
        "C2 — L18H13 GENERATION HOOK LIFECYCLE AUDIT"
    )
    print(
        "OBSERVATIONAL ONLY / NO ACTIVATION MODIFICATION"
    )
    print(
        "=" * 72
    )

    audit_assets()

    # --------------------------------------------------------
    # Frozen population first sample
    # --------------------------------------------------------

    population = pd.read_csv(
        POPULATION_PATH,
        dtype={
            "question_id": "string",
            "image_id": "string",
            "question": "string",
            "answer": "string",
            "answer_normalized": "string",
            "selection_hash": "string",
            "structural_type": "string",
            "semantic_type": "string",
            "detailed_type": "string",
        },
    )

    if len(population) != 1000:
        raise RuntimeError(
            f"Expected N=1000, "
            f"got {len(population)}."
        )

    row = population.iloc[0]

    candidate_rank = int(
        row["candidate_rank"]
    )

    print()
    print(
        "===== FROZEN SAMPLE ====="
    )
    print(
        "candidate_rank:",
        candidate_rank,
    )
    print(
        "question_id   :",
        row["question_id"],
    )
    print(
        "image_id      :",
        row["image_id"],
    )
    print(
        "question      :",
        row["question"],
    )
    print(
        "gold          :",
        row["answer_normalized"],
    )

    # --------------------------------------------------------
    # Frozen baseline reference
    # --------------------------------------------------------

    baseline_scan = load_jsonl(
        BASELINE_SCAN_PATH
    )

    reference_rows = [
        r
        for r in baseline_scan
        if int(
            r["candidate_rank"]
        ) == candidate_rank
    ]

    if len(reference_rows) != 1:
        raise RuntimeError(
            "Frozen baseline reference "
            "not uniquely found."
        )

    reference = reference_rows[0]

    if not bool(
        reference["baseline_correct"]
    ):
        raise RuntimeError(
            "Population sample is not "
            "baseline-correct in frozen scan."
        )

    print()
    print(
        "Frozen raw generation:",
        repr(
            reference[
                "raw_generation"
            ]
        ),
    )
    print(
        "Frozen token IDs     :",
        reference[
            "generated_token_ids"
        ],
    )

    # --------------------------------------------------------
    # Exact baseline model load
    # --------------------------------------------------------

    (
        processor,
        model,
        input_device,
        generation_config,
    ) = load_model()

    # --------------------------------------------------------
    # Resolve exact intervention module
    # --------------------------------------------------------

    layer = (
        model
        .model
        .language_model
        .layers[
            LAYER
        ]
    )

    o_proj = (
        layer
        .cross_attn
        .o_proj
    )

    if int(
        o_proj.in_features
    ) != 4096:
        raise RuntimeError(
            "Unexpected o_proj input "
            f"dimension: "
            f"{o_proj.in_features}"
        )

    records = []

    # --------------------------------------------------------
    # OBSERVATIONAL hook.
    #
    # CRITICAL:
    # Returning None means inputs are left unchanged.
    # No clone, assignment, multiplication, addition,
    # projection, MN direction, or gain is performed.
    # --------------------------------------------------------

    def observational_hook(
        module,
        inputs,
    ):
        if not inputs:
            raise RuntimeError(
                "Empty o_proj hook inputs."
            )

        x = inputs[0]

        if not torch.is_tensor(x):
            raise RuntimeError(
                "o_proj input is not tensor."
            )

        if x.ndim != 3:
            raise RuntimeError(
                "Unexpected o_proj input rank: "
                f"{tuple(x.shape)}"
            )

        if int(
            x.shape[-1]
        ) != 4096:
            raise RuntimeError(
                "Unexpected hidden dimension: "
                f"{tuple(x.shape)}"
            )

        H = x[
            ...,
            START:END
        ]

        h = (
            H
            .float()
            .mean(
                dim=1
            )
        )

        h_norm = (
            torch.linalg
            .vector_norm(
                h,
                dim=-1,
            )
        )

        rec = {
            "call":
                len(records) + 1,

            "x_shape":
                tuple(
                    int(v)
                    for v
                    in x.shape
                ),

            "H_shape":
                tuple(
                    int(v)
                    for v
                    in H.shape
                ),

            "token_length":
                int(
                    H.shape[1]
                ),

            "dtype":
                str(
                    x.dtype
                ),

            "device":
                str(
                    x.device
                ),

            "h_norm":
                float(
                    h_norm[0]
                    .item()
                ),
        }

        records.append(
            rec
        )

        print(
            f"HOOK call={rec['call']:2d} "
            f"x={rec['x_shape']} "
            f"H={rec['H_shape']} "
            f"T={rec['token_length']} "
            f"||h||={rec['h_norm']:.8f}",
            flush=True,
        )

        # OBSERVATIONAL ONLY.
        return None

    handle = (
        o_proj
        .register_forward_pre_hook(
            observational_hook
        )
    )

    try:
        rerun = run_one(
            row=row,
            processor=processor,
            model=model,
            input_device=input_device,
            generation_config=
                generation_config,
        )

    finally:
        handle.remove()

    # --------------------------------------------------------
    # Exact deterministic reproduction audit
    # --------------------------------------------------------

    print()
    print(
        "===== REPRODUCTION AUDIT ====="
    )

    print(
        "rerun raw generation:",
        repr(
            rerun[
                "raw_generation"
            ]
        ),
    )

    print(
        "rerun token IDs      :",
        rerun[
            "generated_token_ids"
        ],
    )

    print(
        "rerun normalized     :",
        repr(
            rerun[
                "prediction_normalized"
            ]
        ),
    )

    if (
        rerun[
            "generated_token_ids"
        ]
        !=
        reference[
            "generated_token_ids"
        ]
    ):
        raise RuntimeError(
            "OBSERVATIONAL HOOK CHANGED "
            "GENERATED TOKEN IDS."
        )

    if (
        rerun[
            "raw_generation"
        ]
        !=
        reference[
            "raw_generation"
        ]
    ):
        raise RuntimeError(
            "OBSERVATIONAL HOOK CHANGED "
            "RAW GENERATION."
        )

    if (
        rerun[
            "prediction_normalized"
        ]
        !=
        reference[
            "prediction_normalized"
        ]
    ):
        raise RuntimeError(
            "OBSERVATIONAL HOOK CHANGED "
            "NORMALIZED PREDICTION."
        )

    if (
        bool(
            rerun[
                "baseline_correct"
            ]
        )
        !=
        bool(
            reference[
                "baseline_correct"
            ]
        )
    ):
        raise RuntimeError(
            "OBSERVATIONAL HOOK CHANGED "
            "CORRECTNESS."
        )

    if len(records) <= 0:
        raise RuntimeError(
            "L18H13 o_proj hook "
            "was never called."
        )

    # --------------------------------------------------------
    # Lifecycle summary
    # --------------------------------------------------------

    token_lengths = [
        r[
            "token_length"
        ]
        for r in records
    ]

    print()
    print(
        "===== HOOK LIFECYCLE SUMMARY ====="
    )

    print(
        "hook calls        :",
        len(records),
    )

    print(
        "token lengths     :",
        token_lengths,
    )

    print(
        "unique lengths    :",
        sorted(
            set(
                token_lengths
            )
        ),
    )

    print(
        "first call T      :",
        token_lengths[0],
    )

    if len(
        token_lengths
    ) > 1:

        print(
            "later call Ts     :",
            token_lengths[1:],
        )

    print()
    print(
        "=" * 72
    )
    print(
        "OBSERVATIONAL GENERATION HOOK AUDIT: PASS"
    )
    print(
        "GENERATED TOKEN IDS EXACTLY MATCH FROZEN BASELINE"
    )
    print(
        "NO ACTIVATION MODIFICATION WAS PERFORMED"
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
