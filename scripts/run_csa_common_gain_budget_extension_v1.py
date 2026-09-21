#!/usr/bin/env python3

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

sys.path.insert(0, "scripts")

from run_l18h13_gain_all300 import (
    MODEL_ID,
    LAYER,
    HEAD,
    prepare_inputs,
    move_inputs,
    HeadGainModifier,
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)


# ============================================================
# COMMON ONE-PARAMETER GAIN BUDGET
# ============================================================

STRENGTHS = [
    2.75,
    3.00,
    3.25,
    3.50,
    4.00,
]

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

EXPECTED_N = 653
EXPECTED_UPWARD_N = 629

DUMMY_GT = 1

START = 1664
END = 1792


# ============================================================
# INPUTS
# ============================================================

V3_RESULTS = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller/"
    "v3_final_results.csv"
)

IMAGE_DIR = Path(
    "data/proc_count_causal_v3/images"
)

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

STRENGTH_CSV = Path(
    "outputs/aroma2/"
    "csa_actuation_strength_v1/"
    "actuation_strength.csv"
)

B2_CSV = Path(
    "outputs/aroma2/"
    "csa_gate_b2_mn_v1/"
    "gate_b2_mn_results.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

OUT_DIR = Path(
    "outputs/aroma2/"
    "csa_common_gain_budget_extension_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = (
    OUT_DIR
    / "partial_results.jsonl"
)

SUMMARY_CSV = (
    OUT_DIR
    / "common_gain_summary.csv"
)

RESULTS_CSV = (
    OUT_DIR
    / "common_gain_results.csv"
)

RESULT_JSON = (
    OUT_DIR
    / "common_gain_result.json"
)


# ============================================================
# HELPERS
# ============================================================

def skey(s):
    return (
        f"s{s:.2f}"
        .replace(".", "p")
    )


def append_jsonl(path, obj):
    with Path(path).open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write(
            json.dumps(
                obj,
                sort_keys=True,
            )
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path):
    path = Path(path)

    if not path.exists():
        return []

    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def resolve_image(sample_id):
    candidates = [
        p
        for p in IMAGE_DIR.glob(
            str(sample_id) + ".*"
        )
        if p.suffix.lower()
        in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"{sample_id}: expected one image, "
            f"got {candidates}"
        )

    return candidates[0]


# ============================================================
# SHARED SUBSPACE MODIFIER
# ============================================================

class SubspaceModifier:

    def __init__(
        self,
        model,
        alpha,
        U,
        mode,
        scalar,
    ):
        layer = (
            model
            .model
            .language_model
            .layers[LAYER]
        )

        self.o_proj = (
            layer
            .cross_attn
            .o_proj
        )

        self.alpha = float(alpha)
        self.mode = str(mode)
        self.scalar = float(scalar)

        self.U = torch.tensor(
            np.asarray(
                U,
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )

        self.handle = None
        self.calls = 0

        self.last_rho = None
        self.last_shift_ratio = None


    def hook(
        self,
        module,
        inputs,
    ):
        if not inputs:
            raise RuntimeError(
                "Empty hook inputs"
            )

        x = inputs[0]

        H = x[
            ...,
            START:END
        ]

        h = (
            H
            .float()
            .mean(dim=1)
        )

        U = self.U.to(
            device=H.device
        )

        projected = (
            (h @ U)
            @ U.T
        )

        h_norm = (
            torch.linalg
            .vector_norm(
                h,
                dim=-1,
                keepdim=True,
            )
        )

        projected_norm = (
            torch.linalg
            .vector_norm(
                projected,
                dim=-1,
                keepdim=True,
            )
        )

        if float(
            projected_norm.min().item()
        ) <= 1e-12:
            raise RuntimeError(
                "Projected norm too small"
            )

        rho = (
            projected_norm
            /
            h_norm
        )

        # scalar means:
        #
        # fixed:
        #   gamma
        #
        # beta:
        #   beta = realized whole-relative
        #   displacement budget
        if self.mode == "fixed":
            delta = (
                self.scalar
                *
                (self.alpha - 1.0)
                *
                projected
            )

        elif self.mode == "beta":
            direction = (
                projected
                /
                projected_norm
            )

            delta = (
                self.scalar
                *
                (self.alpha - 1.0)
                *
                h_norm
                *
                direction
            )

        else:
            raise RuntimeError(
                f"Unknown mode: {self.mode}"
            )

        actual_norm = (
            torch.linalg
            .vector_norm(
                delta,
                dim=-1,
                keepdim=True,
            )
        )

        target_whole_norm = (
            abs(
                self.alpha - 1.0
            )
            *
            h_norm
        )

        ratio = (
            actual_norm
            /
            torch.clamp(
                target_whole_norm,
                min=1e-12,
            )
        )

        self.last_rho = float(
            rho[0, 0].item()
        )

        self.last_shift_ratio = float(
            ratio[0, 0].item()
        )

        H_new = (
            H
            +
            delta
            .to(H.dtype)
            .unsqueeze(1)
        )

        x_new = x.clone()

        x_new[
            ...,
            START:END
        ] = H_new

        self.calls += 1

        if len(inputs) == 1:
            return (x_new,)

        return (
            x_new,
            *inputs[1:],
        )


    def register(self):
        self.calls = 0

        self.handle = (
            self.o_proj
            .register_forward_pre_hook(
                self.hook
            )
        )


    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ============================================================
# LOAD DEVELOPMENT POPULATION
# ============================================================

print("=" * 96)
print(
    "COMMON GAIN-BUDGET "
    "DEVELOPMENT SCREEN"
)
print("=" * 96)

for p in [
    V3_RESULTS,
    U4_PATH,
    STRENGTH_CSV,
    B2_CSV,
]:
    if not p.exists():
        raise RuntimeError(
            f"Missing {p}"
        )


v3 = pd.read_csv(
    V3_RESULTS
)

v3[
    "selected_alpha"
] = (
    v3[
        "selected_alpha"
    ].astype(float)
)

data = (
    v3[
        ~np.isclose(
            v3[
                "selected_alpha"
            ],
            1.0,
        )
    ]
    .copy()
    .sort_values(
        "sample_id"
    )
    .reset_index(
        drop=True
    )
)

if len(data) != EXPECTED_N:
    raise RuntimeError(
        f"N mismatch: {len(data)}"
    )

upward_mask = (
    data[
        "selected_alpha"
    ].astype(float)
    > 1.0
)

if int(upward_mask.sum()) != EXPECTED_UPWARD_N:
    raise RuntimeError(
        "Upward N mismatch"
    )


# ============================================================
# ARCHIVED REFERENCE
# ============================================================

b2 = pd.read_csv(
    B2_CSV
)

b2[
    "sample_id"
] = (
    b2[
        "sample_id"
    ].astype(str)
)

archive = (
    data[
        [
            "sample_id",
            "ground_truth",
            "selected_alpha",
        ]
    ]
    .copy()
)

archive[
    "sample_id"
] = (
    archive[
        "sample_id"
    ].astype(str)
)

archive = archive.merge(
    b2[
        [
            "sample_id",
            "baseline_prediction",
            "whole_prediction",
            "mn_csa_prediction",
        ]
    ],
    on="sample_id",
    validate="one_to_one",
)

if len(archive) != EXPECTED_N:
    raise RuntimeError(
        "Archive merge mismatch"
    )


# ============================================================
# REFERENCE RHO FOR FIXED CSA
#
# Use only upward actions because gain calibration
# is intentionally restricted to alpha > 1.
# ============================================================

strength = pd.read_csv(
    STRENGTH_CSV
)

strength[
    "sample_id"
] = (
    strength[
        "sample_id"
    ].astype(str)
)

rho_table = (
    archive[
        [
            "sample_id",
            "selected_alpha",
        ]
    ]
    .merge(
        strength[
            [
                "sample_id",
                "csa_to_whole_shift_ratio",
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    )
)

rho_ref = float(
    rho_table.loc[
        rho_table[
            "selected_alpha"
        ].astype(float)
        > 1.0,
        "csa_to_whole_shift_ratio",
    ]
    .median()
)

print(
    "Upward-only rho_ref:",
    rho_ref
)

print()
print(
    "Common strength grid:",
    STRENGTHS
)

print()
print(
    "Mapped fixed-CSA gammas:"
)

for s in STRENGTHS:
    print(
        f"  s={s:.2f} -> "
        f"gamma={s / rho_ref:.6f}"
    )


# ============================================================
# U4
# ============================================================

U4 = np.load(
    U4_PATH
).astype(
    np.float32
)

if U4.shape != (
    128,
    4,
):
    raise RuntimeError(
        f"Bad U4 shape: {U4.shape}"
    )


# ============================================================
# RESUME
# ============================================================

records = load_jsonl(
    PARTIAL
)

expected_ids = (
    archive[
        "sample_id"
    ]
    .astype(str)
    .tolist()
)

actual_ids = [
    str(x["sample_id"])
    for x in records
]

if (
    actual_ids
    !=
    expected_ids[
        :len(actual_ids)
    ]
):
    raise RuntimeError(
        "Resume order mismatch"
    )

print(
    "Resume:",
    len(records),
    "/",
    EXPECTED_N
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading processor...")

processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
)

numeral_ids = (
    numeral_token_ids(
        processor
    )
)

print("Loading model...")

model = (
    MllamaForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
)

model.eval()

for p in model.parameters():
    p.requires_grad_(False)


def get_inputs(sid):
    image = (
        Image.open(
            resolve_image(sid)
        )
        .convert("RGB")
    )

    inputs = prepare_inputs(
        processor,
        image,
    )

    return move_inputs(
        inputs,
        model,
    )


# ============================================================
# DEVELOPMENT SCREEN
# ============================================================

start_idx = len(records)

for idx in tqdm(
    range(
        start_idx,
        EXPECTED_N,
    ),
    initial=start_idx,
    total=EXPECTED_N,
):

    row = archive.iloc[idx]

    sid = str(
        row[
            "sample_id"
        ]
    )

    gt = int(
        row[
            "ground_truth"
        ]
    )

    alpha = float(
        row[
            "selected_alpha"
        ]
    )

    baseline_pred = int(
        row[
            "baseline_prediction"
        ]
    )

    inputs = get_inputs(
        sid
    )

    methods = {
        "whole": {},
        "fixed": {},
        "beta": {},
    }

    for s in STRENGTHS:

        key = skey(s)

        # ----------------------------------------------------
        # alpha=0:
        # preserve original action semantics.
        #
        # alpha>1:
        # apply common gain budget.
        # ----------------------------------------------------

        calibrate = (
            alpha > 1.0
        )

        # ====================================================
        # 1. TUNED WHOLE-HEAD
        # ====================================================

        lam = (
            float(s)
            if calibrate
            else 1.0
        )

        effective_alpha = (
            1.0
            +
            lam
            *
            (
                alpha
                - 1.0
            )
        )

        whole = HeadGainModifier(
            model=model,
            layer_idx=LAYER,
            head_idx=HEAD,
            alpha=effective_alpha,
        )

        whole.register()

        try:
            state = score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        finally:
            whole.remove()

        if whole.calls <= 0:
            raise RuntimeError(
                f"{sid}: whole hook failed"
            )

        pred = int(
            state[
                "best_numeral"
            ]
        )

        methods[
            "whole"
        ][
            key
        ] = {
            "s": float(s),
            "lambda": lam,
            "effective_alpha":
                effective_alpha,
            "prediction": pred,
        }


        # ====================================================
        # 2. FIXED-GAIN CSA
        # ====================================================

        gamma = (
            float(s)
            /
            rho_ref
            if calibrate
            else 1.0
        )

        fixed = SubspaceModifier(
            model=model,
            alpha=alpha,
            U=U4,
            mode="fixed",
            scalar=gamma,
        )

        fixed.register()

        try:
            state = score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        finally:
            fixed.remove()

        if fixed.calls <= 0:
            raise RuntimeError(
                f"{sid}: fixed hook failed"
            )

        pred = int(
            state[
                "best_numeral"
            ]
        )

        methods[
            "fixed"
        ][
            key
        ] = {
            "s": float(s),
            "gamma": gamma,
            "prediction": pred,
            "rho":
                fixed.last_rho,
            "shift_ratio":
                fixed.last_shift_ratio,
        }


        # ====================================================
        # 3. BETA-MN
        # ====================================================

        beta = (
            float(s)
            if calibrate
            else 1.0
        )

        mn = SubspaceModifier(
            model=model,
            alpha=alpha,
            U=U4,
            mode="beta",
            scalar=beta,
        )

        mn.register()

        try:
            state = score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        finally:
            mn.remove()

        if mn.calls <= 0:
            raise RuntimeError(
                f"{sid}: beta hook failed"
            )

        pred = int(
            state[
                "best_numeral"
            ]
        )

        methods[
            "beta"
        ][
            key
        ] = {
            "s": float(s),
            "beta": beta,
            "prediction": pred,
            "rho":
                mn.last_rho,
            "shift_ratio":
                mn.last_shift_ratio,
        }


    rec = {
        "sample_id": sid,
        "ground_truth": gt,
        "selected_alpha": alpha,
        "baseline_prediction":
            baseline_pred,
        "archived_whole_prediction":
            int(
                row[
                    "whole_prediction"
                ]
            ),
        "archived_mn_prediction":
            int(
                row[
                    "mn_csa_prediction"
                ]
            ),
        "methods": methods,
    }

    append_jsonl(
        PARTIAL,
        rec,
    )

    records.append(
        rec
    )

    del inputs

    if (
        len(records) == 1
        or
        len(records) % 10 == 0
        or
        len(records) == EXPECTED_N
    ):
        print()
        print(
            f"[{len(records)}/"
            f"{EXPECTED_N}] complete"
        )


# ============================================================
# EXTENSION INTEGRITY NOTE
# ============================================================

print()
print("=" * 96)
print("EXTENSION RUN")
print("=" * 96)
print(
    "s=1 reproduction is not rerun here. "
    "The parent common-gain screen already passed "
    "whole 653/653 and beta-MN 653/653."
)


# ============================================================
# SUMMARIZE
# ============================================================

flat = []
summary = []

for family in [
    "whole",
    "fixed",
    "beta",
]:

    for s in STRENGTHS:

        key = skey(s)

        full_correct = 0
        full_repairs = 0
        full_breaks = 0

        up_correct = 0
        up_repairs = 0
        up_breaks = 0
        up_n = 0

        for r in records:

            gt = int(
                r[
                    "ground_truth"
                ]
            )

            base = int(
                r[
                    "baseline_prediction"
                ]
            )

            alpha = float(
                r[
                    "selected_alpha"
                ]
            )

            pred = int(
                r[
                    "methods"
                ][
                    family
                ][
                    key
                ][
                    "prediction"
                ]
            )

            correct = (
                pred == gt
            )

            repair = (
                base != gt
                and
                correct
            )

            brk = (
                base == gt
                and
                not correct
            )

            full_correct += int(correct)
            full_repairs += int(repair)
            full_breaks += int(brk)

            if alpha > 1.0:

                up_n += 1
                up_correct += int(correct)
                up_repairs += int(repair)
                up_breaks += int(brk)

            flat.append(
                {
                    "sample_id":
                        r[
                            "sample_id"
                        ],
                    "family":
                        family,
                    "s":
                        float(s),
                    "selected_alpha":
                        alpha,
                    "ground_truth":
                        gt,
                    "baseline_prediction":
                        base,
                    "prediction":
                        pred,
                    "correct":
                        correct,
                    "repair":
                        repair,
                    "break":
                        brk,
                }
            )

        summary.append(
            {
                "family":
                    family,
                "s":
                    float(s),

                "upward_n":
                    up_n,
                "upward_correct":
                    up_correct,
                "upward_accuracy":
                    up_correct
                    /
                    up_n,
                "upward_repairs":
                    up_repairs,
                "upward_breaks":
                    up_breaks,
                "upward_net":
                    up_repairs
                    -
                    up_breaks,

                "full_n":
                    EXPECTED_N,
                "full_correct":
                    full_correct,
                "full_accuracy":
                    full_correct
                    /
                    EXPECTED_N,
                "full_repairs":
                    full_repairs,
                "full_breaks":
                    full_breaks,
                "full_net":
                    full_repairs
                    -
                    full_breaks,
            }
        )


summary_df = pd.DataFrame(
    summary
)

pd.DataFrame(
    flat
).to_csv(
    RESULTS_CSV,
    index=False,
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================
# SAME SELECTION RULE FOR ALL THREE FAMILIES
#
# Primary:
#   upward exact-count accuracy
#
# Tie:
#   fewer upward breaks
#
# Tie:
#   smaller strength s
# ============================================================

winners = {}

for family in [
    "whole",
    "fixed",
    "beta",
]:

    sub = (
        summary_df[
            summary_df[
                "family"
            ]
            ==
            family
        ]
        .copy()
        .sort_values(
            [
                "upward_correct",
                "upward_breaks",
                "s",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    winner = sub.iloc[0]

    winners[
        family
    ] = {
        "s":
            float(
                winner[
                    "s"
                ]
            ),
        "upward_correct":
            int(
                winner[
                    "upward_correct"
                ]
            ),
        "upward_accuracy":
            float(
                winner[
                    "upward_accuracy"
                ]
            ),
        "upward_repairs":
            int(
                winner[
                    "upward_repairs"
                ]
            ),
        "upward_breaks":
            int(
                winner[
                    "upward_breaks"
                ]
            ),
        "full_correct":
            int(
                winner[
                    "full_correct"
                ]
            ),
        "full_accuracy":
            float(
                winner[
                    "full_accuracy"
                ]
            ),
    }


result = {
    "stage":
        "common_gain_budget_development_v1",

    "status":
        "development_only",

    "n":
        EXPECTED_N,

    "upward_n":
        EXPECTED_UPWARD_N,

    "strength_grid":
        STRENGTHS,

    "rho_ref_upward_median":
        rho_ref,

    "selection_endpoint":
        "upward exact-count accuracy",

    "tie_breaks": [
        "fewer upward breaks",
        "smaller s",
    ],

    "alpha_zero_policy":
        (
            "alpha=0 keeps original "
            "uncalibrated action; "
            "gain tuning applies only "
            "to alpha>1"
        ),

    "winners":
        winners,

    "confirmation_data_touched":
        False,
}


RESULT_JSON.write_text(
    json.dumps(
        result,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 96)
print("COMMON GAIN-BUDGET SUMMARY")
print("=" * 96)

print(
    summary_df[
        [
            "family",
            "s",
            "upward_accuracy",
            "upward_correct",
            "upward_repairs",
            "upward_breaks",
            "full_accuracy",
        ]
    ]
    .to_string(
        index=False
    )
)

print()
print("=" * 96)
print("DEVELOPMENT WINNERS")
print("=" * 96)

for family, winner in winners.items():
    print()
    print(
        family,
        json.dumps(
            winner,
            indent=2,
        )
    )

print()
print(
    "Saved:",
    SUMMARY_CSV,
)

print(
    "Saved:",
    RESULTS_CSV,
)

print(
    "Saved:",
    RESULT_JSON,
)

print()
print(
    "DEVELOPMENT SCREEN COMPLETE"
)

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
