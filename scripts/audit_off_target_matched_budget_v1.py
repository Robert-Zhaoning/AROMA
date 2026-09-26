#!/usr/bin/env python3

"""
C2 matched-displacement implementation audit.

One frozen sample, lambda=1.0.

Purpose:
    Validate WHOLE and MN intervention mathematics at every actual
    L18H13 cross-attention o_proj invocation during free generation.

IMPORTANT:
    - intervention DOES occur internally;
    - generated behavioral outputs are deliberately not decoded,
      printed, stored, or analyzed;
    - this is an implementation invariant audit, not an experiment result.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from run_off_target_baseline_v1 import (
    IMAGE_ROOT,
    load_model,
    prepare_inputs,
)


# ============================================================
# Frozen constants
# ============================================================

POPULATION_PATH = Path(
    "manifests/off_target_selectivity_v1/"
    "baseline_correct_population_v1.csv"
)

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

EXPECTED_POP_SHA256 = (
    "76a409ea7ee097899cc091fb19ecddb8d95f2305d3167989730bf3f22abf204d"
)

EXPECTED_U4_SHA256 = (
    "af50da2cf49268cc55dc33d44dad50c0cd5a4fa29cda6cad3c01cc0b4ae63f14"
)

LAYER = 18
HEAD = 13
HEAD_DIM = 128

START = HEAD * HEAD_DIM
END = START + HEAD_DIM

LAMBDA = 1.0

# Frozen before observing this audit's results.
MAX_PLANNED_REL_ERROR = 1e-5
MAX_PLANNED_SUBSPACE_RESIDUAL = 1e-5

MAX_REALIZED_STRENGTH_REL_ERROR = 0.03
MAX_REALIZED_MN_SUBSPACE_RESIDUAL = 0.05


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


def relative_norm_error(
    actual,
    target,
):
    denom = torch.clamp(
        torch.linalg.vector_norm(
            target,
            dim=-1,
        ),
        min=1e-12,
    )

    num = torch.linalg.vector_norm(
        actual - target,
        dim=-1,
    )

    return num / denom


def subspace_residual(
    delta,
    U,
):
    projected = (
        (delta @ U)
        @ U.T
    )

    residual = (
        delta
        - projected
    )

    denom = torch.clamp(
        torch.linalg.vector_norm(
            delta,
            dim=-1,
        ),
        min=1e-12,
    )

    return (
        torch.linalg.vector_norm(
            residual,
            dim=-1,
        )
        /
        denom
    )


# ============================================================
# Audited WHOLE modifier
# ============================================================

class AuditedWholeModifier:

    def __init__(
        self,
        model,
        lam,
    ):
        self.lam = float(lam)

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

        if int(
            self.o_proj.in_features
        ) != 4096:
            raise RuntimeError(
                "Unexpected WHOLE "
                "o_proj input dimension."
            )

        self.handle = None
        self.records = []

    def hook(
        self,
        module,
        inputs,
    ):
        if not inputs:
            raise RuntimeError(
                "Empty WHOLE hook inputs."
            )

        x = inputs[0]

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
                keepdim=True,
            )
        )

        target_norm = (
            self.lam
            *
            h_norm
        )

        # Exact C2 WHOLE semantics:
        #
        # H'_t = (1 + lambda) H_t
        #
        # This intentionally mirrors the validated
        # HeadGainModifier implementation.
        H_new = (
            H
            *
            (
                1.0
                +
                self.lam
            )
        )

        # Mathematical/intended pooled displacement.
        intended_delta = (
            self.lam
            *
            h
        )

        intended_norm = (
            torch.linalg
            .vector_norm(
                intended_delta,
                dim=-1,
                keepdim=True,
            )
        )

        planned_strength_error = (
            torch.abs(
                intended_norm
                -
                target_norm
            )
            /
            torch.clamp(
                target_norm,
                min=1e-12,
            )
        )

        # Realized displacement after model-dtype arithmetic.
        realized_delta = (
            (
                H_new.float()
                -
                H.float()
            )
            .mean(
                dim=1
            )
        )

        realized_norm = (
            torch.linalg
            .vector_norm(
                realized_delta,
                dim=-1,
                keepdim=True,
            )
        )

        realized_strength_error = (
            torch.abs(
                realized_norm
                -
                target_norm
            )
            /
            torch.clamp(
                target_norm,
                min=1e-12,
            )
        )

        realized_vector_error = (
            relative_norm_error(
                realized_delta,
                intended_delta,
            )
        )

        rec = {
            "call":
                len(
                    self.records
                )
                + 1,

            "T":
                int(
                    H.shape[1]
                ),

            "h_norm":
                float(
                    h_norm[
                        0,
                        0
                    ].item()
                ),

            "target_shift":
                float(
                    target_norm[
                        0,
                        0
                    ].item()
                ),

            "realized_shift":
                float(
                    realized_norm[
                        0,
                        0
                    ].item()
                ),

            "planned_strength_error":
                float(
                    planned_strength_error[
                        0,
                        0
                    ].item()
                ),

            "realized_strength_error":
                float(
                    realized_strength_error[
                        0,
                        0
                    ].item()
                ),

            "realized_vector_error":
                float(
                    realized_vector_error[
                        0
                    ].item()
                ),
        }

        self.records.append(
            rec
        )

        x_new = x.clone()

        x_new[
            ...,
            START:END
        ] = H_new

        if len(inputs) == 1:
            return (
                x_new,
            )

        return (
            x_new,
            *inputs[1:],
        )

    def register(self):
        self.records = []

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
# Audited MN modifier
# ============================================================

class AuditedMNModifier:

    def __init__(
        self,
        model,
        lam,
        U,
    ):
        self.lam = float(lam)

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

        if int(
            self.o_proj.in_features
        ) != 4096:
            raise RuntimeError(
                "Unexpected MN "
                "o_proj input dimension."
            )

        U = np.asarray(
            U,
            dtype=np.float32,
        )

        if U.shape != (
            HEAD_DIM,
            4,
        ):
            raise RuntimeError(
                f"Unexpected U shape: "
                f"{U.shape}"
            )

        self.U = torch.tensor(
            U,
            dtype=torch.float32,
            device=
                self.o_proj
                .weight
                .device,
        )

        self.handle = None
        self.records = []

    def hook(
        self,
        module,
        inputs,
    ):
        if not inputs:
            raise RuntimeError(
                "Empty MN hook inputs."
            )

        x = inputs[0]

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
            projected_norm
            .min()
            .item()
        ) <= 1e-12:
            raise RuntimeError(
                "MN projected norm "
                "too small."
            )

        direction = (
            projected
            /
            projected_norm
        )

        # Exact C2 MN semantics:
        #
        # delta =
        # lambda ||h||
        # P_U h / ||P_U h||
        intended_delta = (
            self.lam
            *
            h_norm
            *
            direction
        )

        intended_norm = (
            torch.linalg
            .vector_norm(
                intended_delta,
                dim=-1,
                keepdim=True,
            )
        )

        target_norm = (
            self.lam
            *
            h_norm
        )

        planned_strength_error = (
            torch.abs(
                intended_norm
                -
                target_norm
            )
            /
            torch.clamp(
                target_norm,
                min=1e-12,
            )
        )

        planned_subspace = (
            subspace_residual(
                intended_delta,
                U,
            )
        )

        # Mirror validated MN implementation:
        # cast delta to H dtype before injection.
        H_new = (
            H
            +
            intended_delta
            .to(
                H.dtype
            )
            .unsqueeze(
                1
            )
        )

        realized_delta = (
            (
                H_new.float()
                -
                H.float()
            )
            .mean(
                dim=1
            )
        )

        realized_norm = (
            torch.linalg
            .vector_norm(
                realized_delta,
                dim=-1,
                keepdim=True,
            )
        )

        realized_strength_error = (
            torch.abs(
                realized_norm
                -
                target_norm
            )
            /
            torch.clamp(
                target_norm,
                min=1e-12,
            )
        )

        realized_vector_error = (
            relative_norm_error(
                realized_delta,
                intended_delta,
            )
        )

        realized_subspace = (
            subspace_residual(
                realized_delta,
                U,
            )
        )

        rec = {
            "call":
                len(
                    self.records
                )
                + 1,

            "T":
                int(
                    H.shape[1]
                ),

            "h_norm":
                float(
                    h_norm[
                        0,
                        0
                    ].item()
                ),

            "projected_norm":
                float(
                    projected_norm[
                        0,
                        0
                    ].item()
                ),

            "rho":
                float(
                    (
                        projected_norm
                        /
                        torch.clamp(
                            h_norm,
                            min=1e-12,
                        )
                    )[
                        0,
                        0
                    ].item()
                ),

            "target_shift":
                float(
                    target_norm[
                        0,
                        0
                    ].item()
                ),

            "realized_shift":
                float(
                    realized_norm[
                        0,
                        0
                    ].item()
                ),

            "planned_strength_error":
                float(
                    planned_strength_error[
                        0,
                        0
                    ].item()
                ),

            "planned_subspace_residual":
                float(
                    planned_subspace[
                        0
                    ].item()
                ),

            "realized_strength_error":
                float(
                    realized_strength_error[
                        0,
                        0
                    ].item()
                ),

            "realized_vector_error":
                float(
                    realized_vector_error[
                        0
                    ].item()
                ),

            "realized_subspace_residual":
                float(
                    realized_subspace[
                        0
                    ].item()
                ),
        }

        self.records.append(
            rec
        )

        x_new = x.clone()

        x_new[
            ...,
            START:END
        ] = H_new

        if len(inputs) == 1:
            return (
                x_new,
            )

        return (
            x_new,
            *inputs[1:],
        )

    def register(self):
        self.records = []

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
# Input construction
# ============================================================

def build_inputs(
    processor,
    row,
    input_device,
):
    image_path = (
        IMAGE_ROOT
        /
        f"{row.image_id}.jpg"
    )

    if not image_path.is_file():
        raise RuntimeError(
            f"Missing image: "
            f"{image_path}"
        )

    with Image.open(
        image_path
    ) as im:
        image = im.convert(
            "RGB"
        )

    (
        inputs,
        _,
        _,
    ) = prepare_inputs(
        processor,
        image,
        row.question,
    )

    moved = {}

    for key, value in inputs.items():

        if torch.is_tensor(
            value
        ):
            moved[key] = (
                value.to(
                    input_device
                )
            )

        else:
            moved[key] = value

    return moved


# ============================================================
# Generation solely to exercise real hook lifecycle
# ============================================================

def exercise_modifier(
    *,
    model,
    inputs,
    generation_config,
    modifier,
):
    modifier.register()

    try:

        with torch.inference_mode():

            # Deliberately discard generated IDs.
            #
            # This audit does NOT decode, save,
            # score, or analyze behavioral output.
            _ = model.generate(
                **inputs,
                generation_config=
                    generation_config,
            )

    finally:

        modifier.remove()


# ============================================================
# Invariant checks
# ============================================================

def audit_records(
    method,
    records,
):
    if not records:
        raise RuntimeError(
            f"{method}: hook never called."
        )

    print()
    print(
        f"===== {method} CALL AUDIT ====="
    )

    for r in records:

        print(
            f"call={r['call']:2d} "
            f"T={r['T']:3d} "
            f"||h||={r['h_norm']:.8f} "
            f"target={r['target_shift']:.8f} "
            f"realized={r['realized_shift']:.8f} "
            f"planned_err="
            f"{r['planned_strength_error']:.3e} "
            f"realized_err="
            f"{r['realized_strength_error']:.3e}",
            end="",
        )

        if method == "MN":

            print(
                f" "
                f"planned_subspace="
                f"{r['planned_subspace_residual']:.3e} "
                f"realized_subspace="
                f"{r['realized_subspace_residual']:.3e} "
                f"rho={r['rho']:.6f}"
            )

        else:

            print(
                f" "
                f"vector_err="
                f"{r['realized_vector_error']:.3e}"
            )

        if (
            r[
                "planned_strength_error"
            ]
            >
            MAX_PLANNED_REL_ERROR
        ):
            raise RuntimeError(
                f"{method}: planned strength "
                "invariant failed."
            )

        if (
            r[
                "realized_strength_error"
            ]
            >
            MAX_REALIZED_STRENGTH_REL_ERROR
        ):
            raise RuntimeError(
                f"{method}: realized "
                "strength error exceeds "
                f"{MAX_REALIZED_STRENGTH_REL_ERROR:.1%}."
            )

        if method == "MN":

            if (
                r[
                    "planned_subspace_residual"
                ]
                >
                MAX_PLANNED_SUBSPACE_RESIDUAL
            ):
                raise RuntimeError(
                    "MN planned displacement "
                    "left U4 subspace."
                )

            if (
                r[
                    "realized_subspace_residual"
                ]
                >
                MAX_REALIZED_MN_SUBSPACE_RESIDUAL
            ):
                raise RuntimeError(
                    "MN realized BF16 "
                    "displacement subspace "
                    "residual exceeds "
                    f"{MAX_REALIZED_MN_SUBSPACE_RESIDUAL:.1%}."
                )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "C2 — MATCHED-DISPLACEMENT INVARIANT AUDIT"
    )
    print(
        "ONE FROZEN SAMPLE / LAMBDA = 1.0"
    )
    print(
        "NO BEHAVIORAL OUTPUT IS DECODED OR ANALYZED"
    )
    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Frozen artifacts
    # --------------------------------------------------------

    pop_sha = sha256_file(
        POPULATION_PATH
    )

    U_sha = sha256_file(
        U4_PATH
    )

    print(
        "population SHA:",
        pop_sha,
    )

    print(
        "U4 SHA        :",
        U_sha,
    )

    if (
        pop_sha
        != EXPECTED_POP_SHA256
    ):
        raise RuntimeError(
            "Population SHA mismatch."
        )

    if (
        U_sha
        != EXPECTED_U4_SHA256
    ):
        raise RuntimeError(
            "U4 SHA mismatch."
        )

    if (
        START != 1664
        or END != 1792
    ):
        raise RuntimeError(
            "Unexpected L18H13 slice."
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

    population = pd.read_csv(
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
        },
    )

    if len(
        population
    ) != 1000:
        raise RuntimeError(
            "Frozen population is not N=1000."
        )

    row = population.iloc[
        0
    ]

    print()
    print(
        "candidate_rank:",
        int(
            row.candidate_rank
        ),
    )

    print(
        "question_id   :",
        row.question_id,
    )

    print(
        "lambda        :",
        LAMBDA,
    )

    # --------------------------------------------------------
    # Exact model
    # --------------------------------------------------------

    (
        processor,
        model,
        input_device,
        generation_config,
    ) = load_model()

    inputs = build_inputs(
        processor,
        row,
        input_device,
    )

    # --------------------------------------------------------
    # WHOLE
    # --------------------------------------------------------

    whole = AuditedWholeModifier(
        model=model,
        lam=LAMBDA,
    )

    exercise_modifier(
        model=model,
        inputs=inputs,
        generation_config=
            generation_config,
        modifier=whole,
    )

    audit_records(
        "WHOLE",
        whole.records,
    )

    # --------------------------------------------------------
    # MN
    # --------------------------------------------------------

    mn = AuditedMNModifier(
        model=model,
        lam=LAMBDA,
        U=U,
    )

    exercise_modifier(
        model=model,
        inputs=inputs,
        generation_config=
            generation_config,
        modifier=mn,
    )

    audit_records(
        "MN",
        mn.records,
    )

    # --------------------------------------------------------
    # First-call matched-budget audit
    #
    # Both methods see the identical unperturbed prefill state
    # before any prior intervention can alter autoregressive
    # trajectory.
    # --------------------------------------------------------

    w0 = whole.records[0]
    m0 = mn.records[0]

    print()
    print(
        "===== FIRST-CALL CROSS-METHOD AUDIT ====="
    )

    print(
        "WHOLE T:",
        w0["T"],
    )

    print(
        "MN T   :",
        m0["T"],
    )

    print(
        "WHOLE ||h||:",
        w0["h_norm"],
    )

    print(
        "MN ||h||   :",
        m0["h_norm"],
    )

    if (
        w0["T"]
        != m0["T"]
    ):
        raise RuntimeError(
            "First-call token lengths differ."
        )

    if not math.isclose(
        w0["h_norm"],
        m0["h_norm"],
        rel_tol=1e-6,
        abs_tol=1e-5,
    ):
        raise RuntimeError(
            "WHOLE/MN first-call baseline "
            "h norms differ."
        )

    if not math.isclose(
        w0["target_shift"],
        m0["target_shift"],
        rel_tol=1e-6,
        abs_tol=1e-5,
    ):
        raise RuntimeError(
            "WHOLE/MN first-call matched "
            "budget target differs."
        )

    print(
        "identical pre-intervention "
        "first-call state: PASS"
    )

    print(
        "matched first-call target "
        "displacement: PASS"
    )

    print()
    print(
        "WHOLE hook calls:",
        len(
            whole.records
        ),
    )

    print(
        "WHOLE T sequence:",
        [
            r["T"]
            for r
            in whole.records
        ],
    )

    print(
        "MN hook calls   :",
        len(
            mn.records
        ),
    )

    print(
        "MN T sequence   :",
        [
            r["T"]
            for r
            in mn.records
        ],
    )

    print()
    print(
        "=" * 72
    )
    print(
        "C2 MATCHED-DISPLACEMENT INVARIANT AUDIT: PASS"
    )
    print(
        "WHOLE: H'_t = (1 + lambda) H_t"
    )
    print(
        "MN: ||Delta h|| = lambda ||h|| "
        "with frozen U4 direction"
    )
    print(
        "AUTOREGRESSIVE HOOK CALLS AUDITED"
    )
    print(
        "NO BEHAVIORAL OUTPUT WAS DECODED OR ANALYZED"
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
