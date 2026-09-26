"""
Natural-UGR Stage-2 shared implementation v1.

Historical UGR definitions are extracted byte-for-byte in function/class
body from unified-geometry-router-v1 at implementation-freeze time.

This module centralizes:
- exact historical differentiable numeral-expectation target;
- exact historical L18H13 gradient hook;
- exact historical MN / SA DirectionModifier;
- exact historical candidate-outcome schema;
- live-tested Natural-UGR execution wrappers.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from run_l18h13_gain_all300 import (
    MODEL_ID,
    LAYER,
    HEAD,
    move_inputs,
    HeadGainModifier,
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

HIDDEN_SIZE = 4096
HEAD_DIM = 128

START = (
    HEAD
    *
    HEAD_DIM
)

END = (
    START
    +
    HEAD_DIM
)

DUMMY_GT = 1

GAINS = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    4.0,
]


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def differentiable_mu(
    logits,
    numeral_ids,
):
    last = logits[
        0,
        -1,
        :
    ].float()

    numeral_logits = torch.stack(
        [
            last[
                numeral_ids[n]
            ]
            for n in range(
                1,
                11,
            )
        ]
    ).to(
        torch.float64
    )

    probs = torch.softmax(
        numeral_logits,
        dim=0,
    )

    values = torch.arange(
        1,
        11,
        dtype=torch.float64,
        device=probs.device,
    )

    return (
        values
        *
        probs
    ).sum()


def make_hook(saved):

    def hook(
        module,
        hook_inputs,
    ):
        if not hook_inputs:
            raise RuntimeError(
                "o_proj hook received no inputs."
            )

        x = hook_inputs[
            0
        ]

        if x.shape[-1] != HIDDEN_SIZE:
            raise RuntimeError(
                "Unexpected o_proj input shape: "
                f"{tuple(x.shape)}"
            )

        H = x[
            ...,
            START:END
        ]

        h_base = (
            H.detach()
            .float()
            .mean(
                dim=1
            )
        )

        h_leaf = (
            h_base
            .clone()
            .requires_grad_(
                True
            )
        )

        delta = (
            h_leaf
            -
            h_base.detach()
        )

        H_new = (
            H.detach()
            +
            delta
            .to(
                H.dtype
            )
            .unsqueeze(
                1
            )
        )

        x_new = torch.cat(
            [
                x[
                    ...,
                    :START
                ].detach(),

                H_new,

                x[
                    ...,
                    END:
                ].detach(),
            ],
            dim=-1,
        )

        saved[
            "h_base"
        ] = h_base

        saved[
            "h_leaf"
        ] = h_leaf

        saved[
            "token_length"
        ] = int(
            H.shape[
                1
            ]
        )

        saved[
            "calls"
        ] = (
            saved.get(
                "calls",
                0,
            )
            +
            1
        )

        if len(
            hook_inputs
        ) == 1:
            return (
                x_new,
            )

        return (
            x_new,
            *hook_inputs[
                1:
            ],
        )

    return hook


def make_outcome_row(
    *,
    candidate_index,
    sid,
    uid,
    gt,
    baseline_pred,
    selected_alpha,
    selected_score,
    method,
    gain,
    prediction,
    matched_effective_alpha,
    shift_norm,
    shift_ratio,
    hook_calls,
):
    prediction = int(
        prediction
    )

    correct = bool(
        prediction
        ==
        gt
    )

    baseline_correct = bool(
        baseline_pred
        ==
        gt
    )

    return {
        "candidate_index":
            int(
                candidate_index
            ),

        "raw_sample_id":
            sid,

        "cohort_uid":
            uid,

        "ground_truth":
            int(
                gt
            ),

        "baseline_prediction":
            int(
                baseline_pred
            ),

        "baseline_correct":
            baseline_correct,

        "selected_alpha":
            float(
                selected_alpha
            ),

        "selected_score":
            float(
                selected_score
            ),

        "candidate_method":
            str(
                method
            ),

        "candidate_gain":
            float(
                gain
            ),

        "matched_effective_alpha":
            float(
                matched_effective_alpha
            ),

        "candidate_prediction":
            prediction,

        "candidate_correct":
            correct,

        "repair":
            bool(
                (not baseline_correct)
                and
                correct
            ),

        "break":
            bool(
                baseline_correct
                and
                (not correct)
            ),

        "shift_norm":
            float(
                shift_norm
            ),

        "shift_ratio":
            float(
                shift_ratio
            ),

        "hook_calls":
            int(
                hook_calls
            ),
    }


class DirectionModifier:

    def __init__(
        self,
        model,
        alpha,
        gain,
        method,
        U,
        sa_direction,
    ):

        layer = (
            model
            .model
            .language_model
            .layers[
                LAYER
            ]
        )

        self.o_proj = (
            layer
            .cross_attn
            .o_proj
        )

        self.alpha = float(
            alpha
        )

        self.gain = float(
            gain
        )

        self.method = str(
            method
        )

        self.U = torch.tensor(
            np.asarray(
                U,
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )

        d = np.asarray(
            sa_direction,
            dtype=np.float32,
        )

        dn = float(
            np.linalg.norm(
                d
            )
        )

        if not np.isclose(
            dn,
            1.0,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Bad frozen SA "
                f"direction norm: {dn}"
            )

        self.sa_direction = (
            torch.tensor(
                d,
                dtype=torch.float32,
            )
        )

        self.handle = None
        self.calls = 0

        self.last_h_norm = None
        self.last_shift_norm = None
        self.last_shift_ratio = None


    def hook(
        self,
        module,
        inputs,
    ):

        if not inputs:
            raise RuntimeError(
                "Empty hook inputs."
            )

        x = inputs[
            0
        ]

        if x.shape[
            -1
        ] != HIDDEN_SIZE:
            raise RuntimeError(
                "Unexpected o_proj "
                f"input width: {x.shape[-1]}"
            )

        H = x[
            ...,
            START:END
        ]

        h = (
            H.float()
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

        if float(
            h_norm.min().item()
        ) <= 1e-12:
            raise RuntimeError(
                "Runtime h norm too small."
            )


        if self.method == "MN":

            U = self.U.to(
                device=H.device
            )

            projected = (
                (h @ U)
                @
                U.T
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
                    "MN projection too small."
                )

            direction = (
                projected
                /
                projected_norm
            )


        elif self.method == "SA":

            direction = (
                self.sa_direction
                .to(
                    device=H.device
                )
                .view(
                    1,
                    HEAD_DIM,
                )
            )


        else:
            raise RuntimeError(
                f"Unknown method: "
                f"{self.method}"
            )


        delta = (
            self.gain
            *
            (
                self.alpha
                -
                1.0
            )
            *
            h_norm
            *
            direction
        )


        shift_norm = (
            torch.linalg
            .vector_norm(
                delta,
                dim=-1,
                keepdim=True,
            )
        )


        whole_reference = (
            abs(
                self.alpha
                -
                1.0
            )
            *
            h_norm
        )


        shift_ratio = (
            shift_norm
            /
            torch.clamp(
                whole_reference,
                min=1e-12,
            )
        )


        H_new = (
            H
            +
            delta
            .to(
                H.dtype
            )
            .unsqueeze(
                1
            )
        )


        x_new = x.clone()

        x_new[
            ...,
            START:END
        ] = H_new


        self.calls += 1

        self.last_h_norm = float(
            h_norm[
                0,
                0,
            ].item()
        )

        self.last_shift_norm = float(
            shift_norm[
                0,
                0,
            ].item()
        )

        self.last_shift_ratio = float(
            shift_ratio[
                0,
                0,
            ].item()
        )


        if len(
            inputs
        ) == 1:
            return (
                x_new,
            )

        return (
            x_new,
            *inputs[
                1:
            ],
        )


    def register(
        self,
    ):
        self.calls = 0

        self.handle = (
            self.o_proj
            .register_forward_pre_hook(
                self.hook
            )
        )


    def remove(
        self,
    ):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def score_no_intervention(
    model,
    inputs,
    numeral_ids,
):

    with torch.inference_mode():

        state = score_state(
            model,
            inputs,
            numeral_ids,
            DUMMY_GT,
        )

    return state


def score_lowrank(
    *,
    model,
    inputs,
    numeral_ids,
    alpha,
    gain,
    method,
    U4,
    sa_direction,
):

    modifier = DirectionModifier(
        model=model,
        alpha=alpha,
        gain=gain,
        method=method,
        U=U4,
        sa_direction=sa_direction,
    )

    modifier.register()

    try:

        with torch.inference_mode():

            state = score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )

    finally:
        modifier.remove()

    if modifier.calls <= 0:
        raise RuntimeError(
            f"{method}: hook did not fire."
        )

    if not np.isclose(
        modifier.last_shift_ratio,
        gain,
        atol=2e-3,
        rtol=2e-3,
    ):
        raise RuntimeError(
            f"{method}: shift-ratio mismatch: "
            f"{modifier.last_shift_ratio} vs {gain}"
        )

    return (
        state,
        modifier,
    )


def score_whole(
    *,
    model,
    inputs,
    numeral_ids,
    effective_alpha,
):

    modifier = HeadGainModifier(
        model=model,
        layer_idx=LAYER,
        head_idx=HEAD,
        alpha=float(
            effective_alpha
        ),
    )

    modifier.register()

    try:

        with torch.inference_mode():

            state = score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )

    finally:
        modifier.remove()

    if modifier.calls <= 0:
        raise RuntimeError(
            "WHOLE hook did not fire."
        )

    return (
        state,
        modifier,
    )
