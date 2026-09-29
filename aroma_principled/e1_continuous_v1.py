#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


# ============================================================
# Frozen scientific definition
# ============================================================

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

LAYER = 18
HEAD = 13

HEAD_DIM = 128
START = HEAD * HEAD_DIM
END = START + HEAD_DIM

NUMERALS = list(range(1, 11))

QUESTION = """Inspect the image carefully.

How many red circles are visible?

Return the number only."""

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/U4_primary.npy"
)

META_PATH = Path(
    "data/proc_count_sa_dev_v1/metadata.jsonl"
)

ACTION_PATH = Path(
    "outputs/proc_count_sa_dev_v1/"
    "action_manifest_v1/"
    "sa_vs_mn_action_manifest_v1.csv"
)

BETAS = [
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
]


# ============================================================
# Utilities
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path):
    rows = []

    with Path(path).open(
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def append_jsonl(path, row):
    with Path(path).open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                sort_keys=True,
            )
            + "\n"
        )


def numeral_token_ids(tokenizer):
    out = []

    for k in NUMERALS:
        ids = tokenizer.encode(
            str(k),
            add_special_tokens=False,
        )

        if len(ids) != 1:
            raise RuntimeError(
                f"Numeral {k} is not a single token: {ids}"
            )

        out.append(
            int(ids[0])
        )

    return out


def build_inputs(
    processor,
    image,
    device,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": QUESTION,
                },
            ],
        }
    ]

    formatted = (
        processor
        .apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    )

    inputs = processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )

    return {
        k: v.to(device)
        for k, v in inputs.items()
    }


def score_from_logits(
    logits,
    numeral_ids,
):
    z = logits[
        0,
        -1,
        numeral_ids,
    ].float().to(torch.float64)

    probs = torch.softmax(
        z,
        dim=-1,
    )

    values = torch.arange(
        1,
        11,
        dtype=torch.float64,
        device=probs.device,
    )

    mu = (
        probs
        *
        values
    ).sum()

    pred = int(
        NUMERALS[
            torch.argmax(
                probs
            ).item()
        ]
    )

    return mu, pred, probs


# ============================================================
# Baseline local-state capture
#
# Important:
#
# We detach the input to o_proj and make that tensor the local
# differentiation variable.  This computes the derivative of
# the output with respect to the intervention site directly,
# without backpropagating through earlier layers.
#
# The actual intervention adds one shared 128-D delta to every
# token position. Therefore:
#
#     d mu / d delta
#       = sum_t d mu / d H_t
#
# We call this g_shift.
# ============================================================

class LocalCapture:

    def __init__(self, model):

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

        self.handle = None
        self.local_x = None

    def hook(
        self,
        module,
        inputs,
    ):
        if not inputs:
            raise RuntimeError(
                "Empty o_proj hook input."
            )

        x = inputs[0]

        if x.shape[-1] != 4096:
            raise RuntimeError(
                f"Unexpected o_proj input shape {tuple(x.shape)}"
            )

        local_x = (
            x.detach()
            .clone()
            .requires_grad_(True)
        )

        self.local_x = local_x

        return (
            local_x,
            *inputs[1:],
        )

    def register(self):

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
# Shared-shift intervention
#
# H_t' = H_t + delta
#
# The same delta is applied to every token.
# ============================================================

class SharedShift:

    def __init__(
        self,
        model,
        delta,
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

        self.delta = (
            torch.as_tensor(
                delta,
                dtype=torch.float32,
            )
            .reshape(
                1,
                1,
                HEAD_DIM,
            )
        )

        self.handle = None

        self.requested_norm = float(
            torch.linalg.vector_norm(
                self.delta.float()
            ).item()
        )

        self.realized_norm = None
        self.calls = 0

    def hook(
        self,
        module,
        inputs,
    ):
        if not inputs:
            raise RuntimeError(
                "Empty intervention hook input."
            )

        x = inputs[0]

        H = x[
            ...,
            START:END
        ]

        before = (
            H.float()
            .mean(dim=1)
        )

        delta = self.delta.to(
            device=H.device,
            dtype=H.dtype,
        )

        H_new = (
            H
            +
            delta
        )

        after = (
            H_new.float()
            .mean(dim=1)
        )

        realized = (
            after
            -
            before
        )

        self.realized_norm = float(
            torch.linalg.vector_norm(
                realized,
                dim=-1,
            )[0].item()
        )

        x_new = x.clone()

        x_new[
            ...,
            START:END
        ] = H_new

        self.calls += 1

        return (
            x_new,
            *inputs[1:],
        )

    def register(self):

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
# Baseline + local gradient
# ============================================================

def baseline_local_state(
    model,
    inputs,
    numeral_ids,
    P,
):
    capture = LocalCapture(
        model
    )

    capture.register()

    try:
        outputs = model(
            **inputs,
            use_cache=False,
        )

        mu0, pred0, probs0 = (
            score_from_logits(
                outputs.logits,
                numeral_ids,
            )
        )

        if capture.local_x is None:
            raise RuntimeError(
                "Local capture hook did not fire."
            )

        grad_x = torch.autograd.grad(
            mu0,
            capture.local_x,
            retain_graph=False,
            create_graph=False,
        )[0]

    finally:
        capture.remove()

    x_local = (
        capture.local_x
        .detach()
        .float()
    )

    H = x_local[
        ...,
        START:END
    ]

    h = (
        H
        .mean(dim=1)[0]
    )

    # Gradient with respect to one shared additive shift
    # across all token positions.
    g_shift = (
        grad_x[
            ...,
            START:END
        ]
        .detach()
        .float()
        .sum(dim=1)[0]
    )

    h_np = (
        h.cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    g_np = (
        g_shift.cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    PH = (
        h_np
        @
        P
    )

    PG = (
        g_np
        @
        P
    )

    h_norm = float(
        np.linalg.norm(
            h_np
        )
    )

    ph_norm = float(
        np.linalg.norm(
            PH
        )
    )

    g_norm = float(
        np.linalg.norm(
            g_np
        )
    )

    pg_norm = float(
        np.linalg.norm(
            PG
        )
    )

    if min(
        h_norm,
        ph_norm,
        g_norm,
        pg_norm,
    ) <= 1e-12:
        raise RuntimeError(
            "Degenerate mechanistic norm."
        )

    d_activation = (
        PH
        /
        ph_norm
    )

    d_gradient = (
        PG
        /
        pg_norm
    )

    cos_pg_ph = float(
        np.dot(
            PG,
            PH,
        )
        /
        (
            pg_norm
            *
            ph_norm
        )
    )

    rho = float(
        ph_norm
        /
        h_norm
    )

    return {
        "mu0":
            float(
                mu0.detach().item()
            ),

        "prediction0":
            int(pred0),

        "probs0":
            probs0.detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist(),

        "h":
            h_np,

        "g_shift":
            g_np,

        "PH":
            PH,

        "PG":
            PG,

        "d_activation":
            d_activation,

        "d_gradient":
            d_gradient,

        "h_norm":
            h_norm,

        "ph_norm":
            ph_norm,

        "g_shift_norm":
            g_norm,

        "pg_shift_norm":
            pg_norm,

        "rho":
            rho,

        "cos_pg_ph":
            cos_pg_ph,
    }


# ============================================================
# Intervention forward
# ============================================================

def run_shift(
    model,
    inputs,
    numeral_ids,
    delta,
):
    modifier = SharedShift(
        model,
        delta,
    )

    modifier.register()

    try:
        with torch.inference_mode():

            outputs = model(
                **inputs,
                use_cache=False,
            )

            mu, pred, probs = (
                score_from_logits(
                    outputs.logits,
                    numeral_ids,
                )
            )

    finally:
        modifier.remove()

    if modifier.calls != 1:
        raise RuntimeError(
            "Expected exactly one intervention hook call; "
            f"got {modifier.calls}"
        )

    return {
        "mu":
            float(
                mu.item()
            ),

        "prediction":
            int(pred),

        "probs":
            probs.detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist(),

        "requested_shift_norm":
            float(
                modifier.requested_norm
            ),

        "realized_shift_norm":
            float(
                modifier.realized_norm
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.out.exists():

        if not args.overwrite:

            raise RuntimeError(
                f"Output already exists: {args.out}\n"
                "Use --overwrite only intentionally."
            )

        args.out.unlink()

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load frozen basis
    # --------------------------------------------------------

    U = np.load(
        U4_PATH
    ).astype(
        np.float64
    )

    if U.shape != (
        128,
        4,
    ):
        raise RuntimeError(
            f"Unexpected U4 shape: {U.shape}"
        )

    gram_error = float(
        np.max(
            np.abs(
                U.T
                @
                U
                -
                np.eye(4)
            )
        )
    )

    if gram_error > 1e-5:
        raise RuntimeError(
            f"U4 is not orthonormal; "
            f"max Gram error={gram_error}"
        )

    P = (
        U
        @
        U.T
    )

    # --------------------------------------------------------
    # Development population
    # --------------------------------------------------------

    meta = load_jsonl(
        META_PATH
    )

    meta_map = {
        str(x["sample_id"]):
            x
        for x in meta
    }

    actions = pd.read_csv(
        ACTION_PATH
    )

    actions[
        "sample_id"
    ] = actions[
        "sample_id"
    ].astype(str)

    actions[
        "selected_alpha"
    ] = actions[
        "selected_alpha"
    ].astype(float)

    actions = (
        actions[
            actions[
                "selected_alpha"
            ]
            >
            1.0
        ]
        .sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(actions) != 160:
        raise RuntimeError(
            f"Expected upward development N=160; "
            f"got {len(actions)}"
        )

    if args.limit is not None:

        actions = (
            actions
            .iloc[
                :args.limit
            ]
            .copy()
        )

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    protocol = {
        "name":
            "AROMA Principled E1 Continuous Response v1",

        "status":
            (
                "development-only mechanistic experiment; "
                "no final-set evaluation"
            ),

        "model_id":
            MODEL_ID,

        "model_revision":
            MODEL_REVISION,

        "dtype":
            "float32",

        "attention_implementation":
            "eager",

        "layer":
            LAYER,

        "head":
            HEAD,

        "head_slice":
            [
                START,
                END,
            ],

        "question":
            QUESTION,

        "numerals":
            NUMERALS,

        "population":
            "proc_count_sa_dev_v1 selected_alpha > 1",

        "n_selected":
            int(
                len(actions)
            ),

        "betas":
            BETAS,

        "direction_activation":
            "P h / ||P h||",

        "direction_gradient":
            "P g_shift / ||P g_shift||",

        "g_shift_definition":
            (
                "sum over token-position gradients because "
                "the intervention adds one shared delta "
                "to every token position"
            ),

        "shift_budget":
            (
                "beta * (alpha_prop - 1) * ||h||"
            ),

        "u4_path":
            str(U4_PATH),

        "u4_sha256":
            sha256(
                U4_PATH
            ),

        "metadata_sha256":
            sha256(
                META_PATH
            ),

        "action_manifest_sha256":
            sha256(
                ACTION_PATH
            ),

        "u4_gram_max_error":
            gram_error,
    }

    protocol_path = (
        Path(
            "aroma_principled/"
            "protocols/"
            "e1_continuous_v1_smoke.json"
        )
        if args.limit is not None
        else
        Path(
            "aroma_principled/"
            "protocols/"
            "e1_continuous_v1_full.json"
        )
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("AROMA PRINCIPLED E1 — CONTINUOUS RESPONSE v1")
    print("=" * 100)

    print(
        "N:",
        len(actions),
    )

    print(
        "beta grid:",
        BETAS,
    )

    print(
        "U4 Gram max error:",
        gram_error,
    )

    print()
    print("Loading processor...")

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
    )

    tokenizer = (
        processor.tokenizer
    )

    numeral_ids = (
        numeral_token_ids(
            tokenizer
        )
    )

    print(
        "numeral token ids:",
        numeral_ids,
    )

    print()
    print(
        "Loading model in FP32 + eager attention..."
    )

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.float32,
            device_map="auto",
            attn_implementation="eager",
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Memory-critical scientific setting
    #
    # E1 needs gradients ONLY with respect to the local
    # intervention variable inserted at L18H13.
    #
    # Freezing all model parameters prevents autograd from
    # retaining the vision encoder and pre-intervention graph.
    # The LocalCapture hook later creates:
    #
    #     local_x = x.detach().clone().requires_grad_(True)
    #
    # so d(mu)/d(local_x) remains exact.
    # --------------------------------------------------------
    model.requires_grad_(False)

    input_device = next(
        p.device
        for p in model.parameters()
        if p.device.type != "meta"
    )

    print(
        "input device:",
        input_device,
    )

    print(
        "runtime model revision:",
        MODEL_REVISION,
    )

    print(
        "attention implementation: eager"
    )

    print()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    summary_rows = []

    for sample_index, action in (
        actions.iterrows()
    ):

        sid = str(
            action[
                "sample_id"
            ]
        )

        alpha_prop = float(
            action[
                "selected_alpha"
            ]
        )

        if sid not in meta_map:
            raise RuntimeError(
                f"Missing metadata for {sid}"
            )

        item = meta_map[
            sid
        ]

        image_path = Path(
            item[
                "image_path"
            ]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        image = (
            Image.open(
                image_path
            )
            .convert(
                "RGB"
            )
        )

        inputs = build_inputs(
            processor,
            image,
            input_device,
        )

        # ----------------------------------------------------
        # Baseline + local derivative
        # ----------------------------------------------------

        local = (
            baseline_local_state(
                model,
                inputs,
                numeral_ids,
                P,
            )
        )

        # ----------------------------------------------------
        # Exact identity-hook sanity
        # ----------------------------------------------------

        zero = np.zeros(
            128,
            dtype=np.float64,
        )

        identity = run_shift(
            model,
            inputs,
            numeral_ids,
            zero,
        )

        identity_error = abs(
            identity["mu"]
            -
            local["mu0"]
        )

        if identity_error > 1e-7:

            raise RuntimeError(
                f"{sid}: identity-hook mu mismatch "
                f"{identity_error}"
            )

        print(
            f"[{sample_index+1}/{len(actions)}] "
            f"{sid} "
            f"alpha={alpha_prop:.2f} "
            f"pred0={local['prediction0']} "
            f"mu0={local['mu0']:.5f} "
            f"rho={local['rho']:.4f} "
            f"cos={local['cos_pg_ph']:.4f}"
        )

        sample_records = []

        for direction_name, direction in [
            (
                "activation",
                local[
                    "d_activation"
                ],
            ),
            (
                "gradient",
                local[
                    "d_gradient"
                ],
            ),
        ]:

            directional_derivative = float(
                np.dot(
                    local[
                        "g_shift"
                    ],
                    direction,
                )
            )

            for beta in BETAS:

                step = float(
                    beta
                    *
                    (
                        alpha_prop
                        -
                        1.0
                    )
                    *
                    local[
                        "h_norm"
                    ]
                )

                delta = (
                    step
                    *
                    direction
                )

                run = run_shift(
                    model,
                    inputs,
                    numeral_ids,
                    delta,
                )

                observed_delta_mu = (
                    run["mu"]
                    -
                    local["mu0"]
                )

                linear_pred_delta_mu = (
                    step
                    *
                    directional_derivative
                )

                record = {
                    "sample_id":
                        sid,

                    "ground_truth":
                        int(
                            item[
                                "ground_truth"
                            ]
                        ),

                    "alpha_prop":
                        alpha_prop,

                    "beta":
                        float(beta),

                    "direction":
                        direction_name,

                    "prediction0":
                        local[
                            "prediction0"
                        ],

                    "prediction":
                        run[
                            "prediction"
                        ],

                    "mu0":
                        local[
                            "mu0"
                        ],

                    "mu":
                        run[
                            "mu"
                        ],

                    "delta_mu":
                        float(
                            observed_delta_mu
                        ),

                    "linear_pred_delta_mu":
                        float(
                            linear_pred_delta_mu
                        ),

                    "directional_derivative":
                        directional_derivative,

                    "h_norm":
                        local[
                            "h_norm"
                        ],

                    "ph_norm":
                        local[
                            "ph_norm"
                        ],

                    "g_shift_norm":
                        local[
                            "g_shift_norm"
                        ],

                    "pg_shift_norm":
                        local[
                            "pg_shift_norm"
                        ],

                    "rho":
                        local[
                            "rho"
                        ],

                    "cos_pg_ph":
                        local[
                            "cos_pg_ph"
                        ],

                    "requested_shift_norm":
                        run[
                            "requested_shift_norm"
                        ],

                    "realized_shift_norm":
                        run[
                            "realized_shift_norm"
                        ],

                    "shift_norm_abs_error":
                        abs(
                            run[
                                "realized_shift_norm"
                            ]
                            -
                            run[
                                "requested_shift_norm"
                            ]
                        ),
                }

                append_jsonl(
                    args.out,
                    record,
                )

                sample_records.append(
                    record
                )

        # ----------------------------------------------------
        # Per-sample smoke summary
        # ----------------------------------------------------

        frame = pd.DataFrame(
            sample_records
        )

        a = (
            frame[
                frame[
                    "direction"
                ]
                ==
                "activation"
            ]
            .set_index(
                "beta"
            )
        )

        g = (
            frame[
                frame[
                    "direction"
                ]
                ==
                "gradient"
            ]
            .set_index(
                "beta"
            )
        )

        for beta in BETAS:

            da = float(
                a.loc[
                    beta,
                    "delta_mu"
                ]
            )

            dg = float(
                g.loc[
                    beta,
                    "delta_mu"
                ]
            )

            ratio = (
                da
                /
                dg
                if abs(dg) > 1e-10
                else np.nan
            )

            summary_rows.append(
                {
                    "sample_id":
                        sid,

                    "beta":
                        beta,

                    "cos_pg_ph":
                        local[
                            "cos_pg_ph"
                        ],

                    "delta_mu_activation":
                        da,

                    "delta_mu_gradient":
                        dg,

                    "observed_ratio":
                        ratio,
                }
            )

        del inputs
        del image

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Smoke summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        args.out.parent
        /
        (
            args.out.stem
            +
            "_summary.csv"
        )
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    print()
    print("=" * 100)
    print("SMOKE SUMMARY")
    print("=" * 100)

    for beta in BETAS:

        x = summary[
            np.isclose(
                summary[
                    "beta"
                ],
                beta,
            )
        ].copy()

        good = x[
            np.isfinite(
                x[
                    "observed_ratio"
                ]
            )
        ]

        if len(good):

            median_ratio = float(
                np.median(
                    good[
                        "observed_ratio"
                    ]
                )
            )

        else:

            median_ratio = np.nan

        median_cos = float(
            np.median(
                x[
                    "cos_pg_ph"
                ]
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    x[
                        "observed_ratio"
                    ]
                    -
                    x[
                        "cos_pg_ph"
                    ]
                )
            )
        )

        print(
            f"beta={beta:>4.2f}  "
            f"median observed A/G={median_ratio:>8.4f}  "
            f"median cos={median_cos:>8.4f}  "
            f"ratio MAE={mae:>8.4f}"
        )

    records = [
        json.loads(line)
        for line in args.out.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    max_shift_error = max(
        float(
            r[
                "shift_norm_abs_error"
            ]
        )
        for r in records
    )

    small = [
        r
        for r in records
        if float(
            r[
                "beta"
            ]
        )
        <= 0.10
    ]

    linear_abs_errors = [
        abs(
            float(
                r[
                    "delta_mu"
                ]
            )
            -
            float(
                r[
                    "linear_pred_delta_mu"
                ]
            )
        )
        for r in small
    ]

    print()
    print(
        "max realized/requested shift-norm error:",
        f"{max_shift_error:.6e}",
    )

    print(
        "mean |observed-linear| for beta<=0.10:",
        f"{np.mean(linear_abs_errors):.6e}",
    )

    print()
    print("Saved:")
    print(" ", args.out)
    print(" ", summary_csv)
    print(" ", protocol_path)

    print()
    print("=" * 100)
    print("E1 CONTINUOUS SMOKE COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
