import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

import e1_continuous_v1 as e1


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
REVISION = "9eb2daaa8597bf192a8b0e73f848f3a102794df5"

N_WRONG = 4
BETA_TRUST = 0.10
MAX_ITERS = 3
PG_FLOOR = 1e-12

OUT = Path(
    "aroma_principled/results/mea_v1/"
    "smoke4.jsonl"
)
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


class ShiftedLocalMargin:

    def __init__(
        self,
        model,
        cumulative_delta,
    ):
        layer = (
            model
            .model
            .language_model
            .layers[e1.LAYER]
        )

        self.o_proj = layer.cross_attn.o_proj

        self.cumulative_delta = torch.as_tensor(
            cumulative_delta,
            dtype=torch.float32,
        ).reshape(1, 1, e1.HEAD_DIM)

        self.handle = None
        self.h_leaf = None
        self.h_base = None
        self.calls = 0

    def hook(
        self,
        module,
        inputs,
    ):
        x = inputs[0]

        H = x[
            ...,
            e1.START:e1.END
        ]

        base_delta = self.cumulative_delta.to(
            device=H.device,
            dtype=H.dtype,
        )

        H_shifted = (
            H.detach()
            +
            base_delta
        )

        h_base = (
            H_shifted
            .float()
            .mean(dim=1)
        )

        h_leaf = (
            h_base
            .detach()
            .clone()
            .requires_grad_(True)
        )

        local_delta = (
            h_leaf
            -
            h_base.detach()
        )

        H_new = (
            H_shifted
            +
            local_delta
            .to(H.dtype)
            .unsqueeze(1)
        )

        x_new = x.detach().clone()

        x_new[
            ...,
            e1.START:e1.END
        ] = H_new

        self.h_base = h_base
        self.h_leaf = h_leaf
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


def numeral_logits(
    outputs,
    numeral_ids,
):
    return (
        outputs.logits[
            0,
            -1,
            numeral_ids,
        ]
        .float()
        .to(torch.float64)
    )


def evaluate_state(
    model,
    inputs,
    numeral_ids,
    cumulative_delta,
    target,
):
    cap = ShiftedLocalMargin(
        model,
        cumulative_delta,
    )

    cap.register()

    try:
        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        z = numeral_logits(
            out,
            numeral_ids,
        )

        pred_idx = int(
            torch.argmax(z).item()
        )

        prediction = pred_idx + 1

        target_idx = target - 1

        mask = torch.ones(
            len(numeral_ids),
            dtype=torch.bool,
            device=z.device,
        )

        mask[target_idx] = False

        competitor_indices = (
            torch.arange(
                len(numeral_ids),
                device=z.device,
            )[mask]
        )

        competitor_local = int(
            torch.argmax(
                z[mask]
            ).item()
        )

        competitor_idx = int(
            competitor_indices[
                competitor_local
            ].item()
        )

        competitor = competitor_idx + 1

        margin = (
            z[competitor_idx]
            -
            z[target_idx]
        )

        g = torch.autograd.grad(
            outputs=margin,
            inputs=cap.h_leaf,
            retain_graph=False,
            create_graph=False,
        )[0][0]

        h = (
            cap.h_base[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        g = (
            g.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        return {
            "prediction": prediction,
            "competitor": competitor,
            "margin": float(
                margin.detach().item()
            ),
            "target_logit": float(
                z[target_idx].detach().item()
            ),
            "competitor_logit": float(
                z[competitor_idx].detach().item()
            ),
            "h": h,
            "g": g,
        }

    finally:
        cap.remove()


print("=" * 100)
print("AROMA MEA v1 — 4-SAMPLE SMOKE")
print("=" * 100)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=REVISION,
)

numeral_ids = e1.numeral_token_ids(
    processor.tokenizer
)

model = (
    MllamaForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=torch.float32,
        device_map="auto",
        attn_implementation="eager",
    )
    .eval()
)

model.requires_grad_(False)

device = next(
    p.device
    for p in model.parameters()
    if p.device.type != "meta"
)

U = np.load(
    e1.U4_PATH
).astype(np.float64)

P = U @ U.T

meta = load_jsonl(
    e1.META_PATH
)

meta_map = {
    str(x["sample_id"]): x
    for x in meta
}

actions = pd.read_csv(
    e1.ACTION_PATH
)

actions["sample_id"] = (
    actions["sample_id"].astype(str)
)

actions = actions[
    actions["selected_alpha"].astype(float) > 1.0
].reset_index(drop=True)

OUT.unlink(missing_ok=True)

n_wrong = 0

for _, action in actions.iterrows():

    if n_wrong >= N_WRONG:
        break

    sid = str(
        action["sample_id"]
    )

    record = meta_map[sid]

    target = int(
        record["ground_truth"]
    )

    if target < 1 or target > 10:
        continue

    alpha_prop = float(
        action["selected_alpha"]
    )

    image = Image.open(
        record["image_path"]
    ).convert("RGB")

    inputs = e1.build_inputs(
        processor,
        image,
        device,
    )

    cumulative = np.zeros(
        e1.HEAD_DIM,
        dtype=np.float64,
    )

    baseline = evaluate_state(
        model,
        inputs,
        numeral_ids,
        cumulative,
        target,
    )

    if baseline["prediction"] == target:
        continue

    n_wrong += 1

    print()
    print(
        f"[{n_wrong}/{N_WRONG}] "
        f"{sid} "
        f"pred0={baseline['prediction']} "
        f"target={target} "
        f"margin0={baseline['margin']:.6f}"
    )

    path_length = 0.0
    success = False
    abstained = False

    initial_margin = baseline["margin"]
    initial_prediction = baseline["prediction"]

    iteration_rows = []

    for iteration in range(
        1,
        MAX_ITERS + 1,
    ):

        state = evaluate_state(
            model,
            inputs,
            numeral_ids,
            cumulative,
            target,
        )

        if state["prediction"] == target:
            success = True
            break

        h = state["h"]
        g = state["g"]

        h_norm = float(
            np.linalg.norm(h)
        )

        PG = g @ P

        pg_norm = float(
            np.linalg.norm(PG)
        )

        if pg_norm <= PG_FLOOR:
            abstained = True
            break

        margin_before = float(
            state["margin"]
        )

        predicted_effort = (
            abs(margin_before)
            /
            pg_norm
        )

        B_i = (
            abs(
                alpha_prop
                -
                1.0
            )
            *
            h_norm
        )

        tau = (
            BETA_TRUST
            *
            B_i
        )

        unconstrained = (
            -margin_before
            *
            PG
            /
            (pg_norm ** 2)
        )

        unconstrained_norm = float(
            np.linalg.norm(
                unconstrained
            )
        )

        if unconstrained_norm <= tau:
            step = unconstrained
            clipped = False
        else:
            step = (
                -tau
                *
                PG
                /
                pg_norm
            )
            clipped = True

        step_norm = float(
            np.linalg.norm(step)
        )

        cumulative = (
            cumulative
            +
            step
        )

        path_length += step_norm

        after = evaluate_state(
            model,
            inputs,
            numeral_ids,
            cumulative,
            target,
        )

        iteration_rows.append({
            "iteration": iteration,
            "competitor_before":
                state["competitor"],
            "margin_before":
                margin_before,
            "margin_after":
                float(after["margin"]),
            "predicted_effort":
                predicted_effort,
            "tau":
                tau,
            "step_norm":
                step_norm,
            "clipped":
                int(clipped),
            "prediction_after":
                after["prediction"],
        })

        print(
            f"  iter={iteration} "
            f"c={state['competitor']} "
            f"margin "
            f"{margin_before:+.6f}"
            f" -> "
            f"{after['margin']:+.6f} "
            f"eps*={predicted_effort:.4f} "
            f"tau={tau:.4f} "
            f"step={step_norm:.4f} "
            f"clip={int(clipped)} "
            f"pred={after['prediction']}"
        )

        if after["prediction"] == target:
            success = True
            break

    final_state = evaluate_state(
        model,
        inputs,
        numeral_ids,
        cumulative,
        target,
    )

    row = {
        "sample_id": sid,
        "ground_truth": target,
        "alpha_prop": alpha_prop,
        "prediction_before":
            initial_prediction,
        "prediction_after":
            final_state["prediction"],
        "margin_before":
            initial_margin,
        "margin_after":
            final_state["margin"],
        "success":
            int(success),
        "iterations":
            len(iteration_rows),
        "abstained":
            int(abstained),
        "path_length":
            path_length,
        "cumulative_intervention_norm":
            float(
                np.linalg.norm(
                    cumulative
                )
            ),
        "iteration_trace":
            iteration_rows,
    }

    with OUT.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(row)
            +
            "\n"
        )


rows = load_jsonl(
    OUT
)

print()
print("=" * 100)
print("MEA SMOKE SUMMARY")
print("=" * 100)

print(
    "N:",
    len(rows),
)

print(
    "success:",
    sum(
        x["success"]
        for x in rows
    ),
    "/",
    len(rows),
)

print(
    "one-step success:",
    sum(
        x["success"] == 1
        and x["iterations"] == 1
        for x in rows
    ),
    "/",
    len(rows),
)

print(
    "abstained:",
    sum(
        x["abstained"]
        for x in rows
    ),
)

print(
    "margin decreased:",
    sum(
        x["margin_after"]
        <
        x["margin_before"]
        for x in rows
    ),
    "/",
    len(rows),
)

print()
print("Saved:", OUT)
print("=" * 100)
