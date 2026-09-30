import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

import e1_continuous_v1 as e1


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
REVISION = "9eb2daaa8597bf192a8b0e73f848f3a102794df5"

MEA_PATH = Path(
    "aroma_principled/results/mea_v1/horizon20.jsonl"
)

OUT = Path(
    "aroma_principled/results/mea_ablation_v1/"
    "frozen_direction_matched_budget.jsonl"
)

PG_FLOOR = 1e-12
MARGIN_TOL = 1e-3


def load_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


class MarginCapture:
    """
    Baseline-only capture.

    h_leaf parameterizes one shared 128-D additive perturbation
    across all token positions, matching MEA/E1 semantics.
    """

    def __init__(self, model):
        layer = (
            model.model.language_model.layers[e1.LAYER]
        )
        self.o_proj = layer.cross_attn.o_proj
        self.handle = None
        self.h_leaf = None
        self.h_base = None

    def hook(self, module, inputs):
        x = inputs[0]
        H = x[..., e1.START:e1.END]

        h_base = (
            H.detach()
            .float()
            .mean(dim=1)
        )

        h_leaf = (
            h_base.detach()
            .clone()
            .requires_grad_(True)
        )

        shared_delta = (
            h_leaf - h_base.detach()
        )

        x_new = x.detach().clone()
        x_new[..., e1.START:e1.END] = (
            H.detach()
            + shared_delta.to(H.dtype).unsqueeze(1)
        )

        self.h_base = h_base
        self.h_leaf = h_leaf

        return (x_new, *inputs[1:])

    def register(self):
        self.handle = (
            self.o_proj.register_forward_pre_hook(
                self.hook
            )
        )

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


class ShiftOnly:
    """Apply a fixed cumulative 128-D shared edit."""

    def __init__(self, model, delta):
        layer = (
            model.model.language_model.layers[e1.LAYER]
        )
        self.o_proj = layer.cross_attn.o_proj
        self.delta = np.asarray(
            delta,
            dtype=np.float64,
        )
        self.handle = None

    def hook(self, module, inputs):
        x = inputs[0]
        H = x[..., e1.START:e1.END]

        d = torch.as_tensor(
            self.delta,
            device=H.device,
            dtype=H.dtype,
        ).reshape(1, 1, e1.HEAD_DIM)

        x_new = x.clone()
        x_new[..., e1.START:e1.END] = H + d

        return (x_new, *inputs[1:])

    def register(self):
        self.handle = (
            self.o_proj.register_forward_pre_hook(
                self.hook
            )
        )

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def numeral_logits(outputs, numeral_ids):
    return (
        outputs.logits[0, -1, numeral_ids]
        .float()
        .to(torch.float64)
    )


def initial_state_and_gradient(
    model,
    inputs,
    numeral_ids,
    target,
):
    cap = MarginCapture(model)
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

        target_idx = target - 1
        prediction = int(torch.argmax(z).item()) + 1

        mask = torch.ones(
            len(numeral_ids),
            dtype=torch.bool,
            device=z.device,
        )
        mask[target_idx] = False

        idxs = torch.arange(
            len(numeral_ids),
            device=z.device,
        )[mask]

        j = int(torch.argmax(z[mask]).item())
        competitor_idx = int(idxs[j].item())
        competitor = competitor_idx + 1

        margin = (
            z[competitor_idx]
            - z[target_idx]
        )

        g = torch.autograd.grad(
            margin,
            cap.h_leaf,
            retain_graph=False,
            create_graph=False,
        )[0][0]

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
            "margin": float(margin.detach().item()),
            "g": g,
        }

    finally:
        cap.remove()


def evaluate_shifted(
    model,
    inputs,
    numeral_ids,
    target,
    cumulative_delta,
):
    hook = ShiftOnly(
        model,
        cumulative_delta,
    )
    hook.register()

    try:
        with torch.no_grad():
            out = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

            z = numeral_logits(
                out,
                numeral_ids,
            )

        target_idx = target - 1
        prediction = int(torch.argmax(z).item()) + 1

        mask = torch.ones(
            len(numeral_ids),
            dtype=torch.bool,
            device=z.device,
        )
        mask[target_idx] = False

        idxs = torch.arange(
            len(numeral_ids),
            device=z.device,
        )[mask]

        j = int(torch.argmax(z[mask]).item())
        competitor_idx = int(idxs[j].item())

        margin = float(
            (
                z[competitor_idx]
                - z[target_idx]
            ).item()
        )

        return {
            "prediction": prediction,
            "competitor": competitor_idx + 1,
            "margin": margin,
        }

    finally:
        hook.remove()


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)

    p = 2.0 * sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2.0 ** n)

    return min(1.0, p)


print("=" * 100)
print("MEA ABLATION v1 — FROZEN DIRECTION, MATCHED BUDGET")
print("=" * 100)

mea_rows = load_jsonl(MEA_PATH)

assert len(mea_rows) == 112, len(mea_rows)
assert len({r["sample_id"] for r in mea_rows}) == 112

meta = load_jsonl(e1.META_PATH)
meta_map = {
    str(x["sample_id"]): x
    for x in meta
}

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

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)
OUT.unlink(missing_ok=True)

results = []

for i, mea in enumerate(mea_rows, 1):

    sid = str(mea["sample_id"])
    target = int(mea["ground_truth"])

    record = meta_map[sid]

    image = Image.open(
        record["image_path"]
    ).convert("RGB")

    inputs = e1.build_inputs(
        processor,
        image,
        device,
    )

    initial = initial_state_and_gradient(
        model,
        inputs,
        numeral_ids,
        target,
    )

    # Strong semantic consistency checks against frozen MEA v2.
    if initial["prediction"] != int(mea["prediction_before"]):
        raise RuntimeError(
            f"{sid}: baseline prediction mismatch: "
            f"{initial['prediction']} vs {mea['prediction_before']}"
        )

    if abs(
        initial["margin"]
        - float(mea["margin_before"])
    ) > MARGIN_TOL:
        raise RuntimeError(
            f"{sid}: baseline margin mismatch: "
            f"{initial['margin']} vs {mea['margin_before']}"
        )

    PG0 = initial["g"] @ P
    pg0_norm = float(np.linalg.norm(PG0))

    if pg0_norm <= PG_FLOOR:
        direction = np.zeros(
            e1.HEAD_DIM,
            dtype=np.float64,
        )
        abstained = True
    else:
        direction = -PG0 / pg0_norm
        abstained = False

    step_norms = [
        float(x["step_norm"])
        for x in mea["iteration_trace"]
    ]

    budget_path_length = float(
        sum(step_norms)
    )

    cumulative = np.zeros(
        e1.HEAD_DIM,
        dtype=np.float64,
    )

    used_path_length = 0.0
    frozen_success = False
    iterations_used = 0

    state = {
        "prediction": initial["prediction"],
        "competitor": initial["competitor"],
        "margin": initial["margin"],
    }

    if not abstained:
        for k, step_norm in enumerate(
            step_norms,
            1,
        ):
            cumulative = (
                cumulative
                + step_norm * direction
            )

            used_path_length += step_norm
            iterations_used = k

            state = evaluate_shifted(
                model,
                inputs,
                numeral_ids,
                target,
                cumulative,
            )

            if state["prediction"] == target:
                frozen_success = True
                break

    row = {
        "sample_id": sid,
        "ground_truth": target,

        "prediction_before":
            initial["prediction"],
        "initial_competitor":
            initial["competitor"],
        "initial_margin":
            initial["margin"],
        "initial_pg_norm":
            pg0_norm,

        "iterative_success":
            int(mea["success"]),
        "iterative_iterations":
            int(mea["iterations"]),
        "iterative_path_length":
            float(mea["path_length"]),

        "matched_max_steps":
            len(step_norms),
        "matched_budget_path_length":
            budget_path_length,

        "frozen_success":
            int(frozen_success),
        "frozen_iterations_used":
            iterations_used,
        "frozen_used_path_length":
            used_path_length,
        "frozen_prediction_after":
            int(state["prediction"]),
        "frozen_competitor_after":
            int(state["competitor"]),
        "frozen_margin_after":
            float(state["margin"]),
        "abstained":
            int(abstained),
    }

    results.append(row)

    with OUT.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(row)
            + "\n"
        )

    print(
        f"[{i:3d}/112] {sid}  "
        f"MEA={row['iterative_success']}  "
        f"FROZEN={row['frozen_success']}  "
        f"budget={budget_path_length:.2f}  "
        f"used={used_path_length:.2f}  "
        f"pred={row['frozen_prediction_after']}"
    )


mea_success = sum(
    r["iterative_success"]
    for r in results
)

frozen_success = sum(
    r["frozen_success"]
    for r in results
)

both = sum(
    r["iterative_success"] == 1
    and r["frozen_success"] == 1
    for r in results
)

mea_only = sum(
    r["iterative_success"] == 1
    and r["frozen_success"] == 0
    for r in results
)

frozen_only = sum(
    r["iterative_success"] == 0
    and r["frozen_success"] == 1
    for r in results
)

neither = sum(
    r["iterative_success"] == 0
    and r["frozen_success"] == 0
    for r in results
)

p = exact_mcnemar(
    mea_only,
    frozen_only,
)

print()
print("=" * 100)
print("MEA ABLATION v1 SUMMARY")
print("=" * 100)

print(
    f"N: {len(results)}"
)

print(
    f"iterative MEA: "
    f"{mea_success}/{len(results)} "
    f"({100*mea_success/len(results):.2f}%)"
)

print(
    f"frozen direction: "
    f"{frozen_success}/{len(results)} "
    f"({100*frozen_success/len(results):.2f}%)"
)

print(
    f"delta: "
    f"{100*(mea_success-frozen_success)/len(results):+.2f} pp"
)

print()
print("PAIRED OUTCOMES")
print(f"both success : {both}")
print(f"MEA only     : {mea_only}")
print(f"frozen only  : {frozen_only}")
print(f"neither      : {neither}")

print()
print(
    f"McNemar exact p = {p:.8g}"
)

print()
print("Saved:", OUT)
print("=" * 100)
