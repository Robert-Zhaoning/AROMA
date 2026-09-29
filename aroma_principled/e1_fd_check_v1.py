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

N = 4

# Primary numerical scale + slightly larger diagnostic scale.
EPS_FRACS = [0.01, 0.02]

OUT = Path(
    "aroma_principled/results/e1_continuous_v1/"
    "fd_check_v1.csv"
)
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def shifted_mu(
    model,
    inputs,
    numeral_ids,
    delta,
):
    mod = e1.SharedShift(
        model,
        delta.detach().cpu().numpy(),
    )

    mod.register()

    try:
        with torch.inference_mode():
            out = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

            mu, _, _ = e1.score_from_logits(
                out.logits,
                numeral_ids,
            )

        return float(mu.item())

    finally:
        mod.remove()


print("=" * 96)
print("AROMA E1 — CENTRAL FINITE-DIFFERENCE CHECK v1")
print("=" * 96)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=REVISION,
)

numeral_ids = e1.numeral_token_ids(
    processor.tokenizer
)

print("Loading FP32 + eager model...")

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

actions = (
    actions[
        actions["selected_alpha"].astype(float) > 1.0
    ]
    .head(N)
    .reset_index(drop=True)
)

rows = []

for i, action in actions.iterrows():

    sid = str(
        action["sample_id"]
    )

    record = meta_map[sid]

    image = Image.open(
        record["image_path"]
    ).convert("RGB")

    inputs = e1.build_inputs(
        processor,
        image,
        device,
    )

    cap = e1.LocalCapture(
        model
    )

    cap.register()

    try:
        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        mu, pred, _ = e1.score_from_logits(
            out.logits,
            numeral_ids,
        )

        x = cap.local_x

        grad_x = torch.autograd.grad(
            outputs=mu,
            inputs=x,
            retain_graph=False,
            create_graph=False,
        )[0]

    finally:
        cap.remove()

    h = (
        x[
            ...,
            e1.START:e1.END
        ]
        .detach()
        .float()
        .mean(dim=1)[0]
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    # Correct derivative for the intervention variable:
    # the SAME 128-D delta is added to every token.
    g = (
        grad_x[
            ...,
            e1.START:e1.END
        ]
        .detach()
        .float()
        .sum(dim=1)[0]
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    h_norm = float(
        np.linalg.norm(h)
    )

    PH = h @ P
    PG = g @ P

    ph_norm = float(
        np.linalg.norm(PH)
    )

    pg_norm = float(
        np.linalg.norm(PG)
    )

    if ph_norm <= 1e-15:
        raise RuntimeError(
            f"{sid}: PH norm too small"
        )

    if pg_norm <= 1e-15:
        raise RuntimeError(
            f"{sid}: PG norm too small"
        )

    directions = {
        "activation":
            PH / ph_norm,

        "gradient":
            PG / pg_norm,
    }

    print()
    print(
        f"[{i+1}/{N}] {sid} "
        f"pred={pred} "
        f"mu={float(mu.item()):.8f} "
        f"h={h_norm:.4f}"
    )

    for name, d_np in directions.items():

        analytic = float(
            np.dot(
                g,
                d_np,
            )
        )

        d = torch.tensor(
            d_np,
            dtype=torch.float32,
        )

        for eps_frac in EPS_FRACS:

            eps = (
                eps_frac
                *
                h_norm
            )

            mu_plus = shifted_mu(
                model,
                inputs,
                numeral_ids,
                eps * d,
            )

            mu_minus = shifted_mu(
                model,
                inputs,
                numeral_ids,
                -eps * d,
            )

            fd = (
                mu_plus
                -
                mu_minus
            ) / (
                2.0
                *
                eps
            )

            abs_error = abs(
                fd
                -
                analytic
            )

            scale = max(
                abs(fd),
                abs(analytic),
                1e-12,
            )

            rel_error = (
                abs_error
                /
                scale
            )

            sign_match = int(
                np.sign(fd)
                ==
                np.sign(analytic)
            )

            rows.append({
                "sample_id":
                    sid,

                "direction":
                    name,

                "eps_frac":
                    eps_frac,

                "eps":
                    eps,

                "h_norm":
                    h_norm,

                "analytic_derivative":
                    analytic,

                "finite_difference":
                    fd,

                "absolute_error":
                    abs_error,

                "relative_error":
                    rel_error,

                "sign_match":
                    sign_match,
            })

            print(
                f"  {name:10s} "
                f"eps/h={eps_frac:.3f} "
                f"analytic={analytic:+.8e} "
                f"FD={fd:+.8e} "
                f"relerr={rel_error:.3%} "
                f"sign={sign_match}"
            )


df = pd.DataFrame(
    rows
)

df.to_csv(
    OUT,
    index=False,
)

print()
print("=" * 96)
print("FD SUMMARY")
print("=" * 96)

for eps_frac in EPS_FRACS:

    z = df[
        df["eps_frac"]
        ==
        eps_frac
    ]

    print(
        f"eps/h={eps_frac:.3f}  "
        f"median relerr="
        f"{z['relative_error'].median():.3%}  "
        f"max relerr="
        f"{z['relative_error'].max():.3%}  "
        f"sign agreement="
        f"{z['sign_match'].mean():.1%}"
    )

print()
print("Saved:", OUT)
print("=" * 96)
