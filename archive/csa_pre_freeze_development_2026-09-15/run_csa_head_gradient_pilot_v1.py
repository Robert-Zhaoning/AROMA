import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

IMAGE_PATH = Path("data/sanity_count.png")

OUT_DIR = Path("outputs/aroma2/csa_head_gradient_pilot_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER = 18
HEAD = 13
NUM_HEADS = 32
HIDDEN_SIZE = 4096
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS

START = HEAD * HEAD_DIM
END = START + HEAD_DIM

# Pilot only.
# Original L18H13 directional characterization used 1..10.
NUMERAL_VALUES = list(range(1, 11))

QUESTION = (
    "How many red objects are there? "
    "Answer with a single number."
)


print("=" * 80)
print("AROMA 2.0 — L18H13 HEAD-ALIGNED CSA GRADIENT PILOT v1")
print("=" * 80)

print("DEVICE:", "cuda" if torch.cuda.is_available() else "cpu")
print("LAYER:", LAYER)
print("HEAD:", HEAD)
print("HEAD_DIM:", HEAD_DIM)
print("HEAD SLICE:", START, END)


# ============================================================
# LOAD
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    local_files_only=True,
)

model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

# We do NOT need parameter gradients.
# The o_proj input will be replaced by an identical leaf tensor.
for p in model.parameters():
    p.requires_grad_(False)

text_layers = model.model.language_model.layers
layer = text_layers[LAYER]
o_proj = layer.cross_attn.o_proj

assert o_proj.in_features == HIDDEN_SIZE

print("MODEL LOADED")
print("O_PROJ:", type(o_proj))


# ============================================================
# NUMERAL TOKENS
# ============================================================

numeral_ids = []

for n in NUMERAL_VALUES:
    ids = processor.tokenizer.encode(
        str(n),
        add_special_tokens=False,
    )

    if len(ids) != 1:
        raise RuntimeError(
            f"Numeral {n} is not single-token: {ids}"
        )

    numeral_ids.append(ids[0])

print("NUMERAL VALUES:", NUMERAL_VALUES)
print("NUMERAL IDS:", numeral_ids)


# ============================================================
# INPUT
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": QUESTION,
            },
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)

inputs = {
    k: v.to(model.device)
    for k, v in inputs.items()
}

print("INPUT READY")
print("INPUT IDS SHAPE:", tuple(inputs["input_ids"].shape))


# ============================================================
# HELPER: expected numeral
# ============================================================

numeral_values_tensor = torch.tensor(
    NUMERAL_VALUES,
    dtype=torch.float32,
    device=model.device,
)


def expected_numeral(logits):
    """
    Normalized expected numeral over the selected numeral set.
    Equivalent to conditioning the full next-token distribution
    on the selected numeral-token set.
    """
    last_logits = logits[:, -1, :].float()

    selected = last_logits[:, numeral_ids]

    conditional_probs = torch.softmax(
        selected,
        dim=-1,
    )

    mu = (
        conditional_probs
        * numeral_values_tensor.unsqueeze(0)
    ).sum(dim=-1)

    return mu.mean(), conditional_probs


# ============================================================
# BASELINE FORWARD
# ============================================================

with torch.no_grad():
    baseline_out = model(**inputs)

baseline_logits = baseline_out.logits.detach().float()

baseline_mu, baseline_conditional = expected_numeral(
    baseline_logits
)

print()
print("BASELINE MU:", baseline_mu.item())


# ============================================================
# HEAD-ALIGNED PRE-HOOK
#
# IMPORTANT:
# This is the SAME control surface as HeadGainModifier:
#
# layer.cross_attn.o_proj INPUT
#
# Instead of scaling H13, we replace the full o_proj input
# with an IDENTICAL leaf tensor requiring grad.
#
# Downstream computation therefore depends directly on this
# intervention-site tensor.
# ============================================================

saved = {}


def o_proj_pre_hook(module, hook_inputs):
    if not hook_inputs:
        raise RuntimeError("o_proj pre-hook received no inputs")

    x = hook_inputs[0]

    if x.shape[-1] != HIDDEN_SIZE:
        raise RuntimeError(
            f"Unexpected o_proj input shape: {tuple(x.shape)}"
        )

    # Identical numeric value, but becomes the local autograd leaf.
    x_leaf = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    saved["x_leaf"] = x_leaf
    saved["x_original"] = x.detach().float().cpu()

    if len(hook_inputs) == 1:
        return (x_leaf,)

    return (
        x_leaf,
        *hook_inputs[1:],
    )


handle = o_proj.register_forward_pre_hook(
    o_proj_pre_hook
)


# ============================================================
# HOOKED FORWARD
# ============================================================

try:
    hooked_out = model(**inputs)

    hooked_logits = hooked_out.logits

    hooked_mu, hooked_conditional = expected_numeral(
        hooked_logits
    )

    if "x_leaf" not in saved:
        raise RuntimeError(
            "o_proj hook did not capture intervention-site tensor"
        )

    x_leaf = saved["x_leaf"]

    print()
    print("CAPTURED O_PROJ INPUT:", tuple(x_leaf.shape))
    print("CAPTURED DTYPE:", x_leaf.dtype)
    print("CAPTURED REQUIRES_GRAD:", x_leaf.requires_grad)

    # --------------------------------------------------------
    # Identity check
    # --------------------------------------------------------

    max_logit_diff = (
        hooked_logits.detach().float()
        - baseline_logits
    ).abs().max().item()

    mu_diff = abs(
        hooked_mu.detach().float().item()
        - baseline_mu.detach().float().item()
    )

    print()
    print("IDENTITY CHECK")
    print("MAX LOGIT ABS DIFF:", max_logit_diff)
    print("MU ABS DIFF:", mu_diff)

    # --------------------------------------------------------
    # Gradient wrt FULL o_proj input
    # --------------------------------------------------------

    grad_full = torch.autograd.grad(
        outputs=hooked_mu,
        inputs=x_leaf,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]

finally:
    handle.remove()


# ============================================================
# EXTRACT L18H13
# ============================================================

grad_head = (
    grad_full[..., START:END]
    .detach()
    .float()
)

activation_head = (
    x_leaf[..., START:END]
    .detach()
    .float()
)

print()
print("FULL GRAD SHAPE:", tuple(grad_full.shape))
print("HEAD GRAD SHAPE:", tuple(grad_head.shape))
print("HEAD ACTIVATION SHAPE:", tuple(activation_head.shape))

# batch size is expected to be 1
if grad_head.shape[0] != 1:
    raise RuntimeError(
        f"Expected batch size 1, got {grad_head.shape[0]}"
    )

grad_TD = grad_head[0]
act_TD = activation_head[0]

# Save several summaries for diagnostic purposes only.
# Official Gate A must use the aggregation frozen in protocol v1.1.
grad_mean = grad_TD.mean(dim=0)
grad_last = grad_TD[-1]

print()
print("GRADIENT HEALTH")
print("ALL-TOKEN NORM:", grad_TD.norm().item())
print("MEAN-OVER-T NORM:", grad_mean.norm().item())
print("LAST-TOKEN NORM:", grad_last.norm().item())
print("ABS MEAN:", grad_TD.abs().mean().item())
print("ABS MAX:", grad_TD.abs().max().item())
print(
    "NONZERO RATIO:",
    (grad_TD != 0).float().mean().item()
)
print(
    "FINITE:",
    torch.isfinite(grad_TD).all().item()
)

# Per-token norms help inspect whether sensitivity is concentrated
# at particular query positions without changing the frozen protocol.
token_norms = torch.linalg.vector_norm(
    grad_TD,
    dim=-1,
)

print()
print("TOKEN GRAD NORMS:")
print(token_norms.cpu().numpy())


# ============================================================
# SAVE
# ============================================================

np.save(
    OUT_DIR / "head_gradient_all_tokens.npy",
    grad_TD.cpu().numpy(),
)

np.save(
    OUT_DIR / "head_activation_all_tokens.npy",
    act_TD.cpu().numpy(),
)

np.save(
    OUT_DIR / "head_gradient_mean.npy",
    grad_mean.cpu().numpy(),
)

np.save(
    OUT_DIR / "head_gradient_last.npy",
    grad_last.cpu().numpy(),
)

np.save(
    OUT_DIR / "token_gradient_norms.npy",
    token_norms.cpu().numpy(),
)

metadata = {
    "stage": "AROMA2_CSA_HEAD_GRADIENT_PILOT_V1",
    "status": "engineering_pilot_not_gate_a",
    "model_id": MODEL_ID,
    "layer": LAYER,
    "head": HEAD,
    "hidden_size": HIDDEN_SIZE,
    "num_heads": NUM_HEADS,
    "head_dim": HEAD_DIM,
    "head_slice_start": START,
    "head_slice_end": END,
    "image_path": str(IMAGE_PATH),
    "question": QUESTION,
    "numeral_values": NUMERAL_VALUES,
    "numeral_ids": numeral_ids,
    "baseline_mu": float(baseline_mu.item()),
    "hooked_mu": float(hooked_mu.detach().item()),
    "identity_max_logit_abs_diff": float(max_logit_diff),
    "identity_mu_abs_diff": float(mu_diff),
    "o_proj_input_shape": list(x_leaf.shape),
    "head_gradient_shape": list(grad_head.shape),
    "all_token_gradient_norm": float(grad_TD.norm().item()),
    "mean_gradient_norm": float(grad_mean.norm().item()),
    "last_gradient_norm": float(grad_last.norm().item()),
    "gradient_abs_mean": float(grad_TD.abs().mean().item()),
    "gradient_abs_max": float(grad_TD.abs().max().item()),
    "gradient_nonzero_ratio": float(
        (grad_TD != 0).float().mean().item()
    ),
    "gradient_all_finite": bool(
        torch.isfinite(grad_TD).all().item()
    ),
}

with open(
    OUT_DIR / "metadata.json",
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


print()
print("=" * 80)
print("PILOT RESULT")
print("=" * 80)

print(
    "SUCCESS"
    if (
        torch.isfinite(grad_TD).all()
        and grad_TD.norm() > 0
        and max_logit_diff == 0.0
    )
    else "CHECK_REQUIRED"
)

print("OUTPUT:", OUT_DIR)
print("=" * 80)
print("CSA HEAD-ALIGNED GRADIENT PILOT COMPLETE")
print("=" * 80)
