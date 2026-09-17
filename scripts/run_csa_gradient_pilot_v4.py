import os
import json
import torch
import numpy as np

from PIL import Image

from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)


print("="*80)
print("AROMA 2.0 CSA GRADIENT PILOT v4")
print("="*80)


MODEL_PATH = (
    "/workspace/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-11B-Vision-Instruct/"
    "snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

IMAGE_PATH = "data/sanity_count.png"

LAYER = 18

OUT_DIR = "outputs/csa_gradient_pilot_v4"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE:", device)


# ============================================================
# LOAD
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


model.eval()


for p in model.parameters():
    p.requires_grad_(False)


print("MODEL LOADED")


# ============================================================
# CHECK STRUCTURE
# ============================================================

print(
    "MODEL:",
    type(model)
)

print(
    "TEXT LAYERS:",
    len(model.model.language_model.layers)
)


layer = model.model.language_model.layers[LAYER]


if not hasattr(layer, "cross_attn"):
    raise RuntimeError(
        "No cross_attn at layer %d" % LAYER
    )


cross_attn = layer.cross_attn


print(
    "CROSS ATTENTION:",
    type(cross_attn)
)


saved = {}


# ============================================================
# HOOK
# ============================================================

def hook_fn(module, inputs, output):

    if isinstance(output, tuple):
        hidden = output[0]
        rest = output[1:]
    else:
        hidden = output
        rest = None


    hidden = (
        hidden
        .detach()
        .clone()
        .requires_grad_(True)
    )


    saved["hidden"] = hidden


    print(
        "CAPTURED HIDDEN:",
        hidden.shape
    )


    if rest:
        return (
            hidden,
            *rest
        )

    return hidden



handle = cross_attn.register_forward_hook(
    hook_fn
)


print(
    "HOOK REGISTERED"
)



# ============================================================
# INPUT
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert(
    "RGB"
)


messages = [
{
    "role":"user",
    "content":[
        {
            "type":"image"
        },
        {
            "type":"text",
            "text":"How many objects are there?"
        }
    ]
}
]


prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)



inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)


inputs = {
    k:v.to(model.device)
    for k,v in inputs.items()
}


print(
    "INPUT READY"
)



# ============================================================
# NUMERAL TOKENS
# ============================================================

numeral_ids = []


for d in range(10):

    for s in [
        str(d),
        " "+str(d)
    ]:

        ids = processor.tokenizer.encode(
            s,
            add_special_tokens=False
        )

        if len(ids)==1:
            numeral_ids.append(ids[0])
            break



numeral_ids = sorted(
    list(set(numeral_ids))
)


print(
    "NUMERAL IDS:",
    numeral_ids
)


if len(numeral_ids)==0:
    raise RuntimeError(
        "No numeral ids found"
    )



# ============================================================
# FORWARD
# ============================================================

outputs = model(
    **inputs
)


logits = outputs.logits


print(
    "LOGITS:",
    logits.shape
)



# ============================================================
# LOSS
# ============================================================

last_logits = logits[:, -1, :]


loss = (
    -torch.logsumexp(
        last_logits[:, numeral_ids],
        dim=-1
    )
    .mean()
)


print(
    "LOSS:",
    float(loss.detach())
)



# ============================================================
# BACKWARD
# ============================================================

loss.backward()


if "hidden" not in saved:
    raise RuntimeError(
        "Hook failed"
    )


grad = saved["hidden"].grad


if grad is None:
    raise RuntimeError(
        "Gradient is None"
    )


print(
    "RAW GRAD:",
    grad.shape
)



# ============================================================
# SAVE BOTH
# ============================================================

gradient_all = (
    grad
    .detach()
    .float()
    .cpu()
    .numpy()
)


gradient_last = (
    grad[:, -1, :]
    .mean(dim=0)
    .detach()
    .float()
    .cpu()
    .numpy()
)



print(
    "FINAL GRAD:",
    gradient_last.shape
)


norm = float(
    np.linalg.norm(
        gradient_last
    )
)


ratio = float(
    np.mean(
        np.abs(gradient_last)>1e-10
    )
)


print(
    "GRAD NORM:",
    norm
)

print(
    "NONZERO RATIO:",
    ratio
)



np.save(
    f"{OUT_DIR}/gradient_last.npy",
    gradient_last
)


np.save(
    f"{OUT_DIR}/gradient_all.npy",
    gradient_all
)



with open(
    f"{OUT_DIR}/metadata.json",
    "w"
) as f:

    json.dump(
        {
            "layer":LAYER,
            "definition":
            "gradient of numeral utility wrt L18 cross attention output",
            "gradient_shape":
            list(gradient_last.shape),
            "gradient_norm":
            norm,
            "nonzero_ratio":
            ratio,
            "numeral_ids":
            numeral_ids
        },
        f,
        indent=2
    )


handle.remove()


print("="*80)
print("CSA GRADIENT PILOT v4 COMPLETE")
print("="*80)
