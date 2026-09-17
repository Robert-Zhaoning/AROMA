import os
import json
import torch
import numpy as np

from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


print("="*70)
print("AROMA 2.0 CSA GRADIENT PILOT v2")
print("="*70)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    "/workspace/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-11B-Vision-Instruct/"
    "snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)


IMAGE_PATH = (
    "data/sanity_count.png"
)


LAYER = 18


OUT = (
    "outputs/csa_gradient_pilot"
)


os.makedirs(
    OUT,
    exist_ok=True
)


device = "cuda" if torch.cuda.is_available() else "cpu"

print(
    "DEVICE:",
    device
)


# ============================================================
# LOAD MODEL
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
)


model.eval()


print("MODEL LOADED")


# ============================================================
# LOCATE CROSS ATTENTION
# ============================================================

layer = (
    model.model
    .language_model
    .layers[LAYER]
)


cross_attn = layer.cross_attn


print(
    "CROSS ATTENTION:",
    cross_attn.__class__.__name__
)


saved = {}


# ============================================================
# HOOK
# ============================================================

def hook_fn(
    module,
    inputs,
    output,
):

    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output


    hidden.retain_grad()


    saved["cross_output"] = hidden


    print(
        "CAPTURED CROSS OUTPUT:",
        hidden.shape
    )



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
                "text":
                "How many objects are there?"
            }
        ]
    }
]


prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True
)



inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)



inputs = {
    k:v.to(model.device)
    for k,v in inputs.items()
}


print(
    "INPUT READY"
)



# ============================================================
# FORWARD
# ============================================================

model.zero_grad()



outputs = model(
    **inputs
)


logits = outputs.logits


print(
    "LOGITS:",
    logits.shape
)



# ============================================================
# NUMERAL LOSS
# ============================================================

numeral_ids = list(
    range(15,25)
)


print(
    "numeral token ids:",
    numeral_ids
)



last_logits = logits[:,-1,:]


loss = -torch.logsumexp(
    last_logits[:, numeral_ids],
    dim=-1
).mean()



print(
    "LOSS:",
    float(loss.detach())
)



# ============================================================
# BACKWARD
# ============================================================


loss.backward()



if "cross_output" not in saved:
    raise RuntimeError(
        "Hook failed"
    )


grad = saved["cross_output"].grad



if grad is None:
    raise RuntimeError(
        "Gradient still None"
    )


print(
    "RAW GRAD:",
    grad.shape
)



# ============================================================
# REDUCE
# ============================================================


# average batch and sequence
while grad.ndim > 1:
    grad = grad.mean(dim=0)



gradient = (
    grad
    .detach()
    .float()
    .cpu()
    .numpy()
)



print(
    "FINAL GRADIENT:",
    gradient.shape
)



np.save(
    f"{OUT}/gradient.npy",
    gradient
)



with open(
    f"{OUT}/metadata.json",
    "w"
) as f:

    json.dump(
        {
            "layer":LAYER,
            "shape":
                list(gradient.shape),
            "definition":
                "gradient of numeral utility wrt L18 cross attention output"
        },
        f,
        indent=2
    )


handle.remove()


print("="*70)
print("CSA GRADIENT PILOT COMPLETE")
print("="*70)

