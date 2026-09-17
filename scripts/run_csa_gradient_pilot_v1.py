import os
import json
import torch
import numpy as np

from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


MODEL_PATH = (
"/workspace/.cache/huggingface/hub/"
"models--meta-llama--Llama-3.2-11B-Vision-Instruct/"
"snapshots/"
"9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)


IMAGE_PATH = "data/sanity_count.png"

LAYER = 18
HEAD = 13

OUT = "outputs/csa_gradient_pilot_v1"

os.makedirs(
    OUT,
    exist_ok=True
)


device = "cuda" if torch.cuda.is_available() else "cpu"


print("="*70)
print("AROMA 2.0 CSA GRADIENT PILOT")
print("="*70)

print("DEVICE:", device)


processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)


model.eval()

print("MODEL LOADED")


layer = (
    model
    .model
    .language_model
    .layers[LAYER]
)


o_proj = (
    layer
    .cross_attn
    .o_proj
)


hidden_size = o_proj.in_features
num_heads = 32

head_dim = hidden_size // num_heads

start = HEAD * head_dim
end = start + head_dim


print("hidden_size:", hidden_size)
print("head_dim:", head_dim)
print("HEAD SLICE:", start, end)


saved = {}


def hook_fn(module, inputs):

    x = inputs[0]

    head = x[..., start:end]

    x_new = x.clone()

    head = x_new[..., start:end]

    head.retain_grad()

    saved["head"] = head

    return (
        x_new,
    )


handle = (
    o_proj.register_forward_pre_hook(
        hook_fn
    )
)


print(
    "HOOK REGISTERED",
    start,
    end
)


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


print("INPUT READY")


model.zero_grad()


outputs = model(
    **inputs
)


logits = outputs.logits


print(
"LOGITS:",
logits.shape
)


last_logits = logits[:,-1,:]


tokenizer = processor.tokenizer


numeral_ids=[]


for i in range(10):

    ids = tokenizer.encode(
        str(i),
        add_special_tokens=False
    )

    if len(ids)==1:
        numeral_ids.append(ids[0])


numeral_ids = sorted(
list(set(numeral_ids))
)


print(
"numeral token ids:",
numeral_ids
)


loss = -torch.logsumexp(
    last_logits[:,numeral_ids],
    dim=-1
).mean()


print(
"LOSS:",
loss.detach().float().item()
)


loss.backward()


if "head" not in saved:
    raise RuntimeError(
        "Hook did not capture tensor"
    )


grad = saved["head"].grad


if grad is None:
    raise RuntimeError(
        "Gradient is None after backward"
    )


grad = grad[..., start:end]


print(
"RAW HEAD GRAD:",
grad.shape
)


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
os.path.join(
OUT,
"gradient.npy"
),
gradient
)


with open(
os.path.join(
OUT,
"metadata.json"
),
"w"
) as f:

    json.dump(
    {
    "layer":LAYER,
    "head":HEAD,
    "slice":[start,end],
    "gradient_shape":
        list(gradient.shape)
    },
    f,
    indent=2
    )


handle.remove()


print("="*70)
print("CSA PILOT COMPLETE")
print("="*70)

