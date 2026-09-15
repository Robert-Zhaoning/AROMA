
import torch

from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)


print("="*80)
print("AROMA CSA GRAPH CONNECTIVITY DIAGNOSTIC v2")
print("="*80)


MODEL_PATH="/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"

IMAGE_PATH="data/sanity_count.png"

LAYER=18


# ==================================================
# Load model
# ==================================================

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


model.eval()


print("MODEL LOADED")


layer = (
    model
    .model
    .language_model
    .layers[LAYER]
)


print("LAYER:", LAYER)


gate = (
    layer
    .cross_attn_attn_gate
    .detach()
    .float()
    .item()
)


print("GATE:", gate)



# ==================================================
# Hook
# ==================================================

saved={}


def hook_fn(module, inputs, output):

    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output


    print("\n===== FORWARD HOOK =====")

    print(
        "hidden shape:",
        hidden.shape
    )

    print(
        "hidden dtype:",
        hidden.dtype
    )

    print(
        "requires_grad:",
        hidden.requires_grad
    )

    print(
        "grad_fn:",
        hidden.grad_fn
    )


    hidden.retain_grad()

    saved["hidden"] = hidden



handle = (
    layer
    .cross_attn
    .register_forward_hook(
        hook_fn
    )
)



# ==================================================
# Prepare input
# ==================================================

image = Image.open(
    IMAGE_PATH
).convert(
    "RGB"
)


messages=[
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
    tokenize=False,
    add_generation_prompt=True
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


print("INPUT READY")



# ==================================================
# Forward
# ==================================================

model.zero_grad()


outputs = model(
    **inputs
)


logits = outputs.logits


print(
    "LOGITS:",
    logits.shape
)



# ==================================================
# Correct numeral ids
# ==================================================

numeral_ids=[]


for d in range(10):

    ids = processor.tokenizer.encode(
        str(d),
        add_special_tokens=False
    )

    assert len(ids)==1, (
        f"Digit {d} tokenized incorrectly: {ids}"
    )

    numeral_ids.append(
        ids[0]
    )


print(
    "NUMERAL IDS:",
    numeral_ids
)


# ==================================================
# Loss
# ==================================================

last_logits = logits[:,-1,:]


selected = last_logits[:, numeral_ids]


print(
    "SELECTED LOGITS:",
    selected.shape
)


loss = (
    -torch
    .logsumexp(
        selected,
        dim=-1
    )
    .mean()
)


print(
    "LOSS:",
    loss.detach()
    .float()
    .item()
)



# ==================================================
# Backward
# ==================================================

print("\nBACKWARD")


loss.backward()



print("\n===== BACKWARD RESULT =====")


hidden=saved["hidden"]


print(
    "HAS GRAD:",
    hidden.grad is not None
)


grad = (
    hidden
    .grad
    .detach()
    .float()
)


print(
    "GRAD SHAPE:",
    grad.shape
)


print(
    "GRAD NORM:",
    grad.norm().item()
)


print(
    "GRAD ABS MEAN:",
    grad.abs().mean().item()
)


print(
    "GRAD MAX:",
    grad.abs().max().item()
)


print(
    "ALL ZERO:",
    torch.all(
        grad==0
    ).item()
)


handle.remove()


print("="*80)
print("CSA GRAPH DIAGNOSTIC v2 COMPLETE")
print("="*80)

