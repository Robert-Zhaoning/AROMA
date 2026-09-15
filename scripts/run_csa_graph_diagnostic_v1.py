
import torch

from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)


print("="*80)
print("AROMA CSA GRAPH CONNECTIVITY DIAGNOSTIC v1")
print("="*80)


MODEL_PATH="/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"

IMAGE_PATH="data/sanity_count.png"

LAYER=18


# --------------------------------------------------
# Load
# --------------------------------------------------

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


print(
    "LAYER:",
    LAYER
)


print(
    "GATE:",
    layer.cross_attn_attn_gate
    .detach()
    .float()
    .item()
)


saved={}


# --------------------------------------------------
# Hook
# --------------------------------------------------

def hook_fn(
    module,
    inputs,
    output
):

    if isinstance(output, tuple):
        hidden=output[0]
    else:
        hidden=output


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


    if hidden.requires_grad:
        hidden.retain_grad()


    saved["hidden"]=hidden



handle = (
    layer
    .cross_attn
    .register_forward_hook(
        hook_fn
    )
)


# --------------------------------------------------
# Input
# --------------------------------------------------

image=Image.open(
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


prompt=processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs=processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)


inputs={
    k:v.to(model.device)
    for k,v in inputs.items()
}


print("INPUT READY")


# --------------------------------------------------
# Forward
# --------------------------------------------------

model.zero_grad()


outputs=model(
    **inputs
)


logits=outputs.logits


print(
    "LOGITS:",
    logits.shape
)


# check numeral tokens

numeral_ids=[]

for d in range(10):

    ids=processor.tokenizer.encode(
        " "+str(d),
        add_special_tokens=False
    )

    if len(ids)==1:
        numeral_ids.append(ids[0])


print(
    "NUMERAL IDS:",
    numeral_ids
)


last_logits=logits[:,-1,:]


loss=(
    -torch
    .logsumexp(
        last_logits[:,numeral_ids],
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


print("\nBACKWARD")


loss.backward()


print("\n===== BACKWARD RESULT =====")


if "hidden" not in saved:

    print(
        "ERROR: hook did not capture tensor"
    )

else:

    hidden=saved["hidden"]


    print(
        "HAS GRAD OBJECT:",
        hidden.grad is not None
    )


    if hidden.grad is not None:

        grad=(
            hidden.grad
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
print("DONE")
print("="*80)

