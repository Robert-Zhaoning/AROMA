
import torch
import numpy as np

from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)


print("="*80)
print("AROMA CSA LAYER SWEEP v1")
print("="*80)


MODEL_PATH="/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"


IMAGE_PATH="data/sanity_count.png"


LAYERS=[
    3,
    8,
    13,
    18,
    23,
    28,
    33,
    38
]


device="cuda"


# --------------------------------------------------
# load
# --------------------------------------------------

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


model=MllamaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


model.eval()


for p in model.parameters():
    p.requires_grad_(True)


print("MODEL LOADED")


text_layers=model.model.language_model.layers


# --------------------------------------------------
# numeral ids
# --------------------------------------------------

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



results=[]


# --------------------------------------------------
# sweep
# --------------------------------------------------

for LAYER in LAYERS:

    print("\n", "="*60)
    print("TEST LAYER",LAYER)


    layer=text_layers[LAYER]

    gate=(
        layer
        .cross_attn_attn_gate
        .detach()
        .float()
        .item()
    )


    saved={}


    def hook(module, inputs, output):

        if isinstance(output, tuple):
            hidden=output[0]
        else:
            hidden=output


        hidden.retain_grad()

        saved["hidden"]=hidden



    handle=(
        layer.cross_attn
        .register_forward_hook(hook)
    )


    try:

        model.zero_grad()


        outputs=model(
            **inputs
        )


        logits=outputs.logits


        last_logits=logits[:,-1,:]


        loss=(
            -torch
            .logsumexp(
                last_logits[:,numeral_ids],
                dim=-1
            )
            .mean()
        )


        loss.backward()


        if "hidden" not in saved:

            grad_norm=float("nan")

        else:

            grad=saved["hidden"].grad

            if grad is None:
                grad_norm=0.0

            else:
                grad_norm=(
                    grad
                    .float()
                    .norm()
                    .item()
                )


        print(
            f"LAYER={LAYER} "
            f"GATE={gate:.8f} "
            f"GRAD_NORM={grad_norm:.8e}"
        )


        results.append(
        {
            "layer":LAYER,
            "gate":gate,
            "grad_norm":grad_norm
        }
        )


    finally:

        handle.remove()

        torch.cuda.empty_cache()



print("\n")
print("="*80)
print("FINAL RESULTS")
print("="*80)


for r in results:

    print(
        r
    )


np.save(
    "outputs/csa_layer_sweep_v1.npy",
    np.array(
        [
        [
        r["layer"],
        r["gate"],
        r["grad_norm"]
        ]
        for r in results
        ]
    )
)


print("DONE")

