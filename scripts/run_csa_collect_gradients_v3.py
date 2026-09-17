
import os
import json
import numpy as np
import torch

from tqdm import tqdm
from PIL import Image

from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)


print("="*80)
print("AROMA CSA GRADIENT COLLECTION v3")
print("="*80)


MODEL_PATH="/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"


IMAGE_ROOT="data"

OUTPUT_DIR="outputs/csa_gradient_v3"

LAYER=18

MAX_IMAGES=300


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# --------------------------------------------------
# Load
# --------------------------------------------------

processor=AutoProcessor.from_pretrained(
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


layer=(
    model
    .model
    .language_model
    .layers[LAYER]
)


print("MODEL READY")
print("LAYER:",LAYER)



# --------------------------------------------------
# Numeral ids
# --------------------------------------------------

numeral_ids=[]

for d in range(10):

    ids=processor.tokenizer.encode(
        str(d),
        add_special_tokens=False
    )

    assert len(ids)==1

    numeral_ids.append(ids[0])


print(
    "NUMERAL IDS:",
    numeral_ids
)



# --------------------------------------------------
# Images
# --------------------------------------------------

images=[]


for root,_,files in os.walk(IMAGE_ROOT):

    for f in files:

        if f.lower().endswith(
            (".png",".jpg",".jpeg")
        ):

            images.append(
                os.path.join(root,f)
            )


images=images[:MAX_IMAGES]


print(
    "TOTAL IMAGES:",
    len(images)
)



# --------------------------------------------------
# Collect
# --------------------------------------------------

gradients=[]
paths=[]


for path in tqdm(images):

    saved={}


    def hook_fn(
        module,
        inputs,
        output
    ):

        hidden=(
            output[0]
            if isinstance(output,tuple)
            else output
        )

        hidden.retain_grad()

        saved["hidden"]=hidden



    handle=(
        layer
        .cross_attn
        .register_forward_hook(
            hook_fn
        )
    )


    try:

        image=Image.open(
            path
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


        grad=(
            saved["hidden"]
            .grad
            .detach()
            .float()
        )


        # mean over sequence
        grad=grad.mean(
            dim=1
        )


        gradients.append(
            grad.cpu().numpy()[0]
        )

        paths.append(path)


        print(
            "OK:",
            path,
            "norm=",
            float(
                grad.norm()
            )
        )


    except Exception as e:

        print(
            "FAILED:",
            path,
            e
        )


    finally:

        handle.remove()



# --------------------------------------------------
# Save
# --------------------------------------------------

G=np.stack(
    gradients
)


np.save(
    f"{OUTPUT_DIR}/gradients.npy",
    G
)


with open(
    f"{OUTPUT_DIR}/paths.json",
    "w"
) as f:

    json.dump(
        paths,
        f,
        indent=2
    )


print("="*80)

print(
    "GRADIENT MATRIX:",
    G.shape
)

print(
    "CSA COLLECTION COMPLETE"
)

print("="*80)

