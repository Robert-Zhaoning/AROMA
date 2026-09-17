
import os
import json
import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)


print("="*80)
print("AROMA CSA SUBSPACE INTERVENTION v1")
print("="*80)


MODEL_PATH="/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"


IMAGE_PATH="data/sanity_count.png"

BASIS_PATH="outputs/csa_gradient_v3/csa_basis.npy"


LAYER=18


# ==================================================
# Load
# ==================================================

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


basis=np.load(
    BASIS_PATH
)


print(
    "BASIS:",
    basis.shape
)


# ==================================================
# random null
# ==================================================

random_basis=np.random.randn(
    *basis.shape
)

random_basis /= np.linalg.norm(
    random_basis,
    axis=1,
    keepdims=True
)



# ==================================================
# token ids
# ==================================================

numeral_ids=[]

for d in range(10):

    ids=processor.tokenizer.encode(
        str(d),
        add_special_tokens=False
    )

    numeral_ids.append(ids[0])


# ==================================================
# intervention function
# ==================================================

def run(condition,k):


    saved={}


    def hook(
        module,
        inputs,
        output
    ):


        hidden = (
            output[0]
            if isinstance(output,tuple)
            else output
        )


        hidden_float=hidden.float()


        if condition=="csa":

            V=torch.tensor(
                basis[:k],
                device=hidden.device,
                dtype=torch.float32
            )


        elif condition=="random":

            V=torch.tensor(
                random_basis[:k],
                device=hidden.device,
                dtype=torch.float32
            )


        else:

            return output


        # projection removal

        proj = (
            hidden_float
            @ V.T
            @ V
        )


        new_hidden = (
            hidden_float
            - proj
        ).to(hidden.dtype)


        if isinstance(output,tuple):

            return (
                new_hidden,
                *output[1:]
            )

        return new_hidden



    handle=(
        layer
        .cross_attn
        .register_forward_hook(
            hook
        )
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
        {"type":"image"},
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


    with torch.no_grad():

        out=model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )


    text=processor.decode(
        out[0],
        skip_special_tokens=True
    )


    handle.remove()


    return text



# ==================================================
# Run
# ==================================================

for cond in [
    "baseline",
    "csa",
    "random"
]:

    for k in [1,5,10,20]:

        print(
            "\n",
            cond,
            "k=",
            k
        )

        print(
            run(cond,k)
        )


print("="*80)
print("DONE")

