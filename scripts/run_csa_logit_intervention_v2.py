
import numpy as np
import torch

from PIL import Image

from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)


print("="*80)
print("AROMA CSA LOGIT INTERVENTION v2")
print("="*80)


MODEL_PATH="/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"

IMAGE_PATH="data/sanity_count.png"

BASIS_PATH="outputs/csa_gradient_v3/csa_basis.npy"

LAYER=18



# -----------------------------
# Load
# -----------------------------

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



# -----------------------------
# random basis
# -----------------------------

random_basis=np.random.randn(
    *basis.shape
)


random_basis /= np.linalg.norm(
    random_basis,
    axis=1,
    keepdims=True
)



# -----------------------------
# tokens
# -----------------------------

digit_ids=[]

for d in range(10):

    ids=processor.tokenizer.encode(
        str(d),
        add_special_tokens=False
    )

    digit_ids.append(ids[0])


print(
    "DIGIT IDS:",
    digit_ids
)



# -----------------------------
# input
# -----------------------------

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



# -----------------------------
# run
# -----------------------------

def evaluate(condition,k):


    def hook(module,inputs,output):

        hidden=(
            output[0]
            if isinstance(output,tuple)
            else output
        )


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



        h=hidden.float()


        projection=(
            h @ V.T @ V
        )


        new_h=(
            h-projection
        ).to(hidden.dtype)


        if isinstance(output,tuple):

            return (
                new_h,
                *output[1:]
            )

        return new_h



    handle=None


    if condition!="baseline":

        handle=(
            layer.cross_attn
            .register_forward_hook(
                hook
            )
        )


    with torch.no_grad():

        outputs=model(
            **inputs
        )


    logits=outputs.logits


    last=logits[:,-1,:]


    probs=torch.softmax(
        last,
        dim=-1
    )


    digit_probs=[
        probs[0,i].float().item()
        for i in digit_ids
    ]


    if handle:
        handle.remove()


    return digit_probs



for condition in [
    "baseline",
    "csa",
    "random"
]:

    for k in [1,5,10,20]:

        if condition=="baseline" and k!=1:
            continue


        print("\n================")
        print(
            condition,
            "k=",
            k
        )


        p=evaluate(
            condition,
            k
        )


        print(
            "digit probabilities:"
        )

        for i,x in enumerate(p):
            print(
                i,
                x
            )

        print(
            "sum:",
            sum(p)
        )


print("="*80)
print("DONE")

