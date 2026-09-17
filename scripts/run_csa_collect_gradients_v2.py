import os
import json
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)


print("="*80)
print("AROMA 2.0 CSA GRADIENT COLLECTION v2")
print("="*80)


MODEL_PATH = "/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"


IMAGE_DIR = "data"

LAYER = 18

MAX_SAMPLES = 10


OUT_DIR = (
    "outputs/csa_gradient_collection_v2"
)


os.makedirs(
    OUT_DIR,
    exist_ok=True
)


device = "cuda"


# ============================================================
# LOAD MODEL
# ============================================================

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


for p in model.parameters():
    p.requires_grad_(False)


print("MODEL LOADED")


# ============================================================
# MODULE
# ============================================================

layer = (
    model
    .model
    .language_model
    .layers[LAYER]
)


cross_attn = layer.cross_attn


print(
    "CROSS ATTENTION:",
    type(cross_attn)
)


# ============================================================
# GRADIENT FUNCTION
# ============================================================

def compute_gradient(image):

    saved={}


    def hook(module, inputs, output):

        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None


        # IMPORTANT:
        # recreate graph like v4
        hidden = (
            hidden
            .detach()
            .clone()
            .requires_grad_(True)
        )


        hidden.retain_grad()


        saved["hidden"] = hidden


        if rest:
            return (
                hidden,
                *rest
            )

        return hidden



    handle = cross_attn.register_forward_hook(
        hook
    )


    try:


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


        inputs={
            k:v.to(model.device)
            for k,v in inputs.items()
        }


        model.zero_grad()


        outputs=model(
            **inputs
        )


        logits=outputs.logits


        # verify
        last_logits = logits[:,-1,:]


        numeral_ids=[]


        for d in range(10):

            ids=processor.tokenizer.encode(
                " "+str(d),
                add_special_tokens=False
            )

            if len(ids)==1:
                numeral_ids.append(ids[0])


        numeral_ids=list(
            sorted(set(numeral_ids))
        )


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
            raise RuntimeError(
                "hook failed"
            )


        grad=saved["hidden"].grad


        if grad is None:
            raise RuntimeError(
                "gradient missing"
            )


        # only last token position
        grad=(
            grad[:,-1,:]
            .mean(dim=0)
            .detach()
            .float()
            .cpu()
            .numpy()
        )


        return grad


    finally:

        handle.remove()

        torch.cuda.empty_cache()



# ============================================================
# FIND IMAGES
# ============================================================

images=[]


for root,_,files in os.walk(
    IMAGE_DIR
):

    for f in files:

        if f.endswith(".png") or f.endswith(".jpg"):

            images.append(
                os.path.join(root,f)
            )


images=sorted(images)[:MAX_SAMPLES]


print(
    "TOTAL IMAGES:",
    len(images)
)


# ============================================================
# COLLECTION
# ============================================================


G=[]

success=[]


for path in tqdm(images):

    try:

        img=Image.open(
            path
        ).convert(
            "RGB"
        )


        g=compute_gradient(
            img
        )


        G.append(g)

        success.append(
            path
        )


        print(
            "OK:",
            path,
            np.linalg.norm(g)
        )


    except Exception as e:

        print(
            "FAILED:",
            path
        )

        print(
            repr(e)
        )



if len(G)==0:

    raise RuntimeError(
        "No gradients collected"
    )


G=np.stack(
    G
)


print(
    "GRADIENT MATRIX:",
    G.shape
)



np.save(
    f"{OUT_DIR}/gradient_matrix.npy",
    G
)



with open(
    f"{OUT_DIR}/metadata.json",
    "w"
) as f:

    json.dump(
    {
        "layer":LAYER,
        "hidden_dim":4096,
        "samples":len(G),
        "successful_images":success
    },
    f,
    indent=2
    )



print("="*80)
print("CSA COLLECTION COMPLETE")
print("="*80)
