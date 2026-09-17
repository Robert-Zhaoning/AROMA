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


MODEL_PATH = "/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"

IMAGE_DIR = "data"

LAYER = 18

MAX_SAMPLES = 1


OUT = "outputs/csa_gradient_collection_v1"

os.makedirs(
    OUT,
    exist_ok=True
)


device="cuda"


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



layer = model.model.language_model.layers[LAYER]


cross_attn = layer.cross_attn



def get_gradient(image):

    saved={}


    def hook(module, inputs, output):

        hidden = output[0]

        hidden.retain_grad()

        saved["hidden"]=hidden


    h = cross_attn.register_forward_hook(
        hook
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


    prompt = processor.apply_chat_template(
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


    outputs=model(
        **inputs
    )


    logits=outputs.logits


    numeral_ids=[
        processor.tokenizer.encode(
            " "+str(i),
            add_special_tokens=False
        )[0]
        for i in range(10)
    ]


    loss=-torch.logsumexp(
        logits[:,-1,numeral_ids],
        dim=-1
    ).mean()


    loss.backward()


    grad=saved["hidden"].grad


    grad=(
        grad[:,-1,:]
        .mean(dim=0)
        .detach()
        .float()
        .cpu()
        .numpy()
    )


    h.remove()


    return grad



images=[]


for root,_,files in os.walk(IMAGE_DIR):

    for f in files:

        if f.endswith(".png") or f.endswith(".jpg"):

            images.append(
                os.path.join(root,f)
            )


images=sorted(images)[:MAX_SAMPLES]


print(
    "Samples:",
    len(images)
)


G=[]


for p in tqdm(images):

    try:

        img=Image.open(p).convert("RGB")

        g=get_gradient(img)

        G.append(g)


    except Exception as e:

        print(
            "FAILED SAMPLE:",
            p
        )

        import traceback
        traceback.print_exc()


G=np.stack(G)


print(
    "Gradient matrix:",
    G.shape
)


np.save(
    f"{OUT}/gradient_matrix.npy",
    G
)


with open(
    f"{OUT}/metadata.json",
    "w"
) as f:

    json.dump(
    {
        "layer":18,
        "hidden_dim":4096,
        "samples":int(G.shape[0])
    },
    f,
    indent=2
    )


print("DONE")
