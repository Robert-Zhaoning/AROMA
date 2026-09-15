import numpy as np
import torch

from PIL import Image

from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)


print("="*80)
print("AROMA CSA TEACHER FORCING LOGIT TEST v3")
print("="*80)


MODEL_PATH="/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf1928a8b0e73f848f3a102794df5"

IMAGE_PATH="data/sanity_count.png"

BASIS_PATH="outputs/csa_gradient_v3/csa_basis.npy"

LAYER=18



# =============================
# Load model
# =============================

processor = AutoProcessor.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    local_files_only=True
)


model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()


layer = (
    model
    .model
    .language_model
    .layers[LAYER]
)


basis=np.load(BASIS_PATH)


print("BASIS:",basis.shape)



# =============================
# Random basis
# =============================

random_basis=np.random.randn(
    *basis.shape
)

random_basis /= np.linalg.norm(
    random_basis,
    axis=1,
    keepdims=True
)



# =============================
# Digit ids
# =============================

digit_ids=[]

for d in range(10):

    ids=processor.tokenizer.encode(
        str(d),
        add_special_tokens=False
    )

    digit_ids.append(ids[-1])


print(
    "DIGIT IDS:",
    digit_ids
)



# =============================
# Build teacher forcing input
# =============================

image=Image.open(
    IMAGE_PATH
).convert("RGB")


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
},
{
"role":"assistant",
"content":[
    {
    "type":"text",
    "text":"There are"
    }
]
}
]


prompt=processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
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



# =============================
# Evaluate
# =============================

def evaluate(condition,k):


    def hook(module,inputs,output):

        hidden = (
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


        remove = (
            h @ V.T @ V
        )


        new_hidden=(
            h-remove
        ).to(hidden.dtype)


        if isinstance(output,tuple):

            return (
                new_hidden,
                *output[1:]
            )

        return new_hidden



    handle=None

    if condition!="baseline":

        handle=(
            layer
            .cross_attn
            .register_forward_hook(
                hook
            )
        )


    with torch.no_grad():

        out=model(
            **inputs
        )


    logits=out.logits


    # prediction of next token after "There are"
    

    # =====================================================
    # FIX: USE TOKEN POSITION AFTER "There are"
    # =====================================================

    tokens = inputs["input_ids"][0].tolist()

    there_id = processor.tokenizer.convert_tokens_to_ids("There")
    are_id = processor.tokenizer.convert_tokens_to_ids("Ġare")

    answer_position = None

    for i in range(len(tokens)-1):
        if tokens[i] == there_id and tokens[i+1] == are_id:
            answer_position = i + 1


    if answer_position is None:
        raise RuntimeError(
            "Cannot find There are position"
        )


    print(
        "ANSWER POSITION:",
        answer_position,
        processor.tokenizer.convert_ids_to_tokens(
            [tokens[answer_position]]
        )
    )


    target_logits = logits[:, answer_position, :]


    probs=torch.softmax(
        target_logits,
        dim=-1
    )


    digit_probs=[
        probs[0,i].float().item()
        for i in digit_ids
    ]


    if handle:
        handle.remove()


    return digit_probs



# =============================
# Run
# =============================

for condition in [
    "baseline",
    "csa",
    "random"
]:

    for k in [1,5,10,20]:


        if condition=="baseline" and k!=1:
            continue


        print("\n================")
        print(condition,"k=",k)


        probs=evaluate(
            condition,
            k
        )


        for i,p in enumerate(probs):

            print(
                i,
                p
            )


        print(
            "GT(5):",
            probs[5]
        )



print("="*80)
print("COMPLETE")

