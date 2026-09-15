
import torch

from transformers import (
    MllamaForConditionalGeneration,
)


MODEL_PATH="/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"


LAYER_TARGET=18


model=MllamaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()


layers = model.model.language_model.layers


print("="*80)
print("AROMA CSA CROSS ATTENTION GATE AUDIT")
print("="*80)


values=[]


for i,layer in enumerate(layers):

    if hasattr(
        layer,
        "cross_attn_attn_gate"
    ):

        g = (
            layer
            .cross_attn_attn_gate
            .detach()
            .float()
            .item()
        )

        t = torch.tanh(
            torch.tensor(g)
        ).item()

        values.append(g)

        mark=" <-- TARGET" if i==LAYER_TARGET else ""

        print(
            f"layer {i:02d}: gate={g:.8f}, tanh={t:.8f}{mark}"
        )


print("="*80)

print(
    "NUM CROSS ATTENTION LAYERS:",
    len(values)
)

print(
    "MIN:",
    min(values)
)

print(
    "MAX:",
    max(values)
)

print(
    "MEAN:",
    sum(values)/len(values)
)

