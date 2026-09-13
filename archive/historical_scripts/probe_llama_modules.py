import torch
from transformers import MllamaForConditionalGeneration


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

print("Loading model...")

model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

print("\n=== CROSS-ATTENTION-LIKE MODULES ===")

matches = []

for name, module in model.named_modules():
    lname = name.lower()

    if "cross" in lname and (
        "attn" in lname
        or "attention" in lname
    ):
        matches.append(
            (
                name,
                module.__class__.__name__,
            )
        )

for name, cls in matches:
    print(
        f"{name:100s} {cls}"
    )

print("\nTotal cross-attention-like modules:", len(matches))
