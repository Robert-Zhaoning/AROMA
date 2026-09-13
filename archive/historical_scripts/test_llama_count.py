import re
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
IMAGE_PATH = "data/sanity_count.png"
QUESTION = "How many red circles are there in the image?"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": QUESTION + "\nAnswer with only one integer.",
            },
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

inputs = processor(
    images=image,
    text=prompt,
    add_special_tokens=False,
    return_tensors="pt",
)

inputs = {
    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
    for k, v in inputs.items()
}

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
    )

input_len = inputs["input_ids"].shape[-1]
generated = output[:, input_len:]

answer = processor.batch_decode(
    generated,
    skip_special_tokens=True,
)[0].strip()

match = re.search(r"-?\d+", answer)
parsed = int(match.group()) if match else None

print("Question:", QUESTION)
print("Raw answer:", repr(answer))
print("Parsed count:", parsed)
print("Ground truth:", 5)
print("Correct:", parsed == 5)
