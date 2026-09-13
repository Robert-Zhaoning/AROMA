import json
import torch

from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
IMAGE_PATH = "data/sanity_count.png"
QUESTION = "How many red circles are there in the image?"


PROMPT = f"""
You are given an image and a visual counting question.

Question:
{QUESTION}

Carefully inspect the actual image.

Your task is ONLY to enumerate the distinct visible target objects.

Do not estimate bounding boxes.
Do not output coordinates.
Do not count non-target objects.
Do not invent objects that are not visible.

Scan the image systematically from left to right and top to bottom.

For every distinct matching object, create exactly one object record.

Return ONLY one valid JSON object with this schema:

{{
  "objects": [
    {{
      "id": "O1"
    }}
  ],
  "ledger_count": 1
}}

Requirements:
- Each visible matching object must appear exactly once.
- ledger_count must equal the number of entries in objects.
- Do not provide any explanation.
- Do not provide a separate final answer.
- Do not use Markdown.
"""


def extract_json_object(text: str):
    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found.")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False

            continue

        if ch == '"':
            in_string = True

        elif ch == "{":
            depth += 1

        elif ch == "}":
            depth -= 1

            if depth == 0:
                return json.loads(text[start:i + 1])

    raise ValueError("Unclosed JSON object.")


print("Loading processor...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")

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
            {"type": "text", "text": PROMPT},
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
        max_new_tokens=256,
        do_sample=False,
    )

input_len = inputs["input_ids"].shape[-1]

generated = output[:, input_len:]

text = processor.batch_decode(
    generated,
    skip_special_tokens=True,
)[0].strip()

print("\n=== RAW OUTPUT ===")
print(text)

try:
    data = extract_json_object(text)

    objects = data.get("objects", [])
    ledger_count = data.get("ledger_count")

    print("\n=== PARSED ===")
    print(json.dumps(data, indent=2))

    print("\n=== CHECK ===")
    print("Number of records :", len(objects))
    print("Ledger count      :", ledger_count)
    print("Internal consistency:", len(objects) == ledger_count)
    print("Ground truth      :", 5)
    print("Count correct     :", ledger_count == 5)

except Exception as exc:
    print("\nPARSE FAILED")
    print(repr(exc))
