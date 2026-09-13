import json
import re
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
IMAGE_PATH = "data/sanity_count.png"
QUESTION = "How many red circles are there in the image?"


LEDGER_PROMPT = f"""
You are given an image and a counting question.

Question:
{QUESTION}

Your task is to enumerate EVERY visible object that satisfies the target description.

Important rules:
- Inspect the full image carefully from left to right and top to bottom.
- Include every distinct matching object exactly once.
- Do not include non-matching objects.
- Do not copy or repeat examples from this prompt.
- The number of entries in "objects" must exactly equal "ledger_count".
- Bounding boxes must be approximate normalized coordinates:
  [x1, y1, x2, y2], with values between 0 and 1.
- Confidence must be between 0 and 1.
- Do NOT provide a separate final answer.
- Return ONLY one valid JSON object.
- Do NOT use Markdown code fences.
- Do NOT add any text before or after the JSON.

Use this exact output structure:

{{
  "objects": [
    {{
      "id": "O1",
      "category": "<target category>",
      "box": [x1, y1, x2, y2],
      "confidence": confidence
    }}
  ],
  "ledger_count": integer
}}

Now inspect the actual image and populate the JSON with the real matching objects.
"""


def extract_json_object(text: str):
    """
    Robustly extract the first top-level JSON object from model text.
    This tolerates accidental text before/after the JSON.
    """
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
                candidate = text[start:i + 1]
                return json.loads(candidate)

    raise ValueError("Unclosed JSON object.")


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
            {"type": "text", "text": LEDGER_PROMPT},
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
        max_new_tokens=768,
        do_sample=False,
    )

input_len = inputs["input_ids"].shape[-1]
generated = output[:, input_len:]

text = processor.batch_decode(
    generated,
    skip_special_tokens=True,
)[0].strip()

print("=== RAW OUTPUT ===")
print(text)

try:
    data = extract_json_object(text)

    print("\n=== PARSED JSON ===")
    print(json.dumps(data, indent=2))

    objects = data.get("objects", [])
    ledger_count = data.get("ledger_count")

    print("\n=== LEDGER CHECK ===")
    print("Number of object records:", len(objects))
    print("Ledger count:", ledger_count)
    print("Internal consistency:", len(objects) == ledger_count)
    print("Ground-truth count:", 5)
    print("Ledger count correct:", ledger_count == 5)

    valid_boxes = 0

    for obj in objects:
        box = obj.get("box")

        if (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(x, (int, float)) for x in box)
            and all(0 <= x <= 1 for x in box)
            and box[0] < box[2]
            and box[1] < box[3]
        ):
            valid_boxes += 1

    print("Valid boxes:", valid_boxes, "/", len(objects))

except Exception as e:
    print("\nJSON extraction/parsing failed:")
    print(repr(e))
