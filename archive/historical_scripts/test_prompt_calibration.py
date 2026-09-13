import json
import re
from collections import OrderedDict

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
IMAGE_PATH = "data/sanity_count.png"

GROUND_TRUTH = 5
QUESTION = "How many red circles are there in the image?"


def extract_integer(text):
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def extract_json(text):
    start = text.find("{")

    if start == -1:
        return None

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

                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None

    return None


PROMPTS = OrderedDict({

    # ----------------------------------------------------------
    # P0: simplest possible direct-count baseline
    # ----------------------------------------------------------
    "direct": """
Look carefully at the image.

How many red circles are visible?

Return the number only.
""",

    # ----------------------------------------------------------
    # P1: count-only JSON, with NO numerical example
    # ----------------------------------------------------------
    "count_json_no_example": """
Inspect the image carefully.

Count every visible red circle exactly once.

Return one JSON object containing a single field named
"count".

The value of "count" must be the number of visible red circles.

Do not copy a demonstration.
No demonstration is provided.

Return JSON only.
""",

    # ----------------------------------------------------------
    # P2: enumeration, but NO object-count example
    # ----------------------------------------------------------
    "enumeration_no_example": """
Inspect the actual image carefully.

Identify every distinct visible red circle.

Scan systematically from left to right and from top to bottom.

Create one identifier for each red circle you actually see.
Identifiers must begin with O1 and increase consecutively.

Return a JSON object with two fields:

"objects":
a list of the identifiers for all visible red circles.

"ledger_count":
the number of identifiers in "objects".

Do not infer the number from this instruction.
Do not copy an example.
No example count is provided.

Do not include blue squares.

Return JSON only.
""",

    # ----------------------------------------------------------
    # P3: spatial verbal enumeration
    #
    # Relative descriptions force the model to provide evidence
    # beyond merely emitting N copies of O_i.
    # ----------------------------------------------------------
    "spatial_enumeration": """
Inspect the actual image carefully.

Find every distinct visible red circle.

Scan the complete image from left to right and top to bottom.

For every red circle you see, provide a very short relative
location description such as "upper-left", "upper-center",
"upper-right", or another concise description appropriate to
its actual position.

Do not provide bounding-box coordinates.

Return a JSON object containing:

"objects":
a list of objects, each containing:
- "id": a unique consecutive identifier
- "position": a short relative position description

"ledger_count":
the number of listed objects.

The number of listed objects must be based only on what is
actually visible in the image.

Do not use Markdown.
Return JSON only.
""",
})


print("=" * 72)
print("AROMA Prompt Calibration")
print("=" * 72)

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")

results = {}


for prompt_name, prompt_text in PROMPTS.items():

    print("\n" + "=" * 72)
    print("PROMPT:", prompt_name)
    print("=" * 72)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text.strip()},
            ],
        }
    ]

    formatted_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=image,
        text=formatted_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]

    generated = output[:, input_length:]

    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    parsed_json = extract_json(text)

    inferred_count = None

    if prompt_name == "direct":
        inferred_count = extract_integer(text)

    elif parsed_json is not None:

        if prompt_name == "count_json_no_example":
            inferred_count = parsed_json.get("count")

        else:
            objects = parsed_json.get("objects", [])
            ledger_count = parsed_json.get("ledger_count")

            if isinstance(ledger_count, int):
                inferred_count = ledger_count

            print("Object records:", len(objects))
            print("Ledger count:", ledger_count)

    results[prompt_name] = {
        "raw": text,
        "parsed_count": inferred_count,
        "correct": inferred_count == GROUND_TRUTH,
    }

    print("\nRAW OUTPUT:")
    print(text)

    print("\nPARSED COUNT:", inferred_count)
    print("GROUND TRUTH:", GROUND_TRUTH)
    print("CORRECT:", inferred_count == GROUND_TRUTH)


print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

for name, result in results.items():
    print(
        f"{name:28s} "
        f"count={str(result['parsed_count']):>4s} "
        f"correct={result['correct']}"
    )

print("\nGround truth:", GROUND_TRUTH)
