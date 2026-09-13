import argparse
import json
import re
import time
from collections import Counter, OrderedDict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

METADATA_PATH = Path(
    "data/calibration50/metadata.jsonl"
)

OUTPUT_PATH = Path(
    "outputs/calibration50/results.jsonl"
)


PROMPTS = OrderedDict(
    {
        "direct": """
Inspect the actual image carefully.

How many red circles are visible?

Return the number only.
""",

        "count_json": """
Inspect the actual image carefully.

Count every distinct visible red circle exactly once.

Do not count objects of any other color or shape.

Return exactly one JSON object with one field named "count".
The value of "count" must be the number of visible red circles.

No example answer is provided.

Return JSON only.
""",

        "enumeration": """
Inspect the actual image carefully.

Identify every distinct visible red circle.

Scan the entire image systematically.

Create exactly one unique identifier for each red circle
that is actually visible.

Do not count any blue squares or other non-target objects.

Return exactly one JSON object containing:

"objects":
a list of unique identifiers for all visible red circles.

"ledger_count":
the number of identifiers in "objects".

The object list must be derived from the actual image.
No example count is provided.

Return JSON only.
""",

        "spatial_enumeration": """
Inspect the actual image carefully.

Identify every distinct visible red circle.

Scan the complete image systematically.

For each distinct red circle, provide:
- a unique identifier
- a short relative spatial description of its actual position

Do not provide bounding-box coordinates.

Do not count blue squares or other non-target objects.

Return exactly one JSON object containing:

"objects":
a list of object records with fields "id" and "position".

"ledger_count":
the number of listed objects.

No example count is provided.

Return JSON only.
""",
    }
)


def extract_first_integer(text):
    match = re.search(
        r"(?<![\w.])-?\d+(?![\w.])",
        text,
    )

    if match is None:
        return None

    try:
        return int(match.group())
    except ValueError:
        return None


def extract_json_object(text):
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


def parse_prediction(prompt_name, text):
    parsed_json = None
    parsed_count = None
    object_count = None
    internally_consistent = None

    if prompt_name == "direct":
        parsed_count = extract_first_integer(text)

    else:
        parsed_json = extract_json_object(text)

        if parsed_json is None:
            return {
                "raw": text,
                "json_parse_success": False,
                "parsed_json": None,
                "parsed_count": None,
                "object_count": None,
                "internally_consistent": None,
            }

        if prompt_name == "count_json":
            value = parsed_json.get("count")

            if isinstance(value, int):
                parsed_count = value

        else:
            objects = parsed_json.get("objects", [])

            if isinstance(objects, list):
                object_count = len(objects)

            ledger_count = parsed_json.get(
                "ledger_count"
            )

            if isinstance(ledger_count, int):
                parsed_count = ledger_count

            if (
                object_count is not None
                and parsed_count is not None
            ):
                internally_consistent = (
                    object_count == parsed_count
                )

    return {
        "raw": text,
        "json_parse_success": (
            True
            if prompt_name == "direct"
            else parsed_json is not None
        ),
        "parsed_json": parsed_json,
        "parsed_count": parsed_count,
        "object_count": object_count,
        "internally_consistent": internally_consistent,
    }


def load_metadata():
    records = []

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def load_completed_ids():
    if not OUTPUT_PATH.exists():
        return set()

    completed = set()

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                completed.add(
                    record["sample_id"]
                )
            except Exception:
                pass

    return completed


def run_prompt(
    model,
    processor,
    image,
    prompt_text,
    max_new_tokens,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt_text.strip(),
                },
            ],
        }
    ]

    formatted = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )

    inputs = {
        key: (
            value.to(model.device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start

    input_length = inputs[
        "input_ids"
    ].shape[-1]

    generated = output[:, input_length:]

    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    peak_vram_gb = None

    if torch.cuda.is_available():
        peak_vram_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

    return (
        text,
        elapsed,
        peak_vram_gb,
    )


def calculate_prompt_stability(predictions):
    values = [
        value
        for value in predictions
        if isinstance(value, int)
    ]

    if not values:
        return None

    counter = Counter(values)

    modal_frequency = max(
        counter.values()
    )

    return modal_frequency / len(values)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results.jsonl file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N samples after filtering.",
    )

    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Only run samples whose ground-truth count "
            "is in this list. Example: --counts 1 5 10"
        ),
    )

    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata()

    if args.counts is not None:
        allowed_counts = set(args.counts)

        metadata = [
            sample
            for sample in metadata
            if sample["ground_truth"] in allowed_counts
        ]

    if args.limit is not None:
        metadata = metadata[:args.limit]

    completed_ids = (
        load_completed_ids()
        if args.resume
        else set()
    )

    print("=" * 72)
    print("AROMA Calibration-50 Runner")
    print("=" * 72)

    print("Model      :", MODEL_ID)
    print("Samples    :", len(metadata))
    print("Counts     :", args.counts)
    print("Resume     :", args.resume)
    print("Completed  :", len(completed_ids))
    print("Output     :", OUTPUT_PATH)

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    print("Loading model...")

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    model.eval()

    print("\nModel loaded.")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "Allocated VRAM GB:",
            round(
                torch.cuda.memory_allocated()
                / (1024 ** 3),
                3,
            ),
        )

    mode = (
        "a"
        if args.resume
        else "w"
    )

    with OUTPUT_PATH.open(
        mode,
        encoding="utf-8",
    ) as fout:

        for sample in tqdm(
            metadata,
            desc="Calibration-50",
        ):
            sample_id = sample[
                "sample_id"
            ]

            if sample_id in completed_ids:
                continue

            image = Image.open(
                sample["image_path"]
            ).convert("RGB")

            outputs = {}

            for (
                prompt_name,
                prompt_text,
            ) in PROMPTS.items():

                max_new_tokens = (
                    32
                    if prompt_name
                    in {
                        "direct",
                        "count_json",
                    }
                    else 256
                )

                (
                    raw_text,
                    elapsed,
                    peak_vram_gb,
                ) = run_prompt(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt_text=prompt_text,
                    max_new_tokens=max_new_tokens,
                )

                parsed = parse_prediction(
                    prompt_name,
                    raw_text,
                )

                parsed[
                    "elapsed_seconds"
                ] = elapsed

                parsed[
                    "peak_vram_gb"
                ] = peak_vram_gb

                parsed[
                    "correct"
                ] = (
                    parsed[
                        "parsed_count"
                    ]
                    == sample[
                        "ground_truth"
                    ]
                )

                outputs[
                    prompt_name
                ] = parsed

            predicted_counts = [
                outputs[name][
                    "parsed_count"
                ]
                for name in PROMPTS.keys()
            ]

            prompt_stability = (
                calculate_prompt_stability(
                    predicted_counts
                )
            )

            record = {
                "sample_id":
                    sample_id,

                "image_path":
                    sample[
                        "image_path"
                    ],

                "ground_truth":
                    sample[
                        "ground_truth"
                    ],

                "condition":
                    sample[
                        "condition"
                    ],

                "seed":
                    sample[
                        "seed"
                    ],

                "model_id":
                    MODEL_ID,

                "decoding": {
                    "do_sample": False,
                    "dtype": "bfloat16",
                },

                "outputs":
                    outputs,

                "prompt_stability":
                    prompt_stability,
            }

            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            fout.flush()

    print("\nCalibration run complete.")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
