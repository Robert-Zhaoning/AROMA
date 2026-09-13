import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

META_PATH = Path(
    "data/proc_count_causal_v1/"
    "metadata.jsonl"
)

OUTPUT_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "baseline_results_eager.jsonl"
)

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


def extract_integer(text):
    match = re.search(
        r"(?<![\w.])\d+(?![\w.])",
        text,
    )

    if match is None:
        return None

    return int(
        match.group()
    )


def load_metadata():
    return [
        json.loads(line)
        for line in META_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def load_completed():
    if not OUTPUT_PATH.exists():
        return set()

    completed = set()

    for line in OUTPUT_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            r = json.loads(line)
            completed.add(
                r["sample_id"]
            )
        except Exception:
            pass

    return completed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    records = load_metadata()

    if args.limit is not None:
        records = records[
            :args.limit
        ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = (
        load_completed()
        if args.resume
        else set()
    )

    print("=" * 80)
    print(
        "Proc-Count-Causal v1 Baseline"
    )
    print("=" * 80)

    print(
        "Samples:",
        len(records),
    )

    print(
        "Resume :",
        args.resume,
    )

    print(
        "Done   :",
        len(completed),
    )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    print(
        "Loading model..."
    )

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    )

    model.eval()

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
            records,
            desc="Baseline",
        ):

            sample_id = (
                sample[
                    "sample_id"
                ]
            )

            if (
                sample_id
                in completed
            ):
                continue

            image = (
                Image.open(
                    sample[
                        "image_path"
                    ]
                )
                .convert("RGB")
            )

            messages = [
                {
                    "role":
                        "user",

                    "content": [
                        {
                            "type":
                                "image",
                        },
                        {
                            "type":
                                "text",
                            "text":
                                PROMPT,
                        },
                    ],
                }
            ]

            formatted = (
                processor
                .apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
            )

            inputs = processor(
                images=image,
                text=formatted,
                add_special_tokens=False,
                return_tensors="pt",
            )

            inputs = {
                k: (
                    v.to(
                        model.device
                    )
                    if torch.is_tensor(v)
                    else v
                )
                for k, v
                in inputs.items()
            }

            prompt_len = (
                inputs[
                    "input_ids"
                ].shape[-1]
            )

            with torch.inference_mode():

                generated = (
                    model.generate(
                        **inputs,
                        max_new_tokens=8,
                        do_sample=False,
                    )
                )

            new_ids = (
                generated[
                    :,
                    prompt_len:
                ]
            )

            text = (
                processor
                .batch_decode(
                    new_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                .strip()
            )

            pred = (
                extract_integer(
                    text
                )
            )

            record = {
                "sample_id":
                    sample_id,

                "ground_truth":
                    sample[
                        "ground_truth"
                    ],

                "condition":
                    sample[
                        "condition"
                    ],

                "replicate":
                    sample[
                        "replicate"
                    ],

                "seed":
                    sample[
                        "seed"
                    ],

                "prediction":
                    pred,

                "raw":
                    text,

                "parse_success":
                    pred is not None,

                "correct":
                    (
                        pred
                        == sample[
                            "ground_truth"
                        ]
                    ),
            }

            fout.write(
                json.dumps(
                    record
                )
                + "\n"
            )

            fout.flush()

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
