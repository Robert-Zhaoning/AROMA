import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

sys.path.insert(0, "scripts")

from run_l18h13_gain_all300 import (
    MODEL_ID,
    LAYER,
    HEAD,
    prepare_inputs,
    move_inputs,
    HeadGainModifier,
)


META_PATH = Path(
    "data/proc_count_causal_v2/metadata.jsonl"
)

SAMPLES = [
    "pccv2_n10_distractors_r06",
    "pccv2_n10_distractors_r07",
    "pccv2_n10_distractors_r10",
    "pccv2_n10_distractors_r11",
    "pccv2_n10_distractors_r17",
]

ALPHA = 1.5


def extract_integer(text):
    m = re.search(
        r"(?<![\w.])\d+(?![\w.])",
        text,
    )

    if m is None:
        return None

    return int(
        m.group()
    )


def generate(
    model,
    processor,
    inputs,
):
    prompt_len = (
        inputs["input_ids"]
        .shape[-1]
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=False,
        )

    text = (
        processor.batch_decode(
            output[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .strip()
    )

    return text, extract_integer(text)


metadata = {}

for line in META_PATH.read_text(
    encoding="utf-8"
).splitlines():

    if line.strip():
        r = json.loads(line)
        metadata[
            r["sample_id"]
        ] = r


print("=" * 100)
print("V2 BOUNDARY ACTUAL-GENERATION VALIDATION")
print("=" * 100)

processor = AutoProcessor.from_pretrained(
    MODEL_ID
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


for sid in SAMPLES:

    sample = metadata[sid]

    image = (
        Image.open(
            sample["image_path"]
        )
        .convert("RGB")
    )

    inputs = prepare_inputs(
        processor,
        image,
    )

    inputs = move_inputs(
        inputs,
        model,
    )

    base_text, base_pred = generate(
        model,
        processor,
        inputs,
    )

    modifier = HeadGainModifier(
        model=model,
        layer_idx=LAYER,
        head_idx=HEAD,
        alpha=ALPHA,
    )

    modifier.register()

    try:
        gain_text, gain_pred = generate(
            model,
            processor,
            inputs,
        )
    finally:
        modifier.remove()

    print(
        f"\n{sid}"
    )
    print(
        " GT       :",
        sample["ground_truth"]
    )
    print(
        " baseline :",
        base_pred,
        repr(base_text),
    )
    print(
        " gain     :",
        gain_pred,
        repr(gain_text),
    )
    print(
        " hook calls:",
        modifier.calls,
    )


print("\n" + "=" * 100)
print("BOUNDARY GENERATION VALIDATION COMPLETE")
print("=" * 100)
