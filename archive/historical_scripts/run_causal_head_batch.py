import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

META_PATH = Path("data/calibration50/metadata.jsonl")
PAIR_PATH = Path(
    "outputs/calibration50/attention_pilot/clean_matched_pairs.csv"
)
OUTPUT_DIR = Path(
    "outputs/calibration50/causal_head_batch"
)

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()

CANDIDATES = [
    (33, 21),
    (33, 23),
    (13, 11),
    (13, 10),
]

RANDOM_CONTROLS = [
    (38, 5),
    (3, 30),
    (13, 0),
    (3, 19),
]

ALL_HEADS = [
    ("candidate", *x)
    for x in CANDIDATES
] + [
    ("random", *x)
    for x in RANDOM_CONTROLS
]


def load_metadata():
    out = {}

    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r

    return out


def extract_integer(text):
    m = re.search(r"(?<![\w.])\d+(?![\w.])", text)

    if m is None:
        return None

    return int(m.group())


def prepare_inputs(processor, image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    formatted = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    return processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )


def move_inputs(inputs, model):
    return {
        k: (
            v.to(model.device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in inputs.items()
    }


def get_numeral_token_ids(processor):
    token_ids = {}

    for n in range(1, 11):
        ids = processor.tokenizer.encode(
            str(n),
            add_special_tokens=False,
        )

        if len(ids) != 1:
            raise RuntimeError(
                f"Numeral {n} is not single-token: {ids}"
            )

        token_ids[n] = ids[0]

    return token_ids


def score_state(
    model,
    processor,
    device_inputs,
    numeral_token_ids,
    gt,
):
    with torch.inference_mode():
        outputs = model(
            **device_inputs,
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits[0, -1].float()

    numeral_logits = {
        n: float(logits[token_id].item())
        for n, token_id in numeral_token_ids.items()
    }

    gt_logit = numeral_logits[gt]

    wrong_logits = [
        value
        for n, value in numeral_logits.items()
        if n != gt
    ]

    best_wrong_logit = max(wrong_logits)

    gt_margin = (
        gt_logit
        - best_wrong_logit
    )

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    gt_token_id = numeral_token_ids[gt]

    gt_logp = float(
        log_probs[gt_token_id].item()
    )

    return {
        "gt_logit": gt_logit,
        "best_wrong_logit": best_wrong_logit,
        "gt_margin": gt_margin,
        "gt_logp": gt_logp,
    }


def generate_answer(
    model,
    processor,
    device_inputs,
):
    prompt_len = device_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated = model.generate(
            **device_inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )

    new_ids = generated[:, prompt_len:]

    text = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return text, extract_integer(text)


class HeadAblator:
    def __init__(
        self,
        model,
        layer_idx,
        head_idx,
    ):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.handle = None
        self.calls = 0

        layer = (
            model.model.language_model.layers[
                layer_idx
            ]
        )

        cross_attn = layer.cross_attn
        self.o_proj = cross_attn.o_proj

        self.num_heads = 32
        self.hidden_size = self.o_proj.in_features
        self.head_dim = self.hidden_size // self.num_heads

        self.start = head_idx * self.head_dim
        self.end = self.start + self.head_dim

    def hook(self, module, inputs):
        x = inputs[0]

        x_mod = x.clone()

        x_mod[
            ...,
            self.start:self.end
        ] = 0

        self.calls += 1

        if len(inputs) == 1:
            return (x_mod,)

        return (
            x_mod,
            *inputs[1:],
        )

    def register(self):
        self.calls = 0
        self.handle = (
            self.o_proj
            .register_forward_pre_hook(
                self.hook
            )
        )

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata()
    pairs = pd.read_csv(PAIR_PATH)

    sample_info = {}

    for _, row in pairs.iterrows():
        sample_info[row["wrong_sample_id"]] = {
            "role": row["wrong_label"],
        }

        sample_info[row["correct_sample_id"]] = {
            "role": "correct",
        }

    sample_ids = sorted(sample_info)

    print("=" * 80)
    print("AROMA Candidate-vs-Random Causal Head Batch")
    print("=" * 80)

    print("\nSamples:", len(sample_ids))

    print("\nFrozen candidate heads:")
    for _, layer, head in ALL_HEADS[:4]:
        print(f"  L{layer}H{head}")

    print("\nFrozen random controls:")
    for _, layer, head in ALL_HEADS[4:]:
        print(f"  L{layer}H{head}")

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    numeral_token_ids = get_numeral_token_ids(
        processor
    )

    print("Numeral tokens:", numeral_token_ids)

    print("\nLoading model...")

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    model.eval()

    rows = []

    for sample_id in tqdm(
        sample_ids,
        desc="Samples",
    ):
        sample = metadata[sample_id]
        gt = int(sample["ground_truth"])

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        inputs = prepare_inputs(
            processor,
            image,
        )

        device_inputs = move_inputs(
            inputs,
            model,
        )

        # -----------------------------------------------------
        # Baseline once per sample
        # -----------------------------------------------------

        baseline_text, baseline_pred = (
            generate_answer(
                model,
                processor,
                device_inputs,
            )
        )

        baseline_score = score_state(
            model,
            processor,
            device_inputs,
            numeral_token_ids,
            gt,
        )

        for (
            head_type,
            layer_idx,
            head_idx,
        ) in ALL_HEADS:

            ablator = HeadAblator(
                model,
                layer_idx,
                head_idx,
            )

            ablator.register()

            try:
                ablated_text, ablated_pred = (
                    generate_answer(
                        model,
                        processor,
                        device_inputs,
                    )
                )

                ablated_score = score_state(
                    model,
                    processor,
                    device_inputs,
                    numeral_token_ids,
                    gt,
                )

            finally:
                ablator.remove()

            if ablator.calls == 0:
                raise RuntimeError(
                    f"Hook not called for "
                    f"L{layer_idx}H{head_idx}"
                )

            rows.append(
                {
                    "sample_id": sample_id,
                    "role": sample_info[
                        sample_id
                    ]["role"],
                    "ground_truth": gt,
                    "condition": sample["condition"],

                    "head_type": head_type,
                    "layer": layer_idx,
                    "head": head_idx,

                    "baseline_text": baseline_text,
                    "baseline_pred": baseline_pred,
                    "baseline_correct":
                        baseline_pred == gt,

                    "ablated_text": ablated_text,
                    "ablated_pred": ablated_pred,
                    "ablated_correct":
                        ablated_pred == gt,

                    "prediction_changed":
                        baseline_pred
                        != ablated_pred,

                    "baseline_gt_logp":
                        baseline_score["gt_logp"],

                    "ablated_gt_logp":
                        ablated_score["gt_logp"],

                    "delta_gt_logp":
                        ablated_score["gt_logp"]
                        - baseline_score["gt_logp"],

                    "baseline_margin":
                        baseline_score["gt_margin"],

                    "ablated_margin":
                        ablated_score["gt_margin"],

                    "delta_margin":
                        ablated_score["gt_margin"]
                        - baseline_score["gt_margin"],

                    "hook_calls":
                        ablator.calls,
                }
            )

    df = pd.DataFrame(rows)

    out = OUTPUT_DIR / "causal_head_results.csv"

    df.to_csv(
        out,
        index=False,
    )

    print("\nSaved:", out)
    print("Rows:", len(df))
    print("\nCAUSAL BATCH COMPLETE")


if __name__ == "__main__":
    main()
