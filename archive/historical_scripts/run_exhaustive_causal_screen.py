import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

META_PATH = Path(
    "data/calibration50/metadata.jsonl"
)

PAIR_PATH = Path(
    "outputs/calibration50/attention_pilot/"
    "clean_matched_pairs.csv"
)

OUTPUT_DIR = Path(
    "outputs/calibration50/exhaustive_causal"
)

CROSS_LAYERS = [
    3,
    8,
    13,
    18,
    23,
    28,
    33,
    38,
]

NUM_HEADS = 32

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


def load_metadata():
    result = {}

    with META_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            result[
                record["sample_id"]
            ] = record

    return result


def prepare_inputs(
    processor,
    image,
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
                    "text": PROMPT,
                },
            ],
        }
    ]

    formatted = (
        processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    )

    return processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )


def move_inputs(
    inputs,
    model,
):
    return {
        key: (
            value.to(model.device)
            if torch.is_tensor(value)
            else value
        )
        for key, value
        in inputs.items()
    }


def get_numeral_token_ids(
    processor,
):
    result = {}

    for n in range(1, 11):

        ids = (
            processor.tokenizer.encode(
                str(n),
                add_special_tokens=False,
            )
        )

        if len(ids) != 1:
            raise RuntimeError(
                f"Numeral {n} "
                f"is not single token: {ids}"
            )

        result[n] = ids[0]

    return result


def score_gt(
    model,
    device_inputs,
    numeral_ids,
    gt,
):
    """
    Primary metric: GT full-vocabulary log probability.

    Secondary metric:
      GT numeral logit minus best competing numeral logit.
    """

    with torch.inference_mode():
        outputs = model(
            **device_inputs,
            use_cache=False,
            return_dict=True,
        )

    logits = (
        outputs.logits[
            0,
            -1,
            :
        ]
        .float()
    )

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    gt_id = numeral_ids[gt]

    gt_logp = float(
        log_probs[
            gt_id
        ].item()
    )

    numeral_logits = {
        n: float(
            logits[token_id].item()
        )
        for n, token_id
        in numeral_ids.items()
    }

    gt_logit = (
        numeral_logits[gt]
    )

    best_wrong = max(
        value
        for n, value
        in numeral_logits.items()
        if n != gt
    )

    margin = (
        gt_logit
        - best_wrong
    )

    return {
        "gt_logp":
            gt_logp,

        "gt_margin":
            margin,
    }


class HeadAblator:

    def __init__(
        self,
        model,
        layer_idx,
        head_idx,
    ):
        self.model = model

        self.layer_idx = (
            layer_idx
        )

        self.head_idx = (
            head_idx
        )

        layer = (
            model
            .model
            .language_model
            .layers[
                layer_idx
            ]
        )

        self.o_proj = (
            layer
            .cross_attn
            .o_proj
        )

        hidden_size = (
            self.o_proj
            .in_features
        )

        self.head_dim = (
            hidden_size
            // NUM_HEADS
        )

        self.start = (
            head_idx
            * self.head_dim
        )

        self.end = (
            self.start
            + self.head_dim
        )

        self.handle = None
        self.calls = 0

    def hook(
        self,
        module,
        inputs,
    ):
        x = inputs[0]

        modified = (
            x.clone()
        )

        modified[
            ...,
            self.start:self.end
        ] = 0

        self.calls += 1

        if len(inputs) == 1:
            return (
                modified,
            )

        return (
            modified,
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

    metadata = (
        load_metadata()
    )

    pairs = pd.read_csv(
        PAIR_PATH
    )

    sample_info = {}

    for _, row in (
        pairs.iterrows()
    ):

        sample_info[
            row["wrong_sample_id"]
        ] = {
            "role":
                row["wrong_label"],
        }

        sample_info[
            row["correct_sample_id"]
        ] = {
            "role":
                "correct",
        }

    sample_ids = sorted(
        sample_info.keys()
    )

    print("=" * 80)
    print(
        "AROMA Exhaustive "
        "Cross-Attention Causal Screen"
    )
    print("=" * 80)

    print(
        "Samples:",
        len(sample_ids),
    )

    print(
        "Cross layers:",
        CROSS_LAYERS,
    )

    print(
        "Heads/layer:",
        NUM_HEADS,
    )

    print(
        "Total heads:",
        len(CROSS_LAYERS)
        * NUM_HEADS,
    )

    print(
        "Expected interventions:",
        len(sample_ids)
        * len(CROSS_LAYERS)
        * NUM_HEADS,
    )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID
        )
    )

    numeral_ids = (
        get_numeral_token_ids(
            processor
        )
    )

    print(
        "Numeral token IDs:",
        numeral_ids,
    )

    print(
        "\nLoading model..."
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

    rows = []

    total = (
        len(sample_ids)
        * len(CROSS_LAYERS)
        * NUM_HEADS
    )

    progress = tqdm(
        total=total,
        desc="Causal interventions",
    )

    for sample_id in sample_ids:

        sample = metadata[
            sample_id
        ]

        gt = int(
            sample[
                "ground_truth"
            ]
        )

        role = (
            sample_info[
                sample_id
            ]["role"]
        )

        image = (
            Image.open(
                sample[
                    "image_path"
                ]
            )
            .convert("RGB")
        )

        inputs = (
            prepare_inputs(
                processor,
                image,
            )
        )

        device_inputs = (
            move_inputs(
                inputs,
                model,
            )
        )

        # -----------------------------------------------------
        # Baseline once per sample
        # -----------------------------------------------------

        baseline = score_gt(
            model,
            device_inputs,
            numeral_ids,
            gt,
        )

        # -----------------------------------------------------
        # Exhaustive cross-attention head ablation
        # -----------------------------------------------------

        for layer_idx in CROSS_LAYERS:

            for head_idx in range(
                NUM_HEADS
            ):

                ablator = (
                    HeadAblator(
                        model=model,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                    )
                )

                ablator.register()

                try:

                    ablated = score_gt(
                        model,
                        device_inputs,
                        numeral_ids,
                        gt,
                    )

                finally:
                    ablator.remove()

                if (
                    ablator.calls
                    == 0
                ):
                    raise RuntimeError(
                        f"Hook not called for "
                        f"{sample_id} "
                        f"L{layer_idx}H{head_idx}"
                    )

                rows.append(
                    {
                        "sample_id":
                            sample_id,

                        "role":
                            role,

                        "ground_truth":
                            gt,

                        "condition":
                            sample[
                                "condition"
                            ],

                        "layer":
                            layer_idx,

                        "head":
                            head_idx,

                        "baseline_gt_logp":
                            baseline[
                                "gt_logp"
                            ],

                        "ablated_gt_logp":
                            ablated[
                                "gt_logp"
                            ],

                        "delta_gt_logp":
                            (
                                ablated[
                                    "gt_logp"
                                ]
                                - baseline[
                                    "gt_logp"
                                ]
                            ),

                        "baseline_margin":
                            baseline[
                                "gt_margin"
                            ],

                        "ablated_margin":
                            ablated[
                                "gt_margin"
                            ],

                        "delta_margin":
                            (
                                ablated[
                                    "gt_margin"
                                ]
                                - baseline[
                                    "gt_margin"
                                ]
                            ),

                        "hook_calls":
                            ablator.calls,
                    }
                )

                progress.update(
                    1
                )

    progress.close()

    df = pd.DataFrame(
        rows
    )

    output = (
        OUTPUT_DIR
        / "exhaustive_causal_results.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        "\nSaved:",
        output,
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "\nEXHAUSTIVE SCREEN COMPLETE"
    )


if __name__ == "__main__":
    main()
