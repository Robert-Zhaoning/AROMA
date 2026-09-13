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
    "data/proc_count_causal_v1/metadata.jsonl"
)

PAIR_PATH = Path(
    "outputs/proc_count_causal_v1/exact_matched_pairs.csv"
)

HEAD_PATH = Path(
    "outputs/proc_count_causal_v1/frozen_heads.json"
)

OUTPUT_DIR = Path(
    "outputs/proc_count_causal_v1/causal_validation"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "causal_validation_results.csv"
)

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


def load_metadata():
    records = {}

    with META_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)
            records[r["sample_id"]] = r

    return records


def load_heads():
    config = json.loads(
        HEAD_PATH.read_text(
            encoding="utf-8"
        )
    )

    heads = []

    for h in config[
        "candidate_heads"
    ]:
        heads.append(
            (
                "candidate",
                int(h["layer"]),
                int(h["head"]),
            )
        )

    for h in config[
        "random_control_heads"
    ]:
        heads.append(
            (
                "random",
                int(h["layer"]),
                int(h["head"]),
            )
        )

    return heads


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

    inputs = processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )

    return inputs


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
    ids = {}

    for n in range(1, 11):
        tokens = (
            processor.tokenizer.encode(
                str(n),
                add_special_tokens=False,
            )
        )

        if len(tokens) != 1:
            raise RuntimeError(
                f"Numeral {n} not single-token: "
                f"{tokens}"
            )

        ids[n] = tokens[0]

    return ids


def score_state(
    model,
    inputs,
    numeral_ids,
    gt,
):
    with torch.inference_mode():
        outputs = model(
            **inputs,
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

    log_probs = (
        torch.log_softmax(
            logits,
            dim=-1,
        )
    )

    gt_token = (
        numeral_ids[gt]
    )

    gt_logp = float(
        log_probs[
            gt_token
        ].item()
    )

    numeral_logits = {
        n: float(
            logits[
                token_id
            ].item()
        )
        for n, token_id
        in numeral_ids.items()
    }

    gt_logit = (
        numeral_logits[
            gt
        ]
    )

    best_wrong_logit = max(
        value
        for n, value
        in numeral_logits.items()
        if n != gt
    )

    margin = (
        gt_logit
        - best_wrong_logit
    )

    return {
        "gt_logp":
            gt_logp,

        "margin":
            margin,
    }


class HeadAblator:

    def __init__(
        self,
        model,
        layer_idx,
        head_idx,
    ):

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

        self.num_heads = 32

        hidden_size = (
            self.o_proj.in_features
        )

        self.head_dim = (
            hidden_size
            // self.num_heads
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

    frozen_heads = (
        load_heads()
    )

    sample_roles = {}

    for _, pair in (
        pairs.iterrows()
    ):
        sample_roles[
            pair[
                "correct_sample_id"
            ]
        ] = "correct"

        sample_roles[
            pair[
                "wrong_sample_id"
            ]
        ] = "wrong"

    sample_ids = sorted(
        sample_roles.keys()
    )

    print("=" * 80)
    print(
        "Proc-Count-Causal v1 "
        "Held-out Causal Validation"
    )
    print("=" * 80)

    print(
        "Exact pairs     :",
        len(pairs),
    )

    print(
        "Unique samples  :",
        len(sample_ids),
    )

    print(
        "Frozen heads    :",
        len(frozen_heads),
    )

    print(
        "Interventions   :",
        len(sample_ids)
        * len(frozen_heads),
    )

    print("\nFrozen heads:")

    for (
        head_type,
        layer,
        head,
    ) in frozen_heads:

        print(
            f"  {head_type:9s} "
            f"L{layer}H{head}"
        )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    numeral_ids = (
        get_numeral_token_ids(
            processor
        )
    )

    print(
        "Numeral IDs:",
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
        * len(frozen_heads)
    )

    progress = tqdm(
        total=total,
        desc="Held-out interventions",
    )

    for sample_id in sample_ids:

        sample = (
            metadata[
                sample_id
            ]
        )

        role = (
            sample_roles[
                sample_id
            ]
        )

        gt = int(
            sample[
                "ground_truth"
            ]
        )

        image = (
            Image.open(
                sample[
                    "image_path"
                ]
            )
            .convert("RGB")
        )

        inputs = prepare_inputs(
            processor,
            image,
        )

        device_inputs = (
            move_inputs(
                inputs,
                model,
            )
        )

        baseline = score_state(
            model,
            device_inputs,
            numeral_ids,
            gt,
        )

        for (
            head_type,
            layer,
            head,
        ) in frozen_heads:

            ablator = (
                HeadAblator(
                    model,
                    layer,
                    head,
                )
            )

            ablator.register()

            try:
                ablated = score_state(
                    model,
                    device_inputs,
                    numeral_ids,
                    gt,
                )

            finally:
                ablator.remove()

            if ablator.calls == 0:
                raise RuntimeError(
                    f"Hook not called: "
                    f"{sample_id} "
                    f"L{layer}H{head}"
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

                    "replicate":
                        sample[
                            "replicate"
                        ],

                    "head_type":
                        head_type,

                    "layer":
                        layer,

                    "head":
                        head,

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
                            "margin"
                        ],

                    "ablated_margin":
                        ablated[
                            "margin"
                        ],

                    "delta_margin":
                        (
                            ablated[
                                "margin"
                            ]
                            - baseline[
                                "margin"
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

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )

    print(
        "Rows:",
        len(out),
    )

    print(
        "\nHELD-OUT CAUSAL VALIDATION COMPLETE"
    )


if __name__ == "__main__":
    main()
