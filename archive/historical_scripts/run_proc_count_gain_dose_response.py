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
    "outputs/proc_count_causal_v1/gain_dose_response"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "gain_dose_response_results.csv"
)


ALPHAS = [
    0.0,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
]


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

            record = json.loads(
                line
            )

            result[
                record["sample_id"]
            ] = record

    return result


def load_heads():

    config = json.loads(
        HEAD_PATH.read_text(
            encoding="utf-8"
        )
    )

    result = []

    for item in config[
        "candidate_heads"
    ]:

        result.append(
            (
                "candidate",
                int(
                    item["layer"]
                ),
                int(
                    item["head"]
                ),
            )
        )

    for item in config[
        "random_control_heads"
    ]:

        result.append(
            (
                "random",
                int(
                    item["layer"]
                ),
                int(
                    item["head"]
                ),
            )
        )

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
            value.to(
                model.device
            )
            if torch.is_tensor(
                value
            )
            else value
        )
        for key, value
        in inputs.items()
    }


def numeral_token_ids(
    processor,
):

    result = {}

    for n in range(
        1,
        11,
    ):

        ids = (
            processor.tokenizer.encode(
                str(n),
                add_special_tokens=False,
            )
        )

        if len(ids) != 1:

            raise RuntimeError(
                f"Numeral {n} is "
                f"not single-token: {ids}"
            )

        result[n] = ids[0]

    return result


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

    gt_token_id = (
        numeral_ids[gt]
    )

    gt_logp = float(
        log_probs[
            gt_token_id
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

    numeral_log_probs = {
        n: float(
            log_probs[
                token_id
            ].item()
        )
        for n, token_id
        in numeral_ids.items()
    }

    best_numeral = max(
        numeral_logits,
        key=numeral_logits.get,
    )

    best_wrong_numeral = max(
        (
            n
            for n in numeral_logits
            if n != gt
        ),
        key=lambda n:
            numeral_logits[n],
    )

    gt_margin = (
        numeral_logits[gt]
        - numeral_logits[
            best_wrong_numeral
        ]
    )

    return {
        "gt_logp":
            gt_logp,

        "gt_margin":
            gt_margin,

        "best_numeral":
            int(
                best_numeral
            ),

        "best_wrong_numeral":
            int(
                best_wrong_numeral
            ),

        "numeral_correct":
            int(best_numeral)
            == int(gt),

        "numeral_logits":
            numeral_logits,

        "numeral_log_probs":
            numeral_log_probs,
    }


class HeadGainModifier:

    def __init__(
        self,
        model,
        layer_idx,
        head_idx,
        alpha,
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

        self.alpha = float(
            alpha
        )

        self.num_heads = 32

        hidden_size = (
            self.o_proj.in_features
        )

        if (
            hidden_size
            % self.num_heads
            != 0
        ):
            raise RuntimeError(
                "Hidden size is not divisible "
                "by number of heads."
            )

        self.head_dim = (
            hidden_size
            // self.num_heads
        )

        self.start = (
            int(head_idx)
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

        if not inputs:
            return inputs

        x = inputs[0]

        modified = (
            x.clone()
        )

        modified[
            ...,
            self.start:self.end
        ] *= self.alpha

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

    heads = (
        load_heads()
    )

    sample_role = {}

    for _, pair in (
        pairs.iterrows()
    ):

        sample_role[
            pair[
                "correct_sample_id"
            ]
        ] = "correct"

        sample_role[
            pair[
                "wrong_sample_id"
            ]
        ] = "wrong"

    sample_ids = sorted(
        sample_role.keys()
    )

    print("=" * 88)
    print(
        "Proc-Count-Causal v1 "
        "Gain Dose-Response"
    )
    print("=" * 88)

    print(
        "Exact pairs       :",
        len(pairs),
    )

    print(
        "Unique samples    :",
        len(sample_ids),
    )

    print(
        "Frozen heads      :",
        len(heads),
    )

    print(
        "Alpha values      :",
        ALPHAS,
    )

    nonbaseline_alphas = [
        a
        for a in ALPHAS
        if a != 1.0
    ]

    total_interventions = (
        len(sample_ids)
        * len(heads)
        * len(
            nonbaseline_alphas
        )
    )

    print(
        "Interventions     :",
        total_interventions,
    )

    print(
        "\nFrozen heads:"
    )

    for (
        head_type,
        layer,
        head,
    ) in heads:

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
        numeral_token_ids(
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

    progress = tqdm(
        total=total_interventions,
        desc="Gain interventions",
    )

    for sample_id in sample_ids:

        sample = (
            metadata[
                sample_id
            ]
        )

        role = (
            sample_role[
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

        # ------------------------------------------------------
        # Record alpha=1 baseline once for every head so the CSV
        # remains rectangular.
        # ------------------------------------------------------

        for (
            head_type,
            layer,
            head,
        ) in heads:

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

                    "alpha":
                        1.0,

                    "baseline_gt_logp":
                        baseline[
                            "gt_logp"
                        ],

                    "modulated_gt_logp":
                        baseline[
                            "gt_logp"
                        ],

                    "delta_gt_logp":
                        0.0,

                    "baseline_margin":
                        baseline[
                            "gt_margin"
                        ],

                    "modulated_margin":
                        baseline[
                            "gt_margin"
                        ],

                    "delta_margin":
                        0.0,

                    "baseline_best_numeral":
                        baseline[
                            "best_numeral"
                        ],

                    "modulated_best_numeral":
                        baseline[
                            "best_numeral"
                        ],

                    "baseline_numeral_correct":
                        baseline[
                            "numeral_correct"
                        ],

                    "modulated_numeral_correct":
                        baseline[
                            "numeral_correct"
                        ],

                    "numeral_changed":
                        False,

                    "hook_calls":
                        0,
                }
            )

        # ------------------------------------------------------
        # Non-baseline gain levels
        # ------------------------------------------------------

        for (
            head_type,
            layer,
            head,
        ) in heads:

            for alpha in (
                nonbaseline_alphas
            ):

                modifier = (
                    HeadGainModifier(
                        model=model,
                        layer_idx=layer,
                        head_idx=head,
                        alpha=alpha,
                    )
                )

                modifier.register()

                try:

                    modulated = (
                        score_state(
                            model,
                            device_inputs,
                            numeral_ids,
                            gt,
                        )
                    )

                finally:

                    modifier.remove()

                if (
                    modifier.calls
                    == 0
                ):

                    raise RuntimeError(
                        f"Hook not called: "
                        f"{sample_id} "
                        f"L{layer}H{head} "
                        f"alpha={alpha}"
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

                        "alpha":
                            alpha,

                        "baseline_gt_logp":
                            baseline[
                                "gt_logp"
                            ],

                        "modulated_gt_logp":
                            modulated[
                                "gt_logp"
                            ],

                        "delta_gt_logp":
                            (
                                modulated[
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

                        "modulated_margin":
                            modulated[
                                "gt_margin"
                            ],

                        "delta_margin":
                            (
                                modulated[
                                    "gt_margin"
                                ]
                                - baseline[
                                    "gt_margin"
                                ]
                            ),

                        "baseline_best_numeral":
                            baseline[
                                "best_numeral"
                            ],

                        "modulated_best_numeral":
                            modulated[
                                "best_numeral"
                            ],

                        "baseline_numeral_correct":
                            baseline[
                                "numeral_correct"
                            ],

                        "modulated_numeral_correct":
                            modulated[
                                "numeral_correct"
                            ],

                        "numeral_changed":
                            (
                                modulated[
                                    "best_numeral"
                                ]
                                != baseline[
                                    "best_numeral"
                                ]
                            ),

                        "hook_calls":
                            modifier.calls,
                    }
                )

                progress.update(
                    1
                )

    progress.close()

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )

    print(
        "Rows:",
        len(df),
    )

    expected_rows = (
        len(sample_ids)
        * len(heads)
        * len(ALPHAS)
    )

    print(
        "Expected rows:",
        expected_rows,
    )

    if (
        len(df)
        != expected_rows
    ):

        raise RuntimeError(
            "Unexpected output row count."
        )

    print(
        "\nGAIN DOSE-RESPONSE COMPLETE"
    )


if __name__ == "__main__":
    main()
