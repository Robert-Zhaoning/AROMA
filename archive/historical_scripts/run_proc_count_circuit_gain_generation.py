import json
import re
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
    "outputs/proc_count_causal_v1/circuit_gain_generation"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "circuit_gain_generation_results.csv"
)

ALPHAS = [
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


def extract_integer(text):
    m = re.search(
        r"(?<![\w.])\d+(?![\w.])",
        text,
    )

    if m is None:
        return None

    return int(m.group())


def load_metadata():
    result = {}

    for line in META_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        record = json.loads(line)

        result[
            record["sample_id"]
        ] = record

    return result


def load_head_sets():

    config = json.loads(
        HEAD_PATH.read_text(
            encoding="utf-8"
        )
    )

    candidates = [
        (
            int(x["layer"]),
            int(x["head"]),
        )
        for x in config[
            "candidate_heads"
        ]
    ]

    randoms = [
        (
            int(x["layer"]),
            int(x["head"]),
        )
        for x in config[
            "random_control_heads"
        ]
    ]

    return {
        "candidate":
            candidates,

        "random":
            randoms,
    }


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
        k: (
            v.to(model.device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in inputs.items()
    }


def numeral_ids(
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
                f"{n} not single token: {ids}"
            )

        result[n] = ids[0]

    return result


def score(
    model,
    inputs,
    ids,
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
            -1
        ].float()
    )

    log_probs = (
        torch.log_softmax(
            logits,
            dim=-1,
        )
    )

    gt_logp = float(
        log_probs[
            ids[gt]
        ].item()
    )

    numeral_logits = {
        n: float(
            logits[token_id].item()
        )
        for n, token_id
        in ids.items()
    }

    best_wrong = max(
        (
            n
            for n in numeral_logits
            if n != gt
        ),
        key=lambda n:
            numeral_logits[n],
    )

    margin = (
        numeral_logits[gt]
        - numeral_logits[
            best_wrong
        ]
    )

    return gt_logp, margin


def generate(
    model,
    processor,
    inputs,
):

    prompt_len = (
        inputs[
            "input_ids"
        ].shape[-1]
    )

    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )

    new_ids = (
        generated[
            :,
            prompt_len:
        ]
    )

    text = (
        processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .strip()
    )

    return (
        text,
        extract_integer(text),
    )


class CircuitGain:

    def __init__(
        self,
        model,
        heads,
        alpha,
    ):

        self.model = model
        self.heads = heads
        self.alpha = float(alpha)

        self.handles = []
        self.call_counts = {}

        self.by_layer = {}

        for layer, head in heads:

            self.by_layer.setdefault(
                layer,
                [],
            ).append(head)

    def register(self):

        self.remove()

        for (
            layer_idx,
            heads,
        ) in self.by_layer.items():

            layer = (
                self.model
                .model
                .language_model
                .layers[
                    layer_idx
                ]
            )

            o_proj = (
                layer
                .cross_attn
                .o_proj
            )

            hidden_size = (
                o_proj.in_features
            )

            num_heads = 32

            head_dim = (
                hidden_size
                // num_heads
            )

            self.call_counts[
                layer_idx
            ] = 0

            def make_hook(
                current_layer,
                current_heads,
                current_head_dim,
            ):

                def hook(
                    module,
                    inputs,
                ):

                    x = inputs[0]
                    modified = x.clone()

                    for head_idx in (
                        current_heads
                    ):

                        start = (
                            head_idx
                            * current_head_dim
                        )

                        end = (
                            start
                            + current_head_dim
                        )

                        modified[
                            ...,
                            start:end
                        ] *= self.alpha

                    self.call_counts[
                        current_layer
                    ] += 1

                    if len(inputs) == 1:
                        return (
                            modified,
                        )

                    return (
                        modified,
                        *inputs[1:],
                    )

                return hook

            handle = (
                o_proj
                .register_forward_pre_hook(
                    make_hook(
                        layer_idx,
                        list(heads),
                        head_dim,
                    )
                )
            )

            self.handles.append(
                handle
            )

    def remove(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata()

    pairs = pd.read_csv(
        PAIR_PATH
    )

    head_sets = load_head_sets()

    sample_roles = {}

    for _, pair in pairs.iterrows():

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
        sample_roles
    )

    print("=" * 90)
    print(
        "Proc-Count-Causal v1 "
        "Distributed Circuit Gain + Generation"
    )
    print("=" * 90)

    print(
        "Samples:",
        len(sample_ids),
    )

    print(
        "Alphas:",
        ALPHAS,
    )

    print("\nCandidate circuit:")
    print(
        head_sets[
            "candidate"
        ]
    )

    print("\nRandom circuit:")
    print(
        head_sets[
            "random"
        ]
    )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    ids = numeral_ids(
        processor
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
        * (
            1
            + 2 * (
                len(ALPHAS) - 1
            )
        )
    )

    progress = tqdm(
        total=total,
        desc="Circuit repair",
    )

    for sample_id in sample_ids:

        sample = metadata[
            sample_id
        ]

        role = sample_roles[
            sample_id
        ]

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

        inputs = move_inputs(
            inputs,
            model,
        )

        # --------------------------------------------------
        # Baseline
        # --------------------------------------------------

        baseline_logp, baseline_margin = (
            score(
                model,
                inputs,
                ids,
                gt,
            )
        )

        baseline_text, baseline_pred = (
            generate(
                model,
                processor,
                inputs,
            )
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

                "circuit_type":
                    "baseline",

                "alpha":
                    1.0,

                "prediction":
                    baseline_pred,

                "raw_text":
                    baseline_text,

                "correct":
                    baseline_pred
                    == gt,

                "gt_logp":
                    baseline_logp,

                "margin":
                    baseline_margin,

                "delta_gt_logp":
                    0.0,

                "delta_margin":
                    0.0,

                "repaired":
                    False,

                "broken":
                    False,
            }
        )

        progress.update(1)

        # --------------------------------------------------
        # Candidate/random distributed circuit
        # --------------------------------------------------

        for circuit_type in [
            "candidate",
            "random",
        ]:

            heads = head_sets[
                circuit_type
            ]

            for alpha in ALPHAS:

                if alpha == 1.0:
                    continue

                modifier = CircuitGain(
                    model,
                    heads,
                    alpha,
                )

                modifier.register()

                try:

                    logp, margin = score(
                        model,
                        inputs,
                        ids,
                        gt,
                    )

                    text, pred = generate(
                        model,
                        processor,
                        inputs,
                    )

                finally:

                    modifier.remove()

                correct = (
                    pred == gt
                )

                repaired = (
                    (
                        baseline_pred
                        != gt
                    )
                    and correct
                )

                broken = (
                    (
                        baseline_pred
                        == gt
                    )
                    and (
                        pred != gt
                    )
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

                        "circuit_type":
                            circuit_type,

                        "alpha":
                            alpha,

                        "prediction":
                            pred,

                        "raw_text":
                            text,

                        "correct":
                            correct,

                        "gt_logp":
                            logp,

                        "margin":
                            margin,

                        "delta_gt_logp":
                            logp
                            - baseline_logp,

                        "delta_margin":
                            margin
                            - baseline_margin,

                        "repaired":
                            repaired,

                        "broken":
                            broken,
                    }
                )

                progress.update(1)

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

    print(
        "\nDISTRIBUTED CIRCUIT REPAIR COMPLETE"
    )


if __name__ == "__main__":
    main()
