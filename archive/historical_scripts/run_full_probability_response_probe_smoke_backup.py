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
    / "full_probability_response_results.csv"
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

    # ------------------------------------------------------
    # Full 1-10 conditional numeral distribution.
    #
    # vocab probability:
    #   P(token=n | image, prompt)
    #
    # numeral probability:
    #   P(n | answer is one of 1,...,10)
    #
    # The second quantity is GT-free and is the distribution
    # used by the intervention-response router.
    # ------------------------------------------------------

    ordered_numerals = list(range(1, 11))

    numeral_logit_tensor = torch.tensor(
        [
            numeral_logits[n]
            for n in ordered_numerals
        ],
        dtype=torch.float64,
    )

    numeral_cond_probs_tensor = torch.softmax(
        numeral_logit_tensor,
        dim=0,
    )

    numeral_cond_probs = {
        n: float(
            numeral_cond_probs_tensor[i].item()
        )
        for i, n in enumerate(
            ordered_numerals
        )
    }

    # Absolute probability mass assigned to numeral tokens
    # in the full vocabulary distribution.
    numeral_vocab_probs = {
        n: float(
            torch.exp(
                log_probs[
                    numeral_ids[n]
                ]
            ).item()
        )
        for n in ordered_numerals
    }

    numeral_vocab_mass = float(
        sum(
            numeral_vocab_probs.values()
        )
    )

    eps = 1e-12

    numeral_entropy = float(
        -sum(
            p * torch.log(
                torch.tensor(
                    p + eps,
                    dtype=torch.float64,
                )
            ).item()
            for p in numeral_cond_probs.values()
        )
    )

    # Entropy normalized to [0,1].
    numeral_entropy_norm = float(
        numeral_entropy
        / torch.log(
            torch.tensor(
                10.0,
                dtype=torch.float64,
            )
        ).item()
    )

    sorted_probs = sorted(
        numeral_cond_probs.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    numeral_top1 = int(
        sorted_probs[0][0]
    )

    numeral_top2 = int(
        sorted_probs[1][0]
    )

    numeral_top1_prob = float(
        sorted_probs[0][1]
    )

    numeral_top2_prob = float(
        sorted_probs[1][1]
    )

    numeral_probability_margin = float(
        numeral_top1_prob
        - numeral_top2_prob
    )

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

        "numeral_vocab_probs":
            numeral_vocab_probs,

        "numeral_vocab_mass":
            numeral_vocab_mass,

        "numeral_cond_probs":
            numeral_cond_probs,

        "numeral_entropy":
            numeral_entropy,

        "numeral_entropy_norm":
            numeral_entropy_norm,

        "numeral_top1":
            numeral_top1,

        "numeral_top2":
            numeral_top2,

        "numeral_top1_prob":
            numeral_top1_prob,

        "numeral_top2_prob":
            numeral_top2_prob,

        "numeral_probability_margin":
            numeral_probability_margin,
    }


def distribution_response_metrics(
    baseline,
    modulated,
):
    """
    GT-free comparison between two 1-10 conditional numeral
    distributions.
    """

    import math

    p = [
        float(
            baseline[
                "numeral_cond_probs"
            ][n]
        )
        for n in range(1, 11)
    ]

    q = [
        float(
            modulated[
                "numeral_cond_probs"
            ][n]
        )
        for n in range(1, 11)
    ]

    eps = 1e-12

    l1 = sum(
        abs(a - b)
        for a, b in zip(p, q)
    )

    l2 = (
        sum(
            (a - b) ** 2
            for a, b in zip(p, q)
        )
        ** 0.5
    )

    tv = 0.5 * l1

    m = [
        0.5 * (a + b)
        for a, b in zip(p, q)
    ]

    kl_pm = sum(
        a * math.log(
            (a + eps)
            / (c + eps)
        )
        for a, c in zip(p, m)
    )

    kl_qm = sum(
        b * math.log(
            (b + eps)
            / (c + eps)
        )
        for b, c in zip(q, m)
    )

    js = 0.5 * (
        kl_pm
        + kl_qm
    )

    signed_mean_shift = sum(
        (
            q[i]
            - p[i]
        )
        * (
            (i + 1)
            - 5.5
        )
        for i in range(10)
    )

    expected_before = sum(
        (i + 1) * p[i]
        for i in range(10)
    )

    expected_after = sum(
        (i + 1) * q[i]
        for i in range(10)
    )

    return {
        "response_l1":
            float(l1),

        "response_l2":
            float(l2),

        "response_tv":
            float(tv),

        "response_js":
            float(js),

        "response_expected_numeral_before":
            float(expected_before),

        "response_expected_numeral_after":
            float(expected_after),

        "response_expected_numeral_shift":
            float(
                expected_after
                - expected_before
            ),

        "response_signed_center_shift":
            float(
                signed_mean_shift
            ),

        "delta_entropy":
            float(
                modulated[
                    "numeral_entropy"
                ]
                - baseline[
                    "numeral_entropy"
                ]
            ),

        "delta_entropy_norm":
            float(
                modulated[
                    "numeral_entropy_norm"
                ]
                - baseline[
                    "numeral_entropy_norm"
                ]
            ),

        "delta_vocab_numeral_mass":
            float(
                modulated[
                    "numeral_vocab_mass"
                ]
                - baseline[
                    "numeral_vocab_mass"
                ]
            ),
    }


def add_distribution_columns(
    row,
    baseline,
    modulated,
):
    """
    Add complete GT-free numeral distribution information
    to one output row.
    """

    for n in range(1, 11):

        bp = float(
            baseline[
                "numeral_cond_probs"
            ][n]
        )

        mp = float(
            modulated[
                "numeral_cond_probs"
            ][n]
        )

        row[
            f"baseline_numprob_{n}"
        ] = bp

        row[
            f"modulated_numprob_{n}"
        ] = mp

        row[
            f"delta_numprob_{n}"
        ] = mp - bp

        row[
            f"baseline_vocabprob_{n}"
        ] = float(
            baseline[
                "numeral_vocab_probs"
            ][n]
        )

        row[
            f"modulated_vocabprob_{n}"
        ] = float(
            modulated[
                "numeral_vocab_probs"
            ][n]
        )

    row[
        "baseline_numeral_entropy"
    ] = float(
        baseline[
            "numeral_entropy"
        ]
    )

    row[
        "modulated_numeral_entropy"
    ] = float(
        modulated[
            "numeral_entropy"
        ]
    )

    row[
        "baseline_numeral_entropy_norm"
    ] = float(
        baseline[
            "numeral_entropy_norm"
        ]
    )

    row[
        "modulated_numeral_entropy_norm"
    ] = float(
        modulated[
            "numeral_entropy_norm"
        ]
    )

    row[
        "baseline_numeral_top1_prob"
    ] = float(
        baseline[
            "numeral_top1_prob"
        ]
    )

    row[
        "modulated_numeral_top1_prob"
    ] = float(
        modulated[
            "numeral_top1_prob"
        ]
    )

    row[
        "baseline_numeral_top2_prob"
    ] = float(
        baseline[
            "numeral_top2_prob"
        ]
    )

    row[
        "modulated_numeral_top2_prob"
    ] = float(
        modulated[
            "numeral_top2_prob"
        ]
    )

    row[
        "baseline_probability_margin"
    ] = float(
        baseline[
            "numeral_probability_margin"
        ]
    )

    row[
        "modulated_probability_margin"
    ] = float(
        modulated[
            "numeral_probability_margin"
        ]
    )

    row[
        "baseline_vocab_numeral_mass"
    ] = float(
        baseline[
            "numeral_vocab_mass"
        ]
    )

    row[
        "modulated_vocab_numeral_mass"
    ] = float(
        modulated[
            "numeral_vocab_mass"
        ]
    )

    row.update(
        distribution_response_metrics(
            baseline,
            modulated,
        )
    )

    return row


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


def main(limit=None):

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

    full_sample_count = len(sample_ids)

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit must be a positive integer."
            )

        sample_ids = sample_ids[:limit]

        print(
            "\n[SMOKE / LIMITED MODE]"
        )
        print(
            "Full sample pool  :",
            full_sample_count,
        )
        print(
            "Samples selected  :",
            len(sample_ids),
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

            row = {
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

            row = add_distribution_columns(
                row=row,
                baseline=baseline,
                modulated=baseline,
            )

            rows.append(row)

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

                row = {
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

                row = add_distribution_columns(
                    row=row,
                    baseline=baseline,
                    modulated=modulated,
                )

                rows.append(row)

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

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AROMA full-probability "
            "intervention-response probe"
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run only the first N selected samples. "
            "Use this for smoke testing."
        ),
    )

    args = parser.parse_args()

    main(
        limit=args.limit,
    )
