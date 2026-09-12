from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
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

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "l18h13_gain_all300"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_PATH = (
    OUT_DIR
    / "l18h13_gain_all300_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "l18h13_gain_all300_summary.csv"
)


LAYER = 18
HEAD = 13

ALPHAS = [
    0.5,
    1.0,
    1.5,
]


PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


# ============================================================
# Data helpers
# ============================================================

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

    for n in range(1, 11):

        ids = (
            processor.tokenizer.encode(
                str(n),
                add_special_tokens=False,
            )
        )

        if len(ids) != 1:
            raise RuntimeError(
                f"Numeral {n} is not "
                f"single-token: {ids}"
            )

        result[n] = ids[0]

    return result


# ============================================================
# Score
# ============================================================

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

    vocab_log_probs = (
        torch.log_softmax(
            logits,
            dim=-1,
        )
    )

    numeral_logits = torch.tensor(
        [
            float(
                logits[
                    numeral_ids[n]
                ].item()
            )
            for n in range(1, 11)
        ],
        dtype=torch.float64,
    )

    conditional_probs = (
        torch.softmax(
            numeral_logits,
            dim=0,
        )
        .numpy()
    )

    best_idx = int(
        np.argmax(
            conditional_probs
        )
    )

    best_numeral = (
        best_idx + 1
    )

    wrong_logits = [
        float(
            numeral_logits[
                n - 1
            ]
        )
        for n in range(1, 11)
        if n != gt
    ]

    margin = (
        float(
            numeral_logits[
                gt - 1
            ]
        )
        -
        max(
            wrong_logits
        )
    )

    gt_logp = float(
        vocab_log_probs[
            numeral_ids[gt]
        ].item()
    )

    entropy = float(
        -np.sum(
            conditional_probs
            * np.log(
                conditional_probs
                + 1e-12
            )
        )
    )

    entropy_norm = (
        entropy
        / math.log(10.0)
    )

    expected_numeral = float(
        sum(
            n
            * conditional_probs[
                n - 1
            ]
            for n in range(1, 11)
        )
    )

    order = np.argsort(
        conditional_probs
    )[::-1]

    top1 = int(
        order[0] + 1
    )

    top2 = int(
        order[1] + 1
    )

    top1_prob = float(
        conditional_probs[
            order[0]
        ]
    )

    top2_prob = float(
        conditional_probs[
            order[1]
        ]
    )

    conditional_margin = (
        top1_prob
        - top2_prob
    )

    vocab_probs = (
        torch.softmax(
            logits,
            dim=-1,
        )
    )

    vocab_numeral_mass = float(
        sum(
            vocab_probs[
                numeral_ids[n]
            ].item()
            for n in range(1, 11)
        )
    )

    return {
        "gt_logp":
            gt_logp,

        "margin":
            margin,

        "best_numeral":
            best_numeral,

        "conditional_probs":
            conditional_probs,

        "entropy":
            entropy,

        "entropy_norm":
            entropy_norm,

        "expected_numeral":
            expected_numeral,

        "top1":
            top1,

        "top2":
            top2,

        "top1_prob":
            top1_prob,

        "top2_prob":
            top2_prob,

        "conditional_margin":
            conditional_margin,

        "vocab_numeral_mass":
            vocab_numeral_mass,
    }


# ============================================================
# L18H13 output gain
# ============================================================

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
                "Hidden size not divisible "
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


# ============================================================
# Response metrics
# ============================================================

def js_divergence(
    p,
    q,
):

    p = np.asarray(
        p,
        dtype=np.float64,
    )

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    m = 0.5 * (
        p + q
    )

    def kl(a, b):
        return float(
            np.sum(
                a
                * np.log(
                    (
                        a + 1e-12
                    )
                    /
                    (
                        b + 1e-12
                    )
                )
            )
        )

    return 0.5 * (
        kl(p, m)
        + kl(q, m)
    )


def add_row(
    rows,
    sample,
    baseline,
    state,
    alpha,
    calls,
):

    gt = int(
        sample[
            "ground_truth"
        ]
    )

    p0 = np.asarray(
        baseline[
            "conditional_probs"
        ],
        dtype=float,
    )

    p1 = np.asarray(
        state[
            "conditional_probs"
        ],
        dtype=float,
    )

    delta = (
        p1 - p0
    )

    row = {
        "sample_id":
            sample[
                "sample_id"
            ],

        "ground_truth":
            gt,

        "condition":
            sample[
                "condition"
            ],

        "replicate":
            int(
                sample[
                    "replicate"
                ]
            ),

        "layer":
            LAYER,

        "head":
            HEAD,

        "head_name":
            "L18H13",

        "alpha":
            float(alpha),

        "baseline_best_numeral":
            int(
                baseline[
                    "best_numeral"
                ]
            ),

        "modulated_best_numeral":
            int(
                state[
                    "best_numeral"
                ]
            ),

        "baseline_correct":
            bool(
                baseline[
                    "best_numeral"
                ]
                == gt
            ),

        "modulated_correct":
            bool(
                state[
                    "best_numeral"
                ]
                == gt
            ),

        "baseline_gt_logp":
            float(
                baseline[
                    "gt_logp"
                ]
            ),

        "modulated_gt_logp":
            float(
                state[
                    "gt_logp"
                ]
            ),

        "delta_gt_logp":
            float(
                state[
                    "gt_logp"
                ]
                -
                baseline[
                    "gt_logp"
                ]
            ),

        "baseline_margin":
            float(
                baseline[
                    "margin"
                ]
            ),

        "modulated_margin":
            float(
                state[
                    "margin"
                ]
            ),

        "delta_margin":
            float(
                state[
                    "margin"
                ]
                -
                baseline[
                    "margin"
                ]
            ),

        "baseline_entropy":
            float(
                baseline[
                    "entropy"
                ]
            ),

        "modulated_entropy":
            float(
                state[
                    "entropy"
                ]
            ),

        "delta_entropy":
            float(
                state[
                    "entropy"
                ]
                -
                baseline[
                    "entropy"
                ]
            ),

        "baseline_entropy_norm":
            float(
                baseline[
                    "entropy_norm"
                ]
            ),

        "modulated_entropy_norm":
            float(
                state[
                    "entropy_norm"
                ]
            ),

        "delta_entropy_norm":
            float(
                state[
                    "entropy_norm"
                ]
                -
                baseline[
                    "entropy_norm"
                ]
            ),

        "baseline_expected_numeral":
            float(
                baseline[
                    "expected_numeral"
                ]
            ),

        "modulated_expected_numeral":
            float(
                state[
                    "expected_numeral"
                ]
            ),

        "response_expected_numeral_shift":
            float(
                state[
                    "expected_numeral"
                ]
                -
                baseline[
                    "expected_numeral"
                ]
            ),

        "baseline_top1_prob":
            float(
                baseline[
                    "top1_prob"
                ]
            ),

        "modulated_top1_prob":
            float(
                state[
                    "top1_prob"
                ]
            ),

        "baseline_conditional_margin":
            float(
                baseline[
                    "conditional_margin"
                ]
            ),

        "modulated_conditional_margin":
            float(
                state[
                    "conditional_margin"
                ]
            ),

        "baseline_vocab_numeral_mass":
            float(
                baseline[
                    "vocab_numeral_mass"
                ]
            ),

        "modulated_vocab_numeral_mass":
            float(
                state[
                    "vocab_numeral_mass"
                ]
            ),

        "delta_vocab_numeral_mass":
            float(
                state[
                    "vocab_numeral_mass"
                ]
                -
                baseline[
                    "vocab_numeral_mass"
                ]
            ),

        "response_l1":
            float(
                np.abs(
                    delta
                ).sum()
            ),

        "response_l2":
            float(
                np.sqrt(
                    np.square(
                        delta
                    ).sum()
                )
            ),

        "response_tv":
            float(
                0.5
                * np.abs(
                    delta
                ).sum()
            ),

        "response_js":
            float(
                js_divergence(
                    p0,
                    p1,
                )
            ),

        "numeral_changed":
            bool(
                state[
                    "best_numeral"
                ]
                !=
                baseline[
                    "best_numeral"
                ]
            ),

        "hook_calls":
            int(calls),
    }

    for n in range(1, 11):

        row[
            f"baseline_numprob_{n}"
        ] = float(
            p0[
                n - 1
            ]
        )

        row[
            f"modulated_numprob_{n}"
        ] = float(
            p1[
                n - 1
            ]
        )

        row[
            f"delta_numprob_{n}"
        ] = float(
            delta[
                n - 1
            ]
        )

    rows.append(
        row
    )


# ============================================================
# Main
# ============================================================

def main(
    limit=None,
    protocol_only=False,
):

    metadata = (
        load_metadata()
    )

    sample_ids = sorted(
        metadata.keys()
    )

    print("=" * 112)
    print(
        "AROMA L18H13 GAIN DATASET — "
        "ALL PROC-COUNT-CAUSAL v1"
    )
    print("=" * 112)

    print(
        "Full samples:",
        len(sample_ids),
    )

    if len(sample_ids) != 300:
        raise RuntimeError(
            f"Expected 300 samples, "
            f"found {len(sample_ids)}."
        )

    if limit is not None:

        if limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        sample_ids = (
            sample_ids[
                :limit
            ]
        )

        print(
            "[LIMITED MODE]"
        )

    print(
        "Samples selected:",
        len(sample_ids),
    )

    print(
        "Head: L18H13"
    )

    print(
        "Actions:",
        ALPHAS,
    )

    print(
        "Expected rows:",
        len(sample_ids)
        * len(ALPHAS),
    )

    print(
        "Actual gain interventions:",
        len(sample_ids)
        * 2,
    )

    if protocol_only:

        print(
            "\nPROTOCOL-ONLY AUDIT PASS"
        )

        return

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

    for sid in tqdm(
        sample_ids,
        desc="L18H13 gain dataset",
    ):

        sample = (
            metadata[
                sid
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

        baseline = score_state(
            model,
            inputs,
            numeral_ids,
            int(
                sample[
                    "ground_truth"
                ]
            ),
        )

        # Identity action.
        add_row(
            rows,
            sample,
            baseline,
            baseline,
            alpha=1.0,
            calls=0,
        )

        for alpha in [
            0.5,
            1.5,
        ]:

            modifier = (
                HeadGainModifier(
                    model=model,
                    layer_idx=LAYER,
                    head_idx=HEAD,
                    alpha=alpha,
                )
            )

            modifier.register()

            try:

                state = (
                    score_state(
                        model,
                        inputs,
                        numeral_ids,
                        int(
                            sample[
                                "ground_truth"
                            ]
                        ),
                    )
                )

            finally:

                modifier.remove()

            if modifier.calls <= 0:
                raise RuntimeError(
                    f"Hook not called: "
                    f"{sid}, alpha={alpha}"
                )

            add_row(
                rows,
                sample,
                baseline,
                state,
                alpha=alpha,
                calls=
                    modifier.calls,
            )

    df = pd.DataFrame(
        rows
    )

    expected = (
        len(sample_ids)
        * 3
    )

    if len(df) != expected:
        raise RuntimeError(
            f"Expected {expected} rows, "
            f"got {len(df)}."
        )

    if df.isna().sum().sum() != 0:
        raise RuntimeError(
            "NaNs detected."
        )

    if limit is None:

        out_path = (
            RESULT_PATH
        )

    else:

        out_path = (
            OUT_DIR
            / f"smoke_{limit}_results.csv"
        )

    df.to_csv(
        out_path,
        index=False,
    )

    # ========================================================
    # Task-level summary
    # ========================================================

    baseline_df = (
        df[
            np.isclose(
                df[
                    "alpha"
                ],
                1.0,
            )
        ]
        .copy()
    )

    baseline_acc = float(
        baseline_df[
            "baseline_correct"
        ].mean()
    )

    n_baseline_correct = int(
        baseline_df[
            "baseline_correct"
        ].sum()
    )

    n_baseline_wrong = (
        len(
            baseline_df
        )
        - n_baseline_correct
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "TASK-LEVEL SUMMARY"
    )

    print(
        "=" * 112
    )

    print(
        "Baseline correct:",
        n_baseline_correct,
    )

    print(
        "Baseline wrong:",
        n_baseline_wrong,
    )

    print(
        "Baseline accuracy:",
        baseline_acc,
    )

    summary_rows = []

    for alpha in ALPHAS:

        g = df[
            np.isclose(
                df[
                    "alpha"
                ],
                alpha,
            )
        ].copy()

        repairs = int(
            (
                (~g[
                    "baseline_correct"
                ])
                &
                g[
                    "modulated_correct"
                ]
            ).sum()
        )

        breaks = int(
            (
                g[
                    "baseline_correct"
                ]
                &
                (~g[
                    "modulated_correct"
                ])
            ).sum()
        )

        post_acc = float(
            g[
                "modulated_correct"
            ].mean()
        )

        summary_rows.append({
            "alpha":
                alpha,

            "post_accuracy":
                post_acc,

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                repairs
                - breaks,

            "mean_delta_gt_logp":
                float(
                    g[
                        "delta_gt_logp"
                    ].mean()
                ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ========================================================
    # Oracle over noop / 0.5 / 1.5
    # ========================================================

    oracle_correct = []

    repairable_wrong = 0

    for sid, g in df.groupby(
        "sample_id"
    ):

        baseline_correct = bool(
            g[
                "baseline_correct"
            ].iloc[0]
        )

        any_correct = bool(
            g[
                "modulated_correct"
            ].any()
        )

        oracle_correct.append(
            int(
                any_correct
            )
        )

        if (
            (not baseline_correct)
            and any_correct
        ):
            repairable_wrong += 1

    oracle_acc = float(
        np.mean(
            oracle_correct
        )
    )

    print(
        "\nORACLE THREE-ACTION CEILING"
    )

    print(
        "Repairable baseline-wrong samples:",
        repairable_wrong,
        "/",
        n_baseline_wrong,
    )

    print(
        "Oracle accuracy:",
        oracle_acc,
    )

    print(
        "Oracle gain over baseline:",
        oracle_acc
        - baseline_acc,
    )

    if limit is None:

        summary.to_csv(
            SUMMARY_PATH,
            index=False,
        )

        print(
            "\nSaved:"
        )

        print(
            RESULT_PATH
        )

        print(
            SUMMARY_PATH
        )

    else:

        print(
            "\nSaved:",
            out_path,
        )

    print(
        "\nL18H13 ALL-300 DATASET COMPLETE"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--protocol-only",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        limit=args.limit,
        protocol_only=
            args.protocol_only,
    )
