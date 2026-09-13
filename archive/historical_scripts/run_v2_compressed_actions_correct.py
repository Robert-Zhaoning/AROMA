import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
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

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)


META_PATH = Path(
    "data/proc_count_causal_v2/metadata.jsonl"
)

REF_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5_expanded_0_15/"
    "confirmation_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/"
    "compressed_actions_correct"
)

OUT_PATH = (
    OUT_DIR
    / "compressed_actions_correct_results.csv"
)

ALPHAS = [
    0.0,
    2.0,
    4.0,
]


def load_metadata():

    result = {}

    for line in META_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        r = json.loads(line)

        result[
            str(r["sample_id"])
        ] = r

    return result


def load_reference():

    df = pd.read_csv(
        REF_PATH
    )

    identity = df[
        df["alpha"].astype(float)
        .sub(1.0)
        .abs()
        < 1e-8
    ].copy()

    if len(identity) != 1000:
        raise RuntimeError(
            f"Expected 1000 identity rows, "
            f"found {len(identity)}."
        )

    identity[
        "baseline_correct_check"
    ] = (
        identity[
            "baseline_best_numeral"
        ].astype(int)
        ==
        identity[
            "ground_truth"
        ].astype(int)
    )

    correct = identity[
        identity[
            "baseline_correct_check"
        ]
    ].copy()

    if len(correct) != 503:
        raise RuntimeError(
            f"Expected 503 baseline-correct "
            f"samples, found {len(correct)}."
        )

    return (
        correct
        .sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )


def load_metadata_dict():

    return load_metadata()


def load_completed():

    if not OUT_PATH.exists():
        return set(), []

    old = pd.read_csv(
        OUT_PATH
    )

    completed = set(
        zip(
            old["sample_id"].astype(str),
            old["alpha"].astype(float),
        )
    )

    return (
        completed,
        old.to_dict(
            orient="records"
        ),
    )


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

    parser.add_argument(
        "--protocol-only",
        action="store_true",
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = (
        load_metadata_dict()
    )

    correct = load_reference()

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        correct = correct.iloc[
            :args.limit
        ].copy()

    print("=" * 112)
    print(
        "AROMA V2 COMPRESSED-ACTION "
        "CORRECT-SAMPLE COMPLETION"
    )
    print("=" * 112)

    print(
        "Baseline-correct samples:",
        len(correct),
    )

    print(
        "New actions:",
        ALPHAS,
    )

    print(
        "Expected forward passes:",
        len(correct)
        * len(ALPHAS),
    )

    print(
        "Head:",
        f"L{LAYER}H{HEAD}",
    )

    print(
        "\nPROTOCOL AUDIT PASS"
    )

    if args.protocol_only:

        print(
            "No model was loaded."
        )

        return

    completed = set()
    rows = []

    if args.resume:

        completed, rows = (
            load_completed()
        )

        print(
            "Resume rows:",
            len(rows),
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

    progress = tqdm(
        total=(
            len(correct)
            *
            len(ALPHAS)
        ),
        desc="Correct-sample actions",
    )

    for _, ref in correct.iterrows():

        sid = str(
            ref["sample_id"]
        )

        sample = metadata[
            sid
        ]

        gt = int(
            sample[
                "ground_truth"
            ]
        )

        base_pred = int(
            ref[
                "baseline_best_numeral"
            ]
        )

        if base_pred != gt:

            raise RuntimeError(
                f"{sid} is not baseline-correct."
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

        for alpha in ALPHAS:

            key = (
                sid,
                float(alpha),
            )

            if key in completed:

                progress.update(1)

                continue

            modifier = HeadGainModifier(
                model=model,
                layer_idx=LAYER,
                head_idx=HEAD,
                alpha=float(alpha),
            )

            modifier.register()

            try:

                state = score_state(
                    model=model,
                    inputs=inputs,
                    numeral_ids=numeral_ids,
                    gt=gt,
                )

            finally:

                modifier.remove()

            if modifier.calls <= 0:

                raise RuntimeError(
                    f"Hook not called: "
                    f"{sid}, alpha={alpha}"
                )

            pred = int(
                state[
                    "best_numeral"
                ]
            )

            rows.append({
                "sample_id":
                    sid,

                "ground_truth":
                    gt,

                "condition":
                    str(
                        sample[
                            "condition"
                        ]
                    ),

                "replicate":
                    int(
                        sample[
                            "replicate"
                        ]
                    ),

                "alpha":
                    float(alpha),

                "baseline_pred":
                    base_pred,

                "modulated_pred":
                    pred,

                "prediction_shift":
                    pred
                    -
                    base_pred,

                "baseline_correct":
                    True,

                "modulated_correct":
                    bool(
                        pred
                        ==
                        gt
                    ),

                "break_case":
                    bool(
                        pred
                        !=
                        gt
                    ),

                "baseline_gt_logp":
                    float(
                        ref[
                            "baseline_gt_logp"
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
                        float(
                            ref[
                                "baseline_gt_logp"
                            ]
                        )
                    ),

                "baseline_margin":
                    float(
                        ref[
                            "baseline_margin"
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
                        float(
                            ref[
                                "baseline_margin"
                            ]
                        )
                    ),

                "baseline_expected_numeral":
                    float(
                        ref[
                            "baseline_expected_numeral"
                        ]
                    ),

                "modulated_expected_numeral":
                    float(
                        state[
                            "expected_numeral"
                        ]
                    ),

                "expected_shift":
                    float(
                        state[
                            "expected_numeral"
                        ]
                        -
                        float(
                            ref[
                                "baseline_expected_numeral"
                            ]
                        )
                    ),

                "hook_calls":
                    int(
                        modifier.calls
                    ),
            })

            pd.DataFrame(
                rows
            ).to_csv(
                OUT_PATH,
                index=False,
            )

            completed.add(
                key
            )

            progress.update(1)

    progress.close()

    out = pd.DataFrame(
        rows
    )

    out = (
        out.drop_duplicates(
            [
                "sample_id",
                "alpha",
            ],
            keep="last",
        )
        .sort_values(
            [
                "sample_id",
                "alpha",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    out.to_csv(
        OUT_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "COMPRESSED ACTION CORRECT-SAMPLE RESULTS"
    )

    print(
        "=" * 112
    )

    print(
        "Rows:",
        len(out),
    )

    print(
        "Expected full rows:",
        503
        *
        len(ALPHAS),
    )

    print(
        "\nBreaks by alpha:"
    )

    print(
        out.groupby(
            "alpha"
        )[
            "break_case"
        ]
        .sum()
        .astype(int)
        .to_string()
    )

    print(
        "\nChanged predictions by alpha:"
    )

    print(
        out.assign(
            changed=(
                out[
                    "modulated_pred"
                ]
                !=
                out[
                    "baseline_pred"
                ]
            )
        )
        .groupby(
            "alpha"
        )[
            "changed"
        ]
        .sum()
        .astype(int)
        .to_string()
    )

    print(
        "\nMean expected shift by alpha:"
    )

    print(
        out.groupby(
            "alpha"
        )[
            "expected_shift"
        ]
        .mean()
        .to_string()
    )

    print(
        "\nSaved:",
        OUT_PATH,
    )

    print(
        "\nCORRECT-SAMPLE COMPLETION COMPLETE"
    )


if __name__ == "__main__":
    main()
