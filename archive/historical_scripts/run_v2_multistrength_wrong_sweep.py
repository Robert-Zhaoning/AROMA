import argparse
import json
import sys
from pathlib import Path

import numpy as np
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

REFERENCE_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5_expanded_0_15/"
    "confirmation_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/"
    "multistrength_wrong_sweep"
)

OUT_PATH = (
    OUT_DIR
    / "multistrength_wrong_results.csv"
)

# Already have .5 / 1 / 1.5 elsewhere.
# These are the NEW actions we need to evaluate.
ALPHAS = [
    0.0,
    0.25,
    0.75,
    1.25,
    2.0,
    3.0,
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
        REFERENCE_PATH
    )

    identity = df[
        np.isclose(
            df["alpha"],
            1.0,
        )
    ].copy()

    if len(identity) != 1000:
        raise RuntimeError(
            f"Expected 1000 identity rows, "
            f"found {len(identity)}."
        )

    if (
        identity["sample_id"]
        .nunique()
        != 1000
    ):
        raise RuntimeError(
            "Identity sample IDs not unique."
        )

    mismatch = (
        identity[
            "baseline_best_numeral"
        ].astype(int)
        !=
        identity[
            "modulated_best_numeral"
        ].astype(int)
    )

    if mismatch.any():
        raise RuntimeError(
            "Alpha=1 identity audit failed."
        )

    return identity


def load_completed():

    if not OUT_PATH.exists():
        return set(), []

    old = pd.read_csv(
        OUT_PATH
    )

    completed = set(
        zip(
            old[
                "sample_id"
            ].astype(str),
            old[
                "alpha"
            ].astype(float),
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

    metadata = load_metadata()
    reference = load_reference()

    reference[
        "baseline_correct"
    ] = (
        reference[
            "baseline_best_numeral"
        ].astype(int)
        ==
        reference[
            "ground_truth"
        ].astype(int)
    )

    wrong = reference[
        ~reference[
            "baseline_correct"
        ]
    ].copy()

    wrong = (
        wrong.sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(wrong) != 497:
        raise RuntimeError(
            f"Expected 497 baseline-wrong samples, "
            f"found {len(wrong)}."
        )

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        wrong = wrong.iloc[
            :args.limit
        ].copy()

    print("=" * 110)
    print(
        "AROMA V2 L18H13 MULTI-STRENGTH "
        "WRONG-SAMPLE SWEEP"
    )
    print("=" * 110)

    print(
        "Wrong samples :",
        len(wrong),
    )

    print(
        "Head          :",
        f"L{LAYER}H{HEAD}",
    )

    print(
        "New alphas    :",
        ALPHAS,
    )

    print(
        "Forward passes:",
        len(wrong)
        * len(ALPHAS),
    )

    print(
        "Reference     :",
        REFERENCE_PATH,
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
            "\nResume rows:",
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

    total = (
        len(wrong)
        * len(ALPHAS)
    )

    progress = tqdm(
        total=total,
        desc="Multi-strength sweep",
    )

    for _, ref in wrong.iterrows():

        sid = str(
            ref[
                "sample_id"
            ]
        )

        sample = metadata[
            sid
        ]

        gt = int(
            sample[
                "ground_truth"
            ]
        )

        baseline_pred = int(
            ref[
                "baseline_best_numeral"
            ]
        )

        baseline_error = (
            baseline_pred
            -
            gt
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

            modifier = (
                HeadGainModifier(
                    model=model,
                    layer_idx=LAYER,
                    head_idx=HEAD,
                    alpha=float(alpha),
                )
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

            row = {
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

                "baseline_pred":
                    baseline_pred,

                "baseline_error":
                    baseline_error,

                "alpha":
                    float(alpha),

                "modulated_pred":
                    pred,

                "prediction_shift":
                    pred
                    -
                    baseline_pred,

                "correct":
                    bool(
                        pred
                        ==
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
            }

            rows.append(
                row
            )

            # Save continuously so resume is safe.
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
        "\nSaved:",
        OUT_PATH,
    )

    print(
        "Rows:",
        len(out),
    )

    print(
        "Expected full rows:",
        497
        * len(ALPHAS),
    )

    print(
        "\nRepair count by alpha:"
    )

    print(
        out.groupby(
            "alpha"
        )[
            "correct"
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
        "\nMULTI-STRENGTH SWEEP COMPLETE"
    )


if __name__ == "__main__":
    main()
