import json
import re
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


META_PATH = Path(
    "data/proc_count_causal_v2/metadata.jsonl"
)

RESULT_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5/"
    "confirmation_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "generation_concordance_audit"
)

OUT_PATH = (
    OUT_DIR
    / "generation_concordance_100.csv"
)

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


def load_metadata():
    result = {}

    for line in META_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        r = json.loads(line)

        result[
            r["sample_id"]
        ] = r

    return result


def generate_answer(
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
            use_cache=False,
        )

    new_ids = generated[
        :,
        prompt_len:
    ]

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


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = (
        load_metadata()
    )

    results = pd.read_csv(
        RESULT_PATH
    )

    gain = results[
        np.isclose(
            results["alpha"],
            1.5,
        )
    ].copy()

    # --------------------------------------------------------
    # Fixed balanced subset:
    # every count × condition × {r00, r10}
    # = 10 × 5 × 2 = 100.
    # --------------------------------------------------------

    selected = []

    for sid, sample in (
        metadata.items()
    ):

        if int(
            sample["replicate"]
        ) in {
            0,
            10,
        }:
            selected.append(
                sid
            )

    selected = sorted(
        selected
    )

    assert len(
        selected
    ) == 100

    print("=" * 110)
    print(
        "V2 ACTUAL-GENERATION CONCORDANCE AUDIT"
    )
    print("=" * 110)

    print(
        "Samples:",
        len(selected),
    )

    print(
        "Selection:",
        "all count × condition cells, replicates 00 and 10",
    )

    print(
        "Alpha:",
        ALPHA,
    )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    print(
        "Loading model..."
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

    gain_index = (
        gain.set_index(
            "sample_id"
        )
    )

    for sid in tqdm(
        selected,
        desc="Generation concordance",
    ):

        sample = (
            metadata[
                sid
            ]
        )

        row = (
            gain_index.loc[
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

        inputs = (
            prepare_inputs(
                processor,
                image,
            )
        )

        inputs = (
            move_inputs(
                inputs,
                model,
            )
        )

        # ----------------------------------------------------
        # Baseline actual generation
        # ----------------------------------------------------

        (
            baseline_text,
            baseline_generated,
        ) = generate_answer(
            model,
            processor,
            inputs,
        )

        # ----------------------------------------------------
        # Gain actual generation
        # ----------------------------------------------------

        modifier = (
            HeadGainModifier(
                model=model,
                layer_idx=LAYER,
                head_idx=HEAD,
                alpha=ALPHA,
            )
        )

        modifier.register()

        try:

            (
                gain_text,
                gain_generated,
            ) = generate_answer(
                model,
                processor,
                inputs,
            )

        finally:

            modifier.remove()

        if modifier.calls <= 0:
            raise RuntimeError(
                f"Hook not called: {sid}"
            )

        gt = int(
            sample[
                "ground_truth"
            ]
        )

        proxy_base = int(
            row[
                "baseline_best_numeral"
            ]
        )

        proxy_gain = int(
            row[
                "modulated_best_numeral"
            ]
        )

        rows.append({
            "sample_id":
                sid,

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

            "proxy_baseline":
                proxy_base,

            "generated_baseline":
                baseline_generated,

            "baseline_raw":
                baseline_text,

            "baseline_concordant":
                (
                    baseline_generated
                    == proxy_base
                ),

            "proxy_gain":
                proxy_gain,

            "generated_gain":
                gain_generated,

            "gain_raw":
                gain_text,

            "gain_concordant":
                (
                    gain_generated
                    == proxy_gain
                ),

            "proxy_baseline_correct":
                proxy_base == gt,

            "generated_baseline_correct":
                baseline_generated == gt,

            "proxy_gain_correct":
                proxy_gain == gt,

            "generated_gain_correct":
                gain_generated == gt,

            "hook_calls":
                modifier.calls,
        })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUT_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "FINAL CONCORDANCE RESULTS"
    )

    print(
        "=" * 110
    )

    print(
        "Baseline numeral-proxy concordance:",
        int(
            out[
                "baseline_concordant"
            ].sum()
        ),
        "/ 100 =",
        float(
            out[
                "baseline_concordant"
            ].mean()
        ),
    )

    print(
        "Gain numeral-proxy concordance:",
        int(
            out[
                "gain_concordant"
            ].sum()
        ),
        "/ 100 =",
        float(
            out[
                "gain_concordant"
            ].mean()
        ),
    )

    print(
        "Baseline parse failures:",
        int(
            out[
                "generated_baseline"
            ].isna().sum()
        ),
    )

    print(
        "Gain parse failures:",
        int(
            out[
                "generated_gain"
            ].isna().sum()
        ),
    )

    print(
        "\nBaseline disagreements:"
    )

    disagreement = out[
        ~out[
            "baseline_concordant"
        ]
    ]

    if len(
        disagreement
    ) == 0:

        print(
            "NONE"
        )

    else:

        print(
            disagreement[
                [
                    "sample_id",
                    "ground_truth",
                    "proxy_baseline",
                    "generated_baseline",
                    "baseline_raw",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nGain disagreements:"
    )

    disagreement = out[
        ~out[
            "gain_concordant"
        ]
    ]

    if len(
        disagreement
    ) == 0:

        print(
            "NONE"
        )

    else:

        print(
            disagreement[
                [
                    "sample_id",
                    "ground_truth",
                    "proxy_gain",
                    "generated_gain",
                    "gain_raw",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nSaved:",
        OUT_PATH,
    )

    print(
        "\nGENERATION CONCORDANCE AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
