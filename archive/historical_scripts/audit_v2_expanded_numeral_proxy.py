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


META_PATH = Path(
    "data/proc_count_causal_v2/"
    "metadata.jsonl"
)

GEN_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "generation_concordance_audit/"
    "generation_concordance_100.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "generation_concordance_audit"
)

OUT_PATH = (
    OUT_DIR
    / "expanded_numeral_proxy_100.csv"
)

ALPHA = 1.5

NUMERAL_MIN = 0
NUMERAL_MAX = 15


def load_metadata():

    result = {}

    for line in META_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        r = json.loads(
            line
        )

        result[
            str(
                r["sample_id"]
            )
        ] = r

    return result


def build_numeral_ids(
    processor,
):

    result = {}

    for n in range(
        NUMERAL_MIN,
        NUMERAL_MAX + 1,
    ):

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

        result[n] = int(
            ids[0]
        )

    return result


def score_extended(
    model,
    inputs,
    numeral_ids,
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

    numeral_logits = {
        n: float(
            logits[token_id]
            .item()
        )
        for n, token_id
        in numeral_ids.items()
    }

    best_numeral = max(
        numeral_logits,
        key=numeral_logits.get,
    )

    # Conditional distribution over the expanded
    # numeral set only.
    ordered = list(
        range(
            NUMERAL_MIN,
            NUMERAL_MAX + 1,
        )
    )

    local_logits = torch.tensor(
        [
            numeral_logits[n]
            for n in ordered
        ],
        dtype=torch.float64,
    )

    probs = torch.softmax(
        local_logits,
        dim=0,
    )

    expected = float(
        sum(
            n * float(
                probs[i].item()
            )
            for i, n
            in enumerate(
                ordered
            )
        )
    )

    return {
        "best_numeral":
            int(
                best_numeral
            ),

        "expected_numeral":
            expected,

        "top1_probability":
            float(
                probs.max()
                .item()
            ),
    }


def parse_saved_generated(
    value,
):

    if pd.isna(
        value
    ):
        return None

    return int(
        value
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generation = pd.read_csv(
        GEN_PATH
    )

    metadata = (
        load_metadata()
    )

    assert len(
        generation
    ) == 100

    assert (
        generation[
            "sample_id"
        ]
        .nunique()
        == 100
    )

    print("=" * 115)
    print(
        "V2 EXPANDED NUMERAL-PROXY "
        "CONCORDANCE AUDIT"
    )
    print("=" * 115)

    print(
        "Samples:",
        len(
            generation
        ),
    )

    print(
        "Expanded numeral range:",
        f"{NUMERAL_MIN}..{NUMERAL_MAX}",
    )

    print(
        "Gain alpha:",
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

    numeral_ids = (
        build_numeral_ids(
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

    for _, saved in tqdm(
        generation.iterrows(),
        total=len(
            generation
        ),
        desc="Expanded proxy",
    ):

        sid = str(
            saved[
                "sample_id"
            ]
        )

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
        # Expanded baseline proxy
        # ----------------------------------------------------

        baseline = (
            score_extended(
                model,
                inputs,
                numeral_ids,
            )
        )

        # ----------------------------------------------------
        # Expanded gain proxy
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

            gain = (
                score_extended(
                    model,
                    inputs,
                    numeral_ids,
                )
            )

        finally:

            modifier.remove()

        if modifier.calls <= 0:

            raise RuntimeError(
                f"Hook not called: "
                f"{sid}"
            )

        generated_baseline = (
            parse_saved_generated(
                saved[
                    "generated_baseline"
                ]
            )
        )

        generated_gain = (
            parse_saved_generated(
                saved[
                    "generated_gain"
                ]
            )
        )

        old_proxy_baseline = int(
            saved[
                "proxy_baseline"
            ]
        )

        old_proxy_gain = int(
            saved[
                "proxy_gain"
            ]
        )

        rows.append({
            "sample_id":
                sid,

            "ground_truth":
                int(
                    sample[
                        "ground_truth"
                    ]
                ),

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

            "generated_baseline":
                generated_baseline,

            "old_proxy_baseline_1_10":
                old_proxy_baseline,

            "expanded_proxy_baseline_0_15":
                baseline[
                    "best_numeral"
                ],

            "baseline_expected_0_15":
                baseline[
                    "expected_numeral"
                ],

            "baseline_expanded_concordant":
                (
                    generated_baseline
                    ==
                    baseline[
                        "best_numeral"
                    ]
                ),

            "generated_gain":
                generated_gain,

            "old_proxy_gain_1_10":
                old_proxy_gain,

            "expanded_proxy_gain_0_15":
                gain[
                    "best_numeral"
                ],

            "gain_expected_0_15":
                gain[
                    "expected_numeral"
                ],

            "gain_expanded_concordant":
                (
                    generated_gain
                    ==
                    gain[
                        "best_numeral"
                    ]
                ),

            "old_gain_proxy_changed":
                (
                    old_proxy_gain
                    !=
                    baseline[
                        "best_numeral"
                    ]
                ),

            "expanded_gain_proxy_changed":
                (
                    gain[
                        "best_numeral"
                    ]
                    !=
                    baseline[
                        "best_numeral"
                    ]
                ),

            "hook_calls":
                int(
                    modifier.calls
                ),
        })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUT_PATH,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    baseline_matches = int(
        out[
            "baseline_expanded_concordant"
        ].sum()
    )

    gain_matches = int(
        out[
            "gain_expanded_concordant"
        ].sum()
    )

    print(
        "\n" + "=" * 115
    )

    print(
        "FINAL EXPANDED-PROXY RESULTS"
    )

    print(
        "=" * 115
    )

    print(
        "Baseline expanded-proxy concordance:",
        baseline_matches,
        "/ 100 =",
        baseline_matches / 100,
    )

    print(
        "Gain expanded-proxy concordance:",
        gain_matches,
        "/ 100 =",
        gain_matches / 100,
    )

    print(
        "\nOld proxy vs expanded proxy changes:"
    )

    baseline_changed = (
        out[
            "old_proxy_baseline_1_10"
        ]
        !=
        out[
            "expanded_proxy_baseline_0_15"
        ]
    )

    gain_changed = (
        out[
            "old_proxy_gain_1_10"
        ]
        !=
        out[
            "expanded_proxy_gain_0_15"
        ]
    )

    print(
        "Baseline proxy changed:",
        int(
            baseline_changed.sum()
        ),
        "/ 100",
    )

    print(
        "Gain proxy changed:",
        int(
            gain_changed.sum()
        ),
        "/ 100",
    )

    if gain_changed.any():

        print(
            "\nGAIN CASES CHANGED BY "
            "EXPANDED NUMERAL RANGE:"
        )

        print(
            out.loc[
                gain_changed,
                [
                    "sample_id",
                    "ground_truth",
                    "condition",
                    "generated_gain",
                    "old_proxy_gain_1_10",
                    "expanded_proxy_gain_0_15",
                    "gain_expected_0_15",
                ]
            ].to_string(
                index=False
            )
        )

    disagreement = out[
        ~out[
            "gain_expanded_concordant"
        ]
    ]

    print(
        "\nExpanded gain disagreements:"
    )

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
                    "generated_gain",
                    "expanded_proxy_gain_0_15",
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
        "\nEXPANDED NUMERAL-PROXY "
        "CONCORDANCE AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
