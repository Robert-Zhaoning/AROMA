import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

from oracle_dam_v2_smoke import (
    build_object_token_sets,
    generate_answer,
    load_metadata,
    move_inputs,
    numeral_token_ids,
    prepare_inputs,
    score_state,
)

from distributed_dam_v2_smoke_fixed import (
    DistributedLayerDAM,
)


MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

SAMPLE_ID = "pccv1_n05_row_r00"

BETA = 1.25
GAMMA = 0.0
QUERY_SCOPE = "answer"

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "dam_v2_subset_search"
)

# ----------------------------------------------------------
# Candidate subsets motivated by the pairwise interaction map
# ----------------------------------------------------------

SUBSETS = {
    "A_L3H4_L8H30": [
        (3, 4),
        (8, 30),
    ],

    "B_L3H4_L13H11": [
        (3, 4),
        (13, 11),
    ],

    "C_L8H30_L13H11": [
        (8, 30),
        (13, 11),
    ],

    "D_L18H13_L13H11": [
        (18, 13),
        (13, 11),
    ],

    "E_L3H4_L8H30_L13H11": [
        (3, 4),
        (8, 30),
        (13, 11),
    ],

    "F_L3H4_L18H13_L13H11": [
        (3, 4),
        (18, 13),
        (13, 11),
    ],

    "G_L8H30_L18H13_L13H11": [
        (8, 30),
        (18, 13),
        (13, 11),
    ],

    "H_L3H4_L8H30_L18H13_L13H11": [
        (3, 4),
        (8, 30),
        (18, 13),
        (13, 11),
    ],
}


def head_name(head):
    layer, h = head
    return f"L{layer}H{h}"


def run_subset(
    model,
    processor,
    inputs,
    numeral_ids,
    gt,
    object_token_sets,
    subset,
):
    grouped = defaultdict(list)

    for layer, head in subset:
        grouped[int(layer)].append(
            int(head)
        )

    dams = []

    for layer, heads in grouped.items():
        dams.append(
            DistributedLayerDAM(
                model=model,
                layer_idx=layer,
                head_indices=sorted(heads),
                object_token_sets=object_token_sets,
                beta=BETA,
                gamma=GAMMA,
                query_scope=QUERY_SCOPE,
            )
        )

    for dam in dams:
        dam.register()

    try:
        score = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

        text, pred = generate_answer(
            model,
            processor,
            inputs,
        )

    finally:
        for dam in reversed(dams):
            dam.remove()

    calls_ok = True
    observed_calls = {}

    for dam in dams:
        for head in dam.head_indices:
            calls = dam.head_calls[head]

            observed_calls[
                f"L{dam.layer_idx}H{head}"
            ] = calls

            if calls <= 0:
                calls_ok = False

    return {
        "score": score,
        "prediction": pred,
        "text": text,
        "calls_ok": calls_ok,
        "calls": observed_calls,
    }


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "AROMA DAM-v2 Compatible "
        "Subset Search"
    )
    print("=" * 100)

    print(
        "Sample:",
        SAMPLE_ID,
    )

    print(
        "Beta:",
        BETA,
    )

    print(
        "Gamma:",
        GAMMA,
    )

    print(
        "Query scope:",
        QUERY_SCOPE,
    )

    metadata = load_metadata()

    sample = metadata[
        SAMPLE_ID
    ]

    gt = int(
        sample["ground_truth"]
    )

    print(
        "Ground truth:",
        gt,
    )

    print(
        "Condition:",
        sample["condition"],
    )

    print(
        "\nCandidate subsets:"
    )

    for name, subset in SUBSETS.items():
        print(
            f"  {name}: "
            +
            ", ".join(
                head_name(h)
                for h in subset
            )
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

    image = (
        Image.open(
            sample["image_path"]
        )
        .convert("RGB")
    )

    (
        _geometry,
        object_token_sets,
    ) = build_object_token_sets(
        sample,
        processor,
        image,
    )

    (
        _formatted,
        inputs,
    ) = prepare_inputs(
        processor,
        image,
    )

    inputs = move_inputs(
        inputs,
        model,
    )

    numeral_ids = numeral_token_ids(
        processor
    )

    # ------------------------------------------------------
    # Baseline
    # ------------------------------------------------------

    baseline_score = score_state(
        model,
        inputs,
        numeral_ids,
        gt,
    )

    baseline_text, baseline_pred = (
        generate_answer(
            model,
            processor,
            inputs,
        )
    )

    baseline_logp = float(
        baseline_score["gt_logp"]
    )

    baseline_margin = float(
        baseline_score["margin"]
    )

    print(
        "\n" + "=" * 100
    )
    print("BASELINE")
    print("=" * 100)

    print(
        "Prediction:",
        baseline_pred,
    )

    print(
        "Text:",
        repr(baseline_text),
    )

    print(
        "GT logp:",
        baseline_logp,
    )

    print(
        "GT margin:",
        baseline_margin,
    )

    # ------------------------------------------------------
    # Single-head effects needed for additivity expectation
    # ------------------------------------------------------

    unique_heads = []

    seen = set()

    for subset in SUBSETS.values():
        for head in subset:
            if head not in seen:
                seen.add(head)
                unique_heads.append(head)

    single_effects = {}

    print(
        "\n" + "=" * 100
    )

    print(
        "SINGLE-HEAD REFERENCE EFFECTS"
    )

    print("=" * 100)

    for head in unique_heads:
        result = run_subset(
            model=model,
            processor=processor,
            inputs=inputs,
            numeral_ids=numeral_ids,
            gt=gt,
            object_token_sets=object_token_sets,
            subset=[head],
        )

        dlogp = (
            float(
                result[
                    "score"
                ][
                    "gt_logp"
                ]
            )
            -
            baseline_logp
        )

        dmargin = (
            float(
                result[
                    "score"
                ][
                    "margin"
                ]
            )
            -
            baseline_margin
        )

        single_effects[
            head
        ] = {
            "delta_logp":
                dlogp,

            "delta_margin":
                dmargin,
        }

        print(
            f"{head_name(head):10s} "
            f"dlogp={dlogp:+.8f} "
            f"dmargin={dmargin:+.8f}"
        )

    # ------------------------------------------------------
    # Subset search
    # ------------------------------------------------------

    rows = []

    print(
        "\n" + "=" * 100
    )

    print(
        "SUBSET SEARCH"
    )

    print("=" * 100)

    for subset_name, subset in (
        SUBSETS.items()
    ):
        result = run_subset(
            model=model,
            processor=processor,
            inputs=inputs,
            numeral_ids=numeral_ids,
            gt=gt,
            object_token_sets=object_token_sets,
            subset=subset,
        )

        score = result[
            "score"
        ]

        pred = result[
            "prediction"
        ]

        delta_logp = (
            float(
                score[
                    "gt_logp"
                ]
            )
            -
            baseline_logp
        )

        delta_margin = (
            float(
                score[
                    "margin"
                ]
            )
            -
            baseline_margin
        )

        expected_logp = sum(
            single_effects[
                head
            ][
                "delta_logp"
            ]
            for head
            in subset
        )

        expected_margin = sum(
            single_effects[
                head
            ][
                "delta_margin"
            ]
            for head
            in subset
        )

        interaction_residual_logp = (
            delta_logp
            -
            expected_logp
        )

        interaction_residual_margin = (
            delta_margin
            -
            expected_margin
        )

        repaired = (
            baseline_pred != gt
            and
            pred == gt
        )

        broken = (
            baseline_pred == gt
            and
            pred != gt
        )

        rows.append(
            {
                "subset":
                    subset_name,

                "heads":
                    "|".join(
                        head_name(h)
                        for h in subset
                    ),

                "size":
                    len(subset),

                "prediction":
                    pred,

                "delta_logp":
                    delta_logp,

                "delta_margin":
                    delta_margin,

                "expected_additive_logp":
                    expected_logp,

                "expected_additive_margin":
                    expected_margin,

                "interaction_residual_logp":
                    interaction_residual_logp,

                "interaction_residual_margin":
                    interaction_residual_margin,

                "repaired":
                    repaired,

                "broken":
                    broken,

                "calls_ok":
                    result[
                        "calls_ok"
                    ],
            }
        )

        print(
            f"{subset_name:32s} | "
            f"size={len(subset)} | "
            f"pred={pred} | "
            f"dlogp={delta_logp:+.8f} | "
            f"dmargin={delta_margin:+.8f} | "
            f"ires={interaction_residual_logp:+.8f} | "
            f"repair={repaired} | "
            f"calls_ok={result['calls_ok']}"
        )

    df = pd.DataFrame(
        rows
    )

    df = df.sort_values(
        [
            "repaired",
            "delta_logp",
        ],
        ascending=[
            False,
            False,
        ],
    )

    output_csv = (
        OUT_DIR
        /
        "subset_search_results.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "RANKED SUBSETS"
    )

    print("=" * 100)

    print(
        df[
            [
                "subset",
                "size",
                "prediction",
                "delta_logp",
                "delta_margin",
                "interaction_residual_logp",
                "repaired",
                "calls_ok",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nBest subset by delta_logp:"
    )

    best = df.iloc[0]

    print(
        best.to_string()
    )

    print(
        "\nSaved:"
    )

    print(
        output_csv
    )

    print(
        "\nDAM-v2 SUBSET SEARCH COMPLETE"
    )


if __name__ == "__main__":
    main()
