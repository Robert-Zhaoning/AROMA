import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)
import torch

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
    load_candidate_heads,
)


MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

SAMPLE_ID = "pccv1_n05_row_r00"

BETA = 1.25
GAMMA = 0.0
QUERY_SCOPE = "answer"


def run_intervention(
    model,
    processor,
    inputs,
    numeral_ids,
    gt,
    object_token_sets,
    selected_heads,
):
    """
    selected_heads:
        list of (layer, head)
    """

    grouped = defaultdict(list)

    for layer, head in selected_heads:
        grouped[int(layer)].append(
            int(head)
        )

    dams = []

    for layer, heads in grouped.items():
        dam = DistributedLayerDAM(
            model=model,
            layer_idx=layer,
            head_indices=sorted(heads),
            object_token_sets=object_token_sets,
            beta=BETA,
            gamma=GAMMA,
            query_scope=QUERY_SCOPE,
        )

        dams.append(dam)

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

    all_calls = {}

    for dam in dams:
        for head in dam.head_indices:
            all_calls[
                (dam.layer_idx, head)
            ] = dam.head_calls[head]

    return {
        "score": score,
        "text": text,
        "prediction": pred,
        "calls": all_calls,
    }


def main():
    print("=" * 100)
    print(
        "AROMA Distributed DAM-v2 "
        "Decomposition Diagnostic"
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

    metadata = load_metadata()

    sample = metadata[
        SAMPLE_ID
    ]

    gt = int(
        sample["ground_truth"]
    )

    frozen_heads = (
        load_candidate_heads()
    )

    print(
        "\nFrozen heads:"
    )

    for layer, head in frozen_heads:
        print(
            f"  L{layer}H{head}"
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
        geometry,
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

    # ======================================================
    # Baseline
    # ======================================================

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
        "GT logp:",
        baseline_score["gt_logp"],
    )

    print(
        "GT margin:",
        baseline_score["margin"],
    )

    results = []

    # ======================================================
    # Each frozen head separately
    # ======================================================

    print(
        "\n" + "=" * 100
    )
    print(
        "SINGLE-HEAD DECOMPOSITION"
    )
    print("=" * 100)

    for layer, head in frozen_heads:

        result = run_intervention(
            model=model,
            processor=processor,
            inputs=inputs,
            numeral_ids=numeral_ids,
            gt=gt,
            object_token_sets=object_token_sets,
            selected_heads=[
                (layer, head)
            ],
        )

        delta_logp = (
            result["score"]["gt_logp"]
            -
            baseline_score["gt_logp"]
        )

        delta_margin = (
            result["score"]["margin"]
            -
            baseline_score["margin"]
        )

        calls = result[
            "calls"
        ].get(
            (layer, head),
            0,
        )

        row = {
            "name":
                f"L{layer}H{head}",

            "prediction":
                result[
                    "prediction"
                ],

            "delta_logp":
                delta_logp,

            "delta_margin":
                delta_margin,

            "calls":
                calls,
        }

        results.append(row)

        print(
            f"L{layer}H{head:02d} | "
            f"pred={result['prediction']} | "
            f"dlogp={delta_logp:+.8f} | "
            f"dmargin={delta_margin:+.8f} | "
            f"calls={calls}"
        )

    # ======================================================
    # Same-layer combinations
    # ======================================================

    print(
        "\n" + "=" * 100
    )
    print(
        "SAME-LAYER COMBINATIONS"
    )
    print("=" * 100)

    grouped = defaultdict(list)

    for layer, head in frozen_heads:
        grouped[
            int(layer)
        ].append(
            int(head)
        )

    for layer, heads in sorted(
        grouped.items()
    ):
        if len(heads) < 2:
            continue

        selected = [
            (
                layer,
                head,
            )
            for head in heads
        ]

        result = run_intervention(
            model=model,
            processor=processor,
            inputs=inputs,
            numeral_ids=numeral_ids,
            gt=gt,
            object_token_sets=object_token_sets,
            selected_heads=selected,
        )

        delta_logp = (
            result["score"]["gt_logp"]
            -
            baseline_score["gt_logp"]
        )

        delta_margin = (
            result["score"]["margin"]
            -
            baseline_score["margin"]
        )

        label = (
            f"L{layer}["
            +
            ",".join(
                f"H{x}"
                for x in heads
            )
            +
            "]"
        )

        print(
            f"{label:20s} | "
            f"pred={result['prediction']} | "
            f"dlogp={delta_logp:+.8f} | "
            f"dmargin={delta_margin:+.8f}"
        )

    # ======================================================
    # All seven heads simultaneously
    # ======================================================

    print(
        "\n" + "=" * 100
    )
    print(
        "ALL FROZEN HEADS"
    )
    print("=" * 100)

    all_result = run_intervention(
        model=model,
        processor=processor,
        inputs=inputs,
        numeral_ids=numeral_ids,
        gt=gt,
        object_token_sets=object_token_sets,
        selected_heads=frozen_heads,
    )

    all_delta_logp = (
        all_result[
            "score"
        ][
            "gt_logp"
        ]
        -
        baseline_score[
            "gt_logp"
        ]
    )

    all_delta_margin = (
        all_result[
            "score"
        ][
            "margin"
        ]
        -
        baseline_score[
            "margin"
        ]
    )

    print(
        "Prediction:",
        all_result[
            "prediction"
        ],
    )

    print(
        "Delta GT logp:",
        all_delta_logp,
    )

    print(
        "Delta GT margin:",
        all_delta_margin,
    )

    print(
        "Calls:"
    )

    for key, value in sorted(
        all_result[
            "calls"
        ].items()
    ):
        print(
            f"  L{key[0]}H{key[1]}: "
            f"{value}"
        )

    # ======================================================
    # Additivity comparison
    # ======================================================

    sum_single_logp = sum(
        r["delta_logp"]
        for r in results
    )

    sum_single_margin = sum(
        r["delta_margin"]
        for r in results
    )

    print(
        "\n" + "=" * 100
    )
    print(
        "ADDITIVITY DIAGNOSTIC"
    )
    print("=" * 100)

    print(
        "Sum of single-head delta logp:",
        sum_single_logp,
    )

    print(
        "Actual 7-head delta logp:",
        all_delta_logp,
    )

    print(
        "Interaction residual logp:",
        (
            all_delta_logp
            -
            sum_single_logp
        ),
    )

    print(
        "\nSum of single-head delta margin:",
        sum_single_margin,
    )

    print(
        "Actual 7-head delta margin:",
        all_delta_margin,
    )

    print(
        "Interaction residual margin:",
        (
            all_delta_margin
            -
            sum_single_margin
        ),
    )

    print(
        "\nDECOMPOSITION DIAGNOSTIC COMPLETE"
    )


if __name__ == "__main__":
    main()
