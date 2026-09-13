import itertools
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

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "dam_v2_pairwise"
)


def run_subset(
    model,
    inputs,
    numeral_ids,
    gt,
    object_token_sets,
    subset,
):
    grouped = defaultdict(list)

    for layer, head in subset:
        grouped[
            int(layer)
        ].append(
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

    finally:
        for dam in reversed(dams):
            dam.remove()

    calls_ok = True

    for dam in dams:
        for head in dam.head_indices:
            if dam.head_calls[head] <= 0:
                calls_ok = False

    return score, calls_ok


def head_name(head):
    layer, h = head
    return f"L{layer}H{h}"


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print(
        "AROMA DAM-v2 Pairwise "
        "Interaction Map"
    )
    print("=" * 90)

    metadata = load_metadata()
    sample = metadata[SAMPLE_ID]

    gt = int(
        sample["ground_truth"]
    )

    heads = load_candidate_heads()

    print(
        "Sample:",
        SAMPLE_ID,
    )

    print(
        "GT:",
        gt,
    )

    print(
        "Heads:",
        len(heads),
    )

    for h in heads:
        print(
            " ",
            head_name(h),
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

    baseline = score_state(
        model,
        inputs,
        numeral_ids,
        gt,
    )

    baseline_logp = float(
        baseline["gt_logp"]
    )

    baseline_margin = float(
        baseline["margin"]
    )

    print(
        "\nBaseline GT logp:",
        baseline_logp,
    )

    print(
        "Baseline margin:",
        baseline_margin,
    )

    # ------------------------------------------------------
    # Singles
    # ------------------------------------------------------

    singles = {}

    print(
        "\n" + "=" * 90
    )

    print("SINGLE HEADS")
    print("=" * 90)

    for head in heads:
        score, calls_ok = run_subset(
            model=model,
            inputs=inputs,
            numeral_ids=numeral_ids,
            gt=gt,
            object_token_sets=object_token_sets,
            subset=[head],
        )

        dlogp = (
            float(score["gt_logp"])
            - baseline_logp
        )

        dmargin = (
            float(score["margin"])
            - baseline_margin
        )

        singles[head] = {
            "delta_logp": dlogp,
            "delta_margin": dmargin,
        }

        print(
            f"{head_name(head):8s} "
            f"dlogp={dlogp:+.8f} "
            f"dmargin={dmargin:+.8f} "
            f"calls_ok={calls_ok}"
        )

    # ------------------------------------------------------
    # Pairs
    # ------------------------------------------------------

    rows = []

    print(
        "\n" + "=" * 90
    )

    print("PAIRWISE INTERACTIONS")
    print("=" * 90)

    for h1, h2 in itertools.combinations(
        heads,
        2,
    ):
        score, calls_ok = run_subset(
            model=model,
            inputs=inputs,
            numeral_ids=numeral_ids,
            gt=gt,
            object_token_sets=object_token_sets,
            subset=[
                h1,
                h2,
            ],
        )

        pair_dlogp = (
            float(score["gt_logp"])
            - baseline_logp
        )

        pair_dmargin = (
            float(score["margin"])
            - baseline_margin
        )

        expected_logp = (
            singles[h1]["delta_logp"]
            +
            singles[h2]["delta_logp"]
        )

        expected_margin = (
            singles[h1]["delta_margin"]
            +
            singles[h2]["delta_margin"]
        )

        interaction_logp = (
            pair_dlogp
            -
            expected_logp
        )

        interaction_margin = (
            pair_dmargin
            -
            expected_margin
        )

        same_layer = (
            h1[0] == h2[0]
        )

        row = {
            "head_1": head_name(h1),
            "head_2": head_name(h2),

            "layer_1": h1[0],
            "head_idx_1": h1[1],

            "layer_2": h2[0],
            "head_idx_2": h2[1],

            "same_layer": same_layer,

            "single_1_delta_logp":
                singles[h1][
                    "delta_logp"
                ],

            "single_2_delta_logp":
                singles[h2][
                    "delta_logp"
                ],

            "pair_delta_logp":
                pair_dlogp,

            "expected_additive_logp":
                expected_logp,

            "interaction_logp":
                interaction_logp,

            "pair_delta_margin":
                pair_dmargin,

            "expected_additive_margin":
                expected_margin,

            "interaction_margin":
                interaction_margin,

            "calls_ok":
                calls_ok,
        }

        rows.append(row)

        print(
            f"{head_name(h1):8s} + "
            f"{head_name(h2):8s} | "
            f"pair={pair_dlogp:+.8f} | "
            f"interaction="
            f"{interaction_logp:+.8f} | "
            f"same_layer={same_layer}"
        )

    df = pd.DataFrame(rows)

    output_csv = (
        OUT_DIR
        /
        "pairwise_interactions.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # ------------------------------------------------------
    # Ranked antagonism / synergy
    # ------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "STRONGEST ANTAGONISTIC PAIRS"
    )

    print("=" * 90)

    antagonistic = (
        df.sort_values(
            "interaction_logp",
            ascending=True,
        )
        .head(10)
    )

    print(
        antagonistic[
            [
                "head_1",
                "head_2",
                "same_layer",
                "pair_delta_logp",
                "interaction_logp",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "MOST COMPATIBLE / "
        "SYNERGISTIC PAIRS"
    )

    print("=" * 90)

    synergistic = (
        df.sort_values(
            "interaction_logp",
            ascending=False,
        )
        .head(10)
    )

    print(
        synergistic[
            [
                "head_1",
                "head_2",
                "same_layer",
                "pair_delta_logp",
                "interaction_logp",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Interaction matrix
    # ------------------------------------------------------

    names = [
        head_name(h)
        for h in heads
    ]

    matrix = pd.DataFrame(
        0.0,
        index=names,
        columns=names,
    )

    for _, row in df.iterrows():
        matrix.loc[
            row["head_1"],
            row["head_2"],
        ] = row[
            "interaction_logp"
        ]

        matrix.loc[
            row["head_2"],
            row["head_1"],
        ] = row[
            "interaction_logp"
        ]

    matrix_path = (
        OUT_DIR
        /
        "interaction_matrix.csv"
    )

    matrix.to_csv(
        matrix_path
    )

    print(
        "\nSaved:"
    )

    print(
        output_csv
    )

    print(
        matrix_path
    )

    print(
        "\nPAIRWISE INTERACTION "
        "MAP COMPLETE"
    )


if __name__ == "__main__":
    main()
