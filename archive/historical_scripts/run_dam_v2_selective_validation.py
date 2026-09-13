from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
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
)


MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

DISCOVERY_SAMPLE = (
    "pccv1_n05_row_r00"
)

BETA = 1.25
GAMMA = 0.0
QUERY_SCOPE = "answer"

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "dam_v2_selective_validation"
)


SUBSETS = {
    "single_L3H4": [
        (3, 4),
    ],

    "single_L8H30": [
        (8, 30),
    ],

    "selective_A": [
        (3, 4),
        (8, 30),
    ],

    "destructive_F": [
        (3, 4),
        (18, 13),
        (13, 11),
    ],

    "all7": [
        (33, 1),
        (3, 4),
        (18, 13),
        (8, 30),
        (3, 11),
        (13, 11),
        (33, 21),
    ],
}


def find_pair_file():
    candidates = [
        Path(
            "outputs/proc_count_causal_v1/"
            "exact_matched_pairs_eager.csv"
        ),
        Path(
            "outputs/proc_count_causal_v1/"
            "exact_matched_pairs.csv"
        ),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find exact matched-pair CSV."
    )


def find_sample_columns(df):
    wrong_candidates = [
        "wrong_sample_id",
        "wrong_id",
    ]

    correct_candidates = [
        "correct_sample_id",
        "correct_id",
    ]

    wrong_col = next(
        (
            x
            for x in wrong_candidates
            if x in df.columns
        ),
        None,
    )

    correct_col = next(
        (
            x
            for x in correct_candidates
            if x in df.columns
        ),
        None,
    )

    if wrong_col is None:
        wrong_col = next(
            (
                x
                for x in df.columns
                if "wrong" in x.lower()
                and "sample" in x.lower()
            ),
            None,
        )

    if correct_col is None:
        correct_col = next(
            (
                x
                for x in df.columns
                if "correct" in x.lower()
                and "sample" in x.lower()
            ),
            None,
        )

    if wrong_col is None or correct_col is None:
        raise RuntimeError(
            "Could not infer wrong/correct sample ID columns.\n"
            f"Columns = {df.columns.tolist()}"
        )

    return wrong_col, correct_col


def register_subset(
    model,
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
        dam = DistributedLayerDAM(
            model=model,
            layer_idx=layer,
            head_indices=sorted(heads),
            object_token_sets=object_token_sets,
            beta=BETA,
            gamma=GAMMA,
            query_scope=QUERY_SCOPE,
        )

        dam.register()
        dams.append(dam)

    return dams


def remove_dams(dams):
    for dam in reversed(dams):
        dam.remove()


def run_score(
    model,
    inputs,
    numeral_ids,
    gt,
    object_token_sets,
    subset=None,
):
    dams = []

    if subset:
        dams = register_subset(
            model,
            object_token_sets,
            subset,
        )

    try:
        score = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

    finally:
        remove_dams(dams)

    return score


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "AROMA DAM-v2 Selective "
        "Held-out Validation"
    )
    print("=" * 100)

    pair_path = find_pair_file()
    pairs = pd.read_csv(pair_path)

    wrong_col, correct_col = (
        find_sample_columns(pairs)
    )

    print(
        "Pair file:",
        pair_path,
    )

    print(
        "Wrong column:",
        wrong_col,
    )

    print(
        "Correct column:",
        correct_col,
    )

    # ------------------------------------------------------
    # Remove discovery pair completely
    # ------------------------------------------------------

    before = len(pairs)

    mask = (
        (pairs[wrong_col].astype(str)
         == DISCOVERY_SAMPLE)
        |
        (pairs[correct_col].astype(str)
         == DISCOVERY_SAMPLE)
    )

    removed = pairs[mask].copy()

    pairs = (
        pairs[~mask]
        .reset_index(drop=True)
    )

    print(
        "\nOriginal pairs:",
        before,
    )

    print(
        "Removed discovery pairs:",
        len(removed),
    )

    print(
        "Held-out pairs:",
        len(pairs),
    )

    if len(removed):
        print(
            "\nRemoved:"
        )
        print(
            removed.to_string(
                index=False
            )
        )

    metadata = load_metadata()

    sample_rows = []

    for pair_idx, row in pairs.iterrows():
        sample_rows.append(
            {
                "pair_idx": pair_idx,
                "role": "wrong",
                "sample_id":
                    str(row[wrong_col]),
            }
        )

        sample_rows.append(
            {
                "pair_idx": pair_idx,
                "role": "correct",
                "sample_id":
                    str(row[correct_col]),
            }
        )

    print(
        "\nHeld-out samples:",
        len(sample_rows),
    )

    print(
        "Interventions/sample:",
        len(SUBSETS),
    )

    print(
        "Expected intervention rows:",
        len(sample_rows)
        *
        len(SUBSETS),
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

    numeral_ids = numeral_token_ids(
        processor
    )

    rows = []

    iterator = tqdm(
        sample_rows,
        desc="Selective DAM validation",
    )

    for item in iterator:
        sid = item["sample_id"]

        if sid not in metadata:
            raise KeyError(
                f"Sample not found in metadata: {sid}"
            )

        sample = metadata[sid]

        gt = int(
            sample["ground_truth"]
        )

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

        baseline = run_score(
            model=model,
            inputs=inputs,
            numeral_ids=numeral_ids,
            gt=gt,
            object_token_sets=object_token_sets,
            subset=None,
        )

        baseline_logp = float(
            baseline["gt_logp"]
        )

        baseline_margin = float(
            baseline["margin"]
        )

        for subset_name, subset in (
            SUBSETS.items()
        ):
            result = run_score(
                model=model,
                inputs=inputs,
                numeral_ids=numeral_ids,
                gt=gt,
                object_token_sets=
                    object_token_sets,
                subset=subset,
            )

            delta_logp = (
                float(
                    result["gt_logp"]
                )
                -
                baseline_logp
            )

            delta_margin = (
                float(
                    result["margin"]
                )
                -
                baseline_margin
            )

            rows.append(
                {
                    "pair_idx":
                        item["pair_idx"],

                    "role":
                        item["role"],

                    "sample_id":
                        sid,

                    "ground_truth":
                        gt,

                    "condition":
                        sample[
                            "condition"
                        ],

                    "subset":
                        subset_name,

                    "size":
                        len(subset),

                    "baseline_gt_logp":
                        baseline_logp,

                    "baseline_margin":
                        baseline_margin,

                    "intervened_gt_logp":
                        float(
                            result[
                                "gt_logp"
                            ]
                        ),

                    "intervened_margin":
                        float(
                            result[
                                "margin"
                            ]
                        ),

                    "delta_logp":
                        delta_logp,

                    "delta_margin":
                        delta_margin,
                }
            )

    out = pd.DataFrame(rows)

    output_path = (
        OUT_DIR
        /
        "selective_validation_results.csv"
    )

    out.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved:",
        output_path,
    )

    print(
        "Rows:",
        len(out),
    )

    print(
        "\nVALIDATION RUN COMPLETE"
    )


if __name__ == "__main__":
    main()
