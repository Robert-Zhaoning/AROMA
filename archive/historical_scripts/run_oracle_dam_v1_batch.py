import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

from oracle_dam_v1_smoke import (
    OracleDAM,
    build_object_token_sets,
    generate_answer,
    load_metadata,
    move_inputs,
    numeral_token_ids,
    prepare_inputs,
    score_state,
)


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

PAIR_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "exact_matched_pairs_eager.csv"
)

HEAD_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "frozen_heads.json"
)

OUTPUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "oracle_dam_v1_batch"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "oracle_dam_v1_batch_results.csv"
)

GAMMAS = [
    0.0,
    0.25,
    0.5,
    1.0,
]

QUERY_SCOPE = "answer"

RANDOM_REGION_SEED = 20260910


def load_candidate_heads():

    config = json.loads(
        HEAD_PATH.read_text(
            encoding="utf-8"
        )
    )

    return [
        (
            int(item["layer"]),
            int(item["head"]),
        )
        for item in config[
            "candidate_heads"
        ]
    ]


def get_sample_roles(
    pairs,
):

    roles = {}

    for _, row in pairs.iterrows():

        roles[
            row[
                "correct_sample_id"
            ]
        ] = "correct"

        roles[
            row[
                "wrong_sample_id"
            ]
        ] = "wrong"

    return roles


def make_random_region_sets(
    object_token_sets,
    visual_length,
    seed,
):
    """
    Random-region control.

    Preserve:
      - number of regions
      - number of visual tokens per region

    But replace GT-object patches with random visual-token sets.

    Random regions are sampled without replacement where possible.
    """

    rng = random.Random(
        seed
    )

    # Exclude tile class/global tokens at:
    # 0, 1601, 3202, 4803 approximately.
    #
    # More generally, just avoid every 1601st token.
    candidate_tokens = [
        t
        for t in range(
            visual_length
        )
        if (
            t % 1601
            != 0
        )
    ]

    sizes = [
        len(x)
        for x in object_token_sets
    ]

    total_needed = sum(
        sizes
    )

    if total_needed <= len(
        candidate_tokens
    ):

        chosen = rng.sample(
            candidate_tokens,
            total_needed,
        )

        result = []

        offset = 0

        for size in sizes:

            result.append(
                sorted(
                    chosen[
                        offset:
                        offset + size
                    ]
                )
            )

            offset += size

        return result

    # Fallback, unlikely here.

    result = []

    for size in sizes:

        result.append(
            sorted(
                rng.sample(
                    candidate_tokens,
                    size,
                )
            )
        )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata()

    pairs = pd.read_csv(
        PAIR_PATH
    )

    roles = get_sample_roles(
        pairs
    )

    sample_ids = sorted(
        roles.keys()
    )

    if args.limit is not None:

        sample_ids = sample_ids[
            :args.limit
        ]

    candidate_heads = (
        load_candidate_heads()
    )

    print("=" * 96)
    print(
        "AROMA Oracle DAM-v1 Batch"
    )
    print("=" * 96)

    print(
        "Exact pairs:",
        len(pairs),
    )

    print(
        "Unique samples:",
        len(sample_ids),
    )

    print(
        "Candidate heads:",
        len(candidate_heads),
    )

    print(
        "Gammas:",
        GAMMAS,
    )

    print(
        "Query scope:",
        QUERY_SCOPE,
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

    total = (
        len(sample_ids)
        * len(candidate_heads)
        * len(GAMMAS)
        * 2
    )

    progress = tqdm(
        total=total,
        desc="Oracle DAM batch",
    )

    for sample_id in sample_ids:

        sample = metadata[
            sample_id
        ]

        role = roles[
            sample_id
        ]

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

        (
            _geometry,
            gt_object_sets,
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

        baseline_score = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

        (
            baseline_text,
            baseline_pred,
        ) = generate_answer(
            model,
            processor,
            inputs,
        )

        # Get actual visual sequence length from
        # one no-op forward's cross-attention source length.
        #
        # For PCC-v1 this is 6404, but derive from
        # active tile count whenever possible.
        pixel_values = inputs[
            "pixel_values"
        ]

        max_tiles = int(
            pixel_values.shape[2]
        )

        visual_length = (
            max_tiles
            * 1601
        )

        random_sets = (
            make_random_region_sets(
                object_token_sets=gt_object_sets,
                visual_length=visual_length,
                seed=(
                    RANDOM_REGION_SEED
                    + sum(
                        ord(c)
                        for c in sample_id
                    )
                ),
            )
        )

        for (
            layer,
            head,
        ) in candidate_heads:

            for gamma in GAMMAS:

                for region_type, region_sets in [
                    (
                        "oracle_gt",
                        gt_object_sets,
                    ),
                    (
                        "random_region",
                        random_sets,
                    ),
                ]:

                    dam = OracleDAM(
                        model=model,
                        layer_idx=layer,
                        head_idx=head,
                        object_token_sets=region_sets,
                        gamma=gamma,
                        query_scope=QUERY_SCOPE,
                    )

                    dam.register()

                    try:

                        dam_score = score_state(
                            model,
                            inputs,
                            numeral_ids,
                            gt,
                        )

                        (
                            dam_text,
                            dam_pred,
                        ) = generate_answer(
                            model,
                            processor,
                            inputs,
                        )

                    finally:

                        dam.remove()

                    repaired = (
                        baseline_pred
                        != gt
                        and
                        dam_pred
                        == gt
                    )

                    broken = (
                        baseline_pred
                        == gt
                        and
                        dam_pred
                        != gt
                    )

                    before = (
                        dam.last_before_masses
                    )

                    after = (
                        dam.last_after_masses
                    )

                    if (
                        before is not None
                        and
                        len(before) > 1
                    ):

                        before_cv = float(
                            before.float().std(
                                unbiased=False
                            )
                            /
                            (
                                before.float().mean()
                                + 1e-12
                            )
                        )

                    else:

                        before_cv = None

                    if (
                        after is not None
                        and
                        len(after) > 1
                    ):

                        after_cv = float(
                            after.float().std(
                                unbiased=False
                            )
                            /
                            (
                                after.float().mean()
                                + 1e-12
                            )
                        )

                    else:

                        after_cv = None

                    rows.append(
                        {
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

                            "layer":
                                layer,

                            "head":
                                head,

                            "gamma":
                                gamma,

                            "query_scope":
                                QUERY_SCOPE,

                            "region_type":
                                region_type,

                            "baseline_prediction":
                                baseline_pred,

                            "dam_prediction":
                                dam_pred,

                            "baseline_gt_logp":
                                baseline_score[
                                    "gt_logp"
                                ],

                            "dam_gt_logp":
                                dam_score[
                                    "gt_logp"
                                ],

                            "delta_gt_logp":
                                (
                                    dam_score[
                                        "gt_logp"
                                    ]
                                    -
                                    baseline_score[
                                        "gt_logp"
                                    ]
                                ),

                            "baseline_margin":
                                baseline_score[
                                    "margin"
                                ],

                            "dam_margin":
                                dam_score[
                                    "margin"
                                ],

                            "delta_margin":
                                (
                                    dam_score[
                                        "margin"
                                    ]
                                    -
                                    baseline_score[
                                        "margin"
                                    ]
                                ),

                            "repaired":
                                repaired,

                            "broken":
                                broken,

                            "dam_calls":
                                dam.calls,

                            "max_row_sum_error":
                                dam.last_row_sum_error,

                            "before_object_cv":
                                before_cv,

                            "after_object_cv":
                                after_cv,

                            "delta_object_cv":
                                (
                                    None
                                    if (
                                        before_cv is None
                                        or
                                        after_cv is None
                                    )
                                    else
                                    after_cv
                                    - before_cv
                                ),

                            "raw_text":
                                dam_text,
                        }
                    )

                    progress.update(
                        1
                    )

    progress.close()

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )

    print(
        "Rows:",
        len(out),
    )

    expected = (
        len(sample_ids)
        * len(candidate_heads)
        * len(GAMMAS)
        * 2
    )

    print(
        "Expected rows:",
        expected,
    )

    if len(out) != expected:

        raise RuntimeError(
            "Unexpected output row count."
        )

    print(
        "\nORACLE DAM-v1 BATCH COMPLETE"
    )


if __name__ == "__main__":
    main()
