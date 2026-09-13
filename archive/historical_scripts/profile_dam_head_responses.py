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

PAIR_FILE = Path(
    "outputs/proc_count_causal_v1/"
    "exact_matched_pairs_eager.csv"
)

DISCOVERY_SAMPLE = (
    "pccv1_n05_row_r00"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "dam_head_profiles"
)

BETA = 1.25
GAMMA = 0.0
QUERY_SCOPE = "answer"


FROZEN_HEADS = [
    (33, 1),
    (3, 4),
    (18, 13),
    (8, 30),
    (3, 11),
    (13, 11),
    (33, 21),
]


def head_name(layer, head):
    return f"L{layer}H{head}"


def run_single_head(
    model,
    inputs,
    numeral_ids,
    gt,
    object_token_sets,
    layer,
    head,
):
    dam = DistributedLayerDAM(
        model=model,
        layer_idx=int(layer),
        head_indices=[int(head)],
        object_token_sets=object_token_sets,
        beta=BETA,
        gamma=GAMMA,
        query_scope=QUERY_SCOPE,
    )

    dam.register()

    try:
        score = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )
    finally:
        dam.remove()

    calls = 0

    try:
        calls = int(
            dam.head_calls[
                int(head)
            ]
        )
    except Exception:
        pass

    return score, calls


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "AROMA DAM-v2 "
        "Per-Sample Head Response Profiling"
    )
    print("=" * 100)

    pairs = pd.read_csv(
        PAIR_FILE
    )

    # Exclude discovery pair completely
    mask = (
        (
            pairs["wrong_sample_id"]
            .astype(str)
            == DISCOVERY_SAMPLE
        )
        |
        (
            pairs["correct_sample_id"]
            .astype(str)
            == DISCOVERY_SAMPLE
        )
    )

    removed = pairs[mask].copy()

    pairs = (
        pairs[~mask]
        .reset_index(drop=True)
    )

    print(
        "Original pairs:",
        len(pairs)
        + len(removed),
    )

    print(
        "Removed discovery pairs:",
        len(removed),
    )

    print(
        "Held-out pairs:",
        len(pairs),
    )

    sample_rows = []

    for _, row in pairs.iterrows():
        sample_rows.append(
            {
                "pair_id":
                    int(row["pair_id"]),

                "role":
                    "wrong",

                "sample_id":
                    str(
                        row[
                            "wrong_sample_id"
                        ]
                    ),

                "ground_truth":
                    int(
                        row[
                            "ground_truth"
                        ]
                    ),

                "condition":
                    str(
                        row[
                            "condition"
                        ]
                    ),
            }
        )

        sample_rows.append(
            {
                "pair_id":
                    int(row["pair_id"]),

                "role":
                    "correct",

                "sample_id":
                    str(
                        row[
                            "correct_sample_id"
                        ]
                    ),

                "ground_truth":
                    int(
                        row[
                            "ground_truth"
                        ]
                    ),

                "condition":
                    str(
                        row[
                            "condition"
                        ]
                    ),
            }
        )

    print(
        "Held-out samples:",
        len(sample_rows),
    )

    print(
        "Heads:",
        len(FROZEN_HEADS),
    )

    print(
        "Expected rows:",
        len(sample_rows)
        * len(FROZEN_HEADS),
    )

    print(
        "\nFrozen heads:"
    )

    for layer, head in FROZEN_HEADS:
        print(
            " ",
            head_name(
                layer,
                head,
            ),
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

    metadata = load_metadata()

    rows = []

    for item in tqdm(
        sample_rows,
        desc="Head profiling",
    ):
        sid = item[
            "sample_id"
        ]

        sample = metadata[
            sid
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
            .convert(
                "RGB"
            )
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

        baseline = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

        baseline_logp = float(
            baseline[
                "gt_logp"
            ]
        )

        baseline_margin = float(
            baseline[
                "margin"
            ]
        )

        sample_start = len(
            rows
        )

        for layer, head in (
            FROZEN_HEADS
        ):
            score, calls = (
                run_single_head(
                    model=model,
                    inputs=inputs,
                    numeral_ids=
                        numeral_ids,
                    gt=gt,
                    object_token_sets=
                        object_token_sets,
                    layer=layer,
                    head=head,
                )
            )

            intervened_logp = float(
                score[
                    "gt_logp"
                ]
            )

            intervened_margin = float(
                score[
                    "margin"
                ]
            )

            rows.append(
                {
                    "pair_id":
                        item[
                            "pair_id"
                        ],

                    "role":
                        item[
                            "role"
                        ],

                    "sample_id":
                        sid,

                    "ground_truth":
                        gt,

                    "condition":
                        item[
                            "condition"
                        ],

                    "layer":
                        int(
                            layer
                        ),

                    "head":
                        int(
                            head
                        ),

                    "head_name":
                        head_name(
                            layer,
                            head,
                        ),

                    "baseline_gt_logp":
                        baseline_logp,

                    "baseline_margin":
                        baseline_margin,

                    "intervened_gt_logp":
                        intervened_logp,

                    "intervened_margin":
                        intervened_margin,

                    "delta_logp":
                        (
                            intervened_logp
                            -
                            baseline_logp
                        ),

                    "delta_margin":
                        (
                            intervened_margin
                            -
                            baseline_margin
                        ),

                    "calls":
                        calls,

                    "calls_ok":
                        calls > 0,
                }
            )

        # rank heads within this sample
        sample_end = len(
            rows
        )

        local = rows[
            sample_start:
            sample_end
        ]

        ordered = sorted(
            range(
                len(local)
            ),
            key=lambda i:
                local[i][
                    "delta_logp"
                ],
            reverse=True,
        )

        for rank, idx in enumerate(
            ordered,
            start=1,
        ):
            local[
                idx
            ][
                "rank_within_sample"
            ] = rank

            local[
                idx
            ][
                "is_best_head"
            ] = (
                rank == 1
            )

    df = pd.DataFrame(
        rows
    )

    out_path = (
        OUT_DIR
        /
        "head_response_profiles.csv"
    )

    df.to_csv(
        out_path,
        index=False,
    )

    print(
        "\nSaved:",
        out_path,
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "All calls ok:",
        bool(
            df[
                "calls_ok"
            ].all()
        ),
    )

    print(
        "\nPROFILE COMPLETE"
    )


if __name__ == "__main__":
    main()
