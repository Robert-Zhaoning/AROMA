import json
import math
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


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

META_PATH = Path(
    "data/calibration50/metadata.jsonl"
)

PAIR_PATH = Path(
    "outputs/calibration50/attention_pilot/"
    "clean_matched_pairs.csv"
)

OUTPUT_DIR = Path(
    "outputs/calibration50/matched_attention"
)


TILE_SIZE = 560
PATCH_SIZE = 14
GRID = 40
TOKENS_PER_TILE = 1601

ACTIVE_TILES = [0, 1]

CROSS_LAYERS = [
    3,
    8,
    13,
    18,
    23,
    28,
    33,
    38,
]


PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


def load_metadata():
    records = {}

    with META_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                r = json.loads(line)
                records[r["sample_id"]] = r

    return records


def denorm_bbox(
    bbox,
    width,
    height,
):
    x1, y1, x2, y2 = bbox

    return (
        x1 * width,
        y1 * height,
        x2 * width,
        y2 * height,
    )


def bbox_to_tokens(
    bbox,
    width,
    height,
):
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    result = []

    for tile_idx in ACTIVE_TILES:

        tile_x0 = tile_idx * TILE_SIZE
        tile_x1 = tile_x0 + TILE_SIZE

        ix1 = max(x1, tile_x0)
        ix2 = min(x2, tile_x1)

        iy1 = max(y1, 0)
        iy2 = min(y2, TILE_SIZE)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        lx1 = ix1 - tile_x0
        lx2 = ix2 - tile_x0

        col_start = int(
            math.floor(
                lx1 / PATCH_SIZE
            )
        )

        col_end = (
            int(
                math.ceil(
                    lx2 / PATCH_SIZE
                )
            )
            - 1
        )

        row_start = int(
            math.floor(
                iy1 / PATCH_SIZE
            )
        )

        row_end = (
            int(
                math.ceil(
                    iy2 / PATCH_SIZE
                )
            )
            - 1
        )

        col_start = max(
            0,
            min(GRID - 1, col_start),
        )

        col_end = max(
            0,
            min(GRID - 1, col_end),
        )

        row_start = max(
            0,
            min(GRID - 1, row_start),
        )

        row_end = max(
            0,
            min(GRID - 1, row_end),
        )

        for row in range(
            row_start,
            row_end + 1,
        ):
            for col in range(
                col_start,
                col_end + 1,
            ):

                spatial_idx = (
                    row * GRID
                    + col
                )

                token_idx = (
                    tile_idx
                    * TOKENS_PER_TILE
                    + 1
                    + spatial_idx
                )

                result.append(
                    token_idx
                )

    return sorted(
        set(result)
    )


def get_content_tokens(
    width,
    height,
):
    result = []

    for tile_idx in ACTIVE_TILES:

        tile_x0 = (
            tile_idx
            * TILE_SIZE
        )

        for row in range(GRID):
            for col in range(GRID):

                x1 = (
                    tile_x0
                    + col * PATCH_SIZE
                )

                y1 = (
                    row * PATCH_SIZE
                )

                x2 = x1 + PATCH_SIZE
                y2 = y1 + PATCH_SIZE

                if (
                    x1 < width
                    and y1 < height
                    and x2 > 0
                    and y2 > 0
                ):
                    spatial_idx = (
                        row * GRID
                        + col
                    )

                    result.append(
                        tile_idx
                        * TOKENS_PER_TILE
                        + 1
                        + spatial_idx
                    )

    return sorted(
        set(result)
    )


def prepare_inputs(
    processor,
    image,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    formatted = (
        processor
        .apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    )

    return processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )


def decode_tokens(
    processor,
    input_ids,
):
    result = []

    for token_id in (
        input_ids[0]
        .tolist()
    ):
        result.append(
            processor.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    return result


def find_token(
    tokens,
    text,
):
    for idx, token in enumerate(tokens):
        if token == text:
            return idx

    return None


def query_groups(tokens):

    many = find_token(
        tokens,
        " many",
    )

    red = find_token(
        tokens,
        " red",
    )

    circles = find_token(
        tokens,
        " circles",
    )

    are = find_token(
        tokens,
        " are",
    )

    visible = find_token(
        tokens,
        " visible",
    )

    groups = {}

    if many is not None:
        groups["q_many"] = [many]

    if red is not None:
        groups["q_red"] = [red]

    if circles is not None:
        groups["q_circles"] = [
            circles
        ]

    if visible is not None:
        groups["q_visible"] = [
            visible
        ]

    phrase = [
        x
        for x in [
            many,
            red,
            circles,
            are,
            visible,
        ]
        if x is not None
    ]

    groups["q_phrase"] = phrase

    return groups


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata()

    pairs = pd.read_csv(
        PAIR_PATH
    )

    # ---------------------------------------------------------
    # Build unique sample set
    # ---------------------------------------------------------

    sample_role = {}

    for _, row in pairs.iterrows():

        sample_role[
            row["wrong_sample_id"]
        ] = row["wrong_label"]

        sample_role[
            row["correct_sample_id"]
        ] = "correct"

    sample_ids = sorted(
        sample_role.keys()
    )

    print("=" * 80)
    print("AROMA Matched Attention Runner")
    print("=" * 80)

    print(
        "\nClean pairs:",
        len(pairs),
    )

    print(
        "Unique samples:",
        len(sample_ids),
    )

    for sample_id in sample_ids:
        print(
            f"{sample_id:30s} "
            f"{sample_role[sample_id]}"
        )

    print("\nLoading processor...")

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    print("Loading model...")

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

    for sample_id in tqdm(
        sample_ids,
        desc="Matched attention",
    ):

        sample = metadata[
            sample_id
        ]

        role = sample_role[
            sample_id
        ]

        image = (
            Image.open(
                sample["image_path"]
            )
            .convert("RGB")
        )

        width, height = image.size

        content_tokens = (
            get_content_tokens(
                width,
                height,
            )
        )

        assert len(
            content_tokens
        ) == 2035

        object_token_sets = []

        for obj in sample[
            "target_objects"
        ]:

            bbox = denorm_bbox(
                obj["bbox_norm"],
                width,
                height,
            )

            object_token_sets.append(
                bbox_to_tokens(
                    bbox,
                    width,
                    height,
                )
            )

        inputs = prepare_inputs(
            processor,
            image,
        )

        tokens = decode_tokens(
            processor,
            inputs["input_ids"],
        )

        groups = query_groups(
            tokens
        )

        device_inputs = {
            key: (
                value.to(model.device)
                if torch.is_tensor(value)
                else value
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(
                **device_inputs,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

        for layer in CROSS_LAYERS:

            attn = (
                outputs
                .attentions[layer][0]
                .float()
                .cpu()
                .numpy()
            )

            for (
                query_group,
                q_positions,
            ) in groups.items():

                q_attn = (
                    attn[
                        :,
                        q_positions,
                        :
                    ]
                    .mean(axis=1)
                )

                for head in range(
                    q_attn.shape[0]
                ):

                    vec = q_attn[
                        head
                    ]

                    baseline = float(
                        np.mean(
                            vec[
                                content_tokens
                            ]
                        )
                    )

                    baseline = max(
                        baseline,
                        1e-12,
                    )

                    enrichments = []

                    for object_tokens in (
                        object_token_sets
                    ):

                        e = float(
                            np.mean(
                                vec[
                                    object_tokens
                                ]
                            )
                            / baseline
                        )

                        enrichments.append(
                            e
                        )

                    mean_e = float(
                        np.mean(
                            enrichments
                        )
                    )

                    std_e = float(
                        np.std(
                            enrichments
                        )
                    )

                    cv = (
                        std_e / mean_e
                        if mean_e > 0
                        else np.nan
                    )

                    rows.append(
                        {
                            "sample_id":
                                sample_id,

                            "role":
                                role,

                            "ground_truth":
                                sample[
                                    "ground_truth"
                                ],

                            "condition":
                                sample[
                                    "condition"
                                ],

                            "query_group":
                                query_group,

                            "layer":
                                layer,

                            "head":
                                head,

                            "mean_enrichment":
                                mean_e,

                            "cv":
                                cv,

                            "min_enrichment":
                                float(
                                    np.min(
                                        enrichments
                                    )
                                ),

                            "max_enrichment":
                                float(
                                    np.max(
                                        enrichments
                                    )
                                ),
                        }
                    )

        del outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(
        rows
    )

    output_path = (
        OUTPUT_DIR
        / "matched_head_metrics.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)

    print("\nMATCHED EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()
