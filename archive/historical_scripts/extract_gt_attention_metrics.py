import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

METADATA_PATH = Path(
    "data/calibration50/metadata.jsonl"
)

OUTPUT_DIR = Path(
    "outputs/calibration50/attention_metrics"
)


# ---------------------------------------------------------------------
# Verified Mllama geometry for Calibration-50
# ---------------------------------------------------------------------

TILE_SIZE = 560
PATCH_SIZE = 14
GRID_SIZE = 40

# Per active/padded tile:
#   1 special token + 40*40 spatial patch tokens
TOKENS_PER_TILE = 1601

MAX_IMAGE_TILES = 4

# For the current 768x512 Calibration-50 images,
# verified by processor audit:
#
#   aspect ratio = (1, 2)
#   active tiles = [0, 1]
#
ACTIVE_TILES = [0, 1]


# ---------------------------------------------------------------------
# Verified cross-attention layers
# ---------------------------------------------------------------------

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


# =====================================================================
# Metadata
# =====================================================================

def load_metadata():
    records = []

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def get_sample_by_id(sample_id):
    records = load_metadata()

    for record in records:
        if record["sample_id"] == sample_id:
            return record

    raise ValueError(
        f"Sample not found: {sample_id}"
    )


# =====================================================================
# Geometry
# =====================================================================

def denorm_bbox(
    bbox_norm,
    image_width,
    image_height,
):
    x1, y1, x2, y2 = bbox_norm

    return (
        x1 * image_width,
        y1 * image_height,
        x2 * image_width,
        y2 * image_height,
    )


def bbox_to_patch_token_indices(
    bbox,
    image_width,
    image_height,
):
    """
    Map an original-image bbox to Mllama visual token indices.

    Verified Calibration-50 geometry:

        original image = 768 x 512
        resized image  = 768 x 512
        padded canvas  = 1120 x 560
        tiling         = 1 row x 2 active columns
        active tiles   = [0, 1]
        tile size      = 560 x 560
        patch size     = 14 x 14
        patch grid     = 40 x 40

    Token layout assumption:

        tile_offset
        + 1 special token
        + row-major 40x40 spatial tokens

    The function returns only spatial-token indices overlapping
    the target bbox.
    """

    x1, y1, x2, y2 = bbox

    # Clip bbox to actual image.
    x1 = max(
        0.0,
        min(float(image_width), float(x1)),
    )

    x2 = max(
        0.0,
        min(float(image_width), float(x2)),
    )

    y1 = max(
        0.0,
        min(float(image_height), float(y1)),
    )

    y2 = max(
        0.0,
        min(float(image_height), float(y2)),
    )

    token_indices = []

    for tile_idx in ACTIVE_TILES:

        tile_x0 = (
            tile_idx * TILE_SIZE
        )

        tile_x1 = (
            tile_x0 + TILE_SIZE
        )

        tile_y0 = 0
        tile_y1 = TILE_SIZE

        # Intersection between bbox and this tile.
        ix1 = max(
            x1,
            tile_x0,
        )

        iy1 = max(
            y1,
            tile_y0,
        )

        ix2 = min(
            x2,
            tile_x1,
        )

        iy2 = min(
            y2,
            tile_y1,
        )

        if (
            ix2 <= ix1
            or iy2 <= iy1
        ):
            continue

        # Tile-local coordinates.
        lx1 = ix1 - tile_x0
        ly1 = iy1 - tile_y0
        lx2 = ix2 - tile_x0
        ly2 = iy2 - tile_y0

        # Include every patch whose rectangle overlaps the bbox.
        col_start = int(
            math.floor(
                lx1 / PATCH_SIZE
            )
        )

        row_start = int(
            math.floor(
                ly1 / PATCH_SIZE
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

        row_end = (
            int(
                math.ceil(
                    ly2 / PATCH_SIZE
                )
            )
            - 1
        )

        col_start = max(
            0,
            min(
                GRID_SIZE - 1,
                col_start,
            ),
        )

        col_end = max(
            0,
            min(
                GRID_SIZE - 1,
                col_end,
            ),
        )

        row_start = max(
            0,
            min(
                GRID_SIZE - 1,
                row_start,
            ),
        )

        row_end = max(
            0,
            min(
                GRID_SIZE - 1,
                row_end,
            ),
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
                    row * GRID_SIZE
                    + col
                )

                visual_token_idx = (
                    tile_idx
                    * TOKENS_PER_TILE
                    + 1
                    + spatial_idx
                )

                token_indices.append(
                    visual_token_idx
                )

    return sorted(
        set(token_indices)
    )


def get_content_visual_token_indices(
    image_width,
    image_height,
):
    """
    Return visual spatial-token indices whose patch rectangles overlap
    the *actual image content*.

    This explicitly excludes:

      1. padded tile slots 2 and 3
      2. right-side padding inside active tile 1
      3. bottom padding inside active tiles
      4. per-tile special tokens

    For Calibration-50:

        image = 768 x 512
        canvas = 1120 x 560

    Expected count:

        ceil(560 / 14) * ceil(512 / 14)
        +
        ceil((768 - 560) / 14) * ceil(512 / 14)

        = 40 * 37 + 15 * 37
        = 2035
    """

    indices = []

    for tile_idx in ACTIVE_TILES:

        tile_x0 = (
            tile_idx * TILE_SIZE
        )

        tile_y0 = 0

        for row in range(
            GRID_SIZE
        ):
            for col in range(
                GRID_SIZE
            ):

                patch_x1 = (
                    tile_x0
                    + col * PATCH_SIZE
                )

                patch_y1 = (
                    tile_y0
                    + row * PATCH_SIZE
                )

                patch_x2 = (
                    patch_x1
                    + PATCH_SIZE
                )

                patch_y2 = (
                    patch_y1
                    + PATCH_SIZE
                )

                # Keep patch if it overlaps actual image content.
                overlaps_actual_image = (
                    patch_x1 < image_width
                    and patch_y1 < image_height
                    and patch_x2 > 0
                    and patch_y2 > 0
                )

                if not overlaps_actual_image:
                    continue

                spatial_idx = (
                    row * GRID_SIZE
                    + col
                )

                visual_token_idx = (
                    tile_idx
                    * TOKENS_PER_TILE
                    + 1
                    + spatial_idx
                )

                indices.append(
                    visual_token_idx
                )

    return sorted(
        set(indices)
    )


def get_all_active_spatial_indices():
    """
    All spatial tokens in active tiles, INCLUDING canvas padding.

    Useful only for diagnostic comparison.
    """

    indices = []

    for tile_idx in ACTIVE_TILES:

        for spatial_idx in range(
            GRID_SIZE * GRID_SIZE
        ):
            visual_token_idx = (
                tile_idx
                * TOKENS_PER_TILE
                + 1
                + spatial_idx
            )

            indices.append(
                visual_token_idx
            )

    return indices


# =====================================================================
# Prompt / token handling
# =====================================================================

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
        processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    )

    inputs = processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )

    return (
        formatted,
        inputs,
    )


def decode_tokens(
    processor,
    input_ids,
):
    ids = (
        input_ids[0]
        .tolist()
    )

    tokens = []

    for token_id in ids:

        token = (
            processor
            .tokenizer
            .decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

        tokens.append(
            token
        )

    return (
        ids,
        tokens,
    )


def find_token_index(
    tokens,
    target,
):
    matches = [
        i
        for i, token
        in enumerate(tokens)
        if token == target
    ]

    if not matches:
        return None

    return matches[0]


def select_query_groups(
    tokens,
):
    """
    Derive query groups from decoded token text instead of relying
    entirely on fixed positions.

    Expected current prompt:

        'How'
        ' many'
        ' red'
        ' circles'
        ' are'
        ' visible'
    """

    idx_many = find_token_index(
        tokens,
        " many",
    )

    idx_red = find_token_index(
        tokens,
        " red",
    )

    idx_circles = find_token_index(
        tokens,
        " circles",
    )

    idx_are = find_token_index(
        tokens,
        " are",
    )

    idx_visible = find_token_index(
        tokens,
        " visible",
    )

    groups = {}

    if idx_many is not None:
        groups[
            "q_many"
        ] = [
            idx_many
        ]

    if idx_red is not None:
        groups[
            "q_red"
        ] = [
            idx_red
        ]

    if idx_circles is not None:
        groups[
            "q_circles"
        ] = [
            idx_circles
        ]

    if idx_visible is not None:
        groups[
            "q_visible"
        ] = [
            idx_visible
        ]

    phrase_indices = [
        idx
        for idx in [
            idx_many,
            idx_red,
            idx_circles,
            idx_are,
            idx_visible,
        ]
        if idx is not None
    ]

    if phrase_indices:
        groups[
            "q_phrase"
        ] = phrase_indices

    return groups


# =====================================================================
# Attention metrics
# =====================================================================

def compute_metrics_for_attention_vector(
    attn_vec,
    object_token_sets,
    content_indices,
    all_active_indices,
):
    """
    Compute object-level attention metrics.

    attn_vec:
        shape [visual_token_length]

    content_indices:
        actual image-content patches only

    all_active_indices:
        all spatial patches in active tiles, including canvas padding

    Metrics:

      mass_raw
          raw attention mass inside object region

      mass_content_norm
          object attention mass normalized by total actual-image-content
          attention mass

      density
          mean attention value over object's spatial patches

      enrichment
          object density / mean density over actual image-content patches

      padding_fraction
          fraction of active-tile spatial attention falling on
          within-tile padding rather than actual image content
    """

    eps = 1e-12

    attn_vec = np.asarray(
        attn_vec,
        dtype=np.float64,
    )

    content_indices = np.asarray(
        content_indices,
        dtype=np.int64,
    )

    all_active_indices = np.asarray(
        all_active_indices,
        dtype=np.int64,
    )

    content_attn = (
        attn_vec[
            content_indices
        ]
    )

    active_attn = (
        attn_vec[
            all_active_indices
        ]
    )

    content_total = float(
        content_attn.sum()
    )

    active_total = float(
        active_attn.sum()
    )

    if content_total <= 0:
        content_total = eps

    if active_total <= 0:
        active_total = eps

    content_mean = float(
        content_attn.mean()
    )

    if content_mean <= 0:
        content_mean = eps

    padding_mass = max(
        0.0,
        active_total
        - content_total,
    )

    padding_fraction = (
        padding_mass
        / active_total
    )

    rows = []

    for (
        object_id,
        token_ids,
    ) in enumerate(
        object_token_sets
    ):

        token_ids = list(
            token_ids
        )

        if len(token_ids) == 0:

            raw_mass = 0.0
            density = 0.0
            enrichment = 0.0
            normalized_mass = 0.0

        else:

            values = (
                attn_vec[
                    token_ids
                ]
            )

            raw_mass = float(
                values.sum()
            )

            density = float(
                values.mean()
            )

            enrichment = float(
                density
                / content_mean
            )

            normalized_mass = float(
                raw_mass
                / content_total
            )

        rows.append(
            {
                "object_id":
                    object_id,

                "num_patches":
                    len(token_ids),

                "mass_raw":
                    raw_mass,

                "mass_content_norm":
                    normalized_mass,

                "density":
                    density,

                "enrichment":
                    enrichment,

                "padding_fraction":
                    padding_fraction,
            }
        )

    return rows


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-id",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "AROMA GT Attention Metric Extraction v2"
    )
    print("=" * 80)

    sample = get_sample_by_id(
        args.sample_id
    )

    print(
        "Sample ID   :",
        sample["sample_id"],
    )

    print(
        "GT count    :",
        sample["ground_truth"],
    )

    print(
        "Condition   :",
        sample["condition"],
    )

    print(
        "Image path  :",
        sample["image_path"],
    )

    image = (
        Image
        .open(
            sample["image_path"]
        )
        .convert("RGB")
    )

    width, height = image.size

    print(
        "Image size  :",
        (width, height),
    )

    # -------------------------------------------------------------
    # Spatial token geometry
    # -------------------------------------------------------------

    content_indices = (
        get_content_visual_token_indices(
            image_width=width,
            image_height=height,
        )
    )

    all_active_indices = (
        get_all_active_spatial_indices()
    )

    print(
        "\nActual-content spatial tokens:",
        len(content_indices),
    )

    print(
        "All active-tile spatial tokens:",
        len(all_active_indices),
    )

    print(
        "Within-active-tile padding tokens:",
        len(all_active_indices)
        - len(content_indices),
    )

    if (
        width == 768
        and height == 512
    ):
        assert (
            len(content_indices)
            == 2035
        ), (
            "Expected 2035 actual-content tokens "
            "for 768x512 Calibration-50 image, "
            f"got {len(content_indices)}"
        )

    # -------------------------------------------------------------
    # Object patch sets
    # -------------------------------------------------------------

    object_token_sets = []

    print(
        "\nGT object patch coverage:"
    )

    for object_id, obj in enumerate(
        sample["target_objects"]
    ):

        bbox = denorm_bbox(
            obj["bbox_norm"],
            width,
            height,
        )

        token_ids = (
            bbox_to_patch_token_indices(
                bbox,
                width,
                height,
            )
        )

        object_token_sets.append(
            token_ids
        )

        print(
            f"  object {object_id:02d}: "
            f"bbox="
            f"({bbox[0]:.1f}, "
            f"{bbox[1]:.1f}, "
            f"{bbox[2]:.1f}, "
            f"{bbox[3]:.1f}) "
            f"patches={len(token_ids)}"
        )

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
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

    # -------------------------------------------------------------
    # Prepare input
    # -------------------------------------------------------------

    formatted, inputs = (
        prepare_inputs(
            processor,
            image,
        )
    )

    ids, tokens = (
        decode_tokens(
            processor,
            inputs["input_ids"],
        )
    )

    print(
        "\nToken sequence:"
    )

    for i, token in enumerate(
        tokens
    ):
        print(
            f"{i:02d} "
            f"{repr(token)}"
        )

    query_groups = (
        select_query_groups(
            tokens
        )
    )

    print(
        "\nQuery groups:"
    )

    for (
        group_name,
        positions,
    ) in query_groups.items():

        token_text = [
            tokens[i]
            for i in positions
        ]

        print(
            f"  {group_name:12s} "
            f"positions={positions} "
            f"tokens={token_text}"
        )

    # -------------------------------------------------------------
    # Move tensors to device
    # -------------------------------------------------------------

    device_inputs = {}

    for key, value in inputs.items():

        if torch.is_tensor(
            value
        ):
            device_inputs[
                key
            ] = value.to(
                model.device
            )

        else:
            device_inputs[
                key
            ] = value

    # -------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------

    print(
        "\nRunning forward pass "
        "with output_attentions=True ..."
    )

    with torch.inference_mode():

        outputs = model(
            **device_inputs,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    # -------------------------------------------------------------
    # Validate attention shapes
    # -------------------------------------------------------------

    print(
        "\nCross-attention shapes:"
    )

    for layer_idx in CROSS_LAYERS:

        attn = (
            outputs.attentions[
                layer_idx
            ]
        )

        print(
            f"  layer {layer_idx:02d}: "
            f"{tuple(attn.shape)}"
        )

        assert (
            attn.shape[-1]
            == MAX_IMAGE_TILES
            * TOKENS_PER_TILE
        ), (
            f"Unexpected visual token length "
            f"at layer {layer_idx}: "
            f"{attn.shape[-1]}"
        )

    # -------------------------------------------------------------
    # Extract metrics
    # -------------------------------------------------------------

    print(
        "\nComputing object-level metrics ..."
    )

    rows = []

    for layer_idx in CROSS_LAYERS:

        attention_tensor = (
            outputs
            .attentions[
                layer_idx
            ][0]
            .float()
            .cpu()
            .numpy()
        )

        # [heads, query_len, visual_len]
        num_heads = (
            attention_tensor.shape[0]
        )

        for (
            query_group,
            query_positions,
        ) in query_groups.items():

            if not query_positions:
                continue

            # Average over selected query positions.
            #
            # -> [heads, visual_len]
            query_attention = (
                attention_tensor[
                    :,
                    query_positions,
                    :
                ]
                .mean(
                    axis=1
                )
            )

            for head_idx in range(
                num_heads
            ):

                attn_vec = (
                    query_attention[
                        head_idx
                    ]
                )

                object_rows = (
                    compute_metrics_for_attention_vector(
                        attn_vec=attn_vec,
                        object_token_sets=object_token_sets,
                        content_indices=content_indices,
                        all_active_indices=all_active_indices,
                    )
                )

                for row in object_rows:

                    row.update(
                        {
                            "sample_id":
                                sample[
                                    "sample_id"
                                ],

                            "ground_truth":
                                sample[
                                    "ground_truth"
                                ],

                            "condition":
                                sample[
                                    "condition"
                                ],

                            "layer":
                                layer_idx,

                            "head":
                                head_idx,

                            "query_group":
                                query_group,
                        }
                    )

                    rows.append(
                        row
                    )

    df = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------
    # Save full metric table
    # -------------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / (
            f"{sample['sample_id']}"
            "_attention_metrics_v2.csv"
        )
    )

    df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        "\nSaved full metrics:"
    )

    print(
        metrics_path
    )

    # -------------------------------------------------------------
    # Mean summary across heads + layers
    # -------------------------------------------------------------

    summary = (
        df
        .groupby(
            [
                "query_group",
                "object_id",
            ]
        )
        .agg(
            num_patches=(
                "num_patches",
                "first",
            ),

            mean_mass=(
                "mass_content_norm",
                "mean",
            ),

            mean_density=(
                "density",
                "mean",
            ),

            mean_enrichment=(
                "enrichment",
                "mean",
            ),

            median_enrichment=(
                "enrichment",
                "median",
            ),

            std_enrichment=(
                "enrichment",
                "std",
            ),

            mean_padding_fraction=(
                "padding_fraction",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        OUTPUT_DIR
        / (
            f"{sample['sample_id']}"
            "_attention_summary_v2.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nPer-object summary "
        "(mean over all cross layers / heads):"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nSaved summary:"
    )

    print(
        summary_path
    )

    # -------------------------------------------------------------
    # Layer-level summary
    # -------------------------------------------------------------

    layer_summary = (
        df
        .groupby(
            [
                "query_group",
                "layer",
                "object_id",
            ]
        )
        .agg(
            mean_enrichment=(
                "enrichment",
                "mean",
            ),

            mean_mass=(
                "mass_content_norm",
                "mean",
            ),
        )
        .reset_index()
    )

    layer_summary_path = (
        OUTPUT_DIR
        / (
            f"{sample['sample_id']}"
            "_attention_by_layer_v2.csv"
        )
    )

    layer_summary.to_csv(
        layer_summary_path,
        index=False,
    )

    print(
        "\nSaved layer summary:"
    )

    print(
        layer_summary_path
    )

    # -------------------------------------------------------------
    # Head ranking
    # -------------------------------------------------------------

    # For each layer/head/query group,
    # compute coefficient of variation across GT objects.
    #
    # Lower CV = more even attention across objects.
    # Higher CV = stronger object-wise imbalance.

    head_stats = []

    grouped = df.groupby(
        [
            "query_group",
            "layer",
            "head",
        ]
    )

    for (
        query_group,
        layer,
        head,
    ), group in grouped:

        values = (
            group[
                "enrichment"
            ]
            .to_numpy(
                dtype=float
            )
        )

        mean_val = float(
            np.mean(values)
        )

        std_val = float(
            np.std(values)
        )

        cv = (
            std_val
            / mean_val
            if mean_val > 0
            else np.nan
        )

        head_stats.append(
            {
                "query_group":
                    query_group,

                "layer":
                    layer,

                "head":
                    head,

                "mean_object_enrichment":
                    mean_val,

                "object_enrichment_std":
                    std_val,

                "object_enrichment_cv":
                    cv,

                "min_object_enrichment":
                    float(
                        np.min(values)
                    ),

                "max_object_enrichment":
                    float(
                        np.max(values)
                    ),
            }
        )

    head_df = pd.DataFrame(
        head_stats
    )

    head_path = (
        OUTPUT_DIR
        / (
            f"{sample['sample_id']}"
            "_head_stats_v2.csv"
        )
    )

    head_df.to_csv(
        head_path,
        index=False,
    )

    print(
        "\nSaved head statistics:"
    )

    print(
        head_path
    )

    # -------------------------------------------------------------
    # Print q_many head ranking
    # -------------------------------------------------------------

    q_many = (
        head_df[
            head_df[
                "query_group"
            ]
            == "q_many"
        ]
        .sort_values(
            "object_enrichment_cv",
            ascending=False,
        )
    )

    if not q_many.empty:

        print(
            "\nTop 15 most object-imbalanced "
            "q_many heads:"
        )

        print(
            q_many[
                [
                    "layer",
                    "head",
                    "mean_object_enrichment",
                    "object_enrichment_cv",
                    "min_object_enrichment",
                    "max_object_enrichment",
                ]
            ]
            .head(15)
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "EXTRACTION V2 COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
