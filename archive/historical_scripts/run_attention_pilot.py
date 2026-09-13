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

META_PATH = Path("data/calibration50/metadata.jsonl")
LABEL_PATH = Path("outputs/calibration50/behavioral_labels.csv")
OUTPUT_DIR = Path("outputs/calibration50/attention_pilot")

TILE_SIZE = 560
PATCH_SIZE = 14
GRID_SIZE = 40
TOKENS_PER_TILE = 1601
ACTIVE_TILES = [0, 1]

CROSS_LAYERS = [3, 8, 13, 18, 23, 28, 33, 38]

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


def load_metadata():
    records = {}

    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            r = json.loads(line)
            records[r["sample_id"]] = r

    return records


def denorm_bbox(bbox_norm, width, height):
    x1, y1, x2, y2 = bbox_norm

    return (
        x1 * width,
        y1 * height,
        x2 * width,
        y2 * height,
    )


def bbox_to_patch_token_indices(
    bbox,
    image_width,
    image_height,
):
    x1, y1, x2, y2 = bbox

    x1 = max(0.0, min(float(image_width), float(x1)))
    x2 = max(0.0, min(float(image_width), float(x2)))
    y1 = max(0.0, min(float(image_height), float(y1)))
    y2 = max(0.0, min(float(image_height), float(y2)))

    indices = []

    for tile_idx in ACTIVE_TILES:
        tile_x0 = tile_idx * TILE_SIZE
        tile_x1 = tile_x0 + TILE_SIZE

        ix1 = max(x1, tile_x0)
        iy1 = max(y1, 0)
        ix2 = min(x2, tile_x1)
        iy2 = min(y2, TILE_SIZE)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        lx1 = ix1 - tile_x0
        ly1 = iy1
        lx2 = ix2 - tile_x0
        ly2 = iy2

        col_start = int(math.floor(lx1 / PATCH_SIZE))
        row_start = int(math.floor(ly1 / PATCH_SIZE))
        col_end = int(math.ceil(lx2 / PATCH_SIZE)) - 1
        row_end = int(math.ceil(ly2 / PATCH_SIZE)) - 1

        col_start = max(0, min(GRID_SIZE - 1, col_start))
        col_end = max(0, min(GRID_SIZE - 1, col_end))
        row_start = max(0, min(GRID_SIZE - 1, row_start))
        row_end = max(0, min(GRID_SIZE - 1, row_end))

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                spatial_idx = row * GRID_SIZE + col

                visual_idx = (
                    tile_idx * TOKENS_PER_TILE
                    + 1
                    + spatial_idx
                )

                indices.append(visual_idx)

    return sorted(set(indices))


def get_content_indices(width, height):
    indices = []

    for tile_idx in ACTIVE_TILES:
        tile_x0 = tile_idx * TILE_SIZE

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                px1 = tile_x0 + col * PATCH_SIZE
                py1 = row * PATCH_SIZE
                px2 = px1 + PATCH_SIZE
                py2 = py1 + PATCH_SIZE

                if (
                    px1 < width
                    and py1 < height
                    and px2 > 0
                    and py2 > 0
                ):
                    spatial_idx = row * GRID_SIZE + col

                    visual_idx = (
                        tile_idx * TOKENS_PER_TILE
                        + 1
                        + spatial_idx
                    )

                    indices.append(visual_idx)

    return sorted(set(indices))


def prepare_inputs(processor, image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    formatted = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )

    return inputs


def decode_tokens(processor, input_ids):
    ids = input_ids[0].tolist()

    return [
        processor.tokenizer.decode(
            [tid],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for tid in ids
    ]


def find_idx(tokens, token_text):
    for i, tok in enumerate(tokens):
        if tok == token_text:
            return i

    return None


def get_query_groups(tokens):
    idx_many = find_idx(tokens, " many")
    idx_red = find_idx(tokens, " red")
    idx_circles = find_idx(tokens, " circles")
    idx_are = find_idx(tokens, " are")
    idx_visible = find_idx(tokens, " visible")

    groups = {}

    if idx_many is not None:
        groups["q_many"] = [idx_many]

    if idx_red is not None:
        groups["q_red"] = [idx_red]

    if idx_circles is not None:
        groups["q_circles"] = [idx_circles]

    if idx_visible is not None:
        groups["q_visible"] = [idx_visible]

    phrase = [
        x
        for x in [
            idx_many,
            idx_red,
            idx_circles,
            idx_are,
            idx_visible,
        ]
        if x is not None
    ]

    if phrase:
        groups["q_phrase"] = phrase

    return groups


def choose_pilot_samples(labels):
    """
    Select:
      5 correct
      all stable_wrong up to 5
      5 unstable_wrong

    Prefer higher-cardinality correct samples so comparison
    is not dominated by trivial count=1/2 cases.
    """

    selected_parts = []

    correct = labels[
        labels["behavioral_label"] == "correct"
    ].copy()

    correct = correct.sort_values(
        ["ground_truth", "sample_id"],
        ascending=[False, True],
    ).head(5)

    stable = labels[
        labels["behavioral_label"] == "stable_wrong"
    ].copy()

    stable = stable.sort_values(
        ["ground_truth", "sample_id"],
        ascending=[False, True],
    ).head(5)

    unstable = labels[
        labels["behavioral_label"] == "unstable_wrong"
    ].copy()

    unstable = unstable.sort_values(
        ["ground_truth", "sample_id"],
        ascending=[False, True],
    ).head(5)

    selected_parts.extend(
        [correct, stable, unstable]
    )

    return pd.concat(
        selected_parts,
        ignore_index=True,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata()

    labels = pd.read_csv(
        LABEL_PATH
    )

    selected = choose_pilot_samples(
        labels
    )

    print("=" * 80)
    print("AROMA Attention Pilot")
    print("=" * 80)

    print("\nSelected samples:")

    print(
        selected[
            [
                "sample_id",
                "ground_truth",
                "condition",
                "behavioral_label",
            ]
        ].to_string(index=False)
    )

    selected.to_csv(
        OUTPUT_DIR / "selected_samples.csv",
        index=False,
    )

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    print("Loading model...")

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    model.eval()

    rows = []

    for _, label_row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc="Attention pilot",
    ):
        sample_id = label_row["sample_id"]
        label = label_row["behavioral_label"]

        sample = metadata[sample_id]

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        width, height = image.size

        content_indices = get_content_indices(
            width,
            height,
        )

        object_sets = []

        for obj in sample["target_objects"]:
            bbox = denorm_bbox(
                obj["bbox_norm"],
                width,
                height,
            )

            object_sets.append(
                bbox_to_patch_token_indices(
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

        query_groups = get_query_groups(
            tokens
        )

        device_inputs = {
            key: (
                value.to(model.device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(
                **device_inputs,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

        for layer_idx in CROSS_LAYERS:
            attn = (
                outputs.attentions[layer_idx][0]
                .float()
                .cpu()
                .numpy()
            )

            for query_group, q_positions in query_groups.items():
                attn_q = (
                    attn[:, q_positions, :]
                    .mean(axis=1)
                )

                for head_idx in range(
                    attn_q.shape[0]
                ):
                    vec = attn_q[
                        head_idx
                    ]

                    content_vals = vec[
                        content_indices
                    ]

                    content_mean = float(
                        np.mean(
                            content_vals
                        )
                    )

                    eps = 1e-12

                    if content_mean <= 0:
                        content_mean = eps

                    enrichments = []

                    for object_id, token_ids in enumerate(
                        object_sets
                    ):
                        vals = vec[
                            token_ids
                        ]

                        density = float(
                            np.mean(vals)
                        )

                        enrichment = (
                            density
                            / content_mean
                        )

                        enrichments.append(
                            enrichment
                        )

                        rows.append(
                            {
                                "sample_id": sample_id,
                                "ground_truth": sample["ground_truth"],
                                "condition": sample["condition"],
                                "behavioral_label": label,
                                "query_group": query_group,
                                "layer": layer_idx,
                                "head": head_idx,
                                "object_id": object_id,
                                "enrichment": enrichment,
                            }
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

                    min_e = float(
                        np.min(
                            enrichments
                        )
                    )

                    max_e = float(
                        np.max(
                            enrichments
                        )
                    )

                    # per-head summary row
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "ground_truth": sample["ground_truth"],
                            "condition": sample["condition"],
                            "behavioral_label": label,
                            "query_group": query_group,
                            "layer": layer_idx,
                            "head": head_idx,
                            "object_id": -1,
                            "enrichment": np.nan,
                            "head_mean_enrichment": mean_e,
                            "head_cv": cv,
                            "head_min_enrichment": min_e,
                            "head_max_enrichment": max_e,
                        }
                    )

        del outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(
        rows
    )

    full_path = (
        OUTPUT_DIR
        / "pilot_attention_raw.csv"
    )

    df.to_csv(
        full_path,
        index=False,
    )

    head_df = df[
        df["object_id"] == -1
    ].copy()

    head_path = (
        OUTPUT_DIR
        / "pilot_head_metrics.csv"
    )

    head_df.to_csv(
        head_path,
        index=False,
    )

    print("\nSaved:")
    print(full_path)
    print(head_path)

    print("\nPilot complete.")


if __name__ == "__main__":
    main()
