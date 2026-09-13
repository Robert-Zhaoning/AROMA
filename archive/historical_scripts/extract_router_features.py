import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

PAIR_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "exact_matched_pairs_eager.csv"
)

PROFILE_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "dam_head_profiles/"
    "head_response_profiles.csv"
)

META_PATH = Path(
    "data/proc_count_causal_v1/"
    "metadata.jsonl"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "router"
)

OUT_PATH = OUT_DIR / "preintervention_features.csv"

DISCOVERY_SAMPLE = "pccv1_n05_row_r00"

FROZEN_HEADS = [
    (33, 1),
    (3, 4),
    (18, 13),
    (8, 30),
    (3, 11),
    (13, 11),
    (33, 21),
]

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def head_name(layer, head):
    return f"L{layer}H{head}"


def entropy_from_probs(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def coefficient_of_variation(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.nan
    m = x.mean()
    if abs(m) < 1e-12:
        return 0.0
    return float(x.std(ddof=0) / abs(m))


def build_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]


def find_cross_attention_tensor(outputs, layer):
    """
    Mllama returns 40 entries in outputs.attentions.
    Cross-attention layers are at model language layers:
    3,8,13,18,23,28,33,38.

    The returned tensor at those indices has visual-token dimension
    rather than normal text-token square attention.
    """
    attn = outputs.attentions[layer]

    if attn is None:
        raise RuntimeError(
            f"Attention tensor for layer {layer} is None"
        )

    if attn.ndim != 4:
        raise RuntimeError(
            f"Unexpected attention ndim at layer {layer}: "
            f"{attn.shape}"
        )

    return attn


def infer_active_visual_token_count(
    inputs,
    attn_tensor,
):
    """
    We rely on the actual attention tensor width.
    Padding / inactive visual slots may still exist, so later
    we restrict using cross_attention_mask when possible.
    """
    return int(attn_tensor.shape[-1])


def get_answer_query_index(input_ids):
    """
    For teacher-forced baseline forward, use the final text token
    before generation as the answer-query proxy.

    This is exactly the same basic answer-position logic used in the
    previous cross-attention experiments.
    """
    return int(input_ids.shape[1] - 1)


def numeral_distribution(
    logits,
    tokenizer,
):
    """
    Compute normalized probability over numeral tokens 1..10.
    """
    token_ids = {}
    for n in range(1, 11):
        ids = tokenizer.encode(
            str(n),
            add_special_tokens=False,
        )

        if len(ids) != 1:
            raise RuntimeError(
                f"Numeral {n} maps to {ids}, "
                "expected exactly one token."
            )

        token_ids[n] = ids[0]

    last_logits = logits[0, -1].float()

    numeral_logits = torch.tensor(
        [
            last_logits[token_ids[n]].item()
            for n in range(1, 11)
        ],
        dtype=torch.float64,
    )

    probs = torch.softmax(
        numeral_logits,
        dim=0,
    ).cpu().numpy()

    order = np.argsort(-probs)

    top1_idx = int(order[0])
    top2_idx = int(order[1])

    out = {
        "numeral_entropy": entropy_from_probs(probs),
        "numeral_top1": top1_idx + 1,
        "numeral_top1_prob": float(probs[top1_idx]),
        "numeral_top2": top2_idx + 1,
        "numeral_top2_prob": float(probs[top2_idx]),
        "numeral_margin": float(
            probs[top1_idx] - probs[top2_idx]
        ),
    }

    for n in range(1, 11):
        out[f"numeral_prob_{n}"] = float(
            probs[n - 1]
        )

    return out


def get_visual_geometry(
    processor,
    image,
):
    """
    Mllama preprocessing geometry for current 768x512 data.

    Previous probes established:
      tile size = 560
      active aspect ratio = 1 x 2
      image itself remains 768 x 512
      padded canvas = 1120 x 560

    We derive dynamically where possible.
    """
    image_processor = processor.image_processor

    arr = np.array(image)
    tensor = torch.from_numpy(
        arr
    ).permute(2, 0, 1).unsqueeze(0)

    resized, aspect_ratio = (
        image_processor.resize(
            tensor,
            size=image_processor.size,
            max_image_tiles=(
                image_processor.max_image_tiles
            ),
        )
    )

    rows = int(aspect_ratio[0])
    cols = int(aspect_ratio[1])

    tile_h = int(
        image_processor.size.height
    )
    tile_w = int(
        image_processor.size.width
    )

    return {
        "rows": rows,
        "cols": cols,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "canvas_h": rows * tile_h,
        "canvas_w": cols * tile_w,
        "resized_h": int(resized.shape[-2]),
        "resized_w": int(resized.shape[-1]),
        "orig_h": int(image.height),
        "orig_w": int(image.width),
    }


def infer_patch_grid_from_visual_length(
    visual_len,
    num_active_tiles,
):
    """
    Mllama visual token layout contains more than a plain ViT grid,
    so exact global indexing is architecture-specific.

    Earlier geometry experiments already established the object-to-
    visual token mapping. Here we reconstruct only enough for spatial
    region approximation.

    This function intentionally searches plausible square grids.
    """
    if num_active_tiles <= 0:
        return None

    per_tile = visual_len / num_active_tiles

    candidates = []

    for side in range(10, 100):
        base = side * side

        diff = abs(per_tile - base)

        candidates.append(
            (diff, side)
        )

    candidates.sort()

    return candidates[0][1]


def bbox_to_patch_indices(
    bbox_px,
    geom,
    visual_len,
):
    """
    Approximate original-image bbox -> visual token indices.

    IMPORTANT:
    This is an oracle diagnostic feature, not the final deployment
    representation.

    We map the original image into the processor canvas and collect
    patch-grid cells intersecting each bbox.
    """
    x1, y1, x2, y2 = bbox_px

    canvas_w = geom["canvas_w"]
    canvas_h = geom["canvas_h"]

    resized_w = geom["resized_w"]
    resized_h = geom["resized_h"]

    rows = geom["rows"]
    cols = geom["cols"]

    sx = resized_w / geom["orig_w"]
    sy = resized_h / geom["orig_h"]

    rx1 = x1 * sx
    ry1 = y1 * sy
    rx2 = x2 * sx
    ry2 = y2 * sy

    num_tiles = rows * cols

    patch_side = infer_patch_grid_from_visual_length(
        visual_len,
        num_tiles,
    )

    if patch_side is None:
        return []

    patch_w = geom["tile_w"] / patch_side
    patch_h = geom["tile_h"] / patch_side

    indices = []

    approximate_tokens_per_tile = (
        visual_len // num_tiles
    )

    for tile_row in range(rows):
        for tile_col in range(cols):
            tx1 = tile_col * geom["tile_w"]
            ty1 = tile_row * geom["tile_h"]
            tx2 = tx1 + geom["tile_w"]
            ty2 = ty1 + geom["tile_h"]

            ix1 = max(rx1, tx1)
            iy1 = max(ry1, ty1)
            ix2 = min(rx2, tx2)
            iy2 = min(ry2, ty2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            local_x1 = ix1 - tx1
            local_y1 = iy1 - ty1
            local_x2 = ix2 - tx1
            local_y2 = iy2 - ty1

            px1 = max(
                0,
                int(math.floor(
                    local_x1 / patch_w
                )),
            )

            py1 = max(
                0,
                int(math.floor(
                    local_y1 / patch_h
                )),
            )

            px2 = min(
                patch_side - 1,
                int(math.floor(
                    max(
                        local_x2 - 1e-6,
                        local_x1,
                    ) / patch_w
                )),
            )

            py2 = min(
                patch_side - 1,
                int(math.floor(
                    max(
                        local_y2 - 1e-6,
                        local_y1,
                    ) / patch_h
                )),
            )

            tile_id = (
                tile_row * cols
                + tile_col
            )

            tile_offset = (
                tile_id
                * approximate_tokens_per_tile
            )

            for py in range(
                py1,
                py2 + 1,
            ):
                for px in range(
                    px1,
                    px2 + 1,
                ):
                    local_idx = (
                        py * patch_side + px
                    )

                    idx = (
                        tile_offset
                        + local_idx
                    )

                    if 0 <= idx < visual_len:
                        indices.append(idx)

    return sorted(set(indices))


def attention_features_for_head(
    attn_tensor,
    head,
    query_idx,
    object_sets,
):
    """
    attn_tensor:
      [batch, heads, text_query, visual_tokens]
    """
    vec = (
        attn_tensor[
            0,
            head,
            query_idx,
            :
        ]
        .float()
        .detach()
        .cpu()
        .numpy()
    )

    vec = np.maximum(vec, 0)

    total = vec.sum()

    if total <= 0:
        p = np.ones_like(vec) / len(vec)
    else:
        p = vec / total

    global_entropy = entropy_from_probs(p)

    global_entropy_norm = (
        global_entropy
        / math.log(len(p))
        if len(p) > 1
        else 0.0
    )

    global_max = float(p.max())
    global_std = float(p.std())
    global_top1pct_mass = float(
        np.sort(p)[
            -max(
                1,
                int(round(
                    0.01 * len(p)
                )),
            ):
        ].sum()
    )

    object_masses = []

    object_densities = []

    union = set()

    for indices in object_sets:
        valid = [
            i
            for i in indices
            if 0 <= i < len(p)
        ]

        union.update(valid)

        if len(valid) == 0:
            mass = 0.0
            density = 0.0
        else:
            mass = float(
                p[valid].sum()
            )

            density = float(
                mass / len(valid)
            )

        object_masses.append(mass)
        object_densities.append(
            density
        )

    if len(object_masses) > 0:
        obj_arr = np.asarray(
            object_masses,
            dtype=np.float64,
        )

        obj_mean = float(
            obj_arr.mean()
        )
        obj_std = float(
            obj_arr.std()
        )
        obj_cv = coefficient_of_variation(
            obj_arr
        )
        obj_min = float(
            obj_arr.min()
        )
        obj_max = float(
            obj_arr.max()
        )
        obj_sum = float(
            obj_arr.sum()
        )
    else:
        obj_mean = 0.0
        obj_std = 0.0
        obj_cv = 0.0
        obj_min = 0.0
        obj_max = 0.0
        obj_sum = 0.0

    if len(union) > 0:
        object_union_mass = float(
            p[list(union)].sum()
        )

        object_union_density = (
            object_union_mass
            / len(union)
        )
    else:
        object_union_mass = 0.0
        object_union_density = 0.0

    background_indices = [
        i
        for i in range(len(p))
        if i not in union
    ]

    if len(background_indices) > 0:
        background_density = float(
            p[
                background_indices
            ].mean()
        )
    else:
        background_density = 0.0

    enrichment = (
        object_union_density
        / (
            background_density
            + 1e-12
        )
    )

    return {
        "attn_entropy": global_entropy,
        "attn_entropy_norm": global_entropy_norm,
        "attn_max": global_max,
        "attn_std": global_std,
        "attn_top1pct_mass": (
            global_top1pct_mass
        ),
        "object_mass_mean": obj_mean,
        "object_mass_std": obj_std,
        "object_mass_cv": obj_cv,
        "object_mass_min": obj_min,
        "object_mass_max": obj_max,
        "object_mass_sum": obj_sum,
        "object_union_mass": (
            object_union_mass
        ),
        "object_enrichment": float(
            enrichment
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--include-discovery",
        action="store_true",
        help=(
            "Include discovery sample. "
            "Default excludes it."
        ),
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "AROMA Adaptive DAM Router "
        "Pre-intervention Feature Extraction"
    )
    print("=" * 100)

    pairs = pd.read_csv(PAIR_PATH)

    metadata = {
        r["sample_id"]: r
        for r in load_jsonl(
            META_PATH
        )
    }

    baseline_records = {
        r["sample_id"]: r
        for r in load_jsonl(
            Path(
                "outputs/"
                "proc_count_causal_v1/"
                "baseline_results.jsonl"
            )
        )
    }

    sample_rows = []

    for _, pair in pairs.iterrows():
        pair_id = int(
            pair["pair_id"]
        )

        for role, col in [
            (
                "correct",
                "correct_sample_id",
            ),
            (
                "wrong",
                "wrong_sample_id",
            ),
        ]:
            sid = str(
                pair[col]
            )

            if (
                not args.include_discovery
                and sid
                == DISCOVERY_SAMPLE
            ):
                continue

            sample_rows.append(
                {
                    "pair_id": pair_id,
                    "role": role,
                    "sample_id": sid,
                }
            )

    samples = pd.DataFrame(
        sample_rows
    ).drop_duplicates(
        "sample_id"
    )

    if args.limit is not None:
        samples = samples.head(
            args.limit
        )

    print(
        f"Samples: {len(samples)}"
    )

    print(
        "Frozen heads:"
    )

    for layer, head in FROZEN_HEADS:
        print(
            f"  {head_name(layer, head)}"
        )

    print(
        "\nLoading processor..."
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
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

    for _, sample_row in tqdm(
        samples.iterrows(),
        total=len(samples),
        desc="Router features",
    ):
        sid = sample_row[
            "sample_id"
        ]

        meta = metadata[sid]

        image = Image.open(
            meta["image_path"]
        ).convert("RGB")

        messages = build_messages()

        formatted = (
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
        )

        inputs = processor(
            images=image,
            text=formatted,
            return_tensors="pt",
        )

        device = next(
            model.parameters()
        ).device

        inputs = {
            k: (
                v.to(device)
                if torch.is_tensor(v)
                else v
            )
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_attentions=True,
                use_cache=False,
            )

        features = {
            "pair_id": int(
                sample_row["pair_id"]
            ),
            "role": sample_row[
                "role"
            ],
            "sample_id": sid,
            "condition": meta[
                "condition"
            ],
            "replicate": int(
                meta["replicate"]
            ),
        }

        baseline = (
            baseline_records[
                sid
            ]
        )

        features[
            "baseline_prediction"
        ] = baseline[
            "prediction"
        ]

        features[
            "baseline_correct"
        ] = baseline[
            "correct"
        ]

        features.update(
            numeral_distribution(
                outputs.logits,
                processor.tokenizer,
            )
        )

        query_idx = (
            get_answer_query_index(
                inputs["input_ids"]
            )
        )

        geom = get_visual_geometry(
            processor,
            image,
        )

        first_layer = (
            FROZEN_HEADS[0][0]
        )

        first_attn = (
            find_cross_attention_tensor(
                outputs,
                first_layer,
            )
        )

        visual_len = (
            infer_active_visual_token_count(
                inputs,
                first_attn,
            )
        )

        object_sets = []

        for obj in meta[
            "target_objects"
        ]:
            indices = (
                bbox_to_patch_indices(
                    obj["bbox_px"],
                    geom,
                    visual_len,
                )
            )

            object_sets.append(
                indices
            )

        features[
            "oracle_num_objects"
        ] = len(
            object_sets
        )

        features[
            "oracle_mean_object_patch_count"
        ] = float(
            np.mean(
                [
                    len(x)
                    for x in object_sets
                ]
            )
        )

        for layer, head in FROZEN_HEADS:
            hname = head_name(
                layer,
                head,
            )

            attn = (
                find_cross_attention_tensor(
                    outputs,
                    layer,
                )
            )

            hfeatures = (
                attention_features_for_head(
                    attn,
                    head,
                    query_idx,
                    object_sets,
                )
            )

            for k, v in (
                hfeatures.items()
            ):
                features[
                    f"{hname}_{k}"
                ] = v

        rows.append(
            features
        )

        del outputs
        torch.cuda.empty_cache()

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUT_PATH,
        index=False,
    )

    print()
    print("=" * 100)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 100)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Saved: {OUT_PATH}"
    )

    print(
        "\nBaseline role counts:"
    )

    print(
        df["role"]
        .value_counts()
        .to_string()
    )

    print(
        "\nBaseline prediction accuracy:"
    )

    print(
        df[
            "baseline_correct"
        ].mean()
    )

    print(
        "\nNumeral feature preview:"
    )

    preview_cols = [
        "sample_id",
        "role",
        "baseline_prediction",
        "numeral_top1",
        "numeral_top1_prob",
        "numeral_top2",
        "numeral_top2_prob",
        "numeral_margin",
        "numeral_entropy",
    ]

    print(
        df[
            preview_cols
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
