import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


METADATA = Path("data/calibration50/metadata.jsonl")
OUTPUT_DIR = Path("outputs/calibration50/patch_masks")

TILE_SIZE = 560
PATCH_SIZE = 14
GRID = 40
TOKENS_PER_TILE = 1601
MAX_TILES = 4

# For Calibration-50 preprocessing:
NUM_TILE_ROWS = 1
NUM_TILE_COLS = 2
ACTIVE_TILES = [0, 1]


def denorm_bbox(bbox_norm, width, height):
    x1, y1, x2, y2 = bbox_norm

    return (
        x1 * width,
        y1 * height,
        x2 * width,
        y2 * height,
    )


def bbox_to_patch_indices(
    bbox,
    image_width,
    image_height,
):
    """
    Map original-image bbox to flattened visual token indices.

    Current Calibration-50 geometry:
      original = 768x512
      resized  = 768x512
      padded   = 1120x560
      tiling   = 1x2
      each tile = 560x560
      each tile = 40x40 spatial patches + 1 special token

    Returns only spatial patch tokens, never special tokens.
    """

    x1, y1, x2, y2 = bbox

    # Clip to actual image.
    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))

    token_indices = []

    for tile_col in range(NUM_TILE_COLS):
        tile_idx = tile_col

        tile_x0 = tile_col * TILE_SIZE
        tile_x1 = tile_x0 + TILE_SIZE

        tile_y0 = 0
        tile_y1 = TILE_SIZE

        # Intersection between bbox and tile.
        ix1 = max(x1, tile_x0)
        iy1 = max(y1, tile_y0)
        ix2 = min(x2, tile_x1)
        iy2 = min(y2, tile_y1)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        # Tile-local coordinates.
        lx1 = ix1 - tile_x0
        ly1 = iy1 - tile_y0
        lx2 = ix2 - tile_x0
        ly2 = iy2 - tile_y0

        # Any patch whose area overlaps the bbox.
        col_start = int(math.floor(lx1 / PATCH_SIZE))
        row_start = int(math.floor(ly1 / PATCH_SIZE))

        col_end = int(math.ceil(lx2 / PATCH_SIZE)) - 1
        row_end = int(math.ceil(ly2 / PATCH_SIZE)) - 1

        col_start = max(0, min(GRID - 1, col_start))
        col_end = max(0, min(GRID - 1, col_end))
        row_start = max(0, min(GRID - 1, row_start))
        row_end = max(0, min(GRID - 1, row_end))

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                spatial_idx = row * GRID + col

                # Assumption to validate:
                # first token in each tile is special token,
                # followed by 1600 row-major spatial patches.
                visual_token_idx = (
                    tile_idx * TOKENS_PER_TILE
                    + 1
                    + spatial_idx
                )

                token_indices.append(
                    {
                        "tile": tile_idx,
                        "row": row,
                        "col": col,
                        "spatial_idx": spatial_idx,
                        "visual_token_idx": visual_token_idx,
                    }
                )

    return token_indices


def draw_patch_overlay(
    image,
    objects,
    output_path,
):
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    for obj in objects:
        bbox = denorm_bbox(
            obj["bbox_norm"],
            width,
            height,
        )

        x1, y1, x2, y2 = bbox

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="black",
            width=3,
        )

        patches = bbox_to_patch_indices(
            bbox,
            width,
            height,
        )

        for p in patches:
            tile = p["tile"]
            row = p["row"]
            col = p["col"]

            px1 = tile * TILE_SIZE + col * PATCH_SIZE
            py1 = row * PATCH_SIZE

            px2 = px1 + PATCH_SIZE
            py2 = py1 + PATCH_SIZE

            # Only draw portion that lies inside original image.
            if px1 >= width or py1 >= height:
                continue

            draw.rectangle(
                [
                    px1,
                    py1,
                    min(px2, width - 1),
                    min(py2, height - 1),
                ],
                outline="white",
                width=1,
            )

    image.save(output_path)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = [
        json.loads(line)
        for line in METADATA.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    selected = {
        "cal50_n01_row",
        "cal50_n05_row",
        "cal50_n10_row",
        "cal50_n05_random_sparse",
        "cal50_n10_distractors",
    }

    audit_rows = []

    for record in records:
        if record["sample_id"] not in selected:
            continue

        image = Image.open(
            record["image_path"]
        ).convert("RGB")

        width, height = image.size

        object_entries = []

        for i, obj in enumerate(
            record["target_objects"]
        ):
            bbox = denorm_bbox(
                obj["bbox_norm"],
                width,
                height,
            )

            patches = bbox_to_patch_indices(
                bbox,
                width,
                height,
            )

            object_entries.append(
                {
                    "object_id": i,
                    "bbox": bbox,
                    "num_patches": len(patches),
                    "patches": patches,
                }
            )

        audit_rows.append(
            {
                "sample_id": record["sample_id"],
                "image_size": [width, height],
                "objects": object_entries,
            }
        )

        output_image = (
            OUTPUT_DIR
            / f"{record['sample_id']}_patch_overlay.png"
        )

        draw_patch_overlay(
            image,
            record["target_objects"],
            output_image,
        )

        print(
            f"{record['sample_id']:30s} "
            f"GT={record['ground_truth']:2d} "
            f"saved={output_image}"
        )

        for obj in object_entries:
            tokens = [
                p["visual_token_idx"]
                for p in obj["patches"]
            ]

            print(
                f"  object {obj['object_id']:2d}: "
                f"patches={obj['num_patches']:2d} "
                f"token_range="
                f"{min(tokens) if tokens else None}"
                f".."
                f"{max(tokens) if tokens else None}"
            )

    audit_path = (
        OUTPUT_DIR
        / "patch_mapping_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            audit_rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved audit:", audit_path)
    print("PATCH MASK BUILD COMPLETE")


if __name__ == "__main__":
    main()
