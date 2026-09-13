import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


OUTPUT_DIR = Path("data/calibration50")
IMAGE_DIR = OUTPUT_DIR / "images"
METADATA_PATH = OUTPUT_DIR / "metadata.jsonl"

WIDTH = 768
HEIGHT = 512

COUNTS = list(range(1, 11))

CONDITIONS = [
    "row",
    "grid",
    "random_sparse",
    "dense",
    "distractors",
]

BASE_SEED = 20260910


def circle_box(cx, cy, r):
    return [
        int(cx - r),
        int(cy - r),
        int(cx + r),
        int(cy + r),
    ]


def normalize_box(box):
    x1, y1, x2, y2 = box

    return [
        round(x1 / WIDTH, 6),
        round(y1 / HEIGHT, 6),
        round(x2 / WIDTH, 6),
        round(y2 / HEIGHT, 6),
    ]


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def valid_against_existing(candidate, existing, max_iou=0.02):
    return all(box_iou(candidate, box) <= max_iou for box in existing)


def row_layout(n, radius):
    margin_x = radius + 35

    if n == 1:
        xs = [WIDTH // 2]
    else:
        usable = WIDTH - 2 * margin_x
        step = usable / (n - 1)
        xs = [int(margin_x + i * step) for i in range(n)]

    cy = 180

    return [(x, cy) for x in xs]


def grid_layout(n, radius):
    cols = min(5, n)
    rows = math.ceil(n / cols)

    x_margin = 90
    y_margin = 100

    if cols == 1:
        xs = [WIDTH // 2]
    else:
        xs = [
            int(x_margin + i * (WIDTH - 2 * x_margin) / (cols - 1))
            for i in range(cols)
        ]

    if rows == 1:
        ys = [HEIGHT // 2]
    else:
        ys = [
            int(y_margin + j * (HEIGHT - 2 * y_margin) / (rows - 1))
            for j in range(rows)
        ]

    centers = []

    for y in ys:
        for x in xs:
            centers.append((x, y))

            if len(centers) == n:
                return centers

    return centers


def random_nonoverlap_centers(rng, n, radius, dense=False):
    centers = []
    boxes = []

    if dense:
        xmin = 170
        xmax = WIDTH - 170
        ymin = 100
        ymax = HEIGHT - 100
        max_iou = 0.00
    else:
        xmin = radius + 30
        xmax = WIDTH - radius - 30
        ymin = radius + 30
        ymax = HEIGHT - radius - 30
        max_iou = 0.00

    attempts = 0

    while len(centers) < n:
        attempts += 1

        if attempts > 10000:
            raise RuntimeError(
                f"Could not place {n} circles without overlap."
            )

        cx = rng.randint(xmin, xmax)
        cy = rng.randint(ymin, ymax)

        candidate = circle_box(cx, cy, radius)

        if valid_against_existing(
            candidate,
            boxes,
            max_iou=max_iou,
        ):
            centers.append((cx, cy))
            boxes.append(candidate)

    return centers


def generate_scene(count, condition, sample_index):
    seed = BASE_SEED + sample_index
    rng = random.Random(seed)

    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        "white",
    )

    draw = ImageDraw.Draw(img)

    if condition == "dense":
        radius = 24
    else:
        radius = 30

    if condition == "row":
        centers = row_layout(count, radius)

    elif condition == "grid":
        centers = grid_layout(count, radius)

    elif condition == "random_sparse":
        centers = random_nonoverlap_centers(
            rng,
            count,
            radius,
            dense=False,
        )

    elif condition == "dense":
        centers = random_nonoverlap_centers(
            rng,
            count,
            radius,
            dense=True,
        )

    elif condition == "distractors":
        centers = grid_layout(count, radius)

    else:
        raise ValueError(condition)

    target_objects = []

    occupied_boxes = []

    for idx, (cx, cy) in enumerate(centers, start=1):
        box = circle_box(cx, cy, radius)

        draw.ellipse(
            box,
            fill="red",
            outline="black",
            width=3,
        )

        occupied_boxes.append(box)

        target_objects.append(
            {
                "id": f"T{idx}",
                "category": "red circle",
                "center_px": [cx, cy],
                "bbox_px": box,
                "bbox_norm": normalize_box(box),
            }
        )

    distractor_objects = []

    if condition == "distractors":
        n_distractors = min(6, max(3, count // 2 + 2))

        square_size = 50

        attempts = 0

        while len(distractor_objects) < n_distractors:
            attempts += 1

            if attempts > 5000:
                raise RuntimeError(
                    "Could not place distractors."
                )

            cx = rng.randint(50, WIDTH - 50)
            cy = rng.randint(50, HEIGHT - 50)

            half = square_size // 2

            box = [
                cx - half,
                cy - half,
                cx + half,
                cy + half,
            ]

            if not valid_against_existing(
                box,
                occupied_boxes,
                max_iou=0.00,
            ):
                continue

            draw.rectangle(
                box,
                fill="blue",
                outline="black",
                width=3,
            )

            occupied_boxes.append(box)

            distractor_objects.append(
                {
                    "id": f"D{len(distractor_objects) + 1}",
                    "category": "blue square",
                    "center_px": [cx, cy],
                    "bbox_px": box,
                    "bbox_norm": normalize_box(box),
                }
            )

    sample_id = f"cal50_n{count:02d}_{condition}"

    image_relpath = f"data/calibration50/images/{sample_id}.png"

    image_path = Path(image_relpath)

    img.save(image_path)

    metadata = {
        "sample_id": sample_id,
        "image_path": image_relpath,
        "question": "How many red circles are there in the image?",
        "target_category": "red circle",
        "ground_truth": count,
        "condition": condition,
        "seed": seed,
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "target_objects": target_objects,
        "distractor_objects": distractor_objects,
    }

    return metadata


def main():
    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    sample_index = 0

    for count in COUNTS:
        for condition in CONDITIONS:
            record = generate_scene(
                count=count,
                condition=condition,
                sample_index=sample_index,
            )

            records.append(record)
            sample_index += 1

    assert len(records) == 50

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(record)
                + "\n"
            )

    print("=" * 72)
    print("AROMA Calibration-50 Generated")
    print("=" * 72)

    print("Samples       :", len(records))
    print("Counts        :", COUNTS)
    print("Conditions    :", CONDITIONS)
    print("Image size    :", f"{WIDTH}x{HEIGHT}")
    print("Metadata      :", METADATA_PATH)
    print("Image dir     :", IMAGE_DIR)

    counts_found = {}

    for record in records:
        n = record["ground_truth"]
        counts_found[n] = counts_found.get(n, 0) + 1

    print("\nSamples per count:")

    for n in COUNTS:
        print(
            f"  count={n:2d}: "
            f"{counts_found[n]} samples"
        )


if __name__ == "__main__":
    main()
