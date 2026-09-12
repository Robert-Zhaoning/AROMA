import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


WIDTH = 768
HEIGHT = 512

TARGET_COLOR = (255, 0, 0)
DISTRACTOR_COLOR = (0, 90, 255)
BACKGROUND = (255, 255, 255)

TARGET_RADIUS = 24
DISTRACTOR_HALF = 22

BASE_SEED = 90260920

CONDITIONS = [
    "row",
    "grid",
    "random_sparse",
    "dense",
    "distractors",
]

REPLICATES = 20

OUTPUT_ROOT = Path(
    "data/proc_count_causal_v2"
)

IMAGE_DIR = OUTPUT_ROOT / "images"
METADATA_PATH = OUTPUT_ROOT / "metadata.jsonl"


def boxes_overlap(
    a,
    b,
    margin=4,
):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return not (
        ax2 + margin <= bx1
        or bx2 + margin <= ax1
        or ay2 + margin <= by1
        or by2 + margin <= ay1
    )


def circle_bbox(cx, cy, radius):
    return (
        cx - radius,
        cy - radius,
        cx + radius,
        cy + radius,
    )


def square_bbox(cx, cy, half):
    return (
        cx - half,
        cy - half,
        cx + half,
        cy + half,
    )


def clip_center(
    cx,
    cy,
    radius,
):
    cx = max(
        radius + 2,
        min(WIDTH - radius - 2, cx),
    )

    cy = max(
        radius + 2,
        min(HEIGHT - radius - 2, cy),
    )

    return cx, cy


def make_object_record(
    bbox,
    cx,
    cy,
    object_type,
):
    x1, y1, x2, y2 = bbox

    return {
        "type": object_type,
        "bbox_px": [
            round(x1, 3),
            round(y1, 3),
            round(x2, 3),
            round(y2, 3),
        ],
        "bbox_norm": [
            x1 / WIDTH,
            y1 / HEIGHT,
            x2 / WIDTH,
            y2 / HEIGHT,
        ],
        "center_px": [
            round(cx, 3),
            round(cy, 3),
        ],
        "center_norm": [
            cx / WIDTH,
            cy / HEIGHT,
        ],
    }


def generate_row(
    n,
    rng,
):
    """
    Horizontal row with seed-specific jitter.

    Replicates are therefore genuinely different,
    unlike a perfectly deterministic row layout.
    """

    if n == 1:
        base_xs = [WIDTH / 2]
    else:
        left = 60
        right = WIDTH - 60

        step = (
            right - left
        ) / (n - 1)

        base_xs = [
            left + i * step
            for i in range(n)
        ]

    base_y = HEIGHT * 0.36

    centers = []

    for x in base_xs:
        cx = x + rng.uniform(-6, 6)
        cy = base_y + rng.uniform(-10, 10)

        cx, cy = clip_center(
            cx,
            cy,
            TARGET_RADIUS,
        )

        centers.append(
            (cx, cy)
        )

    return centers


def generate_grid(
    n,
    rng,
):
    """
    Up to 5 columns × 2 rows with light jitter.
    """

    cols = min(
        5,
        n,
    )

    rows = math.ceil(
        n / cols
    )

    x_positions = [
        (i + 1)
        * WIDTH
        / (cols + 1)
        for i in range(cols)
    ]

    if rows == 1:
        y_positions = [
            HEIGHT * 0.40
        ]
    else:
        y_positions = [
            HEIGHT * 0.30,
            HEIGHT * 0.58,
        ]

    centers = []

    for idx in range(n):
        row = idx // cols
        col = idx % cols

        cx = (
            x_positions[col]
            + rng.uniform(-8, 8)
        )

        cy = (
            y_positions[row]
            + rng.uniform(-8, 8)
        )

        cx, cy = clip_center(
            cx,
            cy,
            TARGET_RADIUS,
        )

        centers.append(
            (cx, cy)
        )

    return centers


def rejection_sample_centers(
    n,
    rng,
    x_min,
    x_max,
    y_min,
    y_max,
    min_center_distance,
    max_tries=20000,
):
    centers = []

    tries = 0

    while (
        len(centers) < n
        and tries < max_tries
    ):
        tries += 1

        cx = rng.uniform(
            x_min,
            x_max,
        )

        cy = rng.uniform(
            y_min,
            y_max,
        )

        valid = True

        for ox, oy in centers:
            distance = math.sqrt(
                (cx - ox) ** 2
                + (cy - oy) ** 2
            )

            if (
                distance
                < min_center_distance
            ):
                valid = False
                break

        if valid:
            centers.append(
                (cx, cy)
            )

    if len(centers) != n:
        raise RuntimeError(
            f"Could not place {n} objects "
            f"after {max_tries} attempts."
        )

    return centers


def generate_random_sparse(
    n,
    rng,
):
    return rejection_sample_centers(
        n=n,
        rng=rng,
        x_min=55,
        x_max=WIDTH - 55,
        y_min=65,
        y_max=HEIGHT - 65,
        min_center_distance=85,
    )


def generate_dense(
    n,
    rng,
):
    """
    Compact layout but still no overlap/occlusion.
    """

    return rejection_sample_centers(
        n=n,
        rng=rng,
        x_min=190,
        x_max=WIDTH - 190,
        y_min=105,
        y_max=HEIGHT - 105,
        min_center_distance=56,
    )


def generate_distractor_targets(
    n,
    rng,
):
    """
    Targets are moderately spread.
    Blue square distractors are added separately.
    """

    return rejection_sample_centers(
        n=n,
        rng=rng,
        x_min=55,
        x_max=WIDTH - 55,
        y_min=65,
        y_max=HEIGHT - 65,
        min_center_distance=72,
    )


def generate_distractors(
    rng,
    target_boxes,
    n_distractors=8,
):
    centers = []
    boxes = []

    tries = 0

    while (
        len(centers) < n_distractors
        and tries < 30000
    ):
        tries += 1

        cx = rng.uniform(
            45,
            WIDTH - 45,
        )

        cy = rng.uniform(
            50,
            HEIGHT - 50,
        )

        box = square_bbox(
            cx,
            cy,
            DISTRACTOR_HALF,
        )

        bad = False

        for target_box in target_boxes:
            if boxes_overlap(
                box,
                target_box,
                margin=6,
            ):
                bad = True
                break

        if bad:
            continue

        for existing_box in boxes:
            if boxes_overlap(
                box,
                existing_box,
                margin=6,
            ):
                bad = True
                break

        if bad:
            continue

        centers.append(
            (cx, cy)
        )

        boxes.append(
            box
        )

    if (
        len(centers)
        != n_distractors
    ):
        raise RuntimeError(
            "Could not place distractors."
        )

    return centers


def draw_sample(
    count,
    condition,
    replicate,
    seed,
):
    rng = random.Random(
        seed
    )

    if condition == "row":
        target_centers = (
            generate_row(
                count,
                rng,
            )
        )

    elif condition == "grid":
        target_centers = (
            generate_grid(
                count,
                rng,
            )
        )

    elif condition == "random_sparse":
        target_centers = (
            generate_random_sparse(
                count,
                rng,
            )
        )

    elif condition == "dense":
        target_centers = (
            generate_dense(
                count,
                rng,
            )
        )

    elif condition == "distractors":
        target_centers = (
            generate_distractor_targets(
                count,
                rng,
            )
        )

    else:
        raise ValueError(
            condition
        )

    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        BACKGROUND,
    )

    draw = ImageDraw.Draw(
        image
    )

    target_objects = []
    target_boxes = []

    for cx, cy in target_centers:
        bbox = circle_bbox(
            cx,
            cy,
            TARGET_RADIUS,
        )

        target_boxes.append(
            bbox
        )

        draw.ellipse(
            bbox,
            fill=TARGET_COLOR,
        )

        target_objects.append(
            make_object_record(
                bbox,
                cx,
                cy,
                "red_circle",
            )
        )

    distractor_objects = []

    if condition == "distractors":

        distractor_centers = (
            generate_distractors(
                rng,
                target_boxes,
                n_distractors=8,
            )
        )

        for cx, cy in (
            distractor_centers
        ):
            bbox = square_bbox(
                cx,
                cy,
                DISTRACTOR_HALF,
            )

            draw.rectangle(
                bbox,
                fill=DISTRACTOR_COLOR,
            )

            distractor_objects.append(
                make_object_record(
                    bbox,
                    cx,
                    cy,
                    "blue_square",
                )
            )

    sample_id = (
        f"pccv2_n{count:02d}_"
        f"{condition}_r{replicate:02d}"
    )

    image_path = (
        IMAGE_DIR
        / f"{sample_id}.png"
    )

    image.save(
        image_path
    )

    return {
        "sample_id":
            sample_id,

        "dataset":
            "proc_count_causal_v2",

        "generation_version":
            "2.0",

        "seed":
            seed,

        "replicate":
            replicate,

        "ground_truth":
            count,

        "condition":
            condition,

        "image_path":
            str(image_path),

        "image_width":
            WIDTH,

        "image_height":
            HEIGHT,

        "target_class":
            "red_circle",

        "target_radius_px":
            TARGET_RADIUS,

        "target_objects":
            target_objects,

        "distractor_objects":
            distractor_objects,

        "num_distractors":
            len(
                distractor_objects
            ),

        "has_occlusion":
            False,
    }


def main():
    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for count in range(
        1,
        11,
    ):
        for condition_idx, condition in enumerate(
            CONDITIONS
        ):
            for replicate in range(
                REPLICATES
            ):

                seed = (
                    BASE_SEED
                    + count * 10000
                    + condition_idx * 100
                    + replicate
                )

                record = draw_sample(
                    count=count,
                    condition=condition,
                    replicate=replicate,
                    seed=seed,
                )

                records.append(
                    record
                )

                print(
                    f"{record['sample_id']:40s} "
                    f"GT={count:2d} "
                    f"seed={seed}"
                )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\n" + "=" * 80)
    print("Proc-Count-Causal v2 generated")
    print("=" * 80)

    print(
        "Samples:",
        len(records),
    )

    print(
        "Images :",
        IMAGE_DIR,
    )

    print(
        "Metadata:",
        METADATA_PATH,
    )

    assert len(records) == 1000


if __name__ == "__main__":
    main()
