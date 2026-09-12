import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


MANIFEST_PATH = Path(
    "data/tallyqa_natural_ood_v1/"
    "manifest.jsonl"
)

IMAGE_ROOT = Path(
    "data/tallyqa_natural_ood_v1/"
    "images"
)

OUT_DIR = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "image_inventory"
)

INVENTORY_PATH = (
    OUT_DIR
    / "image_inventory.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "image_inventory_summary.json"
)

EXPECTED_MANIFEST_SHA256 = (
    "068a489a02d499da6eb0b837a81c8817583b64b51c23f944c77645420fe2c32c"
)


def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def pixel_sha256(path):

    with Image.open(path) as image:

        image = image.convert("RGB")

        width, height = image.size

        h = hashlib.sha256()

        h.update(
            f"{width}x{height}|RGB|".encode(
                "utf-8"
            )
        )

        h.update(
            image.tobytes()
        )

        return (
            h.hexdigest(),
            width,
            height,
        )


def load_manifest():

    observed = sha256_file(
        MANIFEST_PATH
    )

    if observed != EXPECTED_MANIFEST_SHA256:

        raise RuntimeError(
            "Frozen manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Observed: {observed}"
        )

    rows = [
        json.loads(line)
        for line in MANIFEST_PATH
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    if len(rows) != 4000:

        raise RuntimeError(
            f"Expected 4000 manifest rows, "
            f"found {len(rows)}."
        )

    return rows


def main():

    print("=" * 112)
    print(
        "TALLYQA NATURAL-OOD "
        "FROZEN IMAGE INVENTORY AUDIT"
    )
    print("=" * 112)

    manifest = load_manifest()

    expected_paths = [
        str(
            row["image"]
        )
        for row in manifest
    ]

    expected_set = set(
        expected_paths
    )

    if len(expected_set) != 4000:

        raise RuntimeError(
            "Manifest does not contain "
            "4000 unique image paths."
        )

    # --------------------------------------------------------
    # Local files
    # --------------------------------------------------------

    local_files = sorted(
        p
        for p in IMAGE_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    )

    local_relative = {
        str(
            p.relative_to(
                IMAGE_ROOT
            )
        )
        for p in local_files
    }

    missing = sorted(
        expected_set
        -
        local_relative
    )

    unexpected = sorted(
        local_relative
        -
        expected_set
    )

    print(
        "Manifest images :",
        len(expected_set),
    )

    print(
        "Local images    :",
        len(local_relative),
    )

    print(
        "Missing         :",
        len(missing),
    )

    print(
        "Unexpected      :",
        len(unexpected),
    )

    if missing:

        print(
            "\nMissing examples:"
        )

        for x in missing[:20]:
            print(
                " ",
                x,
            )

        raise RuntimeError(
            "Frozen manifest images are "
            "missing locally."
        )

    if unexpected:

        print(
            "\nUnexpected examples:"
        )

        for x in unexpected[:20]:
            print(
                " ",
                x,
            )

        raise RuntimeError(
            "Unexpected image files exist "
            "inside frozen image root."
        )

    # --------------------------------------------------------
    # Inspect every selected image.
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_by_image = {
        str(
            row["image"]
        ):
            row
        for row in manifest
    }

    inventory = []

    corrupt = []

    byte_hash_groups = defaultdict(
        list
    )

    pixel_hash_groups = defaultdict(
        list
    )

    prefix_counts = Counter()

    dimension_counts = Counter()

    total_bytes = 0

    for index, relative in enumerate(
        sorted(expected_paths),
        start=1,
    ):

        path = (
            IMAGE_ROOT
            /
            relative
        )

        try:

            # First force PIL to verify file structure.
            with Image.open(path) as image:
                image.verify()

            # Re-open for actual decode / pixel hashing.
            pixel_hash, width, height = (
                pixel_sha256(
                    path
                )
            )

        except Exception as exc:

            corrupt.append({
                "image":
                    relative,

                "error":
                    repr(exc),
            })

            continue

        byte_hash = sha256_file(
            path
        )

        size_bytes = int(
            path.stat().st_size
        )

        total_bytes += (
            size_bytes
        )

        prefix = (
            relative
            .split(
                "/",
                1,
            )[0]
        )

        prefix_counts[
            prefix
        ] += 1

        dimension_counts[
            (
                width,
                height,
            )
        ] += 1

        byte_hash_groups[
            byte_hash
        ].append(
            relative
        )

        pixel_hash_groups[
            pixel_hash
        ].append(
            relative
        )

        source = (
            manifest_by_image[
                relative
            ]
        )

        inventory.append({
            "image":
                relative,

            "question_id":
                int(
                    source[
                        "question_id"
                    ]
                ),

            "subset":
                str(
                    source[
                        "subset"
                    ]
                ),

            "answer":
                int(
                    source[
                        "answer"
                    ]
                ),

            "data_source":
                str(
                    source[
                        "data_source"
                    ]
                ),

            "width":
                int(
                    width
                ),

            "height":
                int(
                    height
                ),

            "file_bytes":
                size_bytes,

            "file_sha256":
                byte_hash,

            "pixel_sha256":
                pixel_hash,
        })

        if (
            index % 250 == 0
            or index == 4000
        ):

            print(
                f"Audited "
                f"{index}/4000"
            )

    print(
        "\nCorrupt/unreadable:",
        len(corrupt),
    )

    if corrupt:

        for item in corrupt[:20]:
            print(
                item
            )

        raise RuntimeError(
            "Corrupt image(s) detected."
        )

    if len(inventory) != 4000:

        raise RuntimeError(
            "Inventory does not contain "
            "4000 valid images."
        )

    # --------------------------------------------------------
    # Duplicate analysis.
    #
    # IMPORTANT:
    # Do NOT resample if duplicates are found. The question
    # manifest is already frozen. Report them transparently.
    # --------------------------------------------------------

    byte_duplicate_groups = {
        h:
            paths
        for h, paths
        in byte_hash_groups.items()
        if len(paths) > 1
    }

    pixel_duplicate_groups = {
        h:
            paths
        for h, paths
        in pixel_hash_groups.items()
        if len(paths) > 1
    }

    byte_duplicate_images = sum(
        len(paths)
        for paths
        in byte_duplicate_groups.values()
    )

    pixel_duplicate_images = sum(
        len(paths)
        for paths
        in pixel_duplicate_groups.values()
    )

    # --------------------------------------------------------
    # Save deterministic inventory.
    # --------------------------------------------------------

    fieldnames = [
        "image",
        "question_id",
        "subset",
        "answer",
        "data_source",
        "width",
        "height",
        "file_bytes",
        "file_sha256",
        "pixel_sha256",
    ]

    inventory = sorted(
        inventory,
        key=lambda row:
            row["image"],
    )

    with INVENTORY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            inventory
        )

    inventory_sha256 = (
        sha256_file(
            INVENTORY_PATH
        )
    )

    summary = {
        "manifest_sha256":
            EXPECTED_MANIFEST_SHA256,

        "manifest_images":
            4000,

        "local_images":
            4000,

        "missing_images":
            0,

        "unexpected_images":
            0,

        "corrupt_images":
            0,

        "prefix_counts":
            dict(
                sorted(
                    prefix_counts.items()
                )
            ),

        "total_bytes":
            int(
                total_bytes
            ),

        "unique_file_hashes":
            len(
                byte_hash_groups
            ),

        "byte_duplicate_group_count":
            len(
                byte_duplicate_groups
            ),

        "byte_duplicate_image_count":
            byte_duplicate_images,

        "unique_pixel_hashes":
            len(
                pixel_hash_groups
            ),

        "pixel_duplicate_group_count":
            len(
                pixel_duplicate_groups
            ),

        "pixel_duplicate_image_count":
            pixel_duplicate_images,

        "unique_dimension_pairs":
            len(
                dimension_counts
            ),

        "inventory_sha256":
            inventory_sha256,

        "byte_duplicate_groups":
            byte_duplicate_groups,

        "pixel_duplicate_groups":
            pixel_duplicate_groups,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "IMAGE INVENTORY RESULTS"
    )

    print(
        "=" * 112
    )

    print(
        "Valid images             :",
        len(inventory),
    )

    print(
        "VG_100K                  :",
        prefix_counts.get(
            "VG_100K",
            0,
        ),
    )

    print(
        "VG_100K_2                :",
        prefix_counts.get(
            "VG_100K_2",
            0,
        ),
    )

    print(
        "Unique file SHA256       :",
        len(
            byte_hash_groups
        ),
    )

    print(
        "Byte duplicate groups    :",
        len(
            byte_duplicate_groups
        ),
    )

    print(
        "Byte duplicate images    :",
        byte_duplicate_images,
    )

    print(
        "Unique pixel SHA256      :",
        len(
            pixel_hash_groups
        ),
    )

    print(
        "Pixel duplicate groups   :",
        len(
            pixel_duplicate_groups
        ),
    )

    print(
        "Pixel duplicate images   :",
        pixel_duplicate_images,
    )

    print(
        "Unique dimension pairs   :",
        len(
            dimension_counts
        ),
    )

    print(
        "Total bytes              :",
        total_bytes,
    )

    print(
        "Inventory SHA256         :",
        inventory_sha256,
    )

    if pixel_duplicate_groups:

        print(
            "\nNOTE: pixel-identical "
            "images were detected."
        )

        print(
            "They are NOT removed or "
            "resampled because the "
            "evaluation manifest is "
            "already frozen."
        )

        print(
            "\nFirst duplicate groups:"
        )

        for i, paths in enumerate(
            pixel_duplicate_groups.values()
        ):

            print(
                paths
            )

            if i >= 9:
                break

    print(
        "\nSaved:"
    )

    print(
        INVENTORY_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nTALLYQA FROZEN IMAGE "
        "INVENTORY AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
