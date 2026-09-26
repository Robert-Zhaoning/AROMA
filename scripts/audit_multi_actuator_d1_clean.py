import hashlib
import json
from collections import Counter
from pathlib import Path


D1 = Path(
    "data/proc_count_multi_actuator_d1_v1"
)

META = (
    D1
    / "metadata.jsonl"
)

HIST = [
    Path("data/proc_count_causal_v1"),
    Path("data/proc_count_causal_v2"),
    Path("data/proc_count_causal_v3"),
    Path("data/proc_count_causal_v4"),
]

FREEZE = Path(
    "manifests/multi_actuator_d1_v1/"
    "cohort_freeze_v1.json"
)

GEN = Path(
    "scripts/generate_multi_actuator_d1_clean.py"
)

PROTO = Path(
    "manifests/multi_actuator_d1_v1/"
    "protocol_v1.json"
)

MANIFEST = Path(
    "outputs/proc_count_multi_actuator_d1_v1/"
    "d1_clean_generation_manifest.json"
)


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def image_hashes(root):
    files = sorted(
        p
        for p in (root / "images").rglob("*")
        if p.is_file()
    )

    return (
        {
            sha256_file(p)
            for p in files
        },
        len(files),
    )


rows = load_jsonl(
    META
)

if len(rows) != 300:
    raise RuntimeError(
        f"Expected N=300; got {len(rows)}"
    )


seeds = {
    int(r["seed"])
    for r in rows
}

hashes, image_count = (
    image_hashes(D1)
)

if len(seeds) != 300:
    raise RuntimeError(
        "D1 seeds are not unique."
    )

if image_count != 300:
    raise RuntimeError(
        f"Expected 300 images; got {image_count}"
    )

if len(hashes) != 300:
    raise RuntimeError(
        "D1 pixel hashes are not unique."
    )


count_dist = Counter(
    int(r["ground_truth"])
    for r in rows
)

condition_dist = Counter(
    str(r["condition"])
    for r in rows
)

cell_dist = Counter(
    (
        int(r["ground_truth"]),
        str(r["condition"]),
    )
    for r in rows
)


if count_dist != Counter(
    {
        i: 30
        for i in range(
            1,
            11,
        )
    }
):
    raise RuntimeError(
        f"Count balance failure: {count_dist}"
    )


if (
    len(condition_dist) != 5
    or
    set(
        condition_dist.values()
    ) != {60}
):
    raise RuntimeError(
        f"Condition balance failure: {condition_dist}"
    )


if (
    len(cell_dist) != 50
    or
    set(
        cell_dist.values()
    ) != {6}
):
    raise RuntimeError(
        f"Cell balance failure: {cell_dist}"
    )


overlaps = {}

for i, root in enumerate(
    HIST,
    start=1,
):

    hist_rows = (
        load_jsonl(
            root
            / "metadata.jsonl"
        )
    )

    hist_seeds = {
        int(r["seed"])
        for r in hist_rows
    }

    hist_hashes, _ = (
        image_hashes(
            root
        )
    )

    seed_overlap = len(
        seeds
        &
        hist_seeds
    )

    hash_overlap = len(
        hashes
        &
        hist_hashes
    )

    overlaps[
        f"v{i}"
    ] = {
        "seed_overlap":
            seed_overlap,

        "hash_overlap":
            hash_overlap,
    }

    print(
        f"v{i}: "
        f"seed_overlap = {seed_overlap} "
        f"hash_overlap = {hash_overlap}"
    )

    if seed_overlap != 0:
        raise RuntimeError(
            f"Seed overlap with v{i}"
        )

    if hash_overlap != 0:
        raise RuntimeError(
            f"Pixel-hash overlap with v{i}"
        )


freeze = {
    "experiment":
        "Experiment D1 — Multi-actuator causal replication",

    "status":
        "fresh cohort frozen before model inference",

    "n":
        300,

    "unique_seeds":
        300,

    "unique_image_hashes":
        300,

    "count_distribution":
        dict(
            sorted(
                count_dist.items()
            )
        ),

    "condition_distribution":
        dict(
            sorted(
                condition_dist.items()
            )
        ),

    "count_condition_cells":
        50,

    "samples_per_cell":
        6,

    "historical_overlap":
        overlaps,

    "metadata_sha256":
        sha256_file(
            META
        ),

    "generator_sha256":
        sha256_file(
            GEN
        ),

    "protocol_sha256":
        sha256_file(
            PROTO
        ),

    "generation_manifest_sha256":
        sha256_file(
            MANIFEST
        ),

    "model_inference_occurred":
        False,

    "model_outcomes_observed":
        False,
}


FREEZE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FREEZE.write_text(
    json.dumps(
        freeze,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print(
    "========================================"
)

print(
    "D1 COHORT AUDIT: PASS"
)

print(
    "========================================"
)

print(
    "D1 rows:",
    len(rows),
)

print(
    "D1 unique seeds:",
    len(seeds),
)

print(
    "D1 unique hashes:",
    len(hashes),
)

print(
    "cells:",
    len(cell_dist),
)

print(
    "samples/cell:",
    sorted(
        set(
            cell_dist.values()
        )
    ),
)

print(
    "metadata SHA256:",
    freeze["metadata_sha256"],
)
