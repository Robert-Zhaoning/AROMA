import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")

import generate_proc_count_causal_v3 as g


# ============================================================
# FROZEN EXPERIMENT D1 CONFIG
# ============================================================

BASE_SEED = 314159265
REPLICATES = 6

RESAMPLE_STRIDE = 1_000_000_000
MAX_ATTEMPTS = 1000

DATA_ROOT = Path(
    "data/proc_count_multi_actuator_d1_v1"
)

IMAGE_DIR = (
    DATA_ROOT
    / "images"
)

METADATA_PATH = (
    DATA_ROOT
    / "metadata.jsonl"
)

OUTPUT_ROOT = Path(
    "outputs/proc_count_multi_actuator_d1_v1"
)

MANIFEST_PATH = (
    OUTPUT_ROOT
    / "d1_clean_generation_manifest.json"
)

HISTORICAL_ROOTS = [
    Path("data/proc_count_causal_v1"),
    Path("data/proc_count_causal_v2"),
    Path("data/proc_count_causal_v3"),
    Path("data/proc_count_causal_v4"),
]


# ============================================================
# HELPERS
# ============================================================

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
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def image_files(root):
    image_dir = (
        root
        / "images"
    )

    if not image_dir.is_dir():
        raise RuntimeError(
            f"Missing image directory: {image_dir}"
        )

    return sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file()
    )


# ============================================================
# PRECONDITIONS
# ============================================================

if METADATA_PATH.exists():
    raise RuntimeError(
        f"D1 metadata already exists: {METADATA_PATH}"
    )

if DATA_ROOT.exists():
    leftovers = list(
        DATA_ROOT.rglob("*")
    )

    if leftovers:
        raise RuntimeError(
            f"D1 data directory is not empty: {DATA_ROOT}"
        )


# ============================================================
# HISTORICAL SEED + PIXEL INVENTORY
# ============================================================

historical_seeds = set()
historical_hashes = set()

historical_summary = {}

for root in HISTORICAL_ROOTS:

    metadata = (
        root
        / "metadata.jsonl"
    )

    if not metadata.is_file():
        raise RuntimeError(
            f"Missing historical metadata: {metadata}"
        )

    rows = load_jsonl(
        metadata
    )

    seeds = {
        int(r["seed"])
        for r in rows
    }

    files = image_files(
        root
    )

    hashes = {
        sha256_file(p)
        for p in files
    }

    historical_seeds.update(
        seeds
    )

    historical_hashes.update(
        hashes
    )

    historical_summary[
        root.name
    ] = {
        "rows": len(rows),
        "unique_seeds": len(seeds),
        "images": len(files),
        "unique_hashes": len(hashes),
    }


print(
    "Historical unique seeds:",
    len(historical_seeds),
)

print(
    "Historical unique hashes:",
    len(historical_hashes),
)


# ============================================================
# GENERATION
# ============================================================

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

records = []

accepted_seeds = set()
accepted_hashes = set()

seed_collision_count = 0
hash_collision_count = 0

original_image_dir = (
    g.IMAGE_DIR
)

g.IMAGE_DIR = (
    IMAGE_DIR
)


try:

    for count in range(
        1,
        11,
    ):

        for condition_idx, condition in enumerate(
            g.CONDITIONS
        ):

            for replicate in range(
                REPLICATES
            ):

                base_seed = (
                    BASE_SEED
                    +
                    count * 10000
                    +
                    condition_idx * 100
                    +
                    replicate
                )

                accepted = False

                for attempt in range(
                    MAX_ATTEMPTS
                ):

                    candidate_seed = (
                        base_seed
                        +
                        attempt
                        * RESAMPLE_STRIDE
                    )

                    if (
                        candidate_seed
                        in historical_seeds
                        or
                        candidate_seed
                        in accepted_seeds
                    ):
                        seed_collision_count += 1
                        continue

                    record = g.draw_sample(
                        count=count,
                        condition=condition,
                        replicate=replicate,
                        seed=candidate_seed,
                    )

                    if not isinstance(
                        record,
                        dict,
                    ):
                        raise RuntimeError(
                            "draw_sample() did not return a dict."
                        )

                    if "image_path" not in record:
                        raise RuntimeError(
                            "draw_sample() record missing image_path."
                        )

                    image_path = Path(
                        record["image_path"]
                    )

                    if not image_path.is_file():
                        raise RuntimeError(
                            f"Rendered image missing: {image_path}"
                        )

                    candidate_hash = (
                        sha256_file(
                            image_path
                        )
                    )

                    if (
                        candidate_hash
                        in historical_hashes
                        or
                        candidate_hash
                        in accepted_hashes
                    ):

                        hash_collision_count += 1

                        image_path.unlink(
                            missing_ok=True
                        )

                        continue

                    records.append(
                        record
                    )

                    accepted_seeds.add(
                        candidate_seed
                    )

                    accepted_hashes.add(
                        candidate_hash
                    )

                    accepted = True

                    break


                if not accepted:
                    raise RuntimeError(
                        "Could not generate a unique sample after "
                        f"{MAX_ATTEMPTS} attempts for "
                        f"count={count}, "
                        f"condition={condition}, "
                        f"replicate={replicate}."
                    )


                if (
                    len(records)
                    % 100
                    == 0
                ):
                    print(
                        f"accepted {len(records)} /300"
                    )

finally:

    g.IMAGE_DIR = (
        original_image_dir
    )


# ============================================================
# INVARIANTS
# ============================================================

EXPECTED_N = (
    10
    * len(g.CONDITIONS)
    * REPLICATES
)

if EXPECTED_N != 300:
    raise RuntimeError(
        f"Unexpected planned N: {EXPECTED_N}"
    )

if len(records) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} records; "
        f"got {len(records)}"
    )

if len(accepted_seeds) != EXPECTED_N:
    raise RuntimeError(
        "Accepted seeds are not unique."
    )

if len(accepted_hashes) != EXPECTED_N:
    raise RuntimeError(
        "Accepted image hashes are not unique."
    )


count_dist = Counter(
    int(
        r["ground_truth"]
    )
    for r in records
)

condition_dist = Counter(
    str(
        r["condition"]
    )
    for r in records
)

cell_dist = Counter(
    (
        int(r["ground_truth"]),
        str(r["condition"]),
    )
    for r in records
)


expected_count_dist = {
    i: 30
    for i in range(
        1,
        11,
    )
}

if dict(
    sorted(
        count_dist.items()
    )
) != expected_count_dist:

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


# ============================================================
# WRITE METADATA ONLY AFTER ALL CHECKS PASS
# ============================================================

with METADATA_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    for record in records:
        f.write(
            json.dumps(
                record,
                sort_keys=True,
            )
            + "\n"
        )


manifest = {
    "experiment":
        "Experiment D1 — Multi-actuator causal replication",

    "generator":
        "scripts/generate_multi_actuator_d1_clean.py",

    "renderer":
        "scripts/generate_proc_count_causal_v3.py",

    "base_seed":
        BASE_SEED,

    "replicates":
        REPLICATES,

    "n":
        len(records),

    "conditions":
        list(g.CONDITIONS),

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

    "unique_seeds":
        len(accepted_seeds),

    "unique_hashes":
        len(accepted_hashes),

    "seed_collision_resamples":
        seed_collision_count,

    "hash_collision_resamples":
        hash_collision_count,

    "historical_sources":
        historical_summary,

    "historical_seed_overlap":
        len(
            accepted_seeds
            &
            historical_seeds
        ),

    "historical_hash_overlap":
        len(
            accepted_hashes
            &
            historical_hashes
        ),

    "metadata_sha256":
        sha256_file(
            METADATA_PATH
        ),
}


MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
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
    "D1 CLEAN GENERATION: PASS"
)

print(
    "========================================"
)

print(
    "N:",
    len(records),
)

print(
    "Unique seeds:",
    len(accepted_seeds),
)

print(
    "Unique hashes:",
    len(accepted_hashes),
)

print(
    "Seed collision resamples:",
    seed_collision_count,
)

print(
    "Hash collision resamples:",
    hash_collision_count,
)

print(
    "Historical seed overlap:",
    0,
)

print(
    "Historical hash overlap:",
    0,
)

print(
    "Metadata SHA256:",
    manifest["metadata_sha256"],
)
