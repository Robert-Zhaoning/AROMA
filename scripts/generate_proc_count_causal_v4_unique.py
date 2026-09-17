import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(
    0,
    "scripts",
)

import generate_proc_count_causal_v3 as g


# ============================================================
# FROZEN V4 CONFIG
# ============================================================

V4_BASE_SEED = 271828182

RESAMPLE_STRIDE = 1_000_000_000
MAX_ATTEMPTS = 1000

V1_META = Path(
    "data/proc_count_causal_v1/metadata.jsonl"
)

V2_META = Path(
    "data/proc_count_causal_v2/metadata.jsonl"
)

V3_META = Path(
    "data/proc_count_causal_v3/metadata.jsonl"
)

V4_ROOT = Path(
    "data/proc_count_causal_v4"
)

V4_IMAGE_DIR = (
    V4_ROOT
    / "images"
)

V4_META = (
    V4_ROOT
    / "metadata.jsonl"
)

LOG_ROOT = Path(
    "outputs/proc_count_causal_v4"
)

LOG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

COLLISION_LOG = (
    LOG_ROOT
    / "v4_unique_collision_log.json"
)

HASH_INVENTORY = (
    LOG_ROOT
    / "v4_image_hashes.csv"
)

MANIFEST = (
    LOG_ROOT
    / "v4_generation_manifest.json"
)

RENDERER_PATH = Path(
    "scripts/generate_proc_count_causal_v3.py"
)


# ============================================================
# HELPERS
# ============================================================

def sha256(
    path,
):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def load_jsonl(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    return [
        json.loads(line)
        for line in (
            path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    ]


def record_image_path(
    record,
):

    path = Path(
        record[
            "image_path"
        ]
    )

    if not path.exists():

        raise RuntimeError(
            "Historical image "
            "does not exist: "
            f"{path}"
        )

    return path


def image_hash_set(
    records,
):

    return {
        sha256(
            record_image_path(
                r
            )
        )
        for r in records
    }


def seed_set(
    records,
):

    return {
        int(
            r[
                "seed"
            ]
        )
        for r in records
    }


# ============================================================
# PRE-GENERATION AUDIT
# ============================================================

print("=" * 88)
print(
    "PROC-COUNT-CAUSAL v4 "
    "UNIQUE GENERATION"
)
print("=" * 88)


v1 = load_jsonl(
    V1_META
)

v2 = load_jsonl(
    V2_META
)

v3 = load_jsonl(
    V3_META
)


v1_hashes = image_hash_set(
    v1
)

v2_hashes = image_hash_set(
    v2
)

v3_hashes = image_hash_set(
    v3
)


v1_seeds = seed_set(
    v1
)

v2_seeds = seed_set(
    v2
)

v3_seeds = seed_set(
    v3
)


historical_hashes = (
    v1_hashes
    |
    v2_hashes
    |
    v3_hashes
)

historical_seeds = (
    v1_seeds
    |
    v2_seeds
    |
    v3_seeds
)


print(
    "Historical seed counts:",
    len(v1_seeds),
    len(v2_seeds),
    len(v3_seeds),
)

print(
    "Historical hash counts:",
    len(v1_hashes),
    len(v2_hashes),
    len(v3_hashes),
)


if V4_META.exists():

    raise RuntimeError(
        "V4 metadata already exists. "
        "Refusing to regenerate:\n"
        f"{V4_META}"
    )


if (
    V4_IMAGE_DIR.exists()
    and
    any(
        V4_IMAGE_DIR.iterdir()
    )
):

    raise RuntimeError(
        "V4 image directory is "
        "non-empty. Refusing to "
        "regenerate:\n"
        f"{V4_IMAGE_DIR}"
    )


V4_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GENERATE USING EXACT V3 RENDERER
# ============================================================

accepted_hashes = set()
accepted_seeds = set()

records = []
collision_events = []


original_image_dir = (
    g.IMAGE_DIR
)

g.IMAGE_DIR = (
    V4_IMAGE_DIR
)


try:

    for count in range(
        1,
        11,
    ):

        for (
            condition_idx,
            condition
        ) in enumerate(
            g.CONDITIONS
        ):

            for replicate in range(
                g.REPLICATES
            ):

                base_seed = (
                    V4_BASE_SEED
                    +
                    count
                    * 10000
                    +
                    condition_idx
                    * 100
                    +
                    replicate
                )

                attempt = 0

                while True:

                    candidate_seed = (
                        base_seed
                        +
                        attempt
                        * RESAMPLE_STRIDE
                    )


                    # ----------------------------------------
                    # Seed independence
                    # ----------------------------------------

                    if (
                        candidate_seed
                        in historical_seeds
                        or
                        candidate_seed
                        in accepted_seeds
                    ):

                        collision_events.append(
                            {
                                "count":
                                    count,

                                "condition":
                                    condition,

                                "replicate":
                                    replicate,

                                "attempt":
                                    attempt,

                                "rejected_seed":
                                    candidate_seed,

                                "collision_type":
                                    "seed_collision",
                            }
                        )

                        attempt += 1

                        if (
                            attempt
                            >
                            MAX_ATTEMPTS
                        ):

                            raise RuntimeError(
                                "Too many seed "
                                "collision retries."
                            )

                        continue


                    # ----------------------------------------
                    # Render candidate with exact v3 renderer.
                    #
                    # v3 draw_sample temporarily gives it a
                    # pccv3 namespace. We rename namespace
                    # only AFTER rendering. Pixel generation
                    # is therefore exactly the same.
                    # ----------------------------------------

                    record = (
                        g.draw_sample(
                            count=count,
                            condition=condition,
                            replicate=replicate,
                            seed=candidate_seed,
                        )
                    )


                    temporary_path = Path(
                        record[
                            "image_path"
                        ]
                    )


                    candidate_hash = (
                        sha256(
                            temporary_path
                        )
                    )


                    # ----------------------------------------
                    # Exact-pixel independence
                    # ----------------------------------------

                    if (
                        candidate_hash
                        in v1_hashes
                    ):

                        collision_type = (
                            "cross_v1_v4"
                        )

                    elif (
                        candidate_hash
                        in v2_hashes
                    ):

                        collision_type = (
                            "cross_v2_v4"
                        )

                    elif (
                        candidate_hash
                        in v3_hashes
                    ):

                        collision_type = (
                            "cross_v3_v4"
                        )

                    elif (
                        candidate_hash
                        in accepted_hashes
                    ):

                        collision_type = (
                            "within_v4"
                        )

                    else:

                        collision_type = None


                    if (
                        collision_type
                        is not None
                    ):

                        collision_events.append(
                            {
                                "count":
                                    count,

                                "condition":
                                    condition,

                                "replicate":
                                    replicate,

                                "attempt":
                                    attempt,

                                "rejected_seed":
                                    candidate_seed,

                                "collision_type":
                                    collision_type,

                                "sha256":
                                    candidate_hash,
                            }
                        )


                        if (
                            temporary_path
                            .exists()
                        ):

                            temporary_path.unlink()


                        attempt += 1


                        if (
                            attempt
                            >
                            MAX_ATTEMPTS
                        ):

                            raise RuntimeError(
                                "Too many pixel "
                                "collision retries."
                            )

                        continue


                    # ----------------------------------------
                    # ACCEPT
                    # ----------------------------------------

                    sample_id = (
                        f"pccv4_n"
                        f"{count:02d}_"
                        f"{condition}_"
                        f"r{replicate:02d}"
                    )


                    final_path = (
                        V4_IMAGE_DIR
                        /
                        f"{sample_id}.png"
                    )


                    if (
                        final_path.exists()
                    ):

                        raise RuntimeError(
                            "Unexpected existing "
                            f"V4 image: "
                            f"{final_path}"
                        )


                    temporary_path.rename(
                        final_path
                    )


                    record[
                        "sample_id"
                    ] = sample_id

                    record[
                        "dataset"
                    ] = (
                        "proc_count_causal_v4"
                    )

                    record[
                        "generation_version"
                    ] = (
                        "4.0_unique"
                    )

                    record[
                        "seed"
                    ] = int(
                        candidate_seed
                    )

                    record[
                        "base_seed"
                    ] = int(
                        base_seed
                    )

                    record[
                        "collision_resample_attempts"
                    ] = int(
                        attempt
                    )

                    record[
                        "image_path"
                    ] = str(
                        final_path
                    )

                    record[
                        "renderer_source"
                    ] = (
                        "generate_proc_count_"
                        "causal_v3.draw_sample"
                    )

                    record[
                        "renderer_sha256"
                    ] = sha256(
                        RENDERER_PATH
                    )


                    records.append(
                        record
                    )

                    accepted_hashes.add(
                        candidate_hash
                    )

                    accepted_seeds.add(
                        int(
                            candidate_seed
                        )
                    )


                    if (
                        len(records)
                        % 100
                        == 0
                    ):

                        print(
                            "accepted",
                            len(records),
                            "/2000"
                        )


                    break

finally:

    g.IMAGE_DIR = (
        original_image_dir
    )


# ============================================================
# STRUCTURAL AUDITS
# ============================================================

if len(records) != 2000:

    raise RuntimeError(
        f"Expected 2000 records; "
        f"got {len(records)}"
    )


if len(accepted_seeds) != 2000:

    raise RuntimeError(
        "V4 seeds are not "
        "all unique."
    )


if len(accepted_hashes) != 2000:

    raise RuntimeError(
        "V4 image hashes are "
        "not all unique."
    )


sample_ids = {
    r[
        "sample_id"
    ]
    for r in records
}


if len(sample_ids) != 2000:

    raise RuntimeError(
        "V4 sample IDs are "
        "not all unique."
    )


for r in records:

    if not (
        r[
            "sample_id"
        ]
        .startswith(
            "pccv4_"
        )
    ):

        raise RuntimeError(
            "Bad V4 sample ID."
        )


    if (
        r[
            "dataset"
        ]
        !=
        "proc_count_causal_v4"
    ):

        raise RuntimeError(
            "Bad V4 dataset field."
        )


    if (
        int(
            r[
                "ground_truth"
            ]
        )
        !=
        len(
            r[
                "target_objects"
            ]
        )
    ):

        raise RuntimeError(
            "GT/object-count mismatch: "
            f"{r['sample_id']}"
        )


    if (
        int(
            r[
                "image_width"
            ]
        )
        !=
        768
        or
        int(
            r[
                "image_height"
            ]
        )
        !=
        512
    ):

        raise RuntimeError(
            "Image-size mismatch."
        )


condition_counts = Counter(
    r[
        "condition"
    ]
    for r in records
)


for condition in (
    g.CONDITIONS
):

    if (
        condition_counts[
            condition
        ]
        !=
        400
    ):

        raise RuntimeError(
            f"{condition}: "
            f"expected 400."
        )


count_counts = Counter(
    int(
        r[
            "ground_truth"
        ]
    )
    for r in records
)


for n in range(
    1,
    11,
):

    if (
        count_counts[n]
        !=
        200
    ):

        raise RuntimeError(
            f"count {n}: "
            f"expected 200."
        )


combo_counts = Counter(
    (
        int(
            r[
                "ground_truth"
            ]
        ),
        r[
            "condition"
        ],
    )
    for r in records
)


if len(combo_counts) != 50:

    raise RuntimeError(
        "Expected exactly 50 "
        "count-condition cells."
    )


if not all(
    v == 40
    for v in (
        combo_counts
        .values()
    )
):

    raise RuntimeError(
        "Every count-condition "
        "cell must contain 40 "
        "samples."
    )


# ============================================================
# HISTORICAL INDEPENDENCE AUDITS
# ============================================================

seed_overlap_v1 = (
    accepted_seeds
    &
    v1_seeds
)

seed_overlap_v2 = (
    accepted_seeds
    &
    v2_seeds
)

seed_overlap_v3 = (
    accepted_seeds
    &
    v3_seeds
)


hash_overlap_v1 = (
    accepted_hashes
    &
    v1_hashes
)

hash_overlap_v2 = (
    accepted_hashes
    &
    v2_hashes
)

hash_overlap_v3 = (
    accepted_hashes
    &
    v3_hashes
)


if seed_overlap_v1:

    raise RuntimeError(
        "V1/V4 seed overlap."
    )

if seed_overlap_v2:

    raise RuntimeError(
        "V2/V4 seed overlap."
    )

if seed_overlap_v3:

    raise RuntimeError(
        "V3/V4 seed overlap."
    )


if hash_overlap_v1:

    raise RuntimeError(
        "V1/V4 image overlap."
    )

if hash_overlap_v2:

    raise RuntimeError(
        "V2/V4 image overlap."
    )

if hash_overlap_v3:

    raise RuntimeError(
        "V3/V4 image overlap."
    )


# ============================================================
# SAVE METADATA ONLY AFTER ALL AUDITS PASS
# ============================================================

V4_META.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with V4_META.open(
    "w",
    encoding="utf-8",
) as f:

    for r in records:

        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            +
            "\n"
        )


# ============================================================
# HASH INVENTORY
# ============================================================

with HASH_INVENTORY.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sample_id",
            "seed",
            "sha256",
            "image_path",
        ],
    )

    writer.writeheader()

    for r in records:

        writer.writerow(
            {
                "sample_id":
                    r[
                        "sample_id"
                    ],

                "seed":
                    r[
                        "seed"
                    ],

                "sha256":
                    sha256(
                        r[
                            "image_path"
                        ]
                    ),

                "image_path":
                    r[
                        "image_path"
                    ],
            }
        )


# ============================================================
# COLLISION LOG
# ============================================================

COLLISION_LOG.write_text(
    json.dumps(
        {
            "generation_version":
                "4.0_unique",

            "base_seed":
                V4_BASE_SEED,

            "resample_stride":
                RESAMPLE_STRIDE,

            "samples":
                len(records),

            "collision_events":
                collision_events,
        },
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "dataset":
        "proc_count_causal_v4",

    "generation_version":
        "4.0_unique",

    "n":
        len(records),

    "base_seed":
        V4_BASE_SEED,

    "renderer_path":
        str(
            RENDERER_PATH
        ),

    "renderer_sha256":
        sha256(
            RENDERER_PATH
        ),

    "metadata_sha256":
        sha256(
            V4_META
        ),

    "counts":
        dict(
            sorted(
                count_counts
                .items()
            )
        ),

    "conditions":
        dict(
            sorted(
                condition_counts
                .items()
            )
        ),

    "unique_seeds":
        len(
            accepted_seeds
        ),

    "unique_hashes":
        len(
            accepted_hashes
        ),

    "collision_events":
        len(
            collision_events
        ),

    "seed_overlap": {
        "v1":
            len(
                seed_overlap_v1
            ),

        "v2":
            len(
                seed_overlap_v2
            ),

        "v3":
            len(
                seed_overlap_v3
            ),
    },

    "image_hash_overlap": {
        "v1":
            len(
                hash_overlap_v1
            ),

        "v2":
            len(
                hash_overlap_v2
            ),

        "v3":
            len(
                hash_overlap_v3
            ),
    },
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 88)
print(
    "V4 GENERATION PASSED"
)
print("=" * 88)

print(
    "Samples:",
    len(records)
)

print(
    "Unique seeds:",
    len(accepted_seeds)
)

print(
    "Unique hashes:",
    len(accepted_hashes)
)

print(
    "Collision events:",
    len(collision_events)
)

print()

print(
    "Seed overlap v1/v2/v3:",
    len(seed_overlap_v1),
    len(seed_overlap_v2),
    len(seed_overlap_v3),
)

print(
    "Hash overlap v1/v2/v3:",
    len(hash_overlap_v1),
    len(hash_overlap_v2),
    len(hash_overlap_v3),
)

print()

print(
    "Count distribution:",
    dict(
        sorted(
            count_counts.items()
        )
    )
)

print(
    "Condition distribution:",
    dict(
        sorted(
            condition_counts.items()
        )
    )
)

print()

print(
    "Renderer SHA256:",
    sha256(
        RENDERER_PATH
    )
)

print(
    "Metadata SHA256:",
    sha256(
        V4_META
    )
)

print()
print(
    "Metadata:",
    V4_META
)

print(
    "Manifest:",
    MANIFEST
)

print(
    "Hash inventory:",
    HASH_INVENTORY
)

print("=" * 88)

