import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")

import generate_proc_count_causal_v3 as g


V1_META = Path(
    "data/proc_count_causal_v1/metadata.jsonl"
)

V2_META = Path(
    "data/proc_count_causal_v2/metadata.jsonl"
)

LOG_DIR = Path(
    "outputs/proc_count_causal_v3"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COLLISION_LOG = (
    LOG_DIR
    / "v3_unique_collision_log.json"
)

BACKUP_META = (
    LOG_DIR
    / "metadata_v3_pre_unique.jsonl"
)


def sha256(path):
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


def load_jsonl(path):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def image_hashes(records):

    result = set()

    for r in records:

        p = Path(
            r[
                "image_path"
            ]
        )

        if not p.exists():
            raise FileNotFoundError(
                p
            )

        result.add(
            sha256(
                p
            )
        )

    return result


def seeds(records):

    return {
        int(
            r[
                "seed"
            ]
        )
        for r in records
    }


def main():

    print("=" * 108)
    print(
        "PROC-COUNT-CAUSAL v3.0 "
        "UNTOUCHED UNIQUE FINAL-CONFIRMATION GENERATOR"
    )
    print("=" * 108)

    # --------------------------------------------------------
    # Existing V1/V2 are immutable reference datasets.
    # --------------------------------------------------------

    v1 = load_jsonl(
        V1_META
    )

    v2 = load_jsonl(
        V2_META
    )

    if len(v1) != 300:
        raise RuntimeError(
            f"Expected V1=300, "
            f"found {len(v1)}"
        )

    if len(v2) != 1000:
        raise RuntimeError(
            f"Expected V2=1000, "
            f"found {len(v2)}"
        )

    v1_hashes = image_hashes(
        v1
    )

    v2_hashes = image_hashes(
        v2
    )

    v1_seeds = seeds(
        v1
    )

    v2_seeds = seeds(
        v2
    )

    print(
        "V1 image hashes:",
        len(
            v1_hashes
        ),
    )

    print(
        "V2 image hashes:",
        len(
            v2_hashes
        ),
    )

    print(
        "V1 seed count:",
        len(
            v1_seeds
        ),
    )

    print(
        "V2 seed count:",
        len(
            v2_seeds
        ),
    )

    if (
        v1_hashes
        &
        v2_hashes
    ):
        raise RuntimeError(
            "Existing V1/V2 exact image "
            "overlap detected."
        )

    # --------------------------------------------------------
    # Refuse to silently mix with a previous V3 generation.
    # --------------------------------------------------------

    existing_pngs = []

    if g.IMAGE_DIR.exists():

        existing_pngs = list(
            g.IMAGE_DIR.glob(
                "*.png"
            )
        )

    if existing_pngs:

        raise RuntimeError(
            "V3 image directory already "
            f"contains {len(existing_pngs)} PNG files.\n"
            "Do not silently regenerate an "
            "untouched final-confirmation set.\n"
            f"Directory: {g.IMAGE_DIR}"
        )

    if g.METADATA_PATH.exists():

        shutil.copy2(
            g.METADATA_PATH,
            BACKUP_META,
        )

        raise RuntimeError(
            "V3 metadata already exists. "
            "Backed it up to:\n"
            f"{BACKUP_META}\n"
            "Stop and inspect before "
            "regenerating."
        )

    g.IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Generate V3.
    #
    # Candidate seed:
    #
    #   base_seed + attempt * 1_000_000_000
    #
    # Attempt 0 is always the protocol-prespecified seed.
    # Resampling occurs ONLY for:
    #
    #   - seed collision
    #   - exact pixel collision
    #
    # Never for model outcome.
    # --------------------------------------------------------

    accepted_hashes = set()
    accepted_seeds = set()

    records = []
    collision_events = []

    historical_seeds = (
        v1_seeds
        |
        v2_seeds
    )

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
                    g.BASE_SEED
                    +
                    count * 10000
                    +
                    condition_idx * 100
                    +
                    replicate
                )

                attempt = 0

                while True:

                    candidate_seed = (
                        base_seed
                        +
                        attempt
                        *
                        1_000_000_000
                    )

                    # ----------------------------------------
                    # Seed independence.
                    # ----------------------------------------

                    if (
                        candidate_seed
                        in historical_seeds
                        or
                        candidate_seed
                        in accepted_seeds
                    ):

                        collision_events.append({
                            "sample_id":
                                (
                                    f"pccv3_n"
                                    f"{count:02d}_"
                                    f"{condition}_"
                                    f"r{replicate:02d}"
                                ),

                            "attempt":
                                int(
                                    attempt
                                ),

                            "rejected_seed":
                                int(
                                    candidate_seed
                                ),

                            "collision_type":
                                "seed_collision",
                        })

                        attempt += 1

                        if attempt > 1000:
                            raise RuntimeError(
                                "Too many seed "
                                "resampling attempts."
                            )

                        continue

                    # ----------------------------------------
                    # Deterministically draw candidate.
                    # ----------------------------------------

                    record = g.draw_sample(
                        count=count,
                        condition=condition,
                        replicate=replicate,
                        seed=candidate_seed,
                    )

                    image_path = Path(
                        record[
                            "image_path"
                        ]
                    )

                    image_hash = sha256(
                        image_path
                    )

                    # ----------------------------------------
                    # Exact image independence.
                    # ----------------------------------------

                    if image_hash in v1_hashes:

                        collision_type = (
                            "cross_v1_v3"
                        )

                    elif image_hash in v2_hashes:

                        collision_type = (
                            "cross_v2_v3"
                        )

                    elif image_hash in accepted_hashes:

                        collision_type = (
                            "within_v3"
                        )

                    else:

                        collision_type = None

                    if collision_type is not None:

                        collision_events.append({
                            "sample_id":
                                record[
                                    "sample_id"
                                ],

                            "attempt":
                                int(
                                    attempt
                                ),

                            "rejected_seed":
                                int(
                                    candidate_seed
                                ),

                            "collision_type":
                                collision_type,

                            "sha256":
                                image_hash,
                        })

                        attempt += 1

                        if attempt > 1000:

                            raise RuntimeError(
                                "Too many pixel "
                                "collision resampling "
                                "attempts."
                            )

                        continue

                    # ----------------------------------------
                    # Accept.
                    # ----------------------------------------

                    record[
                        "generation_version"
                    ] = "3.0_unique"

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

                    records.append(
                        record
                    )

                    accepted_hashes.add(
                        image_hash
                    )

                    accepted_seeds.add(
                        int(
                            candidate_seed
                        )
                    )

                    break

    # --------------------------------------------------------
    # Structural integrity.
    # --------------------------------------------------------

    if len(records) != 2000:
        raise RuntimeError(
            f"Expected 2000 records, "
            f"found {len(records)}"
        )

    if len(accepted_hashes) != 2000:
        raise RuntimeError(
            "V3 image hashes are "
            "not all unique."
        )

    if len(accepted_seeds) != 2000:
        raise RuntimeError(
            "V3 seeds are not "
            "all unique."
        )

    ids = {
        r[
            "sample_id"
        ]
        for r in records
    }

    if len(ids) != 2000:
        raise RuntimeError(
            "V3 sample IDs are "
            "not all unique."
        )

    # --------------------------------------------------------
    # Namespace/path audit.
    # --------------------------------------------------------

    for r in records:

        if not str(
            r[
                "sample_id"
            ]
        ).startswith(
            "pccv3_"
        ):
            raise RuntimeError(
                "Non-V3 sample ID: "
                f"{r['sample_id']}"
            )

        if (
            r[
                "dataset"
            ]
            !=
            "proc_count_causal_v3"
        ):
            raise RuntimeError(
                "Wrong dataset field."
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

        p = Path(
            r[
                "image_path"
            ]
        )

        if (
            "proc_count_causal_v3"
            not in
            str(p)
        ):
            raise RuntimeError(
                "Wrong image path namespace: "
                f"{p}"
            )

    # --------------------------------------------------------
    # Balance audit.
    # --------------------------------------------------------

    condition_counts = Counter(
        r[
            "condition"
        ]
        for r in records
    )

    for c in g.CONDITIONS:

        if (
            condition_counts[
                c
            ]
            != 400
        ):
            raise RuntimeError(
                f"Condition {c}: "
                f"{condition_counts[c]}, "
                "expected 400."
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
            count_counts[
                n
            ]
            != 200
        ):
            raise RuntimeError(
                f"Count {n}: "
                f"{count_counts[n]}, "
                "expected 200."
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

    if len(
        combo_counts
    ) != 50:

        raise RuntimeError(
            "Expected 50 count-condition "
            "combinations."
        )

    if not all(
        n == 40
        for n in combo_counts.values()
    ):

        raise RuntimeError(
            "Each count-condition "
            "combination must contain "
            "exactly 40 samples."
        )

    # --------------------------------------------------------
    # Explicit historical-independence audit.
    # --------------------------------------------------------

    if (
        accepted_hashes
        &
        v1_hashes
    ):
        raise RuntimeError(
            "V1/V3 image overlap."
        )

    if (
        accepted_hashes
        &
        v2_hashes
    ):
        raise RuntimeError(
            "V2/V3 image overlap."
        )

    if (
        accepted_seeds
        &
        v1_seeds
    ):
        raise RuntimeError(
            "V1/V3 seed overlap."
        )

    if (
        accepted_seeds
        &
        v2_seeds
    ):
        raise RuntimeError(
            "V2/V3 seed overlap."
        )

    # --------------------------------------------------------
    # Save metadata only after every audit above passes.
    # --------------------------------------------------------

    with g.METADATA_PATH.open(
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

    COLLISION_LOG.write_text(
        json.dumps(
            {
                "generation_version":
                    "3.0_unique",

                "samples":
                    len(
                        records
                    ),

                "collision_rejections":
                    len(
                        collision_events
                    ),

                "events":
                    collision_events,
            },
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print(
        "\n"
        +
        "=" * 108
    )

    print(
        "V3.0 UNIQUE GENERATION COMPLETE"
    )

    print(
        "=" * 108
    )

    print(
        "Samples:",
        len(
            records
        ),
    )

    print(
        "Unique V3 image hashes:",
        len(
            accepted_hashes
        ),
    )

    print(
        "Unique V3 seeds:",
        len(
            accepted_seeds
        ),
    )

    print(
        "V1/V3 image overlap:",
        len(
            accepted_hashes
            &
            v1_hashes
        ),
    )

    print(
        "V2/V3 image overlap:",
        len(
            accepted_hashes
            &
            v2_hashes
        ),
    )

    print(
        "V1/V3 seed overlap:",
        len(
            accepted_seeds
            &
            v1_seeds
        ),
    )

    print(
        "V2/V3 seed overlap:",
        len(
            accepted_seeds
            &
            v2_seeds
        ),
    )

    print(
        "Collision rejections:",
        len(
            collision_events
        ),
    )

    print(
        "\nCondition counts:"
    )

    for c in g.CONDITIONS:
        print(
            f"  {c:16s}",
            condition_counts[
                c
            ],
        )

    print(
        "\nCount counts:"
    )

    for n in range(
        1,
        11,
    ):
        print(
            f"  {n:2d}:",
            count_counts[
                n
            ],
        )

    print(
        "\nMetadata:",
        g.METADATA_PATH,
    )

    print(
        "Collision log:",
        COLLISION_LOG,
    )

    print(
        "\nV3 FINAL-CONFIRMATION "
        "DATASET GENERATION AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
