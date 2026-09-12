import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")

import generate_proc_count_causal_v2 as g


V1_META = Path(
    "data/proc_count_causal_v1/metadata.jsonl"
)

LOG_DIR = Path(
    "outputs/proc_count_causal_v2"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COLLISION_LOG = (
    LOG_DIR
    / "v2_unique_collision_log.json"
)

BACKUP_META = (
    LOG_DIR
    / "metadata_v2_pre_unique.jsonl"
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def main():

    print("=" * 100)
    print(
        "PROC-COUNT-CAUSAL v2.1 "
        "UNIQUE CONFIRMATION GENERATOR"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Backup current v2 metadata before overwriting anything.
    # --------------------------------------------------------

    if g.METADATA_PATH.exists():

        shutil.copy2(
            g.METADATA_PATH,
            BACKUP_META,
        )

        print(
            "Backed up old metadata:",
            BACKUP_META,
        )

    g.IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Hash all V1 images.
    # --------------------------------------------------------

    v1 = load_jsonl(
        V1_META
    )

    assert len(v1) == 300

    v1_hashes = {
        sha256(
            Path(
                r["image_path"]
            )
        )
        for r in v1
    }

    print(
        "V1 image hashes:",
        len(v1_hashes),
    )

    # --------------------------------------------------------
    # Generate V2.
    #
    # Attempt 0 uses exactly the originally specified V2 seed.
    # Only exact pixel collisions are resampled.
    #
    # candidate_seed =
    #     base_seed + attempt * 1_000_000_000
    #
    # This keeps collision-resolution deterministic.
    # --------------------------------------------------------

    accepted_hashes = set()
    accepted_seeds = set()

    records = []
    collision_events = []

    for count in range(
        1,
        11,
    ):

        for condition_idx, condition in enumerate(
            g.CONDITIONS
        ):

            for replicate in range(
                g.REPLICATES
            ):

                base_seed = (
                    g.BASE_SEED
                    + count * 10000
                    + condition_idx * 100
                    + replicate
                )

                attempt = 0

                while True:

                    candidate_seed = (
                        base_seed
                        + attempt
                        * 1_000_000_000
                    )

                    if candidate_seed in accepted_seeds:
                        raise RuntimeError(
                            "Internal seed collision: "
                            f"{candidate_seed}"
                        )

                    record = g.draw_sample(
                        count=count,
                        condition=condition,
                        replicate=replicate,
                        seed=candidate_seed,
                    )

                    image_hash = sha256(
                        Path(
                            record[
                                "image_path"
                            ]
                        )
                    )

                    if image_hash in v1_hashes:

                        collision_events.append({
                            "sample_id":
                                record[
                                    "sample_id"
                                ],

                            "attempt":
                                attempt,

                            "rejected_seed":
                                candidate_seed,

                            "collision_type":
                                "cross_v1_v2",

                            "sha256":
                                image_hash,
                        })

                        attempt += 1

                        if attempt > 1000:
                            raise RuntimeError(
                                "Too many resampling "
                                "attempts."
                            )

                        continue

                    if image_hash in accepted_hashes:

                        collision_events.append({
                            "sample_id":
                                record[
                                    "sample_id"
                                ],

                            "attempt":
                                attempt,

                            "rejected_seed":
                                candidate_seed,

                            "collision_type":
                                "within_v2",

                            "sha256":
                                image_hash,
                        })

                        attempt += 1

                        if attempt > 1000:
                            raise RuntimeError(
                                "Too many resampling "
                                "attempts."
                            )

                        continue

                    # ----------------------------------------
                    # Accept candidate.
                    # ----------------------------------------

                    record[
                        "generation_version"
                    ] = "2.1_unique"

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
                        candidate_seed
                    )

                    break

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    assert len(records) == 1000
    assert len(accepted_hashes) == 1000
    assert len(accepted_seeds) == 1000

    ids = {
        r["sample_id"]
        for r in records
    }

    assert len(ids) == 1000

    # 200 per condition.
    cc = Counter(
        r["condition"]
        for r in records
    )

    assert all(
        cc[c] == 200
        for c in g.CONDITIONS
    )

    # 100 per count.
    nc = Counter(
        int(
            r["ground_truth"]
        )
        for r in records
    )

    assert all(
        nc[n] == 100
        for n in range(
            1,
            11,
        )
    )

    # 20 per count × condition.
    combo = Counter(
        (
            int(
                r["ground_truth"]
            ),
            r["condition"],
        )
        for r in records
    )

    assert len(combo) == 50

    assert all(
        n == 20
        for n in combo.values()
    )

    # --------------------------------------------------------
    # Save fresh metadata.
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
                + "\n"
            )

    COLLISION_LOG.write_text(
        json.dumps(
            {
                "generation_version":
                    "2.1_unique",

                "samples":
                    len(records),

                "collision_rejections":
                    len(
                        collision_events
                    ),

                "events":
                    collision_events,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("V2.1 GENERATION COMPLETE")
    print("=" * 100)

    print(
        "Samples:",
        len(records),
    )

    print(
        "Unique image hashes:",
        len(accepted_hashes),
    )

    print(
        "Collision rejections:",
        len(
            collision_events
        ),
    )

    if collision_events:

        print(
            "\nCollision events:"
        )

        for event in collision_events:
            print(
                event
            )

    print(
        "\nMetadata:",
        g.METADATA_PATH,
    )

    print(
        "Collision log:",
        COLLISION_LOG,
    )


if __name__ == "__main__":
    main()
