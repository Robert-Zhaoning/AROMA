import hashlib
import json
from collections import Counter
from pathlib import Path


V1_META = Path(
    "data/proc_count_causal_v1/metadata.jsonl"
)

V2_META = Path(
    "data/proc_count_causal_v2/metadata.jsonl"
)

V3_META = Path(
    "data/proc_count_causal_v3/metadata.jsonl"
)


def load_jsonl(path):

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


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


def hashes(records):

    return {
        sha256(
            r[
                "image_path"
            ]
        )
        for r in records
    }


def main():

    print("=" * 108)
    print(
        "PROC-COUNT-CAUSAL v3 "
        "FINAL CONFIRMATION AUDIT"
    )
    print("=" * 108)

    v1 = load_jsonl(
        V1_META
    )

    v2 = load_jsonl(
        V2_META
    )

    v3 = load_jsonl(
        V3_META
    )

    print(
        "V1 samples:",
        len(v1),
    )

    print(
        "V2 samples:",
        len(v2),
    )

    print(
        "V3 samples:",
        len(v3),
    )

    assert len(v1) == 300
    assert len(v2) == 1000
    assert len(v3) == 2000

    # --------------------------------------------------------
    # Metadata namespace.
    # --------------------------------------------------------

    assert all(
        r[
            "dataset"
        ]
        ==
        "proc_count_causal_v3"
        for r in v3
    )

    assert all(
        r[
            "generation_version"
        ]
        ==
        "3.0_unique"
        for r in v3
    )

    assert all(
        str(
            r[
                "sample_id"
            ]
        ).startswith(
            "pccv3_"
        )
        for r in v3
    )

    # --------------------------------------------------------
    # IDs.
    # --------------------------------------------------------

    ids1 = {
        r[
            "sample_id"
        ]
        for r in v1
    }

    ids2 = {
        r[
            "sample_id"
        ]
        for r in v2
    }

    ids3 = {
        r[
            "sample_id"
        ]
        for r in v3
    }

    assert len(ids3) == 2000
    assert len(
        ids1 & ids3
    ) == 0
    assert len(
        ids2 & ids3
    ) == 0

    print(
        "\nID overlap V1/V3:",
        len(
            ids1 & ids3
        ),
    )

    print(
        "ID overlap V2/V3:",
        len(
            ids2 & ids3
        ),
    )

    # --------------------------------------------------------
    # Seeds.
    # --------------------------------------------------------

    s1 = {
        int(
            r[
                "seed"
            ]
        )
        for r in v1
    }

    s2 = {
        int(
            r[
                "seed"
            ]
        )
        for r in v2
    }

    s3 = {
        int(
            r[
                "seed"
            ]
        )
        for r in v3
    }

    assert len(s3) == 2000
    assert not (
        s1 & s3
    )
    assert not (
        s2 & s3
    )

    print(
        "\nSeed overlap V1/V3:",
        len(
            s1 & s3
        ),
    )

    print(
        "Seed overlap V2/V3:",
        len(
            s2 & s3
        ),
    )

    # --------------------------------------------------------
    # Structural object metadata.
    # --------------------------------------------------------

    for r in v3:

        gt = int(
            r[
                "ground_truth"
            ]
        )

        assert (
            len(
                r[
                    "target_objects"
                ]
            )
            ==
            gt
        )

        assert (
            int(
                r[
                    "num_distractors"
                ]
            )
            ==
            len(
                r[
                    "distractor_objects"
                ]
            )
        )

        assert Path(
            r[
                "image_path"
            ]
        ).exists()

    print(
        "\nObject-count metadata audit: PASS"
    )

    # --------------------------------------------------------
    # Balance.
    # --------------------------------------------------------

    conditions = Counter(
        r[
            "condition"
        ]
        for r in v3
    )

    counts = Counter(
        int(
            r[
                "ground_truth"
            ]
        )
        for r in v3
    )

    combos = Counter(
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
        for r in v3
    )

    assert len(
        conditions
    ) == 5

    assert all(
        n == 400
        for n in conditions.values()
    )

    assert len(
        counts
    ) == 10

    assert all(
        n == 200
        for n in counts.values()
    )

    assert len(
        combos
    ) == 50

    assert all(
        n == 40
        for n in combos.values()
    )

    print(
        "\nBalance audit: PASS"
    )

    print(
        "\nCondition counts:"
    )

    for k, v in sorted(
        conditions.items()
    ):
        print(
            f"  {k:16s}: {v}"
        )

    print(
        "\nCount counts:"
    )

    for k, v in sorted(
        counts.items()
    ):
        print(
            f"  {k:2d}: {v}"
        )

    # --------------------------------------------------------
    # Pixel hashes.
    # --------------------------------------------------------

    print(
        "\nHashing 3300 images..."
    )

    h1 = hashes(
        v1
    )

    h2 = hashes(
        v2
    )

    h3 = hashes(
        v3
    )

    assert len(
        h3
    ) == 2000

    cross13 = (
        h1 & h3
    )

    cross23 = (
        h2 & h3
    )

    print(
        "Unique V3 hashes:",
        len(
            h3
        ),
    )

    print(
        "Exact V1/V3 image overlap:",
        len(
            cross13
        ),
    )

    print(
        "Exact V2/V3 image overlap:",
        len(
            cross23
        ),
    )

    assert len(
        cross13
    ) == 0

    assert len(
        cross23
    ) == 0

    # --------------------------------------------------------
    # Replicate structure.
    # --------------------------------------------------------

    reps = {
        int(
            r[
                "replicate"
            ]
        )
        for r in v3
    }

    assert reps == set(
        range(
            40
        )
    )

    print(
        "\nReplicates:",
        min(reps),
        "to",
        max(reps),
    )

    print(
        "\n" + "=" * 108
    )

    print(
        "PROC-COUNT-CAUSAL v3 "
        "FINAL AUDIT PASS"
    )

    print(
        "=" * 108
    )

    print(
        "2000 independent, balanced, "
        "pixel-unique final-confirmation samples."
    )


if __name__ == "__main__":
    main()
