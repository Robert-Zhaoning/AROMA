import hashlib
import json
from collections import Counter
from pathlib import Path


META = {
    "v1":
        Path(
            "data/proc_count_causal_v1/"
            "metadata.jsonl"
        ),

    "v2":
        Path(
            "data/proc_count_causal_v2/"
            "metadata.jsonl"
        ),

    "v3":
        Path(
            "data/proc_count_causal_v3/"
            "metadata.jsonl"
        ),

    "v4":
        Path(
            "data/proc_count_causal_v4/"
            "metadata.jsonl"
        ),
}


def sha256(
    path,
):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


def load(
    path,
):

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


def hash_set(
    records,
):

    values = set()

    for r in records:

        path = Path(
            r[
                "image_path"
            ]
        )

        if not path.exists():

            raise RuntimeError(
                f"Missing image: "
                f"{path}"
            )

        values.add(
            sha256(
                path
            )
        )

    return values


print("=" * 88)
print(
    "INDEPENDENT PROC-COUNT-CAUSAL "
    "v4 AUDIT"
)
print("=" * 88)


datasets = {
    name:
        load(path)
    for name, path
    in META.items()
}


v4 = datasets[
    "v4"
]


if len(v4) != 2000:

    raise RuntimeError(
        f"V4 N={len(v4)}"
    )


ids = {
    r[
        "sample_id"
    ]
    for r in v4
}


seeds = {
    int(
        r[
            "seed"
        ]
    )
    for r in v4
}


hashes = hash_set(
    v4
)


if len(ids) != 2000:

    raise RuntimeError(
        "V4 IDs not unique."
    )


if len(seeds) != 2000:

    raise RuntimeError(
        "V4 seeds not unique."
    )


if len(hashes) != 2000:

    raise RuntimeError(
        "V4 hashes not unique."
    )


count_counts = Counter(
    int(
        r[
            "ground_truth"
        ]
    )
    for r in v4
)


condition_counts = Counter(
    r[
        "condition"
    ]
    for r in v4
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
    for r in v4
)


if not all(
    count_counts[n] == 200
    for n in range(
        1,
        11,
    )
):

    raise RuntimeError(
        "Count balance failed."
    )


expected_conditions = {
    "row",
    "grid",
    "random_sparse",
    "dense",
    "distractors",
}


if set(
    condition_counts
) != expected_conditions:

    raise RuntimeError(
        "Condition set mismatch."
    )


if not all(
    condition_counts[c]
    == 400
    for c in expected_conditions
):

    raise RuntimeError(
        "Condition balance failed."
    )


if (
    len(combo_counts) != 50
    or
    not all(
        x == 40
        for x in (
            combo_counts
            .values()
        )
    )
):

    raise RuntimeError(
        "Count-condition balance "
        "failed."
    )


print(
    "PASS: N=2000"
)

print(
    "PASS: 2000 unique IDs"
)

print(
    "PASS: 2000 unique seeds"
)

print(
    "PASS: 2000 unique hashes"
)

print(
    "PASS: balanced counts"
)

print(
    "PASS: balanced conditions"
)

print(
    "PASS: balanced "
    "count-condition cells"
)


for historical in [
    "v1",
    "v2",
    "v3",
]:

    historical_seeds = (
        seed_set(
            datasets[
                historical
            ]
        )
    )

    historical_hashes = (
        hash_set(
            datasets[
                historical
            ]
        )
    )


    seed_overlap = (
        seeds
        &
        historical_seeds
    )

    hash_overlap = (
        hashes
        &
        historical_hashes
    )


    print()

    print(
        f"{historical}/v4 "
        f"seed overlap:",
        len(
            seed_overlap
        )
    )

    print(
        f"{historical}/v4 "
        f"hash overlap:",
        len(
            hash_overlap
        )
    )


    if seed_overlap:

        raise RuntimeError(
            f"{historical}/v4 "
            "seed overlap."
        )


    if hash_overlap:

        raise RuntimeError(
            f"{historical}/v4 "
            "image overlap."
        )


print()
print(
    "FIRST V4 SAMPLE:"
)

print(
    json.dumps(
        v4[0],
        indent=2,
    )
)


print()
print("=" * 88)
print(
    "FINAL RESULT: "
    "V4 INDEPENDENCE AUDIT PASSED"
)
print("=" * 88)

