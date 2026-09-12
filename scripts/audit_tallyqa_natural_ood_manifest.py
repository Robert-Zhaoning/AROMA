import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


CONFIG_PATH = Path(
    "configs/tallyqa_natural_ood_v1.json"
)

SOURCE_PATH = Path(
    "external/tallyqa/qa/test.json"
)

MANIFEST_PATH = Path(
    "data/tallyqa_natural_ood_v1/"
    "manifest.jsonl"
)


def sha256_file(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path)
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]


def rank_key(
    seed,
    namespace,
    *values,
):
    text = "|".join(
        [
            str(seed),
            str(namespace),
            *[
                str(v)
                for v in values
            ],
        ]
    )

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def choose_per_image(
    rows,
    subset,
    seed,
):
    grouped = defaultdict(
        list
    )

    for row in rows:
        grouped[
            str(
                row[
                    "image"
                ]
            )
        ].append(
            row
        )

    chosen = {}

    for image, rows_i in (
        grouped.items()
    ):

        chosen[
            image
        ] = sorted(
            rows_i,
            key=lambda r:
                (
                    rank_key(
                        seed,
                        "question",
                        subset,
                        image,
                        r[
                            "question_id"
                        ],
                    ),
                    int(
                        r[
                            "question_id"
                        ]
                    ),
                ),
        )[0]

    return chosen


def ranked_images(
    mapping,
    subset,
    seed,
):
    return sorted(
        mapping,
        key=lambda image:
            (
                rank_key(
                    seed,
                    "image",
                    subset,
                    image,
                ),
                image,
            ),
    )


def main():

    print("=" * 108)
    print(
        "TALLYQA NATURAL-OOD "
        "INDEPENDENT MANIFEST AUDIT"
    )
    print("=" * 108)

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    source_hash = (
        sha256_file(
            SOURCE_PATH
        )
    )

    if (
        source_hash
        !=
        config[
            "source_metadata"
        ][
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Source metadata hash mismatch."
        )

    source = json.loads(
        SOURCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    manifest = load_jsonl(
        MANIFEST_PATH
    )

    if len(
        manifest
    ) != 4000:
        raise RuntimeError(
            "Manifest does not contain "
            "4000 rows."
        )

    if len({
        r[
            "image"
        ]
        for r in manifest
    }) != 4000:
        raise RuntimeError(
            "Manifest images are not "
            "globally unique."
        )

    if len({
        int(
            r[
                "question_id"
            ]
        )
        for r in manifest
    }) != 4000:
        raise RuntimeError(
            "Question IDs are not unique."
        )

    counts = Counter(
        r[
            "subset"
        ]
        for r in manifest
    )

    if counts != {
        "simple":
            2000,
        "complex":
            2000,
    }:
        raise RuntimeError(
            f"Bad subset counts: {counts}"
        )

    if not all(
        0
        <=
        int(
            r[
                "answer"
            ]
        )
        <=
        15
        for r in manifest
    ):
        raise RuntimeError(
            "Out-of-range answer detected."
        )

    # --------------------------------------------------------
    # Verify each manifest row exists exactly in test.json.
    # --------------------------------------------------------

    by_qid = {
        int(
            r[
                "question_id"
            ]
        ):
            r
        for r in source
    }

    for row in manifest:

        qid = int(
            row[
                "question_id"
            ]
        )

        if qid not in by_qid:
            raise RuntimeError(
                f"Unknown question ID: {qid}"
            )

        src = by_qid[
            qid
        ]

        checks = [
            (
                str(
                    row[
                        "image"
                    ]
                ),
                str(
                    src[
                        "image"
                    ]
                ),
                "image",
            ),
            (
                int(
                    row[
                        "image_id"
                    ]
                ),
                int(
                    src[
                        "image_id"
                    ]
                ),
                "image_id",
            ),
            (
                str(
                    row[
                        "question"
                    ]
                ),
                str(
                    src[
                        "question"
                    ]
                ),
                "question",
            ),
            (
                int(
                    row[
                        "answer"
                    ]
                ),
                int(
                    src[
                        "answer"
                    ]
                ),
                "answer",
            ),
            (
                bool(
                    row[
                        "issimple"
                    ]
                ),
                bool(
                    src[
                        "issimple"
                    ]
                ),
                "issimple",
            ),
        ]

        for observed, expected, field in checks:
            if observed != expected:
                raise RuntimeError(
                    f"Source mismatch for "
                    f"question {qid}, "
                    f"field={field}"
                )

    # --------------------------------------------------------
    # Independently recompute the exact frozen selection.
    # --------------------------------------------------------

    eligible = [
        r
        for r in source
        if (
            0
            <=
            int(
                r[
                    "answer"
                ]
            )
            <=
            15
        )
    ]

    simple = [
        r
        for r in eligible
        if bool(
            r[
                "issimple"
            ]
        )
    ]

    complex_rows = [
        r
        for r in eligible
        if not bool(
            r[
                "issimple"
            ]
        )
    ]

    seed = int(
        config[
            "sampling"
        ][
            "random_seed"
        ]
    )

    simple_map = (
        choose_per_image(
            simple,
            "simple",
            seed,
        )
    )

    complex_map = (
        choose_per_image(
            complex_rows,
            "complex",
            seed,
        )
    )

    complex_images = (
        ranked_images(
            complex_map,
            "complex",
            seed,
        )[:2000]
    )

    complex_set = set(
        complex_images
    )

    simple_images = [
        image
        for image in ranked_images(
            simple_map,
            "simple",
            seed,
        )
        if image not in complex_set
    ][:2000]

    expected_simple = [
        int(
            simple_map[
                image
            ][
                "question_id"
            ]
        )
        for image in simple_images
    ]

    expected_complex = [
        int(
            complex_map[
                image
            ][
                "question_id"
            ]
        )
        for image in complex_images
    ]

    observed_simple = [
        int(
            r[
                "question_id"
            ]
        )
        for r in manifest
        if r[
            "subset"
        ]
        ==
        "simple"
    ]

    observed_complex = [
        int(
            r[
                "question_id"
            ]
        )
        for r in manifest
        if r[
            "subset"
        ]
        ==
        "complex"
    ]

    if (
        observed_simple
        !=
        expected_simple
    ):
        raise RuntimeError(
            "Simple deterministic "
            "selection mismatch."
        )

    if (
        observed_complex
        !=
        expected_complex
    ):
        raise RuntimeError(
            "Complex deterministic "
            "selection mismatch."
        )

    print(
        "Rows                   :",
        len(
            manifest
        ),
    )

    print(
        "Unique images          :",
        len({
            r[
                "image"
            ]
            for r in manifest
        }),
    )

    print(
        "Simple                 :",
        counts[
            "simple"
        ],
    )

    print(
        "Complex                :",
        counts[
            "complex"
        ],
    )

    print(
        "Source-row verification: PASS"
    )

    print(
        "Deterministic selection: PASS"
    )

    print(
        "Global image uniqueness: PASS"
    )

    print(
        "Frozen numeral support : PASS"
    )

    print(
        "Manifest SHA256        :",
        sha256_file(
            MANIFEST_PATH
        ),
    )

    print(
        "\nTALLYQA NATURAL-OOD "
        "MANIFEST AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
