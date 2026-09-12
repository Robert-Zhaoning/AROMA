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

OUT_ROOT = Path(
    "data/tallyqa_natural_ood_v1"
)

MANIFEST_PATH = (
    OUT_ROOT
    / "manifest.jsonl"
)

SUMMARY_DIR = Path(
    "outputs/tallyqa_natural_ood_v1"
)

SUMMARY_PATH = (
    SUMMARY_DIR
    / "manifest_summary.json"
)


def sha256_bytes(data):
    return hashlib.sha256(
        data
    ).hexdigest()


def sha256_file(path):
    return sha256_bytes(
        Path(path).read_bytes()
    )


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


def choose_question_per_image(
    rows,
    subset,
    seed,
):
    by_image = defaultdict(
        list
    )

    for row in rows:
        by_image[
            str(
                row["image"]
            )
        ].append(
            row
        )

    chosen = {}

    for image, candidates in (
        by_image.items()
    ):

        ranked = sorted(
            candidates,
            key=lambda r:
                (
                    rank_key(
                        seed,
                        "question",
                        subset,
                        image,
                        r["question_id"],
                    ),
                    int(
                        r["question_id"]
                    ),
                ),
        )

        chosen[
            image
        ] = ranked[0]

    return chosen


def rank_images(
    image_to_question,
    subset,
    seed,
):

    return sorted(
        image_to_question.keys(),
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


def answer_distribution(
    rows,
):
    c = Counter(
        int(
            r["answer"]
        )
        for r in rows
    )

    return {
        str(k):
            int(c[k])
        for k in sorted(c)
    }


def field_distribution(
    rows,
    field,
):
    c = Counter(
        str(
            r.get(
                field,
                "UNKNOWN"
            )
        )
        for r in rows
    )

    return dict(
        sorted(
            c.items()
        )
    )


def prefix_distribution(
    rows,
):
    c = Counter(
        str(
            r["image"]
        ).split(
            "/"
        )[0]
        for r in rows
    )

    return dict(
        sorted(
            c.items()
        )
    )


def main():

    print("=" * 108)
    print(
        "AROMA TALLYQA NATURAL-OOD "
        "MANIFEST BUILDER"
    )
    print("=" * 108)

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    source_hash = sha256_file(
        SOURCE_PATH
    )

    expected_hash = (
        config[
            "source_metadata"
        ][
            "sha256"
        ]
    )

    if (
        source_hash
        !=
        expected_hash
    ):
        raise RuntimeError(
            "TallyQA test.json hash "
            "does not match frozen protocol."
        )

    data = json.loads(
        SOURCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    if len(data) != 38589:
        raise RuntimeError(
            f"Expected 38589 test rows, "
            f"found {len(data)}."
        )

    required_fields = {
        "answer",
        "data_source",
        "image",
        "image_id",
        "issimple",
        "question",
        "question_id",
    }

    for row in data:

        missing = (
            required_fields
            -
            set(
                row.keys()
            )
        )

        if missing:
            raise RuntimeError(
                f"Missing fields: {missing}"
            )

    eligible = []

    for row in data:

        try:
            answer = int(
                row[
                    "answer"
                ]
            )
        except Exception as exc:
            raise RuntimeError(
                "Non-integer TallyQA answer."
            ) from exc

        if (
            0
            <= answer
            <= 15
        ):
            eligible.append(
                row
            )

    # Current official test metadata happens to
    # be entirely within the frozen numeral range.
    if len(eligible) != 38589:
        raise RuntimeError(
            "Unexpected eligible count."
        )

    simple_rows = [
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

    n_simple = int(
        config[
            "sampling"
        ][
            "simple_samples"
        ]
    )

    n_complex = int(
        config[
            "sampling"
        ][
            "complex_samples"
        ]
    )

    # --------------------------------------------------------
    # One deterministic question candidate per image.
    # --------------------------------------------------------

    simple_by_image = (
        choose_question_per_image(
            simple_rows,
            "simple",
            seed,
        )
    )

    complex_by_image = (
        choose_question_per_image(
            complex_rows,
            "complex",
            seed,
        )
    )

    overlap_images = (
        set(
            simple_by_image
        )
        &
        set(
            complex_by_image
        )
    )

    print(
        "Eligible simple unique images :",
        len(
            simple_by_image
        ),
    )

    print(
        "Eligible complex unique images:",
        len(
            complex_by_image
        ),
    )

    print(
        "Simple/complex image overlap  :",
        len(
            overlap_images
        ),
    )

    # --------------------------------------------------------
    # Frozen selection order:
    # COMPLEX first, then SIMPLE excluding selected complex.
    # --------------------------------------------------------

    complex_ranked = (
        rank_images(
            complex_by_image,
            "complex",
            seed,
        )
    )

    if len(
        complex_ranked
    ) < n_complex:
        raise RuntimeError(
            "Insufficient unique complex images."
        )

    selected_complex_images = (
        complex_ranked[
            :n_complex
        ]
    )

    selected_complex_set = set(
        selected_complex_images
    )

    simple_ranked_all = (
        rank_images(
            simple_by_image,
            "simple",
            seed,
        )
    )

    simple_ranked = [
        image
        for image in simple_ranked_all
        if image not in selected_complex_set
    ]

    if len(
        simple_ranked
    ) < n_simple:
        raise RuntimeError(
            "Insufficient globally unique "
            "simple images."
        )

    selected_simple_images = (
        simple_ranked[
            :n_simple
        ]
    )

    selected_simple = [
        simple_by_image[
            image
        ]
        for image in selected_simple_images
    ]

    selected_complex = [
        complex_by_image[
            image
        ]
        for image in selected_complex_images
    ]

    # Store Simple first only for readability.
    # This does NOT affect selection.
    final_rows = []

    for subset, rows in [
        (
            "simple",
            selected_simple,
        ),
        (
            "complex",
            selected_complex,
        ),
    ]:

        for rank, row in enumerate(
            rows
        ):

            final_rows.append({
                "manifest_index":
                    len(
                        final_rows
                    ),

                "subset":
                    subset,

                "selection_rank":
                    rank,

                "question_id":
                    int(
                        row[
                            "question_id"
                        ]
                    ),

                "image":
                    str(
                        row[
                            "image"
                        ]
                    ),

                "image_id":
                    int(
                        row[
                            "image_id"
                        ]
                    ),

                "data_source":
                    str(
                        row[
                            "data_source"
                        ]
                    ),

                "question":
                    str(
                        row[
                            "question"
                        ]
                    ),

                "answer":
                    int(
                        row[
                            "answer"
                        ]
                    ),

                "issimple":
                    bool(
                        row[
                            "issimple"
                        ]
                    ),

                "source_split":
                    "official_test",
            })

    # --------------------------------------------------------
    # Integrity.
    # --------------------------------------------------------

    if len(
        final_rows
    ) != 4000:
        raise RuntimeError(
            "Expected 4000 manifest rows."
        )

    selected_images = [
        r[
            "image"
        ]
        for r in final_rows
    ]

    if len(
        set(
            selected_images
        )
    ) != 4000:
        raise RuntimeError(
            "Global one-question-per-image "
            "constraint failed."
        )

    qids = [
        int(
            r[
                "question_id"
            ]
        )
        for r in final_rows
    ]

    if len(
        set(qids)
    ) != 4000:
        raise RuntimeError(
            "Selected question IDs are "
            "not unique."
        )

    subset_counts = Counter(
        r[
            "subset"
        ]
        for r in final_rows
    )

    if subset_counts != {
        "simple":
            2000,
        "complex":
            2000,
    }:
        raise RuntimeError(
            f"Unexpected subset counts: "
            f"{subset_counts}"
        )

    if not all(
        (
            r[
                "subset"
            ]
            ==
            "simple"
        )
        ==
        bool(
            r[
                "issimple"
            ]
        )
        for r in final_rows
    ):
        raise RuntimeError(
            "Subset / issimple mismatch."
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
        for r in final_rows
    ):
        raise RuntimeError(
            "Answer outside frozen "
            "numeral support."
        )

    # --------------------------------------------------------
    # Write manifest.
    # --------------------------------------------------------

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in final_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                +
                "\n"
            )

    manifest_hash = (
        sha256_file(
            MANIFEST_PATH
        )
    )

    config_hash = (
        sha256_file(
            CONFIG_PATH
        )
    )

    simple_selected = [
        r
        for r in final_rows
        if r[
            "subset"
        ]
        ==
        "simple"
    ]

    complex_selected = [
        r
        for r in final_rows
        if r[
            "subset"
        ]
        ==
        "complex"
    ]

    summary = {
        "protocol":
            config[
                "protocol_name"
            ],

        "source_test_json_sha256":
            source_hash,

        "protocol_json_sha256":
            config_hash,

        "manifest_sha256":
            manifest_hash,

        "sampling_seed":
            seed,

        "selection_order": [
            "complex",
            "simple"
        ],

        "source_rows":
            len(
                data
            ),

        "eligible_rows":
            len(
                eligible
            ),

        "eligible_unique_images": {
            "simple":
                len(
                    simple_by_image
                ),

            "complex":
                len(
                    complex_by_image
                ),

            "simple_complex_overlap":
                len(
                    overlap_images
                ),
        },

        "selected": {
            "total":
                len(
                    final_rows
                ),

            "simple":
                len(
                    simple_selected
                ),

            "complex":
                len(
                    complex_selected
                ),

            "unique_images":
                len(
                    set(
                        selected_images
                    )
                ),
        },

        "answer_distribution": {
            "all":
                answer_distribution(
                    final_rows
                ),

            "simple":
                answer_distribution(
                    simple_selected
                ),

            "complex":
                answer_distribution(
                    complex_selected
                ),
        },

        "data_source_distribution": {
            "all":
                field_distribution(
                    final_rows,
                    "data_source",
                ),

            "simple":
                field_distribution(
                    simple_selected,
                    "data_source",
                ),

            "complex":
                field_distribution(
                    complex_selected,
                    "data_source",
                ),
        },

        "image_prefix_distribution":
            prefix_distribution(
                final_rows
            ),
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
        "\n" + "=" * 108
    )

    print(
        "TALLYQA NATURAL-OOD MANIFEST COMPLETE"
    )

    print(
        "=" * 108
    )

    print(
        "Selected total       :",
        len(
            final_rows
        ),
    )

    print(
        "Selected Simple      :",
        len(
            simple_selected
        ),
    )

    print(
        "Selected Complex     :",
        len(
            complex_selected
        ),
    )

    print(
        "Unique selected images:",
        len(
            set(
                selected_images
            )
        ),
    )

    print(
        "\nAnswer distribution:"
    )

    print(
        json.dumps(
            summary[
                "answer_distribution"
            ],
            indent=2,
        )
    )

    print(
        "\nData-source distribution:"
    )

    print(
        json.dumps(
            summary[
                "data_source_distribution"
            ],
            indent=2,
        )
    )

    print(
        "\nImage-prefix distribution:"
    )

    print(
        json.dumps(
            summary[
                "image_prefix_distribution"
            ],
            indent=2,
        )
    )

    print(
        "\nSource SHA256  :",
        source_hash,
    )

    print(
        "Protocol SHA256:",
        config_hash,
    )

    print(
        "Manifest SHA256:",
        manifest_hash,
    )

    print(
        "\nManifest:",
        MANIFEST_PATH,
    )

    print(
        "Summary :",
        SUMMARY_PATH,
    )

    print(
        "\nMANIFEST BUILD AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
