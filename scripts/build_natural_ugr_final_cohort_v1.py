#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

PROTOCOL = (
    ROOT
    /
    "manifests/natural_ugr_v1/protocol_v1.json"
)

RESULT_FREEZE = (
    ROOT
    /
    "artifacts/natural_ugr_v1/development/"
    "router_analysis_v1/results_v1/"
    "router_results_freeze_v1.json"
)

SOURCE = (
    ROOT
    /
    "external/tallyqa/qa/test.json"
)

PRE = (
    ROOT
    /
    "artifacts/natural_ugr_v1/"
    "final/pre_generation_v1"
)

EXECUTION_FREEZE = (
    PRE
    /
    "final_generation_execution_freeze_v1.json"
)

DOWNLOADER_REF = (
    PRE
    /
    "reference/"
    "download_tallyqa_natural_confirmation_v2_images.py"
)

OOD_MANIFEST = (
    PRE
    /
    "exclusions/"
    "natural_ood_manifest.jsonl"
)

CONF_MANIFEST = (
    PRE
    /
    "exclusions/"
    "confirmation_v2_manifest.jsonl"
)

OOD_INVENTORY = (
    PRE
    /
    "exclusions/"
    "natural_ood_image_inventory.csv"
)

CONF_INVENTORY = (
    PRE
    /
    "exclusions/"
    "confirmation_v2_image_inventory.csv"
)

CALIBRATION = (
    PRE
    /
    "exclusions/"
    "k500_calibration_manifest.csv"
)


COHORT_ROOT = (
    ROOT
    /
    "artifacts/natural_ugr_v1/"
    "final/cohort_v1"
)

MANIFEST = (
    COHORT_ROOT
    /
    "final_manifest_v1.jsonl"
)

SUMMARY = (
    COHORT_ROOT
    /
    "final_manifest_summary_v1.json"
)

AUDIT = (
    COHORT_ROOT
    /
    "final_exclusion_audit_v1.json"
)

INVENTORY = (
    COHORT_ROOT
    /
    "final_image_inventory_v1.csv"
)


DATA_ROOT = (
    ROOT
    /
    "data/natural_ugr_final_v1"
)

CACHE_ROOT = (
    DATA_ROOT
    /
    "candidate_cache"
)

IMAGE_ROOT = (
    DATA_ROOT
    /
    "images"
)


EXPECTED_SOURCE_SHA256 = (
    "cd51c8a4d5a6deb0f5423c3ae2e16e7"
    "1198867d41da3b662c77631b7d26678a2"
)

EXPECTED_PROTOCOL_SHA256 = (
    "1daedfee079bc81f5c6171be6af9f4d8"
    "4175ed50eaec8043b9e3217a16e73d47"
)

EXPECTED_RESULT_FREEZE_SHA256 = (
    "516013c611d29bae65cabca6ffa3419f"
    "b551d85ffb7c0741c63a60d3dade878d"
)

EXPECTED_DOWNLOADER_SHA256 = (
    "5b2490885e35bd64daf790528ddd3992b"
    "c8348f9b0efcaffd7b9d6cce7fb57cc"
)


TARGET_SIMPLE = 4000
TARGET_COMPLEX = 4000
TARGET_TOTAL = 8000

SELECTION_SEED = 20260926

SELECTION_SALT = (
    "AROMA_NATURAL_UGR_FINAL_V1_"
    f"{SELECTION_SEED}"
)


def sha256_bytes(data: bytes) -> str:

    return hashlib.sha256(
        data
    ).hexdigest()


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def pixel_sha256(path: Path) -> str:

    with Image.open(path) as src:

        image = src.convert(
            "RGB"
        )

        return sha256_bytes(
            image.tobytes()
        )


def load_jsonl(path: Path):

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def load_csv(path: Path):

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def selection_hash(
    *,
    subset,
    question_id,
    image_id,
    image,
):

    # Answer is deliberately absent.
    text = (
        f"{SELECTION_SALT}|"
        f"{subset}|"
        f"{int(question_id)}|"
        f"{int(image_id)}|"
        f"{image}"
    )

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def verify_frozen_inputs():

    if sha256_file(
        SOURCE
    ) != EXPECTED_SOURCE_SHA256:

        raise RuntimeError(
            "Official TallyQA source drift."
        )


    if sha256_file(
        PROTOCOL
    ) != EXPECTED_PROTOCOL_SHA256:

        raise RuntimeError(
            "Natural-UGR protocol drift."
        )


    if sha256_file(
        RESULT_FREEZE
    ) != EXPECTED_RESULT_FREEZE_SHA256:

        raise RuntimeError(
            "Development result-freeze drift."
        )


    if sha256_file(
        DOWNLOADER_REF
    ) != EXPECTED_DOWNLOADER_SHA256:

        raise RuntimeError(
            "Historical downloader drift."
        )


    if not EXECUTION_FREEZE.is_file():

        raise RuntimeError(
            "Final-generation execution freeze missing."
        )


    execution = json.loads(
        EXECUTION_FREEZE.read_text(
            encoding="utf-8"
        )
    )


    generator_expected = (
        execution[
            "generator_sha256"
        ]
    )

    generator_actual = sha256_file(
        Path(__file__)
    )


    if (
        generator_actual
        !=
        generator_expected
    ):

        raise RuntimeError(
            "Frozen final generator byte drift."
        )


    for name, info in (
        execution[
            "input_files"
        ].items()
    ):

        path = (
            ROOT
            /
            info[
                "path"
            ]
        )

        actual = sha256_file(
            path
        )

        if actual != info[
            "sha256"
        ]:

            raise RuntimeError(
                "Frozen final input drift: "
                f"{name}: {path}"
            )


def load_exclusions():

    ood = load_jsonl(
        OOD_MANIFEST
    )

    conf = load_jsonl(
        CONF_MANIFEST
    )

    ood_inv = load_csv(
        OOD_INVENTORY
    )

    conf_inv = load_csv(
        CONF_INVENTORY
    )

    cal = load_csv(
        CALIBRATION
    )


    old_qids = {
        int(
            r[
                "question_id"
            ]
        )
        for r in (
            ood
            +
            conf
        )
    }


    old_images = {
        str(
            r[
                "image"
            ]
        )
        for r in (
            ood
            +
            conf
        )
    }


    old_image_ids = {
        int(
            r[
                "image_id"
            ]
        )
        for r in (
            ood
            +
            conf
        )
    }


    old_pixels = {
        str(
            r[
                "pixel_sha256"
            ]
        )
        for r in (
            ood_inv
            +
            conf_inv
        )
    }


    cal_qids = {
        int(
            r[
                "question_id"
            ]
        )
        for r in cal
    }


    if len(
        old_qids
    ) != 8000:

        raise RuntimeError(
            "Pre-F1 question universe != 8000."
        )


    if len(
        old_images
    ) != 8000:

        raise RuntimeError(
            "Pre-F1 image-path universe != 8000."
        )


    if len(
        old_image_ids
    ) != 8000:

        raise RuntimeError(
            "Pre-F1 image-id universe != 8000."
        )


    if len(
        old_pixels
    ) != 8000:

        raise RuntimeError(
            "Pre-F1 pixel universe != 8000."
        )


    if not cal_qids <= old_qids:

        raise RuntimeError(
            "Calibration adds fresh question IDs."
        )


    return {
        "question_ids":
            old_qids,

        "images":
            old_images,

        "image_ids":
            old_image_ids,

        "pixels":
            old_pixels,
    }


def load_source():

    source = json.loads(
        SOURCE.read_text(
            encoding="utf-8"
        )
    )


    if len(source) != 38589:

        raise RuntimeError(
            "Official TallyQA source N != 38589."
        )


    return source


def protocol_only():

    verify_frozen_inputs()

    exclusions = load_exclusions()

    source = load_source()


    required = {
        "image",
        "answer",
        "data_source",
        "question",
        "image_id",
        "question_id",
        "issimple",
    }


    for index, row in enumerate(
        source
    ):

        missing = (
            required
            -
            set(
                row
            )
        )

        if missing:

            raise RuntimeError(
                f"Source row {index} missing "
                f"{sorted(missing)}"
            )


    print(
        "Official TallyQA rows:",
        len(source),
    )

    print(
        "Pre-F1 qids:",
        len(
            exclusions[
                "question_ids"
            ]
        ),
    )

    print(
        "Pre-F1 image ids:",
        len(
            exclusions[
                "image_ids"
            ]
        ),
    )

    print(
        "Pre-F1 pixels:",
        len(
            exclusions[
                "pixels"
            ]
        ),
    )

    print(
        "Target:",
        TARGET_TOTAL,
        "=",
        TARGET_SIMPLE,
        "simple +",
        TARGET_COMPLEX,
        "complex",
    )

    print(
        "Selection seed:",
        SELECTION_SEED,
    )

    print(
        "Selection salt:",
        SELECTION_SALT,
    )

    print(
        "Answer participates in ordering:",
        False,
    )

    print()
    print(
        "FINAL GENERATOR PROTOCOL-ONLY: PASS"
    )

    print(
        "No final candidate was ranked."
    )

    print(
        "No final sample identity was generated."
    )

    print(
        "No image was downloaded."
    )

    print(
        "No model inference was run."
    )


def build_candidates(
    source,
    exclusions,
):

    candidates = []

    rejected = {
        "pre_f1_question_id":
            0,

        "pre_f1_image":
            0,

        "pre_f1_image_id":
            0,
    }


    for source_index, row in enumerate(
        source
    ):

        qid = int(
            row[
                "question_id"
            ]
        )

        image = str(
            row[
                "image"
            ]
        )

        image_id = int(
            row[
                "image_id"
            ]
        )

        issimple = bool(
            row[
                "issimple"
            ]
        )

        subset = (
            "simple"
            if issimple
            else
            "complex"
        )


        if qid in exclusions[
            "question_ids"
        ]:

            rejected[
                "pre_f1_question_id"
            ] += 1

            continue


        if image in exclusions[
            "images"
        ]:

            rejected[
                "pre_f1_image"
            ] += 1

            continue


        if image_id in exclusions[
            "image_ids"
        ]:

            rejected[
                "pre_f1_image_id"
            ] += 1

            continue


        rank = selection_hash(
            subset=subset,
            question_id=qid,
            image_id=image_id,
            image=image,
        )


        candidates.append(
            {
                "source_index":
                    source_index,

                "question_id":
                    qid,

                "image":
                    image,

                "image_id":
                    image_id,

                "issimple":
                    issimple,

                "subset":
                    subset,

                "selection_hash":
                    rank,
            }
        )


    candidates.sort(
        key=lambda x: (
            x[
                "selection_hash"
            ],
            x[
                "question_id"
            ],
        )
    )


    return (
        candidates,
        rejected,
    )


def load_historical_downloader():

    spec = (
        importlib.util
        .spec_from_file_location(
            "natural_ugr_historical_downloader",
            DOWNLOADER_REF,
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):

        raise RuntimeError(
            "Could not import historical downloader."
        )


    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )


    module.IMAGE_ROOT = (
        CACHE_ROOT
    )


    return module


def ensure_selected_copy(
    relative_path,
):

    src = (
        CACHE_ROOT
        /
        relative_path
    )

    dst = (
        IMAGE_ROOT
        /
        relative_path
    )


    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if dst.exists():

        if sha256_file(
            dst
        ) != sha256_file(
            src
        ):

            raise RuntimeError(
                "Existing selected-image byte mismatch: "
                f"{relative_path}"
            )

        return dst


    shutil.copy2(
        src,
        dst,
    )


    if sha256_file(
        dst
    ) != sha256_file(
        src
    ):

        raise RuntimeError(
            "Selected-image copy verification failed: "
            f"{relative_path}"
        )


    return dst


def write_json_atomic(
    path,
    obj,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    tmp.replace(
        path
    )


def write_jsonl_atomic(
    path,
    records,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in records:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                +
                "\n"
            )

    tmp.replace(
        path
    )


def write_inventory_atomic(
    path,
    records,
):

    fields = [
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


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )


    with tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            records
        )


    tmp.replace(
        path
    )


def run(
    *,
    workers,
    chunk_size,
):

    verify_frozen_inputs()

    if MANIFEST.exists():

        raise RuntimeError(
            "Final cohort manifest already exists. "
            "Formal cohort generation is one-shot."
        )


    source = load_source()

    exclusions = load_exclusions()


    candidates, metadata_rejections = (
        build_candidates(
            source,
            exclusions,
        )
    )


    print(
        "Candidate rows after metadata exclusions:",
        len(candidates),
    )


    quota = {
        "simple":
            TARGET_SIMPLE,

        "complex":
            TARGET_COMPLEX,
    }


    selected = []

    selected_qids = set()
    selected_images = set()
    selected_image_ids = set()
    selected_pixels = set()


    rejection = {
        **metadata_rejections,

        "new_duplicate_image":
            0,

        "new_duplicate_image_id":
            0,

        "pre_f1_pixel":
            0,

        "new_duplicate_pixel":
            0,
    }


    downloader = (
        load_historical_downloader()
    )


    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    done = False


    for chunk_start in range(
        0,
        len(candidates),
        chunk_size,
    ):

        if done:
            break


        chunk = candidates[
            chunk_start:
            chunk_start
            +
            chunk_size
        ]


        requested = sorted({
            c[
                "image"
            ]
            for c in chunk
            if quota[
                c[
                    "subset"
                ]
            ] > 0
        })


        results = {}


        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {
                executor.submit(
                    downloader.download_one,
                    image,
                ):
                    image
                for image in requested
            }


            for future in as_completed(
                futures
            ):

                image = futures[
                    future
                ]

                results[
                    image
                ] = future.result()


        for rank_index, candidate in enumerate(
            chunk,
            start=chunk_start,
        ):

            subset = candidate[
                "subset"
            ]


            if quota[
                subset
            ] <= 0:

                continue


            qid = candidate[
                "question_id"
            ]

            image = candidate[
                "image"
            ]

            image_id = candidate[
                "image_id"
            ]


            if qid in selected_qids:

                raise RuntimeError(
                    "Duplicate source question_id "
                    "encountered during deterministic stream."
                )


            if image in selected_images:

                rejection[
                    "new_duplicate_image"
                ] += 1

                continue


            if image_id in selected_image_ids:

                rejection[
                    "new_duplicate_image_id"
                ] += 1

                continue


            download_result = results.get(
                image
            )


            if download_result is None:

                raise RuntimeError(
                    "Candidate image was not downloaded: "
                    f"{image}"
                )


            if download_result[
                "status"
            ] not in {
                "downloaded",
                "existing",
            }:

                raise RuntimeError(
                    "Technical image acquisition failure "
                    "for deterministic candidate: "
                    f"{image}: "
                    f"{download_result}"
                )


            cached = (
                CACHE_ROOT
                /
                image
            )


            if not cached.is_file():

                raise RuntimeError(
                    f"Downloaded candidate missing: {cached}"
                )


            pixel = pixel_sha256(
                cached
            )


            if pixel in exclusions[
                "pixels"
            ]:

                rejection[
                    "pre_f1_pixel"
                ] += 1

                continue


            if pixel in selected_pixels:

                rejection[
                    "new_duplicate_pixel"
                ] += 1

                continue


            # -------------------------------------------------
            # Only AFTER metadata ordering + image/pixel gates
            # do we read the answer, matching Confirmation-v2.
            # -------------------------------------------------

            source_row = source[
                candidate[
                    "source_index"
                ]
            ]


            answer = int(
                source_row[
                    "answer"
                ]
            )


            if not (
                0
                <=
                answer
                <=
                15
            ):

                raise RuntimeError(
                    "Historical Confirmation-v2 "
                    "answer-validity gate failed AFTER "
                    "selection eligibility. "
                    f"question_id={qid}, answer={answer}. "
                    "Do not skip/top-up based on answer."
                )


            selected_path = (
                ensure_selected_copy(
                    image
                )
            )


            selected.append(
                {
                    "manifest_index":
                        len(selected),

                    "selection_rank":
                        rank_index,

                    "selection_hash":
                        candidate[
                            "selection_hash"
                        ],

                    "question_id":
                        qid,

                    "image":
                        image,

                    "image_id":
                        image_id,

                    "subset":
                        subset,

                    "issimple":
                        bool(
                            source_row[
                                "issimple"
                            ]
                        ),

                    "data_source":
                        str(
                            source_row[
                                "data_source"
                            ]
                        ),

                    "question":
                        str(
                            source_row[
                                "question"
                            ]
                        ),

                    "answer":
                        answer,

                    "pixel_sha256":
                        pixel,
                }
            )


            selected_qids.add(
                qid
            )

            selected_images.add(
                image
            )

            selected_image_ids.add(
                image_id
            )

            selected_pixels.add(
                pixel
            )


            quota[
                subset
            ] -= 1


            if (
                quota[
                    "simple"
                ] == 0
                and
                quota[
                    "complex"
                ] == 0
            ):

                done = True

                break


        print(
            "stream processed:",
            min(
                chunk_start
                +
                len(chunk),
                len(candidates),
            ),
            "/",
            len(candidates),
            "| selected=",
            len(selected),
            "| remaining=",
            quota,
        )


    if quota != {
        "simple":
            0,

        "complex":
            0,
    }:

        raise RuntimeError(
            f"Final quotas incomplete: {quota}"
        )


    if len(selected) != TARGET_TOTAL:

        raise RuntimeError(
            f"Selected N != {TARGET_TOTAL}: "
            f"{len(selected)}"
        )


    if len(
        selected_qids
    ) != TARGET_TOTAL:

        raise RuntimeError(
            "Final question IDs not unique."
        )


    if len(
        selected_images
    ) != TARGET_TOTAL:

        raise RuntimeError(
            "Final image paths not unique."
        )


    if len(
        selected_image_ids
    ) != TARGET_TOTAL:

        raise RuntimeError(
            "Final image IDs not unique."
        )


    if len(
        selected_pixels
    ) != TARGET_TOTAL:

        raise RuntimeError(
            "Final pixel hashes not unique."
        )


    if (
        selected_qids
        &
        exclusions[
            "question_ids"
        ]
    ):

        raise RuntimeError(
            "Pre-F1 question overlap."
        )


    if (
        selected_images
        &
        exclusions[
            "images"
        ]
    ):

        raise RuntimeError(
            "Pre-F1 image-path overlap."
        )


    if (
        selected_image_ids
        &
        exclusions[
            "image_ids"
        ]
    ):

        raise RuntimeError(
            "Pre-F1 image-id overlap."
        )


    if (
        selected_pixels
        &
        exclusions[
            "pixels"
        ]
    ):

        raise RuntimeError(
            "Pre-F1 pixel overlap."
        )


    inventory = []


    for row in selected:

        path = (
            IMAGE_ROOT
            /
            row[
                "image"
            ]
        )


        with Image.open(
            path
        ) as src:

            width, height = (
                src.size
            )


        inventory.append(
            {
                "image":
                    row[
                        "image"
                    ],

                "question_id":
                    row[
                        "question_id"
                    ],

                "subset":
                    row[
                        "subset"
                    ],

                "answer":
                    row[
                        "answer"
                    ],

                "data_source":
                    row[
                        "data_source"
                    ],

                "width":
                    int(
                        width
                    ),

                "height":
                    int(
                        height
                    ),

                "file_bytes":
                    path.stat().st_size,

                "file_sha256":
                    sha256_file(
                        path
                    ),

                "pixel_sha256":
                    pixel_sha256(
                        path
                    ),
            }
        )


    simple_n = sum(
        r[
            "subset"
        ]
        ==
        "simple"
        for r in selected
    )

    complex_n = sum(
        r[
            "subset"
        ]
        ==
        "complex"
        for r in selected
    )


    if simple_n != TARGET_SIMPLE:

        raise RuntimeError(
            "Final simple count drift."
        )


    if complex_n != TARGET_COMPLEX:

        raise RuntimeError(
            "Final complex count drift."
        )


    summary = {
        "artifact":
            "Natural UGR untouched final source cohort v1",

        "status":
            (
                "GENERATED AFTER DEVELOPMENT ROUTER "
                "AND COMPARATOR FREEZE"
            ),

        "source":
            "official TallyQA test.json",

        "source_sha256":
            EXPECTED_SOURCE_SHA256,

        "selection_seed":
            SELECTION_SEED,

        "selection_salt":
            SELECTION_SALT,

        "selection_rule":
            (
                "Global SHA256 metadata-only order; "
                "greedy 4000 Simple / 4000 Complex selection; "
                "question/image/image_id uniqueness; "
                "pixel collisions rejected in frozen order."
            ),

        "answer_used_for_ordering":
            False,

        "selected": {
            "total":
                len(selected),

            "simple":
                simple_n,

            "complex":
                complex_n,

            "unique_question_ids":
                len(
                    selected_qids
                ),

            "unique_images":
                len(
                    selected_images
                ),

            "unique_image_ids":
                len(
                    selected_image_ids
                ),

            "unique_pixel_sha256":
                len(
                    selected_pixels
                ),
        },

        "rejections":
            rejection,

        "pre_f1_exclusion_universe": {
            "question_ids":
                len(
                    exclusions[
                        "question_ids"
                    ]
                ),

            "image_paths":
                len(
                    exclusions[
                        "images"
                    ]
                ),

            "image_ids":
                len(
                    exclusions[
                        "image_ids"
                    ]
                ),

            "pixel_sha256":
                len(
                    exclusions[
                        "pixels"
                    ]
                ),
        },

        "stage1_inference_run":
            False,

        "model_outcome_observed":
            False,
    }


    audit = {
        "selected_total":
            len(selected),

        "simple":
            simple_n,

        "complex":
            complex_n,

        "question_id_overlap_pre_f1":
            len(
                selected_qids
                &
                exclusions[
                    "question_ids"
                ]
            ),

        "image_path_overlap_pre_f1":
            len(
                selected_images
                &
                exclusions[
                    "images"
                ]
            ),

        "image_id_overlap_pre_f1":
            len(
                selected_image_ids
                &
                exclusions[
                    "image_ids"
                ]
            ),

        "pixel_sha256_overlap_pre_f1":
            len(
                selected_pixels
                &
                exclusions[
                    "pixels"
                ]
            ),

        "unique_question_ids":
            len(
                selected_qids
            ),

        "unique_images":
            len(
                selected_images
            ),

        "unique_image_ids":
            len(
                selected_image_ids
            ),

        "unique_pixel_sha256":
            len(
                selected_pixels
            ),

        "all_pass":
            True,
    }


    # Remove helper-only pixel key from formal manifest.
    formal_manifest = []

    for row in selected:

        out = dict(
            row
        )

        out.pop(
            "pixel_sha256"
        )

        formal_manifest.append(
            out
        )


    write_jsonl_atomic(
        MANIFEST,
        formal_manifest,
    )

    write_inventory_atomic(
        INVENTORY,
        inventory,
    )

    summary[
        "manifest_sha256"
    ] = sha256_file(
        MANIFEST
    )

    summary[
        "image_inventory_sha256"
    ] = sha256_file(
        INVENTORY
    )


    write_json_atomic(
        SUMMARY,
        summary,
    )

    write_json_atomic(
        AUDIT,
        audit,
    )


    print()
    print(
        "=" * 78
    )

    print(
        "NATURAL UGR FINAL COHORT GENERATION: PASS"
    )

    print(
        "=" * 78
    )

    print(
        "Selected:",
        len(selected),
    )

    print(
        "Simple:",
        simple_n,
    )

    print(
        "Complex:",
        complex_n,
    )

    print(
        "question overlap:",
        audit[
            "question_id_overlap_pre_f1"
        ],
    )

    print(
        "image-id overlap:",
        audit[
            "image_id_overlap_pre_f1"
        ],
    )

    print(
        "pixel overlap:",
        audit[
            "pixel_sha256_overlap_pre_f1"
        ],
    )

    print(
        "Manifest:",
        MANIFEST,
    )

    print(
        "Manifest SHA256:",
        sha256_file(
            MANIFEST
        ),
    )

    print(
        "Stage-1 inference run:",
        False,
    )

    print(
        "Model outcome observed:",
        False,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol-only",
        action="store_true",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
    )

    args = parser.parse_args()


    if args.protocol_only:

        protocol_only()

        return


    if args.workers < 1:

        raise ValueError(
            "--workers must be >= 1."
        )


    if args.chunk_size < 1:

        raise ValueError(
            "--chunk-size must be >= 1."
        )


    run(
        workers=args.workers,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
