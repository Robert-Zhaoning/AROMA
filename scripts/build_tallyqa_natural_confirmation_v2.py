#!/usr/bin/env python3

"""
AROMA TallyQA Natural Confirmation v2 Manifest Builder
======================================================

Purpose
-------
Create a NEW, untouched natural-domain confirmation set after
freezing the K=500 natural-adapted controller.

Selection constraints
---------------------
- Source: official TallyQA test.json
- 4000 total samples
- 2000 Simple + 2000 Complex
- exactly one question per image
- zero overlap with the previously evaluated TallyQA-4000 on:
    * question_id
    * image
    * image_id
- deterministic SHA256-based selection
- selection does NOT use:
    * answer
    * baseline predictions
    * correctness
    * utility
    * repairability
    * controller scores

The answer is copied into the final manifest only AFTER selection.

No VLM inference is performed.
"""

from pathlib import Path
from collections import Counter
import hashlib
import json

import pandas as pd


ROOT = Path("/workspace/AromaExperiments")

SOURCE_PATH = (
    ROOT
    / "external"
    / "tallyqa"
    / "qa"
    / "test.json"
)

OLD_RESULTS_PATH = (
    ROOT
    / "outputs"
    / "tallyqa_natural_ood_v1"
    / "final_frozen_controller"
    / "tallyqa_final_results.csv"
)

CONTROLLER_PATH = (
    ROOT
    / "outputs"
    / "phase2_natural_controller_k500"
    / "final_frozen_controller"
    / "aroma_natural_utility_controller_k500.joblib"
)

CALIBRATION_MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "phase2_natural_controller_k500"
    / "final_frozen_controller"
    / "calibration_manifest.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "tallyqa_natural_confirmation_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_PATH = (
    OUT_DIR
    / "manifest.jsonl"
)

SUMMARY_PATH = (
    OUT_DIR
    / "manifest_summary.json"
)

EXCLUSION_AUDIT_PATH = (
    OUT_DIR
    / "exclusion_audit.csv"
)


EXPECTED_SOURCE_SHA256 = (
    "cd51c8a4d5a6deb0f5423c3ae2e16e71198867d41da3b662c77631b7d26678a2"
)

EXPECTED_CONTROLLER_SHA256 = (
    "d9d8ba3a24814bf71b3f4039d1effa864b57754a00539f1260a1d077c03449f5"
)

SALT = (
    "AROMA_TALLYQA_NATURAL_CONFIRMATION_V2_20260912"
)

TARGET_PER_SUBSET = 2000


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def selection_hash(
    subset,
    question_id,
    image_id,
    image,
):
    # Critically: answer is NOT included.
    text = (
        f"{SALT}|"
        f"{subset}|"
        f"{int(question_id)}|"
        f"{int(image_id)}|"
        f"{image}"
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# Artifact integrity
# ============================================================

for path in [
    SOURCE_PATH,
    OLD_RESULTS_PATH,
    CONTROLLER_PATH,
    CALIBRATION_MANIFEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)


source_sha = sha256_file(
    SOURCE_PATH
)

controller_sha = sha256_file(
    CONTROLLER_PATH
)

if source_sha != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(
        "Official TallyQA source hash mismatch:\n"
        f"expected={EXPECTED_SOURCE_SHA256}\n"
        f"actual={source_sha}"
    )

if controller_sha != EXPECTED_CONTROLLER_SHA256:
    raise RuntimeError(
        "Frozen controller hash mismatch:\n"
        f"expected={EXPECTED_CONTROLLER_SHA256}\n"
        f"actual={controller_sha}"
    )


# ============================================================
# Load source and old evaluated set
# ============================================================

with open(
    SOURCE_PATH,
    "r",
    encoding="utf-8",
) as f:
    source = json.load(f)

if len(source) != 38589:
    raise RuntimeError(
        f"Expected 38589 source rows; found {len(source)}."
    )

old = pd.read_csv(
    OLD_RESULTS_PATH
)

if len(old) != 4000:
    raise RuntimeError(
        f"Expected old result set of 4000; found {len(old)}."
    )

if old["question_id"].nunique() != 4000:
    raise RuntimeError(
        "Old results do not have 4000 unique question IDs."
    )

if old["image"].nunique() != 4000:
    raise RuntimeError(
        "Old results do not have 4000 unique images."
    )

if old["image_id"].nunique() != 4000:
    raise RuntimeError(
        "Old results do not have 4000 unique image IDs."
    )


old_question_ids = set(
    old["question_id"]
    .astype(int)
    .tolist()
)

old_images = set(
    old["image"]
    .astype(str)
    .tolist()
)

old_image_ids = set(
    old["image_id"]
    .astype(int)
    .tolist()
)


# ============================================================
# Verify K=500 calibration is contained in old 4000
# ============================================================

calibration = pd.read_csv(
    CALIBRATION_MANIFEST_PATH
)

calibration_qids = set(
    calibration["question_id"]
    .astype(int)
    .tolist()
)

if len(calibration_qids) != 500:
    raise RuntimeError(
        "Expected 500 unique calibration questions."
    )

if not calibration_qids.issubset(
    old_question_ids
):
    raise RuntimeError(
        "K=500 calibration set is not a subset "
        "of the previously evaluated TallyQA-4000."
    )


# ============================================================
# Build candidate records WITHOUT using answer for selection
# ============================================================

candidates = []

excluded_old_question = 0
excluded_old_image = 0
excluded_old_image_id = 0

for source_index, row in enumerate(source):

    required = [
        "image",
        "answer",
        "data_source",
        "question",
        "image_id",
        "question_id",
        "issimple",
    ]

    missing = [
        k for k in required
        if k not in row
    ]

    if missing:
        raise RuntimeError(
            f"Source row {source_index} missing {missing}"
        )

    qid = int(
        row["question_id"]
    )

    image = str(
        row["image"]
    )

    image_id = int(
        row["image_id"]
    )

    issimple = bool(
        row["issimple"]
    )

    subset = (
        "simple"
        if issimple
        else "complex"
    )

    if qid in old_question_ids:
        excluded_old_question += 1
        continue

    if image in old_images:
        excluded_old_image += 1
        continue

    if image_id in old_image_ids:
        excluded_old_image_id += 1
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


# Global deterministic order.
#
# We do NOT select Simple first or Complex first.
# Both subsets compete in one hash-ordered stream while their
# separate quotas are enforced.
candidates.sort(
    key=lambda x: (
        x["selection_hash"],
        x["question_id"],
    )
)


# ============================================================
# Greedy selection with global image uniqueness
# ============================================================

quota = {
    "simple":
        TARGET_PER_SUBSET,

    "complex":
        TARGET_PER_SUBSET,
}

selected = []

selected_images = set()
selected_image_ids = set()
selected_question_ids = set()

duplicate_new_image_rejections = 0
duplicate_new_image_id_rejections = 0

for candidate in candidates:

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

    if qid in selected_question_ids:
        raise RuntimeError(
            "Duplicate source question_id encountered."
        )

    if image in selected_images:
        duplicate_new_image_rejections += 1
        continue

    if image_id in selected_image_ids:
        duplicate_new_image_id_rejections += 1
        continue

    selected.append(
        candidate
    )

    selected_question_ids.add(
        qid
    )

    selected_images.add(
        image
    )

    selected_image_ids.add(
        image_id
    )

    quota[
        subset
    ] -= 1

    if (
        quota["simple"] == 0
        and
        quota["complex"] == 0
    ):
        break


if quota["simple"] != 0:
    raise RuntimeError(
        f"Simple quota incomplete: {quota['simple']}"
    )

if quota["complex"] != 0:
    raise RuntimeError(
        f"Complex quota incomplete: {quota['complex']}"
    )

if len(selected) != 4000:
    raise RuntimeError(
        f"Expected 4000 selected rows; found {len(selected)}."
    )


# ============================================================
# Construct final manifest AFTER selection
#
# Only now do we read/copy answer.
# ============================================================

manifest = []

for manifest_index, candidate in enumerate(
    selected
):

    row = source[
        candidate[
            "source_index"
        ]
    ]

    answer = int(
        row[
            "answer"
        ]
    )

    if not (
        0 <= answer <= 15
    ):
        raise RuntimeError(
            f"Unexpected answer {answer} "
            f"for question {row['question_id']}."
        )

    manifest.append(
        {
            "manifest_index":
                manifest_index,

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

            "subset":
                (
                    "simple"
                    if bool(
                        row[
                            "issimple"
                        ]
                    )
                    else "complex"
                ),

            "issimple":
                bool(
                    row[
                        "issimple"
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
                answer,
        }
    )


# ============================================================
# Final overlap / balance audits
# ============================================================

new_df = pd.DataFrame(
    manifest
)

if new_df["question_id"].nunique() != 4000:
    raise RuntimeError(
        "New manifest question IDs are not unique."
    )

if new_df["image"].nunique() != 4000:
    raise RuntimeError(
        "New manifest images are not unique."
    )

if new_df["image_id"].nunique() != 4000:
    raise RuntimeError(
        "New manifest image IDs are not unique."
    )

subset_counts = (
    new_df[
        "subset"
    ]
    .value_counts()
    .to_dict()
)

if subset_counts.get(
    "simple",
    0,
) != 2000:
    raise RuntimeError(
        "Simple count is not 2000."
    )

if subset_counts.get(
    "complex",
    0,
) != 2000:
    raise RuntimeError(
        "Complex count is not 2000."
    )


qid_overlap = (
    set(
        new_df[
            "question_id"
        ].astype(int)
    )
    &
    old_question_ids
)

image_overlap = (
    set(
        new_df[
            "image"
        ].astype(str)
    )
    &
    old_images
)

image_id_overlap = (
    set(
        new_df[
            "image_id"
        ].astype(int)
    )
    &
    old_image_ids
)

calibration_overlap = (
    set(
        new_df[
            "question_id"
        ].astype(int)
    )
    &
    calibration_qids
)

if qid_overlap:
    raise RuntimeError(
        f"Old/new question overlap: {len(qid_overlap)}"
    )

if image_overlap:
    raise RuntimeError(
        f"Old/new image overlap: {len(image_overlap)}"
    )

if image_id_overlap:
    raise RuntimeError(
        f"Old/new image_id overlap: {len(image_id_overlap)}"
    )

if calibration_overlap:
    raise RuntimeError(
        f"Calibration/new overlap: "
        f"{len(calibration_overlap)}"
    )


# ============================================================
# Write manifest
# ============================================================

with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as f:

    for row in manifest:

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


manifest_sha = sha256_file(
    MANIFEST_PATH
)


# ============================================================
# Summary
# ============================================================

def count_dict(series):

    counts = (
        series
        .value_counts()
        .sort_index()
    )

    return {
        str(k):
            int(v)
        for k, v in counts.items()
    }


def prefix_of(image):

    image = str(image)

    if "/" in image:
        return image.split(
            "/",
            1,
        )[0]

    return "UNKNOWN"


summary = {
    "protocol":
        "AROMA TallyQA Natural Confirmation v2 "
        "for frozen K=500 natural-adapted controller",

    "confirmation_status":
        "MANIFEST_FROZEN_BEFORE_ANY_V2_PREDICTIONS",

    "source_test_json_sha256":
        source_sha,

    "old_tallyqa_results_sha256":
        sha256_file(
            OLD_RESULTS_PATH
        ),

    "frozen_controller_sha256":
        controller_sha,

    "calibration_manifest_sha256":
        sha256_file(
            CALIBRATION_MANIFEST_PATH
        ),

    "manifest_sha256":
        manifest_sha,

    "selection_salt":
        SALT,

    "selection_rule":
        (
            "Global SHA256 order over metadata only; "
            "greedy selection with Simple/Complex quotas "
            "and global image/image_id uniqueness"
        ),

    "answer_used_for_selection":
        False,

    "source_rows":
        len(source),

    "old_evaluated_rows":
        len(old),

    "candidate_rows_after_old_exclusion":
        len(candidates),

    "selected":
        {
            "total":
                len(new_df),

            "simple":
                int(
                    (
                        new_df[
                            "subset"
                        ]
                        ==
                        "simple"
                    ).sum()
                ),

            "complex":
                int(
                    (
                        new_df[
                            "subset"
                        ]
                        ==
                        "complex"
                    ).sum()
                ),

            "unique_question_ids":
                int(
                    new_df[
                        "question_id"
                    ].nunique()
                ),

            "unique_images":
                int(
                    new_df[
                        "image"
                    ].nunique()
                ),

            "unique_image_ids":
                int(
                    new_df[
                        "image_id"
                    ].nunique()
                ),
        },

    "overlap_audit":
        {
            "old_new_question_id_overlap":
                len(
                    qid_overlap
                ),

            "old_new_image_overlap":
                len(
                    image_overlap
                ),

            "old_new_image_id_overlap":
                len(
                    image_id_overlap
                ),

            "calibration_new_question_overlap":
                len(
                    calibration_overlap
                ),
        },

    "answer_distribution":
        {
            "all":
                count_dict(
                    new_df[
                        "answer"
                    ]
                ),

            "simple":
                count_dict(
                    new_df[
                        new_df[
                            "subset"
                        ]
                        ==
                        "simple"
                    ][
                        "answer"
                    ]
                ),

            "complex":
                count_dict(
                    new_df[
                        new_df[
                            "subset"
                        ]
                        ==
                        "complex"
                    ][
                        "answer"
                    ]
                ),
        },

    "data_source_distribution":
        {
            "all":
                {
                    str(k):
                        int(v)
                    for k, v in
                    new_df[
                        "data_source"
                    ]
                    .value_counts()
                    .items()
                },

            "simple":
                {
                    str(k):
                        int(v)
                    for k, v in
                    new_df[
                        new_df[
                            "subset"
                        ]
                        ==
                        "simple"
                    ][
                        "data_source"
                    ]
                    .value_counts()
                    .items()
                },

            "complex":
                {
                    str(k):
                        int(v)
                    for k, v in
                    new_df[
                        new_df[
                            "subset"
                        ]
                        ==
                        "complex"
                    ][
                        "data_source"
                    ]
                    .value_counts()
                    .items()
                },
        },

    "image_prefix_distribution":
        dict(
            Counter(
                prefix_of(x)
                for x in new_df[
                    "image"
                ]
            )
        ),

    "selection_rejections":
        {
            "old_question":
                excluded_old_question,

            "old_image":
                excluded_old_image,

            "old_image_id":
                excluded_old_image_id,

            "new_duplicate_image":
                duplicate_new_image_rejections,

            "new_duplicate_image_id":
                duplicate_new_image_id_rejections,
        },
}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


# ============================================================
# Explicit exclusion audit table
# ============================================================

audit = pd.DataFrame(
    [
        {
            "check":
                "selected_rows",

            "value":
                len(new_df),

            "expected":
                4000,

            "pass":
                len(new_df)
                ==
                4000,
        },

        {
            "check":
                "simple_rows",

            "value":
                int(
                    (
                        new_df["subset"]
                        ==
                        "simple"
                    ).sum()
                ),

            "expected":
                2000,

            "pass":
                int(
                    (
                        new_df["subset"]
                        ==
                        "simple"
                    ).sum()
                )
                ==
                2000,
        },

        {
            "check":
                "complex_rows",

            "value":
                int(
                    (
                        new_df["subset"]
                        ==
                        "complex"
                    ).sum()
                ),

            "expected":
                2000,

            "pass":
                int(
                    (
                        new_df["subset"]
                        ==
                        "complex"
                    ).sum()
                )
                ==
                2000,
        },

        {
            "check":
                "unique_images",

            "value":
                int(
                    new_df[
                        "image"
                    ].nunique()
                ),

            "expected":
                4000,

            "pass":
                int(
                    new_df[
                        "image"
                    ].nunique()
                )
                ==
                4000,
        },

        {
            "check":
                "old_new_question_overlap",

            "value":
                len(qid_overlap),

            "expected":
                0,

            "pass":
                len(qid_overlap)
                ==
                0,
        },

        {
            "check":
                "old_new_image_overlap",

            "value":
                len(image_overlap),

            "expected":
                0,

            "pass":
                len(image_overlap)
                ==
                0,
        },

        {
            "check":
                "old_new_image_id_overlap",

            "value":
                len(image_id_overlap),

            "expected":
                0,

            "pass":
                len(image_id_overlap)
                ==
                0,
        },

        {
            "check":
                "calibration_new_question_overlap",

            "value":
                len(
                    calibration_overlap
                ),

            "expected":
                0,

            "pass":
                len(
                    calibration_overlap
                )
                ==
                0,
        },
    ]
)


audit.to_csv(
    EXCLUSION_AUDIT_PATH,
    index=False,
)


if not audit[
    "pass"
].all():
    raise RuntimeError(
        "Final manifest audit failed."
    )


# ============================================================
# Print
# ============================================================

print("=" * 118)
print("TALLYQA NATURAL CONFIRMATION V2 MANIFEST FREEZE")
print("=" * 118)

print("Source rows:", len(source))
print(
    "Candidates after old-set exclusion:",
    len(candidates),
)

print("\nSelected:")
print(" total:", len(new_df))
print(
    " simple:",
    int(
        (
            new_df["subset"]
            ==
            "simple"
        ).sum()
    ),
)
print(
    " complex:",
    int(
        (
            new_df["subset"]
            ==
            "complex"
        ).sum()
    ),
)
print(
    " unique images:",
    new_df["image"].nunique(),
)
print(
    " unique image IDs:",
    new_df["image_id"].nunique(),
)
print(
    " unique question IDs:",
    new_df["question_id"].nunique(),
)

print("\nOVERLAP AUDIT")
print(
    "old/new question_id overlap:",
    len(qid_overlap),
)
print(
    "old/new image overlap:",
    len(image_overlap),
)
print(
    "old/new image_id overlap:",
    len(image_id_overlap),
)
print(
    "calibration/new question overlap:",
    len(calibration_overlap),
)

print("\nDATA SOURCE")
print(
    new_df.groupby(
        "subset"
    )[
        "data_source"
    ]
    .value_counts()
    .to_string()
)

print("\nANSWER DISTRIBUTION")
print(
    new_df[
        "answer"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nIMAGE PREFIX")
print(
    new_df[
        "image"
    ]
    .map(
        prefix_of
    )
    .value_counts()
    .to_string()
)

print("\nManifest:")
print(MANIFEST_PATH)

print("\nManifest SHA256:")
print(manifest_sha)

print("\nFrozen controller SHA256:")
print(controller_sha)

print("\nSTATUS:")
print(
    "MANIFEST FROZEN BEFORE ANY "
    "NATURAL-CONFIRMATION-V2 PREDICTIONS"
)

print("\n" + "=" * 118)
print("ALL MANIFEST AUDITS: PASS")
print("=" * 118)

