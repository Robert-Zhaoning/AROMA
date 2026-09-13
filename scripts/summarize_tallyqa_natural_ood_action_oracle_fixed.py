#!/usr/bin/env python3

"""
Deterministic summary-only repair for the frozen TallyQA
natural-OOD action-oracle forensic experiment.

IMPORTANT:
- Performs NO model inference.
- Does NOT alter the frozen controller.
- Does NOT alter intervention actions.
- Does NOT alter oracle definitions.
- Reads the already-completed frozen 20,000-row action matrix.
- Reproduces the pre-specified summarization logic from the
  frozen forensic runner.
- Fixes only the Pandas merge-suffix reference bug in the
  baseline reproducibility audit.
"""

from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# Frozen paths
# ============================================================

ROOT = Path(
    "outputs/tallyqa_natural_ood_v1"
)

FORENSIC_DIR = (
    ROOT
    / "forensics"
    / "action_oracle"
)

ACTION_MATRIX_PATH = (
    FORENSIC_DIR
    / "action_matrix.csv"
)

ARCHIVED_RESULTS_PATH = (
    ROOT
    / "final_frozen_controller"
    / "tallyqa_final_results.csv"
)

ACTION_SUMMARY_PATH = (
    FORENSIC_DIR
    / "action_summary.csv"
)

ORACLE_SUMMARY_PATH = (
    FORENSIC_DIR
    / "oracle_summary.csv"
)

ORACLE_SUBSET_PATH = (
    FORENSIC_DIR
    / "oracle_by_subset.csv"
)

ORACLE_ACTION_PATH = (
    FORENSIC_DIR
    / "oracle_action_distribution.csv"
)

PATCH_METADATA_PATH = (
    FORENSIC_DIR
    / "summary_patch_metadata.json"
)


# ============================================================
# Frozen intervention protocol
# ============================================================

ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]

NONNOOP_ACTIONS = [
    0.0,
    1.5,
    2.0,
    4.0,
]

EXPECTED_N = 4000
EXPECTED_MATRIX_ROWS = (
    EXPECTED_N
    *
    len(ACTIONS)
)


# ============================================================
# Utilities
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# Load frozen artifacts
# ============================================================

if not ACTION_MATRIX_PATH.exists():
    raise FileNotFoundError(
        ACTION_MATRIX_PATH
    )

if not ARCHIVED_RESULTS_PATH.exists():
    raise FileNotFoundError(
        ARCHIVED_RESULTS_PATH
    )

matrix = pd.read_csv(
    ACTION_MATRIX_PATH
)

archived = pd.read_csv(
    ARCHIVED_RESULTS_PATH
)

matrix["alpha"] = (
    matrix["alpha"]
    .astype(float)
)

archived["selected_alpha"] = (
    archived["selected_alpha"]
    .astype(float)
)


# ============================================================
# Raw action-matrix integrity audit
# ============================================================

print(
    "=" * 112
)

print(
    "RAW ACTION MATRIX INTEGRITY AUDIT"
)

print(
    "=" * 112
)

if len(matrix) != EXPECTED_MATRIX_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_MATRIX_ROWS} rows, "
        f"found {len(matrix)}."
    )

if matrix["question_id"].nunique() != EXPECTED_N:
    raise RuntimeError(
        "Expected exactly 4000 unique question IDs."
    )

duplicate_pairs = int(
    matrix.duplicated(
        [
            "question_id",
            "alpha",
        ]
    ).sum()
)

if duplicate_pairs != 0:
    raise RuntimeError(
        "Duplicate (question_id, alpha) rows found."
    )

actions_per_question = (
    matrix.groupby(
        "question_id"
    )["alpha"]
    .nunique()
)

if not (
    actions_per_question
    ==
    len(ACTIONS)
).all():
    raise RuntimeError(
        "At least one question does not contain "
        "exactly five unique actions."
    )

for alpha in ACTIONS:

    n_alpha = int(
        np.isclose(
            matrix["alpha"],
            alpha,
        ).sum()
    )

    if n_alpha != EXPECTED_N:
        raise RuntimeError(
            f"alpha={alpha} has {n_alpha} rows "
            "instead of 4000."
        )

print(
    "Rows:",
    len(matrix),
)

print(
    "Unique question IDs:",
    matrix[
        "question_id"
    ].nunique(),
)

print(
    "Duplicate (question_id, alpha):",
    duplicate_pairs,
)

print(
    "Five actions per question: PASS"
)

print(
    "RAW ACTION MATRIX INTEGRITY: PASS"
)


# ============================================================
# Fresh alpha=1 baseline
# ============================================================

baseline = (
    matrix[
        np.isclose(
            matrix["alpha"],
            1.0,
        )
    ]
    .copy()
)

if len(baseline) != EXPECTED_N:
    raise RuntimeError(
        "Could not recover exactly 4000 "
        "alpha=1 baseline rows."
    )


# ============================================================
# Baseline reproducibility audit
#
# Original frozen runner bug:
# both DataFrames contain baseline_prediction;
# merge suffixes therefore rename the columns to
# baseline_prediction_fresh and
# baseline_prediction_archived.
#
# The original downstream code incorrectly requested
# baseline_prediction without a suffix.
#
# ONLY this column reference is corrected here.
# ============================================================

archived_base = (
    archived[
        [
            "question_id",
            "baseline_prediction",
            "post_prediction",
            "selected_alpha",
            "baseline_correct",
            "post_correct",
            "subset",
        ]
    ]
    .copy()
)

merged_base = (
    baseline.merge(
        archived_base,
        on="question_id",
        how="inner",
        suffixes=(
            "_fresh",
            "_archived",
        ),
    )
)

if len(merged_base) != EXPECTED_N:
    raise RuntimeError(
        "Baseline merge did not recover "
        "exactly 4000 samples."
    )

baseline_mismatches = int(
    (
        merged_base[
            "action_prediction"
        ]
        !=
        merged_base[
            "baseline_prediction_archived"
        ]
    ).sum()
)

stored_baseline_mismatches = int(
    (
        merged_base[
            "baseline_prediction_fresh"
        ]
        !=
        merged_base[
            "baseline_prediction_archived"
        ]
    ).sum()
)

noop_internal_mismatches = int(
    (
        merged_base[
            "action_prediction"
        ]
        !=
        merged_base[
            "baseline_prediction_fresh"
        ]
    ).sum()
)

if baseline_mismatches != 0:
    raise RuntimeError(
        "Fresh alpha=1 prediction does not "
        "reproduce archived baseline."
    )

if stored_baseline_mismatches != 0:
    raise RuntimeError(
        "Fresh stored baseline_prediction does "
        "not reproduce archived baseline."
    )

if noop_internal_mismatches != 0:
    raise RuntimeError(
        "Fresh alpha=1 action_prediction differs "
        "from fresh baseline_prediction."
    )


# ============================================================
# Selected-action reproducibility audit
# ============================================================

selected = (
    archived[
        [
            "question_id",
            "selected_alpha",
            "post_prediction",
        ]
    ]
    .copy()
)

selected_check = (
    matrix.merge(
        selected,
        left_on=[
            "question_id",
            "alpha",
        ],
        right_on=[
            "question_id",
            "selected_alpha",
        ],
        how="inner",
    )
)

if len(selected_check) != EXPECTED_N:
    raise RuntimeError(
        "Could not recover exactly one selected "
        "action per sample."
    )

if (
    selected_check[
        "question_id"
    ]
    .nunique()
    !=
    EXPECTED_N
):
    raise RuntimeError(
        "Selected-action merge does not contain "
        "4000 unique question IDs."
    )

post_mismatches = int(
    (
        selected_check[
            "action_prediction"
        ]
        !=
        selected_check[
            "post_prediction"
        ]
    ).sum()
)

if post_mismatches != 0:
    raise RuntimeError(
        "Fresh action matrix does not reproduce "
        "archived selected-action predictions."
    )


print(
    "\nBaseline prediction mismatches:",
    baseline_mismatches,
)

print(
    "Fresh stored baseline mismatches:",
    stored_baseline_mismatches,
)

print(
    "Alpha=1 internal mismatches:",
    noop_internal_mismatches,
)

print(
    "Selected-action post mismatches:",
    post_mismatches,
)

print(
    "\nREPRODUCIBILITY AUDITS: PASS"
)


# ============================================================
# Fixed-action results
# Exact frozen definition
# ============================================================

action_rows = []

for alpha in ACTIONS:

    g = (
        matrix[
            np.isclose(
                matrix["alpha"],
                alpha,
            )
        ]
        .copy()
    )

    base_correct = (
        g[
            "baseline_correct"
        ]
        .astype(bool)
    )

    action_correct = (
        g[
            "action_correct"
        ]
        .astype(bool)
    )

    repairs = int(
        (
            (~base_correct)
            &
            action_correct
        ).sum()
    )

    breaks = int(
        (
            base_correct
            &
            (~action_correct)
        ).sum()
    )

    action_rows.append(
        {
            "alpha":
                float(alpha),

            "n":
                len(g),

            "baseline_accuracy":
                float(
                    base_correct.mean()
                ),

            "post_accuracy":
                float(
                    action_correct.mean()
                ),

            "accuracy_change":
                float(
                    action_correct.mean()
                    -
                    base_correct.mean()
                ),

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                repairs
                -
                breaks,

            "changed_predictions":
                int(
                    (
                        g[
                            "action_prediction"
                        ]
                        !=
                        g[
                            "baseline_prediction"
                        ]
                    ).sum()
                ),

            "mean_expected_shift":
                float(
                    g[
                        "expected_numeral_shift"
                    ]
                    .mean()
                ),
        }
    )

action_summary = pd.DataFrame(
    action_rows
)

action_summary.to_csv(
    ACTION_SUMMARY_PATH,
    index=False,
)


# ============================================================
# Frozen oracle definition
#
# Correct baseline samples retain alpha=1.
#
# Baseline-wrong sample is oracle-repairable iff at
# least one NON-NOOP frozen action produces GT.
# ============================================================

baseline_rows = (
    baseline[
        [
            "question_id",
            "subset",
            "ground_truth",
            "baseline_prediction",
            "baseline_correct",
        ]
    ]
    .copy()
)

wrong = (
    baseline_rows[
        ~baseline_rows[
            "baseline_correct"
        ]
        .astype(bool)
    ]
    .copy()
)

successful = (
    matrix[
        (
            ~np.isclose(
                matrix[
                    "alpha"
                ],
                1.0,
            )
        )
        &
        (
            matrix[
                "action_correct"
            ]
            .astype(bool)
        )
    ]
    .copy()
)

successful_by_qid = {}

for qid, g in (
    successful.groupby(
        "question_id"
    )
):

    alphas = sorted(
        float(a)
        for a in g[
            "alpha"
        ].tolist()
    )

    successful_by_qid[
        int(qid)
    ] = alphas


repairable_ids = {
    int(qid)
    for qid in wrong[
        "question_id"
    ]
    if int(qid)
    in successful_by_qid
}

unique_repaired_wrong = len(
    repairable_ids
)

baseline_correct_n = int(
    baseline_rows[
        "baseline_correct"
    ]
    .astype(bool)
    .sum()
)

oracle_correct = (
    baseline_correct_n
    +
    unique_repaired_wrong
)

oracle_accuracy = (
    oracle_correct
    /
    EXPECTED_N
)

baseline_accuracy = (
    baseline_correct_n
    /
    EXPECTED_N
)

controller_accuracy = float(
    archived[
        "post_correct"
    ]
    .astype(bool)
    .mean()
)

controller_repairs = int(
    (
        (
            ~archived[
                "baseline_correct"
            ]
            .astype(bool)
        )
        &
        archived[
            "post_correct"
        ]
        .astype(bool)
    ).sum()
)


# ============================================================
# Controller capture of oracle-repairable wrongs
# ============================================================

archived_wrong = (
    archived[
        ~archived[
            "baseline_correct"
        ]
        .astype(bool)
    ]
    .copy()
)

oracle_repairable_archived = (
    archived_wrong[
        archived_wrong[
            "question_id"
        ]
        .astype(int)
        .isin(
            repairable_ids
        )
    ]
)

controller_captured = int(
    oracle_repairable_archived[
        "post_correct"
    ]
    .astype(bool)
    .sum()
)

capture_fraction = (
    controller_captured
    /
    unique_repaired_wrong
    if unique_repaired_wrong
    else 0.0
)


oracle_summary = pd.DataFrame(
    [
        {
            "n":
                EXPECTED_N,

            "baseline_correct":
                baseline_correct_n,

            "baseline_wrong":
                len(wrong),

            "baseline_accuracy":
                baseline_accuracy,

            "unique_oracle_repairable_wrong":
                unique_repaired_wrong,

            "repairable_fraction_of_wrong":
                (
                    unique_repaired_wrong
                    /
                    len(wrong)
                ),

            "oracle_correct":
                oracle_correct,

            "oracle_accuracy":
                oracle_accuracy,

            "oracle_accuracy_gain":
                (
                    oracle_accuracy
                    -
                    baseline_accuracy
                ),

            "controller_accuracy":
                controller_accuracy,

            "controller_accuracy_change":
                (
                    controller_accuracy
                    -
                    baseline_accuracy
                ),

            "controller_repairs":
                controller_repairs,

            "controller_captured_oracle_repairs":
                controller_captured,

            "controller_repair_capture_fraction":
                capture_fraction,

            "oracle_minus_controller_accuracy":
                (
                    oracle_accuracy
                    -
                    controller_accuracy
                ),
        }
    ]
)

oracle_summary.to_csv(
    ORACLE_SUMMARY_PATH,
    index=False,
)


# ============================================================
# Oracle by frozen Simple / Complex split
# ============================================================

subset_rows = []

for subset in [
    "simple",
    "complex",
]:

    b = (
        baseline_rows[
            baseline_rows[
                "subset"
            ]
            ==
            subset
        ]
        .copy()
    )

    b_correct = (
        b[
            "baseline_correct"
        ]
        .astype(bool)
    )

    b_wrong = (
        b[
            ~b_correct
        ]
        .copy()
    )

    repairable_subset = {
        int(qid)
        for qid in b_wrong[
            "question_id"
        ]
        if int(qid)
        in repairable_ids
    }

    n_correct = int(
        b_correct.sum()
    )

    oracle_correct_subset = (
        n_correct
        +
        len(
            repairable_subset
        )
    )

    subset_rows.append(
        {
            "subset":
                subset,

            "n":
                len(b),

            "baseline_correct":
                n_correct,

            "baseline_wrong":
                len(
                    b_wrong
                ),

            "baseline_accuracy":
                (
                    n_correct
                    /
                    len(b)
                ),

            "oracle_repairable_wrong":
                len(
                    repairable_subset
                ),

            "repairable_fraction_of_wrong":
                (
                    len(
                        repairable_subset
                    )
                    /
                    len(
                        b_wrong
                    )
                    if len(
                        b_wrong
                    )
                    else 0.0
                ),

            "oracle_accuracy":
                (
                    oracle_correct_subset
                    /
                    len(b)
                ),

            "oracle_accuracy_gain":
                (
                    oracle_correct_subset
                    /
                    len(b)
                    -
                    n_correct
                    /
                    len(b)
                ),
        }
    )

oracle_subset = pd.DataFrame(
    subset_rows
)

oracle_subset.to_csv(
    ORACLE_SUBSET_PATH,
    index=False,
)


# ============================================================
# Closest-to-NOOP successful action distribution
#
# DESCRIPTIVE ONLY.
# Does not affect oracle accuracy.
# Frozen tie-breaking:
#
#   min(abs(alpha - 1), alpha)
# ============================================================

oracle_action_counter = Counter()

for qid in sorted(
    repairable_ids
):

    candidates = (
        successful_by_qid[
            qid
        ]
    )

    chosen = min(
        candidates,
        key=lambda a: (
            abs(
                a
                -
                1.0
            ),
            a,
        ),
    )

    oracle_action_counter[
        chosen
    ] += 1


oracle_action_df = pd.DataFrame(
    [
        {
            "oracle_alpha":
                alpha,

            "count":
                oracle_action_counter.get(
                    alpha,
                    0,
                ),

            "fraction_of_repairable":
                (
                    oracle_action_counter.get(
                        alpha,
                        0,
                    )
                    /
                    unique_repaired_wrong
                    if unique_repaired_wrong
                    else 0.0
                ),
        }
        for alpha in NONNOOP_ACTIONS
    ]
)

oracle_action_df.to_csv(
    ORACLE_ACTION_PATH,
    index=False,
)


# ============================================================
# Patch provenance metadata
# ============================================================

metadata = {
    "patch_type":
        "deterministic_summary_only",

    "model_inference_performed":
        False,

    "controller_modified":
        False,

    "action_space_modified":
        False,

    "oracle_definition_modified":
        False,

    "bug_fixed":
        (
            "Pandas merge suffix renamed overlapping "
            "baseline_prediction columns to "
            "baseline_prediction_fresh and "
            "baseline_prediction_archived; original "
            "summary code referenced the unsuffixed "
            "column."
        ),

    "action_matrix_path":
        str(
            ACTION_MATRIX_PATH
        ),

    "action_matrix_sha256":
        sha256_file(
            ACTION_MATRIX_PATH
        ),

    "archived_results_path":
        str(
            ARCHIVED_RESULTS_PATH
        ),

    "archived_results_sha256":
        sha256_file(
            ARCHIVED_RESULTS_PATH
        ),

    "action_matrix_rows":
        int(
            len(matrix)
        ),

    "unique_questions":
        int(
            matrix[
                "question_id"
            ]
            .nunique()
        ),

    "baseline_prediction_mismatches":
        baseline_mismatches,

    "stored_baseline_mismatches":
        stored_baseline_mismatches,

    "noop_internal_mismatches":
        noop_internal_mismatches,

    "selected_action_post_mismatches":
        post_mismatches,
}

with open(
    PATCH_METADATA_PATH,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        sort_keys=True,
    )


# ============================================================
# Print final outputs
# ============================================================

print(
    "\n" + "=" * 112
)

print(
    "FIXED ACTION RESULTS"
)

print(
    "=" * 112
)

print(
    action_summary.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "NATURAL-OOD MULTI-ACTION ORACLE CEILING"
)

print(
    "=" * 112
)

print(
    oracle_summary.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "ORACLE BY SUBSET"
)

print(
    "=" * 112
)

print(
    oracle_subset.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "ORACLE SUCCESSFUL ACTION DISTRIBUTION"
)

print(
    "=" * 112
)

print(
    oracle_action_df.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 112
)

print(
    "SUMMARY-ONLY PATCH COMPLETE"
)

print(
    "=" * 112
)

print(
    "No model inference was performed."
)

print(
    "Frozen inference runner was not modified."
)

print(
    "All reproducibility audits passed."
)

print(
    "\nSaved:"
)

for p in [
    ACTION_SUMMARY_PATH,
    ORACLE_SUMMARY_PATH,
    ORACLE_SUBSET_PATH,
    ORACLE_ACTION_PATH,
    PATCH_METADATA_PATH,
]:

    print(
        " ",
        p,
    )

