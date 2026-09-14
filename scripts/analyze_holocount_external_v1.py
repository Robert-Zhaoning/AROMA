#!/usr/bin/env python3

"""
Frozen statistical analysis for AROMA HoloCount external transfer v1.

This script is created and frozen BEFORE the 2480-example full HoloCount
evaluation is run.

It does not modify the model, controller, actuator, features, actions,
threshold, evaluation population, or predictions.
"""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


MANIFEST_PATH = Path(
    "manifests/generalization_extension/"
    "holocount/holocount_manifest.csv"
)

RESULT_PATH = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/"
    "full_raw_results.csv"
)

RUN_META_PATH = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/"
    "full_run_metadata.json"
)

ANALYSIS_CONFIG_PATH = Path(
    "configs/holocount_external_v1_analysis.json"
)

OUT_DIR = Path(
    "outputs/generalization_extension/"
    "holocount/external_v1/analysis_v1"
)

EXPECTED_MANIFEST_SHA256 = (
    "07e9672295230dfdad9e356102165b0190117101f865418efd1d3a6be9d403cd"
)

EXPECTED_CONTROLLER_SHA256 = (
    "d9d8ba3a24814bf71b3f4039d1effa864b57754a00539f1260a1d077c03449f5"
)

EXPECTED_MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

EXPECTED_DATASET_REVISION = (
    "f44cfe591e8f7e64b63a2fb080b98bffa10b45aa"
)

ALLOWED_ACTIONS = {
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
}

QUESTION_BOOTSTRAP_REPS = 20_000
QUESTION_BOOTSTRAP_SEED = 20260914

CLUSTER_BOOTSTRAP_REPS = 20_000
CLUSTER_BOOTSTRAP_SEED = 20260914

ALPHA = 0.05


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def git_head():
    return (
        subprocess
        .check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        )
        .strip()
    )


def load_config():
    return json.loads(
        ANALYSIS_CONFIG_PATH
        .read_text(
            encoding="utf-8"
        )
    )


def bool_series(series, name):
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    out = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if out.isna().any():
        bad = (
            series[
                out.isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"Cannot parse boolean column "
            f"{name}: {bad}"
        )

    return out.astype(bool)


def exact_mcnemar_p(
    repairs,
    breaks,
):
    discordant = int(
        repairs + breaks
    )

    if discordant == 0:
        return 1.0

    return float(
        binomtest(
            k=int(repairs),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def percentile_ci(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    lo, hi = np.quantile(
        values,
        [ALPHA / 2.0, 1.0 - ALPHA / 2.0],
    )

    return (
        float(lo),
        float(hi),
    )


def question_bootstrap(
    paired_diff,
):
    paired_diff = np.asarray(
        paired_diff,
        dtype=np.float64,
    )

    n = len(
        paired_diff
    )

    rng = np.random.default_rng(
        QUESTION_BOOTSTRAP_SEED
    )

    out = np.empty(
        QUESTION_BOOTSTRAP_REPS,
        dtype=np.float64,
    )

    batch_size = 250
    cursor = 0

    while cursor < QUESTION_BOOTSTRAP_REPS:

        m = min(
            batch_size,
            QUESTION_BOOTSTRAP_REPS - cursor,
        )

        idx = rng.integers(
            0,
            n,
            size=(m, n),
        )

        out[
            cursor:cursor + m
        ] = (
            paired_diff[
                idx
            ]
            .mean(
                axis=1
            )
        )

        cursor += m

    return percentile_ci(
        out
    )


def image_cluster_bootstrap(
    frame,
):
    # Each image-content SHA256 is one cluster.
    #
    # For each cluster we only need:
    #   sum(post_correct - baseline_correct)
    #   number of question records.
    #
    # Resampling these cluster sufficient statistics is exactly
    # equivalent to including every question in each sampled cluster.

    tmp = frame.copy()

    tmp[
        "_paired_diff"
    ] = (
        tmp[
            "post_correct"
        ].astype(int)
        -
        tmp[
            "baseline_correct"
        ].astype(int)
    )

    grouped = (
        tmp
        .groupby(
            "image_sha256",
            sort=True,
        )
        .agg(
            delta_sum=(
                "_paired_diff",
                "sum",
            ),
            n_questions=(
                "_paired_diff",
                "size",
            ),
        )
        .reset_index()
    )

    if len(grouped) != 2056:
        raise RuntimeError(
            "Expected 2056 unique image clusters, "
            f"got {len(grouped)}."
        )

    delta = (
        grouped[
            "delta_sum"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    sizes = (
        grouped[
            "n_questions"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    clusters = len(
        grouped
    )

    rng = np.random.default_rng(
        CLUSTER_BOOTSTRAP_SEED
    )

    out = np.empty(
        CLUSTER_BOOTSTRAP_REPS,
        dtype=np.float64,
    )

    batch_size = 250
    cursor = 0

    while cursor < CLUSTER_BOOTSTRAP_REPS:

        m = min(
            batch_size,
            CLUSTER_BOOTSTRAP_REPS - cursor,
        )

        idx = rng.integers(
            0,
            clusters,
            size=(m, clusters),
        )

        sampled_delta = (
            delta[
                idx
            ]
            .sum(
                axis=1
            )
        )

        sampled_n = (
            sizes[
                idx
            ]
            .sum(
                axis=1
            )
        )

        out[
            cursor:cursor + m
        ] = (
            sampled_delta
            /
            sampled_n
        )

        cursor += m

    return percentile_ci(
        out
    )


def summarize_group(
    frame,
    label,
):
    n = len(
        frame
    )

    if n == 0:
        return {
            "group": label,
            "n": 0,
        }

    baseline = (
        frame[
            "baseline_correct"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    post = (
        frame[
            "post_correct"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    alphas = (
        frame[
            "selected_alpha"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    repairs = int(
        np.sum(
            (~baseline)
            &
            post
        )
    )

    breaks = int(
        np.sum(
            baseline
            &
            (~post)
        )
    )

    baseline_correct = int(
        baseline.sum()
    )

    post_correct = int(
        post.sum()
    )

    baseline_acc = float(
        baseline.mean()
    )

    post_acc = float(
        post.mean()
    )

    gain = (
        post_acc
        -
        baseline_acc
    )

    intervention = (
        ~np.isclose(
            alphas,
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    )

    return {
        "group":
            label,

        "n":
            int(n),

        "unique_image_clusters":
            int(
                frame[
                    "image_sha256"
                ].nunique()
            ),

        "baseline_correct":
            baseline_correct,

        "post_correct":
            post_correct,

        "baseline_accuracy":
            baseline_acc,

        "post_accuracy":
            post_acc,

        "accuracy_change":
            float(gain),

        "accuracy_change_pp":
            float(
                100.0
                *
                gain
            ),

        "repairs":
            repairs,

        "breaks":
            breaks,

        "net_repairs":
            int(
                repairs
                -
                breaks
            ),

        "interventions":
            int(
                intervention.sum()
            ),

        "intervention_rate":
            float(
                intervention.mean()
            ),

        "exact_mcnemar_p":
            exact_mcnemar_p(
                repairs,
                breaks,
            ),
    }


def verify_results(
    results,
    manifest,
):
    if len(results) != 2480:
        raise RuntimeError(
            f"Expected 2480 result rows, "
            f"got {len(results)}."
        )

    if len(manifest) != 2480:
        raise RuntimeError(
            "Frozen manifest no longer contains "
            "2480 rows."
        )

    if results[
        "sample_id"
    ].nunique() != 2480:
        raise RuntimeError(
            "Result sample IDs are not unique."
        )

    if (
        results[
            "sample_id"
        ].tolist()
        !=
        manifest[
            "sample_id"
        ].tolist()
    ):
        raise RuntimeError(
            "Result row order/sample IDs do not "
            "exactly match frozen manifest."
        )

    required = {
        "sample_id",
        "split",
        "question",
        "file_name",
        "image_sha256",
        "ground_truth",
        "ground_truth_le_15",
        "baseline_prediction",
        "post_prediction",
        "selected_alpha",
        "baseline_correct",
        "post_correct",
        "repair",
        "break_case",
        "hook_calls",
    }

    missing = sorted(
        required
        -
        set(
            results.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing result columns: {missing}"
        )

    feature_cols = [
        c
        for c in results.columns
        if c.startswith(
            "feature__"
        )
    ]

    if len(
        feature_cols
    ) != 39:
        raise RuntimeError(
            f"Expected 39 feature columns, "
            f"got {len(feature_cols)}."
        )

    for col in [
        "split",
        "question",
        "file_name",
        "image_sha256",
    ]:

        if not (
            results[
                col
            ].astype(str)
            .to_numpy()
            ==
            manifest[
                col
            ].astype(str)
            .to_numpy()
        ).all():

            raise RuntimeError(
                f"Result/manifest mismatch: {col}"
            )

    gt = (
        results[
            "ground_truth"
        ]
        .astype(int)
        .to_numpy()
    )

    manifest_gt = (
        manifest[
            "answer_int"
        ]
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        gt,
        manifest_gt,
    ):
        raise RuntimeError(
            "Ground truth does not match "
            "frozen manifest."
        )

    baseline_pred = (
        results[
            "baseline_prediction"
        ]
        .astype(int)
        .to_numpy()
    )

    post_pred = (
        results[
            "post_prediction"
        ]
        .astype(int)
        .to_numpy()
    )

    if not (
        (
            baseline_pred >= 0
        )
        &
        (
            baseline_pred <= 15
        )
    ).all():
        raise RuntimeError(
            "Baseline prediction outside 0..15."
        )

    if not (
        (
            post_pred >= 0
        )
        &
        (
            post_pred <= 15
        )
    ).all():
        raise RuntimeError(
            "Post prediction outside 0..15."
        )

    selected = set(
        results[
            "selected_alpha"
        ]
        .astype(float)
        .unique()
        .tolist()
    )

    if not selected.issubset(
        ALLOWED_ACTIONS
    ):
        raise RuntimeError(
            f"Unexpected controller actions: {selected}"
        )

    results[
        "baseline_correct"
    ] = bool_series(
        results[
            "baseline_correct"
        ],
        "baseline_correct",
    )

    results[
        "post_correct"
    ] = bool_series(
        results[
            "post_correct"
        ],
        "post_correct",
    )

    results[
        "repair"
    ] = bool_series(
        results[
            "repair"
        ],
        "repair",
    )

    results[
        "break_case"
    ] = bool_series(
        results[
            "break_case"
        ],
        "break_case",
    )

    expected_baseline_correct = (
        baseline_pred
        ==
        gt
    )

    expected_post_correct = (
        post_pred
        ==
        gt
    )

    if not np.array_equal(
        results[
            "baseline_correct"
        ].to_numpy(
            dtype=bool
        ),
        expected_baseline_correct,
    ):
        raise RuntimeError(
            "baseline_correct bookkeeping mismatch."
        )

    if not np.array_equal(
        results[
            "post_correct"
        ].to_numpy(
            dtype=bool
        ),
        expected_post_correct,
    ):
        raise RuntimeError(
            "post_correct bookkeeping mismatch."
        )

    expected_repair = (
        (~expected_baseline_correct)
        &
        expected_post_correct
    )

    expected_break = (
        expected_baseline_correct
        &
        (~expected_post_correct)
    )

    if not np.array_equal(
        results[
            "repair"
        ].to_numpy(
            dtype=bool
        ),
        expected_repair,
    ):
        raise RuntimeError(
            "Repair bookkeeping mismatch."
        )

    if not np.array_equal(
        results[
            "break_case"
        ].to_numpy(
            dtype=bool
        ),
        expected_break,
    ):
        raise RuntimeError(
            "Break bookkeeping mismatch."
        )

    # Prediction-space consequence:
    # GT > 15 cannot be exactly correct under frozen 0..15 proxy.
    high = (
        gt > 15
    )

    if (
        expected_baseline_correct[
            high
        ].any()
        or
        expected_post_correct[
            high
        ].any()
    ):
        raise RuntimeError(
            "GT>15 example unexpectedly marked correct "
            "under 0..15 prediction support."
        )

    # Hook semantics.
    alpha = (
        results[
            "selected_alpha"
        ]
        .astype(float)
        .to_numpy()
    )

    hooks = (
        results[
            "hook_calls"
        ]
        .astype(int)
        .to_numpy()
    )

    noop = np.isclose(
        alpha,
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    if not (
        hooks[
            noop
        ]
        ==
        0
    ).all():
        raise RuntimeError(
            "NOOP row has hook calls."
        )

    if not (
        hooks[
            ~noop
        ]
        >
        0
    ).all():
        raise RuntimeError(
            "Non-NOOP row lacks hook call."
        )

    return results


def audit_only():
    config = load_config()

    if config[
        "status"
    ] != "FROZEN_BEFORE_FULL_HOLOCOUNT_EVALUATION":
        raise RuntimeError(
            "Analysis config status mismatch."
        )

    if config[
        "primary_population_n"
    ] != 2480:
        raise RuntimeError(
            "Primary population mismatch."
        )

    if config[
        "unique_image_clusters"
    ] != 2056:
        raise RuntimeError(
            "Image-cluster count mismatch."
        )

    if config[
        "question_bootstrap"
    ][
        "replicates"
    ] != QUESTION_BOOTSTRAP_REPS:
        raise RuntimeError(
            "Question bootstrap count mismatch."
        )

    if config[
        "image_cluster_bootstrap"
    ][
        "replicates"
    ] != CLUSTER_BOOTSTRAP_REPS:
        raise RuntimeError(
            "Cluster bootstrap count mismatch."
        )

    if sha256_file(
        MANIFEST_PATH
    ) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen HoloCount manifest hash mismatch."
        )

    if RESULT_PATH.exists():
        raise RuntimeError(
            "Full result already exists during "
            "pre-full analysis audit."
        )

    print("Analysis configuration : PASS")
    print("Frozen manifest        : PASS")
    print("Primary N              : 2480")
    print("Image clusters         : 2056")
    print("Question bootstrap     : 20,000")
    print("Cluster bootstrap      : 20,000")
    print("Exact McNemar          : FIXED")
    print("GT<=15 analysis        : PRESPECIFIED")
    print("GT>15 analysis         : PRESPECIFIED")
    print("20-subset analysis     : PRESPECIFIED")
    print()
    print("AUDIT-ONLY PASS")
    print("NO FULL HOLOCOUNT RESULT WAS READ")


def analyze():
    config = load_config()

    if not RESULT_PATH.is_file():
        raise RuntimeError(
            f"Missing full result: {RESULT_PATH}"
        )

    if not RUN_META_PATH.is_file():
        raise RuntimeError(
            f"Missing full run metadata: {RUN_META_PATH}"
        )

    if sha256_file(
        MANIFEST_PATH
    ) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen manifest hash mismatch."
        )

    run_meta = json.loads(
        RUN_META_PATH.read_text(
            encoding="utf-8"
        )
    )

    if run_meta[
        "record_count"
    ] != 2480:
        raise RuntimeError(
            "Full-run record count mismatch."
        )

    if run_meta[
        "model_revision"
    ] != EXPECTED_MODEL_REVISION:
        raise RuntimeError(
            "Model revision mismatch."
        )

    if run_meta[
        "controller_sha256"
    ] != EXPECTED_CONTROLLER_SHA256:
        raise RuntimeError(
            "Controller SHA mismatch."
        )

    if run_meta[
        "dataset_revision"
    ] != EXPECTED_DATASET_REVISION:
        raise RuntimeError(
            "Dataset revision mismatch."
        )

    result_sha = sha256_file(
        RESULT_PATH
    )

    if run_meta[
        "result_sha256"
    ] != result_sha:
        raise RuntimeError(
            "Run metadata / raw result SHA mismatch."
        )

    results = pd.read_csv(
        RESULT_PATH
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    results = verify_results(
        results,
        manifest,
    )

    # --------------------------------------------------------
    # Primary full-population analysis.
    # --------------------------------------------------------

    overall = summarize_group(
        results,
        "ALL_2480",
    )

    paired_diff = (
        results[
            "post_correct"
        ].astype(int)
        -
        results[
            "baseline_correct"
        ].astype(int)
    ).to_numpy(
        dtype=np.float64
    )

    question_lo, question_hi = (
        question_bootstrap(
            paired_diff
        )
    )

    cluster_lo, cluster_hi = (
        image_cluster_bootstrap(
            results
        )
    )

    overall[
        "question_bootstrap_ci95_low"
    ] = float(
        question_lo
    )

    overall[
        "question_bootstrap_ci95_high"
    ] = float(
        question_hi
    )

    overall[
        "question_bootstrap_ci95_low_pp"
    ] = float(
        100.0
        *
        question_lo
    )

    overall[
        "question_bootstrap_ci95_high_pp"
    ] = float(
        100.0
        *
        question_hi
    )

    overall[
        "image_cluster_bootstrap_ci95_low"
    ] = float(
        cluster_lo
    )

    overall[
        "image_cluster_bootstrap_ci95_high"
    ] = float(
        cluster_hi
    )

    overall[
        "image_cluster_bootstrap_ci95_low_pp"
    ] = float(
        100.0
        *
        cluster_lo
    )

    overall[
        "image_cluster_bootstrap_ci95_high_pp"
    ] = float(
        100.0
        *
        cluster_hi
    )

    primary_success = bool(
        (
            overall[
                "accuracy_change"
            ]
            >
            0.0
        )
        and
        (
            overall[
                "exact_mcnemar_p"
            ]
            <
            0.05
        )
    )

    cluster_robust_positive = bool(
        cluster_lo
        >
        0.0
    )

    strong_external_success = bool(
        primary_success
        and
        cluster_robust_positive
    )

    overall[
        "primary_success"
    ] = primary_success

    overall[
        "cluster_robust_positive"
    ] = cluster_robust_positive

    overall[
        "strong_external_generalization_success"
    ] = strong_external_success

    # --------------------------------------------------------
    # Prespecified count-range analyses.
    # --------------------------------------------------------

    gt = (
        results[
            "ground_truth"
        ]
        .astype(int)
    )

    range_rows = [
        summarize_group(
            results[
                gt <= 15
            ],
            "GT_LE_15",
        ),
        summarize_group(
            results[
                gt > 15
            ],
            "GT_GT_15",
        ),
    ]

    # --------------------------------------------------------
    # 20 official subsets.
    # --------------------------------------------------------

    subset_rows = []

    for subset in sorted(
        results[
            "split"
        ]
        .astype(str)
        .unique()
        .tolist()
    ):

        subset_rows.append(
            summarize_group(
                results[
                    results[
                        "split"
                    ]
                    .astype(str)
                    ==
                    subset
                ],
                subset,
            )
        )

    if len(
        subset_rows
    ) != 20:
        raise RuntimeError(
            f"Expected 20 subsets, got "
            f"{len(subset_rows)}."
        )

    # --------------------------------------------------------
    # Ground-truth count.
    # --------------------------------------------------------

    count_rows = []

    for count in sorted(
        gt.unique()
        .tolist()
    ):

        count_rows.append(
            summarize_group(
                results[
                    gt == count
                ],
                str(
                    int(
                        count
                    )
                ),
            )
        )

    # --------------------------------------------------------
    # Action distribution.
    # --------------------------------------------------------

    action_df = (
        results
        .groupby(
            "selected_alpha",
            sort=True,
        )
        .size()
        .rename(
            "count"
        )
        .reset_index()
    )

    action_df[
        "fraction"
    ] = (
        action_df[
            "count"
        ]
        /
        len(
            results
        )
    )

    # --------------------------------------------------------
    # Write analysis artifacts.
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df = pd.DataFrame([
        overall
    ])

    range_df = pd.DataFrame(
        range_rows
    )

    subset_df = pd.DataFrame(
        subset_rows
    )

    count_df = pd.DataFrame(
        count_rows
    )

    overall_df.to_csv(
        OUT_DIR
        /
        "overall_summary.csv",
        index=False,
    )

    range_df.to_csv(
        OUT_DIR
        /
        "by_count_range.csv",
        index=False,
    )

    subset_df.to_csv(
        OUT_DIR
        /
        "by_subset.csv",
        index=False,
    )

    count_df.to_csv(
        OUT_DIR
        /
        "by_ground_truth_count.csv",
        index=False,
    )

    action_df.to_csv(
        OUT_DIR
        /
        "action_distribution.csv",
        index=False,
    )

    summary_json = {
        "overall":
            overall,

        "count_range":
            range_rows,

        "primary_success":
            primary_success,

        "cluster_robust_positive":
            cluster_robust_positive,

        "strong_external_generalization_success":
            strong_external_success,
    }

    (
        OUT_DIR
        /
        "summary.json"
    ).write_text(
        json.dumps(
            summary_json,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    metadata = {
        "experiment":
            "AROMA HoloCount external transfer v1",

        "analysis_completed_utc":
            utc_now(),

        "analysis_git_head":
            git_head(),

        "raw_result_path":
            str(
                RESULT_PATH
            ),

        "raw_result_sha256":
            result_sha,

        "manifest_sha256":
            EXPECTED_MANIFEST_SHA256,

        "controller_sha256":
            EXPECTED_CONTROLLER_SHA256,

        "model_revision":
            EXPECTED_MODEL_REVISION,

        "question_bootstrap_replicates":
            QUESTION_BOOTSTRAP_REPS,

        "question_bootstrap_seed":
            QUESTION_BOOTSTRAP_SEED,

        "image_cluster_bootstrap_replicates":
            CLUSTER_BOOTSTRAP_REPS,

        "image_cluster_bootstrap_seed":
            CLUSTER_BOOTSTRAP_SEED,

        "image_cluster_key":
            "image_sha256",

        "primary_population_n":
            2480,

        "unique_image_clusters":
            2056,
    }

    (
        OUT_DIR
        /
        "analysis_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print("============================================================")
    print("HOLOCOUNT EXTERNAL V1 — FROZEN ANALYSIS COMPLETE")
    print("============================================================")
    print(
        overall_df.to_string(
            index=False
        )
    )
    print()
    print(
        "Primary success                         :",
        primary_success,
    )
    print(
        "Image-cluster robustness               :",
        cluster_robust_positive,
    )
    print(
        "Strong external generalization success :",
        strong_external_success,
    )


def main():
    parser = argparse.ArgumentParser()

    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--audit-only",
        action="store_true",
    )

    group.add_argument(
        "--analyze",
        action="store_true",
    )

    args = parser.parse_args()

    if args.audit_only:
        audit_only()
        return

    if args.analyze:
        analyze()
        return


if __name__ == "__main__":
    main()
