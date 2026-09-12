import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


sys.path.insert(
    0,
    "scripts",
)


# ============================================================
# Reuse the EXACT development implementations.
# ============================================================

from run_l18h13_gain_all300 import (
    MODEL_ID,
    LAYER,
    HEAD,
    prepare_inputs,
    move_inputs,
    HeadGainModifier,
)

from expanded_numeral_metrics import (
    NUMERALS,
    numeral_token_ids,
    score_state,
)

from run_v2_baseline_utility_controller import (
    add_geometry_features,
)


# ============================================================
# Frozen paths
# ============================================================

META_PATH = Path(
    "data/proc_count_causal_v3/"
    "metadata.jsonl"
)

V3_PROTOCOL_PATH = Path(
    "configs/"
    "proc_count_causal_v3_final_confirmation.json"
)

RUNNER_PROTOCOL_PATH = Path(
    "configs/"
    "proc_count_causal_v3_final_runner.json"
)

CONTROLLER_MANIFEST_PATH = Path(
    "configs/"
    "aroma_cardinality_controller_frozen.json"
)

CONTROLLER_FREEZE_RECORD_PATH = Path(
    "configs/"
    "aroma_controller_freeze_record.json"
)

CONTROLLER_BUNDLE_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "controller/final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)


OUT_DIR = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller"
)

RESULT_PATH = (
    OUT_DIR
    / "v3_final_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "v3_final_summary.csv"
)

CONDITION_PATH = (
    OUT_DIR
    / "v3_final_by_condition.csv"
)

COUNT_PATH = (
    OUT_DIR
    / "v3_final_by_count.csv"
)

ACTION_PATH = (
    OUT_DIR
    / "v3_final_action_distribution.csv"
)

RUN_META_PATH = (
    OUT_DIR
    / "v3_final_run_metadata.json"
)

RUN_STARTED_PATH = (
    OUT_DIR
    / "v3_final_run_started.json"
)


# ============================================================
# Frozen commits
# ============================================================

CONTROLLER_FREEZE_COMMIT = (
    "c8c5ede1601b1ada4087e141d61247903a4458a7"
)

V3_PROTOCOL_FREEZE_COMMIT = (
    "582bb53fe51d8b8bbeb45b90b7d073253d4df2a1"
)


# ============================================================
# Frozen statistical settings
# ============================================================

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260912

DUMMY_GT = 1


# ============================================================
# Utilities
# ============================================================

def sha256_file(
    path,
):

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


def git_head():

    return (
        subprocess
        .check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )


def git_is_ancestor(
    commit,
):

    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return (
        result.returncode
        == 0
    )


def load_json(
    path,
):

    return json.loads(
        Path(path)
        .read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(
    path,
):

    return [
        json.loads(line)
        for line in (
            Path(path)
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    ]


def as_bool_series(
    series,
):

    if (
        series.dtype
        ==
        bool
    ):
        return (
            series
            .astype(bool)
        )

    mapped = (
        series
        .astype(str)
        .str.lower()
        .map({
            "true":
                True,

            "false":
                False,

            "1":
                True,

            "0":
                False,
        })
    )

    if mapped.isna().any():

        raise RuntimeError(
            "Failed to parse boolean "
            "column."
        )

    return mapped.astype(bool)


# ============================================================
# Exact paired McNemar
# ============================================================

def exact_mcnemar_p(
    repairs,
    breaks,
):

    repairs = int(
        repairs
    )

    breaks = int(
        breaks
    )

    n = (
        repairs
        +
        breaks
    )

    if n == 0:
        return 1.0

    k = min(
        repairs,
        breaks,
    )

    lower = sum(
        math.comb(
            n,
            i,
        )
        for i in range(
            k + 1
        )
    ) / (
        2 ** n
    )

    return min(
        1.0,
        2.0
        *
        lower,
    )


# ============================================================
# Paired bootstrap CI
# ============================================================

def paired_bootstrap_ci(
    paired_differences,
    n_bootstrap=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):

    values = np.asarray(
        paired_differences,
        dtype=np.float64,
    )

    if len(values) != 2000:

        raise RuntimeError(
            "Expected 2000 paired "
            "differences."
        )

    rng = (
        np.random
        .default_rng(
            seed
        )
    )

    boot = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    batch_size = 500

    cursor = 0

    while (
        cursor
        <
        n_bootstrap
    ):

        current = min(
            batch_size,
            n_bootstrap
            -
            cursor,
        )

        indices = (
            rng.integers(
                0,
                len(values),
                size=(
                    current,
                    len(values),
                ),
            )
        )

        boot[
            cursor:
            cursor
            +
            current
        ] = (
            values[
                indices
            ]
            .mean(
                axis=1
            )
        )

        cursor += current

    lower = float(
        np.quantile(
            boot,
            0.025,
        )
    )

    upper = float(
        np.quantile(
            boot,
            0.975,
        )
    )

    return (
        lower,
        upper,
    )


# ============================================================
# STRICT GT-FREE feature construction
# ============================================================

def state_to_features(
    state,
    feature_names,
):

    numerals = [
        int(n)
        for n in state[
            "numerals"
        ]
    ]

    if numerals != list(
        range(
            16
        )
    ):

        raise RuntimeError(
            "Expanded numeral state "
            "is not 0..15."
        )

    probs = np.asarray(
        state[
            "conditional_probs"
        ],
        dtype=np.float64,
    )

    if probs.shape != (
        16,
    ):

        raise RuntimeError(
            "Expected 16 numeral "
            "probabilities."
        )

    if not np.isfinite(
        probs
    ).all():

        raise RuntimeError(
            "Non-finite numeral "
            "probabilities."
        )

    if not np.isclose(
        probs.sum(),
        1.0,
        atol=1e-8,
    ):

        raise RuntimeError(
            "Numeral probabilities "
            "do not sum to 1."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Only GT-free fields are copied here.
    #
    # state["gt_logp"] and state["margin"] are deliberately
    # excluded.
    # --------------------------------------------------------

    row = {
        "baseline_best_numeral":
            int(
                state[
                    "best_numeral"
                ]
            ),

        "baseline_entropy":
            float(
                state[
                    "entropy"
                ]
            ),

        "baseline_entropy_norm":
            float(
                state[
                    "entropy_norm"
                ]
            ),

        "baseline_expected_numeral":
            float(
                state[
                    "expected_numeral"
                ]
            ),

        "baseline_top1_prob":
            float(
                state[
                    "top1_prob"
                ]
            ),

        "baseline_conditional_margin":
            float(
                state[
                    "conditional_margin"
                ]
            ),

        "baseline_vocab_numeral_mass":
            float(
                state[
                    "vocab_numeral_mass"
                ]
            ),
    }

    for i, numeral in enumerate(
        range(
            16
        )
    ):

        row[
            f"baseline_numprob_{numeral}"
        ] = float(
            probs[
                i
            ]
        )

    frame = pd.DataFrame([
        row
    ])

    frame = (
        add_geometry_features(
            frame
        )
    )

    missing = [
        c
        for c in feature_names
        if c not in frame.columns
    ]

    if missing:

        raise RuntimeError(
            "Frozen controller features "
            "missing from constructed row: "
            f"{missing}"
        )

    X = (
        frame[
            feature_names
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if X.shape != (
        1,
        39,
    ):

        raise RuntimeError(
            f"Unexpected controller "
            f"feature shape: {X.shape}"
        )

    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(
            "Non-finite controller "
            "feature detected."
        )

    feature_values = {
        name:
            float(
                frame.iloc[0][
                    name
                ]
            )
        for name in feature_names
    }

    return (
        X,
        feature_values,
    )


# ============================================================
# Frozen controller decision
# ============================================================

def controller_decision(
    bundle,
    X,
):

    threshold = float(
        bundle[
            "threshold"
        ]
    )

    actions = [
        float(a)
        for a in bundle[
            "nonnoop_actions"
        ]
    ]

    expected_actions = [
        0.0,
        1.5,
        2.0,
        4.0,
    ]

    if actions != expected_actions:

        raise RuntimeError(
            "Unexpected non-NOOP "
            "action ordering."
        )

    models = (
        bundle[
            "models"
        ]
    )

    scores = []

    for action in actions:

        if action not in models:

            raise RuntimeError(
                f"Missing frozen model "
                f"for alpha={action}"
            )

        value = float(
            models[
                action
            ]
            .predict(
                X
            )[0]
        )

        scores.append(
            value
        )

    scores_arr = np.asarray(
        scores,
        dtype=float,
    )

    best_idx = int(
        np.argmax(
            scores_arr
        )
    )

    best_action = float(
        actions[
            best_idx
        ]
    )

    best_score = float(
        scores_arr[
            best_idx
        ]
    )

    # Exact frozen decision rule:
    # strictly GREATER than threshold.
    if (
        best_score
        >
        threshold
    ):

        selected_alpha = (
            best_action
        )

    else:

        selected_alpha = 1.0

    score_map = {
        float(action):
            float(score)
        for action, score
        in zip(
            actions,
            scores,
        )
    }

    return (
        selected_alpha,
        best_score,
        score_map,
    )


# ============================================================
# Protocol / freeze audit
# ============================================================

def protocol_audit(
    records,
):

    print("=" * 118)
    print(
        "AROMA PROC-COUNT-CAUSAL v3 "
        "FROZEN FINAL EVALUATION"
    )
    print("=" * 118)

    if len(records) != 2000:

        raise RuntimeError(
            f"Expected 2000 V3 records, "
            f"found {len(records)}."
        )

    ids = [
        str(
            r[
                "sample_id"
            ]
        )
        for r in records
    ]

    if len(
        set(ids)
    ) != 2000:

        raise RuntimeError(
            "V3 sample IDs are not "
            "unique."
        )

    for r in records:

        if (
            r[
                "dataset"
            ]
            !=
            "proc_count_causal_v3"
        ):

            raise RuntimeError(
                "Wrong V3 dataset "
                "metadata."
            )

        if (
            r[
                "generation_version"
            ]
            !=
            "3.0_unique"
        ):

            raise RuntimeError(
                "Wrong V3 generation "
                "version."
            )

        if not str(
            r[
                "sample_id"
            ]
        ).startswith(
            "pccv3_"
        ):

            raise RuntimeError(
                "Wrong V3 sample "
                "namespace."
            )

        if not Path(
            r[
                "image_path"
            ]
        ).exists():

            raise RuntimeError(
                "Missing V3 image: "
                +
                str(
                    r[
                        "image_path"
                    ]
                )
            )

    v3_protocol = load_json(
        V3_PROTOCOL_PATH
    )

    runner_protocol = load_json(
        RUNNER_PROTOCOL_PATH
    )

    manifest = load_json(
        CONTROLLER_MANIFEST_PATH
    )

    freeze_record = load_json(
        CONTROLLER_FREEZE_RECORD_PATH
    )

    # --------------------------------------------------------
    # V3 protocol
    # --------------------------------------------------------

    if (
        v3_protocol[
            "final_dataset"
        ]
        !=
        "proc_count_causal_v3"
    ):

        raise RuntimeError(
            "V3 protocol dataset "
            "mismatch."
        )

    if (
        v3_protocol[
            "final_generation_version"
        ]
        !=
        "3.0_unique"
    ):

        raise RuntimeError(
            "V3 protocol generation "
            "version mismatch."
        )

    if int(
        v3_protocol[
            "n_samples"
        ]
    ) != 2000:

        raise RuntimeError(
            "V3 protocol sample "
            "count mismatch."
        )

    # --------------------------------------------------------
    # Runner protocol
    # --------------------------------------------------------

    if (
        runner_protocol[
            "controller_freeze_commit"
        ]
        !=
        CONTROLLER_FREEZE_COMMIT
    ):

        raise RuntimeError(
            "Controller freeze commit "
            "mismatch."
        )

    if (
        runner_protocol[
            "v3_dataset_protocol_freeze_commit"
        ]
        !=
        V3_PROTOCOL_FREEZE_COMMIT
    ):

        raise RuntimeError(
            "V3 protocol freeze commit "
            "mismatch."
        )

    if (
        runner_protocol[
            "frozen_before_v3_model_inference"
        ]
        is not True
    ):

        raise RuntimeError(
            "Runner is not marked "
            "frozen-before-inference."
        )

    measurement = (
        runner_protocol[
            "primary_measurement"
        ]
    )

    if int(
        measurement[
            "numeral_min"
        ]
    ) != 0:

        raise RuntimeError(
            "Wrong numeral minimum."
        )

    if int(
        measurement[
            "numeral_max"
        ]
    ) != 15:

        raise RuntimeError(
            "Wrong numeral maximum."
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    if (
        manifest[
            "model"
        ]
        !=
        MODEL_ID
    ):

        raise RuntimeError(
            "Frozen model ID mismatch."
        )

    actuator = (
        manifest[
            "actuator"
        ]
    )

    if int(
        actuator[
            "layer"
        ]
    ) != int(
        LAYER
    ):

        raise RuntimeError(
            "Frozen layer mismatch."
        )

    if int(
        actuator[
            "head"
        ]
    ) != int(
        HEAD
    ):

        raise RuntimeError(
            "Frozen head mismatch."
        )

    if (
        actuator[
            "head_name"
        ]
        !=
        "L18H13"
    ):

        raise RuntimeError(
            "Frozen head-name mismatch."
        )

    frozen_actions = [
        float(a)
        for a in actuator[
            "actions"
        ]
    ]

    if frozen_actions != [
        0.0,
        1.0,
        1.5,
        2.0,
        4.0,
    ]:

        raise RuntimeError(
            "Frozen action space "
            "mismatch."
        )

    controller_cfg = (
        manifest[
            "controller"
        ]
    )

    if (
        controller_cfg[
            "feature_family"
        ]
        !=
        "full_geometry"
    ):

        raise RuntimeError(
            "Frozen feature-family "
            "mismatch."
        )

    if int(
        controller_cfg[
            "feature_count"
        ]
    ) != 39:

        raise RuntimeError(
            "Frozen feature count "
            "mismatch."
        )

    if not math.isclose(
        float(
            controller_cfg[
                "ridge_alpha"
            ]
        ),
        0.01,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        raise RuntimeError(
            "Frozen Ridge alpha "
            "mismatch."
        )

    if not math.isclose(
        float(
            controller_cfg[
                "threshold"
            ]
        ),
        0.1,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        raise RuntimeError(
            "Frozen threshold "
            "mismatch."
        )

    if (
        manifest[
            "frozen_before_v3_evaluation"
        ]
        is not True
    ):

        raise RuntimeError(
            "Controller manifest is "
            "not frozen."
        )

    # --------------------------------------------------------
    # Controller file hashes from freeze record
    # --------------------------------------------------------

    hash_map = (
        freeze_record[
            "sha256"
        ]
    )

    manifest_key = str(
        CONTROLLER_MANIFEST_PATH
    )

    bundle_key = str(
        CONTROLLER_BUNDLE_PATH
    )

    if manifest_key not in hash_map:

        raise RuntimeError(
            "Controller manifest hash "
            "missing from freeze record."
        )

    if bundle_key not in hash_map:

        raise RuntimeError(
            "Controller bundle hash "
            "missing from freeze record."
        )

    actual_manifest_hash = (
        sha256_file(
            CONTROLLER_MANIFEST_PATH
        )
    )

    actual_bundle_hash = (
        sha256_file(
            CONTROLLER_BUNDLE_PATH
        )
    )

    if (
        actual_manifest_hash
        !=
        hash_map[
            manifest_key
        ]
    ):

        raise RuntimeError(
            "Frozen controller manifest "
            "hash mismatch."
        )

    if (
        actual_bundle_hash
        !=
        hash_map[
            bundle_key
        ]
    ):

        raise RuntimeError(
            "Frozen controller bundle "
            "hash mismatch."
        )

    # --------------------------------------------------------
    # Load frozen controller bundle.
    # --------------------------------------------------------

    bundle = joblib.load(
        CONTROLLER_BUNDLE_PATH
    )

    feature_names_manifest = list(
        controller_cfg[
            "feature_names"
        ]
    )

    feature_names_bundle = list(
        bundle[
            "feature_names"
        ]
    )

    if (
        feature_names_manifest
        !=
        feature_names_bundle
    ):

        raise RuntimeError(
            "Frozen feature names/order "
            "mismatch between manifest "
            "and bundle."
        )

    if len(
        feature_names_bundle
    ) != 39:

        raise RuntimeError(
            "Frozen bundle does not "
            "contain 39 features."
        )

    forbidden_fragments = [
        "ground_truth",
        "correct",
        "gt_logp",
        "oracle",
        "utility",
        "repair",
        "break",
    ]

    for name in (
        feature_names_bundle
    ):

        lower = (
            name.lower()
        )

        if any(
            fragment in lower
            for fragment in
            forbidden_fragments
        ):

            raise RuntimeError(
                "Leakage-sensitive frozen "
                f"feature detected: {name}"
            )

        if (
            name
            ==
            "baseline_margin"
        ):

            raise RuntimeError(
                "GT-dependent "
                "baseline_margin detected."
            )

    if not math.isclose(
        float(
            bundle[
                "ridge_alpha"
            ]
        ),
        0.01,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        raise RuntimeError(
            "Bundle Ridge alpha "
            "mismatch."
        )

    if not math.isclose(
        float(
            bundle[
                "threshold"
            ]
        ),
        0.1,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        raise RuntimeError(
            "Bundle threshold "
            "mismatch."
        )

    if [
        float(a)
        for a in bundle[
            "actions"
        ]
    ] != [
        0.0,
        1.0,
        1.5,
        2.0,
        4.0,
    ]:

        raise RuntimeError(
            "Bundle action-space "
            "mismatch."
        )

    # --------------------------------------------------------
    # Git ancestry
    # --------------------------------------------------------

    if not git_is_ancestor(
        CONTROLLER_FREEZE_COMMIT
    ):

        raise RuntimeError(
            "Controller freeze commit "
            "is not an ancestor of HEAD."
        )

    if not git_is_ancestor(
        V3_PROTOCOL_FREEZE_COMMIT
    ):

        raise RuntimeError(
            "V3 protocol freeze commit "
            "is not an ancestor of HEAD."
        )

    print(
        "Samples                    :",
        len(
            records
        ),
    )

    print(
        "Model                      :",
        MODEL_ID,
    )

    print(
        "Numeral space              :",
        "0..15",
    )

    print(
        "Frozen head                :",
        "L18H13",
    )

    print(
        "Frozen actions             :",
        frozen_actions,
    )

    print(
        "Frozen feature family      :",
        "full_geometry",
    )

    print(
        "Frozen feature count       :",
        len(
            feature_names_bundle
        ),
    )

    print(
        "Frozen Ridge alpha         :",
        bundle[
            "ridge_alpha"
        ],
    )

    print(
        "Frozen utility threshold   :",
        bundle[
            "threshold"
        ],
    )

    print(
        "Controller hash audit      : PASS"
    )

    print(
        "Feature leakage audit      : PASS"
    )

    print(
        "Git freeze ancestry audit  : PASS"
    )

    print(
        "V3 metadata audit          : PASS"
    )

    return (
        v3_protocol,
        runner_protocol,
        manifest,
        bundle,
        actual_manifest_hash,
        actual_bundle_hash,
    )


# ============================================================
# Resume handling
# ============================================================

def prepare_resume(
    records,
):

    if not RESULT_PATH.exists():

        return set()

    df = pd.read_csv(
        RESULT_PATH
    )

    if (
        df[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate sample IDs in "
            "partial V3 result file."
        )

    valid_ids = {
        str(
            r[
                "sample_id"
            ]
        )
        for r in records
    }

    observed = set(
        df[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    unexpected = (
        observed
        -
        valid_ids
    )

    if unexpected:

        raise RuntimeError(
            "Unexpected IDs in partial "
            "V3 result file."
        )

    if len(
        observed
    ) >= 2000:

        raise RuntimeError(
            "V3 final result file already "
            "appears complete. Do not rerun."
        )

    return observed


# ============================================================
# Final summary
# ============================================================

def summarize(
    df,
    run_metadata,
):

    if len(df) != 2000:

        raise RuntimeError(
            f"Final V3 results contain "
            f"{len(df)} rows, expected 2000."
        )

    if (
        df[
            "sample_id"
        ]
        .nunique()
        != 2000
    ):

        raise RuntimeError(
            "Final V3 sample IDs are "
            "not unique."
        )

    baseline_correct = (
        as_bool_series(
            df[
                "baseline_correct"
            ]
        )
    )

    post_correct = (
        as_bool_series(
            df[
                "post_correct"
            ]
        )
    )

    repairs = int(
        (
            (~baseline_correct)
            &
            post_correct
        ).sum()
    )

    breaks = int(
        (
            baseline_correct
            &
            (~post_correct)
        ).sum()
    )

    baseline_n = int(
        baseline_correct.sum()
    )

    post_n = int(
        post_correct.sum()
    )

    baseline_wrong = int(
        (
            ~baseline_correct
        ).sum()
    )

    baseline_accuracy = float(
        baseline_correct.mean()
    )

    post_accuracy = float(
        post_correct.mean()
    )

    accuracy_change = (
        post_accuracy
        -
        baseline_accuracy
    )

    paired_difference = (
        post_correct.astype(int)
        -
        baseline_correct.astype(int)
    ).to_numpy(
        dtype=float
    )

    ci_low, ci_high = (
        paired_bootstrap_ci(
            paired_difference
        )
    )

    p_value = (
        exact_mcnemar_p(
            repairs,
            breaks,
        )
    )

    success = bool(
        (
            accuracy_change
            >
            0
        )
        and
        (
            p_value
            <
            0.05
        )
    )

    intervention = (
        ~np.isclose(
            df[
                "selected_alpha"
            ]
            .astype(float),
            1.0,
        )
    )

    intervention_rate = float(
        intervention.mean()
    )

    overall = {
        "scope":
            "ALL2000",

        "n":
            2000,

        "baseline_correct":
            baseline_n,

        "baseline_wrong":
            baseline_wrong,

        "baseline_accuracy":
            baseline_accuracy,

        "post_correct":
            post_n,

        "post_accuracy":
            post_accuracy,

        "accuracy_change":
            accuracy_change,

        "accuracy_change_ci95_low":
            ci_low,

        "accuracy_change_ci95_high":
            ci_high,

        "repairs":
            repairs,

        "breaks":
            breaks,

        "net_repairs":
            repairs
            -
            breaks,

        "repair_rate_among_wrong":
            (
                repairs
                /
                baseline_wrong
                if baseline_wrong
                else 0.0
            ),

        "break_rate_among_correct":
            (
                breaks
                /
                baseline_n
                if baseline_n
                else 0.0
            ),

        "intervention_rate":
            intervention_rate,

        "mcnemar_exact_p":
            p_value,

        "primary_success":
            success,
    }

    summary = pd.DataFrame([
        overall
    ])

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # By condition
    # --------------------------------------------------------

    condition_rows = []

    for condition, g in (
        df.groupby(
            "condition"
        )
    ):

        bc = (
            as_bool_series(
                g[
                    "baseline_correct"
                ]
            )
        )

        pc = (
            as_bool_series(
                g[
                    "post_correct"
                ]
            )
        )

        r = int(
            (
                (~bc)
                &
                pc
            ).sum()
        )

        b = int(
            (
                bc
                &
                (~pc)
            ).sum()
        )

        base_acc = float(
            bc.mean()
        )

        post_acc = float(
            pc.mean()
        )

        condition_rows.append({
            "condition":
                condition,

            "n":
                len(g),

            "baseline_accuracy":
                base_acc,

            "post_accuracy":
                post_acc,

            "accuracy_change":
                post_acc
                -
                base_acc,

            "repairs":
                r,

            "breaks":
                b,

            "net_repairs":
                r - b,

            "mcnemar_exact_p":
                exact_mcnemar_p(
                    r,
                    b,
                ),
        })

    by_condition = pd.DataFrame(
        condition_rows
    )

    by_condition.to_csv(
        CONDITION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # By GT count — exploratory only
    # --------------------------------------------------------

    count_rows = []

    for gt, g in (
        df.groupby(
            "ground_truth"
        )
    ):

        bc = (
            as_bool_series(
                g[
                    "baseline_correct"
                ]
            )
        )

        pc = (
            as_bool_series(
                g[
                    "post_correct"
                ]
            )
        )

        r = int(
            (
                (~bc)
                &
                pc
            ).sum()
        )

        b = int(
            (
                bc
                &
                (~pc)
            ).sum()
        )

        count_rows.append({
            "ground_truth":
                int(
                    gt
                ),

            "n":
                len(g),

            "baseline_accuracy":
                float(
                    bc.mean()
                ),

            "post_accuracy":
                float(
                    pc.mean()
                ),

            "accuracy_change":
                float(
                    pc.mean()
                    -
                    bc.mean()
                ),

            "repairs":
                r,

            "breaks":
                b,

            "net_repairs":
                r - b,
        })

    by_count = pd.DataFrame(
        count_rows
    )

    by_count.to_csv(
        COUNT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Action distribution
    # --------------------------------------------------------

    action_distribution = (
        df[
            "selected_alpha"
        ]
        .astype(float)
        .value_counts()
        .sort_index()
        .rename_axis(
            "selected_alpha"
        )
        .reset_index(
            name="count"
        )
    )

    action_distribution[
        "fraction"
    ] = (
        action_distribution[
            "count"
        ]
        /
        2000.0
    )

    action_distribution.to_csv(
        ACTION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Run metadata
    # --------------------------------------------------------

    run_metadata = dict(
        run_metadata
    )

    run_metadata[
        "completed_utc"
    ] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    run_metadata[
        "primary_result"
    ] = overall

    RUN_META_PATH.write_text(
        json.dumps(
            run_metadata,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print only after COMPLETE dataset.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 118
    )

    print(
        "V3 FINAL PRIMARY RESULT"
    )

    print(
        "=" * 118
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print(
        "\n" + "=" * 118
    )

    print(
        "V3 PRESPECIFIED CONDITION RESULTS"
    )

    print(
        "=" * 118
    )

    print(
        by_condition.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print(
        "\n" + "=" * 118
    )

    print(
        "FROZEN CONTROLLER ACTION DISTRIBUTION"
    )

    print(
        "=" * 118
    )

    print(
        action_distribution.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        CONDITION_PATH
    )

    print(
        COUNT_PATH
    )

    print(
        ACTION_PATH
    )

    print(
        RUN_META_PATH
    )

    print(
        "\nV3 FROZEN FINAL EVALUATION COMPLETE"
    )


# ============================================================
# Main
# ============================================================

def main(
    resume=False,
    protocol_only=False,
):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = (
        load_jsonl(
            META_PATH
        )
    )

    (
        v3_protocol,
        runner_protocol,
        manifest,
        bundle,
        manifest_hash,
        bundle_hash,
    ) = protocol_audit(
        records
    )

    if protocol_only:

        print(
            "\nPROTOCOL-ONLY AUDIT PASS"
        )

        print(
            "No processor was loaded."
        )

        print(
            "No model was loaded."
        )

        print(
            "No V3 prediction was produced."
        )

        return

    # --------------------------------------------------------
    # Protect the final untouched result from accidental rerun.
    # --------------------------------------------------------

    if resume:

        completed = (
            prepare_resume(
                records
            )
        )

    else:

        if RESULT_PATH.exists():

            raise RuntimeError(
                "Final V3 result file already exists:\n"
                f"{RESULT_PATH}\n"
                "Refusing to overwrite final-confirmation "
                "outcomes."
            )

        completed = set()

    ordered = sorted(
        records,
        key=lambda r:
            str(
                r[
                    "sample_id"
                ]
            ),
    )

    remaining = [
        r
        for r in ordered
        if str(
            r[
                "sample_id"
            ]
        )
        not in completed
    ]

    current_head = (
        git_head()
    )

    run_started = {
        "protocol":
            runner_protocol[
                "protocol_name"
            ],

        "dataset":
            "proc_count_causal_v3",

        "generation_version":
            "3.0_unique",

        "samples":
            2000,

        "git_head_at_inference_start":
            current_head,

        "controller_freeze_commit":
            CONTROLLER_FREEZE_COMMIT,

        "v3_protocol_freeze_commit":
            V3_PROTOCOL_FREEZE_COMMIT,

        "controller_manifest_sha256":
            manifest_hash,

        "controller_bundle_sha256":
            bundle_hash,

        "primary_measurement":
            "expanded_next_token_numeral_proxy_0_15",

        "dummy_gt_used_for_scoring":
            DUMMY_GT,

        "started_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "resume":
            bool(
                resume
            ),

        "already_completed":
            len(
                completed
            ),

        "remaining":
            len(
                remaining
            ),
    }

    if (
        not resume
        or
        not RUN_STARTED_PATH.exists()
    ):

        RUN_STARTED_PATH.write_text(
            json.dumps(
                run_started,
                indent=2,
            )
            +
            "\n",
            encoding="utf-8",
        )

    print(
        "\nAlready completed:",
        len(
            completed
        ),
    )

    print(
        "Remaining:",
        len(
            remaining
        ),
    )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    numeral_ids = (
        numeral_token_ids(
            processor
        )
    )

    if sorted(
        numeral_ids.keys()
    ) != list(
        range(
            16
        )
    ):

        raise RuntimeError(
            "Expanded numeral-token audit "
            "failed."
        )

    print(
        "Numeral-token audit: PASS"
    )

    print(
        "Loading model..."
    )

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    )

    model.eval()

    feature_names = list(
        bundle[
            "feature_names"
        ]
    )

    for sample in tqdm(
        remaining,
        desc="V3 frozen final evaluation",
    ):

        sid = str(
            sample[
                "sample_id"
            ]
        )

        image = (
            Image.open(
                sample[
                    "image_path"
                ]
            )
            .convert(
                "RGB"
            )
        )

        inputs = (
            prepare_inputs(
                processor,
                image,
            )
        )

        inputs = (
            move_inputs(
                inputs,
                model,
            )
        )

        # ====================================================
        # IMPORTANT:
        #
        # The actual sample GT is NOT read before action
        # selection and intervention execution.
        #
        # score_state() receives fixed DUMMY_GT=1.
        # All controller features are GT-independent.
        # ====================================================

        baseline_state = (
            score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        )

        (
            X,
            feature_values,
        ) = (
            state_to_features(
                baseline_state,
                feature_names,
            )
        )

        (
            selected_alpha,
            selected_score,
            score_map,
        ) = (
            controller_decision(
                bundle,
                X,
            )
        )

        baseline_prediction = int(
            baseline_state[
                "best_numeral"
            ]
        )

        # ====================================================
        # NOOP identity:
        # alpha=1 uses the exact baseline state.
        # ====================================================

        if math.isclose(
            selected_alpha,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):

            post_state = (
                baseline_state
            )

            hook_calls = 0

        else:

            modifier = (
                HeadGainModifier(
                    model=model,
                    layer_idx=LAYER,
                    head_idx=HEAD,
                    alpha=
                        selected_alpha,
                )
            )

            modifier.register()

            try:

                post_state = (
                    score_state(
                        model,
                        inputs,
                        numeral_ids,
                        DUMMY_GT,
                    )
                )

            finally:

                modifier.remove()

            hook_calls = int(
                modifier.calls
            )

            if hook_calls <= 0:

                raise RuntimeError(
                    "Gain hook not called "
                    f"for {sid}"
                )

        post_prediction = int(
            post_state[
                "best_numeral"
            ]
        )

        # ====================================================
        # ONLY NOW is ground truth read for evaluation.
        # ====================================================

        gt = int(
            sample[
                "ground_truth"
            ]
        )

        baseline_correct = bool(
            baseline_prediction
            ==
            gt
        )

        post_correct = bool(
            post_prediction
            ==
            gt
        )

        repair = bool(
            (
                not baseline_correct
            )
            and
            post_correct
        )

        break_case = bool(
            baseline_correct
            and
            (
                not post_correct
            )
        )

        row = {
            "sample_id":
                sid,

            "condition":
                str(
                    sample[
                        "condition"
                    ]
                ),

            "replicate":
                int(
                    sample[
                        "replicate"
                    ]
                ),

            "ground_truth":
                gt,

            "baseline_prediction":
                baseline_prediction,

            "selected_alpha":
                float(
                    selected_alpha
                ),

            "selected_score":
                float(
                    selected_score
                ),

            "post_prediction":
                post_prediction,

            "baseline_correct":
                baseline_correct,

            "post_correct":
                post_correct,

            "repair":
                repair,

            "break_case":
                break_case,

            "hook_calls":
                hook_calls,

            "score_alpha_0":
                float(
                    score_map[
                        0.0
                    ]
                ),

            "score_alpha_1p5":
                float(
                    score_map[
                        1.5
                    ]
                ),

            "score_alpha_2":
                float(
                    score_map[
                        2.0
                    ]
                ),

            "score_alpha_4":
                float(
                    score_map[
                        4.0
                    ]
                ),

            "baseline_expected_numeral":
                float(
                    baseline_state[
                        "expected_numeral"
                    ]
                ),

            "baseline_entropy":
                float(
                    baseline_state[
                        "entropy"
                    ]
                ),

            "baseline_top1_prob":
                float(
                    baseline_state[
                        "top1_prob"
                    ]
                ),

            "baseline_conditional_margin":
                float(
                    baseline_state[
                        "conditional_margin"
                    ]
                ),

            "post_expected_numeral":
                float(
                    post_state[
                        "expected_numeral"
                    ]
                ),
        }

        # Save the actual frozen feature vector for auditability.
        for feature_name in (
            feature_names
        ):

            row[
                "feature__"
                +
                feature_name
            ] = float(
                feature_values[
                    feature_name
                ]
            )

        pd.DataFrame([
            row
        ]).to_csv(
            RESULT_PATH,
            mode="a",
            header=(
                not RESULT_PATH.exists()
            ),
            index=False,
        )

    final_df = pd.read_csv(
        RESULT_PATH
    )

    final_run_meta = {
        "protocol_name":
            runner_protocol[
                "protocol_name"
            ],

        "dataset":
            "proc_count_causal_v3",

        "generation_version":
            "3.0_unique",

        "n_samples":
            2000,

        "model":
            MODEL_ID,

        "attention_implementation":
            "eager",

        "controller_freeze_commit":
            CONTROLLER_FREEZE_COMMIT,

        "v3_dataset_protocol_freeze_commit":
            V3_PROTOCOL_FREEZE_COMMIT,

        "runner_git_head":
            current_head,

        "head":
            "L18H13",

        "actions":
            [
                0.0,
                1.0,
                1.5,
                2.0,
                4.0,
            ],

        "feature_family":
            "full_geometry",

        "feature_count":
            39,

        "ridge_alpha":
            0.01,

        "utility_threshold":
            0.1,

        "primary_measurement":
            "expanded numeral proxy 0..15",

        "ground_truth_used_for_controller":
            False,

        "controller_manifest_sha256":
            manifest_hash,

        "controller_bundle_sha256":
            bundle_hash,

        "bootstrap_replicates":
            BOOTSTRAP_REPS,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "retuned_on_v3":
            False,
    }

    summarize(
        final_df,
        final_run_meta,
    )


if __name__ == "__main__":

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only after an "
            "infrastructure interruption. "
            "Do not use for outcome-based "
            "reruns."
        ),
    )

    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help=(
            "Run all protocol, manifest, "
            "bundle, leakage, and Git audits "
            "without loading the model or "
            "producing V3 predictions."
        ),
    )

    args = (
        parser.parse_args()
    )

    main(
        resume=
            args.resume,

        protocol_only=
            args.protocol_only,
    )
