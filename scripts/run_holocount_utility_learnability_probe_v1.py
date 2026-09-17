#!/usr/bin/env python3

"""
AROMA HoloCount Conditional-Utility Learnability Probe v1
=========================================================

Question
--------
After frozen TallyQA -> HoloCount routing transfer fails, does the
existing 39-dimensional GT-free representation still contain enough
information to learn intervention utility within HoloCount?

Evidence class
--------------
POST-HOC DEVELOPMENT / LEARNABILITY PROBE.

This is NOT an independent confirmation experiment.

Frozen modeling specification
-----------------------------
- Features: exact existing 39 GT-free numeral-geometry features.
- Model: StandardScaler -> Ridge, independently per non-NOOP action.
- Ridge alpha: 0.01.
- Decision threshold: 0.1, strictly greater than.
- Actions: {0, 1, 1.5, 2, 4}; alpha=1 is NOOP.
- 5-fold OOF only.
- No hyperparameter search.
- No VLM inference.

Primary CV
----------
5-fold StratifiedGroupKFold:
    stratify = 20 official HoloCount subsets
    group    = image_sha256

This prevents identical image content from crossing train/test folds.

Sensitivity CV
--------------
Exact canonical-style 5-fold StratifiedKFold by HoloCount subset,
included only for comparison with the earlier TallyQA OOF protocol.
"""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import spearmanr


ROOT = Path("/workspace/AromaExperiments")

BUNDLE_PATH = (
    ROOT
    / "outputs/proc_count_causal_v2/controller/"
    "final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

FULL_PATH = (
    ROOT
    / "outputs/generalization_extension/holocount/"
    "external_v1/full_raw_results.csv"
)

ORACLE_PATH = (
    ROOT
    / "outputs/generalization_extension/holocount/"
    "external_v1/oracle_gate_v1/"
    "wrong_gtle15_action_results.csv"
)

CORRECT_MATRIX_PATH = (
    ROOT
    / "outputs/generalization_extension/holocount/"
    "external_v1/action_matrix_v1/"
    "baseline_correct_action_results.csv"
)

OUT_DIR = (
    ROOT
    / "outputs/generalization_extension/holocount/"
    "external_v1/utility_learnability_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_FULL_SHA = (
    "82929de79583d0c0641a80e13c8978d4ee4c00cebe7deb8f132697d3b6295466"
)

EXPECTED_ORACLE_SHA = (
    "1c5f367c43a0af43e88432cc7c577c23e349f8d4ce16d754bda32cf7e44aa6ce"
)

EXPECTED_CORRECT_MATRIX_SHA = (
    "fae0cfe8a7d13405cdee9c79cb3706fce76632df53b6747f9309d2c08c5e2556"
)


N_SPLITS = 5

# Same random state as canonical TallyQA utility-learnability probe.
RANDOM_STATE = 20260912

ALL_ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]

NONNOOP = [
    0.0,
    1.5,
    2.0,
    4.0,
]

ACTION_SUFFIX = {
    0.0: "0",
    1.5: "1p5",
    2.0: "2",
    4.0: "4",
}


def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def as_bool(s):

    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        return (
            pd.to_numeric(s)
            .fillna(0)
            .ne(0)
        )

    t = (
        s.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    bad = (
        set(t.unique())
        -
        set(mapping)
    )

    if bad:

        raise RuntimeError(
            f"Unknown bool values: {bad}"
        )

    return (
        t.map(mapping)
        .astype(bool)
    )


def safe_spearman(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(x) < 2
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return np.nan

    return float(
        spearmanr(
            x,
            y,
        ).statistic
    )


# ============================================================
# Frozen input integrity
# ============================================================

if sha256_file(
    FULL_PATH
) != EXPECTED_FULL_SHA:

    raise RuntimeError(
        "Full HoloCount result SHA mismatch."
    )


if sha256_file(
    ORACLE_PATH
) != EXPECTED_ORACLE_SHA:

    raise RuntimeError(
        "Oracle-gate result SHA mismatch."
    )


if sha256_file(
    CORRECT_MATRIX_PATH
) != EXPECTED_CORRECT_MATRIX_SHA:

    raise RuntimeError(
        "Baseline-correct action-matrix SHA mismatch."
    )


# ============================================================
# Load
# ============================================================

bundle = joblib.load(
    BUNDLE_PATH
)

full = pd.read_csv(
    FULL_PATH
)

oracle = pd.read_csv(
    ORACLE_PATH
)

correct_matrix = pd.read_csv(
    CORRECT_MATRIX_PATH
)


if len(full) != 2480:

    raise RuntimeError(
        f"Expected 2480 HoloCount rows, "
        f"got {len(full)}."
    )


if len(oracle) != 1134:

    raise RuntimeError(
        f"Expected 1134 oracle rows, "
        f"got {len(oracle)}."
    )


if len(correct_matrix) != 1158:

    raise RuntimeError(
        f"Expected 1158 baseline-correct rows, "
        f"got {len(correct_matrix)}."
    )


if full["sample_id"].nunique() != 2480:

    raise RuntimeError(
        "Full sample IDs are not unique."
    )


feature_names = list(
    bundle[
        "feature_names"
    ]
)

feature_cols = [
    f"feature__{f}"
    for f in feature_names
]


if len(
    feature_names
) != 39:

    raise RuntimeError(
        "Expected 39 frozen features."
    )


missing_features = [
    c
    for c in feature_cols
    if c not in full.columns
]

if missing_features:

    raise RuntimeError(
        "Missing feature columns: "
        f"{missing_features}"
    )


ridge_alpha = float(
    bundle[
        "ridge_alpha"
    ]
)

threshold = float(
    bundle[
        "threshold"
    ]
)


if ridge_alpha != 0.01:

    raise RuntimeError(
        f"Unexpected ridge alpha: "
        f"{ridge_alpha}"
    )


if threshold != 0.1:

    raise RuntimeError(
        f"Unexpected threshold: "
        f"{threshold}"
    )


# ============================================================
# Baseline state / features
# ============================================================

full[
    "ground_truth"
] = (
    full[
        "ground_truth"
    ]
    .astype(int)
)

full[
    "baseline_prediction"
] = (
    full[
        "baseline_prediction"
    ]
    .astype(int)
)


baseline_correct = (
    full[
        "baseline_prediction"
    ]
    .to_numpy(
        dtype=int
    )
    ==
    full[
        "ground_truth"
    ]
    .to_numpy(
        dtype=int
    )
)


if int(
    baseline_correct.sum()
) != 1158:

    raise RuntimeError(
        "Expected 1158 baseline-correct "
        "HoloCount examples."
    )


X = (
    full[
        feature_cols
    ]
    .to_numpy(
        dtype=float
    )
)


if X.shape != (
    2480,
    39,
):

    raise RuntimeError(
        f"Unexpected feature matrix: "
        f"{X.shape}"
    )


if not np.isfinite(
    X
).all():

    raise RuntimeError(
        "Non-finite feature values."
    )


sample_ids = (
    full[
        "sample_id"
    ]
    .astype(str)
    .to_numpy()
)

subset = (
    full[
        "split"
    ]
    .astype(str)
    .to_numpy()
)

groups = (
    full[
        "image_sha256"
    ]
    .astype(str)
    .to_numpy()
)


if len(
    np.unique(
        subset
    )
) != 20:

    raise RuntimeError(
        "Expected 20 HoloCount subsets."
    )


if len(
    np.unique(
        groups
    )
) != 2056:

    raise RuntimeError(
        "Expected 2056 image-hash groups."
    )


# ============================================================
# Build exact 2480 x 4 action-correctness matrix.
#
# Three populations:
#
# A. baseline correct: 1158
#    action correctness comes from correct_matrix.
#
# B. baseline wrong, GT<=15: 1134
#    action correctness comes from oracle repair labels.
#
# C. baseline wrong, GT>15: 188
#    frozen prediction space is only 0..15, therefore all
#    four non-NOOP actions are necessarily incorrect.
# ============================================================

id_to_index = {
    sid: i
    for i, sid
    in enumerate(
        sample_ids
    )
}


action_correct = {
    action:
        np.zeros(
            len(full),
            dtype=bool,
        )
    for action in NONNOOP
}


# ------------------------------------------------------------
# Population A: baseline-correct
# ------------------------------------------------------------

correct_ids = set()

for row in correct_matrix.itertuples(
    index=False
):

    sid = str(
        row.sample_id
    )

    if sid not in id_to_index:

        raise RuntimeError(
            f"Unknown correct-matrix "
            f"sample: {sid}"
        )

    i = id_to_index[
        sid
    ]

    if not baseline_correct[
        i
    ]:

        raise RuntimeError(
            "Correct-matrix sample is not "
            "baseline correct."
        )

    correct_ids.add(
        sid
    )

    for action in NONNOOP:

        suffix = ACTION_SUFFIX[
            action
        ]

        raw = getattr(
            row,
            f"correct_alpha_{suffix}",
        )

        action_correct[
            action
        ][
            i
        ] = bool(
            str(raw)
            .strip()
            .lower()
            in {
                "true",
                "1",
            }
        )


if len(
    correct_ids
) != 1158:

    raise RuntimeError(
        "Correct-matrix ID count mismatch."
    )


# ------------------------------------------------------------
# Population B: supported baseline-wrong
# ------------------------------------------------------------

oracle_ids = set()

for row in oracle.itertuples(
    index=False
):

    sid = str(
        row.sample_id
    )

    if sid not in id_to_index:

        raise RuntimeError(
            f"Unknown oracle sample: "
            f"{sid}"
        )

    i = id_to_index[
        sid
    ]

    if baseline_correct[
        i
    ]:

        raise RuntimeError(
            "Oracle sample unexpectedly "
            "baseline correct."
        )

    if int(
        full.iloc[
            i
        ][
            "ground_truth"
        ]
    ) > 15:

        raise RuntimeError(
            "Oracle population contains "
            "GT>15 sample."
        )

    oracle_ids.add(
        sid
    )

    for action in NONNOOP:

        suffix = ACTION_SUFFIX[
            action
        ]

        raw = getattr(
            row,
            f"repair_alpha_{suffix}",
        )

        action_correct[
            action
        ][
            i
        ] = bool(
            str(raw)
            .strip()
            .lower()
            in {
                "true",
                "1",
            }
        )


if len(
    oracle_ids
) != 1134:

    raise RuntimeError(
        "Oracle ID count mismatch."
    )


if correct_ids & oracle_ids:

    raise RuntimeError(
        "Correct/oracle populations overlap."
    )


# ------------------------------------------------------------
# Population C: unsupported baseline-wrong
# ------------------------------------------------------------

assigned = (
    correct_ids
    |
    oracle_ids
)

remaining_ids = (
    set(
        sample_ids
    )
    -
    assigned
)


if len(
    remaining_ids
) != 188:

    raise RuntimeError(
        f"Expected 188 unsupported rows, "
        f"got {len(remaining_ids)}."
    )


for sid in remaining_ids:

    i = id_to_index[
        sid
    ]

    if baseline_correct[
        i
    ]:

        raise RuntimeError(
            "Unsupported remainder is "
            "baseline correct."
        )

    if int(
        full.iloc[
            i
        ][
            "ground_truth"
        ]
    ) <= 15:

        raise RuntimeError(
            "Unsupported remainder has "
            "GT<=15."
        )


# ============================================================
# Utility labels:
#
# u_i(a) = c_i(a) - c_i(1)
#
# +1 repair
#  0 neutral
# -1 break
# ============================================================

U = {}

for action in NONNOOP:

    U[
        action
    ] = (
        action_correct[
            action
        ]
        .astype(int)
        -
        baseline_correct
        .astype(int)
    )

    allowed = set(
        np.unique(
            U[
                action
            ]
        )
        .tolist()
    )

    if not allowed.issubset(
        {
            -1,
            0,
            1,
        }
    ):

        raise RuntimeError(
            f"Unexpected utility labels "
            f"for alpha={action}: "
            f"{allowed}"
        )


# ============================================================
# Sanity-check reconstructed fixed-action results.
# ============================================================

EXPECTED_FIXED = {
    0.0:
        (99, 120),

    1.5:
        (59, 58),

    2.0:
        (95, 111),

    4.0:
        (176, 265),
}


print("=" * 76)
print(
    "HOLOCOUNT CONDITIONAL-UTILITY "
    "LEARNABILITY PROBE V1"
)
print("=" * 76)

print()
print("===== RECONSTRUCTED ACTION UTILITIES =====")

for action in NONNOOP:

    repairs = int(
        (
            U[
                action
            ]
            ==
            1
        ).sum()
    )

    breaks = int(
        (
            U[
                action
            ]
            ==
            -1
        ).sum()
    )

    expected = EXPECTED_FIXED[
        action
    ]

    if (
        repairs,
        breaks,
    ) != expected:

        raise RuntimeError(
            f"Fixed-action reconstruction "
            f"mismatch for alpha={action}: "
            f"{(repairs, breaks)} != "
            f"{expected}"
        )

    print(
        f"alpha={action:<3} "
        f"repairs={repairs:3d} "
        f"breaks={breaks:3d} "
        f"net={repairs-breaks:+4d}"
    )


# ============================================================
# OOF evaluator
# ============================================================

def evaluate_oof(
    name,
    splitter,
    use_groups,
):

    print()
    print(
        "=" * 76
    )
    print(
        f"OOF: {name}"
    )
    print(
        "=" * 76
    )

    oof_scores = {
        action:
            np.full(
                len(full),
                np.nan,
                dtype=float,
            )
        for action in NONNOOP
    }

    oof_fold = np.full(
        len(full),
        -1,
        dtype=int,
    )

    fold_rows = []


    if use_groups:

        split_iter = splitter.split(
            X,
            subset,
            groups=groups,
        )

    else:

        split_iter = splitter.split(
            X,
            subset,
        )


    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        split_iter
    ):

        oof_fold[
            test_idx
        ] = fold


        if set(
            train_idx
        ) & set(
            test_idx
        ):

            raise RuntimeError(
                "Train/test row overlap."
            )


        if use_groups:

            train_groups = set(
                groups[
                    train_idx
                ]
            )

            test_groups = set(
                groups[
                    test_idx
                ]
            )

            overlap = (
                train_groups
                &
                test_groups
            )

            if overlap:

                raise RuntimeError(
                    f"Image-group leakage in "
                    f"fold {fold}: "
                    f"{len(overlap)} groups."
                )


        for action in NONNOOP:

            model = Pipeline(
                [
                    (
                        "scale",
                        StandardScaler(),
                    ),
                    (
                        "ridge",
                        Ridge(
                            alpha=
                                ridge_alpha,
                        ),
                    ),
                ]
            )

            model.fit(
                X[
                    train_idx
                ],
                U[
                    action
                ][
                    train_idx
                ],
            )

            oof_scores[
                action
            ][
                test_idx
            ] = (
                model.predict(
                    X[
                        test_idx
                    ]
                )
            )


        row = {
            "cv":
                name,

            "fold":
                fold,

            "train_n":
                len(
                    train_idx
                ),

            "test_n":
                len(
                    test_idx
                ),

            "train_image_groups":
                int(
                    len(
                        np.unique(
                            groups[
                                train_idx
                            ]
                        )
                    )
                ),

            "test_image_groups":
                int(
                    len(
                        np.unique(
                            groups[
                                test_idx
                            ]
                        )
                    )
                ),

            "test_baseline_accuracy":
                float(
                    baseline_correct[
                        test_idx
                    ].mean()
                ),
        }

        for label in sorted(
            np.unique(
                subset
            )
        ):

            row[
                f"test_subset__{label}"
            ] = int(
                (
                    subset[
                        test_idx
                    ]
                    ==
                    label
                ).sum()
            )

        fold_rows.append(
            row
        )


    if (
        oof_fold
        <
        0
    ).any():

        raise RuntimeError(
            "Incomplete OOF assignment."
        )


    for action in NONNOOP:

        if not np.isfinite(
            oof_scores[
                action
            ]
        ).all():

            raise RuntimeError(
                f"Missing OOF scores for "
                f"alpha={action}."
            )


    # --------------------------------------------------------
    # Exact frozen action-selection rule.
    # --------------------------------------------------------

    score_matrix = np.column_stack(
        [
            oof_scores[
                action
            ]
            for action
            in NONNOOP
        ]
    )

    best_idx = np.argmax(
        score_matrix,
        axis=1,
    )

    best_score = score_matrix[
        np.arange(
            len(full)
        ),
        best_idx,
    ]

    nonnoop_array = np.asarray(
        NONNOOP,
        dtype=float,
    )

    best_action = (
        nonnoop_array[
            best_idx
        ]
    )

    selected_alpha = np.where(
        best_score
        >
        threshold,
        best_action,
        1.0,
    )


    # --------------------------------------------------------
    # Realized selected utility.
    # --------------------------------------------------------

    selected_utility = np.zeros(
        len(full),
        dtype=int,
    )


    for action in NONNOOP:

        mask = np.isclose(
            selected_alpha,
            action,
            rtol=0.0,
            atol=1e-12,
        )

        selected_utility[
            mask
        ] = U[
            action
        ][
            mask
        ]


    post_correct_int = (
        baseline_correct
        .astype(int)
        +
        selected_utility
    )


    if not np.isin(
        post_correct_int,
        [
            0,
            1,
        ],
    ).all():

        raise RuntimeError(
            "Invalid reconstructed "
            "post correctness."
        )


    post_correct = (
        post_correct_int
        .astype(bool)
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

    baseline_accuracy = float(
        baseline_correct.mean()
    )

    post_accuracy = float(
        post_correct.mean()
    )

    intervention = (
        ~np.isclose(
            selected_alpha,
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    )

    intervention_rate = float(
        intervention.mean()
    )


    # --------------------------------------------------------
    # Supported-count result, using same OOF policy.
    # --------------------------------------------------------

    supported = (
        full[
            "ground_truth"
        ]
        .to_numpy(
            dtype=int
        )
        <=
        15
    )

    supported_baseline = float(
        baseline_correct[
            supported
        ].mean()
    )

    supported_post = float(
        post_correct[
            supported
        ].mean()
    )


    # --------------------------------------------------------
    # Score-utility alignment after in-domain OOF fitting.
    # --------------------------------------------------------

    alignment_rows = []

    for action in NONNOOP:

        alignment_rows.append(
            {
                "cv":
                    name,

                "alpha":
                    action,

                "oof_score_utility_spearman":
                    safe_spearman(
                        oof_scores[
                            action
                        ],
                        U[
                            action
                        ],
                    ),

                "utility_mean":
                    float(
                        U[
                            action
                        ].mean()
                    ),

                "score_mean":
                    float(
                        oof_scores[
                            action
                        ].mean()
                    ),
            }
        )


    # --------------------------------------------------------
    # Per-sample artifact.
    # --------------------------------------------------------

    per_sample = pd.DataFrame(
        {
            "sample_id":
                sample_ids,

            "subset":
                subset,

            "image_sha256":
                groups,

            "fold":
                oof_fold,

            "ground_truth":
                full[
                    "ground_truth"
                ].to_numpy(
                    dtype=int
                ),

            "baseline_correct":
                baseline_correct,

            "selected_alpha":
                selected_alpha,

            "selected_score":
                best_score,

            "selected_utility":
                selected_utility,

            "post_correct":
                post_correct,
        }
    )


    for action in NONNOOP:

        tag = (
            str(
                action
            )
            .replace(
                ".",
                "p",
            )
        )

        per_sample[
            f"score_alpha_{tag}"
        ] = oof_scores[
            action
        ]

        per_sample[
            f"utility_alpha_{tag}"
        ] = U[
            action
        ]


    # --------------------------------------------------------
    # Action distribution.
    # --------------------------------------------------------

    action_dist = (
        pd.Series(
            selected_alpha,
            name="selected_alpha",
        )
        .value_counts()
        .sort_index()
        .rename(
            "count"
        )
        .reset_index()
    )

    action_dist[
        "fraction"
    ] = (
        action_dist[
            "count"
        ]
        /
        len(full)
    )

    action_dist[
        "cv"
    ] = name


    summary = {
        "cv":
            name,

        "n":
            len(full),

        "unique_image_groups":
            int(
                len(
                    np.unique(
                        groups
                    )
                )
            ),

        "baseline_accuracy":
            baseline_accuracy,

        "oof_post_accuracy":
            post_accuracy,

        "accuracy_change":
            (
                post_accuracy
                -
                baseline_accuracy
            ),

        "accuracy_change_pp":
            (
                100.0
                *
                (
                    post_accuracy
                    -
                    baseline_accuracy
                )
            ),

        "repairs":
            repairs,

        "breaks":
            breaks,

        "net_repairs":
            repairs
            -
            breaks,

        "interventions":
            int(
                intervention.sum()
            ),

        "intervention_rate":
            intervention_rate,

        "supported_n":
            int(
                supported.sum()
            ),

        "supported_baseline_accuracy":
            supported_baseline,

        "supported_post_accuracy":
            supported_post,

        "supported_accuracy_change_pp":
            (
                100.0
                *
                (
                    supported_post
                    -
                    supported_baseline
                )
            ),

        "ridge_alpha":
            ridge_alpha,

        "threshold":
            threshold,

        "n_splits":
            N_SPLITS,

        "random_state":
            RANDOM_STATE,
    }


    return {
        "summary":
            summary,

        "per_sample":
            per_sample,

        "folds":
            pd.DataFrame(
                fold_rows
            ),

        "alignment":
            pd.DataFrame(
                alignment_rows
            ),

        "action_distribution":
            action_dist,
    }


# ============================================================
# PRIMARY: image-disjoint group-aware OOF.
# ============================================================

primary_splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

primary = evaluate_oof(
    name="PRIMARY_STRATIFIED_GROUP_BY_IMAGE_SHA256",
    splitter=primary_splitter,
    use_groups=True,
)


# ============================================================
# SENSITIVITY: exact canonical-style ordinary stratified OOF.
# ============================================================

sensitivity_splitter = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

sensitivity = evaluate_oof(
    name="SENSITIVITY_STRATIFIED_BY_SUBSET",
    splitter=sensitivity_splitter,
    use_groups=False,
)


# ============================================================
# Write outputs.
# ============================================================

summary_df = pd.DataFrame(
    [
        primary[
            "summary"
        ],
        sensitivity[
            "summary"
        ],
    ]
)

summary_df.to_csv(
    OUT_DIR
    /
    "oof_summary.csv",
    index=False,
)


primary[
    "per_sample"
].to_csv(
    OUT_DIR
    /
    "primary_oof_predictions.csv",
    index=False,
)


sensitivity[
    "per_sample"
].to_csv(
    OUT_DIR
    /
    "sensitivity_oof_predictions.csv",
    index=False,
)


pd.concat(
    [
        primary[
            "folds"
        ],
        sensitivity[
            "folds"
        ],
    ],
    ignore_index=True,
).to_csv(
    OUT_DIR
    /
    "fold_summary.csv",
    index=False,
)


pd.concat(
    [
        primary[
            "alignment"
        ],
        sensitivity[
            "alignment"
        ],
    ],
    ignore_index=True,
).to_csv(
    OUT_DIR
    /
    "score_utility_alignment.csv",
    index=False,
)


pd.concat(
    [
        primary[
            "action_distribution"
        ],
        sensitivity[
            "action_distribution"
        ],
    ],
    ignore_index=True,
).to_csv(
    OUT_DIR
    /
    "action_distribution.csv",
    index=False,
)


metadata = {
    "analysis_type":
        "post_hoc_holocount_conditional_utility_learnability_probe",

    "evidence_class":
        "POST_HOC_DEVELOPMENT",

    "independent_confirmation":
        False,

    "model_inference_performed":
        False,

    "feature_family":
        bundle[
            "feature_family"
        ],

    "feature_count":
        len(
            feature_names
        ),

    "ridge_alpha":
        ridge_alpha,

    "threshold":
        threshold,

    "actions":
        ALL_ACTIONS,

    "utility_definition":
        "action_correct - baseline_correct",

    "primary_cv":
        (
            "5-fold StratifiedGroupKFold by "
            "HoloCount subset, grouped by image_sha256"
        ),

    "sensitivity_cv":
        (
            "5-fold StratifiedKFold by "
            "HoloCount subset"
        ),

    "random_state":
        RANDOM_STATE,

    "hyperparameter_search":
        False,

    "full_result_sha256":
        EXPECTED_FULL_SHA,

    "oracle_result_sha256":
        EXPECTED_ORACLE_SHA,

    "correct_matrix_sha256":
        EXPECTED_CORRECT_MATRIX_SHA,
}


with (
    OUT_DIR
    /
    "metadata.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        sort_keys=True,
    )


# ============================================================
# Console result.
# ============================================================

print()
print("=" * 76)
print(
    "HOLOCOUNT UTILITY LEARNABILITY RESULTS"
)
print("=" * 76)

print()
print(
    summary_df[
        [
            "cv",
            "n",
            "baseline_accuracy",
            "oof_post_accuracy",
            "accuracy_change_pp",
            "repairs",
            "breaks",
            "net_repairs",
            "intervention_rate",
            "supported_accuracy_change_pp",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("===== PRIMARY OOF SCORE-UTILITY ALIGNMENT =====")

print(
    primary[
        "alignment"
    ][
        [
            "alpha",
            "oof_score_utility_spearman",
            "utility_mean",
            "score_mean",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("===== PRIMARY ACTION DISTRIBUTION =====")

print(
    primary[
        "action_distribution"
    ][
        [
            "selected_alpha",
            "count",
            "fraction",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "Artifacts written to:",
    OUT_DIR,
)
