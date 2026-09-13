#!/usr/bin/env python3

"""
AROMA Frozen Ridge Controller:
Exact Feature-Shift and Score-Explosion Attribution
===================================================

Purpose
-------
Explain why the frozen synthetic-trained controller becomes
miscalibrated on natural TallyQA.

Because each frozen action model is:

    StandardScaler -> Ridge

the per-action score has the exact form

    s_a(x) = b_a + sum_j beta_{a,j} z_j(x),

where

    z_j(x) = (phi_j(x) - scaler_mean_j) / scaler_scale_j.

Therefore the mean score drift between two domains decomposes exactly:

    E_T[s_a] - E_V[s_a]
      = sum_j beta_{a,j}
          (E_T[z_j] - E_V[z_j]).

This script performs:

1. exact stored-score reproduction;
2. exact frozen-controller decision reproduction;
3. mean score-drift attribution;
4. standardized feature OOD analysis;
5. extreme-score sample attribution;
6. baseline-correct / baseline-wrong attribution.

IMPORTANT
---------
- NO model inference.
- NO controller fitting.
- NO hyperparameter tuning.
- NO modification of frozen artifacts.
- Post-hoc forensic analysis only.
"""

from pathlib import Path
import json
import hashlib

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Frozen artifact paths
# ============================================================

BUNDLE_PATH = Path(
    "outputs/proc_count_causal_v2/controller/"
    "final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

V3_PATH = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller/"
    "v3_final_results.csv"
)

TALLY_PATH = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "final_frozen_controller/"
    "tallyqa_final_results.csv"
)

OUT_DIR = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "forensics/ridge_feature_attribution"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SCORE_COLS = {
    0.0: "score_alpha_0",
    1.5: "score_alpha_1p5",
    2.0: "score_alpha_2",
    4.0: "score_alpha_4",
}

TOP_K_EXTREME_SAMPLES = 20
TOP_K_CONTRIB_FEATURES = 10

SCORE_TOL = 1e-6


# ============================================================
# Helpers
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


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.to_numeric(
                series
            )
            .fillna(0)
            .ne(0)
        )

    s = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    unknown = set(
        s.unique()
    ) - set(
        mapping.keys()
    )

    if unknown:
        raise ValueError(
            "Unknown boolean values: "
            f"{sorted(unknown)}"
        )

    return (
        s.map(mapping)
        .astype(bool)
    )


def safe_quantile(
    x,
    q,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan

    return float(
        np.quantile(
            x,
            q,
        )
    )


# ============================================================
# Load frozen artifacts
# ============================================================

for path in [
    BUNDLE_PATH,
    V3_PATH,
    TALLY_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

bundle = joblib.load(
    BUNDLE_PATH
)

v3 = pd.read_csv(
    V3_PATH
)

tally = pd.read_csv(
    TALLY_PATH
)


# ============================================================
# Frozen specification
# ============================================================

feature_names = list(
    bundle[
        "feature_names"
    ]
)

if len(
    feature_names
) != 39:
    raise RuntimeError(
        "Expected exactly 39 frozen features."
    )

feature_cols = [
    f"feature__{name}"
    for name in feature_names
]

for domain_name, df in [
    ("v3", v3),
    ("TallyQA", tally),
]:
    missing = [
        c
        for c in feature_cols
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{domain_name} missing features: "
            f"{missing}"
        )

actions = [
    float(a)
    for a in bundle[
        "nonnoop_actions"
    ]
]

if actions != [
    0.0,
    1.5,
    2.0,
    4.0,
]:
    raise RuntimeError(
        f"Unexpected action order: {actions}"
    )

threshold = float(
    bundle[
        "threshold"
    ]
)


# ============================================================
# Feature matrices
# ============================================================

X_v3 = (
    v3[
        feature_cols
    ]
    .to_numpy(
        dtype=float
    )
)

X_tally = (
    tally[
        feature_cols
    ]
    .to_numpy(
        dtype=float
    )
)

if not np.isfinite(
    X_v3
).all():
    raise RuntimeError(
        "Non-finite value in v3 features."
    )

if not np.isfinite(
    X_tally
).all():
    raise RuntimeError(
        "Non-finite value in TallyQA features."
    )


# ============================================================
# 1. Exact stored-score reproduction
# ============================================================

score_repro_rows = []

pred_v3 = {}
pred_tally = {}

for alpha in actions:

    model = bundle[
        "models"
    ][alpha]

    score_col = SCORE_COLS[
        alpha
    ]

    if score_col not in v3.columns:
        raise RuntimeError(
            f"v3 missing {score_col}"
        )

    if score_col not in tally.columns:
        raise RuntimeError(
            f"TallyQA missing {score_col}"
        )

    pv3 = np.asarray(
        model.predict(
            X_v3
        ),
        dtype=float,
    )

    ptq = np.asarray(
        model.predict(
            X_tally
        ),
        dtype=float,
    )

    pred_v3[
        alpha
    ] = pv3

    pred_tally[
        alpha
    ] = ptq

    stored_v3 = pd.to_numeric(
        v3[
            score_col
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    stored_tq = pd.to_numeric(
        tally[
            score_col
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    diff_v3 = np.abs(
        pv3
        -
        stored_v3
    )

    diff_tq = np.abs(
        ptq
        -
        stored_tq
    )

    score_repro_rows.append(
        {
            "alpha":
                alpha,

            "v3_max_abs_diff":
                float(
                    diff_v3.max()
                ),

            "v3_mean_abs_diff":
                float(
                    diff_v3.mean()
                ),

            "v3_mismatches_gt_tol":
                int(
                    (
                        diff_v3
                        >
                        SCORE_TOL
                    ).sum()
                ),

            "tally_max_abs_diff":
                float(
                    diff_tq.max()
                ),

            "tally_mean_abs_diff":
                float(
                    diff_tq.mean()
                ),

            "tally_mismatches_gt_tol":
                int(
                    (
                        diff_tq
                        >
                        SCORE_TOL
                    ).sum()
                ),
        }
    )

score_reproduction = pd.DataFrame(
    score_repro_rows
)

if (
    score_reproduction[
        "v3_mismatches_gt_tol"
    ].sum()
    !=
    0
):
    raise RuntimeError(
        "Frozen model does not reproduce "
        "stored v3 scores."
    )

if (
    score_reproduction[
        "tally_mismatches_gt_tol"
    ].sum()
    !=
    0
):
    raise RuntimeError(
        "Frozen model does not reproduce "
        "stored TallyQA scores."
    )

score_reproduction.to_csv(
    OUT_DIR
    / "stored_score_reproduction.csv",
    index=False,
)


# ============================================================
# 2. Exact frozen controller decision reproduction
# ============================================================

def reconstruct_controller(
    predictions,
    n_rows,
):
    score_matrix = np.column_stack(
        [
            predictions[a]
            for a in actions
        ]
    )

    best_idx = np.argmax(
        score_matrix,
        axis=1,
    )

    best_score = score_matrix[
        np.arange(
            n_rows
        ),
        best_idx,
    ]

    best_action = np.array(
        [
            actions[i]
            for i in best_idx
        ],
        dtype=float,
    )

    selected = np.where(
        best_score
        >
        threshold,
        best_action,
        1.0,
    )

    return (
        selected,
        best_score,
        best_action,
    )


v3_selected_recon, \
v3_best_score, \
v3_best_nonnoop = (
    reconstruct_controller(
        pred_v3,
        len(v3),
    )
)

tq_selected_recon, \
tq_best_score, \
tq_best_nonnoop = (
    reconstruct_controller(
        pred_tally,
        len(tally),
    )
)


v3_selected_stored = pd.to_numeric(
    v3[
        "selected_alpha"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

tq_selected_stored = pd.to_numeric(
    tally[
        "selected_alpha"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)


v3_alpha_mismatch = int(
    (
        ~np.isclose(
            v3_selected_recon,
            v3_selected_stored,
        )
    ).sum()
)

tq_alpha_mismatch = int(
    (
        ~np.isclose(
            tq_selected_recon,
            tq_selected_stored,
        )
    ).sum()
)


v3_selected_score_stored = pd.to_numeric(
    v3[
        "selected_score"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

tq_selected_score_stored = pd.to_numeric(
    tally[
        "selected_score"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)


v3_selected_score_diff = np.abs(
    v3_best_score
    -
    v3_selected_score_stored
)

tq_selected_score_diff = np.abs(
    tq_best_score
    -
    tq_selected_score_stored
)


decision_reproduction = pd.DataFrame(
    [
        {
            "domain":
                "v3",

            "n":
                len(v3),

            "selected_alpha_mismatches":
                v3_alpha_mismatch,

            "selected_score_max_abs_diff":
                float(
                    v3_selected_score_diff.max()
                ),

            "selected_score_mismatches_gt_tol":
                int(
                    (
                        v3_selected_score_diff
                        >
                        SCORE_TOL
                    ).sum()
                ),
        },

        {
            "domain":
                "tallyqa",

            "n":
                len(tally),

            "selected_alpha_mismatches":
                tq_alpha_mismatch,

            "selected_score_max_abs_diff":
                float(
                    tq_selected_score_diff.max()
                ),

            "selected_score_mismatches_gt_tol":
                int(
                    (
                        tq_selected_score_diff
                        >
                        SCORE_TOL
                    ).sum()
                ),
        },
    ]
)

if v3_alpha_mismatch != 0:
    raise RuntimeError(
        "Could not reproduce v3 selected actions."
    )

if tq_alpha_mismatch != 0:
    raise RuntimeError(
        "Could not reproduce TallyQA selected actions."
    )

if (
    decision_reproduction[
        "selected_score_mismatches_gt_tol"
    ].sum()
    !=
    0
):
    raise RuntimeError(
        "Could not reproduce selected_score."
    )

decision_reproduction.to_csv(
    OUT_DIR
    / "controller_decision_reproduction.csv",
    index=False,
)


# ============================================================
# 3. Exact standardized feature / coefficient extraction
# ============================================================

action_components = {}

for alpha in actions:

    pipe = bundle[
        "models"
    ][alpha]

    scaler = pipe.named_steps[
        "scale"
    ]

    ridge = pipe.named_steps[
        "ridge"
    ]

    mean = np.asarray(
        scaler.mean_,
        dtype=float,
    )

    scale = np.asarray(
        scaler.scale_,
        dtype=float,
    )

    coef = np.asarray(
        ridge.coef_,
        dtype=float,
    )

    intercept = float(
        ridge.intercept_
    )

    if (
        len(mean) != 39
        or
        len(scale) != 39
        or
        len(coef) != 39
    ):
        raise RuntimeError(
            f"Unexpected dimensionality "
            f"for alpha={alpha}."
        )

    if (
        scale
        <=
        0
    ).any():
        raise RuntimeError(
            f"Non-positive scaler scale "
            f"for alpha={alpha}."
        )

    # Standardized domain representations.
    Z_v3 = (
        X_v3
        -
        mean
    ) / scale

    Z_tq = (
        X_tally
        -
        mean
    ) / scale

    contrib_v3 = (
        Z_v3
        *
        coef
    )

    contrib_tq = (
        Z_tq
        *
        coef
    )

    # Manual score reconstruction.
    manual_v3 = (
        intercept
        +
        contrib_v3.sum(
            axis=1
        )
    )

    manual_tq = (
        intercept
        +
        contrib_tq.sum(
            axis=1
        )
    )

    if np.max(
        np.abs(
            manual_v3
            -
            pred_v3[
                alpha
            ]
        )
    ) > SCORE_TOL:
        raise RuntimeError(
            f"Manual v3 decomposition failed "
            f"for alpha={alpha}."
        )

    if np.max(
        np.abs(
            manual_tq
            -
            pred_tally[
                alpha
            ]
        )
    ) > SCORE_TOL:
        raise RuntimeError(
            f"Manual TallyQA decomposition failed "
            f"for alpha={alpha}."
        )

    action_components[
        alpha
    ] = {
        "mean":
            mean,

        "scale":
            scale,

        "coef":
            coef,

        "intercept":
            intercept,

        "Z_v3":
            Z_v3,

        "Z_tally":
            Z_tq,

        "contrib_v3":
            contrib_v3,

        "contrib_tally":
            contrib_tq,
    }


# ============================================================
# 4. Mean score drift attribution
# ============================================================

v3_base_correct = as_bool(
    v3[
        "baseline_correct"
    ]
).to_numpy()

tq_base_correct = as_bool(
    tally[
        "baseline_correct"
    ]
).to_numpy()


groups = {
    "overall": (
        np.ones(
            len(v3),
            dtype=bool,
        ),
        np.ones(
            len(tally),
            dtype=bool,
        ),
    ),

    "baseline_correct": (
        v3_base_correct,
        tq_base_correct,
    ),

    "baseline_wrong": (
        ~v3_base_correct,
        ~tq_base_correct,
    ),
}


attribution_rows = []
attribution_check_rows = []


for alpha in actions:

    comp = action_components[
        alpha
    ]

    coef = comp[
        "coef"
    ]

    scaler_mean = comp[
        "mean"
    ]

    scaler_scale = comp[
        "scale"
    ]

    for group_name, (
        mask_v3,
        mask_tq,
    ) in groups.items():

        Z_v3_g = comp[
            "Z_v3"
        ][
            mask_v3
        ]

        Z_tq_g = comp[
            "Z_tally"
        ][
            mask_tq
        ]

        X_v3_g = X_v3[
            mask_v3
        ]

        X_tq_g = X_tally[
            mask_tq
        ]

        mean_z_v3 = (
            Z_v3_g.mean(
                axis=0
            )
        )

        mean_z_tq = (
            Z_tq_g.mean(
                axis=0
            )
        )

        delta_z = (
            mean_z_tq
            -
            mean_z_v3
        )

        score_drift_contrib = (
            coef
            *
            delta_z
        )

        raw_mean_v3 = (
            X_v3_g.mean(
                axis=0
            )
        )

        raw_mean_tq = (
            X_tq_g.mean(
                axis=0
            )
        )

        for j, feature in enumerate(
            feature_names
        ):

            attribution_rows.append(
                {
                    "alpha":
                        alpha,

                    "group":
                        group_name,

                    "feature_index":
                        j,

                    "feature":
                        feature,

                    "ridge_coef_standardized":
                        float(
                            coef[j]
                        ),

                    "scaler_train_mean":
                        float(
                            scaler_mean[j]
                        ),

                    "scaler_train_scale":
                        float(
                            scaler_scale[j]
                        ),

                    "v3_raw_mean":
                        float(
                            raw_mean_v3[j]
                        ),

                    "tally_raw_mean":
                        float(
                            raw_mean_tq[j]
                        ),

                    "raw_mean_shift":
                        float(
                            raw_mean_tq[j]
                            -
                            raw_mean_v3[j]
                        ),

                    "v3_mean_z":
                        float(
                            mean_z_v3[j]
                        ),

                    "tally_mean_z":
                        float(
                            mean_z_tq[j]
                        ),

                    "delta_mean_z":
                        float(
                            delta_z[j]
                        ),

                    "score_drift_contribution":
                        float(
                            score_drift_contrib[j]
                        ),

                    "abs_score_drift_contribution":
                        float(
                            abs(
                                score_drift_contrib[j]
                            )
                        ),
                }
            )

        mean_score_v3 = float(
            pred_v3[
                alpha
            ][
                mask_v3
            ].mean()
        )

        mean_score_tq = float(
            pred_tally[
                alpha
            ][
                mask_tq
            ].mean()
        )

        observed_drift = (
            mean_score_tq
            -
            mean_score_v3
        )

        explained_drift = float(
            score_drift_contrib.sum()
        )

        attribution_check_rows.append(
            {
                "alpha":
                    alpha,

                "group":
                    group_name,

                "v3_mean_score":
                    mean_score_v3,

                "tally_mean_score":
                    mean_score_tq,

                "observed_score_drift":
                    observed_drift,

                "sum_feature_contributions":
                    explained_drift,

                "decomposition_residual":
                    observed_drift
                    -
                    explained_drift,

                "v3_n":
                    int(
                        mask_v3.sum()
                    ),

                "tally_n":
                    int(
                        mask_tq.sum()
                    ),
            }
        )


attribution = pd.DataFrame(
    attribution_rows
)

attribution = attribution.sort_values(
    [
        "alpha",
        "group",
        "abs_score_drift_contribution",
    ],
    ascending=[
        True,
        True,
        False,
    ],
)

attribution.to_csv(
    OUT_DIR
    / "mean_score_drift_feature_attribution.csv",
    index=False,
)


attribution_check = pd.DataFrame(
    attribution_check_rows
)

if (
    attribution_check[
        "decomposition_residual"
    ]
    .abs()
    .max()
    >
    SCORE_TOL
):
    raise RuntimeError(
        "Mean score-drift decomposition "
        "does not close numerically."
    )

attribution_check.to_csv(
    OUT_DIR
    / "mean_score_drift_decomposition_check.csv",
    index=False,
)


# ============================================================
# 5. Standardized feature OOD severity
#
# Use each action's scaler. StandardScaler parameters should
# normally be identical because the same training features were
# used, but we preserve action specificity.
# ============================================================

ood_rows = []

for alpha in actions:

    comp = action_components[
        alpha
    ]

    Z_v = comp[
        "Z_v3"
    ]

    Z_t = comp[
        "Z_tally"
    ]

    for j, feature in enumerate(
        feature_names
    ):

        av = np.abs(
            Z_v[
                :,
                j
            ]
        )

        at = np.abs(
            Z_t[
                :,
                j
            ]
        )

        ood_rows.append(
            {
                "alpha":
                    alpha,

                "feature":
                    feature,

                "v3_abs_z_median":
                    safe_quantile(
                        av,
                        0.50,
                    ),

                "tally_abs_z_median":
                    safe_quantile(
                        at,
                        0.50,
                    ),

                "v3_abs_z_q95":
                    safe_quantile(
                        av,
                        0.95,
                    ),

                "tally_abs_z_q95":
                    safe_quantile(
                        at,
                        0.95,
                    ),

                "v3_abs_z_q99":
                    safe_quantile(
                        av,
                        0.99,
                    ),

                "tally_abs_z_q99":
                    safe_quantile(
                        at,
                        0.99,
                    ),

                "v3_abs_z_q999":
                    safe_quantile(
                        av,
                        0.999,
                    ),

                "tally_abs_z_q999":
                    safe_quantile(
                        at,
                        0.999,
                    ),

                "v3_abs_z_max":
                    float(
                        av.max()
                    ),

                "tally_abs_z_max":
                    float(
                        at.max()
                    ),

                "v3_fraction_abs_z_gt_5":
                    float(
                        (
                            av > 5
                        ).mean()
                    ),

                "tally_fraction_abs_z_gt_5":
                    float(
                        (
                            at > 5
                        ).mean()
                    ),

                "v3_fraction_abs_z_gt_10":
                    float(
                        (
                            av > 10
                        ).mean()
                    ),

                "tally_fraction_abs_z_gt_10":
                    float(
                        (
                            at > 10
                        ).mean()
                    ),

                "v3_fraction_abs_z_gt_50":
                    float(
                        (
                            av > 50
                        ).mean()
                    ),

                "tally_fraction_abs_z_gt_50":
                    float(
                        (
                            at > 50
                        ).mean()
                    ),

                "v3_fraction_abs_z_gt_100":
                    float(
                        (
                            av > 100
                        ).mean()
                    ),

                "tally_fraction_abs_z_gt_100":
                    float(
                        (
                            at > 100
                        ).mean()
                    ),

                "v3_raw_min":
                    float(
                        X_v3[
                            :,
                            j
                        ].min()
                    ),

                "v3_raw_max":
                    float(
                        X_v3[
                            :,
                            j
                        ].max()
                    ),

                "tally_raw_min":
                    float(
                        X_tally[
                            :,
                            j
                        ].min()
                    ),

                "tally_raw_max":
                    float(
                        X_tally[
                            :,
                            j
                        ].max()
                    ),
            }
        )


ood_severity = pd.DataFrame(
    ood_rows
)

ood_severity[
    "q99_inflation"
] = (
    ood_severity[
        "tally_abs_z_q99"
    ]
    /
    (
        ood_severity[
            "v3_abs_z_q99"
        ]
        +
        1e-12
    )
)

ood_severity = ood_severity.sort_values(
    [
        "alpha",
        "tally_abs_z_q99",
    ],
    ascending=[
        True,
        False,
    ],
)

ood_severity.to_csv(
    OUT_DIR
    / "standardized_feature_ood_severity.csv",
    index=False,
)


# ============================================================
# 6. Extreme TallyQA score samples
# ============================================================

id_columns = []

for candidate in [
    "manifest_index",
    "question_id",
    "subset",
    "image",
    "ground_truth",
    "baseline_prediction",
    "baseline_correct",
    "selected_alpha",
    "post_prediction",
    "post_correct",
]:
    if candidate in tally.columns:
        id_columns.append(
            candidate
        )


extreme_sample_rows = []
extreme_contrib_rows = []


for alpha in actions:

    scores = pred_tally[
        alpha
    ]

    comp = action_components[
        alpha
    ]

    contrib = comp[
        "contrib_tally"
    ]

    Z = comp[
        "Z_tally"
    ]

    # Highest positive scores are most relevant because the
    # controller intervenes when score exceeds threshold.
    order = np.argsort(
        scores
    )[
        ::-1
    ]

    top_indices = order[
        :
        TOP_K_EXTREME_SAMPLES
    ]

    for rank, idx in enumerate(
        top_indices,
        start=1,
    ):

        row = {
            "alpha":
                alpha,

            "rank":
                rank,

            "row_index":
                int(
                    idx
                ),

            "pipeline_score":
                float(
                    scores[
                        idx
                    ]
                ),

            "best_nonnoop_score":
                float(
                    tq_best_score[
                        idx
                    ]
                ),

            "best_nonnoop_action":
                float(
                    tq_best_nonnoop[
                        idx
                    ]
                ),
        }

        for col in id_columns:
            value = tally.iloc[
                idx
            ][
                col
            ]

            if isinstance(
                value,
                np.generic,
            ):
                value = value.item()

            row[
                col
            ] = value

        extreme_sample_rows.append(
            row
        )

        feature_order = np.argsort(
            np.abs(
                contrib[
                    idx
                ]
            )
        )[
            ::-1
        ][
            :
            TOP_K_CONTRIB_FEATURES
        ]

        for feature_rank, j in enumerate(
            feature_order,
            start=1,
        ):

            extreme_contrib_rows.append(
                {
                    "alpha":
                        alpha,

                    "sample_rank":
                        rank,

                    "row_index":
                        int(
                            idx
                        ),

                    "feature_rank":
                        feature_rank,

                    "feature":
                        feature_names[
                            j
                        ],

                    "raw_feature_value":
                        float(
                            X_tally[
                                idx,
                                j
                            ]
                        ),

                    "standardized_z":
                        float(
                            Z[
                                idx,
                                j
                            ]
                        ),

                    "ridge_coef":
                        float(
                            comp[
                                "coef"
                            ][
                                j
                            ]
                        ),

                    "score_contribution":
                        float(
                            contrib[
                                idx,
                                j
                            ]
                        ),

                    "abs_score_contribution":
                        float(
                            abs(
                                contrib[
                                    idx,
                                    j
                                ]
                            )
                        ),
                }
            )


extreme_samples = pd.DataFrame(
    extreme_sample_rows
)

extreme_contributions = pd.DataFrame(
    extreme_contrib_rows
)

extreme_samples.to_csv(
    OUT_DIR
    / "extreme_tallyqa_action_scores.csv",
    index=False,
)

extreme_contributions.to_csv(
    OUT_DIR
    / "extreme_tallyqa_score_feature_contributions.csv",
    index=False,
)


# ============================================================
# 7. Top coefficient table
# ============================================================

coef_rows = []

for alpha in actions:

    comp = action_components[
        alpha
    ]

    for j, feature in enumerate(
        feature_names
    ):

        coef_rows.append(
            {
                "alpha":
                    alpha,

                "feature":
                    feature,

                "ridge_coef_standardized":
                    float(
                        comp[
                            "coef"
                        ][
                            j
                        ]
                    ),

                "abs_coef":
                    float(
                        abs(
                            comp[
                                "coef"
                            ][
                                j
                            ]
                        )
                    ),

                "scaler_mean":
                    float(
                        comp[
                            "mean"
                        ][
                            j
                        ]
                    ),

                "scaler_scale":
                    float(
                        comp[
                            "scale"
                        ][
                            j
                        ]
                    ),
            }
        )


coefs = pd.DataFrame(
    coef_rows
)

coefs = coefs.sort_values(
    [
        "alpha",
        "abs_coef",
    ],
    ascending=[
        True,
        False,
    ],
)

coefs.to_csv(
    OUT_DIR
    / "frozen_ridge_coefficients.csv",
    index=False,
)


# ============================================================
# 8. Metadata
# ============================================================

metadata = {
    "analysis_type":
        "post_hoc_exact_ridge_feature_attribution",

    "model_inference_performed":
        False,

    "controller_fitting_performed":
        False,

    "hyperparameter_tuning_performed":
        False,

    "frozen_artifacts_modified":
        False,

    "bundle_sha256":
        sha256_file(
            BUNDLE_PATH
        ),

    "v3_results_sha256":
        sha256_file(
            V3_PATH
        ),

    "tallyqa_results_sha256":
        sha256_file(
            TALLY_PATH
        ),

    "feature_count":
        len(
            feature_names
        ),

    "actions":
        actions,

    "threshold":
        threshold,

    "score_tolerance":
        SCORE_TOL,

    "top_k_extreme_samples":
        TOP_K_EXTREME_SAMPLES,

    "top_k_contribution_features":
        TOP_K_CONTRIB_FEATURES,
}


with open(
    OUT_DIR
    / "analysis_metadata.json",
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        sort_keys=True,
    )


# ============================================================
# Print primary findings
# ============================================================

print(
    "=" * 118
)

print(
    "FROZEN SCORE REPRODUCTION"
)

print(
    "=" * 118
)

print(
    score_reproduction.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "FROZEN CONTROLLER DECISION REPRODUCTION"
)

print(
    "=" * 118
)

print(
    decision_reproduction.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "EXACT MEAN SCORE-DRIFT DECOMPOSITION"
)

print(
    "=" * 118
)

print(
    attribution_check.to_string(
        index=False
    )
)


for alpha in actions:

    print(
        "\n" + "=" * 118
    )

    print(
        f"TOP SCORE-DRIFT CONTRIBUTORS "
        f"FOR ALPHA={alpha} — OVERALL"
    )

    print(
        "=" * 118
    )

    view = attribution[
        (
            attribution[
                "alpha"
            ]
            ==
            alpha
        )
        &
        (
            attribution[
                "group"
            ]
            ==
            "overall"
        )
    ].head(
        12
    )

    print(
        view[
            [
                "feature",
                "ridge_coef_standardized",
                "v3_mean_z",
                "tally_mean_z",
                "delta_mean_z",
                "score_drift_contribution",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\n" + "=" * 118
)

print(
    "TOP STANDARDIZED FEATURE OOD OUTLIERS "
    "(ALPHA=4 SCALER)"
)

print(
    "=" * 118
)

view_ood = (
    ood_severity[
        ood_severity[
            "alpha"
        ]
        ==
        4.0
    ]
    .sort_values(
        "tally_abs_z_q99",
        ascending=False,
    )
    .head(
        15
    )
)

print(
    view_ood[
        [
            "feature",
            "v3_abs_z_q99",
            "tally_abs_z_q99",
            "tally_abs_z_q999",
            "tally_abs_z_max",
            "tally_fraction_abs_z_gt_10",
            "tally_fraction_abs_z_gt_100",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "TOP 20 TALLYQA ALPHA=4 SCORES"
)

print(
    "=" * 118
)

view_extreme = (
    extreme_samples[
        extreme_samples[
            "alpha"
        ]
        ==
        4.0
    ]
    .head(
        20
    )
)

display_cols = [
    c
    for c in [
        "rank",
        "pipeline_score",
        "manifest_index",
        "question_id",
        "subset",
        "ground_truth",
        "baseline_prediction",
        "baseline_correct",
        "selected_alpha",
        "post_prediction",
        "post_correct",
    ]
    if c in view_extreme.columns
]

print(
    view_extreme[
        display_cols
    ].to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "TOP FEATURE CONTRIBUTIONS FOR "
    "THE #1 EXTREME ALPHA=4 SAMPLE"
)

print(
    "=" * 118
)

top_one = (
    extreme_contributions[
        (
            extreme_contributions[
                "alpha"
            ]
            ==
            4.0
        )
        &
        (
            extreme_contributions[
                "sample_rank"
            ]
            ==
            1
        )
    ]
    .sort_values(
        "feature_rank"
    )
)

print(
    top_one[
        [
            "feature_rank",
            "feature",
            "raw_feature_value",
            "standardized_z",
            "ridge_coef",
            "score_contribution",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n" + "=" * 118
)

print(
    "RIDGE FEATURE ATTRIBUTION FORENSICS COMPLETE"
)

print(
    "=" * 118
)

print(
    "No model inference performed."
)

print(
    "No controller fitting performed."
)

print(
    "No hyperparameter tuning performed."
)

print(
    "No frozen artifact modified."
)

print(
    "\nSaved to:"
)

print(
    OUT_DIR
)

