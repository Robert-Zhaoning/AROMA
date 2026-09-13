#!/usr/bin/env python3

"""
AROMA Phase II-A:
Support-Projected Utility Control (SPUC-v1)
===========================================

Motivation
----------
The frozen synthetic-trained controller uses:

    StandardScaler -> Ridge

and natural-domain forensic analysis showed catastrophic
out-of-support standardized feature values, especially for
high-numeral probability features.

SPUC-v1 preserves:
- the frozen feature representation,
- StandardScaler parameters,
- Ridge coefficients,
- Ridge intercepts,
- action space,
- intervention threshold,
- actuator head,

and changes only the standardized feature presented to Ridge:

    z_j -> clip(z_j, L_j, U_j)

where L_j and U_j are the exact empirical min/max standardized
feature values observed in the reconstructed v2 controller
training set.

IMPORTANT
---------
This is a post-hoc DEVELOPMENT experiment.

TallyQA-4000 is no longer an untouched confirmation set and is
used here only for method development / forensic evaluation.

No model inference is performed.
No controller parameters are fitted.
No threshold is tuned.
No support parameter is tuned.
No frozen artifact is modified.
"""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path("/workspace/AromaExperiments")

BUNDLE_PATH = (
    ROOT
    / "outputs"
    / "proc_count_causal_v2"
    / "controller"
    / "final_frozen_controller"
    / "aroma_cardinality_controller.joblib"
)

SUPPORT_PATH = (
    ROOT
    / "outputs"
    / "proc_count_causal_v2"
    / "controller"
    / "training_support_reconstruction"
    / "v2_controller_training_support.csv"
)

V3_PATH = (
    ROOT
    / "outputs"
    / "proc_count_causal_v3"
    / "final_frozen_controller"
    / "v3_final_results.csv"
)

TALLY_PATH = (
    ROOT
    / "outputs"
    / "tallyqa_natural_ood_v1"
    / "final_frozen_controller"
    / "tallyqa_final_results.csv"
)

TALLY_MATRIX_PATH = (
    ROOT
    / "outputs"
    / "tallyqa_natural_ood_v1"
    / "forensics"
    / "action_oracle"
    / "action_matrix.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "phase2_spuc_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
        mapping
    )

    if unknown:
        raise RuntimeError(
            f"Unknown bool values: "
            f"{sorted(unknown)}"
        )

    return (
        s.map(mapping)
        .astype(bool)
    )


def quantiles(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    return {
        "mean":
            float(
                np.mean(x)
            ),

        "median":
            float(
                np.median(x)
            ),

        "q90":
            float(
                np.quantile(
                    x,
                    0.90,
                )
            ),

        "q95":
            float(
                np.quantile(
                    x,
                    0.95,
                )
            ),

        "q99":
            float(
                np.quantile(
                    x,
                    0.99,
                )
            ),

        "max":
            float(
                np.max(x)
            ),

        "min":
            float(
                np.min(x)
            ),
    }


# ============================================================
# Load artifacts
# ============================================================

for path in [
    BUNDLE_PATH,
    SUPPORT_PATH,
    V3_PATH,
    TALLY_PATH,
    TALLY_MATRIX_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

bundle = joblib.load(
    BUNDLE_PATH
)

support = pd.read_csv(
    SUPPORT_PATH
)

v3 = pd.read_csv(
    V3_PATH
)

tally = pd.read_csv(
    TALLY_PATH
)

matrix = pd.read_csv(
    TALLY_MATRIX_PATH
)


# ============================================================
# Frozen specification
# ============================================================

feature_names = list(
    bundle[
        "feature_names"
    ]
)

feature_cols = [
    f"feature__{f}"
    for f in feature_names
]

actions = [
    float(a)
    for a in bundle[
        "nonnoop_actions"
    ]
]

all_actions = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]

threshold = float(
    bundle[
        "threshold"
    ]
)

if actions != [
    0.0,
    1.5,
    2.0,
    4.0,
]:
    raise RuntimeError(
        f"Unexpected actions: {actions}"
    )

if threshold != 0.1:
    raise RuntimeError(
        f"Unexpected threshold: {threshold}"
    )


# ============================================================
# Verify support feature order
# ============================================================

support = (
    support
    .set_index(
        "feature"
    )
    .reindex(
        feature_names
    )
)

if support.index.tolist() != feature_names:
    raise RuntimeError(
        "Support feature ordering failed."
    )

if support[
    [
        "z_min",
        "z_max",
    ]
].isna().any().any():
    raise RuntimeError(
        "Missing support boundaries."
    )

z_lower = (
    support[
        "z_min"
    ]
    .to_numpy(
        dtype=float
    )
)

z_upper = (
    support[
        "z_max"
    ]
    .to_numpy(
        dtype=float
    )
)

if not (
    z_lower
    <=
    z_upper
).all():
    raise RuntimeError(
        "Invalid support interval."
    )


# ============================================================
# Verify common frozen scaler
# ============================================================

scalers = []

for alpha in actions:

    scaler = (
        bundle[
            "models"
        ][alpha]
        .named_steps[
            "scale"
        ]
    )

    scalers.append(
        scaler
    )

reference_scaler = scalers[0]

reference_mean = np.asarray(
    reference_scaler.mean_,
    dtype=float,
)

reference_scale = np.asarray(
    reference_scaler.scale_,
    dtype=float,
)

for scaler in scalers[1:]:

    if not np.array_equal(
        np.asarray(
            scaler.mean_
        ),
        reference_mean,
    ):
        raise RuntimeError(
            "Frozen scaler means differ "
            "across actions."
        )

    if not np.array_equal(
        np.asarray(
            scaler.scale_
        ),
        reference_scale,
    ):
        raise RuntimeError(
            "Frozen scaler scales differ "
            "across actions."
        )


# ============================================================
# Check reconstructed support is compatible with scaler
# ============================================================

support_mean = (
    support[
        "train_mean"
    ]
    .to_numpy(
        dtype=float
    )
)

support_scale = (
    support[
        "train_scale"
    ]
    .to_numpy(
        dtype=float
    )
)

if not np.allclose(
    support_mean,
    reference_mean,
    rtol=1e-10,
    atol=1e-12,
):
    raise RuntimeError(
        "Support means do not reproduce "
        "frozen scaler means."
    )

if not np.allclose(
    support_scale,
    reference_scale,
    rtol=1e-10,
    atol=1e-12,
):
    raise RuntimeError(
        "Support scales do not reproduce "
        "frozen scaler scales."
    )


# ============================================================
# Extract features
# ============================================================

for name, df in [
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
            f"{name} missing features: "
            f"{missing}"
        )


X_v3 = (
    v3[
        feature_cols
    ]
    .to_numpy(
        dtype=float
    )
)

X_tq = (
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
        "Non-finite v3 feature."
    )

if not np.isfinite(
    X_tq
).all():
    raise RuntimeError(
        "Non-finite TallyQA feature."
    )


# ============================================================
# Standardize + empirical support projection
# ============================================================

def standardized_and_projected(
    X,
):

    Z = (
        X
        -
        reference_mean
    ) / reference_scale

    Z_projected = np.clip(
        Z,
        z_lower,
        z_upper,
    )

    clipped_mask = (
        ~np.isclose(
            Z,
            Z_projected,
            rtol=0.0,
            atol=1e-12,
        )
    )

    return (
        Z,
        Z_projected,
        clipped_mask,
    )


Z_v3, Zp_v3, clip_v3 = (
    standardized_and_projected(
        X_v3
    )
)

Z_tq, Zp_tq, clip_tq = (
    standardized_and_projected(
        X_tq
    )
)


# ============================================================
# Compute original + SPUC scores
# ============================================================

def compute_scores(
    X,
    Z_projected,
):

    original = {}
    projected = {}

    for alpha in actions:

        pipe = bundle[
            "models"
        ][alpha]

        ridge = (
            pipe.named_steps[
                "ridge"
            ]
        )

        original[
            alpha
        ] = np.asarray(
            pipe.predict(
                X
            ),
            dtype=float,
        )

        coef = np.asarray(
            ridge.coef_,
            dtype=float,
        )

        intercept = float(
            ridge.intercept_
        )

        projected[
            alpha
        ] = (
            intercept
            +
            Z_projected.dot(
                coef
            )
        )

    return (
        original,
        projected,
    )


orig_v3, spuc_v3 = (
    compute_scores(
        X_v3,
        Zp_v3,
    )
)

orig_tq, spuc_tq = (
    compute_scores(
        X_tq,
        Zp_tq,
    )
)


# ============================================================
# Verify original scores reproduce frozen artifacts
# ============================================================

score_cols = {
    0.0:
        "score_alpha_0",

    1.5:
        "score_alpha_1p5",

    2.0:
        "score_alpha_2",

    4.0:
        "score_alpha_4",
}


for domain_name, df, scores in [
    (
        "v3",
        v3,
        orig_v3,
    ),
    (
        "TallyQA",
        tally,
        orig_tq,
    ),
]:

    for alpha in actions:

        stored = pd.to_numeric(
            df[
                score_cols[
                    alpha
                ]
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        max_diff = float(
            np.max(
                np.abs(
                    stored
                    -
                    scores[
                        alpha
                    ]
                )
            )
        )

        if max_diff > 1e-6:
            raise RuntimeError(
                f"{domain_name} alpha={alpha} "
                f"score reproduction failed: "
                f"{max_diff}"
            )


# ============================================================
# Controller decision
# ============================================================

def decide(
    scores,
    n,
):

    score_matrix = np.column_stack(
        [
            scores[
                a
            ]
            for a in actions
        ]
    )

    best_index = np.argmax(
        score_matrix,
        axis=1,
    )

    best_score = score_matrix[
        np.arange(
            n
        ),
        best_index,
    ]

    best_action = np.array(
        [
            actions[i]
            for i in best_index
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


orig_v3_alpha, \
orig_v3_best_score, \
orig_v3_best_action = (
    decide(
        orig_v3,
        len(v3),
    )
)

spuc_v3_alpha, \
spuc_v3_best_score, \
spuc_v3_best_action = (
    decide(
        spuc_v3,
        len(v3),
    )
)

orig_tq_alpha, \
orig_tq_best_score, \
orig_tq_best_action = (
    decide(
        orig_tq,
        len(tally),
    )
)

spuc_tq_alpha, \
spuc_tq_best_score, \
spuc_tq_best_action = (
    decide(
        spuc_tq,
        len(tally),
    )
)


# ============================================================
# Original selected-action reproduction
# ============================================================

stored_v3_alpha = pd.to_numeric(
    v3[
        "selected_alpha"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

stored_tq_alpha = pd.to_numeric(
    tally[
        "selected_alpha"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)

v3_original_mismatch = int(
    (
        ~np.isclose(
            orig_v3_alpha,
            stored_v3_alpha,
        )
    ).sum()
)

tq_original_mismatch = int(
    (
        ~np.isclose(
            orig_tq_alpha,
            stored_tq_alpha,
        )
    ).sum()
)

if v3_original_mismatch != 0:
    raise RuntimeError(
        "Original v3 controller reconstruction failed."
    )

if tq_original_mismatch != 0:
    raise RuntimeError(
        "Original TallyQA controller reconstruction failed."
    )


# ============================================================
# V3 policy-stability analysis
#
# We currently have the archived outcome only for the original
# selected action. If SPUC chooses a different action, its new
# outcome cannot be inferred without all-action predictions.
#
# Therefore:
# - unchanged selected actions have exact archived outcomes;
# - changed actions are explicitly marked unresolved here.
# ============================================================

v3_changed = (
    ~np.isclose(
        spuc_v3_alpha,
        stored_v3_alpha,
    )
)

v3_changed_n = int(
    v3_changed.sum()
)

v3_agreement = float(
    (
        ~v3_changed
    ).mean()
)

v3_post_correct_old = as_bool(
    v3[
        "post_correct"
    ]
).to_numpy()

v3_known_correct_unchanged = int(
    v3_post_correct_old[
        ~v3_changed
    ].sum()
)

v3_lower_bound_correct = (
    v3_known_correct_unchanged
)

v3_upper_bound_correct = (
    v3_known_correct_unchanged
    +
    v3_changed_n
)

v3_lower_bound_acc = (
    v3_lower_bound_correct
    /
    len(v3)
)

v3_upper_bound_acc = (
    v3_upper_bound_correct
    /
    len(v3)
)


# ============================================================
# Exact TallyQA SPUC outcome from completed 5-action matrix
# ============================================================

matrix[
    "alpha"
] = pd.to_numeric(
    matrix[
        "alpha"
    ],
    errors="raise",
).astype(float)

if len(matrix) != 20000:
    raise RuntimeError(
        "Expected 20,000 TallyQA action rows."
    )

if matrix[
    "question_id"
].nunique() != 4000:
    raise RuntimeError(
        "Expected 4000 TallyQA question IDs."
    )

tq_decisions = pd.DataFrame(
    {
        "question_id":
            tally[
                "question_id"
            ]
            .astype(int)
            .to_numpy(),

        "spuc_selected_alpha":
            spuc_tq_alpha,

        "spuc_selected_score":
            spuc_tq_best_score,

        "original_selected_alpha":
            stored_tq_alpha,

        "original_selected_score":
            pd.to_numeric(
                tally[
                    "selected_score"
                ],
                errors="raise",
            ).to_numpy(
                dtype=float
            ),
    }
)

selected_outcomes = (
    tq_decisions.merge(
        matrix[
            [
                "question_id",
                "alpha",
                "ground_truth",
                "baseline_prediction",
                "action_prediction",
            ]
        ],
        left_on=[
            "question_id",
            "spuc_selected_alpha",
        ],
        right_on=[
            "question_id",
            "alpha",
        ],
        how="inner",
    )
)

if len(
    selected_outcomes
) != 4000:
    raise RuntimeError(
        "Could not recover exactly one "
        "TallyQA SPUC outcome per sample."
    )

if selected_outcomes[
    "question_id"
].nunique() != 4000:
    raise RuntimeError(
        "Duplicate/missing SPUC TallyQA outcomes."
    )

selected_outcomes[
    "baseline_correct"
] = (
    selected_outcomes[
        "baseline_prediction"
    ]
    ==
    selected_outcomes[
        "ground_truth"
    ]
)

selected_outcomes[
    "spuc_correct"
] = (
    selected_outcomes[
        "action_prediction"
    ]
    ==
    selected_outcomes[
        "ground_truth"
    ]
)

selected_outcomes[
    "repair"
] = (
    (~selected_outcomes[
        "baseline_correct"
    ])
    &
    selected_outcomes[
        "spuc_correct"
    ]
)

selected_outcomes[
    "break_case"
] = (
    selected_outcomes[
        "baseline_correct"
    ]
    &
    (~selected_outcomes[
        "spuc_correct"
    ])
)


tq_baseline_accuracy = float(
    selected_outcomes[
        "baseline_correct"
    ].mean()
)

tq_spuc_accuracy = float(
    selected_outcomes[
        "spuc_correct"
    ].mean()
)

tq_repairs = int(
    selected_outcomes[
        "repair"
    ].sum()
)

tq_breaks = int(
    selected_outcomes[
        "break_case"
    ].sum()
)

tq_net = (
    tq_repairs
    -
    tq_breaks
)

tq_spuc_intervention_rate = float(
    (
        selected_outcomes[
            "spuc_selected_alpha"
        ]
        !=
        1.0
    ).mean()
)

tq_original_accuracy = float(
    as_bool(
        tally[
            "post_correct"
        ]
    ).mean()
)


# ============================================================
# Main policy summary
# ============================================================

policy_summary = pd.DataFrame(
    [
        {
            "domain":
                "v3_confirmation",

            "n":
                len(v3),

            "baseline_accuracy":
                float(
                    as_bool(
                        v3[
                            "baseline_correct"
                        ]
                    ).mean()
                ),

            "original_post_accuracy":
                float(
                    as_bool(
                        v3[
                            "post_correct"
                        ]
                    ).mean()
                ),

            "original_accuracy_change":
                float(
                    as_bool(
                        v3[
                            "post_correct"
                        ]
                    ).mean()
                    -
                    as_bool(
                        v3[
                            "baseline_correct"
                        ]
                    ).mean()
                ),

            "spuc_exact_post_accuracy":
                np.nan
                if v3_changed_n
                else float(
                    as_bool(
                        v3[
                            "post_correct"
                        ]
                    ).mean()
                ),

            "spuc_post_accuracy_lower_bound":
                v3_lower_bound_acc,

            "spuc_post_accuracy_upper_bound":
                v3_upper_bound_acc,

            "original_intervention_rate":
                float(
                    (
                        stored_v3_alpha
                        !=
                        1.0
                    ).mean()
                ),

            "spuc_intervention_rate":
                float(
                    (
                        spuc_v3_alpha
                        !=
                        1.0
                    ).mean()
                ),

            "policy_agreement":
                v3_agreement,

            "changed_selected_actions":
                v3_changed_n,

            "repairs":
                np.nan,

            "breaks":
                np.nan,

            "net_repairs":
                np.nan,
        },

        {
            "domain":
                "tallyqa_posthoc_development",

            "n":
                len(tally),

            "baseline_accuracy":
                tq_baseline_accuracy,

            "original_post_accuracy":
                tq_original_accuracy,

            "original_accuracy_change":
                (
                    tq_original_accuracy
                    -
                    tq_baseline_accuracy
                ),

            "spuc_exact_post_accuracy":
                tq_spuc_accuracy,

            "spuc_post_accuracy_lower_bound":
                tq_spuc_accuracy,

            "spuc_post_accuracy_upper_bound":
                tq_spuc_accuracy,

            "original_intervention_rate":
                float(
                    (
                        stored_tq_alpha
                        !=
                        1.0
                    ).mean()
                ),

            "spuc_intervention_rate":
                tq_spuc_intervention_rate,

            "policy_agreement":
                float(
                    np.isclose(
                        spuc_tq_alpha,
                        stored_tq_alpha,
                    ).mean()
                ),

            "changed_selected_actions":
                int(
                    (
                        ~np.isclose(
                            spuc_tq_alpha,
                            stored_tq_alpha,
                        )
                    ).sum()
                ),

            "repairs":
                tq_repairs,

            "breaks":
                tq_breaks,

            "net_repairs":
                tq_net,
        },
    ]
)

policy_summary.to_csv(
    OUT_DIR
    / "policy_summary.csv",
    index=False,
)


# ============================================================
# Action distributions
# ============================================================

action_rows = []

for domain, old_alpha, new_alpha in [
    (
        "v3_confirmation",
        stored_v3_alpha,
        spuc_v3_alpha,
    ),
    (
        "tallyqa_posthoc_development",
        stored_tq_alpha,
        spuc_tq_alpha,
    ),
]:

    for policy_name, selected in [
        (
            "original",
            old_alpha,
        ),
        (
            "spuc_v1",
            new_alpha,
        ),
    ]:

        for alpha in all_actions:

            count = int(
                np.isclose(
                    selected,
                    alpha,
                ).sum()
            )

            action_rows.append(
                {
                    "domain":
                        domain,

                    "policy":
                        policy_name,

                    "alpha":
                        alpha,

                    "count":
                        count,

                    "fraction":
                        count
                        /
                        len(
                            selected
                        ),
                }
            )


action_distribution = pd.DataFrame(
    action_rows
)

action_distribution.to_csv(
    OUT_DIR
    / "action_distribution.csv",
    index=False,
)


# ============================================================
# Score-range comparison
# ============================================================

score_rows = []

for domain, original, projected in [
    (
        "v3_confirmation",
        orig_v3,
        spuc_v3,
    ),
    (
        "tallyqa_posthoc_development",
        orig_tq,
        spuc_tq,
    ),
]:

    for alpha in actions:

        for policy, scores in [
            (
                "original",
                original[
                    alpha
                ],
            ),
            (
                "spuc_v1",
                projected[
                    alpha
                ],
            ),
        ]:

            q = quantiles(
                scores
            )

            score_rows.append(
                {
                    "domain":
                        domain,

                    "alpha":
                        alpha,

                    "policy":
                        policy,

                    **q,

                    "fraction_gt_threshold":
                        float(
                            (
                                scores
                                >
                                threshold
                            ).mean()
                        ),
                }
            )


score_summary = pd.DataFrame(
    score_rows
)

score_summary.to_csv(
    OUT_DIR
    / "score_range_summary.csv",
    index=False,
)


# ============================================================
# Clipping summaries
# ============================================================

clip_domain_rows = []

for domain, clipped in [
    (
        "v3_confirmation",
        clip_v3,
    ),
    (
        "tallyqa_posthoc_development",
        clip_tq,
    ),
]:

    clip_domain_rows.append(
        {
            "domain":
                domain,

            "n":
                clipped.shape[0],

            "n_features":
                clipped.shape[1],

            "samples_with_any_clipped_feature":
                int(
                    clipped.any(
                        axis=1
                    ).sum()
                ),

            "fraction_samples_any_clipped":
                float(
                    clipped.any(
                        axis=1
                    ).mean()
                ),

            "total_clipped_cells":
                int(
                    clipped.sum()
                ),

            "fraction_all_feature_cells_clipped":
                float(
                    clipped.mean()
                ),
        }
    )


clipping_domain_summary = pd.DataFrame(
    clip_domain_rows
)

clipping_domain_summary.to_csv(
    OUT_DIR
    / "clipping_domain_summary.csv",
    index=False,
)


feature_clip_rows = []

for domain, Z, Zp, clipped in [
    (
        "v3_confirmation",
        Z_v3,
        Zp_v3,
        clip_v3,
    ),
    (
        "tallyqa_posthoc_development",
        Z_tq,
        Zp_tq,
        clip_tq,
    ),
]:

    for j, feature in enumerate(
        feature_names
    ):

        lower = (
            Z[
                :,
                j
            ]
            <
            z_lower[
                j
            ]
        )

        upper = (
            Z[
                :,
                j
            ]
            >
            z_upper[
                j
            ]
        )

        feature_clip_rows.append(
            {
                "domain":
                    domain,

                "feature":
                    feature,

                "z_lower":
                    float(
                        z_lower[j]
                    ),

                "z_upper":
                    float(
                        z_upper[j]
                    ),

                "n_low_clipped":
                    int(
                        lower.sum()
                    ),

                "n_high_clipped":
                    int(
                        upper.sum()
                    ),

                "n_any_clipped":
                    int(
                        clipped[
                            :,
                            j
                        ].sum()
                    ),

                "fraction_any_clipped":
                    float(
                        clipped[
                            :,
                            j
                        ].mean()
                    ),

                "raw_z_abs_max":
                    float(
                        np.abs(
                            Z[
                                :,
                                j
                            ]
                        ).max()
                    ),

                "projected_z_abs_max":
                    float(
                        np.abs(
                            Zp[
                                :,
                                j
                            ]
                        ).max()
                    ),

                "mean_abs_projection_change":
                    float(
                        np.abs(
                            Z[
                                :,
                                j
                            ]
                            -
                            Zp[
                                :,
                                j
                            ]
                        ).mean()
                    ),
            }
        )


feature_clipping = pd.DataFrame(
    feature_clip_rows
)

feature_clipping.to_csv(
    OUT_DIR
    / "feature_clipping_summary.csv",
    index=False,
)


# ============================================================
# Save per-sample results
# ============================================================

tq_per_sample = selected_outcomes[
    [
        "question_id",
        "ground_truth",
        "baseline_prediction",
        "original_selected_alpha",
        "original_selected_score",
        "spuc_selected_alpha",
        "spuc_selected_score",
        "action_prediction",
        "baseline_correct",
        "spuc_correct",
        "repair",
        "break_case",
    ]
].copy()

tq_per_sample.to_csv(
    OUT_DIR
    / "tallyqa_spuc_v1_per_sample.csv",
    index=False,
)


v3_per_sample = pd.DataFrame(
    {
        "row_index":
            np.arange(
                len(v3)
            ),

        "original_selected_alpha":
            stored_v3_alpha,

        "spuc_selected_alpha":
            spuc_v3_alpha,

        "original_best_score":
            orig_v3_best_score,

        "spuc_best_score":
            spuc_v3_best_score,

        "action_changed":
            v3_changed,

        "archived_post_correct":
            v3_post_correct_old,
    }
)

for identifier in [
    "manifest_index",
    "sample_id",
    "condition",
    "ground_truth",
    "baseline_prediction",
]:

    if identifier in v3.columns:

        v3_per_sample[
            identifier
        ] = v3[
            identifier
        ].values


v3_per_sample.to_csv(
    OUT_DIR
    / "v3_spuc_v1_policy_changes.csv",
    index=False,
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "method":
        "SPUC-v1 empirical min-max support projection",

    "development_status":
        (
            "post-hoc development only; "
            "TallyQA-4000 is not an untouched "
            "confirmation set for SPUC-v1"
        ),

    "model_inference_performed":
        False,

    "controller_fitting_performed":
        False,

    "threshold_tuning_performed":
        False,

    "support_parameter_tuning_performed":
        False,

    "frozen_artifacts_modified":
        False,

    "threshold":
        threshold,

    "actions":
        all_actions,

    "feature_count":
        len(
            feature_names
        ),

    "support_definition":
        (
            "per-feature empirical min/max "
            "standardized z observed in exact "
            "reconstructed v2 controller "
            "training matrix"
        ),

    "bundle_sha256":
        sha256_file(
            BUNDLE_PATH
        ),

    "support_sha256":
        sha256_file(
            SUPPORT_PATH
        ),

    "v3_sha256":
        sha256_file(
            V3_PATH
        ),

    "tallyqa_final_sha256":
        sha256_file(
            TALLY_PATH
        ),

    "tallyqa_action_matrix_sha256":
        sha256_file(
            TALLY_MATRIX_PATH
        ),
}


with open(
    OUT_DIR
    / "metadata.json",
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        sort_keys=True,
    )


# ============================================================
# Print results
# ============================================================

print("=" * 118)
print("SPUC-v1 PRE-SPECIFIED CONFIGURATION")
print("=" * 118)

print("Projection: empirical v2 standardized min/max")
print("Threshold:", threshold)
print("Actions:", all_actions)
print("Controller parameters changed: NO")
print("Model inference performed: NO")


print("\n" + "=" * 118)
print("POLICY / OUTCOME SUMMARY")
print("=" * 118)

print(
    policy_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("ACTION DISTRIBUTION")
print("=" * 118)

print(
    action_distribution.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("CLIPPING DOMAIN SUMMARY")
print("=" * 118)

print(
    clipping_domain_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("TOP 20 CLIPPED FEATURES — TALLYQA")
print("=" * 118)

print(
    feature_clipping[
        feature_clipping[
            "domain"
        ]
        ==
        "tallyqa_posthoc_development"
    ]
    .sort_values(
        [
            "fraction_any_clipped",
            "mean_abs_projection_change",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(20)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("SCORE RANGE COMPARISON — TALLYQA")
print("=" * 118)

print(
    score_summary[
        score_summary[
            "domain"
        ]
        ==
        "tallyqa_posthoc_development"
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 118)
print("V3 CHANGED-ACTION DETAILS")
print("=" * 118)

print(
    "Changed actions:",
    v3_changed_n,
    "/",
    len(v3),
)

print(
    "Policy agreement:",
    v3_agreement,
)

if v3_changed_n == 0:

    print(
        "SPUC-v1 preserves the exact "
        "archived v3 outcome."
    )

else:

    print(
        "Exact outcomes for changed v3 actions "
        "are not available from the archived "
        "single-selected-action result."
    )

    print(
        "Current exact post-accuracy bounds:",
        v3_lower_bound_acc,
        "to",
        v3_upper_bound_acc,
    )


print("\n" + "=" * 118)
print("TALLYQA POST-HOC DEVELOPMENT RESULT")
print("=" * 118)

print(
    "Baseline accuracy:",
    tq_baseline_accuracy,
)

print(
    "Original frozen controller accuracy:",
    tq_original_accuracy,
)

print(
    "SPUC-v1 accuracy:",
    tq_spuc_accuracy,
)

print(
    "SPUC-v1 accuracy change vs baseline:",
    tq_spuc_accuracy
    -
    tq_baseline_accuracy,
)

print(
    "Repairs:",
    tq_repairs,
)

print(
    "Breaks:",
    tq_breaks,
)

print(
    "Net repairs:",
    tq_net,
)

print(
    "Intervention rate:",
    tq_spuc_intervention_rate,
)

print(
    "Original/SPUC policy agreement:",
    float(
        np.isclose(
            spuc_tq_alpha,
            stored_tq_alpha,
        ).mean()
    ),
)


print("\n" + "=" * 118)
print("SPUC-v1 DEVELOPMENT EVALUATION COMPLETE")
print("=" * 118)

print("No model inference performed.")
print("No controller fitting performed.")
print("No threshold tuning performed.")
print("No support parameter tuning performed.")
print("No frozen artifact modified.")

print("\nSaved to:")
print(OUT_DIR)

