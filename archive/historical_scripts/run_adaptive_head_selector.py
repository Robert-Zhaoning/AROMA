from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore")


# ============================================================
# Paths
# ============================================================

PROFILE_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "dam_head_profiles/"
    "head_response_profiles.csv"
)

FEATURE_PATH = Path(
    "outputs/proc_count_causal_v1/router/"
    "full_probability_response/"
    "full_response_router_features.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "adaptive_head_selector"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PRED_OUT = (
    OUT_DIR
    / "adaptive_head_selector_predictions.csv"
)

SUMMARY_OUT = (
    OUT_DIR
    / "adaptive_head_selector_summary.csv"
)

FOLD_OUT = (
    OUT_DIR
    / "adaptive_head_selector_folds.csv"
)

CONFIG_OUT = (
    OUT_DIR
    / "adaptive_head_selector_config.json"
)


SEED = 20260912

RIDGE_ALPHAS = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
]


CANDIDATE_HEADS = [
    "L33H1",
    "L3H4",
    "L18H13",
    "L8H30",
    "L3H11",
    "L13H11",
    "L33H21",
]


# ============================================================
# Load
# ============================================================

profiles = pd.read_csv(
    PROFILE_PATH
)

features = pd.read_csv(
    FEATURE_PATH
)

print("=" * 120)
print("AROMA ADAPTIVE HEAD SELECTOR")
print("=" * 120)

print(
    "Head-profile rows:",
    len(profiles),
)

print(
    "Feature rows:",
    len(features),
)


# ============================================================
# Strict integrity audit
# ============================================================

required_profile_cols = {
    "pair_id",
    "role",
    "sample_id",
    "head_name",
    "delta_logp",
}

missing = (
    required_profile_cols
    - set(profiles.columns)
)

if missing:
    raise RuntimeError(
        f"Missing profile columns: {sorted(missing)}"
    )

assert len(profiles) == 378
assert profiles["sample_id"].nunique() == 54
assert profiles["pair_id"].nunique() == 27

assert (
    profiles[
        ["sample_id", "head_name"]
    ]
    .duplicated()
    .sum()
    == 0
)

assert set(
    profiles["head_name"]
    .astype(str)
    .unique()
) == set(CANDIDATE_HEADS)

assert len(features) == 54
assert features["sample_id"].nunique() == 54
assert features["pair_id"].nunique() == 27

profile_ids = set(
    profiles["sample_id"].astype(str)
)

feature_ids = set(
    features["sample_id"].astype(str)
)

assert profile_ids == feature_ids

discovery = {
    "pccv1_n05_row_r00",
    "pccv1_n05_row_r01",
}

assert not (
    discovery
    & profile_ids
)

print(
    "Integrity audit: PASS"
)


# ============================================================
# Build sample × head learning table
# ============================================================

# Safe passive numeral features.
numeral_features = [
    "baseline_prediction",
    "numeral_entropy",
    "numeral_top1",
    "numeral_top1_prob",
    "numeral_top2",
    "numeral_top2_prob",
    "numeral_margin",
] + [
    f"numeral_prob_{n}"
    for n in range(1, 11)
]


# Global response geometry.
global_features = [
    c
    for c in features.columns
    if c.startswith("global_")
]


# Alpha-level response geometry across heads.
alpha_features = [
    c
    for c in features.columns
    if c.startswith("alpha_")
]


# ------------------------------------------------------------
# For every sample/head row, convert that head's own
# response features to generic names.
#
# This allows one model to learn:
#
#     reward = f(sample state,
#                candidate head identity,
#                candidate head response)
#
# without giving it GT.
# ------------------------------------------------------------

rows = []

for _, pr in profiles.iterrows():

    sample_id = str(
        pr["sample_id"]
    )

    head_name = str(
        pr["head_name"]
    )

    frow = features[
        features["sample_id"].astype(str)
        == sample_id
    ].iloc[0]

    row = {
        "pair_id":
            int(pr["pair_id"]),

        "role":
            str(pr["role"]),

        "sample_id":
            sample_id,

        "head_name":
            head_name,

        "target_delta_logp":
            float(pr["delta_logp"]),
    }

    # Passive numeral state.
    for c in numeral_features:
        row[c] = float(
            frow[c]
        )

    # Global response state.
    for c in global_features:
        row[c] = float(
            frow[c]
        )

    # Across-head alpha summaries.
    for c in alpha_features:
        row[c] = float(
            frow[c]
        )

    # --------------------------------------------------------
    # Head-specific aggregate features.
    # Convert:
    #   L3H4_agg_mean_l1
    # to:
    #   head_agg_mean_l1
    # --------------------------------------------------------

    prefix = (
        f"{head_name}_agg_"
    )

    head_agg_cols = [
        c
        for c in features.columns
        if c.startswith(prefix)
    ]

    for c in head_agg_cols:

        generic = (
            "head_agg_"
            + c[len(prefix):]
        )

        row[generic] = float(
            frow[c]
        )

    # --------------------------------------------------------
    # Head-specific per-alpha compact metrics.
    #
    # We deliberately exclude raw dp1...dp10 here to keep
    # dimensionality controlled.
    # --------------------------------------------------------

    for alpha in [
        0.5,
        0.75,
        1.25,
        1.5,
    ]:

        a = str(alpha).replace(
            ".",
            "p",
        )

        pfx = (
            f"{head_name}_a{a}_"
        )

        candidate_cols = [
            c
            for c in features.columns
            if c.startswith(pfx)
        ]

        # Exclude raw 10-way delta probabilities.
        compact_cols = [
            c
            for c in candidate_cols
            if "_dp" not in c
        ]

        for c in compact_cols:

            suffix = c[
                len(pfx):
            ]

            generic = (
                f"head_a{a}_"
                f"{suffix}"
            )

            row[generic] = float(
                frow[c]
            )

    rows.append(row)


data = pd.DataFrame(
    rows
)

assert len(data) == 378
assert data.isna().sum().sum() == 0

coverage = (
    data.groupby("sample_id")[
        "head_name"
    ]
    .nunique()
)

assert coverage.min() == 7
assert coverage.max() == 7


# ============================================================
# Feature families
# ============================================================

head_specific_features = [
    c
    for c in data.columns
    if (
        c.startswith("head_agg_")
        or c.startswith("head_a")
    )
]


families = {
    # Very conservative baseline:
    # sample numeral state + head identity.
    "numeral_head":
        numeral_features,

    # Main adaptive policy:
    # numeral + global response +
    # candidate-head-specific response.
    "compact_response_policy":
        (
            numeral_features
            + global_features
            + alpha_features
            + head_specific_features
        ),

    # Response without passive numeral state.
    "response_policy_only":
        (
            global_features
            + alpha_features
            + head_specific_features
        ),
}


print("\nFeature families:")

for name, cols in families.items():

    print(
        f"{name:28s}: "
        f"{len(cols)} numeric + head identity"
    )


# ============================================================
# Leakage audit
# ============================================================

forbidden_tokens = [
    "ground_truth",
    "baseline_correct",
    "oracle_",
    "object_mass",
    "object_enrichment",
    "_gt_",
    "target_delta_logp",
    "role",
]

for family, cols in families.items():

    bad = [
        c
        for c in cols
        if any(
            token in c
            for token in forbidden_tokens
        )
    ]

    if bad:
        raise RuntimeError(
            f"{family}: leakage columns: {bad}"
        )


# ============================================================
# Model
# ============================================================

def make_model(
    numeric_cols,
    ridge_alpha,
):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_cols,
            ),
            (
                "head",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["head_name"],
            ),
        ],
        remainder="drop",
    )

    return Pipeline([
        (
            "preprocess",
            preprocessor,
        ),
        (
            "regressor",
            Ridge(
                alpha=ridge_alpha,
            ),
        ),
    ])


# ============================================================
# Selection metric
# ============================================================

def selection_metrics(
    frame,
    predicted_col,
):

    selected_rows = []

    for sample_id, g in (
        frame.groupby(
            "sample_id"
        )
    ):

        best_pred_idx = (
            g[predicted_col]
            .idxmax()
        )

        selected = frame.loc[
            best_pred_idx
        ]

        oracle_idx = (
            g["target_delta_logp"]
            .idxmax()
        )

        oracle = frame.loc[
            oracle_idx
        ]

        selected_rows.append({
            "sample_id":
                sample_id,

            "pair_id":
                int(
                    selected[
                        "pair_id"
                    ]
                ),

            "role":
                selected[
                    "role"
                ],

            "selected_head":
                selected[
                    "head_name"
                ],

            "selected_reward":
                float(
                    selected[
                        "target_delta_logp"
                    ]
                ),

            "oracle_head":
                oracle[
                    "head_name"
                ],

            "oracle_reward":
                float(
                    oracle[
                        "target_delta_logp"
                    ]
                ),

            "oracle_hit":
                float(
                    selected[
                        "head_name"
                    ]
                    == oracle[
                        "head_name"
                    ]
                ),

            "random_expected_reward":
                float(
                    g[
                        "target_delta_logp"
                    ]
                    .mean()
                ),
        })

    return pd.DataFrame(
        selected_rows
    )


# ============================================================
# Inner hyperparameter selection
# ============================================================

def choose_ridge_alpha(
    train_df,
    numeric_cols,
):

    unique_pairs = (
        train_df[
            "pair_id"
        ].unique()
    )

    n_splits = min(
        5,
        len(unique_pairs),
    )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    candidates = []

    # Important:
    # split by pair, therefore all seven heads for both
    # samples in a pair remain together.
    for ridge_alpha in RIDGE_ALPHAS:

        pred = np.full(
            len(train_df),
            np.nan,
        )

        pair_groups = (
            train_df[
                "pair_id"
            ]
            .to_numpy()
        )

        target = (
            train_df[
                "target_delta_logp"
            ]
            .to_numpy()
        )

        X = train_df[
            numeric_cols
            + ["head_name"]
        ]

        valid = True

        for tr, va in splitter.split(
            X,
            target,
            groups=pair_groups,
        ):

            model = make_model(
                numeric_cols,
                ridge_alpha,
            )

            try:

                model.fit(
                    X.iloc[tr],
                    target[tr],
                )

                pred[va] = (
                    model.predict(
                        X.iloc[va]
                    )
                )

            except Exception:

                valid = False
                break

        if (
            not valid
            or np.isnan(pred).any()
        ):
            continue

        tmp = train_df.copy()

        tmp[
            "_inner_pred"
        ] = pred

        selected = (
            selection_metrics(
                tmp,
                "_inner_pred",
            )
        )

        # Hyperparameter criterion:
        # maximize actual reward obtained by selected heads.
        mean_selected_reward = float(
            selected[
                "selected_reward"
            ].mean()
        )

        mean_regret = float(
            (
                selected[
                    "oracle_reward"
                ]
                - selected[
                    "selected_reward"
                ]
            ).mean()
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    target,
                    pred,
                )
            )
        )

        candidates.append(
            (
                ridge_alpha,
                mean_selected_reward,
                mean_regret,
                rmse,
            )
        )

    if not candidates:
        return 1.0

    # First maximize achieved reward,
    # then minimize regret,
    # then RMSE.
    candidates.sort(
        key=lambda z: (
            -z[1],
            z[2],
            z[3],
            z[0],
        )
    )

    return candidates[0][0]


# ============================================================
# Outer leave-one-pair-out
# ============================================================

all_prediction_rows = []
all_fold_rows = []
summary_rows = []

logo = LeaveOneGroupOut()


for family_name, numeric_cols in (
    families.items()
):

    print(
        "\n" + "=" * 120
    )

    print(
        "FAMILY:",
        family_name,
    )

    print(
        "=" * 120
    )

    # Predictions for every sample×head row.
    data_family = (
        data.copy()
        .reset_index(drop=True)
    )

    data_family[
        "predicted_reward"
    ] = np.nan

    data_family[
        "chosen_ridge_alpha"
    ] = np.nan


    pair_groups = (
        data_family[
            "pair_id"
        ].to_numpy()
    )

    # Leave one complete matched pair out.
    # There are 14 rows per pair:
    # 2 samples × 7 heads.
    for fold_idx, (
        train_idx,
        test_idx,
    ) in enumerate(
        logo.split(
            data_family,
            groups=pair_groups,
        ),
        start=1,
    ):

        train_df = (
            data_family
            .iloc[train_idx]
            .copy()
        )

        test_df = (
            data_family
            .iloc[test_idx]
            .copy()
        )

        test_pair = int(
            test_df[
                "pair_id"
            ].iloc[0]
        )

        best_alpha = (
            choose_ridge_alpha(
                train_df,
                numeric_cols,
            )
        )

        model = make_model(
            numeric_cols,
            best_alpha,
        )

        X_train = train_df[
            numeric_cols
            + ["head_name"]
        ]

        y_train = train_df[
            "target_delta_logp"
        ].to_numpy()

        X_test = test_df[
            numeric_cols
            + ["head_name"]
        ]

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(
            X_test
        )

        data_family.loc[
            test_idx,
            "predicted_reward",
        ] = pred

        data_family.loc[
            test_idx,
            "chosen_ridge_alpha",
        ] = best_alpha

        fold_selected = (
            data_family.loc[
                test_idx
            ]
            .copy()
        )

        fold_selected = (
            selection_metrics(
                fold_selected,
                "predicted_reward",
            )
        )

        fold_row = {
            "family":
                family_name,

            "fold":
                fold_idx,

            "test_pair_id":
                test_pair,

            "ridge_alpha":
                best_alpha,

            "mean_selected_reward":
                float(
                    fold_selected[
                        "selected_reward"
                    ].mean()
                ),

            "mean_oracle_reward":
                float(
                    fold_selected[
                        "oracle_reward"
                    ].mean()
                ),

            "mean_random_reward":
                float(
                    fold_selected[
                        "random_expected_reward"
                    ].mean()
                ),

            "oracle_hit_rate":
                float(
                    fold_selected[
                        "oracle_hit"
                    ].mean()
                ),
        }

        all_fold_rows.append(
            fold_row
        )


    if data_family[
        "predicted_reward"
    ].isna().any():

        raise RuntimeError(
            f"{family_name}: missing OOF predictions."
        )


    # --------------------------------------------------------
    # OOF selection
    # --------------------------------------------------------

    selected = (
        selection_metrics(
            data_family,
            "predicted_reward",
        )
    )

    selected[
        "family"
    ] = family_name


    # --------------------------------------------------------
    # Training-only fixed-head baseline for each outer pair
    #
    # We calculate the fixed head separately for every
    # outer fold so the held-out pair never participates
    # in selecting the fixed baseline.
    # --------------------------------------------------------

    fixed_rewards = {}

    for pair_id in sorted(
        data_family[
            "pair_id"
        ].unique()
    ):

        train = data_family[
            data_family[
                "pair_id"
            ]
            != pair_id
        ]

        test = data_family[
            data_family[
                "pair_id"
            ]
            == pair_id
        ]

        fixed_head = (
            train.groupby(
                "head_name"
            )[
                "target_delta_logp"
            ]
            .mean()
            .idxmax()
        )

        for sample_id, g in (
            test.groupby(
                "sample_id"
            )
        ):

            reward = float(
                g.loc[
                    g[
                        "head_name"
                    ]
                    == fixed_head,
                    "target_delta_logp",
                ].iloc[0]
            )

            fixed_rewards[
                sample_id
            ] = (
                fixed_head,
                reward,
            )


    selected[
        "fixed_head"
    ] = (
        selected[
            "sample_id"
        ]
        .map(
            lambda x:
                fixed_rewards[x][0]
        )
    )

    selected[
        "fixed_reward"
    ] = (
        selected[
            "sample_id"
        ]
        .map(
            lambda x:
                fixed_rewards[x][1]
        )
    )


    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    def role_mean(
        column,
        role=None,
    ):

        x = selected

        if role is not None:
            x = x[
                x["role"]
                == role
            ]

        return float(
            x[column]
            .mean()
        )


    summary = {
        "family":
            family_name,

        "n_numeric_features":
            len(numeric_cols),

        "mean_selected_reward_all":
            role_mean(
                "selected_reward"
            ),

        "mean_selected_reward_wrong":
            role_mean(
                "selected_reward",
                "wrong",
            ),

        "mean_selected_reward_correct":
            role_mean(
                "selected_reward",
                "correct",
            ),

        "mean_fixed_reward_wrong":
            role_mean(
                "fixed_reward",
                "wrong",
            ),

        "mean_random_reward_wrong":
            role_mean(
                "random_expected_reward",
                "wrong",
            ),

        "mean_oracle_reward_wrong":
            role_mean(
                "oracle_reward",
                "wrong",
            ),

        "oracle_hit_rate_all":
            role_mean(
                "oracle_hit"
            ),

        "oracle_hit_rate_wrong":
            role_mean(
                "oracle_hit",
                "wrong",
            ),

        "positive_selected_fraction_wrong":
            float(
                (
                    selected.loc[
                        selected["role"]
                        == "wrong",
                        "selected_reward",
                    ]
                    > 0
                )
                .mean()
            ),

        "mean_regret_wrong":
            float(
                (
                    selected.loc[
                        selected["role"]
                        == "wrong",
                        "oracle_reward",
                    ]
                    -
                    selected.loc[
                        selected["role"]
                        == "wrong",
                        "selected_reward",
                    ]
                )
                .mean()
            ),
    }

    summary_rows.append(
        summary
    )

    all_prediction_rows.append(
        selected
    )


# ============================================================
# Combine
# ============================================================

predictions = pd.concat(
    all_prediction_rows,
    ignore_index=True,
)

folds = pd.DataFrame(
    all_fold_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# Paired bootstrap on wrong samples
#
# Learned adaptive policy vs training-only fixed head.
# ============================================================

def paired_bootstrap_wrong(
    frame,
    family_name,
    B=10000,
):

    x = frame[
        (frame["family"] == family_name)
        & (frame["role"] == "wrong")
    ].copy()

    d = (
        x["selected_reward"]
        .to_numpy()
        -
        x["fixed_reward"]
        .to_numpy()
    )

    observed = float(
        d.mean()
    )

    rng = np.random.default_rng(
        SEED
    )

    boots = []

    for _ in range(B):

        sampled = rng.choice(
            d,
            size=len(d),
            replace=True,
        )

        boots.append(
            float(
                sampled.mean()
            )
        )

    lo, hi = np.quantile(
        boots,
        [0.025, 0.975],
    )

    return (
        observed,
        float(lo),
        float(hi),
    )


bootstrap_rows = []

for family_name in families:

    diff, lo, hi = (
        paired_bootstrap_wrong(
            predictions,
            family_name,
        )
    )

    bootstrap_rows.append({
        "family":
            family_name,

        "learned_minus_fixed_wrong":
            diff,

        "ci_low":
            lo,

        "ci_high":
            hi,
    })


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

summary_df = summary_df.merge(
    bootstrap_df,
    on="family",
    how="left",
)


# ============================================================
# Save
# ============================================================

predictions.to_csv(
    PRED_OUT,
    index=False,
)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)

folds.to_csv(
    FOLD_OUT,
    index=False,
)

CONFIG_OUT.write_text(
    json.dumps(
        {
            "candidate_heads":
                CANDIDATE_HEADS,

            "ridge_alphas":
                RIDGE_ALPHAS,

            "families":
                families,

            "outer_cv":
                "LeaveOneGroupOut by pair_id",

            "inner_cv":
                "GroupKFold by pair_id",

            "target":
                "delta_logp used only as supervised reward label",

            "deployment_features":
                "GT-free only",
        },
        indent=2,
    )
)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 140)
print("FINAL ADAPTIVE HEAD-SELECTION RESULTS")
print("=" * 140)

display_cols = [
    "family",
    "n_numeric_features",
    "mean_selected_reward_wrong",
    "mean_fixed_reward_wrong",
    "mean_random_reward_wrong",
    "mean_oracle_reward_wrong",
    "oracle_hit_rate_wrong",
    "positive_selected_fraction_wrong",
    "mean_regret_wrong",
    "learned_minus_fixed_wrong",
    "ci_low",
    "ci_high",
]

print(
    summary_df[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)

print(
    "\nInterpretation:"
)

print(
    "  mean_selected_reward_wrong:"
    " actual held-out ΔGT-logp obtained by"
    " the head chosen WITHOUT GT."
)

print(
    "  learned_minus_fixed_wrong > 0:"
    " adaptive routing beats the best"
    " training-only fixed-head baseline."
)

print(
    "  oracle reward is an upper bound,"
    " not a deployable result."
)

print("\nSaved:")
print(PRED_OUT)
print(SUMMARY_OUT)
print(FOLD_OUT)
print(CONFIG_OUT)

print(
    "\nADAPTIVE HEAD SELECTOR BENCHMARK COMPLETE"
)
