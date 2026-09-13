from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

INPUT = Path(
    "outputs/proc_count_causal_v1/router/"
    "full_probability_response/"
    "full_response_router_features.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "full_probability_response/"
    "router_benchmark"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PRED_OUT = OUT_DIR / "router_oof_predictions.csv"
FOLD_OUT = OUT_DIR / "router_fold_metrics.csv"
PAIR_OUT = OUT_DIR / "router_pairwise_rankings.csv"
SUMMARY_OUT = OUT_DIR / "router_benchmark_summary.csv"
FEATURE_OUT = OUT_DIR / "feature_sets.json"

SEED = 20260912

C_GRID = [
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
]


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 120)
print("AROMA FULL-RESPONSE ROUTER — NESTED PAIR-GROUPED CV")
print("=" * 120)

print("Rows:", len(df))
print("Pairs:", df["pair_id"].nunique())
print("Correct:", int((df["role"] == "correct").sum()))
print("Wrong:", int((df["role"] == "wrong").sum()))

assert len(df) == 54
assert df["sample_id"].nunique() == 54
assert df["pair_id"].nunique() == 27
assert (df.groupby("pair_id").size() == 2).all()

pair_roles = (
    df.groupby("pair_id")["role"]
    .apply(set)
)

assert all(
    x == {"correct", "wrong"}
    for x in pair_roles
)

y = (
    df["role"] == "wrong"
).astype(int).to_numpy()

groups = df["pair_id"].to_numpy()


# ============================================================
# Feature families
# ============================================================

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


# ------------------------------------------------------------
# Low-dimensional response summary
#
# Only aggregate response geometry.
# No per-number delta vectors.
# No metadata / GT.
# ------------------------------------------------------------

response_summary_features = [
    c for c in df.columns
    if (
        c.startswith("global_")
        or "_agg_" in c
    )
]


# ------------------------------------------------------------
# Full response
#
# Includes:
#   per-head × alpha × delta probabilities
#   per-head response metrics
#   aggregate response metrics
# ------------------------------------------------------------

metadata = {
    "pair_id",
    "role",
    "sample_id",
    "condition",
    "replicate",
}

full_response_features = [
    c for c in df.columns
    if (
        c not in metadata
        and c not in numeral_features
    )
]


families = {
    "numeral_only":
        numeral_features,

    "response_summary_only":
        response_summary_features,

    "full_response_only":
        full_response_features,

    "numeral_plus_summary":
        numeral_features
        + response_summary_features,
}


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
]

for family, cols in families.items():

    bad = [
        c for c in cols
        if any(
            token in c
            for token in forbidden_tokens
        )
    ]

    if bad:
        raise RuntimeError(
            f"{family}: leakage features {bad}"
        )


print("\nFeature families:")

for name, cols in families.items():
    print(
        f"{name:26s}: {len(cols)}"
    )


# ============================================================
# Model
# ============================================================

def make_model(C):

    return Pipeline([
        (
            "variance",
            VarianceThreshold(
                threshold=0.0
            ),
        ),
        (
            "scale",
            StandardScaler(),
        ),
        (
            "clf",
            LogisticRegression(
                C=C,
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=10000,
                random_state=SEED,
            ),
        ),
    ])


# ============================================================
# Pair ranking helper
# ============================================================

def pair_ranking_score(
    probs,
    labels,
    group_ids,
):

    tmp = pd.DataFrame({
        "group": group_ids,
        "y": labels,
        "p": probs,
    })

    scores = []

    for _, g in tmp.groupby("group"):

        if (
            len(g) != 2
            or set(g["y"]) != {0, 1}
        ):
            continue

        p_wrong = float(
            g.loc[
                g["y"] == 1,
                "p",
            ].iloc[0]
        )

        p_correct = float(
            g.loc[
                g["y"] == 0,
                "p",
            ].iloc[0]
        )

        if p_wrong > p_correct:
            scores.append(1.0)

        elif p_wrong == p_correct:
            scores.append(0.5)

        else:
            scores.append(0.0)

    return float(
        np.mean(scores)
    )


# ============================================================
# Inner hyperparameter selection
# ============================================================

def choose_c(
    X,
    y_train,
    train_groups,
):

    unique_groups = np.unique(
        train_groups
    )

    n_splits = min(
        5,
        len(unique_groups),
    )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    candidates = []

    for C in C_GRID:

        probs = np.full(
            len(y_train),
            np.nan,
        )

        valid = True

        for tr, va in splitter.split(
            X,
            y_train,
            groups=train_groups,
        ):

            model = make_model(C)

            try:

                model.fit(
                    X.iloc[tr],
                    y_train[tr],
                )

                probs[va] = (
                    model.predict_proba(
                        X.iloc[va]
                    )[:, 1]
                )

            except Exception:
                valid = False
                break

        if (
            not valid
            or np.isnan(probs).any()
        ):
            continue

        rank_acc = pair_ranking_score(
            probs,
            y_train,
            train_groups,
        )

        ll = log_loss(
            y_train,
            probs,
            labels=[0, 1],
        )

        candidates.append(
            (
                C,
                rank_acc,
                ll,
            )
        )

    if not candidates:
        return 0.01

    candidates.sort(
        key=lambda z: (
            -z[1],
            z[2],
            z[0],
        )
    )

    return candidates[0][0]


# ============================================================
# Outer Leave-One-Pair-Out
# ============================================================

def run_family(
    family_name,
    feature_cols,
):

    X = (
        df[feature_cols]
        .astype(float)
    )

    logo = LeaveOneGroupOut()

    oof_prob = np.full(
        len(df),
        np.nan,
    )

    chosen_cs = np.full(
        len(df),
        np.nan,
    )

    fold_rows = []

    for fold_idx, (
        train_idx,
        test_idx,
    ) in enumerate(
        logo.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):

        X_train = X.iloc[
            train_idx
        ]

        y_train = y[
            train_idx
        ]

        g_train = groups[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_test = y[
            test_idx
        ]

        test_pair = int(
            groups[
                test_idx
            ][0]
        )

        best_c = choose_c(
            X_train,
            y_train,
            g_train,
        )

        model = make_model(
            best_c
        )

        model.fit(
            X_train,
            y_train,
        )

        p = model.predict_proba(
            X_test
        )[:, 1]

        oof_prob[
            test_idx
        ] = p

        chosen_cs[
            test_idx
        ] = best_c

        wrong_pos = np.where(
            y_test == 1
        )[0][0]

        correct_pos = np.where(
            y_test == 0
        )[0][0]

        p_wrong = float(
            p[wrong_pos]
        )

        p_correct = float(
            p[correct_pos]
        )

        if p_wrong > p_correct:
            rank_correct = 1.0

        elif p_wrong == p_correct:
            rank_correct = 0.5

        else:
            rank_correct = 0.0

        fold_rows.append({
            "family":
                family_name,

            "fold":
                fold_idx,

            "test_pair_id":
                test_pair,

            "best_C":
                best_c,

            "p_correct_sample":
                p_correct,

            "p_wrong_sample":
                p_wrong,

            "pair_probability_gap":
                (
                    p_wrong
                    - p_correct
                ),

            "pair_ranking_correct":
                rank_correct,
        })

    if np.isnan(
        oof_prob
    ).any():
        raise RuntimeError(
            f"{family_name}: missing OOF prediction"
        )

    pred = (
        oof_prob >= 0.5
    ).astype(int)

    fold_df = pd.DataFrame(
        fold_rows
    )

    summary = {
        "family":
            family_name,

        "n_features":
            len(feature_cols),

        "accuracy":
            accuracy_score(
                y,
                pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred,
            ),

        "roc_auc":
            roc_auc_score(
                y,
                oof_prob,
            ),

        "log_loss":
            log_loss(
                y,
                oof_prob,
                labels=[0, 1],
            ),

        "pair_ranking_accuracy":
            float(
                fold_df[
                    "pair_ranking_correct"
                ].mean()
            ),

        "mean_pair_probability_gap":
            float(
                fold_df[
                    "pair_probability_gap"
                ].mean()
            ),

        "tn":
            int(
                (
                    (y == 0)
                    & (pred == 0)
                ).sum()
            ),

        "fp":
            int(
                (
                    (y == 0)
                    & (pred == 1)
                ).sum()
            ),

        "fn":
            int(
                (
                    (y == 1)
                    & (pred == 0)
                ).sum()
            ),

        "tp":
            int(
                (
                    (y == 1)
                    & (pred == 1)
                ).sum()
            ),
    }

    prediction_df = df[
        [
            "pair_id",
            "role",
            "sample_id",
            "condition",
            "replicate",
        ]
    ].copy()

    prediction_df[
        "family"
    ] = family_name

    prediction_df[
        "target_wrong"
    ] = y

    prediction_df[
        "failure_probability"
    ] = oof_prob

    prediction_df[
        "predicted_wrong"
    ] = pred

    prediction_df[
        "chosen_C"
    ] = chosen_cs

    return (
        prediction_df,
        fold_df,
        summary,
    )


# ============================================================
# Run
# ============================================================

all_predictions = []
all_folds = []
summaries = []

for family_name, cols in families.items():

    print(
        "\nRunning:",
        family_name,
        f"({len(cols)} features)"
    )

    pred_df, fold_df, summary = (
        run_family(
            family_name,
            cols,
        )
    )

    all_predictions.append(
        pred_df
    )

    all_folds.append(
        fold_df
    )

    summaries.append(
        summary
    )


predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)

folds = pd.concat(
    all_folds,
    ignore_index=True,
)

summary_df = pd.DataFrame(
    summaries
)


# ============================================================
# Paired bootstrap:
# numeral+summary vs numeral-only
# ============================================================

def paired_rank_difference(
    family_a,
    family_b,
    B=10000,
):

    a = folds[
        folds["family"]
        == family_a
    ].sort_values(
        "test_pair_id"
    )

    b = folds[
        folds["family"]
        == family_b
    ].sort_values(
        "test_pair_id"
    )

    assert (
        a["test_pair_id"]
        .to_numpy()
        ==
        b["test_pair_id"]
        .to_numpy()
    ).all()

    d = (
        a["pair_ranking_correct"]
        .to_numpy()
        -
        b["pair_ranking_correct"]
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
            sampled.mean()
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


diff, lo, hi = (
    paired_rank_difference(
        "numeral_plus_summary",
        "numeral_only",
    )
)


# ============================================================
# Save
# ============================================================

predictions.to_csv(
    PRED_OUT,
    index=False,
)

folds.to_csv(
    FOLD_OUT,
    index=False,
)

folds.to_csv(
    PAIR_OUT,
    index=False,
)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)

FEATURE_OUT.write_text(
    json.dumps(
        families,
        indent=2,
    )
)


# ============================================================
# Report
# ============================================================

print("\n" + "=" * 120)
print("FINAL FULL-RESPONSE ROUTER RESULTS")
print("=" * 120)

cols = [
    "family",
    "n_features",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "log_loss",
    "pair_ranking_accuracy",
    "mean_pair_probability_gap",
    "tn",
    "fp",
    "fn",
    "tp",
]

print(
    summary_df[
        cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)

print("\n" + "=" * 120)
print("NUMERAL+SUMMARY vs NUMERAL-ONLY")
print("=" * 120)

print(
    "Pair-ranking improvement:",
    f"{diff:+.4f}",
)

print(
    "95% paired bootstrap CI:",
    f"[{lo:+.4f}, {hi:+.4f}]",
)

print("\nSaved:")
print(PRED_OUT)
print(FOLD_OUT)
print(SUMMARY_OUT)
print(FEATURE_OUT)

print(
    "\nFULL-RESPONSE ROUTER "
    "BENCHMARK COMPLETE"
)
