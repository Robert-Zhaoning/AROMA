from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
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

# ============================================================
# Paths
# ============================================================

INPUT = Path(
    "outputs/proc_count_causal_v1/router/"
    "gtfree_response/gtfree_response_features.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "gtfree_response_benchmark"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_OUT = OUT_DIR / "router_oof_predictions.csv"
FOLD_OUT = OUT_DIR / "router_fold_metrics.csv"
PAIR_OUT = OUT_DIR / "router_pairwise_rankings.csv"
SUMMARY_OUT = OUT_DIR / "router_benchmark_summary.csv"
FEATURE_OUT = OUT_DIR / "feature_sets.json"

# ============================================================
# Settings
# ============================================================

SEED = 20260910

# Conservative grid because N is small.
C_GRID = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]

# ============================================================
# Load
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)

print("=" * 120)
print("AROMA GT-FREE RESPONSE ROUTER — NESTED PAIR-GROUPED BENCHMARK")
print("=" * 120)

print("Rows:", len(df))
print("Pairs:", df["pair_id"].nunique())
print("Correct:", int((df["role"] == "correct").sum()))
print("Wrong:", int((df["role"] == "wrong").sum()))

# ------------------------------------------------------------
# Integrity
# ------------------------------------------------------------

assert len(df) == 54
assert df["sample_id"].nunique() == 54
assert df["pair_id"].nunique() == 27
assert (df.groupby("pair_id").size() == 2).all()

pair_roles = (
    df.groupby("pair_id")["role"]
    .apply(lambda x: set(x))
)

assert all(x == {"correct", "wrong"} for x in pair_roles)

# wrong = positive class
y = (df["role"] == "wrong").astype(int).to_numpy()
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

response_features = [
    c for c in df.columns
    if (
        c.startswith("resp_")
        or "_resp_" in c
    )
]

# Safety / leakage checks
forbidden_tokens = [
    "ground_truth",
    "baseline_correct",
    "oracle_",
    "object_mass",
    "object_enrichment",
    "_gt_",
    "role",
]

model_features = set(numeral_features + response_features)

bad = [
    c for c in model_features
    if any(token in c for token in forbidden_tokens)
]

if bad:
    raise RuntimeError(
        f"Potential leakage feature detected: {sorted(bad)}"
    )

missing = [
    c for c in model_features
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing feature columns: {missing}"
    )

families = {
    "numeral_only": numeral_features,
    "response_only": response_features,
    "numeral_plus_response": (
        numeral_features + response_features
    ),
}

print("\nFeature families:")
for name, cols in families.items():
    print(f"  {name:24s}: {len(cols)}")

# ============================================================
# Model
# ============================================================

def make_model(C):
    return Pipeline([
        (
            "variance",
            VarianceThreshold(threshold=0.0),
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
# Inner C selection
# ============================================================

def choose_c(X, y_train, train_groups):
    """
    Select C using only the outer-training samples.

    Metric:
        mean pair-ranking accuracy first,
        then log-loss as tie-breaker.

    Inner split is group-safe.
    """

    unique_groups = np.unique(train_groups)

    # At most 5 folds, but never more folds than groups.
    n_splits = min(5, len(unique_groups))

    splitter = GroupKFold(
        n_splits=n_splits
    )

    candidates = []

    for C in C_GRID:
        probs = np.full(
            len(y_train),
            np.nan,
            dtype=float,
        )

        valid = True

        for inner_train, inner_val in splitter.split(
            X,
            y_train,
            groups=train_groups,
        ):
            model = make_model(C)

            try:
                model.fit(
                    X.iloc[inner_train],
                    y_train[inner_train],
                )

                probs[inner_val] = model.predict_proba(
                    X.iloc[inner_val]
                )[:, 1]

            except Exception:
                valid = False
                break

        if (
            (not valid)
            or np.isnan(probs).any()
        ):
            continue

        # ----------------------------------------------
        # Pair-ranking accuracy
        # ----------------------------------------------

        tmp = pd.DataFrame({
            "group": train_groups,
            "y": y_train,
            "p": probs,
        })

        pair_scores = []

        for _, g in tmp.groupby("group"):
            if (
                len(g) != 2
                or set(g["y"]) != {0, 1}
            ):
                continue

            p_wrong = float(
                g.loc[g["y"] == 1, "p"].iloc[0]
            )

            p_correct = float(
                g.loc[g["y"] == 0, "p"].iloc[0]
            )

            if p_wrong > p_correct:
                pair_scores.append(1.0)
            elif p_wrong == p_correct:
                pair_scores.append(0.5)
            else:
                pair_scores.append(0.0)

        pair_acc = (
            float(np.mean(pair_scores))
            if pair_scores
            else 0.0
        )

        ll = log_loss(
            y_train,
            probs,
            labels=[0, 1],
        )

        candidates.append(
            (C, pair_acc, ll)
        )

    if not candidates:
        return 0.1

    # maximize pair ranking,
    # then minimize log loss,
    # then prefer stronger regularization (smaller C)
    candidates.sort(
        key=lambda z: (
            -z[1],
            z[2],
            z[0],
        )
    )

    return candidates[0][0]


# ============================================================
# Outer nested LOGO
# ============================================================

def run_family(
    family_name,
    feature_cols,
):
    X = df[feature_cols].astype(float)

    logo = LeaveOneGroupOut()

    oof_prob = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    chosen_cs = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    total_folds = df["pair_id"].nunique()

    print("\n" + "=" * 120)
    print(f"FAMILY: {family_name}")
    print("=" * 120)

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

        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        g_train = groups[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        test_pair = int(
            groups[test_idx][0]
        )

        best_c = choose_c(
            X_train,
            y_train,
            g_train,
        )

        model = make_model(best_c)

        model.fit(
            X_train,
            y_train,
        )

        p = model.predict_proba(
            X_test
        )[:, 1]

        oof_prob[test_idx] = p
        chosen_cs[test_idx] = best_c

        # pair ranking in this held-out pair
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
            pair_correct = 1.0
        elif p_wrong == p_correct:
            pair_correct = 0.5
        else:
            pair_correct = 0.0

        fold_rows.append({
            "family": family_name,
            "fold": fold_idx,
            "test_pair_id": test_pair,
            "best_C": best_c,
            "p_correct_sample": p_correct,
            "p_wrong_sample": p_wrong,
            "pair_probability_gap": (
                p_wrong - p_correct
            ),
            "pair_ranking_correct": (
                pair_correct
            ),
        })

        print(
            f"Fold {fold_idx:02d}/{total_folds} | "
            f"pair={test_pair:02d} | "
            f"C={best_c:g} | "
            f"P(correct→wrong)={p_correct:.3f} | "
            f"P(wrong→wrong)={p_wrong:.3f} | "
            f"gap={p_wrong-p_correct:+.3f}"
        )

    if np.isnan(oof_prob).any():
        raise RuntimeError(
            f"{family_name}: missing OOF predictions"
        )

    pred = (
        oof_prob >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        pred,
    )

    auc = roc_auc_score(
        y,
        oof_prob,
    )

    ll = log_loss(
        y,
        oof_prob,
        labels=[0, 1],
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    pair_ranking_acc = float(
        fold_df[
            "pair_ranking_correct"
        ].mean()
    )

    mean_pair_gap = float(
        fold_df[
            "pair_probability_gap"
        ].mean()
    )

    # confusion matrix manually
    tn = int(
        ((y == 0) & (pred == 0)).sum()
    )
    fp = int(
        ((y == 0) & (pred == 1)).sum()
    )
    fn = int(
        ((y == 1) & (pred == 0)).sum()
    )
    tp = int(
        ((y == 1) & (pred == 1)).sum()
    )

    prediction_df = df[
        [
            "pair_id",
            "role",
            "sample_id",
            "condition",
            "replicate",
        ]
    ].copy()

    prediction_df["family"] = (
        family_name
    )

    prediction_df["target_wrong"] = y
    prediction_df["failure_probability"] = (
        oof_prob
    )
    prediction_df["predicted_wrong"] = pred
    prediction_df["chosen_C"] = (
        chosen_cs
    )

    summary = {
        "family": family_name,
        "n_features": len(feature_cols),
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "roc_auc": auc,
        "log_loss": ll,
        "pair_ranking_accuracy": (
            pair_ranking_acc
        ),
        "mean_pair_probability_gap": (
            mean_pair_gap
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return (
        prediction_df,
        fold_df,
        summary,
    )


# ============================================================
# Run all families
# ============================================================

all_predictions = []
all_folds = []
summaries = []

for family_name, feature_cols in families.items():
    pred_df, fold_df, summary = run_family(
        family_name,
        feature_cols,
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
# Bootstrap pair-level difference:
# combined vs numeral-only
# ============================================================

def paired_bootstrap_metric_difference(
    predictions,
    family_a,
    family_b,
    metric="pair_ranking",
    B=10000,
    seed=SEED,
):
    rng = np.random.default_rng(seed)

    pa = predictions[
        predictions["family"] == family_a
    ].copy()

    pb = predictions[
        predictions["family"] == family_b
    ].copy()

    rows = []

    for pair_id in sorted(
        pa["pair_id"].unique()
    ):
        a = pa[
            pa["pair_id"] == pair_id
        ]

        b = pb[
            pb["pair_id"] == pair_id
        ]

        aw = float(
            a.loc[
                a["role"] == "wrong",
                "failure_probability",
            ].iloc[0]
        )

        ac = float(
            a.loc[
                a["role"] == "correct",
                "failure_probability",
            ].iloc[0]
        )

        bw = float(
            b.loc[
                b["role"] == "wrong",
                "failure_probability",
            ].iloc[0]
        )

        bc = float(
            b.loc[
                b["role"] == "correct",
                "failure_probability",
            ].iloc[0]
        )

        score_a = (
            1.0 if aw > ac
            else 0.5 if aw == ac
            else 0.0
        )

        score_b = (
            1.0 if bw > bc
            else 0.5 if bw == bc
            else 0.0
        )

        rows.append(
            score_a - score_b
        )

    rows = np.asarray(
        rows,
        dtype=float,
    )

    observed = float(
        rows.mean()
    )

    boots = []

    for _ in range(B):
        sample = rng.choice(
            rows,
            size=len(rows),
            replace=True,
        )

        boots.append(
            float(sample.mean())
        )

    low, high = np.quantile(
        boots,
        [0.025, 0.975],
    )

    return (
        observed,
        float(low),
        float(high),
    )


diff, ci_low, ci_high = (
    paired_bootstrap_metric_difference(
        predictions,
        "numeral_plus_response",
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

pair_rankings = folds[
    [
        "family",
        "test_pair_id",
        "best_C",
        "p_correct_sample",
        "p_wrong_sample",
        "pair_probability_gap",
        "pair_ranking_correct",
    ]
].copy()

pair_rankings.to_csv(
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
# Final report
# ============================================================

print("\n" + "=" * 120)
print("FINAL NESTED PAIR-GROUPED RESULTS")
print("=" * 120)

display_cols = [
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
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 120)
print("COMBINED vs NUMERAL-ONLY — PAIR-RANKING BOOTSTRAP")
print("=" * 120)

print(
    "Mean paired improvement:",
    f"{diff:+.4f}",
)

print(
    "95% bootstrap CI:",
    f"[{ci_low:+.4f}, {ci_high:+.4f}]",
)

print("\nInterpretation:")
print(
    "  > 0 means GT-free intervention response improves "
    "within-pair failure ranking over passive numeral features."
)

print("\nSaved:")
print(PRED_OUT)
print(FOLD_OUT)
print(PAIR_OUT)
print(SUMMARY_OUT)
print(FEATURE_OUT)

print("\nROUTER GT-FREE RESPONSE BENCHMARK COMPLETE")
