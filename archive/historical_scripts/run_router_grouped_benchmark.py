#!/usr/bin/env python3

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


INPUT = Path(
    "outputs/proc_count_causal_v1/router/"
    "preintervention_features_heldout_clean.csv"
)

OUTDIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "grouped_benchmark"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260910

# ---------------------------------------------------------------------
# Feature-policy helpers
# ---------------------------------------------------------------------

META_COLUMNS = {
    "pair_id",
    "sample_id",
    "role",
    "ground_truth",
    "condition",
    "replicate",
    "seed",
    "baseline_prediction",
    "baseline_correct",
    "parse_success",
}

# Anything derived from GT objects / oracle annotations cannot enter
# deployable router.
FORBIDDEN_SUBSTRINGS = [
    "oracle",
    "ground_truth",
    "target_bbox",
    "target_object",
    "bbox",
    "object_mass",
    "object_union",
    "object_enrichment",
]

# Explicit label leakage
FORBIDDEN_EXACT = {
    "role",
    "baseline_correct",
}


def is_forbidden(c):
    cl = c.lower()

    if c in FORBIDDEN_EXACT:
        return True

    if c in META_COLUMNS:
        return True

    return any(x in cl for x in FORBIDDEN_SUBSTRINGS)


def numeric_columns(df):
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]


def numeral_feature(c):
    """
    Deployable numeral-side information.

    Full numeral probability distribution, entropy, margin, top-k
    predictions/probabilities are observable before intervention.
    """
    cl = c.lower()

    return (
        cl.startswith("numeral_")
        and not is_forbidden(c)
    )


def attention_feature(c):
    """
    Attention summaries that do NOT depend on GT object boxes.

    Deliberately excludes object_mass/object_enrichment/oracle signals.
    """
    cl = c.lower()

    if is_forbidden(c):
        return False

    if "attn_" in cl:
        return True

    return False


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def compute_metrics(y, pred, prob):
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": safe_auc(y, prob),
        "log_loss": log_loss(
            y,
            np.column_stack([1.0 - prob, prob]),
            labels=[0, 1],
        ),
    }


# ---------------------------------------------------------------------
# Inner grouped C selection
# ---------------------------------------------------------------------

def choose_c(X, y, groups, c_grid):
    unique_groups = np.unique(groups)

    n_splits = min(4, len(unique_groups))

    if n_splits < 2:
        return 1.0

    splitter = GroupKFold(n_splits=n_splits)

    rows = []

    for c in c_grid:

        scores = []

        for train_idx, valid_idx in splitter.split(
            X, y, groups
        ):
            model = Pipeline([
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=c,
                        penalty="l2",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ])

            model.fit(
                X[train_idx],
                y[train_idx],
            )

            p = model.predict_proba(
                X[valid_idx]
            )[:, 1]

            score = safe_auc(
                y[valid_idx],
                p,
            )

            if np.isfinite(score):
                scores.append(score)

        mean_score = (
            np.mean(scores)
            if scores
            else -np.inf
        )

        rows.append(
            (c, mean_score)
        )

    rows.sort(
        key=lambda x: (
            x[1],
            -abs(np.log10(x[0])),
        ),
        reverse=True,
    )

    return float(rows[0][0])


# ---------------------------------------------------------------------
# Nested pair-grouped benchmark
# ---------------------------------------------------------------------

def run_family(df, family_name, features):
    print("\n" + "=" * 100)
    print(f"FEATURE FAMILY: {family_name}")
    print("=" * 100)

    print("Features:", len(features))

    if len(features) == 0:
        raise RuntimeError(
            f"No features found for {family_name}"
        )

    X = (
        df[features]
        .astype(float)
        .to_numpy()
    )

    # 1 = wrong / failure state
    y = (
        df["role"]
        .map({
            "correct": 0,
            "wrong": 1,
        })
        .astype(int)
        .to_numpy()
    )

    groups = (
        df["pair_id"]
        .astype(int)
        .to_numpy()
    )

    # 9 outer folds:
    # 27 pairs -> exactly ~3 matched pairs per test fold.
    outer = GroupKFold(
        n_splits=9
    )

    c_grid = [
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

    records = []

    fold_summaries = []

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):
        X_train = X[train_idx]
        y_train = y[train_idx]
        g_train = groups[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        best_c = choose_c(
            X_train,
            y_train,
            g_train,
            c_grid,
        )

        model = Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "clf",
                LogisticRegression(
                    C=best_c,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ])

        model.fit(
            X_train,
            y_train,
        )

        prob = model.predict_proba(
            X_test
        )[:, 1]

        pred = (
            prob >= 0.5
        ).astype(int)

        metrics = compute_metrics(
            y_test,
            pred,
            prob,
        )

        test_groups = sorted(
            np.unique(groups[test_idx]).tolist()
        )

        print(
            f"Fold {fold:02d} | "
            f"pairs={test_groups} | "
            f"C={best_c:g} | "
            f"acc={metrics['accuracy']:.3f} | "
            f"auc={metrics['roc_auc']:.3f}"
        )

        fold_summaries.append({
            "family": family_name,
            "fold": fold,
            "best_c": best_c,
            "n_test": len(test_idx),
            "n_test_pairs": len(test_groups),
            **metrics,
        })

        for local_i, idx in enumerate(test_idx):
            records.append({
                "family": family_name,
                "fold": fold,
                "pair_id": int(
                    df.iloc[idx]["pair_id"]
                ),
                "sample_id": str(
                    df.iloc[idx]["sample_id"]
                ),
                "role": str(
                    df.iloc[idx]["role"]
                ),
                "ground_truth": int(
                    df.iloc[idx]["ground_truth"]
                ),
                "condition": str(
                    df.iloc[idx]["condition"]
                ),
                "y_true": int(
                    y_test[local_i]
                ),
                "failure_probability": float(
                    prob[local_i]
                ),
                "y_pred": int(
                    pred[local_i]
                ),
                "correct_classification": bool(
                    pred[local_i]
                    == y_test[local_i]
                ),
                "best_c": best_c,
            })

    pred_df = pd.DataFrame(
        records
    )

    y_all = pred_df[
        "y_true"
    ].to_numpy()

    p_all = pred_df[
        "failure_probability"
    ].to_numpy()

    pred_all = pred_df[
        "y_pred"
    ].to_numpy()

    overall = compute_metrics(
        y_all,
        pred_all,
        p_all,
    )

    # -------------------------------------------------------------
    # Pairwise diagnostic:
    # For each exact pair, does router assign a larger failure
    # probability to the wrong member than the correct member?
    # -------------------------------------------------------------

    pair_rows = []

    for pair_id, g in pred_df.groupby(
        "pair_id"
    ):
        wrong = g[
            g["role"] == "wrong"
        ]

        correct = g[
            g["role"] == "correct"
        ]

        if len(wrong) != 1 or len(correct) != 1:
            continue

        pw = float(
            wrong.iloc[0][
                "failure_probability"
            ]
        )

        pc = float(
            correct.iloc[0][
                "failure_probability"
            ]
        )

        pair_rows.append({
            "family": family_name,
            "pair_id": int(pair_id),
            "condition": str(
                wrong.iloc[0]["condition"]
            ),
            "ground_truth": int(
                wrong.iloc[0]["ground_truth"]
            ),
            "wrong_failure_probability": pw,
            "correct_failure_probability": pc,
            "pair_probability_gap": pw - pc,
            "pair_rank_correct": (
                pw > pc
            ),
            "pair_tie": (
                pw == pc
            ),
        })

    pair_df = pd.DataFrame(
        pair_rows
    )

    pair_accuracy = (
        pair_df["pair_rank_correct"]
        .mean()
    )

    pair_gap = (
        pair_df[
            "pair_probability_gap"
        ]
        .mean()
    )

    tn, fp, fn, tp = confusion_matrix(
        y_all,
        pred_all,
        labels=[0, 1],
    ).ravel()

    summary = {
        "family": family_name,
        "n_features": len(features),
        "n_samples": len(pred_df),
        "n_pairs": pred_df[
            "pair_id"
        ].nunique(),
        **overall,
        "pair_ranking_accuracy": float(
            pair_accuracy
        ),
        "mean_pair_probability_gap": float(
            pair_gap
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return (
        pred_df,
        pd.DataFrame(fold_summaries),
        pair_df,
        summary,
    )


def main():

    print("=" * 110)
    print("AROMA LEAKAGE-SAFE PAIR-GROUPED ROUTER BENCHMARK")
    print("=" * 110)

    df = pd.read_csv(
        INPUT
    )

    # Recover pair-level metadata if the leakage-clean feature file
    # intentionally removed ground_truth / condition.
    pair_path = Path(
        "outputs/proc_count_causal_v1/"
        "exact_matched_pairs_eager.csv"
    )

    if pair_path.exists():
        pair_meta = pd.read_csv(pair_path)[
            ["pair_id", "ground_truth", "condition"]
        ].drop_duplicates("pair_id")

        for col in ["ground_truth", "condition"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df.merge(
            pair_meta,
            on="pair_id",
            how="left",
            validate="many_to_one",
        )

    assert len(df) == 54
    assert df["pair_id"].nunique() == 27

    if df["ground_truth"].isna().any():
        raise RuntimeError(
            "Missing ground_truth after pair metadata merge."
        )

    if df["condition"].isna().any():
        raise RuntimeError(
            "Missing condition after pair metadata merge."
        )

    counts = df["role"].value_counts()

    assert counts.get("correct", 0) == 27
    assert counts.get("wrong", 0) == 27

    print("Samples:", len(df))
    print("Pairs:", df["pair_id"].nunique())

    all_numeric = numeric_columns(
        df
    )

    numeral = [
        c for c in all_numeric
        if numeral_feature(c)
    ]

    attention = [
        c for c in all_numeric
        if attention_feature(c)
    ]

    combined = sorted(
        set(numeral + attention)
    )

    print("\nFeature counts:")
    print("Numeral-only :", len(numeral))
    print("Attention-only:", len(attention))
    print("Combined     :", len(combined))

    print("\nNumeral features:")
    for c in numeral:
        print(" ", c)

    print("\nAttention features:")
    for c in attention:
        print(" ", c)

    # Strong leakage assertion
    for c in combined:
        assert not is_forbidden(c), (
            f"Forbidden deployable feature leaked: {c}"
        )

    families = {
        "numeral_only": numeral,
        "attention_only": attention,
        "numeral_plus_attention": combined,
    }

    all_predictions = []
    all_folds = []
    all_pairs = []
    summaries = []

    for family_name, features in families.items():

        pred_df, fold_df, pair_df, summary = run_family(
            df,
            family_name,
            features,
        )

        all_predictions.append(
            pred_df
        )

        all_folds.append(
            fold_df
        )

        all_pairs.append(
            pair_df
        )

        summaries.append(
            summary
        )

    pred_all = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    fold_all = pd.concat(
        all_folds,
        ignore_index=True,
    )

    pair_all = pd.concat(
        all_pairs,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summaries
    ).sort_values(
        "roc_auc",
        ascending=False,
    )

    pred_path = (
        OUTDIR /
        "router_oof_predictions.csv"
    )

    fold_path = (
        OUTDIR /
        "router_fold_metrics.csv"
    )

    pair_path = (
        OUTDIR /
        "router_pairwise_rankings.csv"
    )

    summary_path = (
        OUTDIR /
        "router_benchmark_summary.csv"
    )

    features_path = (
        OUTDIR /
        "router_feature_sets.json"
    )

    pred_all.to_csv(
        pred_path,
        index=False,
    )

    fold_all.to_csv(
        fold_path,
        index=False,
    )

    pair_all.to_csv(
        pair_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    with open(
        features_path,
        "w",
    ) as f:
        json.dump(
            families,
            f,
            indent=2,
        )

    print("\n" + "=" * 110)
    print("FINAL NESTED GROUPED-CV RESULTS")
    print("=" * 110)

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

    print("\nSaved:")
    print(pred_path)
    print(fold_path)
    print(pair_path)
    print(summary_path)
    print(features_path)

    print("\n" + "=" * 110)
    print("ROUTER GROUPED BENCHMARK COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
