from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/proc_count_causal_v1/"
    "dam_head_profiles/"
    "head_response_profiles.csv"
)

SELECTIVE_INPUT = Path(
    "outputs/proc_count_causal_v1/"
    "dam_v2_selective_validation/"
    "selective_validation_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/"
    "dam_head_profiles"
)

SEED = 20260910
N_BOOT = 20000


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    boot = values[idx].mean(axis=1)

    return (
        float(values.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def build_oracle(df):
    idx = (
        df.groupby("sample_id")["delta_logp"]
        .idxmax()
    )

    oracle = (
        df.loc[idx]
        .copy()
        .sort_values(["role", "pair_id"])
        .reset_index(drop=True)
    )

    oracle = oracle.rename(
        columns={
            "head_name": "oracle_head",
            "layer": "oracle_layer",
            "head": "oracle_head_index",
            "delta_logp": "oracle_delta_logp",
            "delta_margin": "oracle_delta_margin",
        }
    )

    return oracle


def summarize_oracle(oracle):
    rows = []

    for role, g in oracle.groupby("role"):
        mean_logp, lo_logp, hi_logp = bootstrap_mean_ci(
            g["oracle_delta_logp"]
        )

        mean_margin, lo_margin, hi_margin = bootstrap_mean_ci(
            g["oracle_delta_margin"]
        )

        rows.append(
            {
                "role": role,
                "n": len(g),
                "mean_oracle_delta_logp": mean_logp,
                "oracle_logp_ci_low": lo_logp,
                "oracle_logp_ci_high": hi_logp,
                "mean_oracle_delta_margin": mean_margin,
                "oracle_margin_ci_low": lo_margin,
                "oracle_margin_ci_high": hi_margin,
                "positive_oracle_fraction": float(
                    (g["oracle_delta_logp"] > 0).mean()
                ),
                "strong_positive_fraction_0p05": float(
                    (g["oracle_delta_logp"] > 0.05).mean()
                ),
                "strong_positive_fraction_0p10": float(
                    (g["oracle_delta_logp"] > 0.10).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def head_win_counts(oracle):
    return (
        oracle.groupby(
            ["role", "oracle_head"],
            as_index=False,
        )
        .agg(
            wins=("sample_id", "count"),
            mean_best_delta_logp=(
                "oracle_delta_logp",
                "mean",
            ),
            mean_best_delta_margin=(
                "oracle_delta_margin",
                "mean",
            ),
        )
        .sort_values(
            ["role", "wins", "mean_best_delta_logp"],
            ascending=[True, False, False],
        )
    )


def per_head_summary(df):
    rows = []

    for (role, head_name), g in df.groupby(
        ["role", "head_name"]
    ):
        mean_logp, lo_logp, hi_logp = bootstrap_mean_ci(
            g["delta_logp"]
        )

        rows.append(
            {
                "role": role,
                "head_name": head_name,
                "layer": int(g["layer"].iloc[0]),
                "head": int(g["head"].iloc[0]),
                "n": len(g),
                "mean_delta_logp": mean_logp,
                "ci_low": lo_logp,
                "ci_high": hi_logp,
                "positive_fraction": float(
                    (g["delta_logp"] > 0).mean()
                ),
                "best_head_fraction": float(
                    g["is_best_head"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def oracle_by_condition(oracle):
    return (
        oracle.groupby(
            ["role", "condition"],
            as_index=False,
        )
        .agg(
            n=("sample_id", "count"),
            mean_oracle_delta_logp=(
                "oracle_delta_logp",
                "mean",
            ),
            mean_oracle_delta_margin=(
                "oracle_delta_margin",
                "mean",
            ),
            positive_fraction=(
                "oracle_delta_logp",
                lambda x: float((x > 0).mean()),
            ),
        )
    )


def oracle_by_count(oracle):
    return (
        oracle.groupby(
            ["role", "ground_truth"],
            as_index=False,
        )
        .agg(
            n=("sample_id", "count"),
            mean_oracle_delta_logp=(
                "oracle_delta_logp",
                "mean",
            ),
            mean_oracle_delta_margin=(
                "oracle_delta_margin",
                "mean",
            ),
            positive_fraction=(
                "oracle_delta_logp",
                lambda x: float((x > 0).mean()),
            ),
        )
    )


def build_oracle_vs_fixed(oracle):
    if not SELECTIVE_INPUT.exists():
        return pd.DataFrame()

    fixed = pd.read_csv(SELECTIVE_INPUT)

    fixed = fixed[
        fixed["subset"].isin(
            [
                "selective_A",
                "all7",
                "single_L3H4",
                "single_L8H30",
            ]
        )
    ].copy()

    oracle_small = oracle[
        [
            "pair_id",
            "sample_id",
            "role",
            "oracle_head",
            "oracle_delta_logp",
            "oracle_delta_margin",
        ]
    ].copy()

    # selective file uses pair_idx rather than original pair_id,
    # so sample_id + role is safer for joining.
    comparisons = []

    for subset in [
        "selective_A",
        "all7",
        "single_L3H4",
        "single_L8H30",
    ]:
        sub = fixed[
            fixed["subset"] == subset
        ][
            [
                "sample_id",
                "role",
                "delta_logp",
                "delta_margin",
            ]
        ].rename(
            columns={
                "delta_logp": "fixed_delta_logp",
                "delta_margin": "fixed_delta_margin",
            }
        )

        merged = oracle_small.merge(
            sub,
            on=["sample_id", "role"],
            how="inner",
        )

        merged["subset"] = subset
        merged["oracle_minus_fixed_logp"] = (
            merged["oracle_delta_logp"]
            - merged["fixed_delta_logp"]
        )

        merged["oracle_minus_fixed_margin"] = (
            merged["oracle_delta_margin"]
            - merged["fixed_delta_margin"]
        )

        comparisons.append(merged)

    if not comparisons:
        return pd.DataFrame()

    return pd.concat(
        comparisons,
        ignore_index=True,
    )


def summarize_oracle_vs_fixed(comp):
    if comp.empty:
        return pd.DataFrame()

    rows = []

    for (role, subset), g in comp.groupby(
        ["role", "subset"]
    ):
        mean_logp, lo_logp, hi_logp = bootstrap_mean_ci(
            g["oracle_minus_fixed_logp"]
        )

        mean_margin, lo_margin, hi_margin = bootstrap_mean_ci(
            g["oracle_minus_fixed_margin"]
        )

        rows.append(
            {
                "role": role,
                "subset": subset,
                "n": len(g),
                "mean_oracle_minus_fixed_logp": mean_logp,
                "ci_low": lo_logp,
                "ci_high": hi_logp,
                "mean_oracle_minus_fixed_margin": mean_margin,
                "margin_ci_low": lo_margin,
                "margin_ci_high": hi_margin,
                "oracle_beats_fixed_fraction": float(
                    (
                        g["oracle_minus_fixed_logp"] > 0
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def head_diversity_metrics(oracle):
    rows = []

    for role, g in oracle.groupby("role"):
        counts = g["oracle_head"].value_counts()

        probs = counts / counts.sum()

        entropy = float(
            -(probs * np.log2(probs)).sum()
        )

        dominant_fraction = float(
            probs.max()
        )

        rows.append(
            {
                "role": role,
                "n_samples": len(g),
                "unique_best_heads": int(
                    g["oracle_head"].nunique()
                ),
                "head_entropy_bits": entropy,
                "dominant_head_fraction": dominant_fraction,
            }
        )

    return pd.DataFrame(rows)


def paired_failure_specificity(oracle):
    wrong = oracle[
        oracle["role"] == "wrong"
    ][
        [
            "pair_id",
            "oracle_delta_logp",
            "oracle_delta_margin",
        ]
    ].rename(
        columns={
            "oracle_delta_logp": "wrong_oracle_logp",
            "oracle_delta_margin": "wrong_oracle_margin",
        }
    )

    correct = oracle[
        oracle["role"] == "correct"
    ][
        [
            "pair_id",
            "oracle_delta_logp",
            "oracle_delta_margin",
        ]
    ].rename(
        columns={
            "oracle_delta_logp": "correct_oracle_logp",
            "oracle_delta_margin": "correct_oracle_margin",
        }
    )

    paired = wrong.merge(
        correct,
        on="pair_id",
        how="inner",
    )

    paired["specificity_logp"] = (
        paired["wrong_oracle_logp"]
        - paired["correct_oracle_logp"]
    )

    paired["specificity_margin"] = (
        paired["wrong_oracle_margin"]
        - paired["correct_oracle_margin"]
    )

    mean_logp, lo_logp, hi_logp = bootstrap_mean_ci(
        paired["specificity_logp"]
    )

    mean_margin, lo_margin, hi_margin = bootstrap_mean_ci(
        paired["specificity_margin"]
    )

    return paired, {
        "n_pairs": len(paired),
        "mean_specificity_logp": mean_logp,
        "specificity_logp_ci_low": lo_logp,
        "specificity_logp_ci_high": hi_logp,
        "mean_specificity_margin": mean_margin,
        "specificity_margin_ci_low": lo_margin,
        "specificity_margin_ci_high": hi_margin,
        "wrong_oracle_better_fraction": float(
            (
                paired["specificity_logp"] > 0
            ).mean()
        ),
    }


def main():
    print("=" * 110)
    print(
        "AROMA DAM-v2 Adaptive Head Oracle Analysis"
    )
    print("=" * 110)

    df = pd.read_csv(INPUT)

    print("\nRows:", len(df))
    print(
        "Samples:",
        df["sample_id"].nunique(),
    )
    print(
        "Pairs:",
        df["pair_id"].nunique(),
    )
    print(
        "Heads:",
        df["head_name"].nunique(),
    )

    assert len(df) == 378
    assert df["sample_id"].nunique() == 54
    assert df["head_name"].nunique() == 7
    assert df["calls_ok"].all()

    oracle = build_oracle(df)

    oracle_path = (
        OUT_DIR /
        "oracle_best_head_per_sample.csv"
    )

    oracle.to_csv(
        oracle_path,
        index=False,
    )

    summary = summarize_oracle(oracle)

    summary_path = (
        OUT_DIR /
        "oracle_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print("ORACLE BEST-HEAD PERFORMANCE")
    print("=" * 110)
    print(
        summary.to_string(
            index=False
        )
    )

    wins = head_win_counts(oracle)

    wins_path = (
        OUT_DIR /
        "oracle_head_win_counts.csv"
    )

    wins.to_csv(
        wins_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print("BEST-HEAD DISTRIBUTION")
    print("=" * 110)
    print(
        wins.to_string(
            index=False
        )
    )

    diversity = head_diversity_metrics(
        oracle
    )

    diversity_path = (
        OUT_DIR /
        "oracle_head_diversity.csv"
    )

    diversity.to_csv(
        diversity_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print("HEAD DIVERSITY")
    print("=" * 110)
    print(
        diversity.to_string(
            index=False
        )
    )

    per_head = per_head_summary(df)

    per_head_path = (
        OUT_DIR /
        "per_head_heldout_summary.csv"
    )

    per_head.to_csv(
        per_head_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print("PER-HEAD HELD-OUT PERFORMANCE — WRONG")
    print("=" * 110)

    wrong_heads = (
        per_head[
            per_head["role"] == "wrong"
        ]
        .sort_values(
            "mean_delta_logp",
            ascending=False,
        )
    )

    print(
        wrong_heads.to_string(
            index=False
        )
    )

    by_condition = oracle_by_condition(
        oracle
    )

    by_condition_path = (
        OUT_DIR /
        "oracle_by_condition.csv"
    )

    by_condition.to_csv(
        by_condition_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print("ORACLE BY CONDITION")
    print("=" * 110)
    print(
        by_condition.to_string(
            index=False
        )
    )

    by_count = oracle_by_count(
        oracle
    )

    by_count_path = (
        OUT_DIR /
        "oracle_by_count.csv"
    )

    by_count.to_csv(
        by_count_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print("ORACLE BY COUNT")
    print("=" * 110)
    print(
        by_count.to_string(
            index=False
        )
    )

    comp = build_oracle_vs_fixed(
        oracle
    )

    if not comp.empty:
        comp_path = (
            OUT_DIR /
            "oracle_vs_fixed_raw.csv"
        )

        comp.to_csv(
            comp_path,
            index=False,
        )

        comp_summary = (
            summarize_oracle_vs_fixed(
                comp
            )
        )

        comp_summary_path = (
            OUT_DIR /
            "oracle_vs_fixed_summary.csv"
        )

        comp_summary.to_csv(
            comp_summary_path,
            index=False,
        )

        print("\n" + "=" * 110)
        print(
            "ORACLE BEST HEAD VS FIXED INTERVENTIONS"
        )
        print("=" * 110)
        print(
            comp_summary.to_string(
                index=False
            )
        )

    paired, spec = (
        paired_failure_specificity(
            oracle
        )
    )

    paired_path = (
        OUT_DIR /
        "oracle_failure_specificity_pairs.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    spec_df = pd.DataFrame([spec])

    spec_path = (
        OUT_DIR /
        "oracle_failure_specificity_summary.csv"
    )

    spec_df.to_csv(
        spec_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print(
        "ORACLE FAILURE-STATE SPECIFICITY"
    )
    print("=" * 110)
    print(
        spec_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    for path in [
        oracle_path,
        summary_path,
        wins_path,
        diversity_path,
        per_head_path,
        by_condition_path,
        by_count_path,
        paired_path,
        spec_path,
    ]:
        print(path)

    print(
        "\nADAPTIVE HEAD ORACLE ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
