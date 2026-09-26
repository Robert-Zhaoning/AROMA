#!/usr/bin/env python3

"""
AROMA C2 — Frozen off-target selectivity endpoint analysis.

Primary endpoint
----------------
For each frozen sample i:

    B_i^method
      = mean over the seven lambda values
        of break_from_baseline(method, i, lambda)

Primary effect:

    Delta
      = mean_i [ B_i^MN - B_i^WHOLE ]

Interpretation fixed prospectively:
    Delta < 0 favors greater off-target selectivity of MN.

Primary inference:
    paired bootstrap over the N=1000 frozen samples
    20,000 replicates
    seed = 20260922
    percentile 95% CI

Secondary:
    - per-lambda break rates
    - paired discordance
    - exact McNemar tests
    - answer-flip audit
    - descriptive structural / semantic subgroups
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# Frozen inputs
# ============================================================

RAW_CSV = Path(
    "outputs/off_target_selectivity_v1/"
    "intervention_stress_test_v1/"
    "intervention_results_v1.csv"
)

RAW_FREEZE = Path(
    "manifests/off_target_selectivity_v1/"
    "raw_intervention_outcome_freeze_v1.json"
)

POPULATION = Path(
    "manifests/off_target_selectivity_v1/"
    "baseline_correct_population_v1.csv"
)

OUT_DIR = Path(
    "artifacts/off_target_selectivity_v1/"
    "endpoint_analysis_v1"
)

EXPECTED_RAW_CSV_SHA = (
    "8d01529b4c96d653085798ff6d0dae0754e828c157660cce06b0e901cd20159d"
)

EXPECTED_RAW_FREEZE_SHA = (
    "b903451e5ce4193f3645cb5a1271b15b97dc9f305026052cc9b076697b90f2ee"
)

EXPECTED_POP_SHA = (
    "76a409ea7ee097899cc091fb19ecddb8d95f2305d3167989730bf3f22abf204d"
)

LAMBDAS = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
]

METHODS = [
    "WHOLE",
    "MN",
]

N = 1000

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260922


# ============================================================
# Utilities
# ============================================================

def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


def as_bool(series):

    if series.dtype == bool:
        return series.astype(bool)

    s = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {
        "true",
        "false",
        "1",
        "0",
    }

    bad = set(
        s.unique()
    ) - allowed

    if bad:

        raise RuntimeError(
            f"Unexpected boolean values: "
            f"{sorted(bad)}"
        )

    return s.isin(
        {
            "true",
            "1",
        }
    )


def paired_bootstrap_mean(
    sample_differences,
):

    x = np.asarray(
        sample_differences,
        dtype=np.float64,
    )

    if x.shape != (
        N,
    ):

        raise RuntimeError(
            f"Expected {N} paired "
            f"sample differences, "
            f"got {x.shape}."
        )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    boot = np.empty(
        BOOTSTRAP_REPS,
        dtype=np.float64,
    )

    batch_size = 500

    pos = 0

    while pos < BOOTSTRAP_REPS:

        m = min(
            batch_size,
            BOOTSTRAP_REPS - pos,
        )

        idx = rng.integers(
            0,
            N,
            size=(
                m,
                N,
            ),
        )

        boot[
            pos:
            pos + m
        ] = (
            x[
                idx
            ]
            .mean(
                axis=1
            )
        )

        pos += m

    lo, hi = np.quantile(
        boot,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(lo),
        float(hi),
        boot,
    )


def exact_mcnemar_p(
    whole_only_break,
    mn_only_break,
):

    """
    Exact two-sided McNemar test.

    Conditional on n discordant pairs,
    X ~ Binomial(n, 0.5).

    Because p=0.5 is symmetric, the exact
    two-sided probability is twice the
    lower-tail probability at the smaller
    discordant count, capped at 1.
    """

    b = int(
        whole_only_break
    )

    c = int(
        mn_only_break
    )

    n = b + c

    if n == 0:
        return 1.0

    k = min(
        b,
        c,
    )

    lower = sum(
        math.comb(
            n,
            j,
        )
        *
        (
            0.5
            **
            n
        )
        for j in range(
            k + 1
        )
    )

    return float(
        min(
            1.0,
            2.0 * lower,
        )
    )


def write_json(
    path,
    obj,
):

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
        newline="\n",
    )


# ============================================================
# Frozen provenance audit
# ============================================================

def provenance_audit():

    print(
        "===== FROZEN PROVENANCE AUDIT ====="
    )

    for p in [
        RAW_CSV,
        RAW_FREEZE,
        POPULATION,
    ]:

        if not p.is_file():

            raise RuntimeError(
                f"Missing frozen input: {p}"
            )

    raw_sha = sha256(
        RAW_CSV
    )

    freeze_sha = sha256(
        RAW_FREEZE
    )

    pop_sha = sha256(
        POPULATION
    )

    print(
        "raw CSV SHA   :",
        raw_sha,
    )

    print(
        "raw freeze SHA:",
        freeze_sha,
    )

    print(
        "population SHA:",
        pop_sha,
    )

    assert (
        raw_sha
        ==
        EXPECTED_RAW_CSV_SHA
    )

    assert (
        freeze_sha
        ==
        EXPECTED_RAW_FREEZE_SHA
    )

    assert (
        pop_sha
        ==
        EXPECTED_POP_SHA
    )

    freeze = json.loads(
        RAW_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    assert (
        freeze[
            "analysis_state_at_freeze"
        ][
            "primary_mean_break_burden_computed"
        ]
        is False
    )

    assert int(
        freeze[
            "planned_primary_inference"
        ][
            "paired_bootstrap_replicates"
        ]
    ) == BOOTSTRAP_REPS

    assert int(
        freeze[
            "planned_primary_inference"
        ][
            "bootstrap_seed"
        ]
    ) == BOOTSTRAP_SEED

    print(
        "pre-analysis freeze: PASS"
    )


# ============================================================
# Data audit
# ============================================================

def load_and_audit():

    df = pd.read_csv(
        RAW_CSV,
        dtype={
            "question_id":
                "string",

            "image_id":
                "string",

            "structural_type":
                "string",

            "semantic_type":
                "string",

            "detailed_type":
                "string",
        },
    )

    if len(
        df
    ) != 14000:

        raise RuntimeError(
            f"Expected 14000 rows, "
            f"got {len(df)}."
        )

    df[
        "break_from_baseline"
    ] = as_bool(
        df[
            "break_from_baseline"
        ]
    )

    df[
        "answer_flip"
    ] = as_bool(
        df[
            "answer_flip"
        ]
    )

    df[
        "intervention_correct"
    ] = as_bool(
        df[
            "intervention_correct"
        ]
    )

    # Frozen population was baseline-correct.
    # Therefore break and answer flip must coincide.
    if not (
        df[
            "break_from_baseline"
        ]
        ==
        df[
            "answer_flip"
        ]
    ).all():

        raise RuntimeError(
            "break_from_baseline and "
            "answer_flip disagree."
        )

    if not (
        df[
            "break_from_baseline"
        ]
        ==
        ~df[
            "intervention_correct"
        ]
    ).all():

        raise RuntimeError(
            "break indicator and correctness "
            "are inconsistent."
        )

    if (
        df[
            "population_index"
        ].nunique()
        != N
    ):

        raise RuntimeError(
            "Expected 1000 population indices."
        )

    counts = (
        df
        .groupby(
            "population_index"
        )
        .size()
    )

    if not (
        counts == 14
    ).all():

        raise RuntimeError(
            "Each sample must have "
            "exactly 14 conditions."
        )

    print()
    print(
        "===== DATA AUDIT ====="
    )

    print(
        "rows             :",
        len(
            df
        ),
    )

    print(
        "samples          :",
        df[
            "population_index"
        ].nunique(),
    )

    print(
        "break == flip    : PASS"
    )

    print(
        "14 rows / sample : PASS"
    )

    return df


# ============================================================
# Primary endpoint
# ============================================================

def primary_endpoint(
    df,
):

    grouped = (
        df
        .groupby(
            [
                "population_index",
                "method",
            ],
            sort=True,
        )[
            "break_from_baseline"
        ]
        .mean()
        .unstack(
            "method"
        )
    )

    if list(
        grouped.index
    ) != list(
        range(
            N
        )
    ):

        raise RuntimeError(
            "Unexpected population ordering."
        )

    if set(
        grouped.columns
    ) != set(
        METHODS
    ):

        raise RuntimeError(
            "Missing method in primary endpoint."
        )

    whole = (
        grouped[
            "WHOLE"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    mn = (
        grouped[
            "MN"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    diff = (
        mn
        -
        whole
    )

    whole_mean = float(
        whole.mean()
    )

    mn_mean = float(
        mn.mean()
    )

    effect = float(
        diff.mean()
    )

    ci_lo, ci_hi, boot = (
        paired_bootstrap_mean(
            diff
        )
    )

    sample_table = (
        grouped
        .reset_index()
        .rename(
            columns={
                "WHOLE":
                    "whole_break_burden",

                "MN":
                    "mn_break_burden",
            }
        )
    )

    sample_table[
        "mn_minus_whole"
    ] = (
        sample_table[
            "mn_break_burden"
        ]
        -
        sample_table[
            "whole_break_burden"
        ]
    )

    negative_n = int(
        (
            diff < 0
        ).sum()
    )

    equal_n = int(
        (
            diff == 0
        ).sum()
    )

    positive_n = int(
        (
            diff > 0
        ).sum()
    )

    summary = {
        "population_n":
            N,

        "lambda_count":
            7,

        "whole_mean_break_burden":
            whole_mean,

        "mn_mean_break_burden":
            mn_mean,

        "mn_minus_whole":
            effect,

        "mn_minus_whole_percentage_points":
            effect
            *
            100.0,

        "paired_bootstrap_replicates":
            BOOTSTRAP_REPS,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "ci95": [
            ci_lo,
            ci_hi,
        ],

        "ci95_percentage_points": [
            ci_lo
            *
            100.0,

            ci_hi
            *
            100.0,
        ],

        "sample_level_direction": {
            "mn_lower_break_burden_n":
                negative_n,

            "equal_break_burden_n":
                equal_n,

            "mn_higher_break_burden_n":
                positive_n,
        },
    }

    print()
    print(
        "=" * 72
    )

    print(
        "PRIMARY ENDPOINT"
    )

    print(
        "=" * 72
    )

    print(
        "WHOLE mean break burden : "
        f"{100 * whole_mean:.4f}%"
    )

    print(
        "MN mean break burden    : "
        f"{100 * mn_mean:.4f}%"
    )

    print(
        "MN - WHOLE              : "
        f"{100 * effect:+.4f} pp"
    )

    print(
        "95% paired bootstrap CI : "
        f"[{100 * ci_lo:+.4f}, "
        f"{100 * ci_hi:+.4f}] pp"
    )

    print()
    print(
        "sample-level direction:"
    )

    print(
        "  MN lower :",
        negative_n,
    )

    print(
        "  equal    :",
        equal_n,
    )

    print(
        "  MN higher:",
        positive_n,
    )

    return (
        summary,
        sample_table,
        boot,
    )


# ============================================================
# Per-lambda secondary analysis
# ============================================================

def per_lambda_analysis(
    df,
):

    rows = []

    print()
    print(
        "=" * 72
    )

    print(
        "SECONDARY — PER-LAMBDA"
    )

    print(
        "=" * 72
    )

    for lam in LAMBDAS:

        g = df[
            np.isclose(
                df[
                    "lambda"
                ].astype(float),
                lam,
            )
        ]

        p = (
            g
            .pivot(
                index=
                    "population_index",

                columns=
                    "method",

                values=
                    "break_from_baseline",
            )
            .sort_index()
        )

        if p.shape != (
            N,
            2,
        ):

            raise RuntimeError(
                f"Bad paired table "
                f"for lambda={lam}: "
                f"{p.shape}"
            )

        whole = (
            p[
                "WHOLE"
            ]
            .astype(bool)
            .to_numpy()
        )

        mn = (
            p[
                "MN"
            ]
            .astype(bool)
            .to_numpy()
        )

        whole_breaks = int(
            whole.sum()
        )

        mn_breaks = int(
            mn.sum()
        )

        whole_only = int(
            (
                whole
                &
                ~mn
            ).sum()
        )

        mn_only = int(
            (
                ~whole
                &
                mn
            ).sum()
        )

        both = int(
            (
                whole
                &
                mn
            ).sum()
        )

        neither = int(
            (
                ~whole
                &
                ~mn
            ).sum()
        )

        pvalue = exact_mcnemar_p(
            whole_only,
            mn_only,
        )

        whole_rate = (
            whole_breaks
            /
            N
        )

        mn_rate = (
            mn_breaks
            /
            N
        )

        diff = (
            mn_rate
            -
            whole_rate
        )

        rec = {
            "lambda":
                lam,

            "whole_break_n":
                whole_breaks,

            "whole_break_rate":
                whole_rate,

            "mn_break_n":
                mn_breaks,

            "mn_break_rate":
                mn_rate,

            "mn_minus_whole":
                diff,

            "mn_minus_whole_pp":
                100.0
                *
                diff,

            "whole_only_break":
                whole_only,

            "mn_only_break":
                mn_only,

            "both_break":
                both,

            "neither_break":
                neither,

            "discordant_n":
                whole_only
                +
                mn_only,

            "mcnemar_exact_two_sided_p":
                pvalue,
        }

        rows.append(
            rec
        )

        print(
            f"lambda={lam:>4.2f}  "
            f"WHOLE={100*whole_rate:6.2f}%  "
            f"MN={100*mn_rate:6.2f}%  "
            f"Δ={100*diff:+6.2f}pp  "
            f"W-only={whole_only:3d}  "
            f"MN-only={mn_only:3d}  "
            f"p={pvalue:.6g}"
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Descriptive subgroup analysis
# ============================================================

def subgroup_analysis(
    df,
    field,
):

    sample_meta = (
        df[
            [
                "population_index",
                field,
            ]
        ]
        .drop_duplicates(
            "population_index"
        )
        .sort_values(
            "population_index"
        )
    )

    burdens = (
        df
        .groupby(
            [
                "population_index",
                "method",
            ]
        )[
            "break_from_baseline"
        ]
        .mean()
        .unstack(
            "method"
        )
        .reset_index()
    )

    merged = sample_meta.merge(
        burdens,
        on="population_index",
        how="inner",
        validate="one_to_one",
    )

    rows = []

    for value, g in merged.groupby(
        field,
        dropna=False,
        sort=True,
    ):

        whole = float(
            g[
                "WHOLE"
            ].mean()
        )

        mn = float(
            g[
                "MN"
            ].mean()
        )

        rows.append(
            {
                field:
                    str(
                        value
                    ),

                "n":
                    int(
                        len(
                            g
                        )
                    ),

                "whole_mean_break_burden":
                    whole,

                "mn_mean_break_burden":
                    mn,

                "mn_minus_whole":
                    mn
                    -
                    whole,

                "mn_minus_whole_pp":
                    100.0
                    *
                    (
                        mn
                        -
                        whole
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "n",
                field,
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "AROMA C2 — OFF-TARGET SELECTIVITY ENDPOINT ANALYSIS"
    )

    print(
        "=" * 72
    )

    provenance_audit()

    df = load_and_audit()

    (
        primary,
        sample_burdens,
        bootstrap_distribution,
    ) = primary_endpoint(
        df
    )

    per_lambda = (
        per_lambda_analysis(
            df
        )
    )

    structural = (
        subgroup_analysis(
            df,
            "structural_type",
        )
    )

    semantic = (
        subgroup_analysis(
            df,
            "semantic_type",
        )
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_burdens.to_csv(
        OUT_DIR /
        "sample_break_burdens_v1.csv",
        index=False,
        lineterminator="\n",
    )

    per_lambda.to_csv(
        OUT_DIR /
        "per_lambda_v1.csv",
        index=False,
        lineterminator="\n",
    )

    structural.to_csv(
        OUT_DIR /
        "by_structural_type_v1.csv",
        index=False,
        lineterminator="\n",
    )

    semantic.to_csv(
        OUT_DIR /
        "by_semantic_type_v1.csv",
        index=False,
        lineterminator="\n",
    )

    np.save(
        OUT_DIR /
        "primary_bootstrap_distribution_v1.npy",
        bootstrap_distribution,
    )

    write_json(
        OUT_DIR /
        "primary_summary_v1.json",
        primary,
    )

    manifest = {
        "experiment":
            "off_target_selectivity_v1",

        "analysis":
            "endpoint_analysis_v1",

        "raw_csv_sha256":
            sha256(
                RAW_CSV
            ),

        "raw_freeze_sha256":
            sha256(
                RAW_FREEZE
            ),

        "population_sha256":
            sha256(
                POPULATION
            ),

        "primary_endpoint":
            (
                "mean sample-level "
                "break burden MN minus WHOLE"
            ),

        "bootstrap_unit":
            "frozen sample",

        "bootstrap_replicates":
            BOOTSTRAP_REPS,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "secondary_mcnemar":
            (
                "exact two-sided, "
                "reported separately "
                "for each lambda"
            ),
    }

    write_json(
        OUT_DIR /
        "analysis_manifest_v1.json",
        manifest,
    )

    print()
    print(
        "===== DESCRIPTIVE STRUCTURAL SUBGROUPS ====="
    )

    print(
        structural.to_string(
            index=False
        )
    )

    print()
    print(
        "===== DESCRIPTIVE SEMANTIC SUBGROUPS ====="
    )

    print(
        semantic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== OUTPUTS ====="
    )

    for p in sorted(
        OUT_DIR.iterdir()
    ):

        print(
            p,
            sha256(p),
        )

    print()
    print(
        "=" * 72
    )

    print(
        "C2 ENDPOINT ANALYSIS COMPLETE"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
