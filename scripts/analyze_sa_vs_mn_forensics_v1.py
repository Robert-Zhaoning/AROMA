import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_vs_mn_sweep_v1"
)

RESULTS = (
    ROOT
    / "sa_vs_mn_results.csv"
)

SUMMARY = (
    ROOT
    / "sa_vs_mn_sweep_summary.csv"
)

SELECTION = (
    ROOT
    / "sa_vs_mn_selection.json"
)

DIRECTION = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_direction_diagnostic_v1/"
    "sa_dev_direction_per_sample.csv"
)

OUT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_vs_mn_forensics_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260920


# ============================================================
# HELPERS
# ============================================================

def exact_mcnemar(b, c):
    """
    Exact two-sided McNemar under p=0.5.

    b = SA correct, MN wrong
    c = MN correct, SA wrong
    """
    n = int(b + c)

    if n == 0:
        return 1.0

    k = int(min(b, c))

    tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return float(
        min(
            1.0,
            2.0 * tail,
        )
    )


def paired_bootstrap_ci(
    diff,
    reps=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):
    diff = np.asarray(
        diff,
        dtype=np.float64,
    )

    n = len(diff)

    rng = np.random.default_rng(
        seed
    )

    values = []

    batch = 1000

    done = 0

    while done < reps:

        m = min(
            batch,
            reps - done,
        )

        idx = rng.integers(
            0,
            n,
            size=(m, n),
        )

        means = (
            diff[idx]
            .mean(axis=1)
        )

        values.append(
            means
        )

        done += m

    boot = np.concatenate(
        values
    )

    return (
        float(
            np.quantile(
                boot,
                0.025,
            )
        ),
        float(
            np.quantile(
                boot,
                0.975,
            )
        ),
    )


def holm_adjust(pvalues):
    pvalues = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    m = len(pvalues)

    order = np.argsort(
        pvalues
    )

    adjusted = np.empty(
        m,
        dtype=np.float64,
    )

    running = 0.0

    for rank, idx in enumerate(
        order
    ):

        value = (
            (m - rank)
            * pvalues[idx]
        )

        running = max(
            running,
            value,
        )

        adjusted[idx] = min(
            running,
            1.0,
        )

    return adjusted


def spearman_tie(
    x,
    y,
):
    xrank = (
        pd.Series(x)
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    yrank = (
        pd.Series(y)
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    if (
        np.std(xrank) == 0
        or
        np.std(yrank) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            xrank,
            yrank,
        )[0, 1]
    )


def paired_table(
    flat,
    beta,
):
    sub = flat[
        np.isclose(
            flat["beta"],
            beta,
        )
    ].copy()

    mn = (
        sub[
            sub["method"] == "MN"
        ]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    sa = (
        sub[
            sub["method"] == "SA"
        ]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    if not np.array_equal(
        mn["sample_id"].astype(str),
        sa["sample_id"].astype(str),
    ):
        raise RuntimeError(
            f"Pairing mismatch beta={beta}"
        )

    out = mn[
        [
            "sample_id",
            "ground_truth",
            "baseline_prediction",
            "selected_alpha",
            "projected_gradient_norm",
        ]
    ].copy()

    out["mn_prediction"] = (
        mn["prediction"]
        .astype(int)
    )

    out["sa_prediction"] = (
        sa["prediction"]
        .astype(int)
    )

    gt = (
        out["ground_truth"]
        .astype(int)
        .to_numpy()
    )

    out["mn_correct"] = (
        out["mn_prediction"]
        .to_numpy()
        == gt
    ).astype(int)

    out["sa_correct"] = (
        out["sa_prediction"]
        .to_numpy()
        == gt
    ).astype(int)

    out["sa_minus_mn"] = (
        out["sa_correct"]
        - out["mn_correct"]
    )

    return out


# ============================================================
# LOAD
# ============================================================

flat = pd.read_csv(
    RESULTS
)

summary = pd.read_csv(
    SUMMARY
)

selection = json.loads(
    SELECTION.read_text(
        encoding="utf-8"
    )
)

direction = pd.read_csv(
    DIRECTION
)

direction[
    "sample_id"
] = (
    direction[
        "sample_id"
    ].astype(str)
)


# ============================================================
# 1. SELECTED PEAK COMPARISON
# ============================================================

mn_beta = float(
    selection[
        "selected"
    ][
        "MN"
    ][
        "beta"
    ]
)

sa_beta = float(
    selection[
        "selected"
    ][
        "SA"
    ][
        "beta"
    ]
)

mn_peak = paired_table(
    flat,
    mn_beta,
)

sa_peak = paired_table(
    flat,
    sa_beta,
)

peak = (
    mn_peak[
        [
            "sample_id",
            "ground_truth",
            "selected_alpha",
            "mn_correct",
        ]
    ]
    .merge(
        sa_peak[
            [
                "sample_id",
                "sa_correct",
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    )
)

b = int(
    (
        (peak["sa_correct"] == 1)
        &
        (peak["mn_correct"] == 0)
    ).sum()
)

c = int(
    (
        (peak["mn_correct"] == 1)
        &
        (peak["sa_correct"] == 0)
    ).sum()
)

peak_diff = (
    peak["sa_correct"]
    - peak["mn_correct"]
).to_numpy(
    dtype=np.float64
)

peak_ci = paired_bootstrap_ci(
    peak_diff
)

peak_p = exact_mcnemar(
    b,
    c,
)


print("=" * 96)
print("SELECTED-PEAK PAIRED COMPARISON")
print("=" * 96)

print(
    f"MN beta*: {mn_beta:.2f}"
)

print(
    f"SA beta*: {sa_beta:.2f}"
)

print(
    "SA-only correct:",
    b,
)

print(
    "MN-only correct:",
    c,
)

print(
    "SA - MN:",
    f"{peak_diff.mean() * 100:+.3f} pp",
)

print(
    "Bootstrap 95% CI:",
    f"[{peak_ci[0] * 100:+.3f}, "
    f"{peak_ci[1] * 100:+.3f}] pp",
)

print(
    "McNemar exact p:",
    peak_p,
)


# ============================================================
# 2. MATCHED-BETA DIRECTION ABLATION
# ============================================================

betas = sorted(
    flat[
        "beta"
    ].astype(float).unique()
)

matched_rows = []

for beta in betas:

    pair = paired_table(
        flat,
        beta,
    )

    b = int(
        (
            (pair["sa_correct"] == 1)
            &
            (pair["mn_correct"] == 0)
        ).sum()
    )

    c = int(
        (
            (pair["mn_correct"] == 1)
            &
            (pair["sa_correct"] == 0)
        ).sum()
    )

    matched_rows.append(
        {
            "beta":
                beta,

            "mn_correct":
                int(
                    pair[
                        "mn_correct"
                    ].sum()
                ),

            "sa_correct":
                int(
                    pair[
                        "sa_correct"
                    ].sum()
                ),

            "sa_minus_mn_pp":
                float(
                    pair[
                        "sa_minus_mn"
                    ].mean()
                    * 100
                ),

            "sa_only_correct":
                b,

            "mn_only_correct":
                c,

            "discordant_n":
                b + c,

            "mcnemar_p":
                exact_mcnemar(
                    b,
                    c,
                ),
        }
    )

matched = pd.DataFrame(
    matched_rows
)

matched[
    "mcnemar_holm_p"
] = holm_adjust(
    matched[
        "mcnemar_p"
    ].to_numpy()
)

matched.to_csv(
    OUT
    / "matched_beta_paired.csv",
    index=False,
)

print()
print("=" * 96)
print("MATCHED-BETA DIRECTION ABLATION")
print("=" * 96)

print(
    matched.to_string(
        index=False
    )
)


# ============================================================
# 3. SAME-ACCURACY / SMALLER-BUDGET MATCHES
# ============================================================

mn_sum = (
    summary[
        summary["method"]
        == "MN"
    ][
        [
            "beta",
            "full_correct",
            "full_accuracy",
        ]
    ]
    .copy()
)

sa_sum = (
    summary[
        summary["method"]
        == "SA"
    ][
        [
            "beta",
            "full_correct",
            "full_accuracy",
        ]
    ]
    .copy()
)

budget_matches = []

for _, m in mn_sum.iterrows():

    candidates = sa_sum[
        sa_sum[
            "full_correct"
        ]
        ==
        m[
            "full_correct"
        ]
    ]

    for _, s in candidates.iterrows():

        if float(
            s["beta"]
        ) < float(
            m["beta"]
        ):

            reduction = (
                1.0
                -
                float(
                    s["beta"]
                )
                /
                float(
                    m["beta"]
                )
            )

            budget_matches.append(
                {
                    "full_correct":
                        int(
                            m[
                                "full_correct"
                            ]
                        ),

                    "accuracy":
                        float(
                            m[
                                "full_accuracy"
                            ]
                        ),

                    "mn_beta":
                        float(
                            m["beta"]
                        ),

                    "sa_beta":
                        float(
                            s["beta"]
                        ),

                    "sa_budget_reduction":
                        reduction,
                }
            )

budget = pd.DataFrame(
    budget_matches
)

budget.to_csv(
    OUT
    / "same_accuracy_budget_matches.csv",
    index=False,
)

print()
print("=" * 96)
print("SAME ACCURACY WITH SMALLER SA BETA")
print("=" * 96)

if len(budget):

    print(
        budget.to_string(
            index=False
        )
    )

else:

    print(
        "No exact same-correct-count matches."
    )


# ============================================================
# 4. BETA=1 FAIR-BUDGET FORENSICS
#
# beta=1 is the natural equal-whole-relative displacement
# operating point and isolates direction at equal budget.
# ============================================================

pair1 = paired_table(
    flat,
    1.0,
)

pair1 = pair1.merge(
    direction[
        [
            "sample_id",
            "cos_pug_puh",
            "pg_norm",
            "gradient_capture",
            "activation_capture",
        ]
    ],
    on="sample_id",
    validate="one_to_one",
)

pair1[
    "log10_pg_norm"
] = np.log10(
    np.maximum(
        pair1[
            "pg_norm"
        ].to_numpy(
            dtype=np.float64
        ),
        1e-30,
    )
)

rho_cos = spearman_tie(
    pair1[
        "cos_pug_puh"
    ],
    pair1[
        "sa_minus_mn"
    ],
)

rho_pg = spearman_tie(
    pair1[
        "log10_pg_norm"
    ],
    pair1[
        "sa_minus_mn"
    ],
)


# ============================================================
# TERTILES
# ============================================================

cos_q1, cos_q2 = np.quantile(
    pair1[
        "cos_pug_puh"
    ],
    [
        1 / 3,
        2 / 3,
    ],
)

pg_q1, pg_q2 = np.quantile(
    pair1[
        "log10_pg_norm"
    ],
    [
        1 / 3,
        2 / 3,
    ],
)


def tertile_label(
    x,
    q1,
    q2,
):
    if x <= q1:
        return "low"
    elif x <= q2:
        return "mid"
    return "high"


pair1[
    "cos_tertile"
] = [
    tertile_label(
        x,
        cos_q1,
        cos_q2,
    )
    for x in pair1[
        "cos_pug_puh"
    ]
]

pair1[
    "pg_tertile"
] = [
    tertile_label(
        x,
        pg_q1,
        pg_q2,
    )
    for x in pair1[
        "log10_pg_norm"
    ]
]


def grouped_forensics(
    df,
    column,
):
    rows = []

    for group in [
        "low",
        "mid",
        "high",
    ]:

        x = df[
            df[column]
            == group
        ]

        sa_only = int(
            (
                (
                    x[
                        "sa_correct"
                    ]
                    == 1
                )
                &
                (
                    x[
                        "mn_correct"
                    ]
                    == 0
                )
            ).sum()
        )

        mn_only = int(
            (
                (
                    x[
                        "mn_correct"
                    ]
                    == 1
                )
                &
                (
                    x[
                        "sa_correct"
                    ]
                    == 0
                )
            ).sum()
        )

        rows.append(
            {
                "group":
                    group,

                "n":
                    len(x),

                "mn_accuracy":
                    float(
                        x[
                            "mn_correct"
                        ].mean()
                    ),

                "sa_accuracy":
                    float(
                        x[
                            "sa_correct"
                        ].mean()
                    ),

                "sa_minus_mn_pp":
                    float(
                        x[
                            "sa_minus_mn"
                        ].mean()
                        * 100
                    ),

                "sa_only_correct":
                    sa_only,

                "mn_only_correct":
                    mn_only,

                "net":
                    sa_only
                    - mn_only,
            }
        )

    return pd.DataFrame(
        rows
    )


cos_groups = grouped_forensics(
    pair1,
    "cos_tertile",
)

pg_groups = grouped_forensics(
    pair1,
    "pg_tertile",
)

cos_groups.to_csv(
    OUT
    / "beta1_cosine_tertiles.csv",
    index=False,
)

pg_groups.to_csv(
    OUT
    / "beta1_projected_gradient_tertiles.csv",
    index=False,
)

pair1.to_csv(
    OUT
    / "beta1_per_sample.csv",
    index=False,
)


print()
print("=" * 96)
print("BETA=1 DIRECTION FORENSICS")
print("=" * 96)

print(
    "Spearman("
    "cos(PUg,PUh), "
    "SA-MN correctness"
    "):",
    rho_cos,
)

print(
    "Spearman("
    "log10||PUg||, "
    "SA-MN correctness"
    "):",
    rho_pg,
)

print()
print(
    "Cosine tertiles:"
)

print(
    cos_groups.to_string(
        index=False
    )
)

print()
print(
    "Projected-gradient tertiles:"
)

print(
    pg_groups.to_string(
        index=False
    )
)


# ============================================================
# 5. LOW-GRADIENT DIAGNOSTIC
# ============================================================

low_threshold = 1e-6

low = pair1[
    pair1[
        "pg_norm"
    ]
    <= low_threshold
]

high = pair1[
    pair1[
        "pg_norm"
    ]
    > low_threshold
]


def simple_group(
    name,
    x,
):
    return {
        "group":
            name,

        "n":
            len(x),

        "mn_accuracy":
            float(
                x[
                    "mn_correct"
                ].mean()
            ),

        "sa_accuracy":
            float(
                x[
                    "sa_correct"
                ].mean()
            ),

        "sa_minus_mn_pp":
            float(
                x[
                    "sa_minus_mn"
                ].mean()
                * 100
            ),
    }


low_grad = pd.DataFrame(
    [
        simple_group(
            "<=1e-6",
            low,
        ),
        simple_group(
            ">1e-6",
            high,
        ),
    ]
)

low_grad.to_csv(
    OUT
    / "beta1_low_gradient_groups.csv",
    index=False,
)

print()
print(
    "Low projected-gradient diagnostic:"
)

print(
    low_grad.to_string(
        index=False
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

final = {
    "selected_peak": {
        "mn_beta":
            mn_beta,

        "sa_beta":
            sa_beta,

        "sa_minus_mn_pp":
            float(
                peak_diff.mean()
                * 100
            ),

        "bootstrap_95_ci_pp": [
            float(
                peak_ci[0]
                * 100
            ),
            float(
                peak_ci[1]
                * 100
            ),
        ],

        "sa_only_correct":
            b,

        "mn_only_correct":
            c,

        "mcnemar_exact_p":
            peak_p,
    },

    "beta1_forensics": {
        "spearman_cos_vs_advantage":
            rho_cos,

        "spearman_log_pg_vs_advantage":
            rho_pg,

        "cos_tertile_cutpoints": [
            float(cos_q1),
            float(cos_q2),
        ],

        "log_pg_tertile_cutpoints": [
            float(pg_q1),
            float(pg_q2),
        ],
    },
}

(
    OUT
    / "forensic_summary.json"
).write_text(
    json.dumps(
        final,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 96)
print("SA vs MN FORENSIC ANALYSIS COMPLETE")
print("=" * 96)

print(
    "Output:",
    OUT,
)
