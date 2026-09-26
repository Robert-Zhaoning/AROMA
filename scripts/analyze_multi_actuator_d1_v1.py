from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest


HEADS = [
    "L33H1",
    "L3H4",
    "L18H13",
    "L8H30",
    "L3H11",
    "L13H11",
    "L33H21",
]

ROOT = Path(
    "outputs/proc_count_multi_actuator_d1_v1/inference"
)

ANALYSIS_ROOT = Path(
    "outputs/proc_count_multi_actuator_d1_v1/analysis"
)

ANALYSIS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_FREEZE = Path(
    "manifests/multi_actuator_d1_v1/"
    "raw_outcomes_freeze_v1.json"
)

PRIMARY_CSV = (
    ANALYSIS_ROOT
    / "primary_endpoint_v1.csv"
)

PRIMARY_JSON = (
    ANALYSIS_ROOT
    / "primary_endpoint_v1.json"
)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def holm_adjust(pvalues):
    """
    Holm step-down adjusted p-values.

    Input:
        list of raw p-values

    Output:
        numpy array of adjusted p-values,
        returned in original order.
    """

    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(pvalues)

    order = np.argsort(
        pvalues
    )

    sorted_p = (
        pvalues[
            order
        ]
    )

    adjusted_sorted = (
        np.empty(
            m,
            dtype=float,
        )
    )

    running_max = 0.0

    for i, p in enumerate(
        sorted_p
    ):

        candidate = (
            (m - i)
            * p
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted_sorted[
            i
        ] = min(
            1.0,
            running_max,
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[
        order
    ] = adjusted_sorted

    return adjusted


# ============================================================
# VERIFY RAW FILES AGAINST FROZEN HASHES
# ============================================================

freeze = json.loads(
    RAW_FREEZE.read_text(
        encoding="utf-8"
    )
)

for head in HEADS:

    p = (
        ROOT
        / head
        / "results.csv"
    )

    expected = (
        freeze[
            "files"
        ][
            head
        ][
            "results_sha256"
        ]
    )

    actual = sha256(
        p
    )

    if actual != expected:
        raise RuntimeError(
            f"{head}: raw outcome SHA256 mismatch."
        )

print(
    "RAW OUTCOME HASH VERIFICATION: PASS"
)


# ============================================================
# PRIMARY DIRECTIONAL ENDPOINTS
# ============================================================

rows = []
tests = []

for head in HEADS:

    p = (
        ROOT
        / head
        / "results.csv"
    )

    df = pd.read_csv(
        p
    )

    if len(df) != 900:
        raise RuntimeError(
            f"{head}: expected 900 rows."
        )

    down = (
        df[
            df["alpha"] == 0.5
        ]
        .sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    identity = (
        df[
            df["alpha"] == 1.0
        ]
        .sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    up = (
        df[
            df["alpha"] == 1.5
        ]
        .sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    if not (
        len(down)
        ==
        len(identity)
        ==
        len(up)
        ==
        300
    ):
        raise RuntimeError(
            f"{head}: alpha partition failure."
        )

    if not (
        down["sample_id"].tolist()
        ==
        identity["sample_id"].tolist()
        ==
        up["sample_id"].tolist()
    ):
        raise RuntimeError(
            f"{head}: sample alignment failure."
        )

    down_shift = (
        down[
            "response_expected_numeral_shift"
        ]
        .to_numpy(
            dtype=float
        )
    )

    up_shift = (
        up[
            "response_expected_numeral_shift"
        ]
        .to_numpy(
            dtype=float
        )
    )


    # --------------------------------------------------------
    # Primary consistency definition:
    #
    # attenuation success:
    #     shift < 0
    #
    # amplification success:
    #     shift > 0
    #
    # Exact zeros are retained in denominator and count as
    # non-successes.
    # --------------------------------------------------------

    down_success = int(
        np.sum(
            down_shift < 0
        )
    )

    up_success = int(
        np.sum(
            up_shift > 0
        )
    )

    down_ties = int(
        np.sum(
            down_shift == 0
        )
    )

    up_ties = int(
        np.sum(
            up_shift == 0
        )
    )


    down_test = binomtest(
        down_success,
        n=300,
        p=0.5,
        alternative="greater",
    )

    up_test = binomtest(
        up_success,
        n=300,
        p=0.5,
        alternative="greater",
    )


    down_ci = (
        down_test
        .proportion_ci(
            confidence_level=0.95,
            method="exact",
        )
    )

    up_ci = (
        up_test
        .proportion_ci(
            confidence_level=0.95,
            method="exact",
        )
    )


    # --------------------------------------------------------
    # Secondary strict ordering:
    #
    # E[n | alpha=.5]
    # <
    # E[n | alpha=1]
    # <
    # E[n | alpha=1.5]
    # --------------------------------------------------------

    expected_down = (
        down[
            "modulated_expected_numeral"
        ]
        .to_numpy(
            dtype=float
        )
    )

    expected_identity = (
        identity[
            "modulated_expected_numeral"
        ]
        .to_numpy(
            dtype=float
        )
    )

    expected_up = (
        up[
            "modulated_expected_numeral"
        ]
        .to_numpy(
            dtype=float
        )
    )

    strict_order = (
        (
            expected_down
            <
            expected_identity
        )
        &
        (
            expected_identity
            <
            expected_up
        )
    )

    strict_order_n = int(
        np.sum(
            strict_order
        )
    )


    row = {
        "head":
            head,

        "n":
            300,

        "attenuation_success_n":
            down_success,

        "attenuation_consistency":
            down_success / 300.0,

        "attenuation_ties":
            down_ties,

        "attenuation_mean_shift":
            float(
                np.mean(
                    down_shift
                )
            ),

        "attenuation_median_shift":
            float(
                np.median(
                    down_shift
                )
            ),

        "attenuation_ci_low":
            float(
                down_ci.low
            ),

        "attenuation_ci_high":
            float(
                down_ci.high
            ),

        "attenuation_p_raw":
            float(
                down_test.pvalue
            ),

        "amplification_success_n":
            up_success,

        "amplification_consistency":
            up_success / 300.0,

        "amplification_ties":
            up_ties,

        "amplification_mean_shift":
            float(
                np.mean(
                    up_shift
                )
            ),

        "amplification_median_shift":
            float(
                np.median(
                    up_shift
                )
            ),

        "amplification_ci_low":
            float(
                up_ci.low
            ),

        "amplification_ci_high":
            float(
                up_ci.high
            ),

        "amplification_p_raw":
            float(
                up_test.pvalue
            ),

        "strict_order_n":
            strict_order_n,

        "strict_order_fraction":
            strict_order_n / 300.0,
    }

    rows.append(
        row
    )

    tests.append(
        {
            "head":
                head,

            "direction":
                "attenuation",

            "p_raw":
                float(
                    down_test.pvalue
                ),
        }
    )

    tests.append(
        {
            "head":
                head,

            "direction":
                "amplification",

            "p_raw":
                float(
                    up_test.pvalue
                ),
        }
    )


# ============================================================
# HOLM CORRECTION ACROSS ALL 14 DIRECTIONAL TESTS
# ============================================================

raw_ps = [
    t["p_raw"]
    for t in tests
]

adjusted_ps = holm_adjust(
    raw_ps
)

for test, p_adj in zip(
    tests,
    adjusted_ps,
):
    test[
        "p_holm"
    ] = float(
        p_adj
    )


lookup = {
    (
        t["head"],
        t["direction"],
    ):
        t
    for t in tests
}


for row in rows:

    head = row[
        "head"
    ]

    down_adj = (
        lookup[
            (
                head,
                "attenuation",
            )
        ][
            "p_holm"
        ]
    )

    up_adj = (
        lookup[
            (
                head,
                "amplification",
            )
        ][
            "p_holm"
        ]
    )

    row[
        "attenuation_p_holm"
    ] = down_adj

    row[
        "amplification_p_holm"
    ] = up_adj

    row[
        "attenuation_pass_holm"
    ] = (
        down_adj
        < 0.05
    )

    row[
        "amplification_pass_holm"
    ] = (
        up_adj
        < 0.05
    )

    row[
        "bidirectional_replication"
    ] = (
        row[
            "attenuation_pass_holm"
        ]
        and
        row[
            "amplification_pass_holm"
        ]
    )


result = pd.DataFrame(
    rows
)

result = result[
    [
        "head",
        "n",

        "attenuation_success_n",
        "attenuation_consistency",
        "attenuation_ties",
        "attenuation_mean_shift",
        "attenuation_median_shift",
        "attenuation_ci_low",
        "attenuation_ci_high",
        "attenuation_p_raw",
        "attenuation_p_holm",
        "attenuation_pass_holm",

        "amplification_success_n",
        "amplification_consistency",
        "amplification_ties",
        "amplification_mean_shift",
        "amplification_median_shift",
        "amplification_ci_low",
        "amplification_ci_high",
        "amplification_p_raw",
        "amplification_p_holm",
        "amplification_pass_holm",

        "strict_order_n",
        "strict_order_fraction",

        "bidirectional_replication",
    ]
]


result.to_csv(
    PRIMARY_CSV,
    index=False,
)


replicated = (
    result[
        result[
            "bidirectional_replication"
        ]
    ][
        "head"
    ]
    .tolist()
)


nonprimary_replicated = [
    h
    for h in replicated
    if h != "L18H13"
]


summary = {
    "experiment":
        "Experiment D1 — Multi-actuator causal replication",

    "primary_endpoint":
        (
            "Per-sample expected-numeral directional "
            "consistency under alpha=.5 attenuation and "
            "alpha=1.5 amplification."
        ),

    "n_per_head":
        300,

    "heads_tested":
        HEADS,

    "multiple_testing":
        "Holm correction across 14 directional tests",

    "alpha":
        0.05,

    "ties":
        (
            "Exact zero shifts retained in denominator "
            "and counted as non-successes."
        ),

    "bidirectional_replication_rule":
        (
            "Both attenuation and amplification one-sided "
            "exact binomial tests must have Holm-adjusted "
            "p < 0.05."
        ),

    "replicated_heads":
        replicated,

    "replicated_head_count":
        len(
            replicated
        ),

    "nonprimary_replicated_heads":
        nonprimary_replicated,

    "nonprimary_replicated_head_count":
        len(
            nonprimary_replicated
        ),

    "primary_csv":
        str(
            PRIMARY_CSV
        ),

    "primary_csv_sha256":
        sha256(
            PRIMARY_CSV
        ),

    "raw_outcomes_freeze_sha256":
        sha256(
            RAW_FREEZE
        ),
}


PRIMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# HUMAN-READABLE OUTPUT
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    220,
)

display = result[
    [
        "head",

        "attenuation_success_n",
        "attenuation_consistency",
        "attenuation_ties",
        "attenuation_mean_shift",
        "attenuation_p_holm",

        "amplification_success_n",
        "amplification_consistency",
        "amplification_ties",
        "amplification_mean_shift",
        "amplification_p_holm",

        "strict_order_fraction",
        "bidirectional_replication",
    ]
].copy()


print()
print(
    "======================================================================"
)

print(
    "D1 PRIMARY ENDPOINT RESULTS"
)

print(
    "======================================================================"
)

print(
    display.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print()
print(
    "Bidirectional replicated heads:",
    replicated,
)

print(
    "Replicated:",
    len(replicated),
    "/",
    len(HEADS),
)

print(
    "Non-primary replicated:",
    len(nonprimary_replicated),
    "/ 6",
)

print()
print(
    "Saved:",
    PRIMARY_CSV,
)

print(
    "Saved:",
    PRIMARY_JSON,
)
