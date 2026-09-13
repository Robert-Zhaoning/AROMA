from pathlib import Path
import json

import numpy as np
import pandas as pd


RESPONSE_PATH = Path(
    "outputs/proc_count_causal_v1/router/"
    "full_probability_response/"
    "full_probability_response_results.csv"
)

BASELINE_PATH = Path(
    "outputs/proc_count_causal_v1/router/"
    "preintervention_features_heldout_clean.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "full_probability_response"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PATH = (
    OUT_DIR
    / "full_response_router_features.csv"
)

MANIFEST_PATH = (
    OUT_DIR
    / "full_response_router_feature_manifest.json"
)


CANDIDATE_HEADS = [
    (33, 1),
    (3, 4),
    (18, 13),
    (8, 30),
    (3, 11),
    (13, 11),
    (33, 21),
]

NONIDENTITY_ALPHAS = [
    0.5,
    0.75,
    1.25,
    1.5,
]


# ============================================================
# Load
# ============================================================

response = pd.read_csv(
    RESPONSE_PATH
)

baseline = pd.read_csv(
    BASELINE_PATH
)

print("=" * 110)
print("AROMA FULL-RESPONSE ROUTER FEATURE BUILDER")
print("=" * 110)

print("Response rows:", len(response))
print("Baseline rows:", len(baseline))


# ============================================================
# Strict integrity
# ============================================================

assert len(response) == 1890
assert response["sample_id"].nunique() == 54
assert baseline["sample_id"].nunique() == 54

response_ids = set(
    response["sample_id"].astype(str)
)

baseline_ids = set(
    baseline["sample_id"].astype(str)
)

assert response_ids == baseline_ids

assert (
    response[
        ["layer", "head"]
    ]
    .drop_duplicates()
    .shape[0]
    == 7
)


# ============================================================
# Safe baseline features
# ============================================================

baseline_cols = [
    "pair_id",
    "role",
    "sample_id",
    "condition",
    "replicate",
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

out = baseline[
    baseline_cols
].copy()


# ============================================================
# Non-identity response only
# ============================================================

r = response[
    response["alpha"].isin(
        NONIDENTITY_ALPHAS
    )
].copy()

assert len(r) == (
    54
    * 7
    * 4
)


# ============================================================
# A. Detailed per-head × per-alpha response features
# ============================================================

metric_cols = [
    "response_l1",
    "response_l2",
    "response_tv",
    "response_js",
    "response_expected_numeral_shift",
    "delta_entropy",
    "delta_entropy_norm",
    "delta_vocab_numeral_mass",
]

for layer, head in CANDIDATE_HEADS:

    head_name = (
        f"L{layer}H{head}"
    )

    for alpha in NONIDENTITY_ALPHAS:

        g = r[
            (r["layer"].astype(int) == layer)
            & (r["head"].astype(int) == head)
            & np.isclose(
                r["alpha"],
                alpha,
            )
        ].copy()

        if g["sample_id"].nunique() != 54:
            raise RuntimeError(
                f"Incomplete block: "
                f"{head_name} alpha={alpha}"
            )

        prefix = (
            f"{head_name}_a"
            f"{str(alpha).replace('.', 'p')}"
        )

        keep = pd.DataFrame({
            "sample_id":
                g["sample_id"].values
        })

        # Full 10-way probability response.
        for n in range(1, 11):

            keep[
                f"{prefix}_dp{n}"
            ] = (
                g[
                    f"delta_numprob_{n}"
                ]
                .astype(float)
                .values
            )

        # Compact distribution metrics.
        for col in metric_cols:

            keep[
                f"{prefix}_{col}"
            ] = (
                g[col]
                .astype(float)
                .values
            )

        out = out.merge(
            keep,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )


# ============================================================
# B. Aggregate causal-susceptibility features
# ============================================================

# Head-level aggregates across alpha.
for layer, head in CANDIDATE_HEADS:

    head_name = (
        f"L{layer}H{head}"
    )

    g = r[
        (r["layer"].astype(int) == layer)
        & (r["head"].astype(int) == head)
    ].copy()

    grouped = (
        g.groupby("sample_id")
        .agg(
            mean_l1=(
                "response_l1",
                "mean",
            ),
            max_l1=(
                "response_l1",
                "max",
            ),
            mean_js=(
                "response_js",
                "mean",
            ),
            max_js=(
                "response_js",
                "max",
            ),
            mean_abs_expected_shift=(
                "response_expected_numeral_shift",
                lambda x:
                    np.abs(x).mean(),
            ),
            max_abs_expected_shift=(
                "response_expected_numeral_shift",
                lambda x:
                    np.abs(x).max(),
            ),
            mean_delta_entropy=(
                "delta_entropy",
                "mean",
            ),
            std_delta_entropy=(
                "delta_entropy",
                "std",
            ),
        )
        .fillna(0.0)
    )

    grouped.columns = [
        f"{head_name}_agg_{c}"
        for c in grouped.columns
    ]

    out = out.merge(
        grouped,
        left_on="sample_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )


# Alpha-level aggregates across heads.
for alpha in NONIDENTITY_ALPHAS:

    g = r[
        np.isclose(
            r["alpha"],
            alpha,
        )
    ].copy()

    grouped = (
        g.groupby("sample_id")
        .agg(
            mean_l1=(
                "response_l1",
                "mean",
            ),
            max_l1=(
                "response_l1",
                "max",
            ),
            std_l1=(
                "response_l1",
                "std",
            ),
            mean_js=(
                "response_js",
                "mean",
            ),
            max_js=(
                "response_js",
                "max",
            ),
            mean_abs_expected_shift=(
                "response_expected_numeral_shift",
                lambda x:
                    np.abs(x).mean(),
            ),
            max_abs_expected_shift=(
                "response_expected_numeral_shift",
                lambda x:
                    np.abs(x).max(),
            ),
            mean_delta_entropy=(
                "delta_entropy",
                "mean",
            ),
        )
        .fillna(0.0)
    )

    a = str(alpha).replace(
        ".",
        "p",
    )

    grouped.columns = [
        f"alpha_{a}_agg_{c}"
        for c in grouped.columns
    ]

    out = out.merge(
        grouped,
        left_on="sample_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )


# Global response summary.
global_agg = (
    r.groupby("sample_id")
    .agg(
        global_mean_l1=(
            "response_l1",
            "mean",
        ),
        global_max_l1=(
            "response_l1",
            "max",
        ),
        global_std_l1=(
            "response_l1",
            "std",
        ),
        global_mean_js=(
            "response_js",
            "mean",
        ),
        global_max_js=(
            "response_js",
            "max",
        ),
        global_mean_abs_expected_shift=(
            "response_expected_numeral_shift",
            lambda x:
                np.abs(x).mean(),
        ),
        global_max_abs_expected_shift=(
            "response_expected_numeral_shift",
            lambda x:
                np.abs(x).max(),
        ),
        global_mean_delta_entropy=(
            "delta_entropy",
            "mean",
        ),
        global_std_delta_entropy=(
            "delta_entropy",
            "std",
        ),
    )
    .fillna(0.0)
)

out = out.merge(
    global_agg,
    left_on="sample_id",
    right_index=True,
    how="left",
    validate="one_to_one",
)


# ============================================================
# Leakage audit
# ============================================================

assert len(out) == 54
assert out["sample_id"].nunique() == 54
assert out["pair_id"].nunique() == 27
assert out.isna().sum().sum() == 0

forbidden_tokens = [
    "ground_truth",
    "baseline_correct",
    "oracle_",
    "object_mass",
    "object_enrichment",
    "_gt_",
]

bad = [
    c
    for c in out.columns
    if any(
        token in c
        for token in forbidden_tokens
    )
]

if bad:
    raise RuntimeError(
        "Leakage columns detected: "
        f"{bad}"
    )


# ============================================================
# Feature accounting
# ============================================================

metadata_cols = {
    "pair_id",
    "role",
    "sample_id",
    "condition",
    "replicate",
}

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
    c
    for c in out.columns
    if (
        c not in metadata_cols
        and c not in numeral_features
    )
]

print("\nRows:", len(out))
print(
    "Pairs:",
    out["pair_id"].nunique(),
)
print(
    "Numeral features:",
    len(numeral_features),
)
print(
    "Full response features:",
    len(response_features),
)
print(
    "Total model features:",
    len(
        numeral_features
        + response_features
    ),
)

print("\nRole counts:")
print(
    out["role"]
    .value_counts()
    .to_string()
)


# ============================================================
# Save
# ============================================================

out.to_csv(
    OUT_PATH,
    index=False,
)

manifest = {
    "rows":
        int(len(out)),

    "pairs":
        int(
            out["pair_id"]
            .nunique()
        ),

    "numeral_features":
        numeral_features,

    "response_features":
        response_features,

    "nonidentity_alphas":
        NONIDENTITY_ALPHAS,

    "candidate_heads": [
        f"L{l}H{h}"
        for l, h
        in CANDIDATE_HEADS
    ],
}

MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
)

print("\nSaved:")
print(OUT_PATH)
print(MANIFEST_PATH)

print(
    "\nFULL-RESPONSE ROUTER "
    "FEATURE BUILD PASS"
)
