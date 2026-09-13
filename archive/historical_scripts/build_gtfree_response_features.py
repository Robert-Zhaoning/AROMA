from pathlib import Path
import json
import numpy as np
import pandas as pd

GAIN = Path(
    "outputs/proc_count_causal_v1/"
    "gain_dose_response/gain_dose_response_results.csv"
)

CLEAN = Path(
    "outputs/proc_count_causal_v1/router/"
    "preintervention_features_heldout_clean.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "gtfree_response"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "gtfree_response_features.csv"
META = OUT_DIR / "feature_manifest.json"

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

# We deliberately use perturbations near the identity alpha=1.
# alpha=1 itself is the identity and provides no response feature.
ALPHAS = [0.5, 0.75, 1.25, 1.5]

# Frozen candidate circuit from the causal validation stage.
CANDIDATE_HEADS = [
    (33, 1),
    (3, 4),
    (18, 13),
    (8, 30),
    (3, 11),
    (13, 11),
    (33, 21),
]

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

if not GAIN.exists():
    raise FileNotFoundError(GAIN)

if not CLEAN.exists():
    raise FileNotFoundError(CLEAN)

gain = pd.read_csv(GAIN)
clean = pd.read_csv(CLEAN)

print("=" * 110)
print("AROMA GT-FREE INTERVENTION-RESPONSE FEATURE BUILDER")
print("=" * 110)

print("Gain rows:", len(gain))
print("Clean held-out rows:", len(clean))
print("Held-out pairs:", clean["pair_id"].nunique())

# ------------------------------------------------------------
# Integrity checks
# ------------------------------------------------------------

assert clean["sample_id"].is_unique
assert len(clean) == 54
assert clean["pair_id"].nunique() == 27

role_counts = clean["role"].value_counts().to_dict()
assert role_counts.get("correct", 0) == 27
assert role_counts.get("wrong", 0) == 27

heldout_ids = set(clean["sample_id"])

gain = gain[gain["sample_id"].isin(heldout_ids)].copy()

assert gain["sample_id"].nunique() == 54

# Candidate only
candidate_mask = gain.apply(
    lambda r: (int(r["layer"]), int(r["head"])) in CANDIDATE_HEADS,
    axis=1,
)

gain = gain[
    (gain["head_type"] == "candidate")
    & candidate_mask
    & gain["alpha"].isin(ALPHAS)
].copy()

print("Filtered response rows:", len(gain))
print("Response samples:", gain["sample_id"].nunique())
print("Alphas:", sorted(gain["alpha"].unique().tolist()))

# Expected:
# 54 samples * 7 heads * 4 alpha values
expected = 54 * 7 * len(ALPHAS)

print("Expected rows:", expected)

if len(gain) != expected:
    print("\nWARNING: unexpected row count.")
    print(
        gain.groupby(["layer", "head", "alpha"])
        ["sample_id"].nunique().to_string()
    )

# ------------------------------------------------------------
# Base metadata + SAFE pre-intervention numeral features
# ------------------------------------------------------------

safe_base_cols = [
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
]

safe_base_cols += [
    f"numeral_prob_{n}"
    for n in range(1, 11)
]

missing = [
    c for c in safe_base_cols
    if c not in clean.columns
]

if missing:
    raise RuntimeError(
        f"Missing safe baseline columns: {missing}"
    )

out = clean[safe_base_cols].copy()

# ------------------------------------------------------------
# Validate gain baseline numeral against clean baseline
# ------------------------------------------------------------

baseline_from_gain = (
    gain.groupby("sample_id")["baseline_best_numeral"]
    .agg(lambda x: x.mode().iloc[0])
)

check = out[
    ["sample_id", "baseline_prediction"]
].merge(
    baseline_from_gain.rename("gain_baseline"),
    left_on="sample_id",
    right_index=True,
    how="left",
)

mismatch = (
    check["baseline_prediction"].astype(float)
    != check["gain_baseline"].astype(float)
)

print("\nBaseline numeral mismatches:", int(mismatch.sum()))

if mismatch.any():
    print(check[mismatch].to_string(index=False))
    raise RuntimeError(
        "Gain-run baseline is inconsistent with held-out baseline. "
        "Stop before constructing response features."
    )

# ------------------------------------------------------------
# Individual intervention response variables
#
# IMPORTANT:
# None of the following uses ground_truth.
# ------------------------------------------------------------

gain["numeral_shift"] = (
    gain["modulated_best_numeral"].astype(float)
    - gain["baseline_best_numeral"].astype(float)
)

gain["abs_numeral_shift"] = gain["numeral_shift"].abs()

gain["changed"] = gain["numeral_changed"].astype(int)

# ------------------------------------------------------------
# A. Per-alpha aggregate causal sensitivity
# ------------------------------------------------------------

for alpha in ALPHAS:

    g = gain[
        np.isclose(gain["alpha"], alpha)
    ].copy()

    agg = (
        g.groupby("sample_id")
        .agg(
            changed_rate=("changed", "mean"),
            changed_count=("changed", "sum"),
            mean_abs_shift=("abs_numeral_shift", "mean"),
            max_abs_shift=("abs_numeral_shift", "max"),
            mean_signed_shift=("numeral_shift", "mean"),
            std_signed_shift=("numeral_shift", "std"),
            response_nunique=("modulated_best_numeral", "nunique"),
        )
        .fillna(0.0)
    )

    agg.columns = [
        f"resp_a{alpha:g}_{c}"
        for c in agg.columns
    ]

    out = out.merge(
        agg,
        left_on="sample_id",
        right_index=True,
        how="left",
    )

# ------------------------------------------------------------
# B. Per-head response profile across doses
# ------------------------------------------------------------

for layer, head in CANDIDATE_HEADS:

    name = f"L{layer}H{head}"

    g = gain[
        (gain["layer"].astype(int) == layer)
        & (gain["head"].astype(int) == head)
    ].copy()

    agg = (
        g.groupby("sample_id")
        .agg(
            changed_rate=("changed", "mean"),
            changed_any=("changed", "max"),
            mean_abs_shift=("abs_numeral_shift", "mean"),
            max_abs_shift=("abs_numeral_shift", "max"),
            mean_signed_shift=("numeral_shift", "mean"),
            response_nunique=("modulated_best_numeral", "nunique"),
        )
    )

    agg.columns = [
        f"{name}_resp_{c}"
        for c in agg.columns
    ]

    out = out.merge(
        agg,
        left_on="sample_id",
        right_index=True,
        how="left",
    )

# ------------------------------------------------------------
# C. Global response geometry
# ------------------------------------------------------------

sample_global = (
    gain.groupby("sample_id")
    .agg(
        resp_global_changed_rate=("changed", "mean"),
        resp_global_changed_count=("changed", "sum"),
        resp_global_mean_abs_shift=("abs_numeral_shift", "mean"),
        resp_global_max_abs_shift=("abs_numeral_shift", "max"),
        resp_global_mean_signed_shift=("numeral_shift", "mean"),
        resp_global_std_signed_shift=("numeral_shift", "std"),
        resp_global_nunique_outputs=(
            "modulated_best_numeral",
            "nunique",
        ),
    )
    .fillna(0.0)
)

out = out.merge(
    sample_global,
    left_on="sample_id",
    right_index=True,
    how="left",
)

# ------------------------------------------------------------
# D. Directional asymmetry around alpha = 1
# ------------------------------------------------------------

def per_sample_alpha(metric, alpha):
    g = gain[np.isclose(gain["alpha"], alpha)]
    return g.groupby("sample_id")[metric].mean()

low_changed = per_sample_alpha("changed", 0.75)
high_changed = per_sample_alpha("changed", 1.25)

low_shift = per_sample_alpha("abs_numeral_shift", 0.75)
high_shift = per_sample_alpha("abs_numeral_shift", 1.25)

asym = pd.DataFrame(index=sorted(heldout_ids))

asym["resp_asym_changed_125_minus_075"] = (
    high_changed - low_changed
)

asym["resp_asym_abs_shift_125_minus_075"] = (
    high_shift - low_shift
)

asym = asym.fillna(0.0)

out = out.merge(
    asym,
    left_on="sample_id",
    right_index=True,
    how="left",
)

# ------------------------------------------------------------
# Final audit
# ------------------------------------------------------------

assert len(out) == 54
assert out["sample_id"].nunique() == 54
assert out["pair_id"].nunique() == 27
assert out.isna().sum().sum() == 0

# Explicit leakage blacklist.
forbidden_exact = {
    "ground_truth",
    "baseline_correct",
    "oracle_num_objects",
    "oracle_mean_object_patch_count",
    "baseline_gt_logp",
    "modulated_gt_logp",
    "delta_gt_logp",
}

bad = [
    c for c in out.columns
    if c in forbidden_exact
    or "object_mass" in c
    or "object_enrichment" in c
    or "_gt_" in c
]

if bad:
    raise RuntimeError(
        f"Leakage features survived: {bad}"
    )

out.to_csv(OUT, index=False)

feature_cols = [
    c for c in out.columns
    if c not in {
        "pair_id",
        "role",
        "sample_id",
        "condition",
        "replicate",
    }
]

manifest = {
    "rows": int(len(out)),
    "pairs": int(out["pair_id"].nunique()),
    "correct": int((out["role"] == "correct").sum()),
    "wrong": int((out["role"] == "wrong").sum()),
    "alphas": ALPHAS,
    "candidate_heads": [
        f"L{l}H{h}"
        for l, h in CANDIDATE_HEADS
    ],
    "n_total_features": len(feature_cols),
    "features": feature_cols,
    "notes": [
        "No ground-truth numeral is used as an input feature.",
        "No GT object boxes/token-set features are used.",
        "Response variables use only changes in model-predicted numerals.",
        "role is retained only as the supervised target.",
    ],
}

META.write_text(
    json.dumps(manifest, indent=2)
)

print("\n" + "=" * 110)
print("GT-FREE RESPONSE DATASET")
print("=" * 110)

print("Rows:", len(out))
print("Pairs:", out["pair_id"].nunique())
print("Columns:", len(out.columns))
print("Model features:", len(feature_cols))

print("\nRole counts:")
print(out["role"].value_counts())

response_cols = [
    c for c in feature_cols
    if c.startswith("resp_")
    or "_resp_" in c
]

print("\nResponse feature count:", len(response_cols))

print("\nResponse variance preview:")
variance = (
    out[response_cols]
    .var()
    .sort_values(ascending=False)
)

print(variance.head(20).to_string())

print("\nSaved:")
print(OUT)
print(META)

print("\nGT-FREE RESPONSE FEATURE BUILD PASS")
