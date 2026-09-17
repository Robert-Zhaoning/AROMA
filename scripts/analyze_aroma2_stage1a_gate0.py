from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


SEED = 20260914
N_BOOT = 20000
RIDGE_ALPHA = 0.01

ACTIONS = [0.0, 1.5, 2.0, 4.0]

OUT = Path("outputs/aroma2/stage1a_gate0")
OUT.mkdir(parents=True, exist_ok=True)

P05 = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/down_alpha0p5_expanded/"
    "down_alpha0p5_results.csv"
)

P15 = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5_expanded_0_15/"
    "confirmation_results.csv"
)

PSTATIC = Path(
    "outputs/proc_count_causal_v2/controller/"
    "training_support_reconstruction/"
    "v2_controller_training_features_39.csv"
)

PUTIL = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/compressed_action_matrix/"
    "compressed_action_matrix.csv"
)


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def margin(p):
    p = np.asarray(p, dtype=float)
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def expected_numeral(p):
    p = np.asarray(p, dtype=float)
    numerals = np.arange(16, dtype=float)
    return p @ numerals


def safe_spearman(a, b):
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


print("=" * 78)
print("AROMA 2.0 — STAGE 1A-2 GATE 0")
print("=" * 78)

a05 = pd.read_csv(P05)
a15raw = pd.read_csv(P15)
static = pd.read_csv(PSTATIC)
util_long = pd.read_csv(PUTIL)

# ------------------------------------------------------------
# 1. Fix the Stage 1A-1 join issue:
#    alpha1p5 file contains alpha=1 and alpha=1.5 rows.
# ------------------------------------------------------------

a15 = a15raw[
    np.isclose(a15raw["alpha"].astype(float), 1.5)
].copy()

assert len(a05) == 1000, len(a05)
assert len(a15) == 1000, len(a15)
assert a05["sample_id"].is_unique
assert a15["sample_id"].is_unique
assert static["sample_id"].is_unique

ids05 = set(a05["sample_id"])
ids15 = set(a15["sample_id"])
idsS = set(static["sample_id"])

assert ids05 == ids15 == idsS

print("Matched sample IDs:", len(ids05))

# ------------------------------------------------------------
# 2. Join alpha=.5 and alpha=1.5 response states.
# ------------------------------------------------------------

r = a05.merge(
    a15,
    on="sample_id",
    how="inner",
    suffixes=("_05", "_15"),
    validate="one_to_one",
)

assert len(r) == 1000

# Check immutable metadata where available.
for base in ["ground_truth", "condition"]:
    c05 = f"{base}_05"
    c15 = f"{base}_15"

    if c05 in r.columns and c15 in r.columns:
        same = (
            r[c05].astype(str).to_numpy()
            ==
            r[c15].astype(str).to_numpy()
        )

        print(
            f"{base} agreement:",
            f"{same.mean():.6f}"
        )

        assert same.all()


# ------------------------------------------------------------
# 3. Extract p(.5), p(1), p(1.5).
# ------------------------------------------------------------

def cols(prefix, suffix):
    return [
        f"{prefix}_numprob_{n}_{suffix}"
        for n in range(16)
    ]


p05_cols = cols("modulated", "05")
p15_cols = cols("modulated", "15")

p1_from05_cols = cols("baseline", "05")
p1_from15_cols = cols("baseline", "15")

for cc in [
    p05_cols,
    p15_cols,
    p1_from05_cols,
    p1_from15_cols,
]:
    missing = [c for c in cc if c not in r.columns]
    assert not missing, missing


p05 = r[p05_cols].to_numpy(float)
p15 = r[p15_cols].to_numpy(float)

p1a = r[p1_from05_cols].to_numpy(float)
p1b = r[p1_from15_cols].to_numpy(float)

baseline_absdiff = np.abs(p1a - p1b)

print(
    "Baseline probability max abs diff:",
    baseline_absdiff.max()
)

print(
    "Baseline probability mean abs diff:",
    baseline_absdiff.mean()
)

# Require practically identical baseline states.
assert baseline_absdiff.max() <= 1e-6

p1 = 0.5 * (p1a + p1b)

for name, p in [
    ("p05", p05),
    ("p1", p1),
    ("p15", p15),
]:
    sums = p.sum(axis=1)

    print(
        name,
        "probability-sum mean/min/max:",
        sums.mean(),
        sums.min(),
        sums.max(),
    )

    assert np.allclose(
        sums,
        1.0,
        atol=1e-5,
    )


# ------------------------------------------------------------
# 4. Construct PRE-REGISTERED 38-D response signature.
#
#    16 Jp
#    16 Kp
#    J/K expected numeral
#    J/K entropy
#    J/K conditional top1-top2 margin
# ------------------------------------------------------------

Jp = p15 - p05
Kp = (p15 - 2.0 * p1 + p05) / 0.25

mu05 = expected_numeral(p05)
mu1 = expected_numeral(p1)
mu15 = expected_numeral(p15)

H05 = entropy(p05)
H1 = entropy(p1)
H15 = entropy(p15)

M05 = margin(p05)
M1 = margin(p1)
M15 = margin(p15)

Jmu = mu15 - mu05
Kmu = (mu15 - 2.0 * mu1 + mu05) / 0.25

JH = H15 - H05
KH = (H15 - 2.0 * H1 + H05) / 0.25

JM = M15 - M05
KM = (M15 - 2.0 * M1 + M05) / 0.25


response_data = {}

for n in range(16):
    response_data[f"Jp_{n}"] = Jp[:, n]

for n in range(16):
    response_data[f"Kp_{n}"] = Kp[:, n]

response_data["J_expected"] = Jmu
response_data["K_expected"] = Kmu
response_data["J_entropy"] = JH
response_data["K_entropy"] = KH
response_data["J_margin"] = JM
response_data["K_margin"] = KM

R = pd.DataFrame(response_data)

assert R.shape == (1000, 38)

R.insert(
    0,
    "sample_id",
    r["sample_id"].to_numpy(),
)

R.to_csv(
    OUT / "response_signature_38.csv",
    index=False,
)

print("Response signature shape:", R.shape)


# ------------------------------------------------------------
# 5. Prepare Static-39.
# ------------------------------------------------------------

static_feature_cols = [
    c for c in static.columns
    if c != "sample_id"
]

assert len(static_feature_cols) == 39

S = static[
    ["sample_id"] + static_feature_cols
].copy()

data = S.merge(
    R,
    on="sample_id",
    how="inner",
    validate="one_to_one",
)

# Add condition for fixed CV stratification.
condition_map = (
    a05[
        ["sample_id", "condition"]
    ]
    .drop_duplicates()
)

data = data.merge(
    condition_map,
    on="sample_id",
    how="left",
    validate="one_to_one",
)

assert len(data) == 1000


# ------------------------------------------------------------
# 6. Utility matrix: same action utilities already measured in v2.
# ------------------------------------------------------------

u = util_long[
    util_long["alpha"].isin(ACTIONS)
].copy()

assert len(u) == 4000

U = u.pivot(
    index="sample_id",
    columns="alpha",
    values="utility",
)

U = U.reindex(
    data["sample_id"]
)

assert not U.isna().any().any()

for a in ACTIONS:
    vals = sorted(
        pd.Series(U[a]).dropna().unique().tolist()
    )

    print(
        f"utility alpha={a}:",
        vals,
    )


# ------------------------------------------------------------
# 7. Same model class for all representations:
#    StandardScaler -> Ridge(alpha=.01).
#    No hyperparameter tuning.
# ------------------------------------------------------------

response_cols = [
    c for c in R.columns
    if c != "sample_id"
]

families = {
    "Static39": static_feature_cols,
    "Response38": response_cols,
    "Static39_Response38":
        static_feature_cols + response_cols,
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

strata = data["condition"].astype(str).to_numpy()

predictions = {}
summary_rows = []

for family, feature_cols in families.items():

    X = data[feature_cols].to_numpy(float)

    family_scores = np.zeros(
        (len(data), len(ACTIONS)),
        dtype=float,
    )

    for ai, action in enumerate(ACTIONS):

        y = U[action].to_numpy(float)

        oof = np.zeros(len(data), dtype=float)

        for train_idx, test_idx in cv.split(X, strata):

            model = Pipeline([
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=RIDGE_ALPHA
                    ),
                ),
            ])

            model.fit(
                X[train_idx],
                y[train_idx],
            )

            oof[test_idx] = model.predict(
                X[test_idx]
            )

        family_scores[:, ai] = oof

        rho = safe_spearman(
            oof,
            y,
        )

        summary_rows.append({
            "family": family,
            "action": action,
            "n_features": len(feature_cols),
            "spearman": rho,
        })

    predictions[family] = family_scores


summary = pd.DataFrame(summary_rows)

means = (
    summary.groupby("family")
    ["spearman"]
    .mean()
    .rename("mean_action_spearman")
    .reset_index()
)

summary = summary.merge(
    means,
    on="family",
    how="left",
)

summary.to_csv(
    OUT / "per_action_spearman.csv",
    index=False,
)

print()
print("===== PER-ACTION SPEARMAN =====")
print(
    summary.to_string(index=False)
)

print()
print("===== MEAN ACTION SPEARMAN =====")
print(
    means.to_string(index=False)
)


# ------------------------------------------------------------
# 8. Save OOF predictions.
# ------------------------------------------------------------

pred_df = pd.DataFrame({
    "sample_id": data["sample_id"],
    "condition": data["condition"],
})

for ai, action in enumerate(ACTIONS):
    tag = str(action).replace(".", "p")

    pred_df[
        f"utility_alpha_{tag}"
    ] = U[action].to_numpy(float)

    for family in families:
        pred_df[
            f"score_{family}_alpha_{tag}"
        ] = predictions[family][:, ai]

pred_df.to_csv(
    OUT / "oof_predictions.csv",
    index=False,
)


# ------------------------------------------------------------
# 9. Paired sample bootstrap for Gate-0 deltas.
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

def mean_action_rho(
    score_matrix,
    idx,
):
    vals = []

    for ai, action in enumerate(ACTIONS):
        vals.append(
            safe_spearman(
                score_matrix[idx, ai],
                U[action].to_numpy(float)[idx],
            )
        )

    return float(np.mean(vals))


n = len(data)

observed = {}

for family in families:
    idx = np.arange(n)

    observed[family] = mean_action_rho(
        predictions[family],
        idx,
    )

delta_R_S = (
    observed["Response38"]
    -
    observed["Static39"]
)

delta_SR_S = (
    observed["Static39_Response38"]
    -
    observed["Static39"]
)

boot_R_S = np.empty(N_BOOT, dtype=float)
boot_SR_S = np.empty(N_BOOT, dtype=float)

for b in range(N_BOOT):

    idx = rng.integers(
        0,
        n,
        size=n,
    )

    rhoS = mean_action_rho(
        predictions["Static39"],
        idx,
    )

    rhoR = mean_action_rho(
        predictions["Response38"],
        idx,
    )

    rhoSR = mean_action_rho(
        predictions["Static39_Response38"],
        idx,
    )

    boot_R_S[b] = rhoR - rhoS
    boot_SR_S[b] = rhoSR - rhoS


def ci(x):
    return [
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
    ]


ci_R_S = ci(boot_R_S)
ci_SR_S = ci(boot_SR_S)


# ------------------------------------------------------------
# 10. Gate decision.
#
# PRIMARY:
# Static+Response vs Static.
#
# >= .10 AND lower 95% CI > 0 -> GO
# .04 .. .10 -> NO-GO / weak signal
# < .04 -> NO-GO
# ------------------------------------------------------------

if (
    delta_SR_S >= 0.10
    and ci_SR_S[0] > 0
):
    gate = "GO"

elif (
    delta_SR_S >= 0.04
    and delta_SR_S < 0.10
):
    gate = "NO_GO_WEAK_SIGNAL"

else:
    gate = "NO_GO"


# Number of actions for which Response-only
# does not underperform Static.
per_action = (
    summary.pivot(
        index="action",
        columns="family",
        values="spearman",
    )
)

response_noninferior_actions = int(
    (
        per_action["Response38"]
        >=
        per_action["Static39"]
    ).sum()
)


result = {
    "stage": "AROMA2_STAGE1A_GATE0",
    "seed": SEED,
    "n_samples": n,
    "ridge_alpha": RIDGE_ALPHA,
    "cv": "5-fold StratifiedKFold by condition",
    "n_bootstrap": N_BOOT,

    "response_representation": {
        "n_features": 38,
        "definition": (
            "16 Jp + 16 Kp + "
            "J/K expected numeral + "
            "J/K entropy + "
            "J/K top1-top2 margin"
        ),
    },

    "mean_action_spearman": observed,

    "secondary_delta_response_minus_static":
        float(delta_R_S),

    "secondary_delta_response_minus_static_ci95":
        ci_R_S,

    "primary_delta_static_response_minus_static":
        float(delta_SR_S),

    "primary_delta_static_response_minus_static_ci95":
        ci_SR_S,

    "response_noninferior_actions_out_of_4":
        response_noninferior_actions,

    "predefined_gate": {
        "GO": (
            "delta >= 0.10 and "
            "95% CI lower bound > 0"
        ),
        "WEAK_NO_GO": (
            "0.04 <= delta < 0.10"
        ),
        "NO_GO": (
            "delta < 0.04 or otherwise "
            "fails GO criterion"
        ),
    },

    "gate_decision": gate,
}

with (
    OUT / "gate0_result.json"
).open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        indent=2,
    )


print()
print("=" * 78)
print("GATE 0 RESULT")
print("=" * 78)

print(
    "Mean rho Static39:",
    observed["Static39"],
)

print(
    "Mean rho Response38:",
    observed["Response38"],
)

print(
    "Mean rho S+R:",
    observed["Static39_Response38"],
)

print()

print(
    "Secondary delta R-S:",
    delta_R_S,
    "CI95:",
    ci_R_S,
)

print(
    "PRIMARY delta (S+R)-S:",
    delta_SR_S,
    "CI95:",
    ci_SR_S,
)

print(
    "Response >= Static actions:",
    f"{response_noninferior_actions}/4",
)

print()

print(
    "PREDEFINED GATE DECISION:",
    gate,
)

print("=" * 78)
print("CPU-ONLY ANALYSIS COMPLETE")
print("=" * 78)

