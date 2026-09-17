from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


SEED = 20260914
RIDGE_ALPHA = 0.01
POLICY_THRESHOLD = 0.1

N_BOOT_POLICY = 20000
N_BOOT_RHO = 5000

ACTIONS = [0.0, 1.5, 2.0, 4.0]

OUT = Path(
    "outputs/aroma2/holocount_discrete_response"
)
OUT.mkdir(parents=True, exist_ok=True)

P_WRONG = Path(
    "outputs/generalization_extension/holocount/"
    "external_v1/oracle_gate_v1/"
    "wrong_gtle15_action_results.csv"
)

P_CORRECT = Path(
    "outputs/generalization_extension/holocount/"
    "external_v1/action_matrix_v1/"
    "baseline_correct_action_results.csv"
)

P_FULL = Path(
    "outputs/generalization_extension/holocount/"
    "external_v1/full_raw_results.csv"
)


def make_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def safe_rho(x, y):
    r = spearmanr(x, y).statistic
    if not np.isfinite(r):
        return 0.0
    return float(r)


def mean_action_rho(scores, U, idx=None):
    if idx is None:
        idx = np.arange(len(U))

    vals = []

    for ai in range(len(ACTIONS)):
        vals.append(
            safe_rho(
                scores[idx, ai],
                U[idx, ai],
            )
        )

    return float(np.mean(vals)), vals


def evaluate_policy(scores, U, baseline_correct):
    n = len(scores)

    best_idx = np.argmax(
        scores,
        axis=1,
    )

    best_score = scores[
        np.arange(n),
        best_idx,
    ]

    intervene = (
        best_score
        >
        POLICY_THRESHOLD
    )

    chosen = np.ones(
        n,
        dtype=float,
    )

    action_array = np.asarray(
        ACTIONS,
        dtype=float,
    )

    chosen[intervene] = (
        action_array[
            best_idx[intervene]
        ]
    )

    selected_utility = np.zeros(
        n,
        dtype=float,
    )

    for ai, action in enumerate(ACTIONS):
        mask = np.isclose(
            chosen,
            action,
        )

        selected_utility[mask] = (
            U[mask, ai]
        )

    post_correct = (
        baseline_correct
        +
        selected_utility
    )

    assert np.all(
        np.isin(
            post_correct,
            [0, 1],
        )
    )

    repairs = int(
        np.sum(
            selected_utility == 1
        )
    )

    breaks = int(
        np.sum(
            selected_utility == -1
        )
    )

    return {
        "chosen": chosen,
        "selected_utility":
            selected_utility,
        "best_score": best_score,
        "post_correct":
            post_correct,
        "repairs": repairs,
        "breaks": breaks,
        "net": repairs - breaks,
        "interventions":
            int(intervene.sum()),
        "intervention_rate":
            float(intervene.mean()),
        "gain":
            float(
                selected_utility.mean()
            ),
        "gain_pp":
            float(
                100
                *
                selected_utility.mean()
            ),
    }


def cluster_bootstrap_indices(
    groups,
    rng,
):
    groups = np.asarray(groups)
    unique = np.unique(groups)

    by_group = {
        g: np.where(groups == g)[0]
        for g in unique
    }

    sampled = rng.choice(
        unique,
        size=len(unique),
        replace=True,
    )

    return np.concatenate([
        by_group[g]
        for g in sampled
    ])


print("=" * 80)
print("AROMA 2.0 — HOLOCOUNT DISCRETE-RESPONSE SCREEN")
print("=" * 80)

wrong = pd.read_csv(P_WRONG)
correct = pd.read_csv(P_CORRECT)
full = pd.read_csv(P_FULL)

response_cols_source = [
    "manifest_index",
    "sample_id",
    "split",
    "ground_truth",
    "baseline_prediction",
    "pred_alpha_0",
    "pred_alpha_1p5",
    "pred_alpha_2",
    "pred_alpha_4",
]

wrong = wrong[
    response_cols_source
].copy()

correct = correct[
    response_cols_source
].copy()

actions = pd.concat(
    [wrong, correct],
    axis=0,
    ignore_index=True,
)

assert len(actions) == 2292
assert actions["sample_id"].is_unique

print(
    "Supported action-response rows:",
    len(actions),
)

# ------------------------------------------------------------
# Merge Static-39 + image groups.
# ------------------------------------------------------------

feature_cols = [
    c for c in full.columns
    if c.startswith("feature__")
]

assert len(feature_cols) == 39, (
    len(feature_cols),
    feature_cols,
)

full_small = full[
    [
        "sample_id",
        "image_sha256",
        "ground_truth",
        "ground_truth_le_15",
        "baseline_prediction",
    ]
    +
    feature_cols
].copy()

data = actions.merge(
    full_small,
    on="sample_id",
    how="inner",
    suffixes=("_action", "_full"),
    validate="one_to_one",
)

assert len(data) == 2292

assert data[
    "ground_truth_le_15"
].all()

assert np.array_equal(
    data[
        "ground_truth_action"
    ].to_numpy(),
    data[
        "ground_truth_full"
    ].to_numpy(),
)

assert np.array_equal(
    data[
        "baseline_prediction_action"
    ].to_numpy(),
    data[
        "baseline_prediction_full"
    ].to_numpy(),
)

gt = data[
    "ground_truth_action"
].to_numpy(int)

base_pred = data[
    "baseline_prediction_action"
].to_numpy(int)

baseline_correct = (
    base_pred == gt
).astype(int)

print(
    "Baseline accuracy supported:",
    baseline_correct.mean(),
)

print(
    "Unique image groups:",
    data["image_sha256"].nunique(),
)

print(
    "HoloCount subsets:",
    data["split"].nunique(),
)

# ------------------------------------------------------------
# Fixed seven-dimensional GT-free discrete response signature.
# ------------------------------------------------------------

pred0 = data[
    "pred_alpha_0"
].to_numpy(float)

pred15 = data[
    "pred_alpha_1p5"
].to_numpy(float)

d_down = pred0 - base_pred
d_up = pred15 - base_pred

response = pd.DataFrame({
    "sample_id":
        data["sample_id"],

    "resp_d_down":
        d_down,

    "resp_d_up":
        d_up,

    "resp_abs_down":
        np.abs(d_down),

    "resp_abs_up":
        np.abs(d_up),

    "resp_span":
        d_up - d_down,

    "resp_center_shift":
        0.5 * (
            pred0
            +
            pred15
        )
        -
        base_pred,

    "resp_directional_consistency":
        (
            (d_down <= 0)
            &
            (d_up >= 0)
        ).astype(float),
})

response_feature_cols = [
    c for c in response.columns
    if c != "sample_id"
]

assert len(response_feature_cols) == 7

response.to_csv(
    OUT / "discrete_response7.csv",
    index=False,
)

# ------------------------------------------------------------
# Exact action utilities from already-run fixed interventions.
# ------------------------------------------------------------

pred_cols = {
    0.0: "pred_alpha_0",
    1.5: "pred_alpha_1p5",
    2.0: "pred_alpha_2",
    4.0: "pred_alpha_4",
}

U = np.zeros(
    (
        len(data),
        len(ACTIONS),
    ),
    dtype=float,
)

for ai, action in enumerate(ACTIONS):
    action_correct = (
        data[
            pred_cols[action]
        ].to_numpy(int)
        ==
        gt
    ).astype(int)

    U[:, ai] = (
        action_correct
        -
        baseline_correct
    )

    print(
        f"alpha={action}:",
        {
            int(v):
            int(np.sum(U[:, ai] == v))
            for v
            in [-1, 0, 1]
        },
    )

# ------------------------------------------------------------
# Feature families.
# ------------------------------------------------------------

XS = data[
    feature_cols
].to_numpy(float)

XR = response[
    response_feature_cols
].to_numpy(float)

XSR = np.column_stack([
    XS,
    XR,
])

families = {
    "Static39": XS,
    "DiscreteResponse7": XR,
    "Static39_Response7": XSR,
}

# ------------------------------------------------------------
# 5-fold image-disjoint SGKF by official HoloCount subset.
# ------------------------------------------------------------

strata = data[
    "split"
].astype(str).to_numpy()

groups = data[
    "image_sha256"
].astype(str).to_numpy()

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

splits = list(
    cv.split(
        XS,
        strata,
        groups,
    )
)

# Hard leakage audit.
for fold, (tr, te) in enumerate(
    splits,
    start=1,
):
    overlap = (
        set(groups[tr])
        &
        set(groups[te])
    )

    assert not overlap

    print(
        f"fold {fold}: "
        f"train={len(tr)} "
        f"test={len(te)} "
        f"train_groups={len(set(groups[tr]))} "
        f"test_groups={len(set(groups[te]))}"
    )

# ------------------------------------------------------------
# OOF action utility prediction.
# ------------------------------------------------------------

scores = {
    name:
        np.zeros(
            (
                len(data),
                len(ACTIONS),
            ),
            dtype=float,
        )
    for name
    in families
}

for name, X in families.items():

    for ai, action in enumerate(ACTIONS):

        y = U[:, ai]

        oof = np.zeros(
            len(data),
            dtype=float,
        )

        for tr, te in splits:
            model = make_model()

            model.fit(
                X[tr],
                y[tr],
            )

            oof[te] = (
                model.predict(
                    X[te]
                )
            )

        scores[name][:, ai] = oof

# ------------------------------------------------------------
# Spearman.
# ------------------------------------------------------------

rho_rows = []
mean_rho = {}
action_rho = {}

for name in families:

    m, per_action = (
        mean_action_rho(
            scores[name],
            U,
        )
    )

    mean_rho[name] = m
    action_rho[name] = per_action

    for ai, action in enumerate(ACTIONS):
        rho_rows.append({
            "family": name,
            "action": action,
            "spearman":
                per_action[ai],
            "mean_action_spearman":
                m,
        })

rho_df = pd.DataFrame(
    rho_rows
)

rho_df.to_csv(
    OUT / "per_action_spearman.csv",
    index=False,
)

# ------------------------------------------------------------
# Fixed-threshold policy.
# ------------------------------------------------------------

policy = {
    name:
        evaluate_policy(
            scores[name],
            U,
            baseline_correct,
        )
    for name
    in families
}

policy_rows = []

oracle_repairable = (
    U.max(axis=1) > 0
)

oracle_count = int(
    oracle_repairable.sum()
)

for name in families:
    p = policy[name]

    policy_rows.append({
        "family": name,
        "mean_action_spearman":
            mean_rho[name],
        "baseline_accuracy":
            baseline_correct.mean(),
        "post_accuracy":
            p["post_correct"].mean(),
        "gain_pp":
            p["gain_pp"],
        "repairs":
            p["repairs"],
        "breaks":
            p["breaks"],
        "net":
            p["net"],
        "interventions":
            p["interventions"],
        "intervention_rate":
            p["intervention_rate"],
        "oracle_repairable":
            oracle_count,
        "oracle_capture":
            (
                p["repairs"]
                /
                oracle_count
            ),
    })

policy_df = pd.DataFrame(
    policy_rows
)

policy_df.to_csv(
    OUT / "policy_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# Primary paired image-cluster bootstrap:
# S+R vs Static.
# ------------------------------------------------------------

static_name = "Static39"
combo_name = "Static39_Response7"

utility_delta_sample = (
    policy[combo_name][
        "selected_utility"
    ]
    -
    policy[static_name][
        "selected_utility"
    ]
)

observed_policy_delta_pp = float(
    100
    *
    utility_delta_sample.mean()
)

observed_rho_delta = (
    mean_rho[combo_name]
    -
    mean_rho[static_name]
)

rng = np.random.default_rng(
    SEED + 901
)

boot_policy = np.empty(
    N_BOOT_POLICY,
    dtype=float,
)

for b in range(N_BOOT_POLICY):
    idx = cluster_bootstrap_indices(
        groups,
        rng,
    )

    boot_policy[b] = (
        100
        *
        utility_delta_sample[
            idx
        ].mean()
    )

policy_ci = [
    float(
        np.quantile(
            boot_policy,
            0.025,
        )
    ),
    float(
        np.quantile(
            boot_policy,
            0.975,
        )
    ),
]

rng_rho = np.random.default_rng(
    SEED + 902
)

boot_rho = np.empty(
    N_BOOT_RHO,
    dtype=float,
)

for b in range(N_BOOT_RHO):
    idx = cluster_bootstrap_indices(
        groups,
        rng_rho,
    )

    rho_combo, _ = (
        mean_action_rho(
            scores[combo_name],
            U,
            idx,
        )
    )

    rho_static, _ = (
        mean_action_rho(
            scores[static_name],
            U,
            idx,
        )
    )

    boot_rho[b] = (
        rho_combo
        -
        rho_static
    )

rho_ci = [
    float(
        np.quantile(
            boot_rho,
            0.025,
        )
    ),
    float(
        np.quantile(
            boot_rho,
            0.975,
        )
    ),
]

noninferior_actions = int(
    np.sum(
        np.asarray(
            action_rho[combo_name]
        )
        >=
        np.asarray(
            action_rho[static_name]
        )
    )
)

criteria = {
    "policy_delta_ge_1pp":
        observed_policy_delta_pp
        >= 1.0,

    "cluster_policy_ci_lower_gt_0":
        policy_ci[0] > 0,

    "rho_delta_ge_0p05":
        observed_rho_delta
        >= 0.05,

    "cluster_rho_ci_lower_gt_0":
        rho_ci[0] > 0,

    "noninferior_actions_ge_3":
        noninferior_actions >= 3,
}

if all(criteria.values()):
    decision = (
        "HOLO_RESPONSE_RESCUE_GO"
    )

elif (
    observed_policy_delta_pp > 0
    and
    observed_rho_delta > 0
):
    decision = (
        "HOLO_RESPONSE_RESCUE_WEAK_NO_GO"
    )

else:
    decision = (
        "HOLO_RESPONSE_RESCUE_NO_GO"
    )

# ------------------------------------------------------------
# Save sample-level predictions.
# ------------------------------------------------------------

sample_out = pd.DataFrame({
    "sample_id":
        data["sample_id"],
    "split":
        data["split"],
    "image_sha256":
        groups,
    "ground_truth":
        gt,
    "baseline_prediction":
        base_pred,
    "baseline_correct":
        baseline_correct,
})

for ai, action in enumerate(ACTIONS):
    tag = str(action).replace(
        ".",
        "p",
    )

    sample_out[
        f"utility_alpha_{tag}"
    ] = U[:, ai]

    for name in families:
        sample_out[
            f"score_{name}_alpha_{tag}"
        ] = scores[name][:, ai]

for name in families:
    sample_out[
        f"{name}__selected_alpha"
    ] = policy[name]["chosen"]

    sample_out[
        f"{name}__selected_utility"
    ] = policy[name][
        "selected_utility"
    ]

sample_out.to_csv(
    OUT / "oof_predictions.csv",
    index=False,
)

result = {
    "stage":
        "AROMA2_HOLOCOUNT_DISCRETE_RESPONSE",

    "status":
        "retrospective_exploratory",

    "population":
        "HoloCount GT<=15 supported population",

    "n_samples":
        int(len(data)),

    "n_unique_images":
        int(
            data[
                "image_sha256"
            ].nunique()
        ),

    "n_subsets":
        int(
            data[
                "split"
            ].nunique()
        ),

    "response_features": 7,

    "ridge_alpha":
        RIDGE_ALPHA,

    "policy_threshold":
        POLICY_THRESHOLD,

    "mean_action_spearman":
        {
            k:
                float(v)
            for k, v
            in mean_rho.items()
        },

    "policy_gain_pp":
        {
            k:
                float(
                    policy[k][
                        "gain_pp"
                    ]
                )
            for k
            in families
        },

    "primary_combo_vs_static": {
        "policy_delta_pp":
            observed_policy_delta_pp,

        "policy_delta_cluster_ci95_pp":
            policy_ci,

        "mean_spearman_delta":
            float(
                observed_rho_delta
            ),

        "mean_spearman_delta_cluster_ci95":
            rho_ci,

        "actionwise_noninferior":
            noninferior_actions,

        "criteria":
            criteria,
    },

    "decision":
        decision,
}

with (
    OUT / "result.json"
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
print("=" * 80)
print("PER-ACTION SPEARMAN")
print("=" * 80)
print(
    rho_df.to_string(
        index=False
    )
)

print()
print("=" * 80)
print("POLICY SUMMARY")
print("=" * 80)
print(
    policy_df.to_string(
        index=False
    )
)

print()
print("=" * 80)
print("PRIMARY HOLOCOUNT RESPONSE RESCUE TEST")
print("=" * 80)

print(
    "Static mean rho:",
    mean_rho["Static39"],
)

print(
    "Response mean rho:",
    mean_rho[
        "DiscreteResponse7"
    ],
)

print(
    "Static+Response mean rho:",
    mean_rho[
        "Static39_Response7"
    ],
)

print()

print(
    "Static gain:",
    f"{policy['Static39']['gain_pp']:+.4f} pp",
)

print(
    "Response gain:",
    f"{policy['DiscreteResponse7']['gain_pp']:+.4f} pp",
)

print(
    "Static+Response gain:",
    f"{policy['Static39_Response7']['gain_pp']:+.4f} pp",
)

print()

print(
    "PRIMARY policy delta:",
    f"{observed_policy_delta_pp:+.4f} pp",
)

print(
    "Cluster CI95:",
    policy_ci,
)

print(
    "PRIMARY rho delta:",
    observed_rho_delta,
)

print(
    "Cluster rho CI95:",
    rho_ci,
)

print(
    "Actionwise noninferior:",
    f"{noninferior_actions}/4",
)

print()
print("Criteria:")

for k, v in criteria.items():
    print(
        f"  {k}: {v}"
    )

print()
print(
    "FINAL DECISION:",
    decision,
)

print("=" * 80)

