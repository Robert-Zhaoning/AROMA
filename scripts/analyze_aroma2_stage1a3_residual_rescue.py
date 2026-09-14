from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, binomtest
from sklearn.model_selection import StratifiedKFold
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
    "outputs/aroma2/stage1a_residual_rescue"
)
OUT.mkdir(parents=True, exist_ok=True)

P_GATE = Path(
    "outputs/aroma2/stage1a_gate0/"
    "oof_predictions.csv"
)

P_RESPONSE = Path(
    "outputs/aroma2/stage1a_gate0/"
    "response_signature_38.csv"
)

P_STATIC = Path(
    "outputs/proc_count_causal_v2/controller/"
    "training_support_reconstruction/"
    "v2_controller_training_features_39.csv"
)

P_ACTIONS = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/compressed_action_matrix/"
    "compressed_action_matrix.csv"
)

P_BASE = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/down_alpha0p5_expanded/"
    "down_alpha0p5_results.csv"
)


def make_ridge():
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def safe_rho(x, y):
    r = spearmanr(x, y).statistic
    if not np.isfinite(r):
        return 0.0
    return float(r)


def action_tag(a):
    return str(float(a)).replace(".", "p")


def mean_action_rho(scores, utilities):
    vals = [
        safe_rho(
            scores[:, i],
            utilities[:, i],
        )
        for i in range(len(ACTIONS))
    ]
    return float(np.mean(vals)), vals


def bootstrap_mean_ci(
    values,
    n_boot,
    seed,
):
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)

    out = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )
        out[b] = values[idx].mean()

    return [
        float(np.quantile(out, 0.025)),
        float(np.quantile(out, 0.975)),
    ]


def bootstrap_rho_delta(
    scores_a,
    scores_b,
    utilities,
    n_boot,
    seed,
):
    rng = np.random.default_rng(seed)
    n = len(utilities)

    obs_a, _ = mean_action_rho(
        scores_a,
        utilities,
    )
    obs_b, _ = mean_action_rho(
        scores_b,
        utilities,
    )

    obs = obs_a - obs_b

    vals = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        ra, _ = mean_action_rho(
            scores_a[idx],
            utilities[idx],
        )

        rb, _ = mean_action_rho(
            scores_b[idx],
            utilities[idx],
        )

        vals[b] = ra - rb

    return (
        float(obs),
        [
            float(np.quantile(vals, 0.025)),
            float(np.quantile(vals, 0.975)),
        ],
    )


def evaluate_policy(
    name,
    scores,
    utilities,
    baseline_correct,
):
    n = len(scores)

    best_idx = np.argmax(
        scores,
        axis=1,
    )

    best_score = scores[
        np.arange(n),
        best_idx,
    ]

    chosen = np.full(
        n,
        1.0,
        dtype=float,
    )

    intervene = (
        best_score
        >
        POLICY_THRESHOLD
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
            utilities[
                mask,
                ai,
            ]
        )

    post_correct = (
        baseline_correct.astype(float)
        +
        selected_utility
    )

    if not np.all(
        np.isin(
            post_correct,
            [0.0, 1.0],
        )
    ):
        raise RuntimeError(
            f"Invalid post-correctness for {name}"
        )

    repairs = int(
        np.sum(selected_utility == 1)
    )

    breaks = int(
        np.sum(selected_utility == -1)
    )

    discordant = repairs + breaks

    if discordant:
        mcnemar_p = float(
            binomtest(
                repairs,
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        mcnemar_p = 1.0

    oracle_repairable = (
        utilities.max(axis=1)
        >
        0
    )

    n_oracle_repairable = int(
        oracle_repairable.sum()
    )

    capture = (
        repairs
        /
        n_oracle_repairable
        if n_oracle_repairable
        else np.nan
    )

    action_distribution = {
        str(a): int(
            np.sum(
                np.isclose(
                    chosen,
                    a,
                )
            )
        )
        for a in [0.0, 1.0, 1.5, 2.0, 4.0]
    }

    gain = float(
        selected_utility.mean()
    )

    gain_ci = bootstrap_mean_ci(
        selected_utility,
        N_BOOT_POLICY,
        SEED + 17,
    )

    return {
        "name": name,
        "baseline_accuracy":
            float(
                baseline_correct.mean()
            ),
        "post_accuracy":
            float(
                post_correct.mean()
            ),
        "gain":
            gain,
        "gain_pp":
            100.0 * gain,
        "gain_ci95_pp": [
            100.0 * gain_ci[0],
            100.0 * gain_ci[1],
        ],
        "repairs": repairs,
        "breaks": breaks,
        "net_repairs_minus_breaks":
            repairs - breaks,
        "interventions":
            int(intervene.sum()),
        "intervention_rate":
            float(intervene.mean()),
        "oracle_repairable":
            n_oracle_repairable,
        "oracle_capture":
            float(capture),
        "mcnemar_exact_p":
            mcnemar_p,
        "action_distribution":
            action_distribution,
        "chosen_action":
            chosen,
        "selected_utility":
            selected_utility,
        "best_score":
            best_score,
    }


print("=" * 80)
print("AROMA 2.0 — STAGE 1A-3")
print("RESIDUAL-RESPONSE RESCUE SCREEN")
print("=" * 80)

for p in [
    P_GATE,
    P_RESPONSE,
    P_STATIC,
    P_ACTIONS,
    P_BASE,
]:
    if not p.exists():
        raise FileNotFoundError(p)

gate = pd.read_csv(P_GATE)
response = pd.read_csv(P_RESPONSE)
static = pd.read_csv(P_STATIC)
action_long = pd.read_csv(P_ACTIONS)
base = pd.read_csv(P_BASE)

# ------------------------------------------------------------------
# Build one aligned 1000-sample table.
# ------------------------------------------------------------------

meta = (
    gate[
        [
            "sample_id",
            "condition",
        ]
    ]
    .drop_duplicates()
)

base_small = (
    base[
        [
            "sample_id",
            "baseline_correct",
        ]
    ]
    .drop_duplicates()
)

data = (
    static
    .merge(
        response,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        meta,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        base_small,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
)

assert len(data) == 1000

gate_idx = (
    gate
    .set_index("sample_id")
    .loc[data["sample_id"]]
)

# ------------------------------------------------------------------
# Utility matrix.
# ------------------------------------------------------------------

u = action_long.copy()
u["alpha_key"] = (
    u["alpha"]
    .astype(float)
    .round(1)
)

u = u[
    u["alpha_key"].isin(ACTIONS)
]

U_df = u.pivot(
    index="sample_id",
    columns="alpha_key",
    values="utility",
)

U_df = U_df.loc[
    data["sample_id"]
]

U = np.column_stack([
    U_df[a].to_numpy(float)
    for a in ACTIONS
])

assert U.shape == (1000, 4)

baseline_correct = (
    data["baseline_correct"]
    .to_numpy(int)
)

print(
    "Baseline accuracy:",
    baseline_correct.mean(),
)

# ------------------------------------------------------------------
# Static / response feature groups.
# ------------------------------------------------------------------

static_cols = [
    c
    for c in static.columns
    if c != "sample_id"
]

response_cols = [
    c
    for c in response.columns
    if c != "sample_id"
]

J_cols = [
    c for c in response_cols
    if c.startswith("J")
]

K_cols = [
    c for c in response_cols
    if c.startswith("K")
]

assert len(static_cols) == 39
assert len(response_cols) == 38
assert len(J_cols) == 19
assert len(K_cols) == 19

print("Static features :", len(static_cols))
print("Response features:", len(response_cols))
print("J features      :", len(J_cols))
print("K features      :", len(K_cols))

XS = data[
    static_cols
].to_numpy(float)

XR_J = data[
    J_cols
].to_numpy(float)

XR_K = data[
    K_cols
].to_numpy(float)

XR_JK = data[
    response_cols
].to_numpy(float)

strata = (
    data["condition"]
    .astype(str)
    .to_numpy()
)

# ------------------------------------------------------------------
# Existing Gate-0 predictions.
# ------------------------------------------------------------------

existing_scores = {}

for family in [
    "Static39",
    "Response38",
    "Static39_Response38",
]:
    cols = [
        (
            f"score_{family}_alpha_"
            f"{action_tag(a)}"
        )
        for a in ACTIONS
    ]

    missing = [
        c for c in cols
        if c not in gate_idx.columns
    ]

    if missing:
        raise KeyError(
            f"{family}: missing {missing}"
        )

    existing_scores[family] = (
        gate_idx[cols]
        .to_numpy(float)
    )

# ------------------------------------------------------------------
# Proper nested cross-fitted residual-response models.
#
# Outer fold:
#   evaluate on held-out samples.
#
# Inner folds inside outer training set:
#   generate OOF static predictions for residual targets.
#
# Then:
#   response -> (true utility - static OOF utility)
#
# Final outer prediction:
#   outer static prediction + response residual prediction.
# ------------------------------------------------------------------

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

outer_splits = list(
    outer_cv.split(
        XS,
        strata,
    )
)

residual_feature_sets = {
    "Residual_J": XR_J,
    "Residual_K": XR_K,
    "Residual_JK": XR_JK,
}

residual_scores = {
    name: np.zeros(
        (len(data), len(ACTIONS)),
        dtype=float,
    )
    for name in residual_feature_sets
}

reconstructed_static = np.zeros(
    (len(data), len(ACTIONS)),
    dtype=float,
)

for ai, action in enumerate(ACTIONS):

    y = U[:, ai]

    for outer_fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer_splits,
        start=1,
    ):

        # Static predictor for actual outer test fold.
        static_outer = make_ridge()

        static_outer.fit(
            XS[train_idx],
            y[train_idx],
        )

        static_test_pred = (
            static_outer.predict(
                XS[test_idx]
            )
        )

        reconstructed_static[
            test_idx,
            ai,
        ] = static_test_pred

        # Inner cross-fitting creates leakage-free
        # residual targets for outer-train rows.
        X_train_static = XS[
            train_idx
        ]

        y_train = y[
            train_idx
        ]

        strata_train = strata[
            train_idx
        ]

        inner_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=(
                SEED
                + 100
                + outer_fold
            ),
        )

        static_train_oof = np.zeros(
            len(train_idx),
            dtype=float,
        )

        for inner_train_rel, inner_val_rel in (
            inner_cv.split(
                X_train_static,
                strata_train,
            )
        ):
            m_inner = make_ridge()

            m_inner.fit(
                X_train_static[
                    inner_train_rel
                ],
                y_train[
                    inner_train_rel
                ],
            )

            static_train_oof[
                inner_val_rel
            ] = m_inner.predict(
                X_train_static[
                    inner_val_rel
                ]
            )

        residual_target = (
            y_train
            -
            static_train_oof
        )

        # Fit J / K / J+K response corrections.
        for name, XR in (
            residual_feature_sets.items()
        ):

            residual_model = make_ridge()

            residual_model.fit(
                XR[train_idx],
                residual_target,
            )

            residual_test = (
                residual_model.predict(
                    XR[test_idx]
                )
            )

            residual_scores[
                name
            ][
                test_idx,
                ai,
            ] = (
                static_test_pred
                +
                residual_test
            )

# ------------------------------------------------------------------
# Sanity check:
# reconstructed static predictions should reproduce Gate-0 Static39.
# ------------------------------------------------------------------

max_static_diff = float(
    np.max(
        np.abs(
            reconstructed_static
            -
            existing_scores[
                "Static39"
            ]
        )
    )
)

print()
print(
    "Reconstructed Static39 "
    "max abs difference:",
    max_static_diff,
)

if max_static_diff > 1e-8:
    raise RuntimeError(
        "Outer Static39 predictions do not "
        "reproduce Gate-0 Static39."
    )

# ------------------------------------------------------------------
# Assemble all score families.
# ------------------------------------------------------------------

all_scores = {
    "Static39":
        existing_scores["Static39"],

    "Response38":
        existing_scores["Response38"],

    "NaiveConcat":
        existing_scores[
            "Static39_Response38"
        ],

    **residual_scores,
}

# ------------------------------------------------------------------
# Action-wise and mean Spearman.
# ------------------------------------------------------------------

rho_rows = []
mean_rhos = {}

for name, scores in all_scores.items():

    mean_rho, per_action = (
        mean_action_rho(
            scores,
            U,
        )
    )

    mean_rhos[name] = mean_rho

    for ai, action in enumerate(ACTIONS):
        rho_rows.append({
            "family": name,
            "action": action,
            "spearman":
                per_action[ai],
            "mean_action_spearman":
                mean_rho,
        })

rho_df = pd.DataFrame(
    rho_rows
)

rho_df.to_csv(
    OUT / "per_action_spearman.csv",
    index=False,
)

# ------------------------------------------------------------------
# Fixed-threshold policy translation.
# ------------------------------------------------------------------

policy_results = {}

for name, scores in all_scores.items():
    policy_results[name] = (
        evaluate_policy(
            name,
            scores,
            U,
            baseline_correct,
        )
    )

policy_rows = []

for name, r in policy_results.items():
    policy_rows.append({
        "family": name,
        "mean_action_spearman":
            mean_rhos[name],
        "baseline_accuracy":
            r["baseline_accuracy"],
        "post_accuracy":
            r["post_accuracy"],
        "gain_pp":
            r["gain_pp"],
        "gain_ci95_low_pp":
            r["gain_ci95_pp"][0],
        "gain_ci95_high_pp":
            r["gain_ci95_pp"][1],
        "repairs":
            r["repairs"],
        "breaks":
            r["breaks"],
        "net":
            r[
                "net_repairs_minus_breaks"
            ],
        "interventions":
            r["interventions"],
        "intervention_rate":
            r["intervention_rate"],
        "oracle_capture":
            r["oracle_capture"],
        "mcnemar_exact_p":
            r["mcnemar_exact_p"],
    })

policy_df = pd.DataFrame(
    policy_rows
)

policy_df.to_csv(
    OUT / "policy_summary.csv",
    index=False,
)

# ------------------------------------------------------------------
# Save sample-level policy decisions.
# ------------------------------------------------------------------

sample_out = pd.DataFrame({
    "sample_id":
        data["sample_id"],
    "condition":
        data["condition"],
    "baseline_correct":
        baseline_correct,
})

for name, r in policy_results.items():

    sample_out[
        f"{name}__selected_alpha"
    ] = r["chosen_action"]

    sample_out[
        f"{name}__selected_utility"
    ] = r["selected_utility"]

    sample_out[
        f"{name}__best_score"
    ] = r["best_score"]

sample_out.to_csv(
    OUT / "policy_per_sample.csv",
    index=False,
)

# ------------------------------------------------------------------
# PRIMARY rescue comparison:
# Residual_JK vs Static39.
# ------------------------------------------------------------------

static_policy = (
    policy_results["Static39"]
)

rescue_policy = (
    policy_results["Residual_JK"]
)

policy_delta_sample = (
    rescue_policy[
        "selected_utility"
    ]
    -
    static_policy[
        "selected_utility"
    ]
)

policy_delta_pp = float(
    100.0
    *
    policy_delta_sample.mean()
)

policy_delta_ci_raw = (
    bootstrap_mean_ci(
        policy_delta_sample,
        N_BOOT_POLICY,
        SEED + 301,
    )
)

policy_delta_ci_pp = [
    100.0 * x
    for x in policy_delta_ci_raw
]

rho_delta, rho_delta_ci = (
    bootstrap_rho_delta(
        all_scores[
            "Residual_JK"
        ],
        all_scores[
            "Static39"
        ],
        U,
        N_BOOT_RHO,
        SEED + 401,
    )
)

# Action-wise non-inferiority.
static_action_rho = (
    rho_df[
        rho_df["family"]
        ==
        "Static39"
    ]
    .set_index("action")[
        "spearman"
    ]
)

rescue_action_rho = (
    rho_df[
        rho_df["family"]
        ==
        "Residual_JK"
    ]
    .set_index("action")[
        "spearman"
    ]
)

noninferior = int(
    (
        rescue_action_rho
        >=
        static_action_rho
    ).sum()
)

# ------------------------------------------------------------------
# Pre-specified rescue decision.
# ------------------------------------------------------------------

criterion_policy_point = (
    policy_delta_pp >= 1.0
)

criterion_policy_ci = (
    policy_delta_ci_pp[0] > 0
)

criterion_rho_point = (
    rho_delta >= 0.05
)

criterion_rho_ci = (
    rho_delta_ci[0] > 0
)

criterion_actions = (
    noninferior >= 3
)

all_go = all([
    criterion_policy_point,
    criterion_policy_ci,
    criterion_rho_point,
    criterion_rho_ci,
    criterion_actions,
])

if all_go:
    decision = "RESCUE_GO"

elif (
    policy_delta_pp > 0
    and
    rho_delta > 0
):
    decision = (
        "RESCUE_WEAK_NO_GO"
    )

else:
    decision = "RESCUE_NO_GO"

result = {
    "stage":
        "AROMA2_STAGE1A3_RESIDUAL_RESCUE",

    "status":
        "exploratory_followup_after_gate0",

    "gate0_original_decision":
        "NO_GO_WEAK_SIGNAL",

    "seed": SEED,
    "ridge_alpha": RIDGE_ALPHA,
    "policy_threshold":
        POLICY_THRESHOLD,

    "n_samples":
        int(len(data)),

    "baseline_accuracy":
        float(
            baseline_correct.mean()
        ),

    "mean_action_spearman":
        {
            k: float(v)
            for k, v
            in mean_rhos.items()
        },

    "policy_gain_pp":
        {
            k: float(
                policy_results[k][
                    "gain_pp"
                ]
            )
            for k
            in policy_results
        },

    "primary_residual_jk_vs_static": {
        "policy_gain_delta_pp":
            policy_delta_pp,

        "policy_gain_delta_ci95_pp":
            policy_delta_ci_pp,

        "mean_spearman_delta":
            float(rho_delta),

        "mean_spearman_delta_ci95":
            rho_delta_ci,

        "actionwise_noninferior_count":
            noninferior,

        "criteria": {
            "policy_delta_ge_1pp":
                criterion_policy_point,

            "policy_delta_ci_lower_gt_0":
                criterion_policy_ci,

            "rho_delta_ge_0p05":
                criterion_rho_point,

            "rho_delta_ci_lower_gt_0":
                criterion_rho_ci,

            "noninferior_actions_ge_3":
                criterion_actions,
        },
    },

    "rescue_decision":
        decision,
}

with (
    OUT / "rescue_result.json"
).open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        indent=2,
    )

# ------------------------------------------------------------------
# Console summary.
# ------------------------------------------------------------------

print()
print("=" * 80)
print("SPEARMAN SUMMARY")
print("=" * 80)

print(
    policy_df[
        [
            "family",
            "mean_action_spearman",
        ]
    ]
    .to_string(
        index=False
    )
)

print()
print("=" * 80)
print("FIXED-THRESHOLD POLICY SUMMARY")
print("=" * 80)

print(
    policy_df.to_string(
        index=False
    )
)

print()
print("=" * 80)
print("PRIMARY RESIDUAL-JK RESCUE TEST")
print("=" * 80)

print(
    "Static39 gain:",
    f"{static_policy['gain_pp']:+.3f} pp",
)

print(
    "Residual_JK gain:",
    f"{rescue_policy['gain_pp']:+.3f} pp",
)

print(
    "Policy delta:",
    f"{policy_delta_pp:+.3f} pp",
)

print(
    "Policy delta CI95:",
    policy_delta_ci_pp,
)

print()

print(
    "Static39 mean rho:",
    mean_rhos["Static39"],
)

print(
    "Residual_JK mean rho:",
    mean_rhos["Residual_JK"],
)

print(
    "Rho delta:",
    rho_delta,
)

print(
    "Rho delta CI95:",
    rho_delta_ci,
)

print(
    "Action-wise noninferior:",
    f"{noninferior}/4",
)

print()

print("Criteria:")
for k, v in (
    result[
        "primary_residual_jk_vs_static"
    ]["criteria"].items()
):
    print(
        f"  {k}: {v}"
    )

print()
print(
    "FINAL EXPLORATORY RESCUE DECISION:",
    decision,
)

print("=" * 80)
print("CPU-ONLY ANALYSIS COMPLETE")
print("=" * 80)

