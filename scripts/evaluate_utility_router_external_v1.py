import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# FROZEN INPUTS
# ============================================================

ROUTER = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_final_v1/"
    "aroma_utility_router_v1.joblib"
)

EXPECTED_ROUTER_SHA = (
    "4192129c525822fed7ab9d013dc3d564981f48faf81"
    "f28b3011a4be1a2168563"
)

FEATURES = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "utility_features_v1/"
    "tsg_utility_features_v1.csv"
)

EXPECTED_FEATURE_SHA = (
    "007dca7e45fdeecde4b88e950eef9ca3a212d223410"
    "c8e1e9d930dea93924674"
)

SWEEP = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "mn_sa_tsg_sweep_v1/"
    "mn_sa_tsg_results.csv"
)

OUT = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "utility_router_external_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EVAL_SPEC = (
    OUT / "external_evaluation_spec.json"
)

RESULTS = (
    OUT / "external_policy_results.csv"
)

SUMMARY = (
    OUT / "summary.json"
)

BETAS = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]

# Frozen from SA-dev development selection.
GLOBAL_METHOD = "MN"
GLOBAL_BETA = 2.00

BOOTSTRAP_REPS = 20000
SEED = 20260920


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def exact_mcnemar(b, c):
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


def paired_bootstrap(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    diff = a - b

    rng = np.random.default_rng(SEED)

    vals = []

    done = 0

    while done < BOOTSTRAP_REPS:
        m = min(
            1000,
            BOOTSTRAP_REPS - done,
        )

        idx = rng.integers(
            0,
            len(diff),
            size=(m, len(diff)),
        )

        vals.append(
            diff[idx].mean(axis=1)
        )

        done += m

    vals = np.concatenate(vals)

    return (
        float(diff.mean()),
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )


# ============================================================
# HARD AUDIT
# ============================================================

if sha256(ROUTER) != EXPECTED_ROUTER_SHA:
    raise RuntimeError(
        "Frozen Utility Router SHA mismatch."
    )

if sha256(FEATURES) != EXPECTED_FEATURE_SHA:
    raise RuntimeError(
        "External utility-feature SHA mismatch."
    )

if not SWEEP.exists():
    raise FileNotFoundError(SWEEP)


# ============================================================
# FREEZE EXTERNAL EVALUATION SPEC BEFORE SCORING
# ============================================================

spec = {
    "experiment":
        "aroma_utility_router_external_v1",

    "status":
        "frozen external diagnostic evaluation",

    "population":
        "TSG prospective cohort, selected_alpha > 1",

    "expected_n":
        156,

    "router":
        str(ROUTER),

    "router_sha256":
        sha256(ROUTER),

    "candidate_set":
        "identity + MN/SA x "
        "{0.5,0.75,1,1.25,1.5,1.75,2}",

    "primary_comparator":
        {
            "method":
                GLOBAL_METHOD,
            "beta":
                GLOBAL_BETA,
            "selection_source":
                "frozen SA-dev development result",
        },

    "primary_endpoint":
        "exact-count accuracy",

    "secondary_endpoints": [
        "repairs",
        "breaks",
        "net repairs",
        "method selection distribution",
        "gain selection distribution",
    ],

    "selection_rule":
        "argmax frozen router probability; "
        "tie -> identity -> smaller beta -> MN -> SA",

    "no_external_tuning":
        True,

    "interpretation":
        "external diagnostic, not final pristine confirmation",
}


if EVAL_SPEC.exists():

    old = json.loads(
        EVAL_SPEC.read_text(
            encoding="utf-8"
        )
    )

    if old != spec:
        raise RuntimeError(
            "Existing external evaluation spec differs."
        )

else:

    EVAL_SPEC.write_text(
        json.dumps(
            spec,
            indent=2,
        ),
        encoding="utf-8",
    )


print(
    "External evaluation spec SHA256:",
    sha256(EVAL_SPEC),
)


# ============================================================
# LOAD FROZEN ROUTER
# ============================================================

bundle = joblib.load(
    ROUTER
)

model = bundle[
    "model"
]

sample_cols = list(
    bundle[
        "sample_cols"
    ]
)

feature_names = list(
    bundle[
        "feature_names"
    ]
)

if len(feature_names) != 431:
    raise RuntimeError(
        f"Expected 431 design features; "
        f"got {len(feature_names)}"
    )


# ============================================================
# LOAD EXTERNAL DATA
# ============================================================

features = pd.read_csv(
    FEATURES
)

sweep = pd.read_csv(
    SWEEP
)


features[
    "cohort_uid"
] = (
    features[
        "cohort_uid"
    ].astype(str)
)

features[
    "raw_sample_id"
] = (
    features[
        "raw_sample_id"
    ].astype(str)
)

sweep[
    "cohort_uid"
] = (
    sweep[
        "cohort_uid"
    ].astype(str)
)


up = features[
    features[
        "selected_alpha"
    ].astype(float)
    > 1.0
].copy()

up = (
    up.sort_values(
        "cohort_uid"
    )
    .reset_index(drop=True)
)


if len(up) != 156:
    raise RuntimeError(
        f"Expected upward N=156; got {len(up)}"
    )


# ============================================================
# BUILD SAME SAMPLE FEATURES AS TRAINING
# ============================================================

ctrl_cols = sorted(
    [
        c
        for c in up.columns
        if c.startswith("ctrl__")
    ]
)

if len(ctrl_cols) != 39:
    raise RuntimeError(
        f"Expected 39 controller features; "
        f"got {len(ctrl_cols)}"
    )


up[
    "mech__log_pg_norm"
] = np.log10(
    up[
        "pg_norm"
    ].astype(float)
)

up[
    "mech__log_h_norm"
] = np.log10(
    up[
        "h_norm"
    ].astype(float)
)

up[
    "mech__log_sa_sensitivity"
] = np.log10(
    up[
        "sa_sensitivity"
    ].astype(float)
)

up[
    "mech__cos_pug_puh"
] = up[
    "cos_pug_puh"
].astype(float)

up[
    "mech__gradient_capture"
] = up[
    "gradient_capture"
].astype(float)

up[
    "mech__activation_capture"
] = up[
    "activation_capture"
].astype(float)


expected_sample_cols = (
    ctrl_cols
    +
    [
        "selected_alpha",
        "selected_score",
        "mech__log_pg_norm",
        "mech__log_h_norm",
        "mech__log_sa_sensitivity",
        "mech__cos_pug_puh",
        "mech__gradient_capture",
        "mech__activation_capture",
    ]
)


if sample_cols != expected_sample_cols:
    raise RuntimeError(
        "External sample feature schema "
        "does not match frozen router."
    )


# ============================================================
# BUILD 15 CANDIDATES / SAMPLE
# ============================================================

rows = []


for _, sample in up.iterrows():

    uid = str(
        sample[
            "cohort_uid"
        ]
    )

    sid = str(
        sample[
            "raw_sample_id"
        ]
    )

    gt = int(
        sample[
            "ground_truth"
        ]
    )

    baseline = int(
        sample[
            "baseline_prediction"
        ]
    )


    shared = {
        c:
            float(sample[c])
        for c in sample_cols
    }


    # Identity.
    row = {
        "cohort_uid":
            uid,

        "raw_sample_id":
            sid,

        **shared,

        "candidate_id":
            "IDENTITY",

        "candidate_method":
            "IDENTITY",

        "candidate_beta":
            0.0,

        "candidate_beta_sq":
            0.0,

        "candidate_is_mn":
            0,

        "candidate_is_sa":
            0,

        "ground_truth":
            gt,

        "candidate_prediction":
            baseline,

        "candidate_correct":
            int(
                baseline == gt
            ),
    }

    rows.append(row)


    for method_name in [
        "MN",
        "SA",
    ]:

        for beta in BETAS:

            x = sweep[
                (
                    sweep[
                        "cohort_uid"
                    ]
                    == uid
                )
                &
                (
                    sweep[
                        "method"
                    ]
                    == method_name
                )
                &
                np.isclose(
                    sweep[
                        "beta0"
                    ].astype(float),
                    beta,
                )
            ]


            if len(x) != 1:
                raise RuntimeError(
                    f"{uid}: missing/duplicate "
                    f"{method_name} beta={beta}"
                )


            r = x.iloc[0]

            pred = int(
                r[
                    "prediction"
                ]
            )


            rows.append(
                {
                    "cohort_uid":
                        uid,

                    "raw_sample_id":
                        sid,

                    **shared,

                    "candidate_id":
                        f"{method_name}_b{beta:.2f}",

                    "candidate_method":
                        method_name,

                    "candidate_beta":
                        float(beta),

                    "candidate_beta_sq":
                        float(beta ** 2),

                    "candidate_is_mn":
                        int(
                            method_name
                            == "MN"
                        ),

                    "candidate_is_sa":
                        int(
                            method_name
                            == "SA"
                        ),

                    "ground_truth":
                        gt,

                    "candidate_prediction":
                        pred,

                    "candidate_correct":
                        int(
                            pred == gt
                        ),
                }
            )


cand = pd.DataFrame(
    rows
)


if len(cand) != 156 * 15:
    raise RuntimeError(
        f"Expected 2340 candidate rows; "
        f"got {len(cand)}"
    )


if not np.all(
    cand.groupby(
        "cohort_uid"
    ).size().to_numpy()
    == 15
):
    raise RuntimeError(
        "Not exactly 15 candidates/sample."
    )


# ============================================================
# SAME DESIGN MAP AS TRAINING
# ============================================================

candidate_channels = [
    "cand__mn",
    "cand__sa",
    "cand__beta",
    "cand__beta_sq",
    "cand__mn_beta",
    "cand__sa_beta",
    "cand__mn_beta_sq",
    "cand__sa_beta_sq",
]


def build_design(frame):

    Xs = (
        frame[
            sample_cols
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    mn = (
        frame[
            "candidate_is_mn"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    sa = (
        frame[
            "candidate_is_sa"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        frame[
            "candidate_beta"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    beta_sq = beta ** 2

    C = np.column_stack(
        [
            mn,
            sa,
            beta,
            beta_sq,
            mn * beta,
            sa * beta,
            mn * beta_sq,
            sa * beta_sq,
        ]
    )

    interactions = np.concatenate(
        [
            Xs * C[:, j:j+1]
            for j in range(
                C.shape[1]
            )
        ],
        axis=1,
    )

    X = np.concatenate(
        [
            Xs,
            C,
            interactions,
        ],
        axis=1,
    )

    names = (
        sample_cols
        +
        candidate_channels
        +
        [
            f"{s}__X__{c}"
            for c in candidate_channels
            for s in sample_cols
        ]
    )

    if names != feature_names:
        raise RuntimeError(
            "External design schema differs "
            "from frozen router."
        )

    return X


X = build_design(
    cand
)


prob = model.predict_proba(
    X
)[:, 1]


cand[
    "utility_probability"
] = prob


# ============================================================
# FROZEN POLICY SELECTION
# ============================================================

results = []


for uid, g in cand.groupby(
    "cohort_uid",
    sort=True,
):

    max_prob = float(
        g[
            "utility_probability"
        ].max()
    )

    tied = g[
        np.isclose(
            g[
                "utility_probability"
            ],
            max_prob,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()


    tied[
        "_identity"
    ] = (
        tied[
            "candidate_method"
        ]
        == "IDENTITY"
    ).astype(int)


    tied[
        "_method_order"
    ] = tied[
        "candidate_method"
    ].map(
        {
            "IDENTITY": 0,
            "MN": 1,
            "SA": 2,
        }
    )


    chosen = (
        tied
        .sort_values(
            [
                "_identity",
                "candidate_beta",
                "_method_order",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .iloc[0]
    )


    identity = g[
        g[
            "candidate_method"
        ]
        == "IDENTITY"
    ].iloc[0]


    global_row = g[
        (
            g[
                "candidate_method"
            ]
            == GLOBAL_METHOD
        )
        &
        np.isclose(
            g[
                "candidate_beta"
            ],
            GLOBAL_BETA,
        )
    ]


    if len(global_row) != 1:
        raise RuntimeError(
            f"{uid}: frozen global comparator missing."
        )


    global_row = (
        global_row.iloc[0]
    )


    oracle = int(
        g[
            "candidate_correct"
        ].max()
    )


    results.append(
        {
            "cohort_uid":
                uid,

            "router_candidate":
                str(
                    chosen[
                        "candidate_id"
                    ]
                ),

            "router_method":
                str(
                    chosen[
                        "candidate_method"
                    ]
                ),

            "router_beta":
                float(
                    chosen[
                        "candidate_beta"
                    ]
                ),

            "router_probability":
                float(
                    chosen[
                        "utility_probability"
                    ]
                ),

            "router_correct":
                int(
                    chosen[
                        "candidate_correct"
                    ]
                ),

            "identity_correct":
                int(
                    identity[
                        "candidate_correct"
                    ]
                ),

            "global_correct":
                int(
                    global_row[
                        "candidate_correct"
                    ]
                ),

            "oracle_correct":
                oracle,
        }
    )


res = pd.DataFrame(
    results
)


router = res[
    "router_correct"
].to_numpy(
    dtype=int
)

global_c = res[
    "global_correct"
].to_numpy(
    dtype=int
)

identity = res[
    "identity_correct"
].to_numpy(
    dtype=int
)

oracle = res[
    "oracle_correct"
].to_numpy(
    dtype=int
)


# ============================================================
# STATS
# ============================================================

repairs = int(
    (
        (identity == 0)
        &
        (router == 1)
    ).sum()
)

breaks = int(
    (
        (identity == 1)
        &
        (router == 0)
    ).sum()
)


global_repairs = int(
    (
        (identity == 0)
        &
        (global_c == 1)
    ).sum()
)

global_breaks = int(
    (
        (identity == 1)
        &
        (global_c == 0)
    ).sum()
)


router_only = int(
    (
        (router == 1)
        &
        (global_c == 0)
    ).sum()
)

global_only = int(
    (
        (global_c == 1)
        &
        (router == 0)
    ).sum()
)


diff, lo, hi = paired_bootstrap(
    router,
    global_c,
)

p = exact_mcnemar(
    router_only,
    global_only,
)


res.to_csv(
    RESULTS,
    index=False,
)


summary = {
    "n":
        156,

    "router_correct":
        int(router.sum()),

    "router_accuracy":
        float(router.mean()),

    "frozen_global_candidate":
        "MN_b2.00",

    "global_correct":
        int(global_c.sum()),

    "global_accuracy":
        float(global_c.mean()),

    "identity_correct":
        int(identity.sum()),

    "oracle_correct":
        int(oracle.sum()),

    "router_repairs":
        repairs,

    "router_breaks":
        breaks,

    "global_repairs":
        global_repairs,

    "global_breaks":
        global_breaks,

    "router_only":
        router_only,

    "global_only":
        global_only,

    "router_minus_global_pp":
        float(diff * 100),

    "bootstrap_95_ci_pp": [
        float(lo * 100),
        float(hi * 100),
    ],

    "mcnemar_exact_p":
        float(p),

    "method_distribution":
        {
            str(k): int(v)
            for k, v in
            res[
                "router_method"
            ]
            .value_counts()
            .sort_index()
            .items()
        },

    "evaluation_spec_sha256":
        sha256(EVAL_SPEC),

    "router_sha256":
        sha256(ROUTER),

    "external_features_sha256":
        sha256(FEATURES),
}


SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 96)
print(
    "FROZEN UTILITY ROUTER EXTERNAL VALIDATION COMPLETE"
)
print("=" * 96)

print(
    "N:",
    156,
)

print()

print(
    "Identity:",
    f"{identity.sum()}/156",
    f"({identity.mean():.6f})",
)

print(
    "Frozen global MN beta=2.0:",
    f"{global_c.sum()}/156",
    f"({global_c.mean():.6f})",
)

print(
    "Frozen Utility Router:",
    f"{router.sum()}/156",
    f"({router.mean():.6f})",
)

print(
    "Oracle:",
    f"{oracle.sum()}/156",
    f"({oracle.mean():.6f})",
)

print()

print(
    "Router repairs / breaks / net:",
    repairs,
    breaks,
    repairs - breaks,
)

print(
    "Global repairs / breaks / net:",
    global_repairs,
    global_breaks,
    global_repairs - global_breaks,
)

print()

print(
    "Router-only / Global-only:",
    router_only,
    global_only,
)

print(
    "Router - Global:",
    f"{diff * 100:+.3f} pp",
)

print(
    "95% CI:",
    f"[{lo * 100:+.3f}, "
    f"{hi * 100:+.3f}] pp",
)

print(
    "McNemar exact p:",
    p,
)

print()
print(
    "Method selection distribution:"
)

for k, v in (
    summary[
        "method_distribution"
    ].items()
):

    print(
        f"  {k}: {v}"
    )

print()
print(
    "Evaluation spec SHA256:",
    sha256(EVAL_SPEC),
)

print(
    "Summary SHA256:",
    sha256(SUMMARY),
)

print("=" * 96)
