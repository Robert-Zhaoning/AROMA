import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# FROZEN ARTIFACTS
# ============================================================

ROUTER = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_final_v1/"
    "aroma_utility_router_v1.joblib"
)

EXPECTED_ROUTER_SHA = (
    "4192129c525822fed7ab9d013dc3d564981f48faf81f28"
    "b3011a4be1a2168563"
)

FEATURES = Path(
    "outputs/proc_count_utility_final_v1/"
    "utility_features_v1/"
    "utility_final_features_v1.csv"
)

EXPECTED_FEATURE_SHA = (
    "9ea79addd2df8830c1cd0d0485b63d331ab9d3fc0c64f"
    "0d9055511f65c9f0681"
)

FEATURE_FREEZE = Path(
    "outputs/proc_count_utility_final_v1/"
    "utility_features_v1/freeze/"
    "utility_final_features_freeze_v1.json"
)

EXPECTED_FEATURE_FREEZE_SHA = (
    "f672bcd8f6549487c393df393831915ba4a0acbbf50db"
    "76b928a092f14a353c2"
)

CANDIDATES = Path(
    "outputs/proc_count_utility_final_v1/"
    "mn_sa_sweep_v1/"
    "mn_sa_results.csv"
)

EXPECTED_CANDIDATE_SHA = (
    "4bd5c477be016a5e567bddbd0b4564aa637c84b3ae608"
    "51ba0243fa34250036a"
)

CANDIDATE_FREEZE = Path(
    "outputs/proc_count_utility_final_v1/"
    "mn_sa_sweep_v1/freeze/"
    "utility_final_candidate_outcomes_freeze_v1.json"
)

EXPECTED_CANDIDATE_FREEZE_SHA = (
    "a66dce80ba45a2f9d9d930d8c29f4ed195795e046759"
    "9d6c68237410e42b7145"
)

PROTOCOL = Path(
    "outputs/aroma2/"
    "utility_router_final_confirmation_v1/"
    "final_confirmation_protocol.json"
)

EXPECTED_PROTOCOL_SHA = (
    "7a37d80f53e86944631571799d55b8eac8d4873ce74cf"
    "1f9a84d11c118654922"
)

ACTION = Path(
    "outputs/proc_count_utility_final_v1/"
    "action_manifest_v1/"
    "utility_final_action_manifest_v1.csv"
)

EXPECTED_ACTION_SHA = (
    "3123d61cfca130376a2b911707ea20927872b1a270ac1"
    "fbd5a852d3e82fc3040"
)

OUT = Path(
    "outputs/proc_count_utility_final_v1/"
    "final_utility_evaluation_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EXECUTION_FREEZE = (
    OUT
    / "final_evaluation_execution_freeze.json"
)

POLICY_RESULTS = (
    OUT
    / "final_policy_results.csv"
)

SUMMARY = (
    OUT
    / "final_summary.json"
)


EXPECTED_N = 294

BETAS = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]

GLOBAL_METHOD = "MN"
GLOBAL_BETA = 2.00

SECONDARY_SA_BETA = 1.75

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260920


# ============================================================
# HELPERS
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def exact_mcnemar(
    b,
    c,
):

    n = int(
        b + c
    )

    if n == 0:
        return 1.0

    k = int(
        min(
            b,
            c,
        )
    )

    tail = sum(
        math.comb(
            n,
            i,
        )
        for i in range(
            k + 1
        )
    ) / (
        2 ** n
    )

    return float(
        min(
            1.0,
            2.0 * tail,
        )
    )


def paired_bootstrap(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    diff = (
        a - b
    )

    rng = (
        np.random
        .default_rng(
            BOOTSTRAP_SEED
        )
    )

    vals = []

    done = 0

    while done < BOOTSTRAP_REPS:

        m = min(
            1000,
            BOOTSTRAP_REPS
            -
            done,
        )

        idx = rng.integers(
            0,
            len(diff),
            size=(
                m,
                len(diff),
            ),
        )

        vals.append(
            diff[
                idx
            ].mean(
                axis=1
            )
        )

        done += m

    vals = np.concatenate(
        vals
    )

    return (
        float(
            diff.mean()
        ),
        float(
            np.quantile(
                vals,
                0.025,
            )
        ),
        float(
            np.quantile(
                vals,
                0.975,
            )
        ),
    )


# ============================================================
# HARD SHA AUDIT
# ============================================================

checks = [
    (
        ROUTER,
        EXPECTED_ROUTER_SHA,
        "utility router",
    ),
    (
        FEATURES,
        EXPECTED_FEATURE_SHA,
        "final features",
    ),
    (
        FEATURE_FREEZE,
        EXPECTED_FEATURE_FREEZE_SHA,
        "feature freeze",
    ),
    (
        CANDIDATES,
        EXPECTED_CANDIDATE_SHA,
        "candidate outcomes",
    ),
    (
        CANDIDATE_FREEZE,
        EXPECTED_CANDIDATE_FREEZE_SHA,
        "candidate freeze",
    ),
    (
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA,
        "final protocol",
    ),
    (
        ACTION,
        EXPECTED_ACTION_SHA,
        "action manifest",
    ),
]

for path, expected, name in checks:

    actual = sha256(
        path
    )

    if actual != expected:

        raise RuntimeError(
            f"{name} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


print(
    "PASS: all frozen final artifacts verified"
)


# ============================================================
# FREEZE FINAL EXECUTION BEFORE REVEAL
# ============================================================

execution = {
    "experiment":
        "aroma_utility_router_final_confirmation_v1",

    "status":
        "single frozen final evaluation",

    "population":
        "final selected_alpha > 1",

    "n":
        EXPECTED_N,

    "candidate_set":
        {
            "identity":
                True,

            "methods": [
                "MN",
                "SA",
            ],

            "beta_grid":
                BETAS,
        },

    "primary_method":
        "Frozen Utility Router v1",

    "primary_comparator":
        "MN beta=2.00",

    "primary_endpoint":
        "paired exact-count accuracy",

    "primary_test":
        "two-sided exact McNemar",

    "effect_interval":
        "paired bootstrap 95% CI",

    "bootstrap_repetitions":
        BOOTSTRAP_REPS,

    "bootstrap_seed":
        BOOTSTRAP_SEED,

    "tie_break":
        "identity -> smaller beta -> MN -> SA",

    "secondary_fixed_SA":
        "SA beta=1.75",

    "no_tuning":
        True,

    "router_sha256":
        sha256(
            ROUTER
        ),

    "feature_freeze_sha256":
        sha256(
            FEATURE_FREEZE
        ),

    "candidate_freeze_sha256":
        sha256(
            CANDIDATE_FREEZE
        ),

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),
}


if EXECUTION_FREEZE.exists():

    old = json.loads(
        EXECUTION_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    if old != execution:

        raise RuntimeError(
            "Existing final execution freeze differs."
        )

else:

    EXECUTION_FREEZE.write_text(
        json.dumps(
            execution,
            indent=2,
        ),
        encoding="utf-8",
    )


print(
    "Final execution freeze SHA256:",
    sha256(
        EXECUTION_FREEZE
    ),
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

design_feature_names = list(
    bundle[
        "feature_names"
    ]
)


if len(
    design_feature_names
) != 431:

    raise RuntimeError(
        f"Expected 431 design features; "
        f"got {len(design_feature_names)}"
    )


# ============================================================
# LOAD FINAL FEATURES
# ============================================================

features = pd.read_csv(
    FEATURES
)

features[
    "raw_sample_id"
] = (
    features[
        "raw_sample_id"
    ].astype(str)
)

features[
    "cohort_uid"
] = (
    features[
        "cohort_uid"
    ].astype(str)
)


features = (
    features
    .sort_values(
        "cohort_uid"
    )
    .reset_index(
        drop=True
    )
)


if len(
    features
) != EXPECTED_N:

    raise RuntimeError(
        f"Expected feature N={EXPECTED_N}; "
        f"got {len(features)}"
    )


if features[
    "cohort_uid"
].nunique() != EXPECTED_N:

    raise RuntimeError(
        "Feature cohort_uid uniqueness failure."
    )


ctrl_cols = sorted(
    [
        c
        for c in features.columns
        if c.startswith(
            "ctrl__"
        )
    ]
)


if len(
    ctrl_cols
) != 39:

    raise RuntimeError(
        f"Expected 39 ctrl features; "
        f"got {len(ctrl_cols)}"
    )


# ============================================================
# RECONSTRUCT EXACT ROUTER SAMPLE FEATURES
# ============================================================

features[
    "mech__log_pg_norm"
] = np.log10(
    features[
        "pg_norm"
    ].astype(float)
)

features[
    "mech__log_h_norm"
] = np.log10(
    features[
        "h_norm"
    ].astype(float)
)

features[
    "mech__log_sa_sensitivity"
] = np.log10(
    features[
        "sa_sensitivity"
    ].astype(float)
)

features[
    "mech__cos_pug_puh"
] = features[
    "cos_pug_puh"
].astype(float)

features[
    "mech__gradient_capture"
] = features[
    "gradient_capture"
].astype(float)

features[
    "mech__activation_capture"
] = features[
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
        "Final sample feature schema differs "
        "from frozen Utility Router."
    )


# ============================================================
# LOAD FROZEN CANDIDATE OUTCOMES
# ============================================================

outcomes = pd.read_csv(
    CANDIDATES
)

outcomes[
    "raw_sample_id"
] = (
    outcomes[
        "raw_sample_id"
    ].astype(str)
)

outcomes[
    "cohort_uid"
] = (
    outcomes[
        "cohort_uid"
    ].astype(str)
)


if len(
    outcomes
) != (
    EXPECTED_N
    *
    14
):

    raise RuntimeError(
        f"Expected 4116 outcome rows; "
        f"got {len(outcomes)}"
    )


if not np.all(
    outcomes.groupby(
        "cohort_uid"
    ).size().to_numpy()
    ==
    14
):

    raise RuntimeError(
        "Not exactly 14 outcomes/sample."
    )


feature_uids = set(
    features[
        "cohort_uid"
    ]
)

outcome_uids = set(
    outcomes[
        "cohort_uid"
    ]
)


if feature_uids != outcome_uids:

    raise RuntimeError(
        "Feature/outcome cohort population mismatch."
    )


# ============================================================
# BUILD 15 CANDIDATES / SAMPLE
# ============================================================

rows = []


for _, sample in features.iterrows():

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
        col:
            float(
                sample[
                    col
                ]
            )
        for col in sample_cols
    }


    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    rows.append(
        {
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
                    baseline
                    ==
                    gt
                ),
        }
    )


    # --------------------------------------------------------
    # MN / SA x beta
    # --------------------------------------------------------

    sample_out = outcomes[
        outcomes[
            "cohort_uid"
        ]
        ==
        uid
    ]


    for method_name in [
        "MN",
        "SA",
    ]:

        for beta in BETAS:

            x = sample_out[
                (
                    sample_out[
                        "method"
                    ]
                    ==
                    method_name
                )
                &
                np.isclose(
                    sample_out[
                        "beta"
                    ].astype(float),
                    beta,
                    atol=1e-12,
                    rtol=0.0,
                )
            ]


            if len(x) != 1:

                raise RuntimeError(
                    f"{uid}: missing/duplicate "
                    f"{method_name} beta={beta}"
                )


            r = x.iloc[
                0
            ]

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
                        float(
                            beta
                        ),

                    "candidate_beta_sq":
                        float(
                            beta ** 2
                        ),

                    "candidate_is_mn":
                        int(
                            method_name
                            ==
                            "MN"
                        ),

                    "candidate_is_sa":
                        int(
                            method_name
                            ==
                            "SA"
                        ),

                    "ground_truth":
                        gt,

                    "candidate_prediction":
                        pred,

                    "candidate_correct":
                        int(
                            pred
                            ==
                            gt
                        ),
                }
            )


cand = pd.DataFrame(
    rows
)


if len(
    cand
) != (
    EXPECTED_N
    *
    15
):

    raise RuntimeError(
        f"Expected 4410 candidate rows; "
        f"got {len(cand)}"
    )


if not np.all(
    cand.groupby(
        "cohort_uid"
    ).size().to_numpy()
    ==
    15
):

    raise RuntimeError(
        "Not exactly 15 candidates/sample."
    )


# ============================================================
# EXACT FROZEN ROUTER DESIGN MAP
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


def build_design(
    frame
):

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

    beta_sq = (
        beta ** 2
    )


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
            Xs
            *
            C[
                :,
                j:j+1
            ]

            for j in range(
                C.shape[
                    1
                ]
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


    if names != design_feature_names:

        raise RuntimeError(
            "Final design schema differs "
            "from frozen Utility Router."
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
        ==
        "IDENTITY"
    ).astype(int)


    tied[
        "_method_order"
    ] = (
        tied[
            "candidate_method"
        ]
        .map(
            {
                "IDENTITY": 0,
                "MN": 1,
                "SA": 2,
            }
        )
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
        .iloc[
            0
        ]
    )


    identity = g[
        g[
            "candidate_method"
        ]
        ==
        "IDENTITY"
    ].iloc[
        0
    ]


    global_row = g[
        (
            g[
                "candidate_method"
            ]
            ==
            GLOBAL_METHOD
        )
        &
        np.isclose(
            g[
                "candidate_beta"
            ],
            GLOBAL_BETA,
            atol=1e-12,
            rtol=0.0,
        )
    ]


    if len(
        global_row
    ) != 1:

        raise RuntimeError(
            f"{uid}: global comparator missing."
        )


    global_row = (
        global_row.iloc[
            0
        ]
    )


    secondary_sa = g[
        (
            g[
                "candidate_method"
            ]
            ==
            "SA"
        )
        &
        np.isclose(
            g[
                "candidate_beta"
            ],
            SECONDARY_SA_BETA,
            atol=1e-12,
            rtol=0.0,
        )
    ]


    if len(
        secondary_sa
    ) != 1:

        raise RuntimeError(
            f"{uid}: secondary SA comparator missing."
        )


    secondary_sa = (
        secondary_sa.iloc[
            0
        ]
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

            "secondary_sa_correct":
                int(
                    secondary_sa[
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


# ============================================================
# PRIMARY ARRAYS
# ============================================================

router = (
    res[
        "router_correct"
    ]
    .to_numpy(
        dtype=int
    )
)

global_c = (
    res[
        "global_correct"
    ]
    .to_numpy(
        dtype=int
    )
)

identity = (
    res[
        "identity_correct"
    ]
    .to_numpy(
        dtype=int
    )
)

secondary_sa = (
    res[
        "secondary_sa_correct"
    ]
    .to_numpy(
        dtype=int
    )
)

oracle = (
    res[
        "oracle_correct"
    ]
    .to_numpy(
        dtype=int
    )
)


# ============================================================
# PRIMARY STATISTICS
# ============================================================

router_repairs = int(
    (
        (identity == 0)
        &
        (router == 1)
    ).sum()
)

router_breaks = int(
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


mcnemar_p = exact_mcnemar(
    router_only,
    global_only,
)


# ============================================================
# SECONDARY FULL-POLICY ACCURACY ON N=1000
# ============================================================

full_actions = pd.read_csv(
    ACTION
)

full_baseline_correct = (
    full_actions[
        "baseline_prediction"
    ].astype(int)
    ==
    full_actions[
        "ground_truth"
    ].astype(int)
)


if len(
    full_actions
) != 1000:

    raise RuntimeError(
        "Full action manifest N mismatch."
    )


baseline_full_correct = int(
    full_baseline_correct.sum()
)

upward_identity_correct = int(
    identity.sum()
)


router_full_correct = (
    baseline_full_correct
    -
    upward_identity_correct
    +
    int(
        router.sum()
    )
)


global_full_correct = (
    baseline_full_correct
    -
    upward_identity_correct
    +
    int(
        global_c.sum()
    )
)


# ============================================================
# SAVE BEFORE PRINTING
# ============================================================

res.to_csv(
    POLICY_RESULTS,
    index=False,
)


method_distribution = {
    str(k):
        int(v)
    for k, v in (
        res[
            "router_method"
        ]
        .value_counts()
        .sort_index()
        .items()
    )
}


beta_distribution = {
    str(k):
        int(v)
    for k, v in (
        res[
            "router_beta"
        ]
        .value_counts()
        .sort_index()
        .items()
    )
}


summary = {
    "experiment":
        "aroma_utility_router_final_confirmation_v1",

    "upward_n":
        EXPECTED_N,

    "identity_correct":
        int(
            identity.sum()
        ),

    "identity_accuracy":
        float(
            identity.mean()
        ),

    "global_MN_beta_2_correct":
        int(
            global_c.sum()
        ),

    "global_MN_beta_2_accuracy":
        float(
            global_c.mean()
        ),

    "router_correct":
        int(
            router.sum()
        ),

    "router_accuracy":
        float(
            router.mean()
        ),

    "secondary_SA_beta_1_75_correct":
        int(
            secondary_sa.sum()
        ),

    "secondary_SA_beta_1_75_accuracy":
        float(
            secondary_sa.mean()
        ),

    "oracle_correct":
        int(
            oracle.sum()
        ),

    "oracle_accuracy":
        float(
            oracle.mean()
        ),

    "router_repairs":
        router_repairs,

    "router_breaks":
        router_breaks,

    "router_net_repairs":
        router_repairs
        -
        router_breaks,

    "global_repairs":
        global_repairs,

    "global_breaks":
        global_breaks,

    "global_net_repairs":
        global_repairs
        -
        global_breaks,

    "router_only":
        router_only,

    "global_only":
        global_only,

    "router_minus_global_pp":
        float(
            diff
            *
            100
        ),

    "bootstrap_95_ci_pp": [
        float(
            lo
            *
            100
        ),
        float(
            hi
            *
            100
        ),
    ],

    "mcnemar_exact_p":
        float(
            mcnemar_p
        ),

    "method_distribution":
        method_distribution,

    "beta_distribution":
        beta_distribution,

    "full_n":
        1000,

    "baseline_full_correct":
        baseline_full_correct,

    "baseline_full_accuracy":
        baseline_full_correct
        /
        1000.0,

    "router_full_correct":
        int(
            router_full_correct
        ),

    "router_full_accuracy":
        float(
            router_full_correct
            /
            1000.0
        ),

    "global_full_correct":
        int(
            global_full_correct
        ),

    "global_full_accuracy":
        float(
            global_full_correct
            /
            1000.0
        ),

    "execution_freeze_sha256":
        sha256(
            EXECUTION_FREEZE
        ),

    "router_sha256":
        sha256(
            ROUTER
        ),

    "feature_freeze_sha256":
        sha256(
            FEATURE_FREEZE
        ),

    "candidate_freeze_sha256":
        sha256(
            CANDIDATE_FREEZE
        ),

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),
}


SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL REVEAL
# ============================================================

print()
print("=" * 104)
print(
    "AROMA UTILITY ROUTER FINAL CONFIRMATION — RESULT"
)
print("=" * 104)

print(
    "Primary population:",
    f"N={EXPECTED_N}",
)

print()

print(
    "Identity:",
    f"{int(identity.sum())}/{EXPECTED_N}",
    f"({identity.mean():.6f})",
)

print(
    "Frozen global MN beta=2.00:",
    f"{int(global_c.sum())}/{EXPECTED_N}",
    f"({global_c.mean():.6f})",
)

print(
    "Frozen Utility Router v1:",
    f"{int(router.sum())}/{EXPECTED_N}",
    f"({router.mean():.6f})",
)

print(
    "Secondary fixed SA beta=1.75:",
    f"{int(secondary_sa.sum())}/{EXPECTED_N}",
    f"({secondary_sa.mean():.6f})",
)

print(
    "Oracle:",
    f"{int(oracle.sum())}/{EXPECTED_N}",
    f"({oracle.mean():.6f})",
)

print()

print(
    "Router repairs / breaks / net:",
    router_repairs,
    router_breaks,
    router_repairs
    -
    router_breaks,
)

print(
    "Global repairs / breaks / net:",
    global_repairs,
    global_breaks,
    global_repairs
    -
    global_breaks,
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
    "Paired bootstrap 95% CI:",
    f"[{lo * 100:+.3f}, "
    f"{hi * 100:+.3f}] pp",
)

print(
    "Exact McNemar p:",
    mcnemar_p,
)

print()
print(
    "Method selection distribution:"
)

for k, v in method_distribution.items():

    print(
        f"  {k}: {v}"
    )


print()
print(
    "Beta selection distribution:"
)

for k, v in beta_distribution.items():

    print(
        f"  beta={k}: {v}"
    )


print()
print(
    "Full-policy N=1000:"
)

print(
    "  Baseline:",
    f"{baseline_full_correct}/1000",
    f"({baseline_full_correct / 1000:.6f})",
)

print(
    "  Frozen global MN beta=2:",
    f"{global_full_correct}/1000",
    f"({global_full_correct / 1000:.6f})",
)

print(
    "  Frozen Utility Router:",
    f"{router_full_correct}/1000",
    f"({router_full_correct / 1000:.6f})",
)

print()

print(
    "Final execution freeze SHA256:",
    sha256(
        EXECUTION_FREEZE
    ),
)

print(
    "Final summary SHA256:",
    sha256(
        SUMMARY
    ),
)

print("=" * 104)
