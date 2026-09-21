import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# FROZEN INPUTS
# ============================================================

PROTOCOL = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_protocol_v1/"
    "utility_router_protocol_v1.json"
)

EXPECTED_PROTOCOL_SHA = (
    "2bf3ba6504c8915c846c810adf2a6652e31caba9658"
    "ba4993fd1de4d4959c5c8"
)

DATA = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_candidate_dataset_v1/"
    "utility_candidate_dataset_v1.csv"
)

EXPECTED_DATA_SHA = (
    "e615029ef04e44d923134a8aa355918af5f6b7a5fd49"
    "055a860ee90efb1c88cb"
)

OUT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_cv_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EXECUTION_SPEC = (
    OUT / "execution_spec.json"
)

OOF_CSV = (
    OUT / "oof_policy_results.csv"
)

FOLD_CSV = (
    OUT / "fold_summary.csv"
)

COEF_CSV = (
    OUT / "fold_coefficients.csv"
)

SUMMARY_JSON = (
    OUT / "summary.json"
)

SEED = 20260920
N_FOLDS = 5
C_VALUE = 1.0
BOOTSTRAP_REPS = 20000


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


def bootstrap_diff(
    a,
    b,
    reps=BOOTSTRAP_REPS,
    seed=SEED,
):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    diff = a - b

    rng = np.random.default_rng(
        seed
    )

    vals = []

    done = 0

    while done < reps:
        m = min(
            1000,
            reps - done,
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
            diff[idx].mean(axis=1)
        )

        done += m

    vals = np.concatenate(
        vals
    )

    return (
        float(diff.mean()),
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
# HASH AUDIT
# ============================================================

if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
    raise RuntimeError(
        "Protocol SHA mismatch."
    )

if sha256(DATA) != EXPECTED_DATA_SHA:
    raise RuntimeError(
        "Candidate dataset SHA mismatch."
    )

print(
    "PASS: frozen protocol and candidate dataset verified"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA
)

df[
    "sample_id"
] = (
    df[
        "sample_id"
    ].astype(str)
)

df[
    "cohort_uid"
] = (
    df[
        "cohort_uid"
    ].astype(str)
)


if len(df) != 2400:
    raise RuntimeError(
        f"Expected 2400 rows; got {len(df)}"
    )

if df[
    "sample_id"
].nunique() != 160:
    raise RuntimeError(
        "Expected 160 independent samples."
    )

if not np.all(
    df.groupby(
        "sample_id"
    ).size().to_numpy()
    == 15
):
    raise RuntimeError(
        "Every sample must have exactly 15 candidates."
    )


# ============================================================
# EXACT FEATURE MAP — FROZEN BEFORE CV
# ============================================================

ctrl_cols = sorted(
    [
        c
        for c in df.columns
        if c.startswith(
            "ctrl__"
        )
    ]
)

if len(ctrl_cols) != 39:
    raise RuntimeError(
        f"Expected 39 controller columns; "
        f"got {len(ctrl_cols)}"
    )


sample_cols = (
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


# Identity is the reference candidate.
#
# These channels allow:
#   direction,
#   gain,
#   nonlinear gain,
#   direction x gain
# effects.
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

    is_mn = (
        frame[
            "candidate_is_mn"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    is_sa = (
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
        frame[
            "candidate_beta_sq"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    C = np.column_stack(
        [
            is_mn,
            is_sa,
            beta,
            beta_sq,
            is_mn * beta,
            is_sa * beta,
            is_mn * beta_sq,
            is_sa * beta_sq,
        ]
    )

    # Candidate-conditioned sample interactions.
    interactions = np.concatenate(
        [
            Xs
            *
            C[:, j:j+1]
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

    if X.shape[1] != len(
        names
    ):
        raise RuntimeError(
            "Design-matrix/name mismatch."
        )

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Non-finite design matrix."
        )

    return X, names


X_all, design_names = build_design(
    df
)

print(
    "Design features:",
    len(
        design_names
    ),
)


# ============================================================
# FREEZE EXECUTION SPEC
# ============================================================

execution = {
    "experiment":
        "aroma_utility_router_cv_v1",

    "population_n":
        160,

    "candidate_rows":
        2400,

    "outer_cv":
        {
            "folds":
                5,
            "split_unit":
                "sample",
            "shuffle":
                True,
            "seed":
                SEED,
        },

    "model":
        {
            "type":
                "L2 logistic regression",
            "C":
                C_VALUE,
            "solver":
                "lbfgs",
            "max_iter":
                10000,
            "class_weight":
                None,
        },

    "identity_encoding":
        "reference candidate: all candidate channels zero",

    "sample_feature_count":
        len(
            sample_cols
        ),

    "candidate_channel_count":
        len(
            candidate_channels
        ),

    "design_feature_count":
        len(
            design_names
        ),

    "candidate_channels":
        candidate_channels,

    "interaction_policy":
        "every GT-free sample feature multiplied by "
        "every candidate channel",

    "selection_rule":
        "argmax predicted P(candidate_correct) "
        "within each held-out sample",

    "global_comparator":
        "within each training fold choose the single "
        "MN/SA candidate with highest training exact "
        "accuracy; ties -> fewer breaks -> smaller beta "
        "-> MN before SA",

    "candidate_dataset_sha256":
        sha256(
            DATA
        ),

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),
}


if EXECUTION_SPEC.exists():

    old = json.loads(
        EXECUTION_SPEC.read_text(
            encoding="utf-8"
        )
    )

    if old != execution:
        raise RuntimeError(
            "Existing execution spec differs."
        )

else:

    EXECUTION_SPEC.write_text(
        json.dumps(
            execution,
            indent=2,
        ),
        encoding="utf-8",
    )


print(
    "Execution spec SHA256:",
    sha256(
        EXECUTION_SPEC
    ),
)


# ============================================================
# SAMPLE-LEVEL FOLDS
# ============================================================

sample_ids = np.array(
    sorted(
        df[
            "sample_id"
        ].unique()
    ),
    dtype=str,
)


kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)


oof_rows = []
fold_rows = []
coef_rows = []


for fold, (
    train_sample_idx,
    test_sample_idx,
) in enumerate(
    kf.split(
        sample_ids
    ),
    start=1,
):

    train_ids = set(
        sample_ids[
            train_sample_idx
        ]
    )

    test_ids = set(
        sample_ids[
            test_sample_idx
        ]
    )


    if train_ids & test_ids:
        raise RuntimeError(
            "Sample leakage across fold."
        )


    train_mask = (
        df[
            "sample_id"
        ].isin(
            train_ids
        )
        .to_numpy()
    )

    test_mask = (
        df[
            "sample_id"
        ].isin(
            test_ids
        )
        .to_numpy()
    )


    train = df.loc[
        train_mask
    ].copy()

    test = df.loc[
        test_mask
    ].copy()


    X_train, names_train = build_design(
        train
    )

    X_test, names_test = build_design(
        test
    )


    if names_train != names_test:
        raise RuntimeError(
            "Fold design-name mismatch."
        )


    y_train = (
        train[
            "candidate_correct"
        ]
        .astype(int)
        .to_numpy()
    )


    if set(
        np.unique(
            y_train
        )
    ) != {
        0,
        1,
    }:
        raise RuntimeError(
            f"Fold {fold}: training labels "
            "do not contain both classes."
        )


    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    penalty="l2",
                    C=C_VALUE,
                    solver="lbfgs",
                    max_iter=10000,
                ),
            ),
        ]
    )


    model.fit(
        X_train,
        y_train,
    )


    prob = model.predict_proba(
        X_test
    )[:, 1]


    test = test.copy()

    test[
        "utility_probability"
    ] = prob


    # ========================================================
    # TRAIN-FOLD GLOBAL COMPARATOR
    # ========================================================

    train_identity = (
        train[
            train[
                "candidate_method"
            ]
            == "IDENTITY"
        ][
            [
                "sample_id",
                "candidate_correct",
            ]
        ]
        .rename(
            columns={
                "candidate_correct":
                    "identity_correct"
            }
        )
    )


    intervention_train = train[
        train[
            "candidate_method"
        ]
        .isin(
            [
                "MN",
                "SA",
            ]
        )
    ].merge(
        train_identity,
        on="sample_id",
        validate="many_to_one",
    )


    global_rows = []


    for candidate_id, g in (
        intervention_train
        .groupby(
            "candidate_id"
        )
    ):

        correct = int(
            g[
                "candidate_correct"
            ].sum()
        )

        breaks = int(
            (
                (g["identity_correct"] == 1)
                &
                (g["candidate_correct"] == 0)
            ).sum()
        )

        beta = float(
            g[
                "candidate_beta"
            ].iloc[0]
        )

        method = str(
            g[
                "candidate_method"
            ].iloc[0]
        )

        global_rows.append(
            {
                "candidate_id":
                    candidate_id,
                "correct":
                    correct,
                "breaks":
                    breaks,
                "beta":
                    beta,
                "method":
                    method,
                "method_order":
                    0
                    if method == "MN"
                    else 1,
            }
        )


    global_choice = (
        pd.DataFrame(
            global_rows
        )
        .sort_values(
            [
                "correct",
                "breaks",
                "beta",
                "method_order",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        )
        .iloc[0]
    )


    global_candidate_id = str(
        global_choice[
            "candidate_id"
        ]
    )


    # ========================================================
    # HELD-OUT SAMPLE POLICY
    # ========================================================

    fold_policy_correct = []
    fold_global_correct = []
    fold_identity_correct = []
    fold_oracle_correct = []


    for sid, g in (
        test.groupby(
            "sample_id",
            sort=True,
        )
    ):

        if len(g) != 15:
            raise RuntimeError(
                f"{sid}: expected 15 test candidates."
            )


        # Learned utility router.
        max_prob = float(
            g[
                "utility_probability"
            ].max()
        )

        tied = g[
            np.isclose(
                g[
                    "utility_probability"
                ].to_numpy(),
                max_prob,
                atol=1e-12,
                rtol=0.0,
            )
        ].copy()


        # Deterministic tie break:
        # identity -> smaller beta -> MN -> SA.
        tied[
            "_tie_identity"
        ] = (
            tied[
                "candidate_method"
            ]
            == "IDENTITY"
        ).astype(int)

        tied[
            "_tie_method"
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
                    "_tie_identity",
                    "candidate_beta",
                    "_tie_method",
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
        ]

        if len(identity) != 1:
            raise RuntimeError(
                f"{sid}: identity count mismatch."
            )

        identity = identity.iloc[0]


        global_candidate = g[
            g[
                "candidate_id"
            ]
            == global_candidate_id
        ]

        if len(global_candidate) != 1:
            raise RuntimeError(
                f"{sid}: global comparator "
                f"{global_candidate_id} missing."
            )

        global_candidate = (
            global_candidate.iloc[0]
        )


        oracle_correct = int(
            g[
                "candidate_correct"
            ].max()
        )


        rec = {
            "fold":
                fold,

            "sample_id":
                sid,

            "cohort_uid":
                str(
                    chosen[
                        "cohort_uid"
                    ]
                ),

            "selected_alpha":
                float(
                    chosen[
                        "selected_alpha"
                    ]
                ),

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

            "global_candidate":
                global_candidate_id,

            "global_correct":
                int(
                    global_candidate[
                        "candidate_correct"
                    ]
                ),

            "oracle_correct":
                oracle_correct,
        }


        oof_rows.append(
            rec
        )

        fold_policy_correct.append(
            rec[
                "router_correct"
            ]
        )

        fold_global_correct.append(
            rec[
                "global_correct"
            ]
        )

        fold_identity_correct.append(
            rec[
                "identity_correct"
            ]
        )

        fold_oracle_correct.append(
            rec[
                "oracle_correct"
            ]
        )


    fold_rows.append(
        {
            "fold":
                fold,

            "train_samples":
                len(
                    train_ids
                ),

            "test_samples":
                len(
                    test_ids
                ),

            "global_candidate":
                global_candidate_id,

            "router_correct":
                int(
                    np.sum(
                        fold_policy_correct
                    )
                ),

            "global_correct":
                int(
                    np.sum(
                        fold_global_correct
                    )
                ),

            "identity_correct":
                int(
                    np.sum(
                        fold_identity_correct
                    )
                ),

            "oracle_correct":
                int(
                    np.sum(
                        fold_oracle_correct
                    )
                ),
        }
    )


    # Save fold coefficients in standardized space.
    coef = (
        model[
            "logistic"
        ]
        .coef_[0]
    )

    for name, value in zip(
        design_names,
        coef,
    ):

        coef_rows.append(
            {
                "fold":
                    fold,
                "feature":
                    name,
                "coefficient":
                    float(
                        value
                    ),
            }
        )


# ============================================================
# OOF AGGREGATION
# ============================================================

oof = pd.DataFrame(
    oof_rows
).sort_values(
    "sample_id"
).reset_index(
    drop=True
)


if len(oof) != 160:
    raise RuntimeError(
        f"OOF N mismatch: {len(oof)}"
    )


if oof[
    "sample_id"
].nunique() != 160:
    raise RuntimeError(
        "OOF duplicate sample."
    )


fold_df = pd.DataFrame(
    fold_rows
)

coef_df = pd.DataFrame(
    coef_rows
)


router = (
    oof[
        "router_correct"
    ]
    .astype(int)
    .to_numpy()
)

global_c = (
    oof[
        "global_correct"
    ]
    .astype(int)
    .to_numpy()
)

identity = (
    oof[
        "identity_correct"
    ]
    .astype(int)
    .to_numpy()
)

oracle = (
    oof[
        "oracle_correct"
    ]
    .astype(int)
    .to_numpy()
)


# ============================================================
# REPAIRS / BREAKS
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


# ============================================================
# PAIRED STATS
# ============================================================

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


diff, lo, hi = bootstrap_diff(
    router,
    global_c,
)


mcnemar = exact_mcnemar(
    router_only,
    global_only,
)


oracle_headroom = (
    oracle.mean()
    -
    global_c.mean()
)

recovered = (
    router.mean()
    -
    global_c.mean()
)

if oracle_headroom > 0:

    headroom_fraction = (
        recovered
        /
        oracle_headroom
    )

else:

    headroom_fraction = np.nan


# ============================================================
# SELECTION DISTRIBUTION
# ============================================================

selection_distribution = (
    oof[
        "router_candidate"
    ]
    .value_counts()
    .sort_index()
    .to_dict()
)


method_distribution = (
    oof[
        "router_method"
    ]
    .value_counts()
    .sort_index()
    .to_dict()
)


# ============================================================
# SAVE
# ============================================================

oof.to_csv(
    OOF_CSV,
    index=False,
)

fold_df.to_csv(
    FOLD_CSV,
    index=False,
)

coef_df.to_csv(
    COEF_CSV,
    index=False,
)


summary = {
    "experiment":
        "aroma_utility_router_cv_v1",

    "n":
        160,

    "router_correct":
        int(
            router.sum()
        ),

    "router_accuracy":
        float(
            router.mean()
        ),

    "global_correct":
        int(
            global_c.sum()
        ),

    "global_accuracy":
        float(
            global_c.mean()
        ),

    "identity_correct":
        int(
            identity.sum()
        ),

    "identity_accuracy":
        float(
            identity.mean()
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

    "router_only_correct":
        router_only,

    "global_only_correct":
        global_only,

    "router_minus_global_pp":
        float(
            diff
            * 100
        ),

    "bootstrap_95_ci_pp": [
        float(
            lo
            * 100
        ),
        float(
            hi
            * 100
        ),
    ],

    "mcnemar_exact_p":
        float(
            mcnemar
        ),

    "oracle_headroom_pp":
        float(
            oracle_headroom
            * 100
        ),

    "recovered_headroom_pp":
        float(
            recovered
            * 100
        ),

    "fraction_oracle_headroom_recovered":
        (
            None
            if np.isnan(
                headroom_fraction
            )
            else float(
                headroom_fraction
            )
        ),

    "selection_distribution":
        {
            str(k):
                int(v)
            for k, v
            in selection_distribution.items()
        },

    "method_distribution":
        {
            str(k):
                int(v)
            for k, v
            in method_distribution.items()
        },

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),

    "candidate_dataset_sha256":
        sha256(
            DATA
        ),

    "execution_spec_sha256":
        sha256(
            EXECUTION_SPEC
        ),
}


SUMMARY_JSON.write_text(
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
print("=" * 100)
print(
    "UTILITY ROUTER 5-FOLD OOF COMPLETE"
)
print("=" * 100)

print(
    "N:",
    160,
)

print()

print(
    "Identity:",
    f"{identity.sum()}/160",
    f"({identity.mean():.6f})",
)

print(
    "Fold-selected global:",
    f"{global_c.sum()}/160",
    f"({global_c.mean():.6f})",
)

print(
    "Utility Router OOF:",
    f"{router.sum()}/160",
    f"({router.mean():.6f})",
)

print(
    "Oracle:",
    f"{oracle.sum()}/160",
    f"({oracle.mean():.6f})",
)

print()
print(
    "Utility Router repairs / breaks / net:",
    router_repairs,
    router_breaks,
    router_repairs - router_breaks,
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
    "Bootstrap 95% CI:",
    f"[{lo * 100:+.3f}, "
    f"{hi * 100:+.3f}] pp",
)

print(
    "McNemar exact p:",
    mcnemar,
)

print()
print(
    "Oracle headroom over global:",
    f"{oracle_headroom * 100:+.3f} pp",
)

print(
    "Headroom recovered by router:",
    f"{recovered * 100:+.3f} pp",
)

print(
    "Fraction of oracle headroom recovered:",
    headroom_fraction,
)

print()
print(
    "Method selection distribution:"
)

for k, v in (
    method_distribution.items()
):

    print(
        f"  {k}: {v}"
    )

print()
print(
    "Fold summary:"
)

print(
    fold_df.to_string(
        index=False
    )
)

print()
print(
    "Execution spec SHA256:",
    sha256(
        EXECUTION_SPEC
    ),
)

print(
    "Summary:",
    SUMMARY_JSON,
)

print("=" * 100)
