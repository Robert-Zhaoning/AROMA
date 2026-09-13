from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore")


FEATURE_PATH = Path(
    "outputs/proc_count_causal_v1/router/"
    "full_probability_response/"
    "full_response_router_features.csv"
)

RESPONSE_PATH = Path(
    "outputs/proc_count_causal_v1/router/"
    "full_probability_response/"
    "full_probability_response_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "learned_gain_gate"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PRED_PATH = (
    OUT_DIR
    / "gain_gate_predictions.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "gain_gate_summary.csv"
)


SEED = 20260912

HEAD_NAME = "L18H13"

ACTIONS = {
    "noop": 1.0,
    "gain_0p5": 0.5,
    "gain_1p5": 1.5,
}

C_GRID = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]


# ============================================================
# Load
# ============================================================

features = pd.read_csv(
    FEATURE_PATH
)

response = pd.read_csv(
    RESPONSE_PATH
)

response["head_name"] = (
    "L"
    + response["layer"].astype(int).astype(str)
    + "H"
    + response["head"].astype(int).astype(str)
)

print("=" * 120)
print("AROMA LEARNED GAIN GATE")
print("=" * 120)

print("Feature rows:", len(features))
print("Response rows:", len(response))

assert len(features) == 54
assert features["pair_id"].nunique() == 27


# ============================================================
# Passive baseline features
# ============================================================

baseline_features = [
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


# ============================================================
# Build sample × action table
# ============================================================

rows = []

for _, f in features.iterrows():

    sid = str(
        f["sample_id"]
    )

    pair_id = int(
        f["pair_id"]
    )

    role = str(
        f["role"]
    )

    r_sample = response[
        (response["sample_id"].astype(str) == sid)
        &
        (response["head_name"] == HEAD_NAME)
    ].copy()

    assert len(r_sample) == 5

    gt_values = (
        r_sample[
            "ground_truth"
        ]
        .astype(int)
        .unique()
    )

    assert len(gt_values) == 1

    gt = int(
        gt_values[0]
    )

    baseline_pred = int(
        r_sample[
            "baseline_best_numeral"
        ].iloc[0]
    )

    baseline_correct = int(
        baseline_pred == gt
    )

    for action_name, alpha in (
        ACTIONS.items()
    ):

        row = {
            "pair_id":
                pair_id,

            "sample_id":
                sid,

            "role":
                role,

            "action":
                action_name,

            "alpha":
                float(alpha),

            # Evaluation-only fields.
            "ground_truth":
                gt,

            "baseline_correct":
                baseline_correct,
        }

        # Passive GT-free state.
        for c in baseline_features:
            row[c] = float(
                f[c]
            )

        if action_name == "noop":

            # Identity response.
            for n in range(1, 11):
                row[
                    f"action_dp{n}"
                ] = 0.0

            row[
                "action_response_l1"
            ] = 0.0

            row[
                "action_response_l2"
            ] = 0.0

            row[
                "action_response_tv"
            ] = 0.0

            row[
                "action_response_js"
            ] = 0.0

            row[
                "action_expected_shift"
            ] = 0.0

            row[
                "action_delta_entropy"
            ] = 0.0

            row[
                "action_delta_entropy_norm"
            ] = 0.0

            row[
                "action_delta_vocab_mass"
            ] = 0.0

            action_pred = (
                baseline_pred
            )

        else:

            rr = r_sample[
                np.isclose(
                    r_sample[
                        "alpha"
                    ],
                    alpha,
                )
            ]

            assert len(rr) == 1

            rr = rr.iloc[0]

            for n in range(1, 11):
                row[
                    f"action_dp{n}"
                ] = float(
                    rr[
                        f"delta_numprob_{n}"
                    ]
                )

            row[
                "action_response_l1"
            ] = float(
                rr[
                    "response_l1"
                ]
            )

            row[
                "action_response_l2"
            ] = float(
                rr[
                    "response_l2"
                ]
            )

            row[
                "action_response_tv"
            ] = float(
                rr[
                    "response_tv"
                ]
            )

            row[
                "action_response_js"
            ] = float(
                rr[
                    "response_js"
                ]
            )

            row[
                "action_expected_shift"
            ] = float(
                rr[
                    "response_expected_numeral_shift"
                ]
            )

            row[
                "action_delta_entropy"
            ] = float(
                rr[
                    "delta_entropy"
                ]
            )

            row[
                "action_delta_entropy_norm"
            ] = float(
                rr[
                    "delta_entropy_norm"
                ]
            )

            row[
                "action_delta_vocab_mass"
            ] = float(
                rr[
                    "delta_vocab_numeral_mass"
                ]
            )

            action_pred = int(
                rr[
                    "modulated_best_numeral"
                ]
            )

        action_correct = int(
            action_pred == gt
        )

        row[
            "action_prediction"
        ] = action_pred

        row[
            "action_correct"
        ] = action_correct

        row[
            "utility"
        ] = (
            action_correct
            - baseline_correct
        )

        rows.append(row)


data = pd.DataFrame(
    rows
)

assert len(data) == (
    54 * 3
)

assert (
    data.groupby(
        "sample_id"
    )["action"]
    .nunique()
    == 3
).all()

assert data.isna().sum().sum() == 0

print(
    "Action rows:",
    len(data),
)

print(
    "Pairs:",
    data[
        "pair_id"
    ].nunique(),
)

print("\nUtility counts:")
print(
    data[
        "utility"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# Inference-safe numeric features
# ============================================================

action_response_features = [
    f"action_dp{n}"
    for n in range(1, 11)
] + [
    "action_response_l1",
    "action_response_l2",
    "action_response_tv",
    "action_response_js",
    "action_expected_shift",
    "action_delta_entropy",
    "action_delta_entropy_norm",
    "action_delta_vocab_mass",
]


numeric_features = (
    baseline_features
    + action_response_features
)


forbidden = [
    "ground_truth",
    "role",
    "baseline_correct",
    "action_correct",
    "utility",
]

assert not (
    set(numeric_features)
    & set(forbidden)
)


# ============================================================
# We predict probability that candidate action yields a
# CORRECT final answer.
#
# All GT information is training target only.
# ============================================================

def make_model(C):

    pre = ColumnTransformer([
        (
            "num",
            StandardScaler(),
            numeric_features,
        ),
        (
            "action",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            ["action"],
        ),
    ])

    return Pipeline([
        (
            "pre",
            pre,
        ),
        (
            "clf",
            LogisticRegression(
                C=C,
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=10000,
                random_state=SEED,
            ),
        ),
    ])


def policy_score(
    frame,
    prob_col,
):

    selected = []

    for sid, g in (
        frame.groupby(
            "sample_id"
        )
    ):

        # Deterministic tie preference:
        # NO-OP is safest.
        order = {
            "noop": 0,
            "gain_0p5": 1,
            "gain_1p5": 2,
        }

        gg = g.copy()

        gg[
            "_tie"
        ] = (
            gg[
                "action"
            ]
            .map(order)
        )

        best = (
            gg.sort_values(
                [
                    prob_col,
                    "_tie",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

        selected.append(best)

    selected = pd.DataFrame(
        selected
    )

    return float(
        selected[
            "action_correct"
        ].mean()
    )


# ============================================================
# Inner C selection
# ============================================================

def choose_c(
    train_df,
):

    groups = (
        train_df[
            "pair_id"
        ]
        .to_numpy()
    )

    unique_groups = np.unique(
        groups
    )

    splitter = GroupKFold(
        n_splits=min(
            5,
            len(
                unique_groups
            ),
        )
    )

    X = train_df[
        numeric_features
        + ["action"]
    ]

    y = train_df[
        "action_correct"
    ].astype(int).to_numpy()

    candidates = []

    for C in C_GRID:

        probs = np.full(
            len(train_df),
            np.nan,
        )

        ok = True

        for tr, va in (
            splitter.split(
                X,
                y,
                groups=groups,
            )
        ):

            model = make_model(
                C
            )

            try:

                model.fit(
                    X.iloc[tr],
                    y[tr],
                )

                probs[va] = (
                    model.predict_proba(
                        X.iloc[va]
                    )[:, 1]
                )

            except Exception:
                ok = False
                break

        if (
            (not ok)
            or np.isnan(
                probs
            ).any()
        ):
            continue

        tmp = (
            train_df.copy()
            .reset_index(
                drop=True
            )
        )

        tmp[
            "_prob"
        ] = probs

        score = policy_score(
            tmp,
            "_prob",
        )

        candidates.append(
            (
                C,
                score,
            )
        )

    if not candidates:
        return 0.1

    candidates.sort(
        key=lambda z: (
            -z[1],
            z[0],
        )
    )

    return candidates[0][0]


# ============================================================
# Outer Leave-One-Pair-Out
# ============================================================

logo = LeaveOneGroupOut()

pair_groups = (
    data[
        "pair_id"
    ].to_numpy()
)

data[
    "oof_correct_probability"
] = np.nan

data[
    "chosen_C"
] = np.nan


for fold, (
    train_idx,
    test_idx,
) in enumerate(
    logo.split(
        data,
        groups=pair_groups,
    ),
    start=1,
):

    train_df = (
        data.iloc[
            train_idx
        ]
        .copy()
    )

    test_df = (
        data.iloc[
            test_idx
        ]
        .copy()
    )

    best_C = choose_c(
        train_df
    )

    model = make_model(
        best_C
    )

    X_train = train_df[
        numeric_features
        + ["action"]
    ]

    y_train = train_df[
        "action_correct"
    ].astype(int)

    X_test = test_df[
        numeric_features
        + ["action"]
    ]

    model.fit(
        X_train,
        y_train,
    )

    probs = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    data.loc[
        test_idx,
        "oof_correct_probability",
    ] = probs

    data.loc[
        test_idx,
        "chosen_C",
    ] = best_C


assert not data[
    "oof_correct_probability"
].isna().any()


# ============================================================
# Select OOF action
# ============================================================

selected_rows = []

for sid, g in (
    data.groupby(
        "sample_id"
    )
):

    # Prefer NO-OP on exact ties.
    tie_order = {
        "noop": 0,
        "gain_0p5": 1,
        "gain_1p5": 2,
    }

    gg = g.copy()

    gg["_tie"] = (
        gg[
            "action"
        ]
        .map(
            tie_order
        )
    )

    best = (
        gg.sort_values(
            [
                "oof_correct_probability",
                "_tie",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    selected_rows.append(
        best
    )


selected = pd.DataFrame(
    selected_rows
)

assert len(selected) == 54


# ============================================================
# Evaluate learned policy
# ============================================================

baseline_correct = (
    selected[
        "baseline_correct"
    ].astype(bool)
)

post_correct = (
    selected[
        "action_correct"
    ].astype(bool)
)

repairs = int(
    (
        (~baseline_correct)
        & post_correct
    ).sum()
)

breaks = int(
    (
        baseline_correct
        & (~post_correct)
    ).sum()
)

baseline_acc = float(
    baseline_correct.mean()
)

post_acc = float(
    post_correct.mean()
)


# ============================================================
# Fixed alpha=1.5 baseline
# ============================================================

fixed = data[
    data["action"]
    == "gain_1p5"
].copy()

fixed_repairs = int(
    (
        (fixed["baseline_correct"] == 0)
        &
        (fixed["action_correct"] == 1)
    ).sum()
)

fixed_breaks = int(
    (
        (fixed["baseline_correct"] == 1)
        &
        (fixed["action_correct"] == 0)
    ).sum()
)

fixed_acc = float(
    fixed[
        "action_correct"
    ].mean()
)


# ============================================================
# Oracle over the three actions
# Analysis upper bound only.
# ============================================================

oracle_correct = []

for sid, g in (
    data.groupby(
        "sample_id"
    )
):

    oracle_correct.append(
        int(
            g[
                "action_correct"
            ].max()
        )
    )

oracle_acc = float(
    np.mean(
        oracle_correct
    )
)


# ============================================================
# Save
# ============================================================

selected.to_csv(
    PRED_PATH,
    index=False,
)

summary = pd.DataFrame([
    {
        "policy":
            "baseline_noop",

        "accuracy":
            baseline_acc,

        "repairs":
            0,

        "breaks":
            0,

        "net_repairs":
            0,
    },
    {
        "policy":
            "fixed_L18H13_a1p5",

        "accuracy":
            fixed_acc,

        "repairs":
            fixed_repairs,

        "breaks":
            fixed_breaks,

        "net_repairs":
            fixed_repairs
            - fixed_breaks,
    },
    {
        "policy":
            "learned_gain_gate",

        "accuracy":
            post_acc,

        "repairs":
            repairs,

        "breaks":
            breaks,

        "net_repairs":
            repairs
            - breaks,
    },
    {
        "policy":
            "oracle_three_actions",

        "accuracy":
            oracle_acc,

        "repairs":
            int(
                round(
                    (
                        oracle_acc
                        - baseline_acc
                    )
                    * 54
                )
            ),

        "breaks":
            0,

        "net_repairs":
            int(
                round(
                    (
                        oracle_acc
                        - baseline_acc
                    )
                    * 54
                )
            ),
    },
])

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# Report
# ============================================================

print("\n" + "=" * 120)
print("FINAL LEARNED GAIN-GATE RESULTS")
print("=" * 120)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)

print(
    "\nLearned action distribution:"
)

print(
    selected[
        "action"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nWrong-sample actions:"
)

print(
    selected.loc[
        selected[
            "baseline_correct"
        ]
        == 0,
        "action",
    ]
    .value_counts()
    .to_string()
)

print(
    "\nCorrect-sample actions:"
)

print(
    selected.loc[
        selected[
            "baseline_correct"
        ]
        == 1,
        "action",
    ]
    .value_counts()
    .to_string()
)

print("\nSaved:")
print(PRED_PATH)
print(SUMMARY_PATH)

print(
    "\nLEARNED GAIN GATE COMPLETE"
)
