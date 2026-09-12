from pathlib import Path
from itertools import product
from math import comb

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CONF_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5_expanded_0_15/"
    "confirmation_results.csv"
)

MATRIX_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/"
    "compressed_action_matrix/"
    "compressed_action_matrix.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "controller/"
    "baseline_utility_cv"
)

OOF_PATH = (
    OUT_DIR /
    "baseline_utility_oof_predictions.csv"
)

FOLD_PATH = (
    OUT_DIR /
    "baseline_utility_fold_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR /
    "baseline_utility_summary.csv"
)

CONDITION_PATH = (
    OUT_DIR /
    "baseline_utility_by_condition.csv"
)


ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]

NONNOOP = [
    0.0,
    1.5,
    2.0,
    4.0,
]


RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

THRESHOLDS = [
    0.0,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
]


def exact_mcnemar(
    repairs,
    breaks,
):
    repairs = int(repairs)
    breaks = int(breaks)

    n = repairs + breaks

    if n == 0:
        return 1.0

    k = min(
        repairs,
        breaks,
    )

    tail = sum(
        comb(n, i)
        for i in range(
            k + 1
        )
    )

    return min(
        1.0,
        2.0
        * tail
        / (2 ** n),
    )


def add_geometry_features(
    df,
):

    out = df.copy()

    prob_cols = [
        f"baseline_numprob_{n}"
        for n in range(16)
    ]

    probs = (
        out[
            prob_cols
        ]
        .to_numpy(
            dtype=float
        )
    )

    pred = (
        out[
            "baseline_best_numeral"
        ]
        .astype(int)
        .to_numpy()
    )

    n_rows = len(out)

    p_pred = np.zeros(
        n_rows,
        dtype=float,
    )

    p_down1 = np.zeros(
        n_rows,
        dtype=float,
    )

    p_up1 = np.zeros(
        n_rows,
        dtype=float,
    )

    p_down2 = np.zeros(
        n_rows,
        dtype=float,
    )

    p_up2 = np.zeros(
        n_rows,
        dtype=float,
    )

    down_mass = np.zeros(
        n_rows,
        dtype=float,
    )

    up_mass = np.zeros(
        n_rows,
        dtype=float,
    )

    down_distance_mass = np.zeros(
        n_rows,
        dtype=float,
    )

    up_distance_mass = np.zeros(
        n_rows,
        dtype=float,
    )

    top2_prob = np.zeros(
        n_rows,
        dtype=float,
    )

    for i in range(
        n_rows
    ):
        k = int(
            pred[i]
        )

        p = probs[i]

        p_pred[i] = p[k]

        order = np.sort(
            p
        )[::-1]

        top2_prob[i] = (
            order[1]
        )

        if k >= 1:
            p_down1[i] = (
                p[k - 1]
            )

        if k <= 14:
            p_up1[i] = (
                p[k + 1]
            )

        if k >= 2:
            p_down2[i] = (
                p[k - 2]
            )

        if k <= 13:
            p_up2[i] = (
                p[k + 2]
            )

        if k > 0:
            down_mass[i] = (
                p[:k].sum()
            )

            down_distance_mass[i] = sum(
                (k - n)
                * p[n]
                for n in range(
                    0,
                    k
                )
            )

        if k < 15:
            up_mass[i] = (
                p[
                    k + 1:
                ].sum()
            )

            up_distance_mass[i] = sum(
                (n - k)
                * p[n]
                for n in range(
                    k + 1,
                    16
                )
            )

    eps = 1e-12

    out[
        "geom_top2_prob"
    ] = top2_prob

    out[
        "geom_expected_minus_pred"
    ] = (
        out[
            "baseline_expected_numeral"
        ].to_numpy(
            dtype=float
        )
        -
        pred
    )

    out[
        "geom_p_down1"
    ] = p_down1

    out[
        "geom_p_up1"
    ] = p_up1

    out[
        "geom_p_down2"
    ] = p_down2

    out[
        "geom_p_up2"
    ] = p_up2

    out[
        "geom_local_asymmetry"
    ] = (
        p_up1
        -
        p_down1
    )

    out[
        "geom_up_mass"
    ] = up_mass

    out[
        "geom_down_mass"
    ] = down_mass

    out[
        "geom_signed_tail_mass"
    ] = (
        up_mass
        -
        down_mass
    )

    out[
        "geom_up_distance_mass"
    ] = (
        up_distance_mass
    )

    out[
        "geom_down_distance_mass"
    ] = (
        down_distance_mass
    )

    out[
        "geom_signed_distance_mass"
    ] = (
        up_distance_mass
        -
        down_distance_mass
    )

    out[
        "geom_up_boundary_gap"
    ] = (
        p_pred
        -
        p_up1
    )

    out[
        "geom_down_boundary_gap"
    ] = (
        p_pred
        -
        p_down1
    )

    out[
        "geom_up_down_logratio"
    ] = np.log(
        (
            p_up1
            +
            eps
        )
        /
        (
            p_down1
            +
            eps
        )
    )

    return out


def build_feature_families(
    df,
):

    prob_cols = [
        f"baseline_numprob_{n}"
        for n in range(16)
    ]

    compact = [
        "baseline_best_numeral",
        "baseline_entropy_norm",
        "baseline_expected_numeral",
        "baseline_top1_prob",
        "baseline_conditional_margin",
        "baseline_vocab_numeral_mass",
        "geom_top2_prob",
        "geom_expected_minus_pred",
        "geom_p_down1",
        "geom_p_up1",
        "geom_local_asymmetry",
        "geom_up_mass",
        "geom_down_mass",
        "geom_signed_tail_mass",
        "geom_up_boundary_gap",
        "geom_down_boundary_gap",
        "geom_up_down_logratio",
    ]

    distribution = [
        "baseline_best_numeral",
        "baseline_expected_numeral",
        "baseline_top1_prob",
        "baseline_conditional_margin",
    ] + prob_cols

    full = [
        "baseline_best_numeral",
        "baseline_entropy",
        "baseline_entropy_norm",
        "baseline_expected_numeral",
        "baseline_top1_prob",
        "baseline_conditional_margin",
        "baseline_vocab_numeral_mass",
    ] + prob_cols + [
        "geom_top2_prob",
        "geom_expected_minus_pred",
        "geom_p_down1",
        "geom_p_up1",
        "geom_p_down2",
        "geom_p_up2",
        "geom_local_asymmetry",
        "geom_up_mass",
        "geom_down_mass",
        "geom_signed_tail_mass",
        "geom_up_distance_mass",
        "geom_down_distance_mass",
        "geom_signed_distance_mass",
        "geom_up_boundary_gap",
        "geom_down_boundary_gap",
        "geom_up_down_logratio",
    ]

    return {
        "compact_geometry":
            compact,

        "distribution":
            distribution,

        "full_geometry":
            full,
    }


def make_models(
    X,
    utilities,
    ridge_alpha,
):

    models = {}

    for action in NONNOOP:

        y = (
            utilities[
                action
            ]
            .to_numpy(
                dtype=float
            )
        )

        model = Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=float(
                        ridge_alpha
                    )
                ),
            ),
        ])

        model.fit(
            X,
            y,
        )

        models[
            action
        ] = model

    return models


def predict_scores(
    models,
    X,
):

    result = np.zeros(
        (
            len(X),
            len(NONNOOP),
        ),
        dtype=float,
    )

    for j, action in enumerate(
        NONNOOP
    ):

        result[
            :,
            j
        ] = (
            models[
                action
            ]
            .predict(
                X
            )
        )

    return result


def choose_actions(
    scores,
    threshold,
):

    best_idx = np.argmax(
        scores,
        axis=1,
    )

    best_score = scores[
        np.arange(
            len(scores)
        ),
        best_idx,
    ]

    selected = np.full(
        len(scores),
        1.0,
        dtype=float,
    )

    use = (
        best_score
        >
        float(
            threshold
        )
    )

    actions_arr = np.asarray(
        NONNOOP,
        dtype=float,
    )

    selected[
        use
    ] = actions_arr[
        best_idx[
            use
        ]
    ]

    return (
        selected,
        best_score,
    )


def selected_utilities(
    utilities,
    actions,
):

    result = np.zeros(
        len(actions),
        dtype=int,
    )

    for i, action in enumerate(
        actions
    ):

        if np.isclose(
            action,
            1.0,
        ):
            result[i] = 0
        else:
            result[i] = int(
                utilities.iloc[i][
                    action
                ]
            )

    return result


def evaluate_policy(
    baseline_correct,
    utility,
    actions,
):

    baseline_correct = np.asarray(
        baseline_correct,
        dtype=bool,
    )

    utility = np.asarray(
        utility,
        dtype=int,
    )

    post_correct = (
        baseline_correct.astype(int)
        +
        utility
    )

    if not np.isin(
        post_correct,
        [
            0,
            1,
        ],
    ).all():
        raise RuntimeError(
            "Invalid post-correct values."
        )

    repairs = int(
        (
            utility == 1
        ).sum()
    )

    breaks = int(
        (
            utility == -1
        ).sum()
    )

    intervention_rate = float(
        (
            ~np.isclose(
                actions,
                1.0,
            )
        ).mean()
    )

    return {
        "mean_utility":
            float(
                utility.mean()
            ),

        "accuracy":
            float(
                post_correct.mean()
            ),

        "repairs":
            repairs,

        "breaks":
            breaks,

        "net_repairs":
            repairs
            -
            breaks,

        "intervention_rate":
            intervention_rate,

        "post_correct":
            post_correct,
    }


def tune_inner(
    X,
    utilities,
    baseline_correct,
    groups,
):

    inner = GroupKFold(
        n_splits=4
    )

    best = None

    for (
        ridge_alpha,
        threshold
    ) in product(
        RIDGE_ALPHAS,
        THRESHOLDS,
    ):

        fold_utilities = []
        fold_interventions = []
        fold_breaks = []

        for (
            train_idx,
            val_idx
        ) in inner.split(
            X,
            groups=groups,
        ):

            X_train = X[
                train_idx
            ]

            X_val = X[
                val_idx
            ]

            U_train = (
                utilities.iloc[
                    train_idx
                ]
                .reset_index(
                    drop=True
                )
            )

            U_val = (
                utilities.iloc[
                    val_idx
                ]
                .reset_index(
                    drop=True
                )
            )

            models = make_models(
                X_train,
                U_train,
                ridge_alpha,
            )

            scores = predict_scores(
                models,
                X_val,
            )

            actions, _ = (
                choose_actions(
                    scores,
                    threshold,
                )
            )

            selected_u = (
                selected_utilities(
                    U_val,
                    actions,
                )
            )

            metrics = evaluate_policy(
                baseline_correct[
                    val_idx
                ],
                selected_u,
                actions,
            )

            fold_utilities.append(
                metrics[
                    "mean_utility"
                ]
            )

            fold_interventions.append(
                metrics[
                    "intervention_rate"
                ]
            )

            fold_breaks.append(
                metrics[
                    "breaks"
                ]
            )

        candidate = {
            "ridge_alpha":
                float(
                    ridge_alpha
                ),

            "threshold":
                float(
                    threshold
                ),

            "mean_utility":
                float(
                    np.mean(
                        fold_utilities
                    )
                ),

            "mean_intervention_rate":
                float(
                    np.mean(
                        fold_interventions
                    )
                ),

            "total_breaks":
                int(
                    np.sum(
                        fold_breaks
                    )
                ),
        }

        if best is None:

            best = candidate

        else:

            current_key = (
                candidate[
                    "mean_utility"
                ],
                -candidate[
                    "total_breaks"
                ],
                -candidate[
                    "mean_intervention_rate"
                ],
                candidate[
                    "threshold"
                ],
            )

            best_key = (
                best[
                    "mean_utility"
                ],
                -best[
                    "total_breaks"
                ],
                -best[
                    "mean_intervention_rate"
                ],
                best[
                    "threshold"
                ],
            )

            if current_key > best_key:
                best = candidate

    return best


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print(
        "AROMA V2 GT-FREE BASELINE "
        "UTILITY CONTROLLER"
    )
    print("=" * 112)

    conf = pd.read_csv(
        CONF_PATH
    )

    base = conf[
        np.isclose(
            conf[
                "alpha"
            ].astype(float),
            1.0,
        )
    ].copy()

    base = (
        base.sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(base) != 1000:
        raise RuntimeError(
            f"Expected 1000 baseline rows, "
            f"found {len(base)}."
        )

    matrix = pd.read_csv(
        MATRIX_PATH
    )

    if len(matrix) != 5000:
        raise RuntimeError(
            f"Expected 5000 matrix rows, "
            f"found {len(matrix)}."
        )

    # --------------------------------------------------------
    # Utilities per sample/action
    # --------------------------------------------------------

    utility = (
        matrix.pivot(
            index="sample_id",
            columns="alpha",
            values="utility",
        )
        .reindex(
            base[
                "sample_id"
            ]
        )
    )

    for action in ACTIONS:

        if action not in utility.columns:
            raise RuntimeError(
                f"Missing utility action "
                f"{action}."
            )

    utility = utility[
        ACTIONS
    ].copy()

    utility.columns = [
        float(c)
        for c in utility.columns
    ]

    utility = utility.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Baseline correctness is EVALUATION ONLY
    # --------------------------------------------------------

    baseline_correct = (
        base[
            "baseline_correct"
        ]
        .astype(bool)
        .to_numpy()
    )

    ground_truth = (
        base[
            "ground_truth"
        ]
        .astype(int)
        .to_numpy()
    )

    condition = (
        base[
            "condition"
        ]
        .astype(str)
        .to_numpy()
    )

    groups = (
        base[
            "replicate"
        ]
        .astype(int)
        .to_numpy()
    )

    if len(
        np.unique(
            groups
        )
    ) != 20:

        raise RuntimeError(
            "Expected 20 replicate groups."
        )

    # --------------------------------------------------------
    # Oracle metadata for analysis only
    # --------------------------------------------------------

    oracle_action = np.ones(
        len(base),
        dtype=float,
    )

    oracle_repairable = np.zeros(
        len(base),
        dtype=bool,
    )

    for i in range(
        len(base)
    ):

        if baseline_correct[i]:
            continue

        successful = [
            a
            for a in NONNOOP
            if int(
                utility.iloc[i][a]
            )
            == 1
        ]

        if successful:

            oracle_repairable[i] = True

            oracle_action[i] = min(
                successful,
                key=lambda a:
                    (
                        abs(
                            a - 1.0
                        ),
                        a,
                    ),
            )

    print(
        "Samples:",
        len(base),
    )

    print(
        "Replicate groups:",
        len(
            np.unique(
                groups
            )
        ),
    )

    print(
        "Baseline accuracy:",
        float(
            baseline_correct.mean()
        ),
    )

    print(
        "Compressed oracle repairable:",
        int(
            oracle_repairable.sum()
        ),
    )

    print(
        "Compressed oracle accuracy:",
        float(
            (
                baseline_correct.sum()
                +
                oracle_repairable.sum()
            )
            /
            len(base)
        ),
    )

    # --------------------------------------------------------
    # GT-free features
    # --------------------------------------------------------

    features_df = (
        add_geometry_features(
            base
        )
    )

    families = (
        build_feature_families(
            features_df
        )
    )

    forbidden = {
        "ground_truth",
        "condition",
        "replicate",
        "baseline_correct",
        "baseline_gt_logp",
        "baseline_margin",
        "modulated_correct",
        "modulated_gt_logp",
        "modulated_margin",
        "delta_gt_logp",
        "delta_margin",
    }

    for family, cols in (
        families.items()
    ):

        overlap = (
            forbidden
            &
            set(cols)
        )

        if overlap:

            raise RuntimeError(
                f"Leakage in {family}: "
                f"{sorted(overlap)}"
            )

        if (
            features_df[
                cols
            ]
            .isna()
            .any()
            .any()
        ):

            raise RuntimeError(
                f"Missing feature values "
                f"in {family}."
            )

    print(
        "\nFeature families:"
    )

    for name, cols in (
        families.items()
    ):
        print(
            f"  {name:20s}: "
            f"{len(cols)}"
        )

    print(
        "\nLeakage audit: PASS"
    )

    # --------------------------------------------------------
    # Outer grouped CV
    # --------------------------------------------------------

    outer = GroupKFold(
        n_splits=5
    )

    oof_rows = []
    fold_rows = []

    for family_name, cols in (
        families.items()
    ):

        print(
            "\n" + "=" * 112
        )

        print(
            "FEATURE FAMILY:",
            family_name,
        )

        print(
            "=" * 112
        )

        X_all = (
            features_df[
                cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        for fold, (
            train_idx,
            test_idx
        ) in enumerate(
            outer.split(
                X_all,
                groups=groups,
            ),
            start=1,
        ):

            X_train = X_all[
                train_idx
            ]

            X_test = X_all[
                test_idx
            ]

            U_train = (
                utility.iloc[
                    train_idx
                ]
                .reset_index(
                    drop=True
                )
            )

            U_test = (
                utility.iloc[
                    test_idx
                ]
                .reset_index(
                    drop=True
                )
            )

            best = tune_inner(
                X=X_train,
                utilities=U_train,
                baseline_correct=
                    baseline_correct[
                        train_idx
                    ],
                groups=groups[
                    train_idx
                ],
            )

            models = make_models(
                X_train,
                U_train,
                best[
                    "ridge_alpha"
                ],
            )

            scores = predict_scores(
                models,
                X_test,
            )

            actions, best_score = (
                choose_actions(
                    scores,
                    best[
                        "threshold"
                    ],
                )
            )

            selected_u = (
                selected_utilities(
                    U_test,
                    actions,
                )
            )

            metrics = evaluate_policy(
                baseline_correct[
                    test_idx
                ],
                selected_u,
                actions,
            )

            print(
                f"Fold {fold}: "
                f"ridge={best['ridge_alpha']}, "
                f"threshold={best['threshold']}, "
                f"repairs={metrics['repairs']}, "
                f"breaks={metrics['breaks']}, "
                f"net={metrics['net_repairs']}, "
                f"acc={metrics['accuracy']:.4f}, "
                f"intervene="
                f"{metrics['intervention_rate']:.3f}"
            )

            fold_rows.append({
                "family":
                    family_name,

                "fold":
                    fold,

                "ridge_alpha":
                    best[
                        "ridge_alpha"
                    ],

                "threshold":
                    best[
                        "threshold"
                    ],

                "inner_mean_utility":
                    best[
                        "mean_utility"
                    ],

                "test_repairs":
                    metrics[
                        "repairs"
                    ],

                "test_breaks":
                    metrics[
                        "breaks"
                    ],

                "test_net_repairs":
                    metrics[
                        "net_repairs"
                    ],

                "test_accuracy":
                    metrics[
                        "accuracy"
                    ],

                "test_intervention_rate":
                    metrics[
                        "intervention_rate"
                    ],
            })

            for local_i, global_i in enumerate(
                test_idx
            ):

                row = {
                    "family":
                        family_name,

                    "fold":
                        fold,

                    "sample_id":
                        str(
                            base.iloc[
                                global_i
                            ][
                                "sample_id"
                            ]
                        ),

                    "replicate":
                        int(
                            groups[
                                global_i
                            ]
                        ),

                    "ground_truth":
                        int(
                            ground_truth[
                                global_i
                            ]
                        ),

                    "condition":
                        str(
                            condition[
                                global_i
                            ]
                        ),

                    "baseline_pred":
                        int(
                            base.iloc[
                                global_i
                            ][
                                "baseline_best_numeral"
                            ]
                        ),

                    "baseline_correct":
                        bool(
                            baseline_correct[
                                global_i
                            ]
                        ),

                    "selected_alpha":
                        float(
                            actions[
                                local_i
                            ]
                        ),

                    "selected_score":
                        float(
                            best_score[
                                local_i
                            ]
                        ),

                    "selected_utility":
                        int(
                            selected_u[
                                local_i
                            ]
                        ),

                    "post_correct":
                        int(
                            metrics[
                                "post_correct"
                            ][
                                local_i
                            ]
                        ),

                    "oracle_repairable":
                        bool(
                            oracle_repairable[
                                global_i
                            ]
                        ),

                    "oracle_alpha":
                        float(
                            oracle_action[
                                global_i
                            ]
                        ),
                }

                for j, action in enumerate(
                    NONNOOP
                ):

                    row[
                        f"score_alpha_{action}"
                    ] = float(
                        scores[
                            local_i,
                            j
                        ]
                    )

                oof_rows.append(
                    row
                )

    # --------------------------------------------------------
    # Save OOF
    # --------------------------------------------------------

    oof = pd.DataFrame(
        oof_rows
    )

    folds = pd.DataFrame(
        fold_rows
    )

    if len(oof) != (
        1000
        *
        len(families)
    ):

        raise RuntimeError(
            f"OOF rows={len(oof)}, "
            f"unexpected."
        )

    oof.to_csv(
        OOF_PATH,
        index=False,
    )

    folds.to_csv(
        FOLD_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Final summaries
    # --------------------------------------------------------

    summary_rows = []

    condition_rows = []

    for family, g in oof.groupby(
        "family"
    ):

        repairs = int(
            (
                g[
                    "selected_utility"
                ]
                ==
                1
            ).sum()
        )

        breaks = int(
            (
                g[
                    "selected_utility"
                ]
                ==
                -1
            ).sum()
        )

        post_accuracy = float(
            g[
                "post_correct"
            ].mean()
        )

        intervention_rate = float(
            (
                ~np.isclose(
                    g[
                        "selected_alpha"
                    ],
                    1.0,
                )
            ).mean()
        )

        summary_rows.append({
            "family":
                family,

            "n_features":
                len(
                    families[
                        family
                    ]
                ),

            "baseline_accuracy":
                0.503,

            "post_accuracy":
                post_accuracy,

            "accuracy_gain":
                post_accuracy
                -
                0.503,

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                repairs
                -
                breaks,

            "intervention_rate":
                intervention_rate,

            "repair_capture_fraction":
                repairs
                /
                138.0,

            "oracle_accuracy":
                0.641,

            "gap_to_oracle":
                0.641
                -
                post_accuracy,

            "mcnemar_exact_p":
                exact_mcnemar(
                    repairs,
                    breaks,
                ),
        })

        for cond, h in g.groupby(
            "condition"
        ):

            base_acc = float(
                h[
                    "baseline_correct"
                ].mean()
            )

            post_acc = float(
                h[
                    "post_correct"
                ].mean()
            )

            r = int(
                (
                    h[
                        "selected_utility"
                    ]
                    ==
                    1
                ).sum()
            )

            b = int(
                (
                    h[
                        "selected_utility"
                    ]
                    ==
                    -1
                ).sum()
            )

            condition_rows.append({
                "family":
                    family,

                "condition":
                    cond,

                "n":
                    len(h),

                "baseline_accuracy":
                    base_acc,

                "post_accuracy":
                    post_acc,

                "accuracy_gain":
                    post_acc
                    -
                    base_acc,

                "repairs":
                    r,

                "breaks":
                    b,

                "net_repairs":
                    r - b,
            })

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "post_accuracy",
                "net_repairs",
            ],
            ascending=False,
        )
    )

    by_condition = (
        pd.DataFrame(
            condition_rows
        )
        .sort_values(
            [
                "family",
                "condition",
            ]
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    by_condition.to_csv(
        CONDITION_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "FINAL OOF CONTROLLER RESULTS"
    )

    print(
        "=" * 112
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    best_family = str(
        summary.iloc[0][
            "family"
        ]
    )

    best = oof[
        oof[
            "family"
        ]
        ==
        best_family
    ]

    print(
        "\nBest family:",
        best_family,
    )

    print(
        "\nSelected action distribution:"
    )

    print(
        best[
            "selected_alpha"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nBest-family results by condition:"
    )

    print(
        by_condition[
            by_condition[
                "family"
            ]
            ==
            best_family
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nSaved:"
    )

    print(
        OOF_PATH
    )

    print(
        FOLD_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        CONDITION_PATH
    )

    print(
        "\nGT-FREE BASELINE UTILITY "
        "CONTROLLER COMPLETE"
    )


if __name__ == "__main__":
    main()
