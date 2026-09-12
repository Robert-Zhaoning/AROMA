import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


sys.path.insert(0, "scripts")

from run_v2_baseline_utility_controller import (
    ACTIONS,
    NONNOOP,
    RIDGE_ALPHAS,
    THRESHOLDS,
    add_geometry_features,
    build_feature_families,
    make_models,
    predict_scores,
    choose_actions,
    selected_utilities,
    evaluate_policy,
    exact_mcnemar,
)


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
    "final_frozen_controller"
)

BUNDLE_PATH = (
    OUT_DIR /
    "aroma_cardinality_controller.joblib"
)

GRID_PATH = (
    OUT_DIR /
    "final_hyperparameter_grid.csv"
)

OOF_PATH = (
    OUT_DIR /
    "selected_hyperparameter_oof.csv"
)

MANIFEST_PATH = Path(
    "configs/"
    "aroma_cardinality_controller_frozen.json"
)


PRIMARY_FAMILY = "full_geometry"

MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

HEAD_LAYER = 18
HEAD_INDEX = 13
HEAD_NAME = "L18H13"

N_SPLITS = 5


def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print(
        "AROMA FINAL CONTROLLER "
        "FREEZE — V2 DEVELOPMENT"
    )
    print("=" * 112)

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script performs DEVELOPMENT-SET "
        "hyperparameter selection only."
    )

    print(
        "No v3/final-confirmation data may be "
        "used here."
    )

    # ========================================================
    # Load baseline state
    # ========================================================

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

    if (
        base[
            "sample_id"
        ].nunique()
        != 1000
    ):

        raise RuntimeError(
            "Baseline sample IDs are not unique."
        )

    # ========================================================
    # Load counterfactual utilities
    # ========================================================

    matrix = pd.read_csv(
        MATRIX_PATH
    )

    if len(matrix) != 5000:

        raise RuntimeError(
            f"Expected 5000 action rows, "
            f"found {len(matrix)}."
        )

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
                f"Missing action {action} "
                "from utility matrix."
            )

    utility = (
        utility[
            ACTIONS
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    utility.columns = [
        float(c)
        for c in utility.columns
    ]

    # ========================================================
    # Evaluation metadata
    # ========================================================

    baseline_correct = (
        base[
            "baseline_correct"
        ]
        .astype(bool)
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
            "Expected exactly 20 replicate groups."
        )

    # ========================================================
    # Feature construction
    # ========================================================

    feature_df = (
        add_geometry_features(
            base
        )
    )

    families = (
        build_feature_families(
            feature_df
        )
    )

    if PRIMARY_FAMILY not in families:

        raise RuntimeError(
            f"Missing feature family "
            f"{PRIMARY_FAMILY}."
        )

    feature_names = (
        families[
            PRIMARY_FAMILY
        ]
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
        "utility",
        "repair",
        "break_case",
        "oracle_alpha",
    }

    leaked = (
        forbidden
        &
        set(
            feature_names
        )
    )

    if leaked:

        raise RuntimeError(
            "LEAKAGE DETECTED: "
            f"{sorted(leaked)}"
        )

    X = (
        feature_df[
            feature_names
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(
            "Non-finite feature value detected."
        )

    print(
        "\nSamples          :",
        len(base),
    )

    print(
        "Replicate groups :",
        len(
            np.unique(
                groups
            )
        ),
    )

    print(
        "Feature family   :",
        PRIMARY_FAMILY,
    )

    print(
        "Feature count    :",
        len(
            feature_names
        ),
    )

    print(
        "Actions          :",
        ACTIONS,
    )

    print(
        "Leakage audit    : PASS"
    )

    # ========================================================
    # Final hyperparameter selection rule
    #
    # This is now the frozen rule:
    #
    # 1. 5-fold GroupKFold by replicate.
    # 2. Maximize total OOF net repairs.
    # 3. Tie: fewer breaks.
    # 4. Tie: fewer interventions.
    # 5. Tie: higher threshold.
    # 6. Tie: larger Ridge regularization.
    #
    # No v3 information is used.
    # ========================================================

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    grid_rows = []

    best_candidate = None
    best_key = None

    selected_candidate_oof = None

    for ridge_alpha in RIDGE_ALPHAS:

        for threshold in THRESHOLDS:

            oof_rows = []

            total_repairs = 0
            total_breaks = 0
            total_interventions = 0

            for fold, (
                train_idx,
                val_idx
            ) in enumerate(
                splitter.split(
                    X,
                    groups=groups,
                ),
                start=1,
            ):

                X_train = X[
                    train_idx
                ]

                X_val = X[
                    val_idx
                ]

                U_train = (
                    utility.iloc[
                        train_idx
                    ]
                    .reset_index(
                        drop=True
                    )
                )

                U_val = (
                    utility.iloc[
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

                actions, best_score = (
                    choose_actions(
                        scores,
                        threshold,
                    )
                )

                chosen_u = (
                    selected_utilities(
                        U_val,
                        actions,
                    )
                )

                metrics = (
                    evaluate_policy(
                        baseline_correct[
                            val_idx
                        ],
                        chosen_u,
                        actions,
                    )
                )

                total_repairs += (
                    metrics[
                        "repairs"
                    ]
                )

                total_breaks += (
                    metrics[
                        "breaks"
                    ]
                )

                total_interventions += int(
                    (
                        ~np.isclose(
                            actions,
                            1.0,
                        )
                    ).sum()
                )

                for local_i, global_i in enumerate(
                    val_idx
                ):

                    row = {
                        "sample_id":
                            str(
                                base.iloc[
                                    global_i
                                ][
                                    "sample_id"
                                ]
                            ),

                        "fold":
                            fold,

                        "ridge_alpha":
                            float(
                                ridge_alpha
                            ),

                        "threshold":
                            float(
                                threshold
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
                                chosen_u[
                                    local_i
                                ]
                            ),
                    }

                    oof_rows.append(
                        row
                    )

            net_repairs = (
                total_repairs
                -
                total_breaks
            )

            post_accuracy = (
                baseline_correct.sum()
                +
                net_repairs
            ) / len(
                baseline_correct
            )

            intervention_rate = (
                total_interventions
                /
                len(
                    baseline_correct
                )
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

                "repairs":
                    int(
                        total_repairs
                    ),

                "breaks":
                    int(
                        total_breaks
                    ),

                "net_repairs":
                    int(
                        net_repairs
                    ),

                "post_accuracy":
                    float(
                        post_accuracy
                    ),

                "interventions":
                    int(
                        total_interventions
                    ),

                "intervention_rate":
                    float(
                        intervention_rate
                    ),

                "mcnemar_exact_p":
                    float(
                        exact_mcnemar(
                            total_repairs,
                            total_breaks,
                        )
                    ),
            }

            grid_rows.append(
                candidate
            )

            candidate_key = (
                candidate[
                    "net_repairs"
                ],
                -candidate[
                    "breaks"
                ],
                -candidate[
                    "interventions"
                ],
                candidate[
                    "threshold"
                ],
                candidate[
                    "ridge_alpha"
                ],
            )

            if (
                best_key is None
                or
                candidate_key > best_key
            ):

                best_key = (
                    candidate_key
                )

                best_candidate = (
                    candidate
                )

                selected_candidate_oof = (
                    oof_rows
                )

    grid = pd.DataFrame(
        grid_rows
    )

    grid = (
        grid.sort_values(
            [
                "net_repairs",
                "breaks",
                "interventions",
                "threshold",
                "ridge_alpha",
            ],
            ascending=[
                False,
                True,
                True,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    grid.to_csv(
        GRID_PATH,
        index=False,
    )

    if best_candidate is None:

        raise RuntimeError(
            "No hyperparameter candidate selected."
        )

    selected_oof = pd.DataFrame(
        selected_candidate_oof
    )

    selected_oof = (
        selected_oof
        .sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        selected_oof
    ) != 1000:

        raise RuntimeError(
            "Selected OOF table does not "
            "contain 1000 rows."
        )

    selected_oof.to_csv(
        OOF_PATH,
        index=False,
    )

    # ========================================================
    # Train final deployable controller on ALL v2 samples
    # ========================================================

    final_models = make_models(
        X,
        utility,
        best_candidate[
            "ridge_alpha"
        ],
    )

    bundle = {
        "bundle_version":
            "1.0",

        "model_id":
            MODEL_ID,

        "attention_implementation":
            "eager",

        "head_layer":
            HEAD_LAYER,

        "head_index":
            HEAD_INDEX,

        "head_name":
            HEAD_NAME,

        "actions":
            ACTIONS,

        "nonnoop_actions":
            NONNOOP,

        "feature_family":
            PRIMARY_FAMILY,

        "feature_names":
            feature_names,

        "ridge_alpha":
            best_candidate[
                "ridge_alpha"
            ],

        "threshold":
            best_candidate[
                "threshold"
            ],

        "models":
            final_models,
    }

    joblib.dump(
        bundle,
        BUNDLE_PATH,
    )

    # ========================================================
    # Frozen manifest
    # ========================================================

    manifest = {
        "protocol_name":
            "AROMA Adaptive Cardinality Controller "
            "— Frozen Before v3 Confirmation",

        "development_dataset":
            "proc_count_causal_v2",

        "development_generation_version":
            "2.1_unique",

        "development_n":
            1000,

        "model":
            MODEL_ID,

        "attention_implementation":
            "eager",

        "actuator": {
            "type":
                "cross_attention_head_output_gain",

            "layer":
                HEAD_LAYER,

            "head":
                HEAD_INDEX,

            "head_name":
                HEAD_NAME,

            "actions":
                ACTIONS,

            "noop_alpha":
                1.0,
        },

        "controller": {
            "type":
                "per-action ridge utility estimator",

            "feature_family":
                PRIMARY_FAMILY,

            "feature_count":
                len(
                    feature_names
                ),

            "feature_names":
                feature_names,

            "ridge_alpha":
                best_candidate[
                    "ridge_alpha"
                ],

            "threshold":
                best_candidate[
                    "threshold"
                ],

            "action_rule":
                (
                    "Choose the non-NOOP action with "
                    "highest predicted utility only if "
                    "its score is strictly greater than "
                    "the frozen threshold; otherwise "
                    "choose alpha=1.0 NOOP."
                ),
        },

        "hyperparameter_selection": {
            "dataset":
                "proc_count_causal_v2",

            "group_variable":
                "replicate",

            "n_groups":
                20,

            "cv":
                "5-fold GroupKFold",

            "ridge_alpha_grid":
                RIDGE_ALPHAS,

            "threshold_grid":
                THRESHOLDS,

            "primary_selection_metric":
                "OOF net repairs",

            "tie_breaking":
                [
                    "fewer breaks",
                    "fewer interventions",
                    "higher threshold",
                    "larger ridge regularization",
                ],
        },

        "development_cv_selected_result":
            best_candidate,

        "final_confirmation_policy": {
            "allow_head_retuning":
                False,

            "allow_action_retuning":
                False,

            "allow_feature_family_retuning":
                False,

            "allow_feature_retuning":
                False,

            "allow_ridge_retuning":
                False,

            "allow_threshold_retuning":
                False,

            "allow_primary_endpoint_change":
                False,

            "allow_subgroup_redefinition":
                False,
        },

        "planned_v3_primary_endpoint": {
            "metric":
                "paired accuracy change",

            "definition":
                (
                    "controller post-accuracy minus "
                    "baseline accuracy"
                ),

            "paired_test":
                "exact McNemar",

            "direction":
                "controller > baseline",
        },

        "planned_v3_secondary_endpoints": [
            "repairs",
            "breaks",
            "net_repairs",
            "repair_rate_among_baseline_wrong",
            "break_rate_among_baseline_correct",
            "intervention_rate",
            "condition-wise accuracy change",
        ],

        "input_hashes": {
            "confirmation_results_sha256":
                sha256_file(
                    CONF_PATH
                ),

            "compressed_action_matrix_sha256":
                sha256_file(
                    MATRIX_PATH
                ),
        },

        "bundle_path":
            str(
                BUNDLE_PATH
            ),

        "frozen_before_v3_evaluation":
            True,
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "\n" + "=" * 112
    )

    print(
        "FINAL DEVELOPMENT HYPERPARAMETER SELECTION"
    )

    print(
        "=" * 112
    )

    print(
        "Selected Ridge alpha :",
        best_candidate[
            "ridge_alpha"
        ],
    )

    print(
        "Selected threshold   :",
        best_candidate[
            "threshold"
        ],
    )

    print(
        "OOF repairs          :",
        best_candidate[
            "repairs"
        ],
    )

    print(
        "OOF breaks           :",
        best_candidate[
            "breaks"
        ],
    )

    print(
        "OOF net repairs      :",
        best_candidate[
            "net_repairs"
        ],
    )

    print(
        "OOF post accuracy    :",
        best_candidate[
            "post_accuracy"
        ],
    )

    print(
        "OOF intervention rate:",
        best_candidate[
            "intervention_rate"
        ],
    )

    print(
        "OOF McNemar exact p  :",
        best_candidate[
            "mcnemar_exact_p"
        ],
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "FREEZE ARTIFACTS"
    )

    print(
        "=" * 112
    )

    print(
        "Controller bundle:",
        BUNDLE_PATH,
    )

    print(
        "Manifest         :",
        MANIFEST_PATH,
    )

    print(
        "Grid             :",
        GRID_PATH,
    )

    print(
        "Selected OOF     :",
        OOF_PATH,
    )

    print(
        "\nFINAL CONTROLLER FIT COMPLETE"
    )

    print(
        "DO NOT inspect or use v3 results "
        "before protocol + Git freeze."
    )


if __name__ == "__main__":
    main()
