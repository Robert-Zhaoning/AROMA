from pathlib import Path
import re

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

from oracle_dam_v2_smoke import (
    build_object_token_sets,
    load_metadata,
    move_inputs,
    numeral_token_ids,
    prepare_inputs,
    score_state,
)

from profile_dam_head_responses import (
    run_single_head,
    BETA,
    GAMMA,
    QUERY_SCOPE,
)


MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

SELECTOR_PATH = Path(
    "outputs/proc_count_causal_v1/router/"
    "adaptive_head_selector/"
    "adaptive_head_selector_predictions.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "adaptive_dam_closed_loop"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_PATH = (
    OUT_DIR
    / "closed_loop_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "closed_loop_summary.csv"
)


FROZEN_HEADS = {
    "L33H1": (33, 1),
    "L3H4": (3, 4),
    "L18H13": (18, 13),
    "L8H30": (8, 30),
    "L3H11": (3, 11),
    "L13H11": (13, 11),
    "L33H21": (33, 21),
}


def parse_head(name):
    name = str(name)

    if name not in FROZEN_HEADS:
        raise RuntimeError(
            f"Unexpected head: {name}"
        )

    return FROZEN_HEADS[name]


def main():

    print("=" * 110)
    print("AROMA CLOSED-LOOP ADAPTIVE DAM")
    print("=" * 110)

    print(
        "DAM parameters:",
        f"beta={BETA},",
        f"gamma={GAMMA},",
        f"query_scope={QUERY_SCOPE}",
    )

    # ========================================================
    # Load OOF selector decisions
    # ========================================================

    selector = pd.read_csv(
        SELECTOR_PATH
    )

    selector = selector[
        selector["family"]
        == "compact_response_policy"
    ].copy()

    selector = selector.reset_index(
        drop=True
    )

    assert len(selector) == 54
    assert selector["sample_id"].nunique() == 54
    assert selector["pair_id"].nunique() == 27

    assert (
        (selector["role"] == "correct").sum()
        == 27
    )

    assert (
        (selector["role"] == "wrong").sum()
        == 27
    )

    required = {
        "selected_head",
        "fixed_head",
        "oracle_head",
    }

    missing = (
        required
        - set(selector.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing selector columns: {missing}"
        )

    for c in [
        "selected_head",
        "fixed_head",
        "oracle_head",
    ]:
        bad = (
            set(selector[c].astype(str))
            - set(FROZEN_HEADS)
        )

        if bad:
            raise RuntimeError(
                f"{c}: unknown heads {bad}"
            )

    discovery = {
        "pccv1_n05_row_r00",
        "pccv1_n05_row_r01",
    }

    assert not (
        discovery
        & set(
            selector[
                "sample_id"
            ].astype(str)
        )
    )

    print(
        "Selector audit: PASS"
    )

    print(
        "Samples:",
        len(selector),
    )

    print(
        "Pairs:",
        selector[
            "pair_id"
        ].nunique(),
    )

    print("\nLearned head distribution:")

    print(
        selector[
            "selected_head"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Load model
    # ========================================================

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    print(
        "Loading model..."
    )

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    )

    model.eval()

    numeral_ids = (
        numeral_token_ids(
            processor
        )
    )

    metadata = load_metadata()

    rows = []

    # ========================================================
    # Closed-loop evaluation
    # ========================================================

    for _, item in tqdm(
        selector.iterrows(),
        total=len(selector),
        desc="Closed-loop DAM",
    ):

        sid = str(
            item["sample_id"]
        )

        role = str(
            item["role"]
        )

        pair_id = int(
            item["pair_id"]
        )

        sample = metadata[
            sid
        ]

        gt = int(
            sample[
                "ground_truth"
            ]
        )

        image = (
            Image.open(
                sample[
                    "image_path"
                ]
            )
            .convert("RGB")
        )

        (
            _geometry,
            object_token_sets,
        ) = build_object_token_sets(
            sample,
            processor,
            image,
        )

        (
            _formatted,
            inputs,
        ) = prepare_inputs(
            processor,
            image,
        )

        inputs = move_inputs(
            inputs,
            model,
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

        baseline_prediction = int(
            baseline[
                "best_numeral"
            ]
        )

        baseline_correct = (
            baseline_prediction
            == gt
        )

        # Exact held-out matched set should preserve role.
        expected_correct = (
            role == "correct"
        )

        if (
            baseline_correct
            != expected_correct
        ):
            raise RuntimeError(
                "Baseline role mismatch: "
                f"{sid} role={role}, "
                f"GT={gt}, "
                f"prediction={baseline_prediction}"
            )

        baseline_logp = float(
            baseline[
                "gt_logp"
            ]
        )

        baseline_margin = float(
            baseline[
                "margin"
            ]
        )

        # ----------------------------------------------------
        # Three policies
        # ----------------------------------------------------

        policy_heads = {
            "learned_adaptive":
                str(
                    item[
                        "selected_head"
                    ]
                ),

            "training_fixed":
                str(
                    item[
                        "fixed_head"
                    ]
                ),

            "oracle_best":
                str(
                    item[
                        "oracle_head"
                    ]
                ),
        }

        # Cache per unique head so if two policies select the
        # same head we do not rerun the DAM intervention.
        cache = {}

        for head_name in set(
            policy_heads.values()
        ):

            layer, head = (
                parse_head(
                    head_name
                )
            )

            score, calls = (
                run_single_head(
                    model=model,
                    inputs=inputs,
                    numeral_ids=numeral_ids,
                    gt=gt,
                    object_token_sets=
                        object_token_sets,
                    layer=layer,
                    head=head,
                )
            )

            if calls <= 0:
                raise RuntimeError(
                    "DAM hook not called: "
                    f"{sid} {head_name}"
                )

            cache[
                head_name
            ] = {
                "score":
                    score,
                "calls":
                    calls,
            }

        # ----------------------------------------------------
        # Record policy outcomes
        # ----------------------------------------------------

        for (
            policy_name,
            head_name,
        ) in policy_heads.items():

            result = cache[
                head_name
            ]

            score = result[
                "score"
            ]

            modulated_prediction = int(
                score[
                    "best_numeral"
                ]
            )

            modulated_correct = (
                modulated_prediction
                == gt
            )

            repair = (
                (not baseline_correct)
                and modulated_correct
            )

            break_case = (
                baseline_correct
                and (not modulated_correct)
            )

            layer, head = (
                parse_head(
                    head_name
                )
            )

            rows.append({
                "pair_id":
                    pair_id,

                "role":
                    role,

                "sample_id":
                    sid,

                "ground_truth":
                    gt,

                "condition":
                    str(
                        sample[
                            "condition"
                        ]
                    ),

                "policy":
                    policy_name,

                "selected_head":
                    head_name,

                "layer":
                    int(layer),

                "head":
                    int(head),

                "baseline_prediction":
                    baseline_prediction,

                "modulated_prediction":
                    modulated_prediction,

                "baseline_correct":
                    bool(
                        baseline_correct
                    ),

                "modulated_correct":
                    bool(
                        modulated_correct
                    ),

                "repair":
                    bool(
                        repair
                    ),

                "break":
                    bool(
                        break_case
                    ),

                "baseline_gt_logp":
                    baseline_logp,

                "intervened_gt_logp":
                    float(
                        score[
                            "gt_logp"
                        ]
                    ),

                "delta_gt_logp":
                    float(
                        score[
                            "gt_logp"
                        ]
                    )
                    - baseline_logp,

                "baseline_margin":
                    baseline_margin,

                "intervened_margin":
                    float(
                        score[
                            "margin"
                        ]
                    ),

                "delta_margin":
                    float(
                        score[
                            "margin"
                        ]
                    )
                    - baseline_margin,

                "calls":
                    int(
                        result[
                            "calls"
                        ]
                    ),
            })

    # ========================================================
    # Result table
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    assert len(df) == (
        54 * 3
    )

    assert (
        df["calls"] > 0
    ).all()

    df.to_csv(
        RESULT_PATH,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summaries = []

    for policy, g in (
        df.groupby(
            "policy",
            sort=False,
        )
    ):

        baseline_acc = float(
            g[
                "baseline_correct"
            ].mean()
        )

        after_acc = float(
            g[
                "modulated_correct"
            ].mean()
        )

        repairs = int(
            g[
                "repair"
            ].sum()
        )

        breaks = int(
            g[
                "break"
            ].sum()
        )

        wrong = g[
            g["role"]
            == "wrong"
        ]

        correct = g[
            g["role"]
            == "correct"
        ]

        summaries.append({
            "policy":
                policy,

            "n_samples":
                len(g),

            "baseline_accuracy":
                baseline_acc,

            "post_accuracy":
                after_acc,

            "accuracy_gain":
                (
                    after_acc
                    - baseline_acc
                ),

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                repairs
                - breaks,

            "repair_rate_wrong":
                float(
                    wrong[
                        "repair"
                    ].mean()
                ),

            "break_rate_correct":
                float(
                    correct[
                        "break"
                    ].mean()
                ),

            "mean_delta_logp_wrong":
                float(
                    wrong[
                        "delta_gt_logp"
                    ].mean()
                ),

            "mean_delta_logp_correct":
                float(
                    correct[
                        "delta_gt_logp"
                    ].mean()
                ),

            "positive_delta_fraction_wrong":
                float(
                    (
                        wrong[
                            "delta_gt_logp"
                        ]
                        > 0
                    ).mean()
                ),
        })

    summary = pd.DataFrame(
        summaries
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # Pairwise learned-vs-fixed outcome comparison
    # ========================================================

    learned = df[
        df["policy"]
        == "learned_adaptive"
    ][
        [
            "sample_id",
            "role",
            "modulated_correct",
            "repair",
            "break",
            "delta_gt_logp",
        ]
    ].copy()

    fixed = df[
        df["policy"]
        == "training_fixed"
    ][
        [
            "sample_id",
            "modulated_correct",
            "repair",
            "break",
            "delta_gt_logp",
        ]
    ].copy()

    compare = learned.merge(
        fixed,
        on="sample_id",
        suffixes=(
            "_learned",
            "_fixed",
        ),
        validate="one_to_one",
    )

    wrong_compare = compare[
        compare["role"]
        == "wrong"
    ]

    learned_better_count = int(
        (
            wrong_compare[
                "modulated_correct_learned"
            ].astype(int)
            >
            wrong_compare[
                "modulated_correct_fixed"
            ].astype(int)
        ).sum()
    )

    fixed_better_count = int(
        (
            wrong_compare[
                "modulated_correct_fixed"
            ].astype(int)
            >
            wrong_compare[
                "modulated_correct_learned"
            ].astype(int)
        ).sum()
    )

    # ========================================================
    # Report
    # ========================================================

    print(
        "\n" + "=" * 130
    )

    print(
        "FINAL CLOSED-LOOP DAM RESULTS"
    )

    print(
        "=" * 130
    )

    display_cols = [
        "policy",
        "baseline_accuracy",
        "post_accuracy",
        "accuracy_gain",
        "repairs",
        "breaks",
        "net_repairs",
        "repair_rate_wrong",
        "break_rate_correct",
        "mean_delta_logp_wrong",
        "positive_delta_fraction_wrong",
    ]

    print(
        summary[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nLEARNED vs FIXED ON WRONG SAMPLES"
    )

    print(
        "Learned repairs where fixed fails:",
        learned_better_count,
    )

    print(
        "Fixed repairs where learned fails:",
        fixed_better_count,
    )

    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nCLOSED-LOOP ADAPTIVE DAM COMPLETE"
    )


if __name__ == "__main__":
    main()
