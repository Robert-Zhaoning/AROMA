from pathlib import Path

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

from distributed_dam_v2_smoke_fixed import (
    DistributedLayerDAM,
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
    "adaptive_dam_beta_sweep"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_PATH = (
    OUT_DIR / "beta_sweep_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR / "beta_sweep_summary.csv"
)


BETAS = [
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
]

GAMMA = 0.0
QUERY_SCOPE = "answer"


HEADS = {
    "L33H1": (33, 1),
    "L3H4": (3, 4),
    "L18H13": (18, 13),
    "L8H30": (8, 30),
    "L3H11": (3, 11),
    "L13H11": (13, 11),
    "L33H21": (33, 21),
}


def run_head_beta(
    model,
    inputs,
    numeral_ids,
    gt,
    object_token_sets,
    head_name,
    beta,
):

    layer, head = HEADS[
        head_name
    ]

    if beta == 1.0:
        score = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

        return score, 0

    dam = DistributedLayerDAM(
        model=model,
        layer_idx=layer,
        head_indices=[head],
        object_token_sets=object_token_sets,
        beta=beta,
        gamma=GAMMA,
        query_scope=QUERY_SCOPE,
    )

    dam.register()

    try:
        score = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

    finally:
        dam.remove()

    calls = int(
        dam.head_calls.get(
            head,
            0,
        )
    )

    if calls <= 0:
        raise RuntimeError(
            f"Hook not called: "
            f"{head_name}, beta={beta}"
        )

    return score, calls


def main():

    print("=" * 110)
    print("AROMA ADAPTIVE DAM BETA SWEEP")
    print("=" * 110)

    print("Betas:", BETAS)
    print("Gamma:", GAMMA)
    print(
        "Query scope:",
        QUERY_SCOPE,
    )

    selector = pd.read_csv(
        SELECTOR_PATH
    )

    selector = selector[
        selector["family"]
        == "compact_response_policy"
    ].copy()

    assert len(selector) == 54
    assert selector["sample_id"].nunique() == 54
    assert selector["pair_id"].nunique() == 27

    policy_columns = {
        "learned_adaptive":
            "selected_head",

        "training_fixed":
            "fixed_head",

        # IMPORTANT:
        # This is the best head according to the original
        # beta=1.25 profile. It is therefore named
        # profile_oracle rather than true oracle at every beta.
        "profile_oracle":
            "oracle_head",
    }

    for c in policy_columns.values():
        assert set(
            selector[c].astype(str)
        ).issubset(
            set(HEADS)
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

    for _, item in tqdm(
        selector.iterrows(),
        total=len(selector),
        desc="DAM beta sweep",
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

        sample = metadata[sid]

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

        baseline = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

        base_pred = int(
            baseline[
                "best_numeral"
            ]
        )

        base_correct = (
            base_pred == gt
        )

        expected_correct = (
            role == "correct"
        )

        if (
            base_correct
            != expected_correct
        ):
            raise RuntimeError(
                f"Baseline mismatch: {sid}"
            )

        base_logp = float(
            baseline[
                "gt_logp"
            ]
        )

        base_margin = float(
            baseline[
                "margin"
            ]
        )

        # Cache each unique (head, beta).
        cache = {}

        needed_heads = set(
            str(item[col])
            for col
            in policy_columns.values()
        )

        for head_name in needed_heads:

            for beta in BETAS:

                key = (
                    head_name,
                    beta,
                )

                score, calls = (
                    run_head_beta(
                        model=model,
                        inputs=inputs,
                        numeral_ids=
                            numeral_ids,
                        gt=gt,
                        object_token_sets=
                            object_token_sets,
                        head_name=
                            head_name,
                        beta=beta,
                    )
                )

                cache[key] = (
                    score,
                    calls,
                )

        for (
            policy,
            column,
        ) in policy_columns.items():

            head_name = str(
                item[column]
            )

            for beta in BETAS:

                score, calls = (
                    cache[
                        (
                            head_name,
                            beta,
                        )
                    ]
                )

                pred = int(
                    score[
                        "best_numeral"
                    ]
                )

                correct = (
                    pred == gt
                )

                repair = (
                    (not base_correct)
                    and correct
                )

                break_case = (
                    base_correct
                    and not correct
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
                        policy,

                    "head_name":
                        head_name,

                    "beta":
                        beta,

                    "baseline_prediction":
                        base_pred,

                    "prediction":
                        pred,

                    "baseline_correct":
                        base_correct,

                    "correct":
                        correct,

                    "repair":
                        repair,

                    "break":
                        break_case,

                    "baseline_gt_logp":
                        base_logp,

                    "gt_logp":
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
                        - base_logp,

                    "baseline_margin":
                        base_margin,

                    "margin":
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
                        - base_margin,

                    "calls":
                        calls,
                })

    df = pd.DataFrame(
        rows
    )

    expected = (
        54
        * 3
        * len(BETAS)
    )

    assert len(df) == expected

    df.to_csv(
        RESULT_PATH,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summaries = []

    for (
        policy,
        beta,
    ), g in df.groupby(
        [
            "policy",
            "beta",
        ],
        sort=False,
    ):

        wrong = g[
            g["role"]
            == "wrong"
        ]

        correct = g[
            g["role"]
            == "correct"
        ]

        repairs = int(
            wrong[
                "repair"
            ].sum()
        )

        breaks = int(
            correct[
                "break"
            ].sum()
        )

        summaries.append({
            "policy":
                policy,

            "beta":
                float(beta),

            "post_accuracy":
                float(
                    g[
                        "correct"
                    ].mean()
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

            "mean_delta_margin_wrong":
                float(
                    wrong[
                        "delta_margin"
                    ].mean()
                ),

            "positive_margin_shift_wrong":
                float(
                    (
                        wrong[
                            "delta_margin"
                        ]
                        > 0
                    ).mean()
                ),

            "post_margin_positive_wrong":
                int(
                    (
                        wrong[
                            "margin"
                        ]
                        > 0
                    ).sum()
                ),
        })

    summary = pd.DataFrame(
        summaries
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 130
    )

    print(
        "FINAL DAM BETA-SWEEP RESULTS"
    )

    print(
        "=" * 130
    )

    display = [
        "policy",
        "beta",
        "post_accuracy",
        "repairs",
        "breaks",
        "net_repairs",
        "repair_rate_wrong",
        "break_rate_correct",
        "mean_delta_logp_wrong",
        "mean_delta_margin_wrong",
        "post_margin_positive_wrong",
    ]

    print(
        summary[
            display
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # Best beta by net task-level gain.
    print(
        "\nBEST BETA PER POLICY "
        "(development set)"
    )

    for policy, g in (
        summary.groupby(
            "policy"
        )
    ):

        best = (
            g.sort_values(
                [
                    "net_repairs",
                    "repairs",
                    "breaks",
                    "mean_delta_logp_wrong",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    False,
                ],
            )
            .iloc[0]
        )

        print(
            f"{policy:18s} "
            f"beta={best['beta']:.2f} "
            f"repairs={int(best['repairs'])} "
            f"breaks={int(best['breaks'])} "
            f"net={int(best['net_repairs'])} "
            f"acc={best['post_accuracy']:.4f}"
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
        "\nADAPTIVE DAM BETA SWEEP COMPLETE"
    )


if __name__ == "__main__":
    main()
