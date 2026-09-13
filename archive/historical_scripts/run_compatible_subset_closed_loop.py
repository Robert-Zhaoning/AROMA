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
)

from run_dam_v2_selective_validation import (
    run_score,
    BETA,
    GAMMA,
    QUERY_SCOPE,
)


MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

PAIR_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "exact_matched_pairs_eager.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "compatible_subset_closed_loop"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_PATH = (
    OUT_DIR
    / "compatible_subset_closed_loop_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "compatible_subset_closed_loop_summary.csv"
)


DISCOVERY_SAMPLES = {
    "pccv1_n05_row_r00",
    "pccv1_n05_row_r01",
}


# ============================================================
# Pre-frozen subset candidates
#
# These are NOT newly searched on the current outcomes.
# They come from the earlier interaction/subset experiments.
# ============================================================

SUBSETS = {

    "compatible_A_pair": [
        (3, 4),
        (8, 30),
    ],

    "compatible_E_triple": [
        (3, 4),
        (8, 30),
        (13, 11),
    ],

    "destructive_F_control": [
        (3, 4),
        (18, 13),
        (13, 11),
    ],

    "all7_control": [
        (33, 1),
        (3, 4),
        (18, 13),
        (8, 30),
        (3, 11),
        (13, 11),
        (33, 21),
    ],
}


def main():

    print("=" * 118)
    print("AROMA COMPATIBLE-SUBSET CLOSED-LOOP FEASIBILITY")
    print("=" * 118)

    print(
        "DAM parameters:",
        f"beta={BETA},",
        f"gamma={GAMMA},",
        f"query_scope={QUERY_SCOPE}",
    )

    print("\nSubsets:")

    for name, subset in SUBSETS.items():
        pretty = [
            f"L{l}H{h}"
            for l, h in subset
        ]

        print(
            f"  {name:24s}: "
            + ", ".join(pretty)
        )

    # ========================================================
    # Strict held-out/development pair set
    # ========================================================

    pairs = pd.read_csv(
        PAIR_PATH
    )

    print(
        "\nPairs before exclusion:",
        len(pairs),
    )

    leak_mask = (
        pairs["wrong_sample_id"]
        .astype(str)
        .isin(DISCOVERY_SAMPLES)
        |
        pairs["correct_sample_id"]
        .astype(str)
        .isin(DISCOVERY_SAMPLES)
    )

    removed = pairs[
        leak_mask
    ].copy()

    pairs = (
        pairs[
            ~leak_mask
        ]
        .reset_index(drop=True)
    )

    print(
        "Discovery pairs removed:",
        len(removed),
    )

    print(
        "Pairs retained:",
        len(pairs),
    )

    if len(pairs) != 27:
        raise RuntimeError(
            f"Expected 27 pairs, found {len(pairs)}"
        )

    sample_rows = []

    for _, row in pairs.iterrows():

        sample_rows.append({
            "pair_id":
                int(row["pair_id"]),

            "role":
                "wrong",

            "sample_id":
                str(
                    row[
                        "wrong_sample_id"
                    ]
                ),

            "ground_truth":
                int(
                    row[
                        "ground_truth"
                    ]
                ),

            "condition":
                str(
                    row[
                        "condition"
                    ]
                ),
        })

        sample_rows.append({
            "pair_id":
                int(row["pair_id"]),

            "role":
                "correct",

            "sample_id":
                str(
                    row[
                        "correct_sample_id"
                    ]
                ),

            "ground_truth":
                int(
                    row[
                        "ground_truth"
                    ]
                ),

            "condition":
                str(
                    row[
                        "condition"
                    ]
                ),
        })

    if len(sample_rows) != 54:
        raise RuntimeError(
            f"Expected 54 samples, found {len(sample_rows)}"
        )

    sample_ids = {
        x["sample_id"]
        for x in sample_rows
    }

    if (
        sample_ids
        & DISCOVERY_SAMPLES
    ):
        raise RuntimeError(
            "Discovery leakage detected."
        )

    print(
        "Held-out/development sample audit: PASS"
    )

    print(
        "Samples:",
        len(sample_rows),
    )

    print(
        "Subset interventions:",
        len(sample_rows)
        * len(SUBSETS),
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

    metadata = (
        load_metadata()
    )

    rows = []

    # ========================================================
    # Run
    # ========================================================

    for item in tqdm(
        sample_rows,
        desc="Compatible subset DAM",
    ):

        sid = item[
            "sample_id"
        ]

        role = item[
            "role"
        ]

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

        baseline = run_score(
            model=model,
            inputs=inputs,
            numeral_ids=numeral_ids,
            gt=gt,
            object_token_sets=
                object_token_sets,
            subset=None,
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
                "Baseline role mismatch: "
                f"{sid}, role={role}, "
                f"GT={gt}, pred={base_pred}"
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

        # ----------------------------------------------------
        # Fixed subset interventions
        # ----------------------------------------------------

        for (
            subset_name,
            subset,
        ) in SUBSETS.items():

            result = run_score(
                model=model,
                inputs=inputs,
                numeral_ids=numeral_ids,
                gt=gt,
                object_token_sets=
                    object_token_sets,
                subset=subset,
            )

            pred = int(
                result[
                    "best_numeral"
                ]
            )

            correct_after = (
                pred == gt
            )

            repair = (
                (not base_correct)
                and correct_after
            )

            break_case = (
                base_correct
                and not correct_after
            )

            rows.append({
                "pair_id":
                    item[
                        "pair_id"
                    ],

                "role":
                    role,

                "sample_id":
                    sid,

                "ground_truth":
                    gt,

                "condition":
                    item[
                        "condition"
                    ],

                "subset":
                    subset_name,

                "subset_size":
                    len(subset),

                "heads":
                    ",".join(
                        f"L{l}H{h}"
                        for l, h
                        in subset
                    ),

                "baseline_prediction":
                    base_pred,

                "prediction":
                    pred,

                "baseline_correct":
                    base_correct,

                "correct":
                    correct_after,

                "repair":
                    repair,

                "break":
                    break_case,

                "baseline_gt_logp":
                    base_logp,

                "intervened_gt_logp":
                    float(
                        result[
                            "gt_logp"
                        ]
                    ),

                "delta_logp":
                    float(
                        result[
                            "gt_logp"
                        ]
                    )
                    - base_logp,

                "baseline_margin":
                    base_margin,

                "intervened_margin":
                    float(
                        result[
                            "margin"
                        ]
                    ),

                "delta_margin":
                    float(
                        result[
                            "margin"
                        ]
                    )
                    - base_margin,
            })

    # ========================================================
    # Save raw results
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    expected_rows = (
        54
        * len(SUBSETS)
    )

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"got {len(df)}"
        )

    if df.isna().sum().sum() != 0:
        raise RuntimeError(
            "NaN detected in output."
        )

    df.to_csv(
        RESULT_PATH,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summaries = []

    for subset_name, g in (
        df.groupby(
            "subset",
            sort=False,
        )
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
            "subset":
                subset_name,

            "subset_size":
                int(
                    g[
                        "subset_size"
                    ].iloc[0]
                ),

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
                        "delta_logp"
                    ].mean()
                ),

            "median_delta_logp_wrong":
                float(
                    wrong[
                        "delta_logp"
                    ].median()
                ),

            "mean_delta_margin_wrong":
                float(
                    wrong[
                        "delta_margin"
                    ].mean()
                ),

            "positive_logp_fraction_wrong":
                float(
                    (
                        wrong[
                            "delta_logp"
                        ]
                        > 0
                    ).mean()
                ),

            "positive_margin_fraction_wrong":
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
                            "intervened_margin"
                        ]
                        > 0
                    ).sum()
                ),

            "closest_wrong_margin":
                float(
                    wrong[
                        "intervened_margin"
                    ].max()
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
    # Oracle-over-fixed-subsets feasibility
    #
    # IMPORTANT:
    # This is analysis only.
    # For each sample, pick the best of the four pre-defined
    # subset candidates using GT delta_logp.
    # It is NOT deployable.
    # ========================================================

    oracle_rows = []

    for sid, g in (
        df.groupby(
            "sample_id"
        )
    ):

        best_idx = (
            g[
                "delta_logp"
            ]
            .idxmax()
        )

        best = df.loc[
            best_idx
        ]

        oracle_rows.append(
            best
        )

    oracle = pd.DataFrame(
        oracle_rows
    )

    wrong_o = oracle[
        oracle["role"]
        == "wrong"
    ]

    correct_o = oracle[
        oracle["role"]
        == "correct"
    ]

    oracle_repairs = int(
        wrong_o[
            "repair"
        ].sum()
    )

    oracle_breaks = int(
        correct_o[
            "break"
        ].sum()
    )

    # ========================================================
    # Report
    # ========================================================

    print(
        "\n" + "=" * 138
    )

    print(
        "FINAL COMPATIBLE-SUBSET CLOSED-LOOP RESULTS"
    )

    print(
        "=" * 138
    )

    display_cols = [
        "subset",
        "subset_size",
        "post_accuracy",
        "repairs",
        "breaks",
        "net_repairs",
        "repair_rate_wrong",
        "break_rate_correct",
        "mean_delta_logp_wrong",
        "mean_delta_margin_wrong",
        "positive_logp_fraction_wrong",
        "post_margin_positive_wrong",
        "closest_wrong_margin",
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
        "\n" + "=" * 138
    )

    print(
        "ORACLE OVER THE FOUR PRE-FROZEN SUBSETS"
    )

    print(
        "=" * 138
    )

    print(
        "Wrong-sample repairs:",
        oracle_repairs,
        "/ 27",
    )

    print(
        "Correct-sample breaks:",
        oracle_breaks,
        "/ 27",
    )

    print(
        "Net repairs:",
        oracle_repairs
        - oracle_breaks,
    )

    print(
        "Post accuracy:",
        float(
            oracle[
                "correct"
            ].mean()
        ),
    )

    print(
        "Mean wrong-sample delta logp:",
        float(
            wrong_o[
                "delta_logp"
            ].mean()
        ),
    )

    print(
        "Positive wrong-sample delta fraction:",
        float(
            (
                wrong_o[
                    "delta_logp"
                ]
                > 0
            ).mean()
        ),
    )

    print(
        "\nOracle subset distribution (analysis only):"
    )

    print(
        oracle[
            "subset"
        ]
        .value_counts()
        .to_string()
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
        "\nCOMPATIBLE-SUBSET FEASIBILITY COMPLETE"
    )


if __name__ == "__main__":
    main()
