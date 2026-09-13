from pathlib import Path
import argparse

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

PAIR_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "exact_matched_pairs_eager.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v1/router/"
    "dam_rescue_r1_true_oracle"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DISCOVERY_SAMPLES = {
    "pccv1_n05_row_r00",
    "pccv1_n05_row_r01",
}


HEADS = [
    (33, 1),
    (3, 4),
    (18, 13),
    (8, 30),
    (3, 11),
    (13, 11),
    (33, 21),
]


BETAS = [
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
]


GAMMA = 0.0
QUERY_SCOPE = "answer"


def head_name(layer, head):
    return f"L{layer}H{head}"


def run_intervention(
    model,
    inputs,
    numeral_ids,
    gt,
    object_token_sets,
    layer,
    head,
    beta,
):
    dam = DistributedLayerDAM(
        model=model,
        layer_idx=int(layer),
        head_indices=[
            int(head)
        ],
        object_token_sets=
            object_token_sets,
        beta=float(beta),
        gamma=GAMMA,
        query_scope=QUERY_SCOPE,
    )

    dam.register()

    try:
        result = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

    finally:
        dam.remove()

    calls = int(
        dam.head_calls.get(
            int(head),
            0,
        )
    )

    if calls <= 0:
        raise RuntimeError(
            "DAM hook was not called: "
            f"L{layer}H{head}, beta={beta}"
        )

    return result, calls


def build_wrong_samples():

    pairs = pd.read_csv(
        PAIR_PATH
    )

    print(
        "Pairs before discovery exclusion:",
        len(pairs),
    )

    leak_mask = (
        pairs[
            "wrong_sample_id"
        ]
        .astype(str)
        .isin(
            DISCOVERY_SAMPLES
        )
        |
        pairs[
            "correct_sample_id"
        ]
        .astype(str)
        .isin(
            DISCOVERY_SAMPLES
        )
    )

    removed = pairs[
        leak_mask
    ].copy()

    pairs = (
        pairs[
            ~leak_mask
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        "Discovery pairs removed:",
        len(removed),
    )

    if len(pairs) != 27:
        raise RuntimeError(
            f"Expected 27 development pairs, "
            f"found {len(pairs)}"
        )

    samples = []

    for _, row in (
        pairs.iterrows()
    ):

        samples.append({
            "pair_id":
                int(
                    row[
                        "pair_id"
                    ]
                ),

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

    if len(samples) != 27:
        raise RuntimeError(
            "Expected 27 wrong samples."
        )

    ids = {
        x["sample_id"]
        for x in samples
    }

    if (
        ids
        & DISCOVERY_SAMPLES
    ):
        raise RuntimeError(
            "Discovery sample leakage."
        )

    return samples


def main(
    limit=None,
    protocol_only=False,
):

    print("=" * 118)
    print(
        "AROMA DAM RESCUE R1 — "
        "TRUE ORACLE HEAD × BETA"
    )
    print("=" * 118)

    print(
        "Gamma:",
        GAMMA,
    )

    print(
        "Query scope:",
        QUERY_SCOPE,
    )

    print(
        "Betas:",
        BETAS,
    )

    print(
        "Heads:",
        [
            head_name(l, h)
            for l, h
            in HEADS
        ],
    )

    samples = (
        build_wrong_samples()
    )

    full_count = len(
        samples
    )

    if limit is not None:

        if limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        samples = samples[
            :limit
        ]

        print(
            "\n[SMOKE / LIMITED MODE]"
        )

        print(
            "Full wrong-sample pool:",
            full_count,
        )

    print(
        "Wrong samples selected:",
        len(samples),
    )

    expected = (
        len(samples)
        * len(HEADS)
        * len(BETAS)
    )

    print(
        "Expected interventions:",
        expected,
    )

    if protocol_only:

        print(
            "\nPROTOCOL-ONLY AUDIT PASS"
        )

        print(
            "No model was loaded."
        )

        return

    # ========================================================
    # Model
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
    # Run all head × beta configurations
    # ========================================================

    for item in tqdm(
        samples,
        desc="True oracle head×beta",
    ):

        sid = item[
            "sample_id"
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
        ) = (
            build_object_token_sets(
                sample,
                processor,
                image,
            )
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

        base_pred = int(
            baseline[
                "best_numeral"
            ]
        )

        if base_pred == gt:
            raise RuntimeError(
                "Expected a wrong sample, "
                f"but {sid} is correct: "
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
        # Exhaustive frozen-head × beta screen
        # ----------------------------------------------------

        for layer, head in HEADS:

            hname = head_name(
                layer,
                head,
            )

            for beta in BETAS:

                result, calls = (
                    run_intervention(
                        model=model,
                        inputs=inputs,
                        numeral_ids=
                            numeral_ids,
                        gt=gt,
                        object_token_sets=
                            object_token_sets,
                        layer=layer,
                        head=head,
                        beta=beta,
                    )
                )

                pred = int(
                    result[
                        "best_numeral"
                    ]
                )

                post_logp = float(
                    result[
                        "gt_logp"
                    ]
                )

                post_margin = float(
                    result[
                        "margin"
                    ]
                )

                rows.append({
                    "pair_id":
                        item[
                            "pair_id"
                        ],

                    "sample_id":
                        sid,

                    "ground_truth":
                        gt,

                    "condition":
                        item[
                            "condition"
                        ],

                    "layer":
                        int(layer),

                    "head":
                        int(head),

                    "head_name":
                        hname,

                    "beta":
                        float(beta),

                    "baseline_prediction":
                        base_pred,

                    "prediction":
                        pred,

                    "repair":
                        bool(
                            pred == gt
                        ),

                    "baseline_gt_logp":
                        base_logp,

                    "gt_logp":
                        post_logp,

                    "delta_logp":
                        (
                            post_logp
                            - base_logp
                        ),

                    "baseline_margin":
                        base_margin,

                    "margin":
                        post_margin,

                    "delta_margin":
                        (
                            post_margin
                            - base_margin
                        ),

                    "calls":
                        calls,
                })

    df = pd.DataFrame(
        rows
    )

    if len(df) != expected:
        raise RuntimeError(
            f"Expected {expected} rows, "
            f"got {len(df)}."
        )

    if df.isna().sum().sum() != 0:
        raise RuntimeError(
            "NaN detected."
        )

    if not (
        df["calls"] > 0
    ).all():
        raise RuntimeError(
            "Some DAM hooks were not called."
        )

    # ========================================================
    # Output path
    # ========================================================

    if limit is None:

        raw_path = (
            OUT_DIR
            / "true_oracle_head_beta_results.csv"
        )

        sample_path = (
            OUT_DIR
            / "true_oracle_per_sample.csv"
        )

        beta_path = (
            OUT_DIR
            / "oracle_by_beta_summary.csv"
        )

    else:

        raw_path = (
            OUT_DIR
            / f"smoke_{limit}_results.csv"
        )

        sample_path = (
            OUT_DIR
            / f"smoke_{limit}_oracle.csv"
        )

        beta_path = (
            OUT_DIR
            / f"smoke_{limit}_beta_summary.csv"
        )

    df.to_csv(
        raw_path,
        index=False,
    )

    # ========================================================
    # Oracle per beta
    #
    # For each sample and beta:
    #   choose configuration with maximum post-GT margin.
    #
    # Margin is the direct decision-boundary quantity.
    # ========================================================

    beta_oracle_rows = []

    for (
        sid,
        beta,
    ), g in df.groupby(
        [
            "sample_id",
            "beta",
        ]
    ):

        best_idx = (
            g[
                "margin"
            ]
            .idxmax()
        )

        beta_oracle_rows.append(
            df.loc[
                best_idx
            ]
        )

    beta_oracle = pd.DataFrame(
        beta_oracle_rows
    )

    beta_summary_rows = []

    for beta, g in (
        beta_oracle.groupby(
            "beta"
        )
    ):

        beta_summary_rows.append({
            "beta":
                float(beta),

            "oracle_repairs":
                int(
                    g[
                        "repair"
                    ].sum()
                ),

            "oracle_repair_rate":
                float(
                    g[
                        "repair"
                    ].mean()
                ),

            "mean_best_margin":
                float(
                    g[
                        "margin"
                    ].mean()
                ),

            "median_best_margin":
                float(
                    g[
                        "margin"
                    ].median()
                ),

            "closest_margin":
                float(
                    g[
                        "margin"
                    ].max()
                ),

            "mean_best_delta_margin":
                float(
                    g[
                        "delta_margin"
                    ].mean()
                ),

            "mean_best_delta_logp":
                float(
                    g[
                        "delta_logp"
                    ].mean()
                ),

            "positive_margin_count":
                int(
                    (
                        g[
                            "margin"
                        ]
                        > 0
                    ).sum()
                ),
        })

    beta_summary = pd.DataFrame(
        beta_summary_rows
    )

    beta_summary.to_csv(
        beta_path,
        index=False,
    )

    # ========================================================
    # TRUE JOINT ORACLE over head × beta
    #
    # Primary oracle:
    # configuration with maximum post-intervention margin.
    # ========================================================

    joint_rows = []

    for sid, g in (
        df.groupby(
            "sample_id"
        )
    ):

        best_idx = (
            g[
                "margin"
            ]
            .idxmax()
        )

        joint_rows.append(
            df.loc[
                best_idx
            ]
        )

    joint = pd.DataFrame(
        joint_rows
    )

    joint.to_csv(
        sample_path,
        index=False,
    )

    repairs = int(
        joint[
            "repair"
        ].sum()
    )

    # ========================================================
    # Secondary oracle by delta-logp
    # ========================================================

    logp_rows = []

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

        logp_rows.append(
            df.loc[
                best_idx
            ]
        )

    logp_oracle = pd.DataFrame(
        logp_rows
    )

    # ========================================================
    # Final report
    # ========================================================

    print(
        "\n" + "=" * 132
    )

    print(
        "ORACLE REPAIR CEILING BY BETA"
    )

    print(
        "=" * 132
    )

    print(
        beta_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\n" + "=" * 132
    )

    print(
        "TRUE JOINT ORACLE "
        "(HEAD × BETA)"
    )

    print(
        "=" * 132
    )

    print(
        "Wrong samples:",
        len(joint),
    )

    print(
        "Repairs:",
        repairs,
        "/",
        len(joint),
    )

    print(
        "Repair rate:",
        float(
            joint[
                "repair"
            ].mean()
        ),
    )

    print(
        "Mean best post margin:",
        float(
            joint[
                "margin"
            ].mean()
        ),
    )

    print(
        "Median best post margin:",
        float(
            joint[
                "margin"
            ].median()
        ),
    )

    print(
        "Closest post margin:",
        float(
            joint[
                "margin"
            ].max()
        ),
    )

    print(
        "Mean best delta margin:",
        float(
            joint[
                "delta_margin"
            ].mean()
        ),
    )

    print(
        "Mean delta logp "
        "under margin oracle:",
        float(
            joint[
                "delta_logp"
            ].mean()
        ),
    )

    print(
        "\nBest head distribution:"
    )

    print(
        joint[
            "head_name"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nBest beta distribution:"
    )

    print(
        joint[
            "beta"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nDELTA-LOGP ORACLE CHECK"
    )

    print(
        "Mean max delta logp:",
        float(
            logp_oracle[
                "delta_logp"
            ].mean()
        ),
    )

    print(
        "Repairs under delta-logp oracle:",
        int(
            logp_oracle[
                "repair"
            ].sum()
        ),
        "/",
        len(logp_oracle),
    )

    print(
        "\nSaved:"
    )

    print(raw_path)
    print(sample_path)
    print(beta_path)

    print(
        "\nDAM RESCUE R1 COMPLETE"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--protocol-only",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        limit=args.limit,
        protocol_only=
            args.protocol_only,
    )
