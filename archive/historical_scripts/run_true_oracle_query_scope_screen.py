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
    "dam_rescue_r2_query_scope"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_PATH = (
    OUT_DIR
    / "query_scope_results.csv"
)

SCOPE_SUMMARY_PATH = (
    OUT_DIR
    / "query_scope_summary.csv"
)

JOINT_PATH = (
    OUT_DIR
    / "joint_oracle_per_sample.csv"
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


SCOPES = [
    "answer",
    "q_many",
    "count_phrase",
]


BETA = 1.25
GAMMA = 0.0


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
    query_scope,
):

    dam = DistributedLayerDAM(
        model=model,
        layer_idx=int(layer),
        head_indices=[
            int(head)
        ],
        object_token_sets=
            object_token_sets,
        beta=BETA,
        gamma=GAMMA,
        query_scope=query_scope,
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
            f"{head_name(layer, head)}, "
            f"scope={query_scope}"
        )

    return result, calls


def load_wrong_samples():

    pairs = pd.read_csv(
        PAIR_PATH
    )

    print(
        "Pairs before exclusion:",
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

    removed = (
        pairs[
            leak_mask
        ]
        .copy()
    )

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
            f"Expected 27 pairs, "
            f"found {len(pairs)}."
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

    if ids & DISCOVERY_SAMPLES:
        raise RuntimeError(
            "Discovery leakage."
        )

    return samples


def main(
    limit=None,
    protocol_only=False,
):

    print("=" * 120)
    print(
        "AROMA DAM RESCUE R2 — "
        "TRUE ORACLE QUERY-SCOPE SCREEN"
    )
    print("=" * 120)

    print(
        "Beta:",
        BETA,
    )

    print(
        "Gamma:",
        GAMMA,
    )

    print(
        "Scopes:",
        SCOPES,
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
        load_wrong_samples()
    )

    full_count = len(
        samples
    )

    if limit is not None:

        if limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        samples = (
            samples[
                :limit
            ]
        )

        print(
            "\n[SMOKE / LIMITED MODE]"
        )

        print(
            "Full wrong-sample pool:",
            full_count,
        )

    expected = (
        len(samples)
        * len(HEADS)
        * len(SCOPES)
    )

    print(
        "Wrong samples selected:",
        len(samples),
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
    # Screen
    # ========================================================

    for item in tqdm(
        samples,
        desc="Query-scope screen",
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
                "Expected wrong sample: "
                f"{sid}, GT={gt}, "
                f"pred={base_pred}"
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
        # Head × query-scope
        # ----------------------------------------------------

        for scope in SCOPES:

            for layer, head in HEADS:

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
                        query_scope=scope,
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

                    "query_scope":
                        scope,

                    "layer":
                        int(layer),

                    "head":
                        int(head),

                    "head_name":
                        head_name(
                            layer,
                            head,
                        ),

                    "beta":
                        BETA,

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

    if (
        df.isna()
        .sum()
        .sum()
        != 0
    ):
        raise RuntimeError(
            "NaN detected."
        )

    if not (
        df["calls"] > 0
    ).all():
        raise RuntimeError(
            "Some hooks were not called."
        )

    # ========================================================
    # Save raw
    # ========================================================

    if limit is None:

        raw_path = RAW_PATH

        scope_summary_path = (
            SCOPE_SUMMARY_PATH
        )

        joint_path = (
            JOINT_PATH
        )

    else:

        raw_path = (
            OUT_DIR
            / f"smoke_{limit}_results.csv"
        )

        scope_summary_path = (
            OUT_DIR
            / f"smoke_{limit}_scope_summary.csv"
        )

        joint_path = (
            OUT_DIR
            / f"smoke_{limit}_joint_oracle.csv"
        )

    df.to_csv(
        raw_path,
        index=False,
    )

    # ========================================================
    # Oracle HEAD within each scope
    #
    # Primary criterion:
    # maximize post-intervention margin.
    # ========================================================

    scope_oracle_rows = []

    for (
        sid,
        scope,
    ), g in (
        df.groupby(
            [
                "sample_id",
                "query_scope",
            ]
        )
    ):

        best_idx = (
            g[
                "margin"
            ]
            .idxmax()
        )

        scope_oracle_rows.append(
            df.loc[
                best_idx
            ]
        )

    scope_oracle = (
        pd.DataFrame(
            scope_oracle_rows
        )
    )

    summary_rows = []

    for scope, g in (
        scope_oracle.groupby(
            "query_scope"
        )
    ):

        summary_rows.append({
            "query_scope":
                scope,

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

            "mean_best_post_margin":
                float(
                    g[
                        "margin"
                    ].mean()
                ),

            "median_best_post_margin":
                float(
                    g[
                        "margin"
                    ].median()
                ),

            "closest_post_margin":
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

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        scope_summary_path,
        index=False,
    )

    # ========================================================
    # TRUE JOINT ORACLE:
    # head × scope
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
        joint_path,
        index=False,
    )

    # ========================================================
    # Secondary logp oracle
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
    # Report
    # ========================================================

    print(
        "\n" + "=" * 132
    )

    print(
        "ORACLE REPAIR CEILING BY QUERY SCOPE"
    )

    print(
        "=" * 132
    )

    print(
        summary.to_string(
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
        "(HEAD × QUERY SCOPE)"
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
        int(
            joint[
                "repair"
            ].sum()
        ),
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
        "Mean delta logp:",
        float(
            joint[
                "delta_logp"
            ].mean()
        ),
    )

    print(
        "\nBest query-scope distribution:"
    )

    print(
        joint[
            "query_scope"
        ]
        .value_counts()
        .to_string()
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

    print(
        raw_path
    )

    print(
        scope_summary_path
    )

    print(
        joint_path
    )

    print(
        "\nDAM RESCUE R2 COMPLETE"
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
