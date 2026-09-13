import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

sys.path.insert(0, "scripts")

from run_l18h13_gain_all300 import (
    MODEL_ID,
    LAYER,
    HEAD,
    prepare_inputs,
    move_inputs,
    HeadGainModifier,
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
    NUMERALS,
)


META_PATH = Path(
    "data/proc_count_causal_v2/"
    "metadata.jsonl"
)

BASELINE_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5_expanded_0_15/"
    "confirmation_results.csv"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "signed_steering/"
    "down_alpha0p5_expanded"
)

OUT_PATH = (
    OUT_DIR
    / "down_alpha0p5_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "down_alpha0p5_summary.csv"
)

CONDITION_PATH = (
    OUT_DIR
    / "down_alpha0p5_by_condition.csv"
)

ALPHA = 0.5


def load_metadata():

    result = {}

    for line in META_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        record = json.loads(
            line
        )

        result[
            str(
                record["sample_id"]
            )
        ] = record

    return result


def load_reference_baseline():

    df = pd.read_csv(
        BASELINE_PATH
    )

    if "alpha" not in df.columns:
        raise RuntimeError(
            "Reference file lacks alpha column."
        )

    identity = df[
        np.isclose(
            df["alpha"],
            1.0,
        )
    ].copy()

    if len(identity) != 1000:
        raise RuntimeError(
            "Expected exactly 1000 alpha=1 "
            f"reference rows, found {len(identity)}."
        )

    if (
        identity[
            "sample_id"
        ].nunique()
        != 1000
    ):
        raise RuntimeError(
            "Reference baseline sample IDs "
            "are not unique."
        )

    mismatch = (
        identity[
            "baseline_best_numeral"
        ].astype(int)
        !=
        identity[
            "modulated_best_numeral"
        ].astype(int)
    )

    if mismatch.any():
        raise RuntimeError(
            "Reference alpha=1 identity "
            "prediction audit failed."
        )

    return (
        identity
        .set_index(
            "sample_id"
        )
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Independent-v2 expanded-space "
            "L18H13 alpha=0.5 downward "
            "steering confirmation."
        )
    )

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

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata()

    baseline = (
        load_reference_baseline()
    )

    sample_ids = sorted(
        metadata.keys()
    )

    if len(sample_ids) != 1000:
        raise RuntimeError(
            f"Expected 1000 v2 samples, "
            f"found {len(sample_ids)}."
        )

    if set(sample_ids) != set(
        baseline.index.astype(str)
    ):
        raise RuntimeError(
            "Metadata/reference sample-set "
            "mismatch."
        )

    print("=" * 112)
    print(
        "AROMA V2 EXPANDED DOWNWARD "
        "STEERING CONFIRMATION"
    )
    print("=" * 112)

    print(
        "Samples             :",
        len(sample_ids),
    )

    print(
        "Model               :",
        MODEL_ID,
    )

    print(
        "Head                :",
        f"L{LAYER}H{HEAD}",
    )

    print(
        "Alpha               :",
        ALPHA,
    )

    print(
        "Numeral space       :",
        f"{min(NUMERALS)}..{max(NUMERALS)}",
    )

    print(
        "Reference baseline  :",
        BASELINE_PATH,
    )

    print(
        "\nProtocol audit: PASS"
    )

    if args.protocol_only:

        print(
            "No model was loaded."
        )

        return

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        sample_ids = sample_ids[
            :args.limit
        ]

        print(
            "\n[SMOKE / LIMITED MODE]"
        )

        print(
            "Selected samples:",
            len(sample_ids),
        )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    numeral_ids = numeral_token_ids(
        processor
    )

    print(
        "Numeral IDs:",
        numeral_ids,
    )

    print(
        "\nLoading model..."
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

    rows = []

    for sid in tqdm(
        sample_ids,
        desc="Downward alpha=0.5",
    ):

        sample = metadata[
            sid
        ]

        ref = baseline.loc[
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

        inputs = prepare_inputs(
            processor,
            image,
        )

        inputs = move_inputs(
            inputs,
            model,
        )

        modifier = HeadGainModifier(
            model=model,
            layer_idx=LAYER,
            head_idx=HEAD,
            alpha=ALPHA,
        )

        modifier.register()

        try:

            state = score_state(
                model=model,
                inputs=inputs,
                numeral_ids=numeral_ids,
                gt=gt,
            )

        finally:

            modifier.remove()

        if modifier.calls <= 0:

            raise RuntimeError(
                f"Hook not called for {sid}."
            )

        base_best = int(
            ref[
                "baseline_best_numeral"
            ]
        )

        mod_best = int(
            state[
                "best_numeral"
            ]
        )

        base_correct = (
            base_best
            == gt
        )

        mod_correct = (
            mod_best
            == gt
        )

        row = {
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

            "replicate":
                int(
                    sample[
                        "replicate"
                    ]
                ),

            "alpha":
                ALPHA,

            "baseline_best_numeral":
                base_best,

            "modulated_best_numeral":
                mod_best,

            "prediction_shift":
                (
                    mod_best
                    -
                    base_best
                ),

            "baseline_error":
                (
                    base_best
                    -
                    gt
                ),

            "baseline_correct":
                bool(
                    base_correct
                ),

            "modulated_correct":
                bool(
                    mod_correct
                ),

            "repair":
                bool(
                    (not base_correct)
                    and
                    mod_correct
                ),

            "break_case":
                bool(
                    base_correct
                    and
                    (not mod_correct)
                ),

            "baseline_gt_logp":
                float(
                    ref[
                        "baseline_gt_logp"
                    ]
                ),

            "modulated_gt_logp":
                float(
                    state[
                        "gt_logp"
                    ]
                ),

            "delta_gt_logp":
                float(
                    state[
                        "gt_logp"
                    ]
                    -
                    float(
                        ref[
                            "baseline_gt_logp"
                        ]
                    )
                ),

            "baseline_margin":
                float(
                    ref[
                        "baseline_margin"
                    ]
                ),

            "modulated_margin":
                float(
                    state[
                        "margin"
                    ]
                ),

            "delta_margin":
                float(
                    state[
                        "margin"
                    ]
                    -
                    float(
                        ref[
                            "baseline_margin"
                        ]
                    )
                ),

            "baseline_expected_numeral":
                float(
                    ref[
                        "baseline_expected_numeral"
                    ]
                ),

            "modulated_expected_numeral":
                float(
                    state[
                        "expected_numeral"
                    ]
                ),

            "expected_numeral_shift":
                float(
                    state[
                        "expected_numeral"
                    ]
                    -
                    float(
                        ref[
                            "baseline_expected_numeral"
                        ]
                    )
                ),

            "hook_calls":
                int(
                    modifier.calls
                ),
        }

        p1 = np.asarray(
            state[
                "conditional_probs"
            ],
            dtype=float,
        )

        for i, n in enumerate(
            NUMERALS
        ):

            base_col = (
                f"baseline_numprob_{n}"
            )

            if base_col not in ref.index:

                raise RuntimeError(
                    f"Missing reference column "
                    f"{base_col}"
                )

            p0 = float(
                ref[
                    base_col
                ]
            )

            row[
                f"baseline_numprob_{n}"
            ] = p0

            row[
                f"modulated_numprob_{n}"
            ] = float(
                p1[i]
            )

            row[
                f"delta_numprob_{n}"
            ] = float(
                p1[i]
                -
                p0
            )

        rows.append(
            row
        )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUT_PATH,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    n = len(out)

    baseline_correct = int(
        out[
            "baseline_correct"
        ].sum()
    )

    post_correct = int(
        out[
            "modulated_correct"
        ].sum()
    )

    repairs = int(
        out[
            "repair"
        ].sum()
    )

    breaks = int(
        out[
            "break_case"
        ].sum()
    )

    changed = out[
        out[
            "prediction_shift"
        ]
        != 0
    ].copy()

    summary = pd.DataFrame([
        {
            "n":
                n,

            "baseline_correct":
                baseline_correct,

            "baseline_accuracy":
                baseline_correct
                / n,

            "post_correct":
                post_correct,

            "post_accuracy":
                post_correct
                / n,

            "accuracy_change":
                (
                    post_correct
                    -
                    baseline_correct
                )
                / n,

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                repairs
                -
                breaks,

            "changed_predictions":
                len(
                    changed
                ),

            "downward_changed":
                int(
                    (
                        changed[
                            "prediction_shift"
                        ]
                        < 0
                    ).sum()
                ),

            "upward_changed":
                int(
                    (
                        changed[
                            "prediction_shift"
                        ]
                        > 0
                    ).sum()
                ),

            "mean_expected_shift":
                float(
                    out[
                        "expected_numeral_shift"
                    ].mean()
                ),

            "median_expected_shift":
                float(
                    out[
                        "expected_numeral_shift"
                    ].median()
                ),

            "negative_expected_shift_fraction":
                float(
                    (
                        out[
                            "expected_numeral_shift"
                        ]
                        < 0
                    ).mean()
                ),

            "positive_expected_shift_fraction":
                float(
                    (
                        out[
                            "expected_numeral_shift"
                        ]
                        > 0
                    ).mean()
                ),

            "mean_delta_gt_logp":
                float(
                    out[
                        "delta_gt_logp"
                    ].mean()
                ),
        }
    ])

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # By condition
    # ========================================================

    condition_rows = []

    for condition, g in out.groupby(
        "condition"
    ):

        r = int(
            g[
                "repair"
            ].sum()
        )

        b = int(
            g[
                "break_case"
            ].sum()
        )

        c = g[
            g[
                "prediction_shift"
            ]
            != 0
        ]

        condition_rows.append({
            "condition":
                condition,

            "n":
                len(g),

            "baseline_accuracy":
                float(
                    g[
                        "baseline_correct"
                    ].mean()
                ),

            "post_accuracy":
                float(
                    g[
                        "modulated_correct"
                    ].mean()
                ),

            "repairs":
                r,

            "breaks":
                b,

            "net_repairs":
                r - b,

            "changed_predictions":
                len(c),

            "downward_changed":
                int(
                    (
                        c[
                            "prediction_shift"
                        ]
                        < 0
                    ).sum()
                ),

            "upward_changed":
                int(
                    (
                        c[
                            "prediction_shift"
                        ]
                        > 0
                    ).sum()
                ),

            "mean_expected_shift":
                float(
                    g[
                        "expected_numeral_shift"
                    ].mean()
                ),

            "negative_expected_shift_fraction":
                float(
                    (
                        g[
                            "expected_numeral_shift"
                        ]
                        < 0
                    ).mean()
                ),
        })

    by_condition = (
        pd.DataFrame(
            condition_rows
        )
        .sort_values(
            "condition"
        )
    )

    by_condition.to_csv(
        CONDITION_PATH,
        index=False,
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "\n" + "=" * 112
    )

    print(
        "FINAL V2 DOWNWARD STEERING RESULTS"
    )

    print(
        "=" * 112
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nPrediction-shift distribution:"
    )

    if len(changed):

        print(
            changed[
                "prediction_shift"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        print(
            "NO CHANGED PREDICTIONS"
        )

    print(
        "\nRepair transition patterns:"
    )

    repair_df = out[
        out[
            "repair"
        ]
    ]

    if len(repair_df):

        print(
            repair_df.groupby(
                [
                    "baseline_error",
                    "prediction_shift",
                ]
            )
            .size()
            .to_string()
        )

    else:

        print(
            "NO REPAIRS"
        )

    print(
        "\nBreak transition patterns:"
    )

    break_df = out[
        out[
            "break_case"
        ]
    ]

    if len(break_df):

        print(
            break_df[
                "prediction_shift"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        print(
            "NO BREAKS"
        )

    print(
        "\nBY CONDITION"
    )

    print(
        by_condition.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        OUT_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        CONDITION_PATH
    )

    print(
        "\nV2 DOWNWARD CONFIRMATION COMPLETE"
    )


if __name__ == "__main__":
    main()
