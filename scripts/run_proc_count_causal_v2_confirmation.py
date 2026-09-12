import argparse
import json
import math
import sys
from collections import Counter
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
    numeral_token_ids,
    score_state,
    HeadGainModifier,
    add_row,
)


META_PATH = Path(
    "data/proc_count_causal_v2/"
    "metadata.jsonl"
)

PROTOCOL_PATH = Path(
    "configs/"
    "proc_count_causal_v2_confirmation.json"
)

OUT_DIR = Path(
    "outputs/proc_count_causal_v2/"
    "confirmation_l18h13_a1p5"
)

RESULT_PATH = (
    OUT_DIR
    / "confirmation_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "confirmation_summary.csv"
)

CONDITION_PATH = (
    OUT_DIR
    / "confirmation_by_condition.csv"
)

RUN_META_PATH = (
    OUT_DIR
    / "confirmation_run_metadata.json"
)


FROZEN_ALPHA = 1.5


def load_metadata():

    return [
        json.loads(line)
        for line in META_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def exact_mcnemar_p(
    repairs,
    breaks,
):

    n = int(
        repairs
        + breaks
    )

    if n == 0:
        return 1.0

    k = min(
        int(repairs),
        int(breaks),
    )

    lower = sum(
        math.comb(
            n,
            i,
        )
        for i in range(
            k + 1
        )
    ) / (
        2 ** n
    )

    return min(
        1.0,
        2.0 * lower,
    )


def protocol_audit(
    records,
):

    print("=" * 118)
    print(
        "AROMA PROC-COUNT-CAUSAL v2.1 "
        "FROZEN CONFIRMATION"
    )
    print("=" * 118)

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        protocol[
            "dataset"
        ]
        ==
        "proc_count_causal_v2"
    )

    assert (
        protocol[
            "generation_version"
        ]
        ==
        "2.1_unique"
    )

    assert (
        protocol[
            "n_samples"
        ]
        == 1000
    )

    intervention = (
        protocol[
            "intervention"
        ]
    )

    assert int(
        intervention[
            "layer"
        ]
    ) == LAYER

    assert int(
        intervention[
            "head"
        ]
    ) == HEAD

    assert math.isclose(
        float(
            intervention[
                "alpha"
            ]
        ),
        FROZEN_ALPHA,
    )

    assert (
        intervention[
            "head_name"
        ]
        ==
        "L18H13"
    )

    assert (
        protocol[
            "frozen_before_confirmation_inference"
        ]
        is True
    )

    assert len(
        records
    ) == 1000

    ids = [
        str(
            r[
                "sample_id"
            ]
        )
        for r in records
    ]

    assert len(
        set(ids)
    ) == 1000

    assert all(
        r[
            "dataset"
        ]
        ==
        "proc_count_causal_v2"
        for r in records
    )

    assert all(
        str(
            r[
                "generation_version"
            ]
        )
        ==
        "2.1_unique"
        for r in records
    )

    conditions = Counter(
        r[
            "condition"
        ]
        for r in records
    )

    expected_conditions = {
        "row",
        "grid",
        "random_sparse",
        "dense",
        "distractors",
    }

    assert set(
        conditions
    ) == expected_conditions

    assert all(
        conditions[c] == 200
        for c in expected_conditions
    )

    counts = Counter(
        int(
            r[
                "ground_truth"
            ]
        )
        for r in records
    )

    assert all(
        counts[n] == 100
        for n in range(
            1,
            11,
        )
    )

    print(
        "Samples              :",
        len(records),
    )

    print(
        "Frozen head          :",
        "L18H13",
    )

    print(
        "Frozen alpha         :",
        FROZEN_ALPHA,
    )

    print(
        "Primary subgroup     :",
        protocol[
            "prespecified_primary_subgroup"
        ],
    )

    print(
        "Retuning allowed     :",
        False,
    )

    print(
        "Expected result rows :",
        2000,
    )

    print(
        "\nPROTOCOL AUDIT PASS"
    )

    return protocol


def prepare_resume():

    if not RESULT_PATH.exists():
        return set()

    existing = pd.read_csv(
        RESULT_PATH
    )

    if len(
        existing
    ) == 0:
        return set()

    complete_ids = []

    for sid, g in (
        existing.groupby(
            "sample_id"
        )
    ):

        alphas = {
            round(
                float(a),
                6,
            )
            for a in g[
                "alpha"
            ]
        }

        if (
            len(g) == 2
            and
            alphas
            ==
            {
                1.0,
                1.5,
            }
        ):
            complete_ids.append(
                str(sid)
            )

    complete_ids = set(
        complete_ids
    )

    # Drop any incomplete / duplicated samples before resuming.
    clean = existing[
        existing[
            "sample_id"
        ]
        .astype(str)
        .isin(
            complete_ids
        )
    ].copy()

    clean.to_csv(
        RESULT_PATH,
        index=False,
    )

    return complete_ids


def summarize(
    df,
):

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    assert len(
        df
    ) == 2000

    assert (
        df[
            "sample_id"
        ]
        .nunique()
        == 1000
    )

    for sid, g in (
        df.groupby(
            "sample_id"
        )
    ):

        assert len(g) == 2

        alphas = {
            round(
                float(x),
                6,
            )
            for x in g[
                "alpha"
            ]
        }

        assert alphas == {
            1.0,
            1.5,
        }

    assert (
        df.isna()
        .sum()
        .sum()
        == 0
    )

    baseline = df[
        np.isclose(
            df[
                "alpha"
            ],
            1.0,
        )
    ].copy()

    gain = df[
        np.isclose(
            df[
                "alpha"
            ],
            1.5,
        )
    ].copy()

    assert len(
        baseline
    ) == 1000

    assert len(
        gain
    ) == 1000

    # --------------------------------------------------------
    # Overall frozen confirmation
    # --------------------------------------------------------

    baseline_correct = (
        gain[
            "baseline_correct"
        ].astype(bool)
    )

    post_correct = (
        gain[
            "modulated_correct"
        ].astype(bool)
    )

    repairs = int(
        (
            (~baseline_correct)
            &
            post_correct
        ).sum()
    )

    breaks = int(
        (
            baseline_correct
            &
            (~post_correct)
        ).sum()
    )

    baseline_n = int(
        baseline_correct.sum()
    )

    baseline_wrong = int(
        (
            ~baseline_correct
        ).sum()
    )

    post_n = int(
        post_correct.sum()
    )

    baseline_acc = float(
        baseline_correct.mean()
    )

    post_acc = float(
        post_correct.mean()
    )

    overall = {
        "scope":
            "ALL1000",

        "n":
            1000,

        "baseline_correct":
            baseline_n,

        "baseline_wrong":
            baseline_wrong,

        "baseline_accuracy":
            baseline_acc,

        "post_correct":
            post_n,

        "post_accuracy":
            post_acc,

        "accuracy_change":
            post_acc
            - baseline_acc,

        "repairs":
            repairs,

        "breaks":
            breaks,

        "net_repairs":
            repairs
            - breaks,

        "repair_rate_among_wrong":
            (
                repairs
                / baseline_wrong
                if baseline_wrong
                else 0.0
            ),

        "break_rate_among_correct":
            (
                breaks
                / baseline_n
                if baseline_n
                else 0.0
            ),

        "mcnemar_exact_p":
            exact_mcnemar_p(
                repairs,
                breaks,
            ),

        "mean_delta_gt_logp":
            float(
                gain[
                    "delta_gt_logp"
                ].mean()
            ),
    }

    summary = pd.DataFrame([
        overall
    ])

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Pre-specified condition analysis
    # --------------------------------------------------------

    rows = []

    for condition, g in (
        gain.groupby(
            "condition"
        )
    ):

        base_correct = (
            g[
                "baseline_correct"
            ].astype(bool)
        )

        after_correct = (
            g[
                "modulated_correct"
            ].astype(bool)
        )

        r = int(
            (
                (~base_correct)
                &
                after_correct
            ).sum()
        )

        b = int(
            (
                base_correct
                &
                (~after_correct)
            ).sum()
        )

        n_correct = int(
            base_correct.sum()
        )

        n_wrong = int(
            (
                ~base_correct
            ).sum()
        )

        base_acc = float(
            base_correct.mean()
        )

        after_acc = float(
            after_correct.mean()
        )

        rows.append({
            "condition":
                condition,

            "n":
                len(g),

            "baseline_correct":
                n_correct,

            "baseline_wrong":
                n_wrong,

            "baseline_accuracy":
                base_acc,

            "post_accuracy":
                after_acc,

            "accuracy_change":
                after_acc
                - base_acc,

            "repairs":
                r,

            "breaks":
                b,

            "net_repairs":
                r - b,

            "repair_rate_among_wrong":
                (
                    r / n_wrong
                    if n_wrong
                    else 0.0
                ),

            "break_rate_among_correct":
                (
                    b / n_correct
                    if n_correct
                    else 0.0
                ),

            "mcnemar_exact_p":
                exact_mcnemar_p(
                    r,
                    b,
                ),

            "mean_delta_gt_logp":
                float(
                    g[
                        "delta_gt_logp"
                    ].mean()
                ),
        })

    by_condition = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "condition"
        )
    )

    by_condition.to_csv(
        CONDITION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print frozen confirmation only.
    # No oracle. No tuning.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 132
    )

    print(
        "FINAL FROZEN v2 CONFIRMATION"
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
        "PRE-SPECIFIED CONDITION RESULTS"
    )

    print(
        "=" * 132
    )

    print(
        by_condition.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    distractor = (
        by_condition[
            by_condition[
                "condition"
            ]
            ==
            "distractors"
        ]
    )

    assert len(
        distractor
    ) == 1

    print(
        "\n" + "=" * 132
    )

    print(
        "PRE-SPECIFIED PRIMARY SUBGROUP: DISTRACTORS"
    )

    print(
        "=" * 132
    )

    print(
        distractor.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
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
        CONDITION_PATH
    )

    print(
        "\nFROZEN CONFIRMATION COMPLETE"
    )


def main(
    resume=False,
    protocol_only=False,
):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = (
        load_metadata()
    )

    protocol = (
        protocol_audit(
            records
        )
    )

    if protocol_only:
        print(
            "\nNo model was loaded."
        )
        return

    if resume:

        completed = (
            prepare_resume()
        )

    else:

        completed = set()

        if RESULT_PATH.exists():
            RESULT_PATH.unlink()

    print(
        "\nAlready completed:",
        len(completed),
    )

    print(
        "Remaining:",
        1000
        - len(completed),
    )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    numeral_ids = (
        numeral_token_ids(
            processor
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

    ordered = sorted(
        records,
        key=lambda r:
            str(
                r[
                    "sample_id"
                ]
            ),
    )

    remaining = [
        r
        for r in ordered
        if str(
            r[
                "sample_id"
            ]
        )
        not in completed
    ]

    for sample in tqdm(
        remaining,
        desc="Frozen v2 confirmation",
    ):

        sid = str(
            sample[
                "sample_id"
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

        inputs = (
            prepare_inputs(
                processor,
                image,
            )
        )

        inputs = (
            move_inputs(
                inputs,
                model,
            )
        )

        gt = int(
            sample[
                "ground_truth"
            ]
        )

        baseline = (
            score_state(
                model,
                inputs,
                numeral_ids,
                gt,
            )
        )

        local_rows = []

        add_row(
            local_rows,
            sample,
            baseline,
            baseline,
            alpha=1.0,
            calls=0,
        )

        modifier = (
            HeadGainModifier(
                model=model,
                layer_idx=LAYER,
                head_idx=HEAD,
                alpha=
                    FROZEN_ALPHA,
            )
        )

        modifier.register()

        try:

            gain_state = (
                score_state(
                    model,
                    inputs,
                    numeral_ids,
                    gt,
                )
            )

        finally:

            modifier.remove()

        if (
            modifier.calls
            <= 0
        ):
            raise RuntimeError(
                "Gain hook not called: "
                + sid
            )

        add_row(
            local_rows,
            sample,
            baseline,
            gain_state,
            alpha=
                FROZEN_ALPHA,
            calls=
                modifier.calls,
        )

        local_df = (
            pd.DataFrame(
                local_rows
            )
        )

        local_df.to_csv(
            RESULT_PATH,
            mode="a",
            header=(
                not RESULT_PATH.exists()
            ),
            index=False,
        )

    final_df = pd.read_csv(
        RESULT_PATH
    )

    summarize(
        final_df
    )

    run_meta = {
        "protocol":
            protocol[
                "protocol_name"
            ],

        "dataset":
            "proc_count_causal_v2",

        "generation_version":
            "2.1_unique",

        "samples":
            1000,

        "head":
            "L18H13",

        "alpha":
            1.5,

        "retuned_on_v2":
            False,
    }

    RUN_META_PATH.write_text(
        json.dumps(
            run_meta,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--protocol-only",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        resume=args.resume,
        protocol_only=
            args.protocol_only,
    )
