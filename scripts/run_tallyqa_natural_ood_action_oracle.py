import argparse
import hashlib
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
    move_inputs,
    HeadGainModifier,
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)

from run_tallyqa_natural_ood_v1_final import (
    prepare_tallyqa_inputs,
)


# ============================================================
# Frozen inputs
# ============================================================

MANIFEST_PATH = Path(
    "data/tallyqa_natural_ood_v1/"
    "manifest.jsonl"
)

IMAGE_ROOT = Path(
    "data/tallyqa_natural_ood_v1/"
    "images"
)

FINAL_RESULT_PATH = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "final_frozen_controller/"
    "tallyqa_final_results.csv"
)

PROTOCOL_PATH = Path(
    "configs/"
    "tallyqa_natural_ood_v1_action_oracle_forensics.json"
)

OUT_DIR = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "forensics/action_oracle"
)

MATRIX_PATH = (
    OUT_DIR
    / "action_matrix.csv"
)

ACTION_SUMMARY_PATH = (
    OUT_DIR
    / "action_summary.csv"
)

ORACLE_SUMMARY_PATH = (
    OUT_DIR
    / "oracle_summary.csv"
)

ORACLE_SUBSET_PATH = (
    OUT_DIR
    / "oracle_by_subset.csv"
)

ORACLE_ACTION_PATH = (
    OUT_DIR
    / "oracle_action_distribution.csv"
)

EXPECTED_MANIFEST_SHA256 = (
    "068a489a02d499da6eb0b837a81c8817583b64b51c23f944c77645420fe2c32c"
)

ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]

NONNOOP_ACTIONS = [
    0.0,
    1.5,
    2.0,
    4.0,
]

DUMMY_GT = 1


# ============================================================
# Utilities
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_jsonl(path):

    return [
        json.loads(line)
        for line in Path(path)
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]


def protocol_audit():

    print("=" * 112)
    print(
        "AROMA TALLYQA NATURAL-OOD "
        "POST-HOC ACTION-ORACLE FORENSICS"
    )
    print("=" * 112)

    required = [
        MANIFEST_PATH,
        FINAL_RESULT_PATH,
        PROTOCOL_PATH,
    ]

    for path in required:

        if not path.exists():

            raise RuntimeError(
                f"Missing required file: {path}"
            )

    if (
        sha256_file(
            MANIFEST_PATH
        )
        !=
        EXPECTED_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Frozen manifest hash mismatch."
        )

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    runner_hash = sha256_file(
        Path(__file__)
    )

    if (
        runner_hash
        !=
        protocol[
            "runner_source_sha256"
        ]
    ):

        raise RuntimeError(
            "Forensic runner hash mismatch."
        )

    final_hash = sha256_file(
        FINAL_RESULT_PATH
    )

    if (
        final_hash
        !=
        protocol[
            "frozen_final_results_sha256"
        ]
    ):

        raise RuntimeError(
            "Frozen final-result hash mismatch."
        )

    if (
        protocol[
            "analysis_status"
        ]
        !=
        "post_hoc_forensic_only"
    ):

        raise RuntimeError(
            "Forensic-analysis status mismatch."
        )

    if (
        protocol[
            "confirmatory_claim_allowed"
        ]
        is not False
    ):

        raise RuntimeError(
            "Forensic protocol incorrectly "
            "allows confirmatory claims."
        )

    records = load_jsonl(
        MANIFEST_PATH
    )

    final = pd.read_csv(
        FINAL_RESULT_PATH
    )

    if len(records) != 4000:
        raise RuntimeError(
            "Expected 4000 manifest rows."
        )

    if len(final) != 4000:
        raise RuntimeError(
            "Expected 4000 archived final rows."
        )

    if (
        final[
            "question_id"
        ]
        .nunique()
        != 4000
    ):
        raise RuntimeError(
            "Archived final question IDs "
            "are not unique."
        )

    if sorted(
        float(x)
        for x in protocol[
            "actions"
        ]
    ) != ACTIONS:

        raise RuntimeError(
            "Action set mismatch."
        )

    print(
        "Manifest rows             :",
        len(records),
    )

    print(
        "Archived final rows       :",
        len(final),
    )

    print(
        "Actions                   :",
        ACTIONS,
    )

    print(
        "Head                      : L18H13"
    )

    print(
        "Analysis status           : "
        "POST-HOC FORENSIC ONLY"
    )

    print(
        "Confirmatory claim allowed: False"
    )

    print(
        "Manifest hash audit       : PASS"
    )

    print(
        "Frozen result hash audit  : PASS"
    )

    print(
        "Runner hash audit         : PASS"
    )

    print(
        "\nFORENSIC PROTOCOL AUDIT: PASS"
    )

    return (
        records,
        final,
    )


# ============================================================
# Resume support
# ============================================================

def completed_samples():

    if not MATRIX_PATH.exists():
        return set()

    df = pd.read_csv(
        MATRIX_PATH
    )

    complete = set()

    for qid, g in (
        df.groupby(
            "question_id"
        )
    ):

        alphas = {
            round(
                float(x),
                6,
            )
            for x in g[
                "alpha"
            ]
        }

        if (
            len(g) == 5
            and
            alphas
            ==
            {
                0.0,
                1.0,
                1.5,
                2.0,
                4.0,
            }
        ):

            complete.add(
                int(qid)
            )

    return complete


# ============================================================
# Analysis
# ============================================================

def summarize(
    matrix,
    archived,
):

    if len(matrix) != 20000:

        raise RuntimeError(
            f"Expected 20000 matrix rows, "
            f"found {len(matrix)}."
        )

    if (
        matrix[
            "question_id"
        ]
        .nunique()
        != 4000
    ):

        raise RuntimeError(
            "Action matrix does not contain "
            "4000 unique questions."
        )

    # --------------------------------------------------------
    # Baseline identity audit.
    # --------------------------------------------------------

    baseline = (
        matrix[
            np.isclose(
                matrix[
                    "alpha"
                ],
                1.0,
            )
        ]
        .copy()
    )

    archived_base = (
        archived[
            [
                "question_id",
                "baseline_prediction",
                "post_prediction",
                "selected_alpha",
                "baseline_correct",
                "post_correct",
                "subset",
            ]
        ]
        .copy()
    )

    merged_base = (
        baseline.merge(
            archived_base,
            on="question_id",
            how="inner",
            suffixes=(
                "_fresh",
                "_archived",
            ),
        )
    )

    baseline_mismatches = int(
        (
            merged_base[
                "action_prediction"
            ]
            !=
            merged_base[
                "baseline_prediction"
            ]
        ).sum()
    )

    if baseline_mismatches != 0:

        raise RuntimeError(
            "Fresh baseline does not reproduce "
            "archived baseline."
        )

    # --------------------------------------------------------
    # Selected-action reproducibility audit.
    # --------------------------------------------------------

    selected = archived[
        [
            "question_id",
            "selected_alpha",
            "post_prediction",
        ]
    ].copy()

    selected[
        "selected_alpha"
    ] = selected[
        "selected_alpha"
    ].astype(float)

    selected_check = (
        matrix.merge(
            selected,
            left_on=[
                "question_id",
                "alpha",
            ],
            right_on=[
                "question_id",
                "selected_alpha",
            ],
            how="inner",
        )
    )

    if len(
        selected_check
    ) != 4000:

        raise RuntimeError(
            "Could not recover exactly one "
            "selected action per sample."
        )

    post_mismatches = int(
        (
            selected_check[
                "action_prediction"
            ]
            !=
            selected_check[
                "post_prediction"
            ]
        ).sum()
    )

    if post_mismatches != 0:

        raise RuntimeError(
            "Fresh action matrix does not "
            "reproduce archived selected-action "
            "predictions."
        )

    print(
        "\nBaseline prediction mismatches:",
        baseline_mismatches,
    )

    print(
        "Selected-action post mismatches:",
        post_mismatches,
    )

    # --------------------------------------------------------
    # Fixed-action results.
    # --------------------------------------------------------

    action_rows = []

    for alpha in ACTIONS:

        g = matrix[
            np.isclose(
                matrix[
                    "alpha"
                ],
                alpha,
            )
        ].copy()

        base_correct = (
            g[
                "baseline_correct"
            ]
            .astype(bool)
        )

        action_correct = (
            g[
                "action_correct"
            ]
            .astype(bool)
        )

        repairs = int(
            (
                (~base_correct)
                &
                action_correct
            ).sum()
        )

        breaks = int(
            (
                base_correct
                &
                (~action_correct)
            ).sum()
        )

        action_rows.append({
            "alpha":
                float(alpha),

            "n":
                len(g),

            "baseline_accuracy":
                float(
                    base_correct.mean()
                ),

            "post_accuracy":
                float(
                    action_correct.mean()
                ),

            "accuracy_change":
                float(
                    action_correct.mean()
                    -
                    base_correct.mean()
                ),

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net_repairs":
                repairs
                -
                breaks,

            "changed_predictions":
                int(
                    (
                        g[
                            "action_prediction"
                        ]
                        !=
                        g[
                            "baseline_prediction"
                        ]
                    ).sum()
                ),

            "mean_expected_shift":
                float(
                    g[
                        "expected_numeral_shift"
                    ].mean()
                ),
        })

    action_summary = (
        pd.DataFrame(
            action_rows
        )
    )

    action_summary.to_csv(
        ACTION_SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Oracle repairability among baseline-wrong samples.
    #
    # Correct samples always keep alpha=1.0.
    # Wrong samples count as repairable iff >=1 fixed
    # non-NOOP action produces the GT numeral.
    # --------------------------------------------------------

    baseline_rows = baseline[
        [
            "question_id",
            "subset",
            "ground_truth",
            "baseline_prediction",
            "baseline_correct",
        ]
    ].copy()

    wrong = baseline_rows[
        ~baseline_rows[
            "baseline_correct"
        ].astype(bool)
    ].copy()

    successful = (
        matrix[
            (
                ~np.isclose(
                    matrix[
                        "alpha"
                    ],
                    1.0,
                )
            )
            &
            (
                matrix[
                    "action_correct"
                ].astype(bool)
            )
        ]
        .copy()
    )

    successful_by_qid = {}

    for qid, g in (
        successful.groupby(
            "question_id"
        )
    ):

        alphas = sorted(
            float(a)
            for a in g[
                "alpha"
            ].tolist()
        )

        successful_by_qid[
            int(qid)
        ] = alphas

    repairable_ids = {
        int(qid)
        for qid in wrong[
            "question_id"
        ]
        if int(qid)
        in successful_by_qid
    }

    unique_repaired_wrong = len(
        repairable_ids
    )

    baseline_correct_n = int(
        baseline_rows[
            "baseline_correct"
        ]
        .astype(bool)
        .sum()
    )

    oracle_correct = (
        baseline_correct_n
        +
        unique_repaired_wrong
    )

    oracle_accuracy = (
        oracle_correct
        /
        4000
    )

    baseline_accuracy = (
        baseline_correct_n
        /
        4000
    )

    controller_accuracy = float(
        archived[
            "post_correct"
        ]
        .astype(bool)
        .mean()
    )

    controller_repairs = int(
        (
            (~archived[
                "baseline_correct"
            ].astype(bool))
            &
            archived[
                "post_correct"
            ].astype(bool)
        ).sum()
    )

    # --------------------------------------------------------
    # How many oracle-repairable wrongs did controller capture?
    # --------------------------------------------------------

    archived_wrong = (
        archived[
            ~archived[
                "baseline_correct"
            ].astype(bool)
        ]
        .copy()
    )

    oracle_repairable_archived = (
        archived_wrong[
            archived_wrong[
                "question_id"
            ]
            .astype(int)
            .isin(
                repairable_ids
            )
        ]
    )

    controller_captured = int(
        oracle_repairable_archived[
            "post_correct"
        ]
        .astype(bool)
        .sum()
    )

    capture_fraction = (
        controller_captured
        /
        unique_repaired_wrong
        if unique_repaired_wrong
        else 0.0
    )

    oracle_summary = pd.DataFrame([
        {
            "n":
                4000,

            "baseline_correct":
                baseline_correct_n,

            "baseline_wrong":
                len(
                    wrong
                ),

            "baseline_accuracy":
                baseline_accuracy,

            "unique_oracle_repairable_wrong":
                unique_repaired_wrong,

            "repairable_fraction_of_wrong":
                (
                    unique_repaired_wrong
                    /
                    len(wrong)
                ),

            "oracle_correct":
                oracle_correct,

            "oracle_accuracy":
                oracle_accuracy,

            "oracle_accuracy_gain":
                (
                    oracle_accuracy
                    -
                    baseline_accuracy
                ),

            "controller_accuracy":
                controller_accuracy,

            "controller_accuracy_change":
                (
                    controller_accuracy
                    -
                    baseline_accuracy
                ),

            "controller_repairs":
                controller_repairs,

            "controller_captured_oracle_repairs":
                controller_captured,

            "controller_repair_capture_fraction":
                capture_fraction,

            "oracle_minus_controller_accuracy":
                (
                    oracle_accuracy
                    -
                    controller_accuracy
                ),
        }
    ])

    oracle_summary.to_csv(
        ORACLE_SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Oracle by Simple / Complex.
    # --------------------------------------------------------

    subset_rows = []

    for subset in [
        "simple",
        "complex",
    ]:

        b = baseline_rows[
            baseline_rows[
                "subset"
            ]
            ==
            subset
        ].copy()

        b_correct = (
            b[
                "baseline_correct"
            ]
            .astype(bool)
        )

        b_wrong = b[
            ~b_correct
        ]

        repairable_subset = {
            int(qid)
            for qid in b_wrong[
                "question_id"
            ]
            if int(qid)
            in repairable_ids
        }

        n_correct = int(
            b_correct.sum()
        )

        oracle_correct_subset = (
            n_correct
            +
            len(
                repairable_subset
            )
        )

        subset_rows.append({
            "subset":
                subset,

            "n":
                len(b),

            "baseline_correct":
                n_correct,

            "baseline_wrong":
                len(
                    b_wrong
                ),

            "baseline_accuracy":
                (
                    n_correct
                    /
                    len(b)
                ),

            "oracle_repairable_wrong":
                len(
                    repairable_subset
                ),

            "repairable_fraction_of_wrong":
                (
                    len(
                        repairable_subset
                    )
                    /
                    len(
                        b_wrong
                    )
                    if len(
                        b_wrong
                    )
                    else 0.0
                ),

            "oracle_accuracy":
                (
                    oracle_correct_subset
                    /
                    len(b)
                ),

            "oracle_accuracy_gain":
                (
                    oracle_correct_subset
                    /
                    len(b)
                    -
                    n_correct
                    /
                    len(b)
                ),
        })

    oracle_subset = (
        pd.DataFrame(
            subset_rows
        )
    )

    oracle_subset.to_csv(
        ORACLE_SUBSET_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Closest-to-NOOP successful oracle action distribution.
    # This is descriptive only.
    # --------------------------------------------------------

    oracle_action_counter = Counter()

    for qid in sorted(
        repairable_ids
    ):

        candidates = (
            successful_by_qid[
                qid
            ]
        )

        chosen = min(
            candidates,
            key=lambda a:
                (
                    abs(
                        a
                        -
                        1.0
                    ),
                    a,
                ),
        )

        oracle_action_counter[
            chosen
        ] += 1

    oracle_action_df = pd.DataFrame([
        {
            "oracle_alpha":
                alpha,

            "count":
                oracle_action_counter.get(
                    alpha,
                    0,
                ),

            "fraction_of_repairable":
                (
                    oracle_action_counter.get(
                        alpha,
                        0,
                    )
                    /
                    unique_repaired_wrong
                    if unique_repaired_wrong
                    else 0.0
                ),
        }
        for alpha in NONNOOP_ACTIONS
    ])

    oracle_action_df.to_csv(
        ORACLE_ACTION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 112
    )

    print(
        "FIXED ACTION RESULTS"
    )

    print(
        "=" * 112
    )

    print(
        action_summary.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "NATURAL-OOD MULTI-ACTION "
        "ORACLE CEILING"
    )

    print(
        "=" * 112
    )

    print(
        oracle_summary.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "ORACLE BY SUBSET"
    )

    print(
        "=" * 112
    )

    print(
        oracle_subset.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 112
    )

    print(
        "ORACLE SUCCESSFUL ACTION DISTRIBUTION"
    )

    print(
        "=" * 112
    )

    print(
        oracle_action_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    for path in [
        MATRIX_PATH,
        ACTION_SUMMARY_PATH,
        ORACLE_SUMMARY_PATH,
        ORACLE_SUBSET_PATH,
        ORACLE_ACTION_PATH,
    ]:

        print(
            path
        )

    print(
        "\nACTION-ORACLE FORENSICS COMPLETE"
    )


# ============================================================
# Main
# ============================================================

def main(
    resume=False,
    protocol_only=False,
):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        records,
        archived,
    ) = protocol_audit()

    if protocol_only:

        print(
            "\nNo model was loaded."
        )

        print(
            "No new TallyQA prediction "
            "was produced."
        )

        return

    if resume:

        completed = (
            completed_samples()
        )

    else:

        if MATRIX_PATH.exists():

            raise RuntimeError(
                "Action matrix already exists. "
                "Use --resume after a technical "
                "interruption."
            )

        completed = set()

    print(
        "\nAlready completed:",
        len(
            completed
        ),
    )

    print(
        "Remaining:",
        4000
        -
        len(
            completed
        ),
    )

    archived_by_qid = (
        archived.set_index(
            "question_id"
        )
    )

    ordered = sorted(
        records,
        key=lambda r:
            int(
                r[
                    "manifest_index"
                ]
            ),
    )

    remaining = [
        r
        for r in ordered
        if int(
            r[
                "question_id"
            ]
        )
        not in completed
    ]

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

    if sorted(
        numeral_ids.keys()
    ) != list(
        range(
            16
        )
    ):

        raise RuntimeError(
            "Expanded numeral-token "
            "audit failed."
        )

    print(
        "Numeral-token audit: PASS"
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

    for sample in tqdm(
        remaining,
        desc=(
            "TallyQA action-oracle sweep"
        ),
    ):

        qid = int(
            sample[
                "question_id"
            ]
        )

        image_path = (
            IMAGE_ROOT
            /
            str(
                sample[
                    "image"
                ]
            )
        )

        image = (
            Image.open(
                image_path
            )
            .convert(
                "RGB"
            )
        )

        inputs = (
            prepare_tallyqa_inputs(
                processor,
                image,
                str(
                    sample[
                        "question"
                    ]
                ),
            )
        )

        inputs = (
            move_inputs(
                inputs,
                model,
            )
        )

        # ----------------------------------------------------
        # GT is deliberately NOT read while actions are run.
        # ----------------------------------------------------

        baseline_state = (
            score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        )

        baseline_prediction = int(
            baseline_state[
                "best_numeral"
            ]
        )

        archived_baseline_prediction = int(
            archived_by_qid.loc[
                qid,
                "baseline_prediction",
            ]
        )

        if (
            baseline_prediction
            !=
            archived_baseline_prediction
        ):

            raise RuntimeError(
                "Baseline reproducibility "
                f"failure for qid={qid}: "
                f"fresh={baseline_prediction}, "
                f"archived="
                f"{archived_baseline_prediction}"
            )

        states = {
            1.0:
                (
                    baseline_state,
                    0,
                )
        }

        for alpha in NONNOOP_ACTIONS:

            modifier = (
                HeadGainModifier(
                    model=model,
                    layer_idx=LAYER,
                    head_idx=HEAD,
                    alpha=alpha,
                )
            )

            modifier.register()

            try:

                state = (
                    score_state(
                        model,
                        inputs,
                        numeral_ids,
                        DUMMY_GT,
                    )
                )

            finally:

                modifier.remove()

            if (
                modifier.calls
                <= 0
            ):

                raise RuntimeError(
                    "Gain hook not called "
                    f"for qid={qid}, "
                    f"alpha={alpha}"
                )

            states[
                float(alpha)
            ] = (
                state,
                int(
                    modifier.calls
                ),
            )

        # ----------------------------------------------------
        # ONLY NOW read actual TallyQA answer.
        # ----------------------------------------------------

        gt = int(
            sample[
                "answer"
            ]
        )

        baseline_correct = bool(
            baseline_prediction
            ==
            gt
        )

        local_rows = []

        for alpha in ACTIONS:

            (
                state,
                hook_calls,
            ) = states[
                float(alpha)
            ]

            prediction = int(
                state[
                    "best_numeral"
                ]
            )

            action_correct = bool(
                prediction
                ==
                gt
            )

            local_rows.append({
                "manifest_index":
                    int(
                        sample[
                            "manifest_index"
                        ]
                    ),

                "question_id":
                    qid,

                "subset":
                    str(
                        sample[
                            "subset"
                        ]
                    ),

                "image":
                    str(
                        sample[
                            "image"
                        ]
                    ),

                "alpha":
                    float(
                        alpha
                    ),

                "ground_truth":
                    gt,

                "baseline_prediction":
                    baseline_prediction,

                "action_prediction":
                    prediction,

                "baseline_correct":
                    baseline_correct,

                "action_correct":
                    action_correct,

                "repair":
                    bool(
                        (
                            not baseline_correct
                        )
                        and
                        action_correct
                    ),

                "break_case":
                    bool(
                        baseline_correct
                        and
                        (
                            not action_correct
                        )
                    ),

                "prediction_shift":
                    int(
                        prediction
                        -
                        baseline_prediction
                    ),

                "baseline_expected_numeral":
                    float(
                        baseline_state[
                            "expected_numeral"
                        ]
                    ),

                "action_expected_numeral":
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
                        baseline_state[
                            "expected_numeral"
                        ]
                    ),

                "hook_calls":
                    int(
                        hook_calls
                    ),
            })

        pd.DataFrame(
            local_rows
        ).to_csv(
            MATRIX_PATH,
            mode="a",
            header=(
                not MATRIX_PATH.exists()
            ),
            index=False,
        )

    matrix = pd.read_csv(
        MATRIX_PATH
    )

    summarize(
        matrix,
        archived,
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
        resume=
            args.resume,

        protocol_only=
            args.protocol_only,
    )
