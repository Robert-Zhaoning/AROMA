#!/usr/bin/env python3

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

sys.path.insert(0, "scripts")

import natural_ugr_stage2_common_v1 as C
import run_tallyqa_natural_confirmation_v2_final as nat


EXPECTED_N = 631
EXPECTED_ROWS = EXPECTED_N * 43

DEV = Path(
    "manifests/natural_ugr_v2/"
    "final_stage2_population_v1.csv"
)

IMAGE_ROOT = Path(
    "data/natural_ugr_final_v1/images"
)

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

MECH_ROOT = Path(
    "outputs/natural_ugr_v2/"
    "final/stage2/"
    "mechanistic_collection_v1"
)

MECH = (
    MECH_ROOT
    /
    "mechanistic_summary_v1.csv"
)

MN_PATH = (
    MECH_ROOT
    /
    "mn_directions.npy"
)

SA_PATH = (
    MECH_ROOT
    /
    "sa_directions.npy"
)

OUT = Path(
    "outputs/natural_ugr_v2/"
    "final/stage2/"
    "candidate_outcomes_v1"
)

PARTIAL = (
    OUT
    /
    "partial_candidate_samples.jsonl"
)

FINAL = (
    OUT
    /
    "candidate_outcomes_v1.csv"
)

METADATA = (
    OUT
    /
    "metadata.json"
)


def append_jsonl(
    path,
    record,
):
    with Path(path).open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            +
            "\n"
        )

        f.flush()


def load_jsonl(
    path,
):
    p = Path(
        path
    )

    if not p.is_file():
        return []

    return [
        json.loads(
            line
        )
        for line in p.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def main():

    for p in [
        DEV,
        U4_PATH,
        MECH,
        MN_PATH,
        SA_PATH,
    ]:
        if not p.is_file():
            raise FileNotFoundError(
                p
            )

    if FINAL.exists() or METADATA.exists():
        raise RuntimeError(
            "Final candidate artifact already exists."
        )

    dev = pd.read_csv(
        DEV
    )

    mech = pd.read_csv(
        MECH
    )

    U4 = np.load(
        U4_PATH
    ).astype(
        np.float64
    )

    MN = np.load(
        MN_PATH
    ).astype(
        np.float64
    )

    SA = np.load(
        SA_PATH
    ).astype(
        np.float64
    )

    if len(dev) != EXPECTED_N:
        raise RuntimeError(
            "Development N mismatch."
        )

    if len(mech) != EXPECTED_N:
        raise RuntimeError(
            "Mechanistic N mismatch."
        )

    if U4.shape != (
        128,
        4,
    ):
        raise RuntimeError(
            f"Bad U4 shape: {U4.shape}"
        )

    if MN.shape != (
        EXPECTED_N,
        128,
    ):
        raise RuntimeError(
            f"Bad MN shape: {MN.shape}"
        )

    if SA.shape != (
        EXPECTED_N,
        128,
    ):
        raise RuntimeError(
            f"Bad SA shape: {SA.shape}"
        )

    if not np.allclose(
        np.linalg.norm(
            MN,
            axis=1,
        ),
        1.0,
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Frozen MN normalization failure."
        )

    if not np.allclose(
        np.linalg.norm(
            SA,
            axis=1,
        ),
        1.0,
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Frozen SA normalization failure."
        )

    raw_ids = np.array(
        [
            str(
                int(x)
            )
            for x in dev[
                "question_id"
            ]
        ],
        dtype=str,
    )

    uids = (
        dev[
            "natural_ugr_uid"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        mech[
            "raw_sample_id"
        ]
        .astype(str)
        .to_numpy(),
        raw_ids,
    ):
        raise RuntimeError(
            "Dev/mechanistic sample ordering mismatch."
        )

    if not np.array_equal(
        mech[
            "cohort_uid"
        ]
        .astype(str)
        .to_numpy(),
        uids,
    ):
        raise RuntimeError(
            "Dev/mechanistic UID ordering mismatch."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_jsonl(
        PARTIAL
    )

    if len(records) > EXPECTED_N:
        raise RuntimeError(
            "Partial candidate archive too long."
        )

    for idx, rec in enumerate(
        records
    ):

        if str(
            rec[
                "raw_sample_id"
            ]
        ) != raw_ids[
            idx
        ]:
            raise RuntimeError(
                "Partial candidate sample ordering mismatch."
            )

        candidates = rec[
            "candidates"
        ]

        if len(
            candidates
        ) != 43:
            raise RuntimeError(
                "Partial candidate count != 43."
            )

    expected_methods = (
        ["IDENTITY"]
        +
        ["MN"] * 14
        +
        ["SA"] * 14
        +
        ["WHOLE"] * 14
    )

    expected_gains = (
        [0.0]
        +
        C.GAINS
        +
        C.GAINS
        +
        C.GAINS
    )

    print(
        "Loading processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            C.MODEL_ID,
            revision=
                C.MODEL_REVISION,
        )
    )

    numeral_ids = (
        C.numeral_token_ids(
            processor
        )
    )

    print(
        "Loading model..."
    )

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            C.MODEL_ID,
            revision=
                C.MODEL_REVISION,
            torch_dtype=
                torch.bfloat16,
            device_map="auto",
            attn_implementation=
                "eager",
        )
    )

    model.eval()

    start_idx = len(
        records
    )

    for idx in range(
        start_idx,
        EXPECTED_N,
    ):

        row = dev.iloc[
            idx
        ]

        sid = raw_ids[
            idx
        ]

        uid = uids[
            idx
        ]

        alpha_prop = float(
            row[
                "selected_alpha"
            ]
        )

        selected_score = float(
            row[
                "selected_score"
            ]
        )

        archived_baseline = int(
            row[
                "baseline_prediction"
            ]
        )

        image = (
            Image.open(
                IMAGE_ROOT
                /
                str(
                    row[
                        "image"
                    ]
                )
            )
            .convert(
                "RGB"
            )
        )

        inputs = (
            nat.prepare_tallyqa_inputs(
                processor,
                image,
                str(
                    row[
                        "question"
                    ]
                ),
            )
        )

        inputs = C.move_inputs(
            inputs,
            model,
        )

        baseline_state = (
            C.score_no_intervention(
                model,
                inputs,
                numeral_ids,
            )
        )

        baseline_pred = int(
            baseline_state[
                "best_numeral"
            ]
        )

        if baseline_pred != archived_baseline:
            # ----------------------------------------------------
            # Frozen B200 numeric-drift amendment.
            #
            # Full 631-sample replay found exactly one baseline
            # prediction drift. The frozen Stage-1 population,
            # alpha proposal, baseline reference, and routers are
            # NOT changed. Only this pre-audited qid may bypass
            # the exact live-baseline reproduction assertion.
            # ----------------------------------------------------
            if str(sid) != "9236394016419":
                raise RuntimeError(
                    f"{sid}: unexpected baseline reproduction drift "
                    "outside the frozen B200 replay audit."
                )
        
            print(
                f"{sid}: accepted pre-audited B200 baseline drift; "
                "continuing with frozen Stage-1 baseline/action reference.",
                flush=True,
            )

        # Ground truth is accessed only now,
        # after GT-free input/baseline execution.
        # It is used only to label candidate outcomes.
        gt = int(
            row[
                "ground_truth"
            ]
        )

        candidate_rows = []

        candidate_rows.append(
            C.make_outcome_row(
                candidate_index=0,
                sid=sid,
                uid=uid,
                gt=gt,
                baseline_pred=
                    baseline_pred,
                selected_alpha=
                    alpha_prop,
                selected_score=
                    selected_score,
                method="IDENTITY",
                gain=0.0,
                prediction=
                    baseline_pred,
                matched_effective_alpha=
                    1.0,
                shift_norm=0.0,
                shift_ratio=0.0,
                hook_calls=0,
            )
        )

        candidate_index = 1

        for method in [
            "MN",
            "SA",
        ]:

            for gain in C.GAINS:

                state, modifier = (
                    C.score_lowrank(
                        model=model,
                        inputs=inputs,
                        numeral_ids=
                            numeral_ids,
                        alpha=
                            alpha_prop,
                        gain=
                            gain,
                        method=
                            method,
                        U4=
                            U4,
                        sa_direction=
                            SA[
                                idx
                            ],
                    )
                )

                candidate_rows.append(
                    C.make_outcome_row(
                        candidate_index=
                            candidate_index,
                        sid=sid,
                        uid=uid,
                        gt=gt,
                        baseline_pred=
                            baseline_pred,
                        selected_alpha=
                            alpha_prop,
                        selected_score=
                            selected_score,
                        method=
                            method,
                        gain=
                            gain,
                        prediction=
                            int(
                                state[
                                    "best_numeral"
                                ]
                            ),
                        matched_effective_alpha=
                            (
                                1.0
                                +
                                gain
                                *
                                (
                                    alpha_prop
                                    -
                                    1.0
                                )
                            ),
                        shift_norm=
                            modifier
                            .last_shift_norm,
                        shift_ratio=
                            modifier
                            .last_shift_ratio,
                        hook_calls=
                            modifier.calls,
                    )
                )

                candidate_index += 1

        frozen_h_norm = float(
            mech.iloc[
                idx
            ][
                "h_norm"
            ]
        )

        for gain in C.GAINS:

            effective_alpha = (
                1.0
                +
                gain
                *
                (
                    alpha_prop
                    -
                    1.0
                )
            )

            state, modifier = (
                C.score_whole(
                    model=model,
                    inputs=inputs,
                    numeral_ids=
                        numeral_ids,
                    effective_alpha=
                        effective_alpha,
                )
            )

            whole_shift_norm = (
                gain
                *
                abs(
                    alpha_prop
                    -
                    1.0
                )
                *
                frozen_h_norm
            )

            candidate_rows.append(
                C.make_outcome_row(
                    candidate_index=
                        candidate_index,
                    sid=sid,
                    uid=uid,
                    gt=gt,
                    baseline_pred=
                        baseline_pred,
                    selected_alpha=
                        alpha_prop,
                    selected_score=
                        selected_score,
                    method="WHOLE",
                    gain=gain,
                    prediction=
                        int(
                            state[
                                "best_numeral"
                            ]
                        ),
                    matched_effective_alpha=
                        effective_alpha,
                    shift_norm=
                        whole_shift_norm,
                    shift_ratio=
                        gain,
                    hook_calls=
                        modifier.calls,
                )
            )

            candidate_index += 1

        if len(
            candidate_rows
        ) != 43:
            raise RuntimeError(
                f"{sid}: candidate count != 43."
            )

        observed_methods = [
            x[
                "candidate_method"
            ]
            for x in candidate_rows
        ]

        observed_gains = [
            float(
                x[
                    "candidate_gain"
                ]
            )
            for x in candidate_rows
        ]

        if (
            observed_methods
            !=
            expected_methods
        ):
            raise RuntimeError(
                f"{sid}: method ordering mismatch."
            )

        if not np.allclose(
            observed_gains,
            expected_gains,
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{sid}: gain ordering mismatch."
            )

        record = {
            "raw_sample_id":
                sid,

            "cohort_uid":
                uid,

            "ground_truth":
                gt,

            "baseline_prediction":
                baseline_pred,

            "selected_alpha":
                alpha_prop,

            "selected_score":
                selected_score,

            "candidates":
                candidate_rows,
        }

        # Write only once the full sample is complete.
        append_jsonl(
            PARTIAL,
            record,
        )

        records.append(
            record
        )

        print(
            f"[{idx + 1:03d}/{EXPECTED_N}] "
            f"{sid} complete",
            flush=True,
        )

        del inputs
        del image
        del baseline_state
        del candidate_rows

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(
        records
    ) != EXPECTED_N:
        raise RuntimeError(
            "Final candidate sample count mismatch."
        )

    rows = []

    for rec in records:
        rows.extend(
            rec[
                "candidates"
            ]
        )

    df = pd.DataFrame(
        rows
    )

    if len(
        df
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} candidate rows; "
            f"got {len(df)}"
        )

    if df[
        [
            "raw_sample_id",
            "candidate_index",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate sample/candidate index."
        )

    per_sample = (
        df
        .groupby(
            "raw_sample_id"
        )
        .size()
    )

    if not (
        per_sample
        ==
        43
    ).all():
        raise RuntimeError(
            "Not every sample has 43 candidates."
        )

    method_counts = (
        df[
            "candidate_method"
        ]
        .value_counts()
        .to_dict()
    )

    expected_counts = {
        "IDENTITY":
            EXPECTED_N,

        "MN":
            EXPECTED_N
            *
            14,

        "SA":
            EXPECTED_N
            *
            14,

        "WHOLE":
            EXPECTED_N
            *
            14,
    }

    if (
        method_counts
        !=
        expected_counts
    ):
        raise RuntimeError(
            f"Candidate method count mismatch: "
            f"{method_counts}"
        )

    numeric_cols = [
        "selected_alpha",
        "selected_score",
        "candidate_gain",
        "matched_effective_alpha",
        "shift_norm",
        "shift_ratio",
    ]

    if not np.isfinite(
        df[
            numeric_cols
        ].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "Non-finite numeric candidate value."
        )

    df.to_csv(
        FINAL,
        index=False,
    )

    metadata = {
        "experiment":
            "natural_ugr_v2_final_stage2_candidate_sweep_v1",

        "population_n":
            EXPECTED_N,

        "candidate_rows":
            len(
                df
            ),

        "candidates_per_sample":
            43,

        "gains":
            C.GAINS,

        "method_counts":
            {
                str(k):
                    int(v)
                for k, v
                in method_counts.items()
            },

        "ground_truth_role":
            (
                "used only for candidate_correct / "
                "repair / break outcome labels"
            ),

        "candidate_csv_sha256":
            C.sha256_file(
                FINAL
            ),
    }

    METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        +
        "\n"
    )

    print()
    print(
        "=" * 78
    )

    print(
        "NATURAL UGR DEVELOPMENT CANDIDATE SWEEP COMPLETE"
    )

    print(
        "Samples:",
        EXPECTED_N,
    )

    print(
        "Candidate rows:",
        len(
            df
        ),
    )

    print(
        "Method counts:",
        method_counts,
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
