import math

import numpy as np
import torch

from run_l18h13_gain_all300 import (
    add_row as _original_add_row,
)


NUMERAL_MIN = 0
NUMERAL_MAX = 15

NUMERALS = list(
    range(
        NUMERAL_MIN,
        NUMERAL_MAX + 1,
    )
)


def numeral_token_ids(
    processor,
):
    """
    Single-token numeral IDs for the corrected
    expanded candidate space {0, ..., 15}.
    """

    result = {}

    for n in NUMERALS:

        ids = (
            processor.tokenizer.encode(
                str(n),
                add_special_tokens=False,
            )
        )

        if len(ids) != 1:

            raise RuntimeError(
                f"Numeral {n} is not "
                f"single-token: {ids}"
            )

        result[n] = int(
            ids[0]
        )

    return result


def score_state(
    model,
    inputs,
    numeral_ids,
    gt,
):
    """
    Corrected score function over numeral candidates 0..15.

    Important:
    - GT remains 1..10.
    - Competing numerals include 0 and 11..15.
    - Expected numeral, entropy, margin and top-k are all
      defined over the same expanded conditional distribution.
    """

    numerals = sorted(
        int(n)
        for n in numeral_ids.keys()
    )

    if numerals != NUMERALS:

        raise RuntimeError(
            "Unexpected numeral candidate set: "
            f"{numerals}"
        )

    if int(gt) not in numeral_ids:

        raise RuntimeError(
            f"GT {gt} is absent from "
            "expanded numeral set."
        )

    with torch.inference_mode():

        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

    logits = (
        outputs.logits[
            0,
            -1,
            :
        ]
        .float()
    )

    vocab_log_probs = (
        torch.log_softmax(
            logits,
            dim=-1,
        )
    )

    # --------------------------------------------------------
    # Numeral logits in explicit NUMERALS order.
    # Index i corresponds to NUMERALS[i], not i+1.
    # --------------------------------------------------------

    numeral_logits = torch.tensor(
        [
            float(
                logits[
                    numeral_ids[n]
                ].item()
            )
            for n in numerals
        ],
        dtype=torch.float64,
    )

    conditional_probs = (
        torch.softmax(
            numeral_logits,
            dim=0,
        )
        .cpu()
        .numpy()
    )

    index_of = {
        n: i
        for i, n
        in enumerate(
            numerals
        )
    }

    # --------------------------------------------------------
    # Best numeral
    # --------------------------------------------------------

    best_idx = int(
        np.argmax(
            conditional_probs
        )
    )

    best_numeral = int(
        numerals[
            best_idx
        ]
    )

    # --------------------------------------------------------
    # GT margin against ALL alternative numerals in 0..15.
    # --------------------------------------------------------

    gt_idx = index_of[
        int(gt)
    ]

    wrong_logits = [
        float(
            numeral_logits[i]
        )
        for i, n
        in enumerate(
            numerals
        )
        if n != int(gt)
    ]

    margin = (
        float(
            numeral_logits[
                gt_idx
            ]
        )
        -
        max(
            wrong_logits
        )
    )

    # --------------------------------------------------------
    # Full-vocabulary GT log probability.
    # This definition is unchanged from the frozen runner.
    # --------------------------------------------------------

    gt_logp = float(
        vocab_log_probs[
            numeral_ids[
                int(gt)
            ]
        ].item()
    )

    # --------------------------------------------------------
    # Conditional numeral entropy
    # --------------------------------------------------------

    entropy = float(
        -np.sum(
            conditional_probs
            *
            np.log(
                conditional_probs
                + 1e-12
            )
        )
    )

    entropy_norm = (
        entropy
        /
        math.log(
            float(
                len(
                    numerals
                )
            )
        )
    )

    # --------------------------------------------------------
    # Expected numeral
    # --------------------------------------------------------

    expected_numeral = float(
        sum(
            n
            *
            conditional_probs[i]
            for i, n
            in enumerate(
                numerals
            )
        )
    )

    # --------------------------------------------------------
    # Top-1 / top-2 within expanded numeral space
    # --------------------------------------------------------

    order = np.argsort(
        conditional_probs
    )[::-1]

    top1_idx = int(
        order[0]
    )

    top2_idx = int(
        order[1]
    )

    top1 = int(
        numerals[
            top1_idx
        ]
    )

    top2 = int(
        numerals[
            top2_idx
        ]
    )

    top1_prob = float(
        conditional_probs[
            top1_idx
        ]
    )

    top2_prob = float(
        conditional_probs[
            top2_idx
        ]
    )

    conditional_margin = (
        top1_prob
        -
        top2_prob
    )

    # --------------------------------------------------------
    # Probability mass assigned by the full vocabulary
    # distribution to numeral tokens 0..15.
    # --------------------------------------------------------

    vocab_probs = (
        torch.softmax(
            logits,
            dim=-1,
        )
    )

    vocab_numeral_mass = float(
        sum(
            vocab_probs[
                numeral_ids[n]
            ].item()
            for n in numerals
        )
    )

    return {
        "gt_logp":
            gt_logp,

        "margin":
            margin,

        "best_numeral":
            best_numeral,

        "conditional_probs":
            conditional_probs,

        "entropy":
            entropy,

        "entropy_norm":
            entropy_norm,

        "expected_numeral":
            expected_numeral,

        "top1":
            top1,

        "top2":
            top2,

        "top1_prob":
            top1_prob,

        "top2_prob":
            top2_prob,

        "conditional_margin":
            conditional_margin,

        "vocab_numeral_mass":
            vocab_numeral_mass,

        # Extra bookkeeping for audits.
        "numerals":
            numerals,
    }


def add_row(
    rows,
    sample,
    baseline,
    state,
    alpha,
    calls,
):
    """
    Reuse the already-tested original add_row implementation
    for all generic metrics, then replace only the incorrectly
    hard-coded 1..10 numprob columns with correct 0..15 columns.
    """

    temporary_rows = []

    _original_add_row(
        rows=temporary_rows,
        sample=sample,
        baseline=baseline,
        state=state,
        alpha=alpha,
        calls=calls,
    )

    if len(temporary_rows) != 1:

        raise RuntimeError(
            "Original add_row produced "
            f"{len(temporary_rows)} rows."
        )

    row = temporary_rows[0]

    # --------------------------------------------------------
    # Remove the original mislabeled 1..10 probability fields.
    #
    # With a 16-dimensional conditional_probs vector,
    # the original implementation would incorrectly map:
    #   p[0] -> numprob_1
    # even though p[0] now means numeral 0.
    # --------------------------------------------------------

    prefixes = (
        "baseline_numprob_",
        "modulated_numprob_",
        "delta_numprob_",
    )

    for key in list(
        row.keys()
    ):

        if key.startswith(
            prefixes
        ):
            del row[key]

    p0 = np.asarray(
        baseline[
            "conditional_probs"
        ],
        dtype=float,
    )

    p1 = np.asarray(
        state[
            "conditional_probs"
        ],
        dtype=float,
    )

    if len(p0) != len(
        NUMERALS
    ):

        raise RuntimeError(
            f"Baseline probability vector "
            f"has length {len(p0)}, "
            f"expected {len(NUMERALS)}."
        )

    if len(p1) != len(
        NUMERALS
    ):

        raise RuntimeError(
            f"Modulated probability vector "
            f"has length {len(p1)}, "
            f"expected {len(NUMERALS)}."
        )

    delta = (
        p1 - p0
    )

    for i, n in enumerate(
        NUMERALS
    ):

        row[
            f"baseline_numprob_{n}"
        ] = float(
            p0[i]
        )

        row[
            f"modulated_numprob_{n}"
        ] = float(
            p1[i]
        )

        row[
            f"delta_numprob_{n}"
        ] = float(
            delta[i]
        )

    rows.append(
        row
    )
