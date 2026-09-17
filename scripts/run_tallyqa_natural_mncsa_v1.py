#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import binomtest
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration


sys.path.insert(0, "scripts")


import run_tallyqa_natural_confirmation_v2_final as nat

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

from run_proc_count_causal_v3_final import (
    state_to_features,
    controller_decision,
)


# ============================================================
# Frozen experiment constants
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS

START = HEAD * HEAD_DIM
END = START + HEAD_DIM

RANK = 4

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/U4_primary.npy"
)

PROTOCOL_PATH = Path(
    "docs/AROMA2_NATURAL_MNCSA_PROTOCOL_v1.md"
)

RUNNER_CONFIG_PATH = Path(
    "configs/tallyqa_natural_mncsa_v1_final_runner.json"
)

OUT_DIR = Path(
    "outputs/aroma2/tallyqa_natural_mncsa_v1"
)

PARTIAL_PATH = OUT_DIR / "partial_results.csv"

RESULT_PATH = (
    OUT_DIR
    / "tallyqa_natural_mncsa_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "tallyqa_natural_mncsa_summary.json"
)

SUBSET_PATH = (
    OUT_DIR
    / "tallyqa_natural_mncsa_by_subset.csv"
)

COUNT_PATH = (
    OUT_DIR
    / "tallyqa_natural_mncsa_by_count.csv"
)

ACTION_PATH = (
    OUT_DIR
    / "tallyqa_natural_mncsa_by_action.csv"
)

RUN_STARTED_PATH = (
    OUT_DIR
    / "run_started.json"
)

RUN_META_PATH = (
    OUT_DIR
    / "run_metadata.json"
)

SMOKE_PATH = (
    OUT_DIR
    / "smoke_result.json"
)


EXPECTED_MANIFEST_SHA256 = (
    "cf42a6c4b07f21d72c88d302815fd141"
    "9dd912431b203ff9f8cc53f3fc05ae0d"
)

EXPECTED_INVENTORY_SHA256 = (
    "475e28e0ebb7934d49d2062734b164ff"
    "04f4f04461d61de9ee4315f96182f0b2"
)

EXPECTED_CONTROLLER_SHA256 = (
    "d9d8ba3a24814bf71b3f4039d1effa8"
    "64b57754a00539f1260a1d077c03449f5"
)

EXPECTED_U4_SHA256 = (
    "af50da2cf49268cc55dc33d44dad50c0"
    "cd5a4fa29cda6cad3c01cc0b4ae63f14"
)


EXPECTED_ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]

EXPECTED_THRESHOLD = 0.1
EXPECTED_RIDGE_ALPHA = 0.01
EXPECTED_FEATURE_COUNT = 39

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260917

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


def git_head():

    return (
        subprocess
        .check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )


def load_json(path):

    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def save_json(path, obj):

    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )


def as_bool_array(series):

    if pd.api.types.is_bool_dtype(
        series.dtype
    ):

        return (
            series
            .to_numpy(
                dtype=bool
            )
        )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    normalized = (
        series
        .astype(str)
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():

        raise RuntimeError(
            "Could not parse boolean "
            f"column {series.name}"
        )

    return normalized.to_numpy(
        dtype=bool
    )


def paired_bootstrap_ci(
    reference_correct,
    candidate_correct,
    reps=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):

    diff = (
        np.asarray(
            candidate_correct,
            dtype=np.float64,
        )
        -
        np.asarray(
            reference_correct,
            dtype=np.float64,
        )
    )

    n = int(
        diff.shape[0]
    )

    rng = (
        np.random
        .default_rng(
            seed
        )
    )

    estimates = np.empty(
        reps,
        dtype=np.float64,
    )

    batch = 250
    offset = 0

    while offset < reps:

        b = min(
            batch,
            reps - offset,
        )

        idx = rng.integers(
            0,
            n,
            size=(
                b,
                n,
            ),
        )

        estimates[
            offset:
            offset + b
        ] = (
            diff[
                idx
            ]
            .mean(
                axis=1
            )
        )

        offset += b

    low, high = np.quantile(
        estimates,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(low),
        float(high),
    )


def pairwise_stats(
    df,
    reference_col,
    candidate_col,
    bootstrap=True,
):

    reference = as_bool_array(
        df[
            reference_col
        ]
    )

    candidate = as_bool_array(
        df[
            candidate_col
        ]
    )

    gain = int(
        (
            ~reference
            &
            candidate
        )
        .sum()
    )

    loss = int(
        (
            reference
            &
            ~candidate
        )
        .sum()
    )

    n = int(
        len(df)
    )

    ref_acc = float(
        reference.mean()
    )

    cand_acc = float(
        candidate.mean()
    )

    delta = float(
        cand_acc
        -
        ref_acc
    )

    if bootstrap:

        ci_low, ci_high = (
            paired_bootstrap_ci(
                reference,
                candidate,
            )
        )

    else:

        ci_low = None
        ci_high = None

    if gain + loss == 0:

        p_value = 1.0

    else:

        p_value = float(
            binomtest(
                k=min(
                    gain,
                    loss,
                ),
                n=(
                    gain
                    +
                    loss
                ),
                p=0.5,
                alternative="two-sided",
            )
            .pvalue
        )

    result = {
        "n":
            n,

        "reference_accuracy":
            ref_acc,

        "candidate_accuracy":
            cand_acc,

        "difference":
            delta,

        "difference_pp":
            100.0
            *
            delta,

        "reference_wrong_candidate_correct":
            gain,

        "reference_correct_candidate_wrong":
            loss,

        "mcnemar_exact_p":
            p_value,
    }

    if bootstrap:

        result[
            "ci95_low"
        ] = ci_low

        result[
            "ci95_high"
        ] = ci_high

        result[
            "ci95_low_pp"
        ] = (
            100.0
            *
            ci_low
        )

        result[
            "ci95_high_pp"
        ] = (
            100.0
            *
            ci_high
        )

    return result


# ============================================================
# Frozen MN-CSA modifier
# ============================================================

class MagnitudeNormalizedCSAModifier:

    def __init__(
        self,
        model,
        alpha,
        U,
    ):

        layer = (
            model
            .model
            .language_model
            .layers[
                LAYER
            ]
        )

        self.o_proj = (
            layer
            .cross_attn
            .o_proj
        )

        hidden_size = int(
            self.o_proj
            .in_features
        )

        if hidden_size != HIDDEN_SIZE:

            raise RuntimeError(
                "Unexpected hidden "
                f"size: {hidden_size}"
            )

        U = np.asarray(
            U,
            dtype=np.float32,
        )

        if U.shape != (
            HEAD_DIM,
            RANK,
        ):

            raise RuntimeError(
                "Expected U "
                f"({HEAD_DIM},{RANK}), "
                f"got {U.shape}"
            )

        self.U = torch.tensor(
            U,
            dtype=torch.float32,
            device=
                self.o_proj
                .weight
                .device,
        )

        self.alpha = float(
            alpha
        )

        self.calls = 0

        self.handle = None

        self.last_rho = None

        self.last_strength_error = None

        self.last_subspace_residual = None


    def hook(
        self,
        module,
        inputs,
    ):

        if not inputs:

            raise RuntimeError(
                "Empty hook inputs."
            )

        x = inputs[
            0
        ]

        H = x[
            ...,
            START:END
        ]

        h = (
            H
            .float()
            .mean(
                dim=1
            )
        )

        U = self.U

        projection = (
            (h @ U)
            @ U.T
        )

        h_norm = (
            torch.linalg
            .vector_norm(
                h,
                dim=-1,
                keepdim=True,
            )
        )

        projected_norm = (
            torch.linalg
            .vector_norm(
                projection,
                dim=-1,
                keepdim=True,
            )
        )

        if float(
            projected_norm
            .min()
            .item()
        ) <= 1e-12:

            raise RuntimeError(
                "MN-CSA projected "
                "norm too small."
            )

        direction = (
            projection
            /
            projected_norm
        )

        delta = (
            (
                self.alpha
                -
                1.0
            )
            *
            h_norm
            *
            direction
        )

        actual_shift = (
            torch.linalg
            .vector_norm(
                delta,
                dim=-1,
                keepdim=True,
            )
        )

        target_shift = (
            abs(
                self.alpha
                -
                1.0
            )
            *
            h_norm
        )

        if math.isclose(
            self.alpha,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):

            relative_error = (
                torch.zeros_like(
                    actual_shift
                )
            )

            subspace_residual = (
                torch.zeros_like(
                    actual_shift
                )
            )

        else:

            relative_error = (
                torch.abs(
                    actual_shift
                    -
                    target_shift
                )
                /
                torch.clamp(
                    target_shift,
                    min=1e-12,
                )
            )

            delta_projected = (
                (delta @ U)
                @ U.T
            )

            residual = (
                delta
                -
                delta_projected
            )

            subspace_residual = (
                torch.linalg
                .vector_norm(
                    residual,
                    dim=-1,
                    keepdim=True,
                )
                /
                torch.clamp(
                    actual_shift,
                    min=1e-12,
                )
            )

        self.last_rho = float(
            (
                projected_norm
                /
                h_norm
            )[
                0,
                0
            ]
            .item()
        )

        self.last_strength_error = float(
            relative_error[
                0,
                0
            ]
            .item()
        )

        self.last_subspace_residual = float(
            subspace_residual[
                0,
                0
            ]
            .item()
        )

        H_new = (
            H
            +
            delta
            .to(
                H.dtype
            )
            .unsqueeze(
                1
            )
        )

        x_new = x.clone()

        x_new[
            ...,
            START:END
        ] = H_new

        self.calls += 1

        if len(
            inputs
        ) == 1:

            return (
                x_new,
            )

        return (
            x_new,
            *inputs[
                1:
            ],
        )


    def register(
        self,
    ):

        self.calls = 0

        self.handle = (
            self.o_proj
            .register_forward_pre_hook(
                self.hook
            )
        )


    def remove(
        self,
    ):

        if self.handle is not None:

            self.handle.remove()

            self.handle = None


# ============================================================
# Frozen artifact / protocol audit
# ============================================================

def protocol_audit():

    print(
        "=" * 100
    )

    print(
        "AROMA NATURAL-DOMAIN "
        "MN-CSA VALIDATION v1"
    )

    print(
        "=" * 100
    )

    required = [
        U4_PATH,
        PROTOCOL_PATH,
        RUNNER_CONFIG_PATH,
        nat.MANIFEST_PATH,
        nat.IMAGE_INVENTORY_PATH,
        nat.CONTROLLER_BUNDLE_PATH,
    ]

    missing = [
        str(path)
        for path in required
        if not Path(
            path
        ).exists()
    ]

    if missing:

        raise RuntimeError(
            "Missing required "
            "artifact(s):\n"
            +
            "\n".join(
                missing
            )
        )

    config = load_json(
        RUNNER_CONFIG_PATH
    )

    current_runner_hash = (
        sha256_file(
            Path(
                __file__
            )
        )
    )

    current_protocol_hash = (
        sha256_file(
            PROTOCOL_PATH
        )
    )

    if (
        current_runner_hash
        !=
        config[
            "runner_source_sha256"
        ]
    ):

        raise RuntimeError(
            "Natural MN-CSA runner "
            "hash mismatch."
        )

    if (
        current_protocol_hash
        !=
        config[
            "protocol_sha256"
        ]
    ):

        raise RuntimeError(
            "Natural MN-CSA protocol "
            "hash mismatch."
        )

    (
        records,
        bundle,
        bundle_hash,
        _,
    ) = nat.protocol_audit()

    if (
        sha256_file(
            nat.MANIFEST_PATH
        )
        !=
        EXPECTED_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "TallyQA manifest "
            "hash mismatch."
        )

    if (
        sha256_file(
            nat.IMAGE_INVENTORY_PATH
        )
        !=
        EXPECTED_INVENTORY_SHA256
    ):

        raise RuntimeError(
            "TallyQA image inventory "
            "hash mismatch."
        )

    if (
        bundle_hash
        !=
        EXPECTED_CONTROLLER_SHA256
    ):

        raise RuntimeError(
            "Natural controller "
            "hash mismatch."
        )

    if (
        sha256_file(
            U4_PATH
        )
        !=
        EXPECTED_U4_SHA256
    ):

        raise RuntimeError(
            "U4 hash mismatch."
        )

    if (
        bundle[
            "model_id"
        ]
        !=
        MODEL_ID
    ):

        raise RuntimeError(
            "Controller model mismatch."
        )

    if int(
        bundle[
            "head_layer"
        ]
    ) != LAYER:

        raise RuntimeError(
            "Controller layer mismatch."
        )

    if int(
        bundle[
            "head_index"
        ]
    ) != HEAD:

        raise RuntimeError(
            "Controller head mismatch."
        )

    actions = [
        float(x)
        for x in
        bundle[
            "actions"
        ]
    ]

    if (
        actions
        !=
        EXPECTED_ACTIONS
    ):

        raise RuntimeError(
            "Action-set mismatch."
        )

    if (
        len(
            bundle[
                "feature_names"
            ]
        )
        !=
        EXPECTED_FEATURE_COUNT
    ):

        raise RuntimeError(
            "Feature-count mismatch."
        )

    if not math.isclose(
        float(
            bundle[
                "ridge_alpha"
            ]
        ),
        EXPECTED_RIDGE_ALPHA,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        raise RuntimeError(
            "Ridge-alpha mismatch."
        )

    if not math.isclose(
        float(
            bundle[
                "threshold"
            ]
        ),
        EXPECTED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        raise RuntimeError(
            "Threshold mismatch."
        )

    U4 = np.load(
        U4_PATH
    )

    if U4.shape != (
        HEAD_DIM,
        RANK,
    ):

        raise RuntimeError(
            f"Bad U4 shape: "
            f"{U4.shape}"
        )

    orth_err = float(
        np.max(
            np.abs(
                U4.T
                @
                U4
                -
                np.eye(
                    RANK,
                    dtype=np.float32,
                )
            )
        )
    )

    if orth_err > 1e-5:

        raise RuntimeError(
            "U4 orthogonality "
            "error too large: "
            f"{orth_err}"
        )

    if len(
        records
    ) != 4000:

        raise RuntimeError(
            "Expected 4000 "
            f"records, got "
            f"{len(records)}"
        )

    print(
        "PASS: protocol and "
        "artifact audit"
    )

    print(
        "TallyQA manifest:",
        EXPECTED_MANIFEST_SHA256,
    )

    print(
        "Controller:",
        EXPECTED_CONTROLLER_SHA256,
    )

    print(
        "U4:",
        EXPECTED_U4_SHA256,
    )

    print(
        "U4 orthogonality "
        "max error:",
        orth_err,
    )

    return (
        records,
        bundle,
        U4,
        config,
    )


# ============================================================
# Model load
# ============================================================

def load_model():

    print(
        "Loading processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            revision=
                MODEL_REVISION,
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
            revision=
                MODEL_REVISION,
            torch_dtype=
                torch.bfloat16,
            device_map="auto",
            attn_implementation=
                "eager",
        )
    )

    model.eval()

    return (
        processor,
        numeral_ids,
        model,
    )


def make_inputs(
    processor,
    model,
    sample,
):

    image_path = (
        nat.IMAGE_ROOT
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
        nat.prepare_tallyqa_inputs(
            processor,
            image,
            str(
                sample[
                    "question"
                ]
            ),
        )
    )

    inputs = move_inputs(
        inputs,
        model,
    )

    return (
        image,
        inputs,
    )


# ============================================================
# Identity and smoke checks
# ============================================================

def identity_precheck(
    records,
    processor,
    model,
    U4,
):

    sample = sorted(
        records,
        key=lambda r:
            int(
                r[
                    "manifest_index"
                ]
            ),
    )[
        0
    ]

    image, inputs = (
        make_inputs(
            processor,
            model,
            sample,
        )
    )

    with torch.inference_mode():

        baseline = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

    whole = (
        HeadGainModifier(
            model=model,
            layer_idx=LAYER,
            head_idx=HEAD,
            alpha=1.0,
        )
    )

    whole.register()

    try:

        with torch.inference_mode():

            whole_out = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

    finally:

        whole.remove()

    mn = (
        MagnitudeNormalizedCSAModifier(
            model=model,
            alpha=1.0,
            U=U4,
        )
    )

    mn.register()

    try:

        with torch.inference_mode():

            mn_out = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

    finally:

        mn.remove()

    whole_diff = float(
        (
            baseline
            .logits
            .float()
            -
            whole_out
            .logits
            .float()
        )
        .abs()
        .max()
        .item()
    )

    mn_diff = float(
        (
            baseline
            .logits
            .float()
            -
            mn_out
            .logits
            .float()
        )
        .abs()
        .max()
        .item()
    )

    if whole_diff != 0.0:

        raise RuntimeError(
            "Whole alpha=1 is "
            "not exact identity: "
            f"{whole_diff}"
        )

    if mn_diff != 0.0:

        raise RuntimeError(
            "MN alpha=1 is "
            "not exact identity: "
            f"{mn_diff}"
        )

    result = {
        "whole_alpha1_max_logit_diff":
            whole_diff,

        "mn_alpha1_max_logit_diff":
            mn_diff,
    }

    del baseline
    del whole_out
    del mn_out
    del inputs
    del image

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return result


def smoke_precheck(
    records,
    processor,
    model,
    U4,
):

    identity = (
        identity_precheck(
            records,
            processor,
            model,
            U4,
        )
    )

    sample = sorted(
        records,
        key=lambda r:
            int(
                r[
                    "manifest_index"
                ]
            ),
    )[
        0
    ]

    image, inputs = (
        make_inputs(
            processor,
            model,
            sample,
        )
    )

    # Implementation-only alpha.
    # No GT is read.
    # No prediction is recorded.
    mn = (
        MagnitudeNormalizedCSAModifier(
            model=model,
            alpha=2.0,
            U=U4,
        )
    )

    mn.register()

    try:

        with torch.inference_mode():

            _ = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

    finally:

        mn.remove()

    if mn.calls <= 0:

        raise RuntimeError(
            "MN-CSA smoke hook "
            "was not called."
        )

    if (
        mn.last_strength_error
        >
        1e-5
    ):

        raise RuntimeError(
            "MN-CSA magnitude "
            "mismatch: "
            f"{mn.last_strength_error}"
        )

    if (
        mn.last_subspace_residual
        >
        1e-5
    ):

        raise RuntimeError(
            "MN-CSA delta left "
            "the frozen U4 "
            "subspace: "
            f"{mn.last_subspace_residual}"
        )

    result = {
        **identity,

        "fixed_alpha_for_smoke":
            2.0,

        "hook_calls":
            int(
                mn.calls
            ),

        "rho":
            float(
                mn.last_rho
            ),

        "strength_relative_error":
            float(
                mn.last_strength_error
            ),

        "subspace_relative_residual":
            float(
                mn.last_subspace_residual
            ),

        "ground_truth_read":
            False,

        "prediction_recorded":
            False,

        "status":
            "PASS",
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        SMOKE_PATH,
        result,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    del inputs
    del image

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return result


# ============================================================
# Incremental persistence
# ============================================================

def append_row(
    row,
):

    pd.DataFrame(
        [
            row
        ]
    ).to_csv(
        PARTIAL_PATH,
        mode="a",
        header=
            not
            PARTIAL_PATH
            .exists(),
        index=False,
    )


def completed_qids():

    if not PARTIAL_PATH.exists():

        return set()

    df = pd.read_csv(
        PARTIAL_PATH
    )

    return set(
        df[
            "question_id"
        ]
        .astype(int)
        .tolist()
    )


# ============================================================
# Group summaries
# ============================================================

def summarize_group(
    df,
    group_col,
):

    rows = []

    for value, g in df.groupby(
        group_col,
        dropna=False,
    ):

        primary = (
            pairwise_stats(
                g,
                "whole_correct",
                "mn_correct",
                bootstrap=False,
            )
        )

        baseline_whole = (
            pairwise_stats(
                g,
                "baseline_correct",
                "whole_correct",
                bootstrap=False,
            )
        )

        baseline_mn = (
            pairwise_stats(
                g,
                "baseline_correct",
                "mn_correct",
                bootstrap=False,
            )
        )

        rows.append(
            {
                group_col:
                    value,

                "n":
                    int(
                        len(g)
                    ),

                "baseline_accuracy":
                    float(
                        as_bool_array(
                            g[
                                "baseline_correct"
                            ]
                        )
                        .mean()
                    ),

                "whole_accuracy":
                    float(
                        as_bool_array(
                            g[
                                "whole_correct"
                            ]
                        )
                        .mean()
                    ),

                "mn_accuracy":
                    float(
                        as_bool_array(
                            g[
                                "mn_correct"
                            ]
                        )
                        .mean()
                    ),

                "mn_minus_whole_pp":
                    primary[
                        "difference_pp"
                    ],

                "mn_vs_whole_gain":
                    primary[
                        "reference_wrong_candidate_correct"
                    ],

                "mn_vs_whole_loss":
                    primary[
                        "reference_correct_candidate_wrong"
                    ],

                "mn_vs_whole_mcnemar_p":
                    primary[
                        "mcnemar_exact_p"
                    ],

                "whole_minus_baseline_pp":
                    baseline_whole[
                        "difference_pp"
                    ],

                "mn_minus_baseline_pp":
                    baseline_mn[
                        "difference_pp"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Full frozen evaluation
# ============================================================

def run_full(
    records,
    bundle,
    U4,
    config,
    resume,
):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RESULT_PATH.exists():

        raise RuntimeError(
            "Final result already "
            "exists. Refusing to rerun."
        )

    if not resume:

        if (
            PARTIAL_PATH.exists()
            or
            RUN_STARTED_PATH.exists()
        ):

            raise RuntimeError(
                "Partial/start files "
                "already exist. "
                "Use --resume only "
                "for a technical "
                "interruption."
            )

        started = {
            "started_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "git_head":
                git_head(),

            "runner_sha256":
                sha256_file(
                    Path(
                        __file__
                    )
                ),

            "protocol_sha256":
                sha256_file(
                    PROTOCOL_PATH
                ),

            "manifest_sha256":
                EXPECTED_MANIFEST_SHA256,

            "image_inventory_sha256":
                EXPECTED_INVENTORY_SHA256,

            "controller_sha256":
                EXPECTED_CONTROLLER_SHA256,

            "u4_sha256":
                EXPECTED_U4_SHA256,

            "model_id":
                MODEL_ID,

            "model_revision":
                MODEL_REVISION,

            "samples":
                4000,

            "primary_endpoint":
                "MN-CSA accuracy minus "
                "same-run whole-head accuracy",

            "retuning_after_start":
                False,
        }

        save_json(
            RUN_STARTED_PATH,
            started,
        )

        completed = set()

    else:

        if not RUN_STARTED_PATH.exists():

            raise RuntimeError(
                "--resume requested "
                "but run-start marker "
                "is missing."
            )

        completed = (
            completed_qids()
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
        not in
        completed
    ]

    print(
        "Already completed:",
        len(
            completed
        ),
    )

    print(
        "Remaining:",
        len(
            remaining
        ),
    )

    (
        processor,
        numeral_ids,
        model,
    ) = load_model()

    identity = (
        identity_precheck(
            records,
            processor,
            model,
            U4,
        )
    )

    print(
        "Identity precheck:",
        identity,
    )

    feature_names = list(
        bundle[
            "feature_names"
        ]
    )

    for sample in tqdm(
        remaining,
        desc=(
            "TallyQA natural "
            "MN-CSA v1"
        ),
    ):

        qid = int(
            sample[
                "question_id"
            ]
        )

        image, inputs = (
            make_inputs(
                processor,
                model,
                sample,
            )
        )

        # --------------------------------------------
        # Ground truth is NOT read here.
        # --------------------------------------------

        with torch.inference_mode():

            baseline_state = (
                score_state(
                    model,
                    inputs,
                    numeral_ids,
                    DUMMY_GT,
                )
            )

        X, _ = (
            state_to_features(
                baseline_state,
                feature_names,
            )
        )

        (
            selected_alpha,
            selected_score,
            score_map,
        ) = (
            controller_decision(
                bundle,
                X,
            )
        )

        selected_alpha = float(
            selected_alpha
        )

        if (
            selected_alpha
            not in
            EXPECTED_ACTIONS
        ):

            raise RuntimeError(
                "Unexpected action "
                f"{selected_alpha} "
                f"for qid={qid}"
            )

        baseline_prediction = int(
            baseline_state[
                "best_numeral"
            ]
        )

        # ====================================================
        # EXACT SAME SELECTED ALPHA FOR BOTH ACTUATORS
        # ====================================================

        if math.isclose(
            selected_alpha,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):

            whole_prediction = (
                baseline_prediction
            )

            mn_prediction = (
                baseline_prediction
            )

            whole_hook_calls = 0
            mn_hook_calls = 0

            mn_rho = np.nan

            mn_strength_error = 0.0

            mn_subspace_residual = 0.0

        else:

            # ----------------------------------------
            # Whole-head
            # ----------------------------------------

            whole = (
                HeadGainModifier(
                    model=model,
                    layer_idx=LAYER,
                    head_idx=HEAD,
                    alpha=
                        selected_alpha,
                )
            )

            whole.register()

            try:

                with torch.inference_mode():

                    whole_state = (
                        score_state(
                            model,
                            inputs,
                            numeral_ids,
                            DUMMY_GT,
                        )
                    )

            finally:

                whole.remove()

            whole_hook_calls = int(
                whole.calls
            )

            if whole_hook_calls <= 0:

                raise RuntimeError(
                    "Whole-head hook "
                    "not called for "
                    f"qid={qid}"
                )

            whole_prediction = int(
                whole_state[
                    "best_numeral"
                ]
            )

            # ----------------------------------------
            # Frozen MN-CSA
            # ----------------------------------------

            mn = (
                MagnitudeNormalizedCSAModifier(
                    model=model,
                    alpha=
                        selected_alpha,
                    U=U4,
                )
            )

            mn.register()

            try:

                with torch.inference_mode():

                    mn_state = (
                        score_state(
                            model,
                            inputs,
                            numeral_ids,
                            DUMMY_GT,
                        )
                    )

            finally:

                mn.remove()

            mn_hook_calls = int(
                mn.calls
            )

            if mn_hook_calls <= 0:

                raise RuntimeError(
                    "MN-CSA hook "
                    "not called for "
                    f"qid={qid}"
                )

            if (
                mn.last_strength_error
                >
                1e-5
            ):

                raise RuntimeError(
                    "MN-CSA magnitude "
                    "mismatch for "
                    f"qid={qid}: "
                    f"{mn.last_strength_error}"
                )

            if (
                mn.last_subspace_residual
                >
                1e-5
            ):

                raise RuntimeError(
                    "MN-CSA delta "
                    "left U4 for "
                    f"qid={qid}: "
                    f"{mn.last_subspace_residual}"
                )

            mn_prediction = int(
                mn_state[
                    "best_numeral"
                ]
            )

            mn_rho = float(
                mn.last_rho
            )

            mn_strength_error = float(
                mn.last_strength_error
            )

            mn_subspace_residual = float(
                mn.last_subspace_residual
            )

        # ====================================================
        # Ground truth is read ONLY after both executions.
        # ====================================================

        gt = int(
            sample[
                "answer"
            ]
        )

        baseline_correct = (
            baseline_prediction
            ==
            gt
        )

        whole_correct = (
            whole_prediction
            ==
            gt
        )

        mn_correct = (
            mn_prediction
            ==
            gt
        )

        row = {
            "manifest_index":
                int(
                    sample[
                        "manifest_index"
                    ]
                ),

            "question_id":
                qid,

            "image":
                str(
                    sample[
                        "image"
                    ]
                ),

            "image_id":
                int(
                    sample[
                        "image_id"
                    ]
                ),

            "subset":
                str(
                    sample[
                        "subset"
                    ]
                ),

            "issimple":
                bool(
                    sample[
                        "issimple"
                    ]
                ),

            "ground_truth":
                gt,

            "baseline_prediction":
                baseline_prediction,

            "whole_prediction":
                whole_prediction,

            "mn_prediction":
                mn_prediction,

            "selected_alpha":
                selected_alpha,

            "selected_score":
                float(
                    selected_score
                ),

            "baseline_correct":
                bool(
                    baseline_correct
                ),

            "whole_correct":
                bool(
                    whole_correct
                ),

            "mn_correct":
                bool(
                    mn_correct
                ),

            "whole_repair":
                bool(
                    (
                        not
                        baseline_correct
                    )
                    and
                    whole_correct
                ),

            "whole_break":
                bool(
                    baseline_correct
                    and
                    (
                        not
                        whole_correct
                    )
                ),

            "mn_repair":
                bool(
                    (
                        not
                        baseline_correct
                    )
                    and
                    mn_correct
                ),

            "mn_break":
                bool(
                    baseline_correct
                    and
                    (
                        not
                        mn_correct
                    )
                ),

            "whole_to_mn_gain":
                bool(
                    (
                        not
                        whole_correct
                    )
                    and
                    mn_correct
                ),

            "whole_to_mn_loss":
                bool(
                    whole_correct
                    and
                    (
                        not
                        mn_correct
                    )
                ),

            "whole_hook_calls":
                whole_hook_calls,

            "mn_hook_calls":
                mn_hook_calls,

            "mn_rho":
                mn_rho,

            "mn_strength_error":
                mn_strength_error,

            "mn_subspace_residual":
                mn_subspace_residual,
        }

        for action in [
            0.0,
            1.5,
            2.0,
            4.0,
        ]:

            key = (
                "score_alpha_"
                +
                str(
                    action
                )
                .replace(
                    ".",
                    "p",
                )
            )

            try:

                row[
                    key
                ] = float(
                    score_map[
                        action
                    ]
                )

            except Exception:

                try:

                    row[
                        key
                    ] = float(
                        score_map[
                            str(
                                action
                            )
                        ]
                    )

                except Exception:

                    row[
                        key
                    ] = np.nan

        append_row(
            row
        )

        del inputs
        del image
        del baseline_state

        if not math.isclose(
            selected_alpha,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):

            del whole_state
            del mn_state

    # ========================================================
    # Finalize
    # ========================================================

    df = pd.read_csv(
        PARTIAL_PATH
    )

    if len(
        df
    ) != 4000:

        raise RuntimeError(
            "Expected 4000 "
            "completed rows, "
            f"found {len(df)}"
        )

    if (
        df[
            "question_id"
        ]
        .nunique()
        !=
        4000
    ):

        raise RuntimeError(
            "Final results do not "
            "have 4000 unique "
            "question IDs."
        )

    df = (
        df
        .sort_values(
            "manifest_index"
        )
        .reset_index(
            drop=True
        )
    )

    df.to_csv(
        RESULT_PATH,
        index=False,
    )

    primary = (
        pairwise_stats(
            df,
            "whole_correct",
            "mn_correct",
            bootstrap=True,
        )
    )

    baseline_whole = (
        pairwise_stats(
            df,
            "baseline_correct",
            "whole_correct",
            bootstrap=True,
        )
    )

    baseline_mn = (
        pairwise_stats(
            df,
            "baseline_correct",
            "mn_correct",
            bootstrap=True,
        )
    )

    baseline_correct_array = (
        as_bool_array(
            df[
                "baseline_correct"
            ]
        )
    )

    whole_correct_array = (
        as_bool_array(
            df[
                "whole_correct"
            ]
        )
    )

    mn_correct_array = (
        as_bool_array(
            df[
                "mn_correct"
            ]
        )
    )

    summary = {
        "experiment":
            "AROMA Natural-Domain "
            "MN-CSA Validation v1",

        "n":
            4000,

        "model_id":
            MODEL_ID,

        "model_revision":
            MODEL_REVISION,

        "primary_comparison":
            "MN-CSA vs same-run "
            "whole-head",

        "primary":
            primary,

        "secondary_baseline_vs_whole":
            baseline_whole,

        "secondary_baseline_vs_mn":
            baseline_mn,

        "baseline_accuracy":
            float(
                baseline_correct_array
                .mean()
            ),

        "whole_accuracy":
            float(
                whole_correct_array
                .mean()
            ),

        "mn_accuracy":
            float(
                mn_correct_array
                .mean()
            ),

        "whole_repairs":
            int(
                as_bool_array(
                    df[
                        "whole_repair"
                    ]
                )
                .sum()
            ),

        "whole_breaks":
            int(
                as_bool_array(
                    df[
                        "whole_break"
                    ]
                )
                .sum()
            ),

        "mn_repairs":
            int(
                as_bool_array(
                    df[
                        "mn_repair"
                    ]
                )
                .sum()
            ),

        "mn_breaks":
            int(
                as_bool_array(
                    df[
                        "mn_break"
                    ]
                )
                .sum()
            ),

        "intervention_rate":
            float(
                (
                    df[
                        "selected_alpha"
                    ]
                    !=
                    1.0
                )
                .mean()
            ),

        "median_mn_rho_nonnoop":
            (
                float(
                    df.loc[
                        df[
                            "selected_alpha"
                        ]
                        !=
                        1.0,
                        "mn_rho",
                    ]
                    .median()
                )
                if
                (
                    df[
                        "selected_alpha"
                    ]
                    !=
                    1.0
                )
                .any()
                else
                None
            ),

        "max_mn_strength_error":
            float(
                df[
                    "mn_strength_error"
                ]
                .max()
            ),

        "max_mn_subspace_residual":
            float(
                df[
                    "mn_subspace_residual"
                ]
                .max()
            ),

        "hashes":
            {
                "manifest":
                    EXPECTED_MANIFEST_SHA256,

                "image_inventory":
                    EXPECTED_INVENTORY_SHA256,

                "controller":
                    EXPECTED_CONTROLLER_SHA256,

                "u4":
                    EXPECTED_U4_SHA256,

                "protocol":
                    sha256_file(
                        PROTOCOL_PATH
                    ),

                "runner":
                    sha256_file(
                        Path(
                            __file__
                        )
                    ),
            },
    }

    save_json(
        SUMMARY_PATH,
        summary,
    )

    summarize_group(
        df,
        "subset",
    ).to_csv(
        SUBSET_PATH,
        index=False,
    )

    summarize_group(
        df,
        "ground_truth",
    ).to_csv(
        COUNT_PATH,
        index=False,
    )

    summarize_group(
        df,
        "selected_alpha",
    ).to_csv(
        ACTION_PATH,
        index=False,
    )

    run_meta = {
        "finished_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_head":
            git_head(),

        "gpu":
            (
                torch.cuda
                .get_device_name(
                    0
                )
                if
                torch.cuda
                .is_available()
                else
                "cpu"
            ),

        "summary_path":
            str(
                SUMMARY_PATH
            ),

        "result_path":
            str(
                RESULT_PATH
            ),

        "config":
            config,
    }

    save_json(
        RUN_META_PATH,
        run_meta,
    )

    print(
        "=" * 100
    )

    print(
        "FINAL NATURAL "
        "MN-CSA RESULT"
    )

    print(
        "=" * 100
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = (
        argparse
        .ArgumentParser()
    )

    parser.add_argument(
        "--protocol-only",
        action="store_true",
    )

    parser.add_argument(
        "--smoke-only",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = (
        parser
        .parse_args()
    )

    if sum(
        [
            bool(
                args.protocol_only
            ),
            bool(
                args.smoke_only
            ),
            bool(
                args.resume
            ),
        ]
    ) > 1:

        raise RuntimeError(
            "Use only one of "
            "--protocol-only, "
            "--smoke-only, "
            "--resume."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        records,
        bundle,
        U4,
        config,
    ) = protocol_audit()

    if args.protocol_only:

        print(
            "No model was loaded."
        )

        print(
            "No prediction was produced."
        )

        return

    if args.smoke_only:

        (
            processor,
            _,
            model,
        ) = load_model()

        smoke_precheck(
            records,
            processor,
            model,
            U4,
        )

        print(
            "SMOKE TEST PASSED"
        )

        return

    run_full(
        records,
        bundle,
        U4,
        config,
        resume=
            bool(
                args.resume
            ),
    )


if __name__ == "__main__":

    main()
