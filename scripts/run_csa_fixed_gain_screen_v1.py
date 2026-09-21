#!/usr/bin/env python3

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
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

sys.path.insert(
    0,
    "scripts",
)

from run_l18h13_gain_all300 import (
    MODEL_ID,
    LAYER,
    HEAD,
    prepare_inputs,
    move_inputs,
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)


# ============================================================
# PREDECLARED DEVELOPMENT SCREEN
# ============================================================

GAMMAS = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.25,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
]

SELECTION_RULE = [
    "maximize exact-count accuracy",
    "if tied: minimize breaks",
    "if tied: choose smaller gamma",
]

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

EXPECTED_N = 653

EXPECTED_ACTION_COUNTS = {
    0.0: 24,
    1.5: 32,
    2.0: 63,
    4.0: 534,
}

# gamma=1 MUST reproduce the already archived
# same-alpha CSA result exactly.
EXPECTED_GAMMA1 = {
    "correct": 217,
    "repairs": 49,
    "breaks": 7,
    "net": 42,
}

DUMMY_GT = 1

START = 1664
END = 1792


# ============================================================
# INPUTS
# ============================================================

V3_RESULTS = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller/"
    "v3_final_results.csv"
)

IMAGE_DIR = Path(
    "data/proc_count_causal_v3/images"
)

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

STRENGTH_CSV = Path(
    "outputs/aroma2/"
    "csa_actuation_strength_v1/"
    "actuation_strength.csv"
)

B2_CSV = Path(
    "outputs/aroma2/"
    "csa_gate_b2_mn_v1/"
    "gate_b2_mn_results.csv"
)

B2_JSON = Path(
    "outputs/aroma2/"
    "csa_gate_b2_mn_v1/"
    "gate_b2_mn_result.json"
)


# ============================================================
# OUTPUTS
# ============================================================

OUT_DIR = Path(
    "outputs/aroma2/"
    "csa_fixed_gain_screen_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL_JSONL = (
    OUT_DIR
    / "partial_results.jsonl"
)

RESULTS_CSV = (
    OUT_DIR
    / "fixed_gain_screen_results.csv"
)

SUMMARY_CSV = (
    OUT_DIR
    / "fixed_gain_screen_summary.csv"
)

RESULT_JSON = (
    OUT_DIR
    / "fixed_gain_screen_result.json"
)

SELECTION_JSON = (
    OUT_DIR
    / "fixed_gain_selection.json"
)

PROTOCOL_JSON = (
    OUT_DIR
    / "protocol.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(chunk)

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


def load_jsonl(path):

    path = Path(path)

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def append_jsonl(
    path,
    obj,
):

    with Path(path).open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as f:

        f.write(
            json.dumps(
                obj,
                sort_keys=True,
            )
        )
        f.write("\n")
        f.flush()
        os.fsync(
            f.fileno()
        )


def save_json(
    path,
    obj,
):

    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def gamma_key(gamma):

    return (
        f"g{gamma:.2f}"
        .replace(
            ".",
            "p",
        )
    )


def resolve_image(sample_id):

    candidates = [
        p
        for p in IMAGE_DIR.glob(
            sample_id + ".*"
        )
        if p.suffix.lower()
        in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }
    ]

    if len(candidates) != 1:

        raise RuntimeError(
            f"{sample_id}: expected "
            f"exactly one image, "
            f"got {candidates}"
        )

    return candidates[0]


def find_prediction_column(
    df,
    method_keyword,
):

    candidates = []

    for col in df.columns:

        low = col.lower()

        if (
            method_keyword in low
            and
            (
                "prediction" in low
                or
                low.endswith("_pred")
            )
        ):

            candidates.append(col)

    if len(candidates) != 1:

        raise RuntimeError(
            f"Could not uniquely locate "
            f"{method_keyword} prediction "
            f"column in B2 CSV. "
            f"Candidates={candidates}. "
            f"All columns={list(df.columns)}"
        )

    return candidates[0]


# ============================================================
# FIXED-GAIN CSA MODIFIER
#
# delta =
# gamma * (alpha - 1) * P_U h
#
# Same hook / pooling / U4 as B2.
# ============================================================

class FixedGainCSAModifier:

    def __init__(
        self,
        model,
        alpha,
        gamma,
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

        if int(
            self.o_proj.in_features
        ) != 4096:

            raise RuntimeError(
                "Unexpected o_proj "
                "input dimension."
            )

        self.alpha = float(
            alpha
        )

        self.gamma = float(
            gamma
        )

        self.U = torch.tensor(
            np.asarray(
                U,
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )

        if tuple(
            self.U.shape
        ) != (
            128,
            4,
        ):

            raise RuntimeError(
                f"Bad U shape: "
                f"{tuple(self.U.shape)}"
            )

        self.handle = None
        self.calls = 0

        self.last_h_norm = None
        self.last_projected_norm = None
        self.last_rho = None
        self.last_shift_norm = None
        self.last_shift_to_whole_ratio = None


    def hook(
        self,
        module,
        inputs,
    ):

        if not inputs:

            raise RuntimeError(
                "Empty hook input."
            )

        x = inputs[0]

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

        U = self.U.to(
            device=H.device
        )

        projected = (
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
                projected,
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
                "Projected h norm "
                "too small."
            )

        delta_h = (
            self.gamma
            *
            (
                self.alpha
                - 1.0
            )
            *
            projected
        )

        shift_norm = (
            torch.linalg
            .vector_norm(
                delta_h,
                dim=-1,
                keepdim=True,
            )
        )

        rho = (
            projected_norm
            /
            h_norm
        )

        whole_target_norm = (
            abs(
                self.alpha
                - 1.0
            )
            *
            h_norm
        )

        if (
            abs(
                self.alpha
                - 1.0
            )
            <= 1e-12
        ):

            shift_to_whole = (
                torch.zeros_like(
                    whole_target_norm
                )
            )

        else:

            shift_to_whole = (
                shift_norm
                /
                torch.clamp(
                    whole_target_norm,
                    min=1e-12,
                )
            )

        self.last_h_norm = float(
            h_norm[
                0,
                0
            ].item()
        )

        self.last_projected_norm = float(
            projected_norm[
                0,
                0
            ].item()
        )

        self.last_rho = float(
            rho[
                0,
                0
            ].item()
        )

        self.last_shift_norm = float(
            shift_norm[
                0,
                0
            ].item()
        )

        self.last_shift_to_whole_ratio = float(
            shift_to_whole[
                0,
                0
            ].item()
        )

        H_new = (
            H
            +
            delta_h
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

        if len(inputs) == 1:

            return (
                x_new,
            )

        return (
            x_new,
            *inputs[1:],
        )


    def register(self):

        self.calls = 0

        self.handle = (
            self.o_proj
            .register_forward_pre_hook(
                self.hook
            )
        )


    def remove(self):

        if self.handle is not None:

            self.handle.remove()

            self.handle = None


# ============================================================
# PREFLIGHT
# ============================================================

print("=" * 96)
print(
    "FIXED-GAIN PROJECTED CSA "
    "DEVELOPMENT SCREEN v1"
)
print("=" * 96)

for p in [
    V3_RESULTS,
    U4_PATH,
    STRENGTH_CSV,
    B2_CSV,
    B2_JSON,
]:

    if not p.exists():

        raise RuntimeError(
            f"Missing required "
            f"artifact: {p}"
        )


protocol = {
    "stage":
        "fixed_gain_projected_csa_screen_v1",

    "role":
        "post-hoc development baseline",

    "cohort":
        "frozen v3 non-NoOp N=653",

    "gammas":
        GAMMAS,

    "selection_rule":
        SELECTION_RULE,

    "operator":
        (
            "delta = gamma * "
            "(alpha - 1) * P_U h"
        ),

    "rank":
        4,

    "model_revision":
        MODEL_REVISION,

    "confirmation_sets_touched":
        False,

    "u4_sha256":
        sha256_file(
            U4_PATH
        ),

    "v3_results_sha256":
        sha256_file(
            V3_RESULTS
        ),

    "strength_csv_sha256":
        sha256_file(
            STRENGTH_CSV
        ),

    "b2_result_sha256":
        sha256_file(
            B2_JSON
        ),

    "git_head":
        git_head(),
}

save_json(
    PROTOCOL_JSON,
    protocol,
)

print(
    "Protocol written before "
    "model inference:"
)

print(
    PROTOCOL_JSON
)

print()
print(
    "Gammas:",
    GAMMAS
)

print(
    "Selection rule:",
    SELECTION_RULE
)


# ============================================================
# LOAD FROZEN POPULATION
# ============================================================

df = pd.read_csv(
    V3_RESULTS
)

df[
    "selected_alpha"
] = (
    df[
        "selected_alpha"
    ]
    .astype(float)
)

data = (
    df[
        ~np.isclose(
            df[
                "selected_alpha"
            ],
            1.0,
        )
    ]
    .copy()
    .sort_values(
        "sample_id"
    )
    .reset_index(
        drop=True
    )
)

if len(data) != EXPECTED_N:

    raise RuntimeError(
        f"Expected {EXPECTED_N}, "
        f"got {len(data)}"
    )

counts = {
    float(k):
        int(v)
    for k, v
    in (
        data[
            "selected_alpha"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
        .items()
    )
}

if counts != EXPECTED_ACTION_COUNTS:

    raise RuntimeError(
        f"Action-count mismatch: "
        f"{counts}"
    )

for sid in (
    data[
        "sample_id"
    ]
    .astype(str)
):

    resolve_image(sid)

print(
    "Frozen N=653 cohort: PASS"
)

print(
    "Action counts:",
    counts
)


# ============================================================
# LOAD OLD B2 REFERENCE
# ============================================================

b2 = pd.read_csv(
    B2_CSV
)

b2[
    "sample_id"
] = (
    b2[
        "sample_id"
    ]
    .astype(str)
)

baseline_col = (
    find_prediction_column(
        b2,
        "baseline",
    )
)

print(
    "Archived baseline prediction "
    "column:",
    baseline_col
)

baseline_map = dict(
    zip(
        b2[
            "sample_id"
        ],
        b2[
            baseline_col
        ].astype(int),
    )
)

if set(
    data[
        "sample_id"
    ].astype(str)
) != set(
    baseline_map.keys()
):

    raise RuntimeError(
        "B2 baseline cohort does not "
        "match frozen N=653 cohort."
    )


# ============================================================
# U4
# ============================================================

U4 = np.load(
    U4_PATH
).astype(
    np.float32
)

if U4.shape != (
    128,
    4,
):

    raise RuntimeError(
        f"Bad U4 shape: "
        f"{U4.shape}"
    )

orth_error = float(
    np.max(
        np.abs(
            U4.T @ U4
            -
            np.eye(
                4,
                dtype=np.float32,
            )
        )
    )
)

print(
    "U4 orthogonality error:",
    orth_error
)

if orth_error > 1e-5:

    raise RuntimeError(
        "U4 orthogonality failed."
    )


# ============================================================
# RESUME
# ============================================================

records = load_jsonl(
    PARTIAL_JSONL
)

if len(records) > EXPECTED_N:

    raise RuntimeError(
        "Partial output too long."
    )

expected_ids = [
    str(x)
    for x in data[
        "sample_id"
    ]
]

actual_ids = [
    str(r[
        "sample_id"
    ])
    for r in records
]

if (
    actual_ids
    !=
    expected_ids[
        :len(actual_ids)
    ]
):

    raise RuntimeError(
        "Resume sample order mismatch."
    )

expected_gamma_keys = {
    gamma_key(g)
    for g in GAMMAS
}

for row in records:

    if set(
        row[
            "gammas"
        ].keys()
    ) != expected_gamma_keys:

        raise RuntimeError(
            "Incomplete gamma set "
            "in partial row."
        )

print(
    "Resume:",
    len(records),
    "/",
    EXPECTED_N
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
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
    range(16)
):

    raise RuntimeError(
        "Numeral support != 0..15"
    )

print(
    "Numeral support audit: PASS"
)

print()
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
        device_map=
            "auto",
        attn_implementation=
            "eager",
    )
)

model.eval()

for p in model.parameters():

    p.requires_grad_(
        False
    )

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )


def get_inputs(sid):

    image = (
        Image.open(
            resolve_image(
                sid
            )
        )
        .convert(
            "RGB"
        )
    )

    inputs = prepare_inputs(
        processor,
        image,
    )

    return move_inputs(
        inputs,
        model,
    )


# ============================================================
# FORMAL DEVELOPMENT SCREEN
# ============================================================

print()
print("=" * 96)
print(
    "RUNNING FIXED-GLOBAL-GAIN "
    "CSA SCREEN"
)
print("=" * 96)

start_idx = len(
    records
)

for idx in tqdm(
    range(
        start_idx,
        EXPECTED_N,
    ),
    initial=
        start_idx,
    total=
        EXPECTED_N,
):

    row = data.iloc[
        idx
    ]

    sid = str(
        row[
            "sample_id"
        ]
    )

    gt = int(
        row[
            "ground_truth"
        ]
    )

    alpha = float(
        row[
            "selected_alpha"
        ]
    )

    baseline_prediction = int(
        baseline_map[
            sid
        ]
    )

    inputs = get_inputs(
        sid
    )

    gamma_results = {}

    for gamma in GAMMAS:

        key = gamma_key(
            gamma
        )

        modifier = (
            FixedGainCSAModifier(
                model=model,
                alpha=alpha,
                gamma=gamma,
                U=U4,
            )
        )

        modifier.register()

        try:

            state = score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )

        finally:

            modifier.remove()

        if modifier.calls <= 0:

            raise RuntimeError(
                f"{sid} gamma={gamma}: "
                "hook not called."
            )

        pred = int(
            state[
                "best_numeral"
            ]
        )

        expected_ratio = (
            float(gamma)
            *
            float(
                modifier.last_rho
            )
        )

        ratio_error = abs(
            float(
                modifier
                .last_shift_to_whole_ratio
            )
            -
            expected_ratio
        )

        if ratio_error > 1e-5:

            raise RuntimeError(
                f"{sid} gamma={gamma}: "
                f"shift ratio invariant "
                f"failed: {ratio_error}"
            )

        gamma_results[
            key
        ] = {
            "gamma":
                float(
                    gamma
                ),

            "prediction":
                pred,

            "correct":
                bool(
                    pred == gt
                ),

            "repair":
                bool(
                    baseline_prediction
                    != gt
                    and
                    pred == gt
                ),

            "break":
                bool(
                    baseline_prediction
                    == gt
                    and
                    pred != gt
                ),

            "rho":
                float(
                    modifier.last_rho
                ),

            "shift_to_whole_ratio":
                float(
                    modifier
                    .last_shift_to_whole_ratio
                ),
        }

    record = {
        "sample_id":
            sid,

        "ground_truth":
            gt,

        "selected_alpha":
            alpha,

        "baseline_prediction":
            baseline_prediction,

        "baseline_correct":
            bool(
                baseline_prediction
                ==
                gt
            ),

        "gammas":
            gamma_results,
    }

    append_jsonl(
        PARTIAL_JSONL,
        record,
    )

    records.append(
        record
    )

    del inputs

    if (
        len(records) == 1
        or
        len(records) % 10 == 0
        or
        len(records) == EXPECTED_N
    ):

        print()
        print(
            f"[{len(records):3d}/"
            f"{EXPECTED_N}]"
        )

        current = []

        for gamma in GAMMAS:

            key = gamma_key(
                gamma
            )

            rr = [
                x[
                    "gammas"
                ][
                    key
                ]
                for x in records
            ]

            correct = sum(
                int(
                    x[
                        "correct"
                    ]
                )
                for x in rr
            )

            repairs = sum(
                int(
                    x[
                        "repair"
                    ]
                )
                for x in rr
            )

            breaks = sum(
                int(
                    x[
                        "break"
                    ]
                )
                for x in rr
            )

            current.append(
                {
                    "gamma":
                        gamma,

                    "accuracy":
                        correct
                        /
                        len(rr),

                    "correct":
                        correct,

                    "repairs":
                        repairs,

                    "breaks":
                        breaks,
                }
            )

        current = sorted(
            current,
            key=lambda x: (
                -x[
                    "correct"
                ],
                x[
                    "breaks"
                ],
                x[
                    "gamma"
                ],
            ),
        )

        print(
            "Current top 3:"
        )

        for x in current[:3]:

            print(
                "  gamma="
                f"{x['gamma']:<4} | "
                f"acc="
                f"{x['accuracy']:.4f} | "
                f"correct="
                f"{x['correct']} | "
                f"repairs="
                f"{x['repairs']} | "
                f"breaks="
                f"{x['breaks']}"
            )


# ============================================================
# FINAL DEVELOPMENT SUMMARY
# ============================================================

if len(records) != EXPECTED_N:

    raise RuntimeError(
        "Incomplete screen."
    )

summary_rows = []

flat_rows = []

for gamma in GAMMAS:

    key = gamma_key(
        gamma
    )

    rr = [
        x[
            "gammas"
        ][
            key
        ]
        for x in records
    ]

    correct = sum(
        int(
            x[
                "correct"
            ]
        )
        for x in rr
    )

    repairs = sum(
        int(
            x[
                "repair"
            ]
        )
        for x in rr
    )

    breaks = sum(
        int(
            x[
                "break"
            ]
        )
        for x in rr
    )

    net = (
        repairs
        -
        breaks
    )

    ratios = np.asarray(
        [
            x[
                "shift_to_whole_ratio"
            ]
            for x in rr
        ],
        dtype=np.float64,
    )

    rhos = np.asarray(
        [
            x[
                "rho"
            ]
            for x in rr
        ],
        dtype=np.float64,
    )

    summary_rows.append(
        {
            "gamma":
                float(
                    gamma
                ),

            "correct":
                correct,

            "accuracy":
                correct
                /
                EXPECTED_N,

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net":
                net,

            "median_rho":
                float(
                    np.median(
                        rhos
                    )
                ),

            "median_shift_to_whole_ratio":
                float(
                    np.median(
                        ratios
                    )
                ),

            "mean_shift_to_whole_ratio":
                float(
                    np.mean(
                        ratios
                    )
                ),

            "q25_shift_to_whole_ratio":
                float(
                    np.quantile(
                        ratios,
                        0.25,
                    )
                ),

            "q75_shift_to_whole_ratio":
                float(
                    np.quantile(
                        ratios,
                        0.75,
                    )
                ),
        }
    )

    for row in records:

        x = row[
            "gammas"
        ][
            key
        ]

        flat_rows.append(
            {
                "sample_id":
                    row[
                        "sample_id"
                    ],

                "ground_truth":
                    row[
                        "ground_truth"
                    ],

                "selected_alpha":
                    row[
                        "selected_alpha"
                    ],

                "baseline_prediction":
                    row[
                        "baseline_prediction"
                    ],

                "gamma":
                    float(
                        gamma
                    ),

                "prediction":
                    x[
                        "prediction"
                    ],

                "correct":
                    x[
                        "correct"
                    ],

                "repair":
                    x[
                        "repair"
                    ],

                "break":
                    x[
                        "break"
                    ],

                "rho":
                    x[
                        "rho"
                    ],

                "shift_to_whole_ratio":
                    x[
                        "shift_to_whole_ratio"
                    ],
            }
        )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df = (
    summary_df
    .sort_values(
        [
            "correct",
            "breaks",
            "gamma",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)

pd.DataFrame(
    flat_rows
).to_csv(
    RESULTS_CSV,
    index=False,
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================
# HARD REPRODUCTION CHECK: gamma=1
# ============================================================

gamma1 = (
    summary_df[
        np.isclose(
            summary_df[
                "gamma"
            ],
            1.0,
        )
    ]
    .iloc[
        0
    ]
)

actual_gamma1 = {
    "correct":
        int(
            gamma1[
                "correct"
            ]
        ),

    "repairs":
        int(
            gamma1[
                "repairs"
            ]
        ),

    "breaks":
        int(
            gamma1[
                "breaks"
            ]
        ),

    "net":
        int(
            gamma1[
                "net"
            ]
        ),
}

print()
print("=" * 96)
print(
    "GAMMA=1 REPRODUCTION CHECK"
)
print("=" * 96)

print(
    "expected:",
    EXPECTED_GAMMA1
)

print(
    "actual  :",
    actual_gamma1
)

if (
    actual_gamma1
    !=
    EXPECTED_GAMMA1
):

    raise RuntimeError(
        "gamma=1 does not reproduce "
        "archived same-alpha CSA."
    )

print(
    "GAMMA=1 EXACT AGGREGATE "
    "REPRODUCTION: PASS"
)


# ============================================================
# SELECT GLOBAL GAMMA
# ============================================================

winner = (
    summary_df
    .iloc[
        0
    ]
)

selected_gamma = float(
    winner[
        "gamma"
    ]
)

selection = {
    "stage":
        "fixed_gain_projected_csa_screen_v1",

    "development_only":
        True,

    "selected_gamma":
        selected_gamma,

    "selection_rule":
        SELECTION_RULE,

    "selected_metrics":
        {
            "correct":
                int(
                    winner[
                        "correct"
                    ]
                ),

            "accuracy":
                float(
                    winner[
                        "accuracy"
                    ]
                ),

            "repairs":
                int(
                    winner[
                        "repairs"
                    ]
                ),

            "breaks":
                int(
                    winner[
                        "breaks"
                    ]
                ),

            "net":
                int(
                    winner[
                        "net"
                    ]
                ),

            "median_shift_to_whole_ratio":
                float(
                    winner[
                        "median_shift_to_whole_ratio"
                    ]
                ),
        },

    "confirmation_data_touched":
        False,
}

save_json(
    SELECTION_JSON,
    selection,
)


# ============================================================
# EXISTING MN / WHOLE DEVELOPMENT REFERENCES
# ============================================================

b2_json = json.loads(
    B2_JSON.read_text(
        encoding="utf-8"
    )
)

result = {
    "stage":
        "fixed_gain_projected_csa_screen_v1",

    "status":
        "development_screen_complete",

    "n":
        EXPECTED_N,

    "gammas":
        GAMMAS,

    "selection_rule":
        SELECTION_RULE,

    "selected_gamma":
        selected_gamma,

    "selected_fixed_gain":
        selection[
            "selected_metrics"
        ],

    "archived_same_alpha":
        b2_json[
            "same_alpha_csa"
        ],

    "archived_mn_csa":
        b2_json[
            "mn_csa"
        ],

    "archived_whole_head":
        b2_json[
            "whole"
        ],

    "important_note":
        (
            "The selected fixed gamma "
            "was chosen on the same "
            "development cohort and "
            "must not be interpreted "
            "as confirmatory evidence. "
            "A fresh confirmation set "
            "is required."
        ),

    "v4_touched":
        False,

    "new_confirmation_touched":
        False,

    "protocol_sha256":
        sha256_file(
            PROTOCOL_JSON
        ),
}

save_json(
    RESULT_JSON,
    result,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 96)
print(
    "FIXED-GAIN CSA DEVELOPMENT "
    "SCREEN RESULT"
)
print("=" * 96)

print()
print(
    summary_df[
        [
            "gamma",
            "accuracy",
            "correct",
            "repairs",
            "breaks",
            "net",
            "median_shift_to_whole_ratio",
        ]
    ]
    .to_string(
        index=False
    )
)

print()
print(
    "SELECTED GLOBAL GAMMA:",
    selected_gamma
)

print()
print(
    "Selected development metrics:",
    selection[
        "selected_metrics"
    ]
)

print()
print(
    "Archived MN-CSA:",
    b2_json[
        "mn_csa"
    ]
)

print(
    "Archived whole-head:",
    b2_json[
        "whole"
    ]
)

print()
print(
    "IMPORTANT:"
)

print(
    "This is DEVELOPMENT selection only."
)

print(
    "No v4 or new confirmation data "
    "were used."
)

print()
print(
    "Saved:"
)

print(
    " ",
    SUMMARY_CSV
)

print(
    " ",
    SELECTION_JSON
)

print(
    " ",
    RESULT_JSON
)

print()
print(
    "FIXED-GAIN SCREEN COMPLETE"
)


if torch.cuda.is_available():

    torch.cuda.empty_cache()

gc.collect()
