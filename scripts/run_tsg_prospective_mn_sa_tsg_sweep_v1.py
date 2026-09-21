import hashlib
import json
import math
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
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)


# ============================================================
# FROZEN ARTIFACTS
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

META = Path(
    "data/proc_count_tsg_prospective_v1/"
    "metadata.jsonl"
)

FREEZE = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "freeze_v1/"
    "tsg_prospective_cohort_freeze_v1.json"
)

PROTOCOL = Path(
    "outputs/proc_count_sa_dev_v1/"
    "tsg_protocol_v1/"
    "tsg_sa_protocol_v1.json"
)

ACTION = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "action_manifest_v1/"
    "tsg_prospective_action_manifest_v1.csv"
)

NAMESPACE = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "namespace_v1/"
    "tsg_prospective_namespace_v1.csv"
)

GRAD_ROOT = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "gradient_collection_v1"
)

U4_PATH = Path(
    "outputs/aroma2/"
    "csa_gate_a_v3/"
    "U4_primary.npy"
)

OUT = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "mn_sa_tsg_sweep_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = OUT / "partial_results.jsonl"

RESULTS = OUT / "mn_sa_tsg_results.csv"

SUMMARY = OUT / "mn_sa_tsg_summary.csv"

SELECTION = OUT / "mn_sa_tsg_selection.json"

EXECUTION_FREEZE = (
    OUT / "execution_freeze.json"
)


# ============================================================
# EXPECTED HASHES
# ============================================================

EXPECTED_META_SHA = (
    "b7e7cb1f2c58cffdfea06aa77a4fceeab64de7102"
    "d70b2fc99c3e3d731d72a0a"
)

EXPECTED_FREEZE_SHA = (
    "8e863fd29561ca03862e00c482ecdb541eb1e15382"
    "d366f218208eea81b50b22"
)

EXPECTED_PROTOCOL_SHA = (
    "ed04264d7b2ac2c03b998d2f128924b997a1e1b06"
    "b841c510c814d469c5d8938"
)

EXPECTED_ACTION_SHA = (
    "0c4f3cc02e5cde5661612e7f6ee1dd5367434b0e1"
    "cd9bdfab99b5b27aae11b04"
)

EXPECTED_NAMESPACE_SHA = (
    "295a79870c0d1762da1c8fe36774b9122edcbbb8ca"
    "23503a32cd3cef6d29b278"
)

EXPECTED_GRAD_SHA = (
    "e5f932a1266860549243be49e142af6ae6f83b31cd8"
    "be56570c932d567559f5b"
)

EXPECTED_H_SHA = (
    "c79f596de6fa348d0ce3065a31397e82790cd5afbc6"
    "194067e3aa3bb76e9c970"
)

EXPECTED_N = 500

EXPECTED_ACTIONS = {
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
}

DUMMY_GT = 1

HEAD_DIM = 128
START = HEAD * HEAD_DIM
END = START + HEAD_DIM


# ============================================================
# HELPERS
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


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
    ) as f:

        f.write(
            json.dumps(obj)
            + "\n"
        )

        f.flush()


def bkey(x):

    return f"{float(x):.2f}"


# ============================================================
# HARD HASH AUDIT
# ============================================================

checks = [
    (
        META,
        EXPECTED_META_SHA,
        "metadata",
    ),
    (
        FREEZE,
        EXPECTED_FREEZE_SHA,
        "cohort freeze",
    ),
    (
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA,
        "TSG protocol",
    ),
    (
        ACTION,
        EXPECTED_ACTION_SHA,
        "action manifest",
    ),
    (
        NAMESPACE,
        EXPECTED_NAMESPACE_SHA,
        "namespace",
    ),
    (
        GRAD_ROOT / "gradient_matrix.npy",
        EXPECTED_GRAD_SHA,
        "gradient matrix",
    ),
    (
        GRAD_ROOT / "pooled_h_matrix.npy",
        EXPECTED_H_SHA,
        "pooled-h matrix",
    ),
]

for path, expected, name in checks:

    actual = sha256(path)

    if actual != expected:

        raise RuntimeError(
            f"{name} SHA256 mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


print(
    "PASS: all frozen artifacts verified"
)


# ============================================================
# PROTOCOL
# ============================================================

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

TAU = float(
    protocol["tau"]
)

S_REF = float(
    protocol["s_ref"]
)

BETA_MIN = float(
    protocol["beta_eff_min"]
)

BETA_MAX = float(
    protocol["beta_eff_max"]
)

BETA0_GRID = [
    float(x)
    for x in protocol[
        "beta0_grid"
    ]
]

if not np.isclose(
    TAU,
    1.0 / 3.0,
):
    raise RuntimeError(
        "Frozen tau is not 1/3."
    )

if not np.isclose(
    S_REF,
    0.06780448298801545,
):
    raise RuntimeError(
        "Frozen s_ref mismatch."
    )


# ============================================================
# FREEZE EXECUTION DETAILS BEFORE EVALUATION
# ============================================================

execution = {
    "experiment":
        "tsg_prospective_mn_sa_tsg_sweep_v1",

    "n":
        500,

    "methods": [
        "MN",
        "SA",
        "TSG-SA",
    ],

    "global_grid_for_all_methods":
        BETA0_GRID,

    "MN_rule":
        "beta_eff = beta0",

    "SA_rule":
        "beta_eff = beta0",

    "TSG_SA_rule":
        "beta_eff = clip("
        "beta0*(s_ref/s)^(1/3),"
        "0.25,2.50)",

    "tau":
        TAU,

    "s_ref":
        S_REF,

    "alpha_1_policy":
        "exact identity",

    "alpha_0_policy":
        "preserve original semantics with beta=1.0; "
        "not gain calibrated",

    "selection_rule":
        "highest full exact-count accuracy; "
        "ties -> fewer breaks; "
        "ties -> smaller beta0",

    "primary_comparison":
        "selected TSG-SA vs selected SA",

    "secondary_comparison":
        "selected TSG-SA vs selected MN",
}

if EXECUTION_FREEZE.exists():

    old = json.loads(
        EXECUTION_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    if old != execution:
        raise RuntimeError(
            "Existing execution freeze differs."
        )

else:

    EXECUTION_FREEZE.write_text(
        json.dumps(
            execution,
            indent=2,
        ),
        encoding="utf-8",
    )


print(
    "Execution-freeze SHA256:",
    sha256(
        EXECUTION_FREEZE
    ),
)

print(
    "Frozen global grid:",
    BETA0_GRID,
)

print(
    "Frozen tau:",
    TAU,
)

print(
    "Frozen s_ref:",
    S_REF,
)


# ============================================================
# DATASET / IDS
# ============================================================

samples = load_jsonl(
    META
)

samples = sorted(
    samples,
    key=lambda r: str(
        r["sample_id"]
    ),
)

if len(samples) != EXPECTED_N:

    raise RuntimeError(
        "Dataset N mismatch."
    )


raw_ids = np.array(
    [
        str(r["sample_id"])
        for r in samples
    ],
    dtype=str,
)

sample_map = {
    str(r["sample_id"]):
        r
    for r in samples
}


namespace = pd.read_csv(
    NAMESPACE
)

namespace[
    "raw_sample_id"
] = (
    namespace[
        "raw_sample_id"
    ].astype(str)
)

namespace[
    "cohort_uid"
] = (
    namespace[
        "cohort_uid"
    ].astype(str)
)

namespace = (
    namespace
    .sort_values(
        "raw_sample_id"
    )
    .reset_index(
        drop=True
    )
)

if not np.array_equal(
    raw_ids,
    namespace[
        "raw_sample_id"
    ].to_numpy(),
):

    raise RuntimeError(
        "Namespace order mismatch."
    )


uid_map = dict(
    zip(
        namespace[
            "raw_sample_id"
        ],
        namespace[
            "cohort_uid"
        ],
    )
)


# ============================================================
# ACTION MANIFEST
# ============================================================

actions = pd.read_csv(
    ACTION
)

actions[
    "raw_sample_id"
] = (
    actions[
        "raw_sample_id"
    ].astype(str)
)

actions[
    "cohort_uid"
] = (
    actions[
        "cohort_uid"
    ].astype(str)
)

if len(actions) != EXPECTED_N:

    raise RuntimeError(
        "Action N mismatch."
    )


if not np.array_equal(
    actions[
        "raw_sample_id"
    ].to_numpy(),
    raw_ids,
):

    raise RuntimeError(
        "Action ordering mismatch."
    )


if not np.array_equal(
    actions[
        "cohort_uid"
    ].to_numpy(),
    namespace[
        "cohort_uid"
    ].to_numpy(),
):

    raise RuntimeError(
        "Action cohort_uid mismatch."
    )


for alpha in actions[
    "selected_alpha"
].astype(float):

    if alpha not in EXPECTED_ACTIONS:

        raise RuntimeError(
            f"Unexpected alpha={alpha}"
        )


# ============================================================
# G / H / U
# ============================================================

G = np.load(
    GRAD_ROOT
    / "gradient_matrix.npy"
).astype(np.float64)

H_ARCH = np.load(
    GRAD_ROOT
    / "pooled_h_matrix.npy"
).astype(np.float64)

G_IDS = (
    np.load(
        GRAD_ROOT
        / "sample_ids.npy",
        allow_pickle=True,
    )
    .astype(str)
)

U4 = np.load(
    U4_PATH
).astype(np.float64)


if G.shape != (500, 128):

    raise RuntimeError(
        f"Bad G shape: {G.shape}"
    )


if H_ARCH.shape != (500, 128):

    raise RuntimeError(
        f"Bad H shape: {H_ARCH.shape}"
    )


if U4.shape != (128, 4):

    raise RuntimeError(
        f"Bad U4 shape: {U4.shape}"
    )


if not np.array_equal(
    G_IDS,
    raw_ids,
):

    raise RuntimeError(
        "Gradient ID mismatch."
    )


P = U4 @ U4.T

PG = G @ P

PG_NORM = np.linalg.norm(
    PG,
    axis=1,
)

H_NORM = np.linalg.norm(
    H_ARCH,
    axis=1,
)


if float(
    PG_NORM.min()
) <= 1e-12:

    raise RuntimeError(
        "Projected gradient norm too small."
    )


SA_DIR = (
    PG
    /
    PG_NORM[:, None]
)


SENSITIVITY = (
    H_NORM
    *
    PG_NORM
)


print(
    "Prospective sensitivity "
    "min/median/max:",
    float(
        SENSITIVITY.min()
    ),
    float(
        np.median(
            SENSITIVITY
        )
    ),
    float(
        SENSITIVITY.max()
    ),
)


# ============================================================
# INTERVENTION MODIFIER
# ============================================================

class DirectionModifier:

    def __init__(
        self,
        model,
        alpha,
        beta_eff,
        direction_mode,
        U=None,
        sa_direction=None,
    ):

        layer = (
            model
            .model
            .language_model
            .layers[LAYER]
        )

        self.o_proj = (
            layer
            .cross_attn
            .o_proj
        )

        self.alpha = float(
            alpha
        )

        self.beta_eff = float(
            beta_eff
        )

        self.direction_mode = str(
            direction_mode
        )

        self.U = None

        if U is not None:

            self.U = torch.tensor(
                np.asarray(
                    U,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            )


        self.sa_direction = None

        if sa_direction is not None:

            d = np.asarray(
                sa_direction,
                dtype=np.float32,
            )

            dn = float(
                np.linalg.norm(d)
            )

            if not np.isclose(
                dn,
                1.0,
                atol=1e-5,
            ):

                raise RuntimeError(
                    f"SA direction norm={dn}"
                )

            self.sa_direction = (
                torch.tensor(
                    d,
                    dtype=torch.float32,
                )
            )


        self.handle = None
        self.calls = 0

        self.last_h_norm = None
        self.last_shift_norm = None
        self.last_shift_ratio = None


    def hook(
        self,
        module,
        inputs,
    ):

        if not inputs:

            raise RuntimeError(
                "Empty hook inputs."
            )


        x = inputs[0]

        H = x[
            ...,
            START:END
        ]

        h = (
            H.float()
            .mean(dim=1)
        )

        h_norm = (
            torch.linalg
            .vector_norm(
                h,
                dim=-1,
                keepdim=True,
            )
        )


        if (
            self.direction_mode
            == "MN"
        ):

            U = self.U.to(
                device=H.device
            )

            projected = (
                (h @ U)
                @ U.T
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
                projected_norm.min().item()
            ) <= 1e-12:

                raise RuntimeError(
                    "MN projection too small."
                )

            direction = (
                projected
                /
                projected_norm
            )


        elif (
            self.direction_mode
            == "SA"
        ):

            direction = (
                self.sa_direction
                .to(
                    device=H.device
                )
                .view(
                    1,
                    HEAD_DIM,
                )
            )


        else:

            raise RuntimeError(
                f"Unknown direction mode: "
                f"{self.direction_mode}"
            )


        delta = (
            self.beta_eff
            *
            (
                self.alpha
                - 1.0
            )
            *
            h_norm
            *
            direction
        )


        shift_norm = (
            torch.linalg
            .vector_norm(
                delta,
                dim=-1,
                keepdim=True,
            )
        )


        whole_reference = (
            abs(
                self.alpha
                - 1.0
            )
            *
            h_norm
        )


        shift_ratio = (
            shift_norm
            /
            torch.clamp(
                whole_reference,
                min=1e-12,
            )
        )


        H_new = (
            H
            +
            delta
            .to(H.dtype)
            .unsqueeze(1)
        )


        x_new = x.clone()

        x_new[
            ...,
            START:END
        ] = H_new


        self.calls += 1

        self.last_h_norm = float(
            h_norm[
                0,
                0,
            ].item()
        )

        self.last_shift_norm = float(
            shift_norm[
                0,
                0,
            ].item()
        )

        self.last_shift_ratio = float(
            shift_ratio[
                0,
                0,
            ].item()
        )


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
# RESUME
# ============================================================

records = load_jsonl(
    PARTIAL
)


if records:

    partial_ids = [
        str(r["raw_sample_id"])
        for r in records
    ]

    expected_prefix = (
        raw_ids[
            :len(records)
        ].tolist()
    )

    if partial_ids != expected_prefix:

        raise RuntimeError(
            "Partial results are not "
            "an exact prospective prefix."
        )


    print(
        f"Resuming from "
        f"{len(records)}/500"
    )


# ============================================================
# MODEL
# ============================================================

print(
    "Loading processor..."
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
)

numeral_ids = numeral_token_ids(
    processor
)


print(
    "Loading model..."
)

model = (
    MllamaForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
)

model.eval()

for p in model.parameters():

    p.requires_grad_(False)


print(
    "MODEL LOADED"
)


# ============================================================
# ONE FORWARD
# ============================================================

def run_intervention(
    inputs,
    alpha,
    beta_eff,
    method,
    sa_direction,
):

    direction_mode = (
        "MN"
        if method == "MN"
        else "SA"
    )


    modifier = DirectionModifier(
        model=model,
        alpha=alpha,
        beta_eff=beta_eff,
        direction_mode=direction_mode,
        U=U4,
        sa_direction=(
            sa_direction
            if direction_mode == "SA"
            else None
        ),
    )


    modifier.register()

    try:

        with torch.inference_mode():

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
            f"{method} hook failed"
        )


    return {
        "prediction":
            int(
                state[
                    "best_numeral"
                ]
            ),

        "beta_eff":
            float(
                beta_eff
            ),

        "shift_norm":
            float(
                modifier.last_shift_norm
            ),

        "shift_ratio":
            float(
                modifier.last_shift_ratio
            ),
    }


# ============================================================
# PROSPECTIVE SWEEP
# ============================================================

start_idx = len(
    records
)


for idx in tqdm(
    range(
        start_idx,
        EXPECTED_N,
    ),
    initial=start_idx,
    total=EXPECTED_N,
    desc="Prospective MN/SA/TSG-SA",
):

    row = actions.iloc[
        idx
    ]


    sid = str(
        row[
            "raw_sample_id"
        ]
    )

    cohort_uid = str(
        row[
            "cohort_uid"
        ]
    )

    gt = int(
        row[
            "ground_truth"
        ]
    )

    baseline = int(
        row[
            "baseline_prediction"
        ]
    )

    alpha = float(
        row[
            "selected_alpha"
        ]
    )


    sample = sample_map[
        sid
    ]


    image = (
        Image.open(
            sample[
                "image_path"
            ]
        )
        .convert(
            "RGB"
        )
    )


    inputs = prepare_inputs(
        processor,
        image,
    )

    inputs = move_inputs(
        inputs,
        model,
    )


    methods = {
        "MN": {},
        "SA": {},
        "TSG-SA": {},
    }


    # ========================================================
    # alpha = 1 : exact identity
    # ========================================================

    if math.isclose(
        alpha,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        for beta0 in BETA0_GRID:

            key = bkey(
                beta0
            )

            for method in methods:

                methods[
                    method
                ][
                    key
                ] = {
                    "prediction":
                        baseline,

                    "beta0":
                        float(
                            beta0
                        ),

                    "beta_eff":
                        0.0,

                    "beta_unclipped":
                        0.0,

                    "shift_norm":
                        0.0,

                    "shift_ratio":
                        0.0,

                    "clip_status":
                        "identity",
                }


    # ========================================================
    # alpha = 0 : frozen original semantics beta=1
    # ========================================================

    elif math.isclose(
        alpha,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        mn_result = run_intervention(
            inputs=inputs,
            alpha=alpha,
            beta_eff=1.0,
            method="MN",
            sa_direction=SA_DIR[idx],
        )

        sa_result = run_intervention(
            inputs=inputs,
            alpha=alpha,
            beta_eff=1.0,
            method="SA",
            sa_direction=SA_DIR[idx],
        )


        for beta0 in BETA0_GRID:

            key = bkey(
                beta0
            )


            methods[
                "MN"
            ][
                key
            ] = {
                **mn_result,
                "beta0":
                    float(beta0),
                "beta_unclipped":
                    1.0,
                "clip_status":
                    "alpha0_fixed",
            }


            methods[
                "SA"
            ][
                key
            ] = {
                **sa_result,
                "beta0":
                    float(beta0),
                "beta_unclipped":
                    1.0,
                "clip_status":
                    "alpha0_fixed",
            }


            # TSG-SA has same direction and same beta=1
            # under the frozen alpha=0 policy.
            methods[
                "TSG-SA"
            ][
                key
            ] = {
                **sa_result,
                "beta0":
                    float(beta0),
                "beta_unclipped":
                    1.0,
                "clip_status":
                    "alpha0_fixed",
            }


    # ========================================================
    # alpha > 1 : actual prospective comparison
    # ========================================================

    else:

        if alpha <= 1.0:

            raise RuntimeError(
                f"Unexpected alpha={alpha}"
            )


        sensitivity = float(
            SENSITIVITY[
                idx
            ]
        )


        for beta0 in BETA0_GRID:

            key = bkey(
                beta0
            )


            # MN
            mn = run_intervention(
                inputs=inputs,
                alpha=alpha,
                beta_eff=beta0,
                method="MN",
                sa_direction=SA_DIR[idx],
            )

            methods[
                "MN"
            ][
                key
            ] = {
                **mn,
                "beta0":
                    float(beta0),
                "beta_unclipped":
                    float(beta0),
                "clip_status":
                    "none",
            }


            # SA
            sa = run_intervention(
                inputs=inputs,
                alpha=alpha,
                beta_eff=beta0,
                method="SA",
                sa_direction=SA_DIR[idx],
            )

            methods[
                "SA"
            ][
                key
            ] = {
                **sa,
                "beta0":
                    float(beta0),
                "beta_unclipped":
                    float(beta0),
                "clip_status":
                    "none",
            }


            # TSG-SA
            beta_unclipped = (
                float(beta0)
                *
                (
                    S_REF
                    /
                    sensitivity
                )
                ** TAU
            )


            beta_eff = float(
                np.clip(
                    beta_unclipped,
                    BETA_MIN,
                    BETA_MAX,
                )
            )


            if (
                beta_unclipped
                <
                BETA_MIN
            ):

                clip_status = "low"

            elif (
                beta_unclipped
                >
                BETA_MAX
            ):

                clip_status = "high"

            else:

                clip_status = "none"


            tsg = run_intervention(
                inputs=inputs,
                alpha=alpha,
                beta_eff=beta_eff,
                method="TSG-SA",
                sa_direction=SA_DIR[idx],
            )


            methods[
                "TSG-SA"
            ][
                key
            ] = {
                **tsg,
                "beta0":
                    float(beta0),
                "beta_unclipped":
                    float(
                        beta_unclipped
                    ),
                "clip_status":
                    clip_status,
            }


    rec = {
        "raw_sample_id":
            sid,

        "cohort_uid":
            cohort_uid,

        "ground_truth":
            gt,

        "baseline_prediction":
            baseline,

        "selected_alpha":
            alpha,

        "projected_gradient_norm":
            float(
                PG_NORM[idx]
            ),

        "h_norm":
            float(
                H_NORM[idx]
            ),

        "sa_sensitivity":
            float(
                SENSITIVITY[idx]
            ),

        "methods":
            methods,
    }


    append_jsonl(
        PARTIAL,
        rec,
    )

    records.append(
        rec
    )


# ============================================================
# FLATTEN
# ============================================================

if len(records) != EXPECTED_N:

    raise RuntimeError(
        f"Final N mismatch: "
        f"{len(records)}"
    )


flat_rows = []


for rec in records:

    for method in [
        "MN",
        "SA",
        "TSG-SA",
    ]:

        for beta0 in BETA0_GRID:

            r = (
                rec[
                    "methods"
                ][
                    method
                ][
                    bkey(beta0)
                ]
            )


            flat_rows.append(
                {
                    "raw_sample_id":
                        rec[
                            "raw_sample_id"
                        ],

                    "cohort_uid":
                        rec[
                            "cohort_uid"
                        ],

                    "ground_truth":
                        rec[
                            "ground_truth"
                        ],

                    "baseline_prediction":
                        rec[
                            "baseline_prediction"
                        ],

                    "selected_alpha":
                        rec[
                            "selected_alpha"
                        ],

                    "projected_gradient_norm":
                        rec[
                            "projected_gradient_norm"
                        ],

                    "h_norm":
                        rec[
                            "h_norm"
                        ],

                    "sa_sensitivity":
                        rec[
                            "sa_sensitivity"
                        ],

                    "method":
                        method,

                    "beta0":
                        float(
                            beta0
                        ),

                    "beta_eff":
                        r[
                            "beta_eff"
                        ],

                    "beta_unclipped":
                        r[
                            "beta_unclipped"
                        ],

                    "clip_status":
                        r[
                            "clip_status"
                        ],

                    "prediction":
                        r[
                            "prediction"
                        ],

                    "shift_norm":
                        r[
                            "shift_norm"
                        ],

                    "shift_ratio":
                        r[
                            "shift_ratio"
                        ],
                }
            )


flat = pd.DataFrame(
    flat_rows
)


flat.to_csv(
    RESULTS,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for method in [
    "MN",
    "SA",
    "TSG-SA",
]:

    for beta0 in BETA0_GRID:

        sub = flat[
            (
                flat[
                    "method"
                ]
                == method
            )
            &
            np.isclose(
                flat[
                    "beta0"
                ],
                beta0,
            )
        ].copy()


        gt = (
            sub[
                "ground_truth"
            ]
            .astype(int)
            .to_numpy()
        )


        base = (
            sub[
                "baseline_prediction"
            ]
            .astype(int)
            .to_numpy()
        )


        pred = (
            sub[
                "prediction"
            ]
            .astype(int)
            .to_numpy()
        )


        alpha = (
            sub[
                "selected_alpha"
            ]
            .astype(float)
            .to_numpy()
        )


        bc = (
            base == gt
        )

        pc = (
            pred == gt
        )


        repairs = int(
            (
                (~bc)
                &
                pc
            ).sum()
        )


        breaks = int(
            (
                bc
                &
                (~pc)
            ).sum()
        )


        upward = (
            alpha > 1.0
        )


        upward_sub = sub.loc[
            upward
        ]


        if method == "TSG-SA":

            clip_low = int(
                (
                    upward_sub[
                        "clip_status"
                    ]
                    == "low"
                ).sum()
            )

            clip_high = int(
                (
                    upward_sub[
                        "clip_status"
                    ]
                    == "high"
                ).sum()
            )

        else:

            clip_low = 0
            clip_high = 0


        summary_rows.append(
            {
                "method":
                    method,

                "beta0":
                    float(
                        beta0
                    ),

                "full_correct":
                    int(
                        pc.sum()
                    ),

                "full_accuracy":
                    float(
                        pc.mean()
                    ),

                "repairs":
                    repairs,

                "breaks":
                    breaks,

                "net_repairs":
                    repairs
                    -
                    breaks,

                "upward_n":
                    int(
                        upward.sum()
                    ),

                "upward_correct":
                    int(
                        pc[
                            upward
                        ].sum()
                    ),

                "upward_accuracy":
                    float(
                        pc[
                            upward
                        ].mean()
                    ),

                "mean_upward_beta_eff":
                    float(
                        upward_sub[
                            "beta_eff"
                        ].mean()
                    ),

                "median_upward_beta_eff":
                    float(
                        upward_sub[
                            "beta_eff"
                        ].median()
                    ),

                "clip_low_n":
                    clip_low,

                "clip_high_n":
                    clip_high,
            }
        )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY,
    index=False,
)


# ============================================================
# METHOD SELECTION
# ============================================================

selected = {}


for method in [
    "MN",
    "SA",
    "TSG-SA",
]:

    m = (
        summary[
            summary[
                "method"
            ]
            == method
        ]
        .sort_values(
            [
                "full_correct",
                "breaks",
                "beta0",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .iloc[0]
    )


    selected[
        method
    ] = {
        "beta0":
            float(
                m[
                    "beta0"
                ]
            ),

        "full_correct":
            int(
                m[
                    "full_correct"
                ]
            ),

        "full_accuracy":
            float(
                m[
                    "full_accuracy"
                ]
            ),

        "repairs":
            int(
                m[
                    "repairs"
                ]
            ),

        "breaks":
            int(
                m[
                    "breaks"
                ]
            ),

        "net_repairs":
            int(
                m[
                    "net_repairs"
                ]
            ),

        "upward_correct":
            int(
                m[
                    "upward_correct"
                ]
            ),

        "upward_n":
            int(
                m[
                    "upward_n"
                ]
            ),

        "upward_accuracy":
            float(
                m[
                    "upward_accuracy"
                ]
            ),

        "mean_upward_beta_eff":
            float(
                m[
                    "mean_upward_beta_eff"
                ]
            ),

        "median_upward_beta_eff":
            float(
                m[
                    "median_upward_beta_eff"
                ]
            ),

        "clip_low_n":
            int(
                m[
                    "clip_low_n"
                ]
            ),

        "clip_high_n":
            int(
                m[
                    "clip_high_n"
                ]
            ),
    }


selection = {
    "experiment":
        "tsg_prospective_mn_sa_tsg_sweep_v1",

    "n":
        500,

    "execution_freeze_sha256":
        sha256(
            EXECUTION_FREEZE
        ),

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),

    "action_manifest_sha256":
        sha256(
            ACTION
        ),

    "selected":
        selected,
}


SELECTION.write_text(
    json.dumps(
        selection,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 108)
print(
    "PROSPECTIVE MN vs SA vs TSG-SA SWEEP COMPLETE"
)
print("=" * 108)

print()

print(
    summary[
        [
            "method",
            "beta0",
            "full_correct",
            "full_accuracy",
            "repairs",
            "breaks",
            "net_repairs",
            "upward_correct",
            "upward_n",
            "upward_accuracy",
            "mean_upward_beta_eff",
            "median_upward_beta_eff",
            "clip_low_n",
            "clip_high_n",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 108)
print("SELECTED")
print("=" * 108)


for method in [
    "MN",
    "SA",
    "TSG-SA",
]:

    x = selected[
        method
    ]

    print(
        f"{method}: "
        f"beta0={x['beta0']:.2f} "
        f"full={x['full_correct']}/500 "
        f"({x['full_accuracy']:.6f}) "
        f"repairs={x['repairs']} "
        f"breaks={x['breaks']} "
        f"upward="
        f"{x['upward_correct']}/"
        f"{x['upward_n']} "
        f"({x['upward_accuracy']:.6f}) "
        f"mean_beta_eff="
        f"{x['mean_upward_beta_eff']:.4f}"
    )


print()
print(
    "Execution freeze SHA256:",
    sha256(
        EXECUTION_FREEZE
    ),
)

print(
    "Results:",
    RESULTS,
)

print(
    "Summary:",
    SUMMARY,
)

print(
    "Selection:",
    SELECTION,
)

print("=" * 108)
