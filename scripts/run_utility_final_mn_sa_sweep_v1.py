import hashlib
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
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)


# ============================================================
# FROZEN CONFIG
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

META = Path(
    "data/proc_count_utility_final_v1/"
    "metadata.jsonl"
)

EXPECTED_META_SHA = (
    "700b20fe99944c9d25cb4b80873156b796b4bba90f4af"
    "743427a4195bb040231"
)

ACTION = Path(
    "outputs/proc_count_utility_final_v1/"
    "action_manifest_v1/"
    "utility_final_action_manifest_v1.csv"
)

EXPECTED_ACTION_SHA = (
    "3123d61cfca130376a2b911707ea20927872b1a270ac1"
    "fbd5a852d3e82fc3040"
)

ACTION_FREEZE = Path(
    "outputs/proc_count_utility_final_v1/"
    "action_manifest_v1/freeze/"
    "utility_final_action_manifest_freeze_v1.json"
)

EXPECTED_ACTION_FREEZE_SHA = (
    "4fb05b5d1c91ab3613432ee580126dcd504dd59abf875"
    "eb5a56ec79e4745edef"
)

GRAD_ROOT = Path(
    "outputs/proc_count_utility_final_v1/"
    "gradient_collection_v1"
)

GRAD_FREEZE = (
    GRAD_ROOT
    / "freeze"
    / "utility_final_gradient_freeze_v1.json"
)

EXPECTED_GRAD_FREEZE_SHA = (
    "086422c7a0b24023ee7e3cc4bc2774654cfb539d4ebd9"
    "a7b2ba844f853416ab9"
)

EXPECTED_G_SHA = (
    "ac771aadb34ef1682065435e84a88dd7f7fbc108e3fa"
    "44879a5ae2d536699b3b"
)

EXPECTED_H_SHA = (
    "07767f42678a330d8298acd2f63d9438b8bc8c30af61"
    "a35efc61cb8a019994c1"
)

FEATURE_FREEZE = Path(
    "outputs/proc_count_utility_final_v1/"
    "utility_features_v1/freeze/"
    "utility_final_features_freeze_v1.json"
)

EXPECTED_FEATURE_FREEZE_SHA = (
    "f672bcd8f6549487c393df393831915ba4a0acbbf50d"
    "b76b928a092f14a353c2"
)

PROTOCOL = Path(
    "outputs/aroma2/"
    "utility_router_final_confirmation_v1/"
    "final_confirmation_protocol.json"
)

EXPECTED_PROTOCOL_SHA = (
    "7a37d80f53e86944631571799d55b8eac8d4873ce74cf"
    "1f9a84d11c118654922"
)

UTILITY_ROUTER = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_final_v1/"
    "aroma_utility_router_v1.joblib"
)

EXPECTED_ROUTER_SHA = (
    "4192129c525822fed7ab9d013dc3d564981f48faf81f28"
    "b3011a4be1a2168563"
)

U4_PATH = Path(
    "outputs/aroma2/"
    "csa_gate_a_v3/"
    "U4_primary.npy"
)

FINAL_FEATURE_META = Path(
    "outputs/proc_count_utility_final_v1/"
    "utility_features_v1/"
    "metadata.json"
)

OUT = Path(
    "outputs/proc_count_utility_final_v1/"
    "mn_sa_sweep_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = (
    OUT
    / "partial_results.jsonl"
)

RESULTS = (
    OUT
    / "mn_sa_results.csv"
)

EXECUTION_FREEZE = (
    OUT
    / "execution_freeze.json"
)

METADATA_OUT = (
    OUT
    / "metadata.json"
)

EXPECTED_N = 294

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


def bkey(beta):

    return f"{float(beta):.2f}"


# ============================================================
# HARD ARTIFACT AUDIT
# ============================================================

checks = [
    (
        META,
        EXPECTED_META_SHA,
        "dataset metadata",
    ),
    (
        ACTION,
        EXPECTED_ACTION_SHA,
        "action manifest",
    ),
    (
        ACTION_FREEZE,
        EXPECTED_ACTION_FREEZE_SHA,
        "action freeze",
    ),
    (
        GRAD_FREEZE,
        EXPECTED_GRAD_FREEZE_SHA,
        "gradient freeze",
    ),
    (
        GRAD_ROOT / "gradient_matrix.npy",
        EXPECTED_G_SHA,
        "gradient matrix",
    ),
    (
        GRAD_ROOT / "pooled_h_matrix.npy",
        EXPECTED_H_SHA,
        "pooled-h matrix",
    ),
    (
        FEATURE_FREEZE,
        EXPECTED_FEATURE_FREEZE_SHA,
        "feature freeze",
    ),
    (
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA,
        "final protocol",
    ),
    (
        UTILITY_ROUTER,
        EXPECTED_ROUTER_SHA,
        "utility router",
    ),
]

for path, expected, name in checks:

    actual = sha256(path)

    if actual != expected:

        raise RuntimeError(
            f"{name} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


# ============================================================
# PROTOCOL-LOCKED CANDIDATE SET
# ============================================================

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

candidate_spec = (
    protocol[
        "utility_router"
    ][
        "candidate_set"
    ]
)

methods = list(
    candidate_spec[
        "methods"
    ]
)

BETA_GRID = [
    float(x)
    for x in candidate_spec[
        "beta_grid"
    ]
]


if methods != [
    "MN",
    "SA",
]:

    raise RuntimeError(
        f"Unexpected frozen methods: "
        f"{methods}"
    )


expected_grid = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]

if not np.allclose(
    BETA_GRID,
    expected_grid,
    atol=0.0,
    rtol=0.0,
):

    raise RuntimeError(
        f"Unexpected frozen beta grid: "
        f"{BETA_GRID}"
    )


# U4 must be exactly the one already used
# in final feature construction.

feature_meta = json.loads(
    FINAL_FEATURE_META.read_text(
        encoding="utf-8"
    )
)

if (
    sha256(U4_PATH)
    !=
    feature_meta[
        "U4_sha256"
    ]
):

    raise RuntimeError(
        "U4 SHA mismatch with frozen "
        "final feature archive."
    )


print(
    "PASS: all final frozen artifacts verified"
)


# ============================================================
# FREEZE EXECUTION BEFORE INTERVENTION
# ============================================================

execution = {
    "experiment":
        "utility_final_mn_sa_sweep_v1",

    "status":
        "frozen before final candidate outcomes",

    "population":
        "final selected_alpha > 1",

    "n":
        EXPECTED_N,

    "methods":
        [
            "MN",
            "SA",
        ],

    "beta_grid":
        BETA_GRID,

    "candidates_per_sample":
        14,

    "total_intervention_evaluations":
        EXPECTED_N
        *
        14,

    "MN_direction":
        "normalize(P_U h)",

    "SA_direction":
        "normalize(P_U g)",

    "shift_rule":
        "delta = beta * "
        "(alpha - 1) * "
        "||h|| * direction",

    "candidate_selection":
        "NONE — outcomes only",

    "no_tuning":
        True,

    "protocol_sha256":
        sha256(PROTOCOL),

    "action_freeze_sha256":
        sha256(ACTION_FREEZE),

    "gradient_freeze_sha256":
        sha256(GRAD_FREEZE),

    "feature_freeze_sha256":
        sha256(FEATURE_FREEZE),

    "utility_router_sha256":
        sha256(UTILITY_ROUTER),

    "U4_sha256":
        sha256(U4_PATH),
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
    "Execution freeze SHA256:",
    sha256(
        EXECUTION_FREEZE
    ),
)


# ============================================================
# FINAL UPWARD POPULATION
# ============================================================

samples = load_jsonl(
    META
)

if len(
    samples
) != 1000:

    raise RuntimeError(
        "Expected full final N=1000."
    )


sample_map = {
    str(
        r[
            "sample_id"
        ]
    ):
        r
    for r in samples
}


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


actions = actions[
    actions[
        "selected_alpha"
    ].astype(float)
    > 1.0
].copy()


actions = (
    actions
    .sort_values(
        "raw_sample_id"
    )
    .reset_index(
        drop=True
    )
)


if len(
    actions
) != EXPECTED_N:

    raise RuntimeError(
        f"Expected upward N={EXPECTED_N}; "
        f"got {len(actions)}"
    )


raw_ids = (
    actions[
        "raw_sample_id"
    ]
    .astype(str)
    .to_numpy()
)

cohort_uids = (
    actions[
        "cohort_uid"
    ]
    .astype(str)
    .to_numpy()
)


if not np.all(
    actions[
        "selected_alpha"
    ].astype(float)
    > 1.0
):

    raise RuntimeError(
        "Non-upward sample in final sweep."
    )


# ============================================================
# G / H / U4
# ============================================================

G = np.load(
    GRAD_ROOT
    / "gradient_matrix.npy"
).astype(
    np.float64
)

H_ARCH = np.load(
    GRAD_ROOT
    / "pooled_h_matrix.npy"
).astype(
    np.float64
)

G_IDS = (
    np.load(
        GRAD_ROOT
        / "sample_ids.npy",
        allow_pickle=True,
    )
    .astype(str)
)

G_UIDS = (
    np.load(
        GRAD_ROOT
        / "cohort_uids.npy",
        allow_pickle=True,
    )
    .astype(str)
)


if G.shape != (
    EXPECTED_N,
    128,
):

    raise RuntimeError(
        f"Bad G shape: "
        f"{G.shape}"
    )


if H_ARCH.shape != (
    EXPECTED_N,
    128,
):

    raise RuntimeError(
        f"Bad H shape: "
        f"{H_ARCH.shape}"
    )


if not np.array_equal(
    G_IDS,
    raw_ids,
):

    raise RuntimeError(
        "Gradient sample ID mismatch."
    )


if not np.array_equal(
    G_UIDS,
    cohort_uids,
):

    raise RuntimeError(
        "Gradient cohort UID mismatch."
    )


U4 = np.load(
    U4_PATH
).astype(
    np.float64
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
            U4.T
            @
            U4
            -
            np.eye(4)
        )
    )
)

if orth_error > 1e-5:

    raise RuntimeError(
        f"U4 orthogonality error: "
        f"{orth_error}"
    )


P = (
    U4
    @
    U4.T
)

PG = (
    G
    @
    P
)

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
    PG_NORM[
        :,
        None
    ]
)


SENSITIVITY = (
    H_NORM
    *
    PG_NORM
)


print(
    "PASS: final upward population "
    "and gradient alignment N=294"
)


# ============================================================
# INTERVENTION MODIFIER
# ============================================================

class DirectionModifier:

    def __init__(
        self,
        model,
        alpha,
        beta,
        method,
        U,
        sa_direction,
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

        self.alpha = float(
            alpha
        )

        self.beta = float(
            beta
        )

        self.method = str(
            method
        )

        self.U = torch.tensor(
            np.asarray(
                U,
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )

        d = np.asarray(
            sa_direction,
            dtype=np.float32,
        )

        dn = float(
            np.linalg.norm(
                d
            )
        )

        if not np.isclose(
            dn,
            1.0,
            atol=1e-5,
        ):

            raise RuntimeError(
                f"Bad SA direction norm: "
                f"{dn}"
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


        x = inputs[
            0
        ]

        H = x[
            ...,
            START:END
        ]


        h = (
            H.float()
            .mean(
                dim=1
            )
        )


        h_norm = (
            torch.linalg
            .vector_norm(
                h,
                dim=-1,
                keepdim=True,
            )
        )


        if self.method == "MN":

            U = self.U.to(
                device=H.device
            )

            projected = (
                (h @ U)
                @
                U.T
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
                    "MN projection too small."
                )

            direction = (
                projected
                /
                projected_norm
            )


        elif self.method == "SA":

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
                f"Unknown method: "
                f"{self.method}"
            )


        delta = (
            self.beta
            *
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
                -
                1.0
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
        self
    ):

        self.calls = 0

        self.handle = (
            self.o_proj
            .register_forward_pre_hook(
                self.hook
            )
        )


    def remove(
        self
    ):

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
        str(
            r[
                "raw_sample_id"
            ]
        )
        for r in records
    ]

    expected_prefix = (
        raw_ids[
            :len(
                records
            )
        ]
        .astype(str)
        .tolist()
    )


    if partial_ids != expected_prefix:

        raise RuntimeError(
            "Partial results do not match "
            "frozen final prefix."
        )


    print(
        f"Resuming from "
        f"{len(records)}/{EXPECTED_N}"
    )


# ============================================================
# MODEL
# ============================================================

print(
    "Loading processor..."
)

processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
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

    p.requires_grad_(
        False
    )


print(
    "MODEL LOADED"
)


# ============================================================
# ONE INTERVENTION
# ============================================================

def run_intervention(
    inputs,
    alpha,
    beta,
    method,
    sa_direction,
):

    modifier = DirectionModifier(
        model=model,
        alpha=alpha,
        beta=beta,
        method=method,
        U=U4,
        sa_direction=sa_direction,
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
            f"{method} hook failed."
        )


    if not np.isclose(
        modifier.last_shift_ratio,
        beta,
        atol=2e-3,
        rtol=2e-3,
    ):

        raise RuntimeError(
            f"{method}: shift-ratio "
            f"mismatch "
            f"{modifier.last_shift_ratio} "
            f"vs beta={beta}"
        )


    return {
        "prediction":
            int(
                state[
                    "best_numeral"
                ]
            ),

        "shift_norm":
            float(
                modifier
                .last_shift_norm
            ),

        "shift_ratio":
            float(
                modifier
                .last_shift_ratio
            ),
    }


# ============================================================
# FINAL FROZEN SWEEP
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
    desc="FINAL frozen MN/SA sweep",
):

    row = actions.iloc[
        idx
    ]


    sid = str(
        row[
            "raw_sample_id"
        ]
    )

    uid = str(
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


    if alpha <= 1.0:

        raise RuntimeError(
            f"{sid}: non-upward alpha "
            f"{alpha}"
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


    candidate_results = {
        "MN": {},
        "SA": {},
    }


    for beta in BETA_GRID:

        key = bkey(
            beta
        )


        mn = run_intervention(
            inputs=inputs,
            alpha=alpha,
            beta=beta,
            method="MN",
            sa_direction=SA_DIR[
                idx
            ],
        )


        candidate_results[
            "MN"
        ][
            key
        ] = {
            **mn,
            "beta":
                float(
                    beta
                ),
        }


        sa = run_intervention(
            inputs=inputs,
            alpha=alpha,
            beta=beta,
            method="SA",
            sa_direction=SA_DIR[
                idx
            ],
        )


        candidate_results[
            "SA"
        ][
            key
        ] = {
            **sa,
            "beta":
                float(
                    beta
                ),
        }


    rec = {
        "raw_sample_id":
            sid,

        "cohort_uid":
            uid,

        "ground_truth":
            gt,

        "baseline_prediction":
            baseline,

        "selected_alpha":
            alpha,

        "projected_gradient_norm":
            float(
                PG_NORM[
                    idx
                ]
            ),

        "h_norm":
            float(
                H_NORM[
                    idx
                ]
            ),

        "sa_sensitivity":
            float(
                SENSITIVITY[
                    idx
                ]
            ),

        "methods":
            candidate_results,
    }


    append_jsonl(
        PARTIAL,
        rec,
    )

    records.append(
        rec
    )


# ============================================================
# FINALIZE LONG-FORM OUTCOMES
# ============================================================

if len(
    records
) != EXPECTED_N:

    raise RuntimeError(
        "Final sweep N mismatch."
    )


long_rows = []


for rec in records:

    for method in [
        "MN",
        "SA",
    ]:

        for beta in BETA_GRID:

            key = bkey(
                beta
            )

            out = (
                rec[
                    "methods"
                ][
                    method
                ][
                    key
                ]
            )


            long_rows.append(
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
                        int(
                            rec[
                                "ground_truth"
                            ]
                        ),

                    "baseline_prediction":
                        int(
                            rec[
                                "baseline_prediction"
                            ]
                        ),

                    "selected_alpha":
                        float(
                            rec[
                                "selected_alpha"
                            ]
                        ),

                    "method":
                        method,

                    "beta":
                        float(
                            beta
                        ),

                    "prediction":
                        int(
                            out[
                                "prediction"
                            ]
                        ),

                    "shift_norm":
                        float(
                            out[
                                "shift_norm"
                            ]
                        ),

                    "shift_ratio":
                        float(
                            out[
                                "shift_ratio"
                            ]
                        ),

                    "projected_gradient_norm":
                        float(
                            rec[
                                "projected_gradient_norm"
                            ]
                        ),

                    "h_norm":
                        float(
                            rec[
                                "h_norm"
                            ]
                        ),

                    "sa_sensitivity":
                        float(
                            rec[
                                "sa_sensitivity"
                            ]
                        ),
                }
            )


df = pd.DataFrame(
    long_rows
)


if len(
    df
) != (
    EXPECTED_N
    *
    14
):

    raise RuntimeError(
        f"Expected 4116 rows; "
        f"got {len(df)}"
    )


candidate_counts = (
    df.groupby(
        "cohort_uid"
    )
    .size()
)


if not np.all(
    candidate_counts
    .to_numpy()
    ==
    14
):

    raise RuntimeError(
        "Not exactly 14 candidates/sample."
    )


df.to_csv(
    RESULTS,
    index=False,
)


metadata = {
    "experiment":
        "utility_final_mn_sa_sweep_v1",

    "population":
        "frozen final selected_alpha > 1",

    "sample_n":
        EXPECTED_N,

    "candidate_rows":
        int(
            len(
                df
            )
        ),

    "methods": [
        "MN",
        "SA",
    ],

    "beta_grid":
        BETA_GRID,

    "result_policy":
        "candidate outcomes only; "
        "no model selection performed",

    "execution_freeze_sha256":
        sha256(
            EXECUTION_FREEZE
        ),

    "action_manifest_sha256":
        sha256(
            ACTION
        ),

    "gradient_matrix_sha256":
        sha256(
            GRAD_ROOT
            / "gradient_matrix.npy"
        ),

    "pooled_h_matrix_sha256":
        sha256(
            GRAD_ROOT
            / "pooled_h_matrix.npy"
        ),

    "U4_sha256":
        sha256(
            U4_PATH
        ),

    "feature_freeze_sha256":
        sha256(
            FEATURE_FREEZE
        ),

    "utility_router_sha256":
        sha256(
            UTILITY_ROUTER
        ),

    "results_sha256":
        sha256(
            RESULTS
        ),
}


METADATA_OUT.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# IMPORTANT:
# Do not print candidate accuracies here.
# Final outcomes are revealed only by the
# frozen final evaluation.

print()
print("=" * 96)
print(
    "UTILITY FINAL MN/SA CANDIDATE OUTCOMES COMPLETE"
)
print("=" * 96)

print(
    "Samples:",
    EXPECTED_N,
)

print(
    "Candidates/sample:",
    14,
)

print(
    "Total candidate rows:",
    len(
        df
    ),
)

print(
    "Execution freeze SHA256:",
    sha256(
        EXECUTION_FREEZE
    ),
)

print(
    "Results SHA256:",
    sha256(
        RESULTS
    ),
)

print(
    "Metadata SHA256:",
    sha256(
        METADATA_OUT
    ),
)

print("=" * 96)
