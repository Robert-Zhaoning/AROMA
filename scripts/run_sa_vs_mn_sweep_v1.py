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
# FROZEN CONFIG
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

PROTOCOL = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_vs_mn_protocol_v1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "39e61d47edbe9625e93e5f18b2c6f5a9"
    "fd9b25e1b19a731b50908d8eb2c61869"
)

ACTION_MANIFEST = Path(
    "outputs/proc_count_sa_dev_v1/"
    "action_manifest_v1/"
    "sa_vs_mn_action_manifest_v1.csv"
)

EXPECTED_ACTION_MANIFEST_SHA256 = (
    "909577217bc38825eab33f03f1ce5633"
    "dab61815bbd83468a30dc4bac6e52021"
)

DATA_META = Path(
    "data/proc_count_sa_dev_v1/metadata.jsonl"
)

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

GRADIENT_ROOT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_gradient_collection_v1"
)

OUT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_vs_mn_sweep_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = OUT / "partial_results.jsonl"

FINAL_RESULTS = (
    OUT / "sa_vs_mn_results.csv"
)

SWEEP_SUMMARY = (
    OUT / "sa_vs_mn_sweep_summary.csv"
)

SELECTION_JSON = (
    OUT / "sa_vs_mn_selection.json"
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
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

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


def append_jsonl(path, obj):
    with Path(path).open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(obj)
            + "\n"
        )
        f.flush()


def beta_key(beta):
    return f"{float(beta):.2f}"


# ============================================================
# FROZEN ARTIFACT CHECKS
# ============================================================

if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError(
        "Protocol SHA256 mismatch."
    )

if (
    sha256(ACTION_MANIFEST)
    != EXPECTED_ACTION_MANIFEST_SHA256
):
    raise RuntimeError(
        "Action manifest SHA256 mismatch."
    )

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

BETAS = [
    float(x)
    for x in protocol[
        "beta_grid"
    ]
]

print(
    "Frozen beta grid:",
    BETAS,
)


# ============================================================
# LOAD DATASET
# ============================================================

samples = load_jsonl(
    DATA_META
)

if len(samples) != EXPECTED_N:
    raise RuntimeError(
        f"Expected 500 samples; got {len(samples)}"
    )

samples = sorted(
    samples,
    key=lambda x: str(x["sample_id"]),
)

sample_map = {
    str(x["sample_id"]): x
    for x in samples
}


# ============================================================
# ACTION MANIFEST
# ============================================================

actions = pd.read_csv(
    ACTION_MANIFEST
)

actions[
    "sample_id"
] = (
    actions[
        "sample_id"
    ].astype(str)
)

if len(actions) != EXPECTED_N:
    raise RuntimeError(
        "Action manifest N mismatch."
    )

for a in actions[
    "selected_alpha"
].astype(float):

    if a not in EXPECTED_ACTIONS:
        raise RuntimeError(
            f"Unexpected action {a}"
        )


# ============================================================
# ARCHIVED GRADIENTS
# ============================================================

G = np.load(
    GRADIENT_ROOT / "gradient_matrix.npy"
).astype(np.float64)

G_IDS = (
    np.load(
        GRADIENT_ROOT / "sample_ids.npy",
        allow_pickle=True,
    )
    .astype(str)
)

if G.shape != (500, 128):
    raise RuntimeError(
        f"Bad gradient shape {G.shape}"
    )

if not np.array_equal(
    actions[
        "sample_id"
    ].to_numpy(),
    G_IDS,
):
    raise RuntimeError(
        "Action/gradient sample ordering mismatch."
    )


# ============================================================
# FROZEN U4 + SA DIRECTIONS
# ============================================================

U4 = np.load(
    U4_PATH
).astype(np.float64)

if U4.shape != (128, 4):
    raise RuntimeError(
        f"Bad U4 shape: {U4.shape}"
    )

P = U4 @ U4.T

PG = G @ P

PG_NORM = np.linalg.norm(
    PG,
    axis=1,
)

if float(
    PG_NORM.min()
) <= 1e-12:

    raise RuntimeError(
        "Projected gradient norm too small."
    )

SA_DIRECTIONS = (
    PG
    /
    PG_NORM[:, None]
)

print(
    "Projected-gradient norm "
    "min/median/max:",
    float(PG_NORM.min()),
    float(np.median(PG_NORM)),
    float(PG_NORM.max()),
)


# ============================================================
# MODIFIER
# ============================================================

class DirectionModifier:

    def __init__(
        self,
        model,
        alpha,
        beta,
        method,
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

        self.beta = float(
            beta
        )

        self.method = str(
            method
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

            d_norm = float(
                np.linalg.norm(d)
            )

            if not np.isclose(
                d_norm,
                1.0,
                atol=1e-5,
            ):
                raise RuntimeError(
                    f"SA direction not unit norm: "
                    f"{d_norm}"
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
        self.last_direction_source_norm = None
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
            H
            .float()
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


        # ====================================================
        # MN DIRECTION
        # ====================================================

        if self.method == "MN":

            U = self.U.to(
                device=H.device
            )

            projected = (
                (h @ U)
                @ U.T
            )

            source_norm = (
                torch.linalg
                .vector_norm(
                    projected,
                    dim=-1,
                    keepdim=True,
                )
            )

            if float(
                source_norm.min().item()
            ) <= 1e-12:

                raise RuntimeError(
                    "MN projected activation "
                    "norm too small."
                )

            direction = (
                projected
                /
                source_norm
            )


        # ====================================================
        # SA DIRECTION
        # ====================================================

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

            source_norm = torch.ones(
                (
                    h.shape[0],
                    1,
                ),
                dtype=torch.float32,
                device=H.device,
            )


        else:
            raise RuntimeError(
                f"Unknown method: "
                f"{self.method}"
            )


        # ====================================================
        # SAME REALIZED WHOLE-RELATIVE BUDGET
        #
        # ||delta|| =
        # beta * |alpha-1| * ||h||
        # ====================================================

        delta = (
            self.beta
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
            h_norm[0, 0].item()
        )

        self.last_direction_source_norm = float(
            source_norm[0, 0].item()
        )

        self.last_shift_norm = float(
            shift_norm[0, 0].item()
        )

        self.last_shift_ratio = float(
            shift_ratio[0, 0].item()
        )

        if len(inputs) == 1:
            return (x_new,)

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
        str(r["sample_id"])
        for r in records
    ]

    expected_prefix = (
        actions[
            "sample_id"
        ]
        .iloc[
            :len(records)
        ]
        .tolist()
    )

    if partial_ids != expected_prefix:
        raise RuntimeError(
            "Partial results are not "
            "an exact dataset prefix."
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
# FINITE-STEP SWEEP
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
    desc="SA vs MN sweep",
):

    row = actions.iloc[idx]

    sid = str(
        row["sample_id"]
    )

    gt = int(
        row["ground_truth"]
    )

    baseline_pred = int(
        row["baseline_prediction"]
    )

    alpha = float(
        row["selected_alpha"]
    )

    sample = sample_map[
        sid
    ]

    image = (
        Image.open(
            sample["image_path"]
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

    method_results = {
        "MN": {},
        "SA": {},
    }


    # ========================================================
    # alpha=1 : exact identity for every beta
    # ========================================================

    if math.isclose(
        alpha,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        for method in [
            "MN",
            "SA",
        ]:

            for beta in BETAS:

                method_results[
                    method
                ][
                    beta_key(beta)
                ] = {
                    "prediction":
                        baseline_pred,

                    "effective_beta":
                        0.0,

                    "shift_norm":
                        0.0,

                    "shift_ratio":
                        0.0,
                }


    # ========================================================
    # alpha=0 : preserve original action semantics.
    # beta fixed to 1 for every grid entry.
    # ========================================================

    elif math.isclose(
        alpha,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        for method in [
            "MN",
            "SA",
        ]:

            modifier = DirectionModifier(
                model=model,
                alpha=alpha,
                beta=1.0,
                method=method,
                U=U4,
                sa_direction=(
                    SA_DIRECTIONS[idx]
                    if method == "SA"
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
                    f"{sid}: {method} "
                    "alpha=0 hook failed."
                )

            pred = int(
                state[
                    "best_numeral"
                ]
            )

            common_result = {
                "prediction":
                    pred,

                "effective_beta":
                    1.0,

                "shift_norm":
                    modifier.last_shift_norm,

                "shift_ratio":
                    modifier.last_shift_ratio,
            }

            for beta in BETAS:

                method_results[
                    method
                ][
                    beta_key(beta)
                ] = dict(
                    common_result
                )


    # ========================================================
    # alpha>1 : actual beta sweep
    # ========================================================

    else:

        if alpha <= 1.0:
            raise RuntimeError(
                f"{sid}: unexpected alpha "
                f"{alpha}"
            )

        for beta in BETAS:

            for method in [
                "MN",
                "SA",
            ]:

                modifier = DirectionModifier(
                    model=model,
                    alpha=alpha,
                    beta=beta,
                    method=method,
                    U=U4,
                    sa_direction=(
                        SA_DIRECTIONS[idx]
                        if method == "SA"
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
                        f"{sid}: "
                        f"{method} beta={beta} "
                        "hook failed."
                    )

                pred = int(
                    state[
                        "best_numeral"
                    ]
                )

                method_results[
                    method
                ][
                    beta_key(beta)
                ] = {
                    "prediction":
                        pred,

                    "effective_beta":
                        float(beta),

                    "shift_norm":
                        modifier.last_shift_norm,

                    "shift_ratio":
                        modifier.last_shift_ratio,
                }


    rec = {
        "sample_id":
            sid,

        "ground_truth":
            gt,

        "baseline_prediction":
            baseline_pred,

        "selected_alpha":
            alpha,

        "projected_gradient_norm":
            float(
                PG_NORM[idx]
            ),

        "methods":
            method_results,
    }

    append_jsonl(
        PARTIAL,
        rec,
    )

    records.append(
        rec
    )


# ============================================================
# FLATTEN RESULTS
# ============================================================

if len(records) != 500:
    raise RuntimeError(
        f"Final N mismatch: "
        f"{len(records)}"
    )

flat_rows = []

for rec in records:

    for method in [
        "MN",
        "SA",
    ]:

        for beta in BETAS:

            r = (
                rec[
                    "methods"
                ][
                    method
                ][
                    beta_key(beta)
                ]
            )

            flat_rows.append(
                {
                    "sample_id":
                        rec["sample_id"],

                    "ground_truth":
                        rec["ground_truth"],

                    "baseline_prediction":
                        rec[
                            "baseline_prediction"
                        ],

                    "selected_alpha":
                        rec["selected_alpha"],

                    "projected_gradient_norm":
                        rec[
                            "projected_gradient_norm"
                        ],

                    "method":
                        method,

                    "beta":
                        float(beta),

                    "effective_beta":
                        r["effective_beta"],

                    "prediction":
                        r["prediction"],

                    "shift_norm":
                        r["shift_norm"],

                    "shift_ratio":
                        r["shift_ratio"],
                }
            )

flat = pd.DataFrame(
    flat_rows
)

flat.to_csv(
    FINAL_RESULTS,
    index=False,
)


# ============================================================
# SWEEP SUMMARY
# ============================================================

summary_rows = []

for method in [
    "MN",
    "SA",
]:

    for beta in BETAS:

        sub = flat[
            (
                flat["method"]
                == method
            )
            &
            np.isclose(
                flat["beta"],
                beta,
            )
        ].copy()

        gt = (
            sub[
                "ground_truth"
            ].astype(int)
            .to_numpy()
        )

        baseline = (
            sub[
                "baseline_prediction"
            ].astype(int)
            .to_numpy()
        )

        pred = (
            sub[
                "prediction"
            ].astype(int)
            .to_numpy()
        )

        alpha = (
            sub[
                "selected_alpha"
            ].astype(float)
            .to_numpy()
        )

        base_correct = (
            baseline == gt
        )

        method_correct = (
            pred == gt
        )

        repairs = int(
            (
                (~base_correct)
                &
                method_correct
            ).sum()
        )

        breaks = int(
            (
                base_correct
                &
                (~method_correct)
            ).sum()
        )

        upward = (
            alpha > 1.0
        )

        active = (
            ~np.isclose(
                alpha,
                1.0,
            )
        )

        summary_rows.append(
            {
                "method":
                    method,

                "beta":
                    float(beta),

                "full_correct":
                    int(
                        method_correct.sum()
                    ),

                "full_accuracy":
                    float(
                        method_correct.mean()
                    ),

                "repairs":
                    repairs,

                "breaks":
                    breaks,

                "net_repairs":
                    repairs - breaks,

                "active_n":
                    int(
                        active.sum()
                    ),

                "active_correct":
                    int(
                        method_correct[
                            active
                        ].sum()
                    ),

                "active_accuracy":
                    float(
                        method_correct[
                            active
                        ].mean()
                    ),

                "upward_n":
                    int(
                        upward.sum()
                    ),

                "upward_correct":
                    int(
                        method_correct[
                            upward
                        ].sum()
                    ),

                "upward_accuracy":
                    float(
                        method_correct[
                            upward
                        ].mean()
                    ),

                "mean_upward_shift_ratio":
                    float(
                        sub.loc[
                            upward,
                            "shift_ratio",
                        ].mean()
                    ),
            }
        )

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SWEEP_SUMMARY,
    index=False,
)


# ============================================================
# FROZEN SELECTION RULE
#
# 1. highest exact-count accuracy
# 2. fewer breaks
# 3. smaller beta
# ============================================================

selected = {}

for method in [
    "MN",
    "SA",
]:

    m = (
        summary[
            summary["method"]
            == method
        ]
        .sort_values(
            [
                "full_correct",
                "breaks",
                "beta",
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
        "beta":
            float(m["beta"]),

        "full_correct":
            int(m["full_correct"]),

        "full_accuracy":
            float(m["full_accuracy"]),

        "repairs":
            int(m["repairs"]),

        "breaks":
            int(m["breaks"]),

        "net_repairs":
            int(m["net_repairs"]),

        "upward_correct":
            int(m["upward_correct"]),

        "upward_n":
            int(m["upward_n"]),

        "upward_accuracy":
            float(m["upward_accuracy"]),
    }

selection = {
    "experiment":
        "sa_vs_mn_sweep_v1",

    "n":
        500,

    "protocol_sha256":
        sha256(PROTOCOL),

    "action_manifest_sha256":
        sha256(ACTION_MANIFEST),

    "beta_grid":
        BETAS,

    "selection_rule":
        "highest full exact-count accuracy; "
        "ties -> fewer breaks; "
        "ties -> smaller beta",

    "selected":
        selected,
}

SELECTION_JSON.write_text(
    json.dumps(
        selection,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("SA vs MN FINITE-STEP SWEEP COMPLETE")
print("=" * 100)

print()
print(
    summary[
        [
            "method",
            "beta",
            "full_correct",
            "full_accuracy",
            "repairs",
            "breaks",
            "net_repairs",
            "upward_correct",
            "upward_n",
            "upward_accuracy",
            "mean_upward_shift_ratio",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 100)
print("SELECTED")
print("=" * 100)

for method in [
    "MN",
    "SA",
]:

    x = selected[
        method
    ]

    print(
        f"{method}: "
        f"beta={x['beta']:.2f} "
        f"full={x['full_correct']}/500 "
        f"({x['full_accuracy']:.6f}) "
        f"repairs={x['repairs']} "
        f"breaks={x['breaks']} "
        f"upward="
        f"{x['upward_correct']}/"
        f"{x['upward_n']} "
        f"({x['upward_accuracy']:.6f})"
    )

print()
print(
    "Results:",
    FINAL_RESULTS,
)

print(
    "Summary:",
    SWEEP_SUMMARY,
)

print(
    "Selection:",
    SELECTION_JSON,
)

print("=" * 100)
