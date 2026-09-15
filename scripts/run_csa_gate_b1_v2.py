import gc
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import binomtest
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
)


MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

V3_RESULTS = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller/"
    "v3_final_results.csv"
)

IMAGE_DIR = Path(
    "data/proc_count_causal_v3/images"
)

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/U4_primary.npy"
)

FREEZE_DOC = Path(
    "docs/AROMA2_CSA_GATEB1_FREEZE_v2.md"
)

OUT_DIR = Path(
    "outputs/aroma2/csa_gate_b1_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL_CSV = OUT_DIR / "partial_results.csv"
PARTIAL_META = OUT_DIR / "partial_metadata.json"

FINAL_CSV = OUT_DIR / "gate_b1_v2_results.csv"
SUMMARY_CSV = OUT_DIR / "gate_b1_v2_summary.csv"
BY_ACTION_CSV = OUT_DIR / "gate_b1_v2_by_action.csv"
BY_CONDITION_CSV = OUT_DIR / "gate_b1_v2_by_condition.csv"
RESULT_JSON = OUT_DIR / "gate_b1_v2_result.json"

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260915

DUMMY_GT = 1

HEAD_DIM = 128
START = 1664
END = 1792


EXPECTED_ACTION_COUNTS = {
    0.0: 24,
    1.0: 1347,
    1.5: 32,
    2.0: 63,
    4.0: 534,
}


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def git_head():

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def image_path_for(sid):

    p = IMAGE_DIR / f"{sid}.png"

    if not p.exists():
        raise RuntimeError(
            f"Missing image: {p}"
        )

    return p


def paired_bootstrap_ci(delta):

    delta = np.asarray(
        delta,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = len(delta)

    boot = np.empty(
        BOOTSTRAP_REPS,
        dtype=np.float64,
    )

    batch = 500
    cursor = 0

    while cursor < BOOTSTRAP_REPS:

        b = min(
            batch,
            BOOTSTRAP_REPS - cursor,
        )

        idx = rng.integers(
            0,
            n,
            size=(b, n),
        )

        boot[
            cursor:cursor+b
        ] = delta[idx].mean(axis=1)

        cursor += b

    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


class PooledCSAModifier:

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
            .layers[LAYER]
        )

        self.o_proj = (
            layer
            .cross_attn
            .o_proj
        )

        hidden = int(
            self.o_proj.in_features
        )

        if hidden != 4096:
            raise RuntimeError(
                f"Unexpected hidden size: {hidden}"
            )

        self.start = START
        self.end = END

        self.alpha = float(alpha)

        U = np.asarray(
            U,
            dtype=np.float32,
        )

        if U.shape != (128, 4):
            raise RuntimeError(
                f"Bad U shape: {U.shape}"
            )

        self.U = U

        self.calls = 0
        self.handle = None


    def hook(
        self,
        module,
        inputs,
    ):

        x = inputs[0]

        H = x[
            ...,
            self.start:self.end
        ]

        h = (
            H
            .float()
            .mean(dim=1)
        )

        U = torch.as_tensor(
            self.U,
            dtype=torch.float32,
            device=H.device,
        )

        projected_h = (
            (h @ U)
            @ U.T
        )

        delta_h = (
            self.alpha - 1.0
        ) * projected_h

        H_new = (
            H
            +
            delta_h
            .to(H.dtype)
            .unsqueeze(1)
        )

        out = x.clone()

        out[
            ...,
            self.start:self.end
        ] = H_new

        self.calls += 1

        if len(inputs) == 1:
            return (out,)

        return (
            out,
            *inputs[1:]
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


print("=" * 96)
print("AROMA 2.0 — FORMAL CSA GATE B1 v2")
print("=" * 96)


# ============================================================
# DATA AUDIT
# ============================================================

df = pd.read_csv(
    V3_RESULTS
)

if len(df) != 2000:
    raise RuntimeError(
        f"Expected 2000 rows, got {len(df)}"
    )

if df["sample_id"].nunique() != 2000:
    raise RuntimeError(
        "Sample IDs are not unique."
    )

df["selected_alpha"] = (
    df["selected_alpha"]
    .astype(float)
)

counts = {
    float(k): int(v)
    for k, v
    in df["selected_alpha"]
    .value_counts()
    .sort_index()
    .to_dict()
    .items()
}

if counts != EXPECTED_ACTION_COUNTS:
    raise RuntimeError(
        f"Action counts mismatch: {counts}"
    )


for sid in df["sample_id"].astype(str):
    image_path_for(sid)


U4 = np.load(
    U4_PATH
).astype(np.float32)

if U4.shape != (128, 4):
    raise RuntimeError(
        f"Bad U4 shape: {U4.shape}"
    )

orth_error = float(
    np.max(
        np.abs(
            U4.T @ U4
            -
            np.eye(4)
        )
    )
)

if orth_error > 1e-4:
    raise RuntimeError(
        "U4 orthogonality check failed."
    )


print("PASS: V3 N=2000")
print("PASS: frozen action distribution")
print("PASS: U4 shape and orthogonality")
print(
    "U4 orthogonality max error:",
    orth_error
)


# ============================================================
# FINGERPRINT
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

fingerprint_data = {
    "script_sha256":
        sha256_file(SCRIPT_PATH),

    "freeze_sha256":
        sha256_file(FREEZE_DOC),

    "v3_results_sha256":
        sha256_file(V3_RESULTS),

    "u4_sha256":
        sha256_file(U4_PATH),

    "model_revision":
        MODEL_REVISION,

    "geometry":
        "pooled_mean_shared_displacement",

    "rank":
        4,

    "bootstrap_reps":
        BOOTSTRAP_REPS,

    "bootstrap_seed":
        BOOTSTRAP_SEED,
}

fingerprint = hashlib.sha256(
    json.dumps(
        fingerprint_data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


print("Git HEAD:", git_head())
print("Method fingerprint:", fingerprint)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
)

numeral_ids = numeral_token_ids(
    processor
)

if sorted(
    numeral_ids.keys()
) != list(range(16)):
    raise RuntimeError(
        "Numeral support != 0..15"
    )


print("Numeral support audit: PASS")

print()
print("Loading model...")

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

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


def get_inputs(sid):

    image = (
        Image.open(
            image_path_for(sid)
        )
        .convert("RGB")
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
# CSA IDENTITY TEST ONLY
# ============================================================

print()
print("=" * 96)
print("CSA ALPHA=1 IDENTITY TEST")
print("=" * 96)

sid = str(
    df.iloc[0]["sample_id"]
)

inputs = get_inputs(sid)

with torch.inference_mode():

    base_out = model(
        **inputs,
        use_cache=False,
        return_dict=True,
    )


modifier = PooledCSAModifier(
    model,
    alpha=1.0,
    U=U4,
)

modifier.register()

try:

    with torch.inference_mode():

        identity_out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

finally:

    modifier.remove()


identity_diff = float(
    (
        base_out.logits.float()
        -
        identity_out.logits.float()
    )
    .abs()
    .max()
    .item()
)

print(
    "Identity max logit abs diff:",
    identity_diff
)

if identity_diff != 0.0:
    raise RuntimeError(
        "CSA alpha=1 not identity."
    )

print("PASS: CSA alpha=1 exact identity")


del base_out
del identity_out
del inputs

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# RESUME
# ============================================================

ordered = (
    df
    .sort_values("sample_id")
    .reset_index(drop=True)
)

records = []


if PARTIAL_CSV.exists():

    if not PARTIAL_META.exists():
        raise RuntimeError(
            "Partial CSV without metadata."
        )

    meta = json.loads(
        PARTIAL_META.read_text()
    )

    if (
        meta["method_fingerprint"]
        != fingerprint
    ):
        raise RuntimeError(
            "Resume fingerprint mismatch."
        )

    partial = pd.read_csv(
        PARTIAL_CSV
    )

    records = partial.to_dict(
        "records"
    )

    ids = [
        str(r["sample_id"])
        for r in records
    ]

    expected = (
        ordered[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    if ids != expected[:len(ids)]:
        raise RuntimeError(
            "Resume order mismatch."
        )

    print(
        "RESUME VALIDATED:",
        len(records),
        "/2000",
    )


def save_partial():

    pd.DataFrame(
        records
    ).to_csv(
        PARTIAL_CSV,
        index=False,
    )

    PARTIAL_META.write_text(
        json.dumps(
            {
                "method_fingerprint":
                    fingerprint,

                "completed":
                    len(records),

                "total":
                    2000,
            },
            indent=2,
        )
    )


# ============================================================
# CURRENT SAME-RUN EVALUATION
# ============================================================

print()
print("=" * 96)
print("CURRENT SAME-RUN BASELINE / WHOLE / CSA")
print("=" * 96)


start_idx = len(records)


for idx in tqdm(
    range(start_idx, 2000),
    initial=start_idx,
    total=2000,
):

    row = ordered.iloc[idx]

    sid = str(
        row["sample_id"]
    )

    gt = int(
        row["ground_truth"]
    )

    alpha = float(
        row["selected_alpha"]
    )

    inputs = get_inputs(sid)


    # --------------------------------------------------------
    # Current baseline
    # --------------------------------------------------------

    baseline_state = score_state(
        model,
        inputs,
        numeral_ids,
        DUMMY_GT,
    )

    baseline_pred = int(
        baseline_state[
            "best_numeral"
        ]
    )


    # --------------------------------------------------------
    # Same-run whole + CSA
    # --------------------------------------------------------

    if math.isclose(
        alpha,
        1.0,
        abs_tol=1e-12,
        rel_tol=0.0,
    ):

        whole_pred = baseline_pred
        csa_pred = baseline_pred

        whole_calls = 0
        csa_calls = 0

    else:

        whole = HeadGainModifier(
            model=model,
            layer_idx=LAYER,
            head_idx=HEAD,
            alpha=alpha,
        )

        whole.register()

        try:

            whole_state = score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )

        finally:

            whole.remove()

        whole_pred = int(
            whole_state[
                "best_numeral"
            ]
        )

        whole_calls = int(
            whole.calls
        )

        if whole_calls <= 0:
            raise RuntimeError(
                f"Whole hook not called: {sid}"
            )


        csa = PooledCSAModifier(
            model,
            alpha=alpha,
            U=U4,
        )

        csa.register()

        try:

            csa_state = score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )

        finally:

            csa.remove()

        csa_pred = int(
            csa_state[
                "best_numeral"
            ]
        )

        csa_calls = int(
            csa.calls
        )

        if csa_calls <= 0:
            raise RuntimeError(
                f"CSA hook not called: {sid}"
            )


    records.append(
        {
            "sample_id":
                sid,

            "condition":
                str(row["condition"]),

            "ground_truth":
                gt,

            "selected_alpha":
                alpha,

            "archive_baseline_prediction":
                int(
                    row[
                        "baseline_prediction"
                    ]
                ),

            "archive_whole_prediction":
                int(
                    row[
                        "post_prediction"
                    ]
                ),

            "current_baseline_prediction":
                baseline_pred,

            "current_whole_prediction":
                whole_pred,

            "current_csa_prediction":
                csa_pred,

            "whole_hook_calls":
                whole_calls,

            "csa_hook_calls":
                csa_calls,
        }
    )


    if (
        len(records) % 25 == 0
        or len(records) == 2000
    ):
        save_partial()


# ============================================================
# FINAL DATA
# ============================================================

result = pd.DataFrame(
    records
)

if len(result) != 2000:
    raise RuntimeError(
        "Incomplete result."
    )


gt = result[
    "ground_truth"
].astype(int)


result[
    "baseline_correct"
] = (
    result[
        "current_baseline_prediction"
    ].astype(int)
    ==
    gt
)


result[
    "whole_correct"
] = (
    result[
        "current_whole_prediction"
    ].astype(int)
    ==
    gt
)


result[
    "csa_correct"
] = (
    result[
        "current_csa_prediction"
    ].astype(int)
    ==
    gt
)


result[
    "whole_repair"
] = (
    (~result["baseline_correct"])
    &
    result["whole_correct"]
)


result[
    "whole_break"
] = (
    result["baseline_correct"]
    &
    (~result["whole_correct"])
)


result[
    "csa_repair"
] = (
    (~result["baseline_correct"])
    &
    result["csa_correct"]
)


result[
    "csa_break"
] = (
    result["baseline_correct"]
    &
    (~result["csa_correct"])
)


result[
    "delta_correct"
] = (
    result[
        "csa_correct"
    ].astype(int)
    -
    result[
        "whole_correct"
    ].astype(int)
)


result[
    "archive_baseline_match"
] = (
    result[
        "archive_baseline_prediction"
    ].astype(int)
    ==
    result[
        "current_baseline_prediction"
    ].astype(int)
)


result[
    "archive_whole_match"
] = (
    result[
        "archive_whole_prediction"
    ].astype(int)
    ==
    result[
        "current_whole_prediction"
    ].astype(int)
)


result.to_csv(
    FINAL_CSV,
    index=False,
)


# ============================================================
# PRIMARY STATISTICS
# ============================================================

N = 2000

baseline_n = int(
    result["baseline_correct"].sum()
)

whole_n = int(
    result["whole_correct"].sum()
)

csa_n = int(
    result["csa_correct"].sum()
)


whole_repairs = int(
    result["whole_repair"].sum()
)

whole_breaks = int(
    result["whole_break"].sum()
)

csa_repairs = int(
    result["csa_repair"].sum()
)

csa_breaks = int(
    result["csa_break"].sum()
)


whole_net = (
    whole_repairs
    -
    whole_breaks
)

csa_net = (
    csa_repairs
    -
    csa_breaks
)


delta = result[
    "delta_correct"
].to_numpy(
    dtype=np.float64
)


delta_accuracy = float(
    delta.mean()
)

ci_low, ci_high = (
    paired_bootstrap_ci(delta)
)


whole_wrong_csa_correct = int(
    (
        (~result["whole_correct"])
        &
        result["csa_correct"]
    ).sum()
)


whole_correct_csa_wrong = int(
    (
        result["whole_correct"]
        &
        (~result["csa_correct"])
    ).sum()
)


discordant = (
    whole_wrong_csa_correct
    +
    whole_correct_csa_wrong
)


if discordant:

    mcnemar_p = float(
        binomtest(
            min(
                whole_wrong_csa_correct,
                whole_correct_csa_wrong,
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )

else:

    mcnemar_p = 1.0


if ci_low > 0:

    status = "B1_CLEAR_POSITIVE"

elif ci_high < 0:

    status = "B1_CLEAR_NEGATIVE"

else:

    status = "B1_NO_CLEAR_DIFFERENCE"


archive_baseline_match_rate = float(
    result[
        "archive_baseline_match"
    ].mean()
)

archive_whole_match_rate = float(
    result[
        "archive_whole_match"
    ].mean()
)


# ============================================================
# GROUP SUMMARIES
# ============================================================

def summarize(group):

    n = len(group)

    b = int(
        group[
            "baseline_correct"
        ].sum()
    )

    w = int(
        group[
            "whole_correct"
        ].sum()
    )

    c = int(
        group[
            "csa_correct"
        ].sum()
    )

    wr = int(
        group[
            "whole_repair"
        ].sum()
    )

    wb = int(
        group[
            "whole_break"
        ].sum()
    )

    cr = int(
        group[
            "csa_repair"
        ].sum()
    )

    cb = int(
        group[
            "csa_break"
        ].sum()
    )

    return {
        "n": n,

        "baseline_accuracy":
            b / n,

        "whole_accuracy":
            w / n,

        "csa_accuracy":
            c / n,

        "csa_minus_whole_pp":
            100.0 * (c - w) / n,

        "whole_repairs":
            wr,

        "whole_breaks":
            wb,

        "whole_net":
            wr - wb,

        "csa_repairs":
            cr,

        "csa_breaks":
            cb,

        "csa_net":
            cr - cb,
    }


action_rows = []

for alpha, group in result.groupby(
    "selected_alpha",
    sort=True,
):

    row = {
        "selected_alpha":
            float(alpha)
    }

    row.update(
        summarize(group)
    )

    action_rows.append(row)


pd.DataFrame(
    action_rows
).to_csv(
    BY_ACTION_CSV,
    index=False,
)


condition_rows = []

for condition, group in result.groupby(
    "condition",
    sort=True,
):

    row = {
        "condition":
            str(condition)
    }

    row.update(
        summarize(group)
    )

    condition_rows.append(row)


pd.DataFrame(
    condition_rows
).to_csv(
    BY_CONDITION_CSV,
    index=False,
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "stage":
        "AROMA2_CSA_GATE_B1_V2",

    "status":
        status,

    "git_head":
        git_head(),

    "method_fingerprint":
        fingerprint,

    "n":
        N,

    "current_baseline_correct":
        baseline_n,

    "current_whole_correct":
        whole_n,

    "current_csa_correct":
        csa_n,

    "current_baseline_accuracy":
        baseline_n / N,

    "current_whole_accuracy":
        whole_n / N,

    "current_csa_accuracy":
        csa_n / N,

    "csa_minus_whole_pp":
        100.0
        * delta_accuracy,

    "paired_bootstrap_95ci_pp":
        [
            100.0 * ci_low,
            100.0 * ci_high,
        ],

    "mcnemar_exact_p":
        mcnemar_p,

    "whole_repairs":
        whole_repairs,

    "whole_breaks":
        whole_breaks,

    "whole_net":
        whole_net,

    "csa_repairs":
        csa_repairs,

    "csa_breaks":
        csa_breaks,

    "csa_net":
        csa_net,

    "delta_net":
        csa_net
        - whole_net,

    "whole_wrong_csa_correct":
        whole_wrong_csa_correct,

    "whole_correct_csa_wrong":
        whole_correct_csa_wrong,

    "archive_baseline_prediction_match_rate":
        archive_baseline_match_rate,

    "archive_whole_prediction_match_rate":
        archive_whole_match_rate,

    "alpha1_identity_max_logit_diff":
        identity_diff,
}


RESULT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


pd.DataFrame(
    [summary]
).to_csv(
    SUMMARY_CSV,
    index=False,
)


print()
print("=" * 96)
print("FORMAL CSA GATE B1 v2 RESULT")
print("=" * 96)

print(
    "Current baseline:",
    baseline_n,
    "/2000",
    f"= {100*baseline_n/N:.3f}%"
)

print(
    "Current whole-head:",
    whole_n,
    "/2000",
    f"= {100*whole_n/N:.3f}%"
)

print(
    "Current CSA:",
    csa_n,
    "/2000",
    f"= {100*csa_n/N:.3f}%"
)

print()

print(
    "CSA - whole:",
    f"{100*delta_accuracy:+.3f} pp"
)

print(
    "Paired bootstrap 95% CI:",
    f"[{100*ci_low:+.3f}, "
    f"{100*ci_high:+.3f}] pp"
)

print(
    "Exact McNemar p:",
    mcnemar_p
)

print()

print(
    "WHOLE repairs / breaks / net:",
    whole_repairs,
    "/",
    whole_breaks,
    "/",
    whole_net
)

print(
    "CSA repairs / breaks / net:",
    csa_repairs,
    "/",
    csa_breaks,
    "/",
    csa_net
)

print(
    "Delta net:",
    csa_net
    - whole_net
)

print()

print(
    "Whole wrong -> CSA correct:",
    whole_wrong_csa_correct
)

print(
    "Whole correct -> CSA wrong:",
    whole_correct_csa_wrong
)

print()

print(
    "Archive/current baseline match:",
    archive_baseline_match_rate
)

print(
    "Archive/current whole match:",
    archive_whole_match_rate
)

print()

print(
    "PRIMARY STATUS:",
    status
)

print("=" * 96)

print()
print("BY ACTION")
print(
    pd.read_csv(
        BY_ACTION_CSV
    ).to_string(
        index=False
    )
)

print()
print("BY CONDITION")
print(
    pd.read_csv(
        BY_CONDITION_CSV
    ).to_string(
        index=False
    )
)

print()
print(
    "Result:",
    RESULT_JSON
)

print("=" * 96)

