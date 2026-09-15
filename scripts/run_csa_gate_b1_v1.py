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


sys.path.insert(
    0,
    "scripts",
)


# ============================================================
# EXACT CANONICAL IMPLEMENTATIONS
# ============================================================

from run_l18h13_gain_all300 import (
    MODEL_ID,
    LAYER,
    HEAD,
    prepare_inputs,
    move_inputs,
    HeadGainModifier,
)

from expanded_numeral_metrics import (
    NUMERALS,
    numeral_token_ids,
    score_state,
)


# ============================================================
# FROZEN CONFIG
# ============================================================

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
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

FREEZE_DOC = Path(
    "docs/AROMA2_CSA_GATEB1_FREEZE_v1.md"
)

CANONICAL_HEAD_RUNNER = Path(
    "scripts/run_l18h13_gain_all300.py"
)

EXPANDED_METRICS = Path(
    "scripts/expanded_numeral_metrics.py"
)

OUT_DIR = Path(
    "outputs/aroma2/csa_gate_b1_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL_CSV = (
    OUT_DIR
    / "partial_nonnoop.csv"
)

PARTIAL_META = (
    OUT_DIR
    / "partial_metadata.json"
)

FINAL_CSV = (
    OUT_DIR
    / "gate_b1_results.csv"
)

SUMMARY_CSV = (
    OUT_DIR
    / "gate_b1_summary.csv"
)

BY_ACTION_CSV = (
    OUT_DIR
    / "gate_b1_by_action.csv"
)

BY_CONDITION_CSV = (
    OUT_DIR
    / "gate_b1_by_condition.csv"
)

RESULT_JSON = (
    OUT_DIR
    / "gate_b1_result.json"
)


BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260915

HEAD_DIM = 128

START = 1664
END = 1792

DUMMY_GT = 1


EXPECTED_ACTION_COUNTS = {
    0.0: 24,
    1.0: 1347,
    1.5: 32,
    2.0: 63,
    4.0: 534,
}


# ============================================================
# UTILITY
# ============================================================

def sha256_file(
    path,
):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        while True:

            x = f.read(
                1024 * 1024
            )

            if not x:
                break

            h.update(
                x
            )

    return h.hexdigest()


def git_head():

    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def image_path_for(
    sample_id,
):

    p = (
        IMAGE_DIR
        / f"{sample_id}.png"
    )

    if not p.exists():

        raise RuntimeError(
            f"Missing V3 image: {p}"
        )

    return p


def paired_bootstrap_ci(
    delta,
):

    delta = np.asarray(
        delta,
        dtype=np.float64,
    )

    n = len(
        delta
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    boot = np.empty(
        BOOTSTRAP_REPS,
        dtype=np.float64,
    )

    batch_size = 500

    cursor = 0

    while cursor < BOOTSTRAP_REPS:

        b = min(
            batch_size,
            BOOTSTRAP_REPS
            - cursor,
        )

        idx = rng.integers(
            0,
            n,
            size=(
                b,
                n,
            ),
        )

        boot[
            cursor:
            cursor + b
        ] = (
            delta[
                idx
            ]
            .mean(
                axis=1
            )
        )

        cursor += b

    return (
        float(
            np.quantile(
                boot,
                0.025,
            )
        ),
        float(
            np.quantile(
                boot,
                0.975,
            )
        ),
    )


# ============================================================
# POOLED CSA ACTUATOR
# ============================================================

class PooledCSAModifier:

    """
    Gate-A-aligned CSA.

    Gate A defined:

        h = mean_t H_t

    and learned U4 in this 128-dimensional coordinate.

    We therefore apply:

        delta_h
        =
        (alpha - 1)
        U U^T h

        H'_t
        =
        H_t + delta_h

    preserving token residuals H_t - h.
    """

    def __init__(
        self,
        model,
        layer_idx,
        head_idx,
        alpha,
        U,
    ):

        layer = (
            model
            .model
            .language_model
            .layers[
                layer_idx
            ]
        )

        self.o_proj = (
            layer
            .cross_attn
            .o_proj
        )

        hidden_size = int(
            self.o_proj.in_features
        )

        if hidden_size != 4096:

            raise RuntimeError(
                f"Unexpected hidden size: "
                f"{hidden_size}"
            )

        self.head_dim = (
            hidden_size
            // 32
        )

        if self.head_dim != HEAD_DIM:

            raise RuntimeError(
                "Head-dimension mismatch."
            )

        self.start = (
            int(head_idx)
            * self.head_dim
        )

        self.end = (
            self.start
            + self.head_dim
        )

        if (
            self.start != START
            or
            self.end != END
        ):

            raise RuntimeError(
                "L18H13 slice mismatch."
            )

        U = np.asarray(
            U,
            dtype=np.float32,
        )

        if U.shape != (
            128,
            4,
        ):

            raise RuntimeError(
                f"Expected U=(128,4), "
                f"got {U.shape}"
            )

        self.U = U

        self.alpha = float(
            alpha
        )

        self.calls = 0

        self.handle = None


    def hook(
        self,
        module,
        inputs,
    ):

        if not inputs:

            return inputs

        x = inputs[0]

        H = x[
            ...,
            self.start:self.end
        ]

        # Gate-A pooled coordinate.
        h = (
            H
            .float()
            .mean(
                dim=1
            )
        )

        U = torch.as_tensor(
            self.U,
            dtype=torch.float32,
            device=H.device,
        )

        coefficients = torch.matmul(
            h,
            U,
        )

        projected_h = torch.matmul(
            coefficients,
            U.transpose(
                0,
                1,
            ),
        )

        delta_h = (
            self.alpha
            - 1.0
        ) * projected_h

        # Match Gate-A local coordinate:
        # cast the displacement to the head dtype,
        # then add the same displacement to every token.
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

        modified = x.clone()

        modified[
            ...,
            self.start:self.end
        ] = H_new

        self.calls += 1

        if len(inputs) == 1:

            return (
                modified,
            )

        return (
            modified,
            *inputs[1:],
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
# LOAD FROZEN DATA
# ============================================================

print("=" * 96)
print(
    "AROMA 2.0 — FORMAL CSA GATE B1"
)
print("=" * 96)


for required in [
    V3_RESULTS,
    U4_PATH,
    FREEZE_DOC,
    CANONICAL_HEAD_RUNNER,
    EXPANDED_METRICS,
]:

    if not required.exists():

        raise RuntimeError(
            f"Missing artifact: "
            f"{required}"
        )


df = pd.read_csv(
    V3_RESULTS
)


if len(df) != 2000:

    raise RuntimeError(
        f"Expected N=2000, "
        f"got {len(df)}"
    )


if (
    df[
        "sample_id"
    ]
    .astype(str)
    .nunique()
    != 2000
):

    raise RuntimeError(
        "V3 sample IDs are not unique."
    )


required_columns = {
    "sample_id",
    "condition",
    "ground_truth",
    "baseline_prediction",
    "selected_alpha",
    "post_prediction",
}

missing = (
    required_columns
    - set(
        df.columns
    )
)

if missing:

    raise RuntimeError(
        f"Missing columns: "
        f"{sorted(missing)}"
    )


df = df.copy()

df[
    "selected_alpha"
] = (
    df[
        "selected_alpha"
    ]
    .astype(float)
)


actual_action_counts = (
    df[
        "selected_alpha"
    ]
    .value_counts()
    .sort_index()
    .to_dict()
)


actual_action_counts = {
    float(k):
        int(v)
    for k, v
    in actual_action_counts.items()
}


if actual_action_counts != (
    EXPECTED_ACTION_COUNTS
):

    raise RuntimeError(
        "Frozen action distribution "
        f"mismatch:\n"
        f"{actual_action_counts}"
    )


for sid in (
    df[
        "sample_id"
    ]
    .astype(str)
):

    image_path_for(
        sid
    )


# ============================================================
# ARCHIVED WHOLE-HEAD AUDIT
# ============================================================

baseline_correct = (
    df[
        "baseline_prediction"
    ].astype(int)
    ==
    df[
        "ground_truth"
    ].astype(int)
)


whole_correct = (
    df[
        "post_prediction"
    ].astype(int)
    ==
    df[
        "ground_truth"
    ].astype(int)
)


whole_repair = (
    (~baseline_correct)
    &
    whole_correct
)


whole_break = (
    baseline_correct
    &
    (~whole_correct)
)


if int(
    baseline_correct.sum()
) != 999:

    raise RuntimeError(
        "Archived baseline != 999."
    )


if int(
    whole_correct.sum()
) != 1161:

    raise RuntimeError(
        "Archived whole-head != 1161."
    )


if int(
    whole_repair.sum()
) != 173:

    raise RuntimeError(
        "Archived repairs != 173."
    )


if int(
    whole_break.sum()
) != 11:

    raise RuntimeError(
        "Archived breaks != 11."
    )


print(
    "PASS: frozen V3 N=2000"
)

print(
    "PASS: baseline 999/2000"
)

print(
    "PASS: whole-head 1161/2000"
)

print(
    "PASS: repairs=173 breaks=11"
)

print(
    "PASS: action distribution:",
    actual_action_counts,
)


# ============================================================
# U4 AUDIT
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
    "U4 orthonormality max error:",
    orth_error
)


if orth_error > 1e-4:

    raise RuntimeError(
        "U4 orthogonality audit failed."
    )


# ============================================================
# METHOD FINGERPRINT
# ============================================================

SCRIPT_PATH = Path(
    __file__
).resolve()


fingerprint_metadata = {
    "script_sha256":
        sha256_file(
            SCRIPT_PATH
        ),

    "freeze_doc_sha256":
        sha256_file(
            FREEZE_DOC
        ),

    "v3_results_sha256":
        sha256_file(
            V3_RESULTS
        ),

    "u4_sha256":
        sha256_file(
            U4_PATH
        ),

    "canonical_runner_sha256":
        sha256_file(
            CANONICAL_HEAD_RUNNER
        ),

    "expanded_metrics_sha256":
        sha256_file(
            EXPANDED_METRICS
        ),

    "model_revision":
        MODEL_REVISION,

    "layer":
        int(
            LAYER
        ),

    "head":
        int(
            HEAD
        ),

    "head_slice":
        [
            START,
            END,
        ],

    "rank":
        4,

    "geometry":
        "pooled_mean_shared_displacement",

    "bootstrap_reps":
        BOOTSTRAP_REPS,

    "bootstrap_seed":
        BOOTSTRAP_SEED,
}


fingerprint_json = json.dumps(
    fingerprint_metadata,
    sort_keys=True,
    separators=(
        ",",
        ":",
    ),
)


METHOD_FINGERPRINT = (
    hashlib.sha256(
        fingerprint_json
        .encode(
            "utf-8"
        )
    ).hexdigest()
)


print(
    "Git HEAD:",
    git_head()
)

print(
    "Method fingerprint:",
    METHOD_FINGERPRINT
)


(
    OUT_DIR
    / "method_fingerprint.json"
).write_text(
    json.dumps(
        {
            "fingerprint":
                METHOD_FINGERPRINT,

            "metadata":
                fingerprint_metadata,
        },
        indent=2,
    ),
    encoding="utf-8",
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
    range(
        16
    )
):

    raise RuntimeError(
        "Expanded numeral support "
        "is not 0..15."
    )


print(
    "Expanded numeral audit: PASS"
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
        device_map="auto",
        attn_implementation=
            "eager",
    )
)


model.eval()


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )


def get_inputs(
    sid,
):

    image = (
        Image.open(
            image_path_for(
                str(
                    sid
                )
            )
        )
        .convert(
            "RGB"
        )
    )

    inputs = (
        prepare_inputs(
            processor,
            image,
        )
    )

    return move_inputs(
        inputs,
        model,
    )


# ============================================================
# CURRENT-VS-ARCHIVED WHOLE-HEAD PREFLIGHT
# ============================================================

print()
print("=" * 96)
print(
    "CURRENT / ARCHIVED V3 PREFLIGHT"
)
print("=" * 96)


preflight_rows = []


for alpha in [
    0.0,
    1.5,
    2.0,
    4.0,
]:

    group = (
        df[
            np.isclose(
                df[
                    "selected_alpha"
                ],
                alpha,
            )
        ]
        .reset_index(
            drop=True
        )
    )

    indices = sorted(
        set(
            [
                0,
                len(group) // 2,
                len(group) - 1,
            ]
        )
    )

    for i in indices:

        preflight_rows.append(
            group.iloc[
                i
            ]
        )


for row in preflight_rows:

    sid = str(
        row[
            "sample_id"
        ]
    )

    alpha = float(
        row[
            "selected_alpha"
        ]
    )

    inputs = get_inputs(
        sid
    )


    baseline_state = (
        score_state(
            model,
            inputs,
            numeral_ids,
            DUMMY_GT,
        )
    )


    current_baseline = int(
        baseline_state[
            "best_numeral"
        ]
    )


    archived_baseline = int(
        row[
            "baseline_prediction"
        ]
    )


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

        whole_state = (
            score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        )

    finally:

        modifier.remove()


    current_whole = int(
        whole_state[
            "best_numeral"
        ]
    )


    archived_whole = int(
        row[
            "post_prediction"
        ]
    )


    print(
        sid,
        f"a={alpha}",
        f"baseline="
        f"{archived_baseline}/"
        f"{current_baseline}",
        f"whole="
        f"{archived_whole}/"
        f"{current_whole}",
    )


    if (
        current_baseline
        != archived_baseline
    ):

        raise RuntimeError(
            "Baseline replay mismatch."
        )


    if (
        current_whole
        != archived_whole
    ):

        raise RuntimeError(
            "Whole-head replay mismatch."
        )


print(
    "PASS: archived whole-head "
    "behavior reproduced"
)


# ============================================================
# CSA ALPHA=1 EXACT IDENTITY
# ============================================================

print()
print("=" * 96)
print(
    "CSA ALPHA=1 IDENTITY TEST"
)
print("=" * 96)


sid = str(
    df.iloc[0][
        "sample_id"
    ]
)


inputs = get_inputs(
    sid
)


with torch.inference_mode():

    baseline_out = model(
        **inputs,
        use_cache=False,
        return_dict=True,
    )


csa_identity = (
    PooledCSAModifier(
        model=model,
        layer_idx=LAYER,
        head_idx=HEAD,
        alpha=1.0,
        U=U4,
    )
)


csa_identity.register()


try:

    with torch.inference_mode():

        identity_out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

finally:

    csa_identity.remove()


identity_max_diff = float(
    (
        baseline_out
        .logits
        .float()
        -
        identity_out
        .logits
        .float()
    )
    .abs()
    .max()
    .item()
)


print(
    "Identity max logit abs diff:",
    identity_max_diff
)


if identity_max_diff != 0.0:

    raise RuntimeError(
        "CSA alpha=1 is not exact identity."
    )


print(
    "PASS: CSA alpha=1 exact identity"
)


del baseline_out
del identity_out
del inputs

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# FORMAL NON-NOOP POPULATION
# ============================================================

nonnoop = (
    df[
        ~np.isclose(
            df[
                "selected_alpha"
            ],
            1.0,
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


if len(
    nonnoop
) != 653:

    raise RuntimeError(
        f"Expected 653 interventions, "
        f"got {len(nonnoop)}"
    )


expected_ids = (
    nonnoop[
        "sample_id"
    ]
    .astype(str)
    .tolist()
)


# ============================================================
# RESUME
# ============================================================

records = []


if PARTIAL_CSV.exists():

    if not PARTIAL_META.exists():

        raise RuntimeError(
            "Partial CSV exists without "
            "metadata."
        )


    meta = json.loads(
        PARTIAL_META.read_text(
            encoding="utf-8"
        )
    )


    stored_fingerprint = (
        meta[
            "method_fingerprint"
        ]
    )


    if (
        stored_fingerprint
        != METHOD_FINGERPRINT
    ):

        raise RuntimeError(
            "\nRESUME REFUSED\n"
            "Method fingerprint changed.\n"
            f"stored : "
            f"{stored_fingerprint}\n"
            f"current: "
            f"{METHOD_FINGERPRINT}"
        )


    partial = pd.read_csv(
        PARTIAL_CSV
    )


    records = (
        partial
        .to_dict(
            "records"
        )
    )


    observed_ids = [
        str(
            r[
                "sample_id"
            ]
        )
        for r in records
    ]


    if observed_ids != (
        expected_ids[
            :len(
                observed_ids
            )
        ]
    ):

        raise RuntimeError(
            "Partial sample order mismatch."
        )


    print(
        "RESUME VALIDATED:",
        len(
            records
        ),
        "/653",
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
                    METHOD_FINGERPRINT,

                "completed":
                    len(
                        records
                    ),

                "total":
                    653,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# FORMAL CSA EVALUATION
# ============================================================

print()
print("=" * 96)
print(
    "FORMAL CSA NON-NOOP EVALUATION"
)
print("=" * 96)


start_idx = len(
    records
)


for idx in tqdm(
    range(
        start_idx,
        len(
            nonnoop
        ),
    ),
    initial=start_idx,
    total=len(
        nonnoop
    ),
):

    row = nonnoop.iloc[
        idx
    ]


    sid = str(
        row[
            "sample_id"
        ]
    )


    alpha = float(
        row[
            "selected_alpha"
        ]
    )


    inputs = get_inputs(
        sid
    )


    modifier = (
        PooledCSAModifier(
            model=model,
            layer_idx=LAYER,
            head_idx=HEAD,
            alpha=alpha,
            U=U4,
        )
    )


    modifier.register()


    try:

        csa_state = (
            score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        )

    finally:

        modifier.remove()


    if modifier.calls <= 0:

        raise RuntimeError(
            f"CSA hook not called: "
            f"{sid}"
        )


    records.append(
        {
            "sample_id":
                sid,

            "selected_alpha":
                alpha,

            "csa_prediction":
                int(
                    csa_state[
                        "best_numeral"
                    ]
                ),

            "csa_expected_numeral":
                float(
                    csa_state[
                        "expected_numeral"
                    ]
                ),

            "hook_calls":
                int(
                    modifier.calls
                ),
        }
    )


    if (
        len(
            records
        )
        % 25
        == 0
        or
        len(
            records
        )
        == 653
    ):

        save_partial()


# ============================================================
# FINALIZE PER-SAMPLE DATA
# ============================================================

partial = pd.DataFrame(
    records
)


if len(
    partial
) != 653:

    raise RuntimeError(
        "CSA evaluation incomplete."
    )


if (
    partial[
        "sample_id"
    ]
    .astype(str)
    .tolist()
    != expected_ids
):

    raise RuntimeError(
        "CSA final sample ordering mismatch."
    )


prediction_map = dict(
    zip(
        partial[
            "sample_id"
        ].astype(str),
        partial[
            "csa_prediction"
        ].astype(int),
    )
)


expected_map = dict(
    zip(
        partial[
            "sample_id"
        ].astype(str),
        partial[
            "csa_expected_numeral"
        ].astype(float),
    )
)


csa_predictions = []

csa_expected = []


for _, row in df.iterrows():

    sid = str(
        row[
            "sample_id"
        ]
    )


    alpha = float(
        row[
            "selected_alpha"
        ]
    )


    if math.isclose(
        alpha,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        csa_predictions.append(
            int(
                row[
                    "baseline_prediction"
                ]
            )
        )

        csa_expected.append(
            np.nan
        )

    else:

        csa_predictions.append(
            int(
                prediction_map[
                    sid
                ]
            )
        )

        csa_expected.append(
            float(
                expected_map[
                    sid
                ]
            )
        )


result = df[
    [
        "sample_id",
        "condition",
        "replicate",
        "ground_truth",
        "baseline_prediction",
        "selected_alpha",
        "selected_score",
        "post_prediction",
    ]
].copy()


result = result.rename(
    columns={
        "post_prediction":
            "whole_prediction",
    }
)


result[
    "csa_prediction"
] = csa_predictions


result[
    "csa_expected_numeral"
] = csa_expected


result[
    "baseline_correct"
] = (
    result[
        "baseline_prediction"
    ].astype(int)
    ==
    result[
        "ground_truth"
    ].astype(int)
)


result[
    "whole_correct"
] = (
    result[
        "whole_prediction"
    ].astype(int)
    ==
    result[
        "ground_truth"
    ].astype(int)
)


result[
    "csa_correct"
] = (
    result[
        "csa_prediction"
    ].astype(int)
    ==
    result[
        "ground_truth"
    ].astype(int)
)


result[
    "whole_repair"
] = (
    (~result[
        "baseline_correct"
    ])
    &
    result[
        "whole_correct"
    ]
)


result[
    "whole_break"
] = (
    result[
        "baseline_correct"
    ]
    &
    (~result[
        "whole_correct"
    ])
)


result[
    "csa_repair"
] = (
    (~result[
        "baseline_correct"
    ])
    &
    result[
        "csa_correct"
    ]
)


result[
    "csa_break"
] = (
    result[
        "baseline_correct"
    ]
    &
    (~result[
        "csa_correct"
    ])
)


result[
    "csa_minus_whole_correct"
] = (
    result[
        "csa_correct"
    ].astype(int)
    -
    result[
        "whole_correct"
    ].astype(int)
)


result.to_csv(
    FINAL_CSV,
    index=False,
)


# ============================================================
# PRIMARY STATISTICS
# ============================================================

N = len(
    result
)


baseline_n = int(
    result[
        "baseline_correct"
    ].sum()
)


whole_n = int(
    result[
        "whole_correct"
    ].sum()
)


csa_n = int(
    result[
        "csa_correct"
    ].sum()
)


whole_repairs = int(
    result[
        "whole_repair"
    ].sum()
)


whole_breaks = int(
    result[
        "whole_break"
    ].sum()
)


csa_repairs = int(
    result[
        "csa_repair"
    ].sum()
)


csa_breaks = int(
    result[
        "csa_break"
    ].sum()
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


delta = (
    result[
        "csa_minus_whole_correct"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


delta_accuracy = float(
    delta.mean()
)


ci_low, ci_high = (
    paired_bootstrap_ci(
        delta
    )
)


whole_wrong_csa_correct = int(
    (
        (~result[
            "whole_correct"
        ])
        &
        result[
            "csa_correct"
        ]
    ).sum()
)


whole_correct_csa_wrong = int(
    (
        result[
            "whole_correct"
        ]
        &
        (~result[
            "csa_correct"
        ])
    ).sum()
)


discordant = (
    whole_wrong_csa_correct
    +
    whole_correct_csa_wrong
)


if discordant == 0:

    mcnemar_p = 1.0

else:

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


if ci_low > 0:

    status = (
        "B1_CLEAR_POSITIVE"
    )

elif ci_high < 0:

    status = (
        "B1_CLEAR_NEGATIVE"
    )

else:

    status = (
        "B1_NO_CLEAR_DIFFERENCE"
    )


# ============================================================
# GROUP SUMMARIES
# ============================================================

def summarize(
    group,
):

    n = len(
        group
    )

    baseline = int(
        group[
            "baseline_correct"
        ].sum()
    )

    whole = int(
        group[
            "whole_correct"
        ].sum()
    )

    csa = int(
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
        "n":
            n,

        "baseline_accuracy":
            baseline / n,

        "whole_accuracy":
            whole / n,

        "csa_accuracy":
            csa / n,

        "csa_minus_whole_pp":
            100.0
            * (
                csa - whole
            )
            / n,

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


for alpha, group in (
    result.groupby(
        "selected_alpha",
        sort=True,
    )
):

    row = {
        "selected_alpha":
            float(
                alpha
            )
    }

    row.update(
        summarize(
            group
        )
    )

    action_rows.append(
        row
    )


pd.DataFrame(
    action_rows
).to_csv(
    BY_ACTION_CSV,
    index=False,
)


condition_rows = []


for condition, group in (
    result.groupby(
        "condition",
        sort=True,
    )
):

    row = {
        "condition":
            str(
                condition
            )
    }

    row.update(
        summarize(
            group
        )
    )

    condition_rows.append(
        row
    )


pd.DataFrame(
    condition_rows
).to_csv(
    BY_CONDITION_CSV,
    index=False,
)


# ============================================================
# SAVE FINAL RESULT
# ============================================================

summary = {
    "stage":
        "AROMA2_CSA_GATE_B1_V1",

    "status":
        status,

    "method_fingerprint":
        METHOD_FINGERPRINT,

    "git_head":
        git_head(),

    "n":
        N,

    "nonnoop_interventions":
        653,

    "baseline_correct":
        baseline_n,

    "whole_correct":
        whole_n,

    "csa_correct":
        csa_n,

    "baseline_accuracy":
        baseline_n / N,

    "whole_accuracy":
        whole_n / N,

    "csa_accuracy":
        csa_n / N,

    "csa_minus_whole_pp":
        100.0
        * delta_accuracy,

    "paired_bootstrap_95ci_pp":
        [
            100.0
            * ci_low,

            100.0
            * ci_high,
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

    "u4_sha256":
        sha256_file(
            U4_PATH
        ),

    "u4_orthonormality_max_error":
        orth_error,

    "alpha1_identity_max_logit_diff":
        identity_max_diff,
}


RESULT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


pd.DataFrame(
    [
        summary
    ]
).to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 96)
print(
    "FORMAL CSA GATE B1 RESULT"
)
print("=" * 96)


print(
    "Baseline:",
    baseline_n,
    "/",
    N,
    f"= {100*baseline_n/N:.3f}%"
)


print(
    "Whole-head:",
    whole_n,
    "/",
    N,
    f"= {100*whole_n/N:.3f}%"
)


print(
    "CSA:",
    csa_n,
    "/",
    N,
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
    "PRIMARY STATUS:",
    status
)


print("=" * 96)


print()
print(
    "BY ACTION"
)

print(
    pd.read_csv(
        BY_ACTION_CSV
    ).to_string(
        index=False
    )
)


print()
print(
    "BY CONDITION"
)

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

