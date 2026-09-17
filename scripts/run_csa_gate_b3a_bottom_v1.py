import gc
import hashlib
import json
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

V3_RESULTS = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller/"
    "v3_final_results.csv"
)

B2_RESULTS = Path(
    "outputs/aroma2/csa_gate_b2_mn_v1/"
    "gate_b2_mn_results.csv"
)

TOP4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

SECOND_MOMENT_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "second_moment_uncentered.npy"
)

IMAGE_DIR = Path(
    "data/proc_count_causal_v3/images"
)

FREEZE_DOC = Path(
    "docs/"
    "AROMA2_CSA_GATEB3A_BOTTOM_FREEZE_v1.md"
)

OUT_DIR = Path(
    "outputs/aroma2/"
    "csa_gate_b3a_bottom_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL_CSV = (
    OUT_DIR
    / "partial_results.csv"
)

PARTIAL_META = (
    OUT_DIR
    / "partial_metadata.json"
)

FINAL_CSV = (
    OUT_DIR
    / "gate_b3a_results.csv"
)

SUMMARY_CSV = (
    OUT_DIR
    / "gate_b3a_summary.csv"
)

BY_ACTION_CSV = (
    OUT_DIR
    / "gate_b3a_by_action.csv"
)

BY_CONDITION_CSV = (
    OUT_DIR
    / "gate_b3a_by_condition.csv"
)

RESULT_JSON = (
    OUT_DIR
    / "gate_b3a_result.json"
)


EXPECTED_N = 653

EXPECTED_ACTION_COUNTS = {
    0.0: 24,
    1.5: 32,
    2.0: 63,
    4.0: 534,
}

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260915

DUMMY_GT = 1

START = 1664
END = 1792


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
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


def paired_bootstrap_ci(
    difference,
):

    difference = np.asarray(
        difference,
        dtype=np.float64,
    )

    n = len(difference)

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    boot = np.empty(
        BOOTSTRAP_REPS,
        dtype=np.float64,
    )

    batch = 500
    cursor = 0

    while cursor < BOOTSTRAP_REPS:

        b = min(
            batch,
            BOOTSTRAP_REPS
            - cursor,
        )

        indices = rng.integers(
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
            difference[
                indices
            ]
            .mean(axis=1)
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


def exact_mcnemar(
    ref,
    cand,
):

    ref = np.asarray(
        ref,
        dtype=bool,
    )

    cand = np.asarray(
        cand,
        dtype=bool,
    )

    ref_wrong_cand_right = int(
        (
            (~ref)
            &
            cand
        ).sum()
    )

    ref_right_cand_wrong = int(
        (
            ref
            &
            (~cand)
        ).sum()
    )

    discordant = (
        ref_wrong_cand_right
        +
        ref_right_cand_wrong
    )

    if discordant == 0:

        p = 1.0

    else:

        p = float(
            binomtest(
                min(
                    ref_wrong_cand_right,
                    ref_right_cand_wrong,
                ),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "reference_wrong_candidate_correct":
            ref_wrong_cand_right,

        "reference_correct_candidate_wrong":
            ref_right_cand_wrong,

        "discordant":
            discordant,

        "p":
            p,
    }


# ============================================================
# MAGNITUDE-NORMALIZED SUBSPACE MODIFIER
# ============================================================

class MNSubspaceModifier:

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

        if int(
            self.o_proj.in_features
        ) != 4096:

            raise RuntimeError(
                "Unexpected hidden size."
            )

        U = np.asarray(
            U,
            dtype=np.float32,
        )

        if U.shape != (128, 4):

            raise RuntimeError(
                f"Bad U shape: "
                f"{U.shape}"
            )

        self.U = torch.tensor(
            U,
            dtype=torch.float32,
        )

        self.alpha = float(
            alpha
        )

        self.calls = 0
        self.handle = None

        self.last_rho = None
        self.last_strength_error = None


    def hook(
        self,
        module,
        inputs,
    ):

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

        U = self.U.to(
            H.device
        )

        projection = (
            (h @ U)
            @ U.T
        )

        h_norm = (
            torch.linalg.vector_norm(
                h,
                dim=-1,
                keepdim=True,
            )
        )

        p_norm = (
            torch.linalg.vector_norm(
                projection,
                dim=-1,
                keepdim=True,
            )
        )

        if float(
            p_norm.min().item()
        ) <= 1e-12:

            raise RuntimeError(
                "Projection norm "
                "too small."
            )

        direction = (
            projection
            / p_norm
        )

        delta = (
            (
                self.alpha
                - 1.0
            )
            *
            h_norm
            *
            direction
        )

        actual = (
            torch.linalg.vector_norm(
                delta,
                dim=-1,
                keepdim=True,
            )
        )

        target = (
            abs(
                self.alpha
                - 1.0
            )
            *
            h_norm
        )

        strength_error = (
            torch.abs(
                actual
                -
                target
            )
            /
            torch.clamp(
                target,
                min=1e-12,
            )
        )

        self.last_rho = float(
            (
                p_norm
                /
                h_norm
            )[
                0,
                0
            ].item()
        )

        self.last_strength_error = (
            float(
                strength_error[
                    0,
                    0
                ].item()
            )
        )

        H_new = (
            H
            +
            delta
            .to(H.dtype)
            .unsqueeze(1)
        )

        out = x.clone()

        out[
            ...,
            START:END
        ] = H_new

        self.calls += 1

        if len(inputs) == 1:

            return (out,)

        return (
            out,
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
# START
# ============================================================

print("=" * 96)
print(
    "AROMA 2.0 — CSA GATE B3a "
    "TOP-4 vs BOTTOM-4"
)
print("=" * 96)


for path in [
    V3_RESULTS,
    B2_RESULTS,
    TOP4_PATH,
    SECOND_MOMENT_PATH,
    FREEZE_DOC,
]:

    if not path.exists():

        raise RuntimeError(
            f"Missing artifact: {path}"
        )


# ============================================================
# POPULATION
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
    for k, v in (
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
        f"Action counts "
        f"mismatch: {counts}"
    )


for sid in (
    data[
        "sample_id"
    ]
    .astype(str)
):

    resolve_image(
        sid
    )


print(
    "PASS: frozen N=653"
)

print(
    "PASS: action counts:",
    counts
)


# ============================================================
# CONSTRUCT TOP/BOTTOM SUBSPACES
# ============================================================

U_saved = np.load(
    TOP4_PATH
).astype(
    np.float64
)

S = np.load(
    SECOND_MOMENT_PATH
).astype(
    np.float64
)


if U_saved.shape != (128, 4):

    raise RuntimeError(
        f"Bad saved U4 shape: "
        f"{U_saved.shape}"
    )


if S.shape != (128, 128):

    raise RuntimeError(
        f"Bad second moment: "
        f"{S.shape}"
    )


S = (
    S + S.T
) / 2.0


eigvals, eigvecs = np.linalg.eigh(
    S
)


# np.linalg.eigh is ascending.
U_bottom = (
    eigvecs[
        :,
        :4
    ]
    .astype(
        np.float32
    )
)


U_reconstructed_top = (
    eigvecs[
        :,
        -4:
    ]
    .astype(
        np.float32
    )
)


P_saved = (
    U_saved
    @ U_saved.T
)


P_reconstructed = (
    U_reconstructed_top
    @ U_reconstructed_top.T
)


projector_error = float(
    np.max(
        np.abs(
            P_saved
            -
            P_reconstructed
        )
    )
)


print(
    "Top-4 projector "
    "reconstruction error:",
    projector_error
)


if projector_error > 1e-5:

    raise RuntimeError(
        "Reconstructed Top-4 "
        "does not match saved U4."
    )


bottom_orth_error = float(
    np.max(
        np.abs(
            U_bottom.T
            @ U_bottom
            -
            np.eye(4)
        )
    )
)


cross_overlap = float(
    np.linalg.norm(
        U_saved.T
        @ U_bottom,
        ord=2,
    )
)


print(
    "Bottom-4 orth error:",
    bottom_orth_error
)

print(
    "Top/Bottom spectral overlap:",
    cross_overlap
)


if bottom_orth_error > 1e-5:

    raise RuntimeError(
        "Bottom-4 not orthonormal."
    )


if cross_overlap > 1e-4:

    raise RuntimeError(
        "Top and Bottom spaces "
        "are not sufficiently "
        "orthogonal."
    )


np.save(
    OUT_DIR
    / "U4_bottom.npy",
    U_bottom,
)


print(
    "Top-4 eigenvalues:",
    eigvals[
        -4:
    ][::-1]
)

print(
    "Bottom-4 eigenvalues:",
    eigvals[
        :4
    ]
)


# ============================================================
# B2 REPLAY REFERENCE
# ============================================================

b2 = pd.read_csv(
    B2_RESULTS
)


if len(b2) != EXPECTED_N:

    raise RuntimeError(
        "B2 result N mismatch."
    )


b2_prediction_map = dict(
    zip(
        b2[
            "sample_id"
        ].astype(str),
        b2[
            "mn_csa_prediction"
        ].astype(int),
    )
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

    "freeze_sha256":
        sha256_file(
            FREEZE_DOC
        ),

    "v3_results_sha256":
        sha256_file(
            V3_RESULTS
        ),

    "b2_results_sha256":
        sha256_file(
            B2_RESULTS
        ),

    "top4_sha256":
        sha256_file(
            TOP4_PATH
        ),

    "second_moment_sha256":
        sha256_file(
            SECOND_MOMENT_PATH
        ),

    "model_revision":
        MODEL_REVISION,

    "n":
        EXPECTED_N,

    "rank":
        4,

    "control":
        "bottom_4_eigenvectors",

    "bootstrap_reps":
        BOOTSTRAP_REPS,

    "bootstrap_seed":
        BOOTSTRAP_SEED,
}


METHOD_FINGERPRINT = hashlib.sha256(
    json.dumps(
        fingerprint_metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode(
        "utf-8"
    )
).hexdigest()


print(
    "Git HEAD:",
    git_head()
)

print(
    "Method fingerprint:",
    METHOD_FINGERPRINT
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading processor...")


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
    "Numeral audit: PASS"
)


print()
print("Loading model...")


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


print(
    "model.training:",
    model.training
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


def get_inputs(sid):

    image = (
        Image.open(
            resolve_image(
                sid
            )
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
# ALPHA=1 IDENTITY
# ============================================================

print()
print("=" * 96)
print(
    "TOP/BOTTOM ALPHA=1 "
    "IDENTITY TEST"
)
print("=" * 96)


test_sid = str(
    data.iloc[
        0
    ][
        "sample_id"
    ]
)


inputs = get_inputs(
    test_sid
)


with torch.inference_mode():

    base = model(
        **inputs,
        use_cache=False,
        return_dict=True,
    )


for name, U in [
    (
        "top4",
        U_saved,
    ),
    (
        "bottom4",
        U_bottom,
    ),
]:

    modifier = (
        MNSubspaceModifier(
            model=model,
            alpha=1.0,
            U=U,
        )
    )

    modifier.register()

    try:

        with torch.inference_mode():

            out = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

    finally:

        modifier.remove()


    diff = float(
        (
            base.logits.float()
            -
            out.logits.float()
        )
        .abs()
        .max()
        .item()
    )


    print(
        name,
        "identity diff:",
        diff
    )


    if diff != 0.0:

        raise RuntimeError(
            f"{name} alpha=1 "
            f"is not identity."
        )


print(
    "PASS: both exact identity"
)


del base
del out
del inputs

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# RESUME
# ============================================================

records = []


if PARTIAL_CSV.exists():

    if not PARTIAL_META.exists():

        raise RuntimeError(
            "Partial without metadata."
        )


    meta = json.loads(
        PARTIAL_META
        .read_text(
            encoding="utf-8"
        )
    )


    if (
        meta[
            "method_fingerprint"
        ]
        != METHOD_FINGERPRINT
    ):

        raise RuntimeError(
            "Resume fingerprint "
            "mismatch."
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


    observed = [
        str(
            r[
                "sample_id"
            ]
        )
        for r in records
    ]


    expected = (
        data[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )


    if observed != (
        expected[
            :len(observed)
        ]
    ):

        raise RuntimeError(
            "Resume order mismatch."
        )


    print(
        "RESUME VALIDATED:",
        len(records),
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
                    len(records),

                "total":
                    EXPECTED_N,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# FORMAL TOP-vs-BOTTOM EVALUATION
# ============================================================

print()
print("=" * 96)
print(
    "FORMAL MN TOP-4 vs "
    "MN BOTTOM-4"
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
    initial=start_idx,
    total=EXPECTED_N,
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


    inputs = get_inputs(
        sid
    )


    predictions = {}
    metadata = {}


    for name, U in [
        (
            "top4",
            U_saved,
        ),
        (
            "bottom4",
            U_bottom,
        ),
    ]:

        modifier = (
            MNSubspaceModifier(
                model=model,
                alpha=alpha,
                U=U,
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
                f"{name} hook not "
                f"called: {sid}"
            )


        if (
            modifier
            .last_strength_error
            >
            1e-5
        ):

            raise RuntimeError(
                f"{name} magnitude "
                f"mismatch: {sid}"
            )


        predictions[name] = int(
            state[
                "best_numeral"
            ]
        )


        metadata[
            name
        ] = {
            "rho":
                float(
                    modifier
                    .last_rho
                ),

            "strength_error":
                float(
                    modifier
                    .last_strength_error
                ),
        }


    records.append(
        {
            "sample_id":
                sid,

            "condition":
                str(
                    row[
                        "condition"
                    ]
                ),

            "ground_truth":
                gt,

            "selected_alpha":
                alpha,

            "top4_prediction":
                predictions[
                    "top4"
                ],

            "bottom4_prediction":
                predictions[
                    "bottom4"
                ],

            "b2_top4_prediction":
                int(
                    b2_prediction_map[
                        sid
                    ]
                ),

            "top4_rho":
                metadata[
                    "top4"
                ][
                    "rho"
                ],

            "bottom4_rho":
                metadata[
                    "bottom4"
                ][
                    "rho"
                ],

            "top4_strength_error":
                metadata[
                    "top4"
                ][
                    "strength_error"
                ],

            "bottom4_strength_error":
                metadata[
                    "bottom4"
                ][
                    "strength_error"
                ],
        }
    )


    if (
        len(records)
        % 20
        == 0
        or
        len(records)
        == EXPECTED_N
    ):

        save_partial()


# ============================================================
# RESULTS
# ============================================================

result = pd.DataFrame(
    records
)


if len(result) != EXPECTED_N:

    raise RuntimeError(
        "Incomplete B3a result."
    )


gt = result[
    "ground_truth"
].astype(int)


result[
    "top4_correct"
] = (
    result[
        "top4_prediction"
    ].astype(int)
    ==
    gt
)


result[
    "bottom4_correct"
] = (
    result[
        "bottom4_prediction"
    ].astype(int)
    ==
    gt
)


result[
    "top4_b2_match"
] = (
    result[
        "top4_prediction"
    ].astype(int)
    ==
    result[
        "b2_top4_prediction"
    ].astype(int)
)


result.to_csv(
    FINAL_CSV,
    index=False,
)


top_n = int(
    result[
        "top4_correct"
    ].sum()
)


bottom_n = int(
    result[
        "bottom4_correct"
    ].sum()
)


difference = (
    result[
        "top4_correct"
    ].astype(int)
    -
    result[
        "bottom4_correct"
    ].astype(int)
).to_numpy(
    dtype=np.float64
)


effect = float(
    difference.mean()
)


ci_low, ci_high = (
    paired_bootstrap_ci(
        difference
    )
)


mcnemar = exact_mcnemar(
    result[
        "bottom4_correct"
    ],
    result[
        "top4_correct"
    ],
)


if ci_low > 0:

    status = (
        "TOP4_CLEAR_SPECIFICITY"
    )

elif ci_high < 0:

    status = (
        "TOP4_WORSE_THAN_BOTTOM4"
    )

else:

    status = (
        "TOP4_NO_CLEAR_SPECIFICITY"
    )


b2_match_rate = float(
    result[
        "top4_b2_match"
    ].mean()
)


# ============================================================
# GROUP SUMMARIES
# ============================================================

def summarize_group(
    group,
):

    n = len(group)

    top = int(
        group[
            "top4_correct"
        ].sum()
    )

    bottom = int(
        group[
            "bottom4_correct"
        ].sum()
    )

    return {
        "n":
            n,

        "top4_accuracy":
            top / n,

        "bottom4_accuracy":
            bottom / n,

        "top4_minus_bottom4_pp":
            100.0
            * (
                top - bottom
            )
            / n,

        "top4_median_rho":
            float(
                group[
                    "top4_rho"
                ].median()
            ),

        "bottom4_median_rho":
            float(
                group[
                    "bottom4_rho"
                ].median()
            ),
    }


action_rows = []


for alpha, group in (
    result
    .groupby(
        "selected_alpha",
        sort=True,
    )
):

    row = {
        "selected_alpha":
            float(alpha)
    }

    row.update(
        summarize_group(
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
    result
    .groupby(
        "condition",
        sort=True,
    )
):

    row = {
        "condition":
            str(condition)
    }

    row.update(
        summarize_group(
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
# FINAL RESULT
# ============================================================

final = {
    "stage":
        "AROMA2_CSA_GATE_B3A_BOTTOM_V1",

    "status":
        status,

    "git_head":
        git_head(),

    "method_fingerprint":
        METHOD_FINGERPRINT,

    "n":
        EXPECTED_N,

    "top4_correct":
        top_n,

    "bottom4_correct":
        bottom_n,

    "top4_accuracy":
        top_n
        / EXPECTED_N,

    "bottom4_accuracy":
        bottom_n
        / EXPECTED_N,

    "top4_minus_bottom4_pp":
        100.0
        * effect,

    "paired_bootstrap_95ci_pp":
        [
            100.0
            * ci_low,

            100.0
            * ci_high,
        ],

    "mcnemar":
        mcnemar,

    "b2_top4_prediction_match_rate":
        b2_match_rate,

    "median_top4_rho":
        float(
            result[
                "top4_rho"
            ].median()
        ),

    "median_bottom4_rho":
        float(
            result[
                "bottom4_rho"
            ].median()
        ),

    "top4_projector_reconstruction_error":
        projector_error,

    "top_bottom_overlap":
        cross_overlap,

    "max_top_strength_error":
        float(
            result[
                "top4_strength_error"
            ].max()
        ),

    "max_bottom_strength_error":
        float(
            result[
                "bottom4_strength_error"
            ].max()
        ),
}


RESULT_JSON.write_text(
    json.dumps(
        final,
        indent=2,
    ),
    encoding="utf-8",
)


pd.DataFrame([
    final
]).to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 96)
print(
    "FORMAL CSA GATE B3a RESULT"
)
print("=" * 96)


print(
    "Top-4:",
    top_n,
    "/",
    EXPECTED_N,
    f"= {100*top_n/EXPECTED_N:.3f}%"
)


print(
    "Bottom-4:",
    bottom_n,
    "/",
    EXPECTED_N,
    f"= {100*bottom_n/EXPECTED_N:.3f}%"
)


print()


print(
    "Top-4 - Bottom-4:",
    f"{100*effect:+.3f} pp"
)


print(
    "Paired bootstrap 95% CI:",
    f"[{100*ci_low:+.3f}, "
    f"{100*ci_high:+.3f}] pp"
)


print(
    "Exact McNemar p:",
    mcnemar[
        "p"
    ]
)


print()


print(
    "B2 Top-4 replay match:",
    b2_match_rate
)


print(
    "Median Top-4 rho:",
    final[
        "median_top4_rho"
    ]
)


print(
    "Median Bottom-4 rho:",
    final[
        "median_bottom4_rho"
    ]
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

