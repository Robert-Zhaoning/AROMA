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

STRENGTH_CSV = Path(
    "outputs/aroma2/"
    "csa_actuation_strength_v1/"
    "actuation_strength.csv"
)

FREEZE_DOC = Path(
    "docs/AROMA2_CSA_GATEB2_MN_FREEZE_v1.md"
)

OUT_DIR = Path(
    "outputs/aroma2/csa_gate_b2_mn_v1"
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
    / "gate_b2_mn_results.csv"
)

SUMMARY_CSV = (
    OUT_DIR
    / "gate_b2_mn_summary.csv"
)

BY_ACTION_CSV = (
    OUT_DIR
    / "gate_b2_mn_by_action.csv"
)

BY_CONDITION_CSV = (
    OUT_DIR
    / "gate_b2_mn_by_condition.csv"
)

RESULT_JSON = (
    OUT_DIR
    / "gate_b2_mn_result.json"
)


BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260915

DUMMY_GT = 1

START = 1664
END = 1792

EXPECTED_N = 653

EXPECTED_ACTION_COUNTS = {
    0.0: 24,
    1.5: 32,
    2.0: 63,
    4.0: 534,
}


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(
    path,
):

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

            h.update(
                chunk
            )

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


def resolve_image(
    sample_id,
):

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

    if len(
        candidates
    ) != 1:

        raise RuntimeError(
            f"{sample_id}: expected "
            f"exactly one image; "
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

    n = len(
        difference
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

    while cursor < (
        BOOTSTRAP_REPS
    ):

        b = min(
            batch_size,
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


def exact_mcnemar(
    reference_correct,
    candidate_correct,
):

    reference_correct = np.asarray(
        reference_correct,
        dtype=bool,
    )

    candidate_correct = np.asarray(
        candidate_correct,
        dtype=bool,
    )

    ref_wrong_candidate_right = int(
        (
            (~reference_correct)
            &
            candidate_correct
        ).sum()
    )

    ref_right_candidate_wrong = int(
        (
            reference_correct
            &
            (~candidate_correct)
        ).sum()
    )

    discordant = (
        ref_wrong_candidate_right
        +
        ref_right_candidate_wrong
    )

    if discordant == 0:

        p = 1.0

    else:

        p = float(
            binomtest(
                min(
                    ref_wrong_candidate_right,
                    ref_right_candidate_wrong,
                ),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "reference_wrong_candidate_correct":
            ref_wrong_candidate_right,

        "reference_correct_candidate_wrong":
            ref_right_candidate_wrong,

        "discordant":
            discordant,

        "p":
            p,
    }


# ============================================================
# CSA MODIFIERS
# ============================================================

class BaseSubspaceModifier:

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
        self.last_target_shift_norm = None
        self.last_strength_relative_error = None


    def calculate_delta(
        self,
        h,
        projected,
        h_norm,
        projected_norm,
    ):

        raise NotImplementedError


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

        h_norm = torch.linalg.vector_norm(
            h,
            dim=-1,
            keepdim=True,
        )

        projected_norm = (
            torch.linalg.vector_norm(
                projected,
                dim=-1,
                keepdim=True,
            )
        )

        if float(
            projected_norm.min().item()
        ) <= 1e-12:

            raise RuntimeError(
                "Projected h norm "
                "too small for stable "
                "subspace actuation."
            )

        delta_h = self.calculate_delta(
            h=h,
            projected=projected,
            h_norm=h_norm,
            projected_norm=
                projected_norm,
        )

        target_shift_norm = (
            abs(
                self.alpha
                - 1.0
            )
            * h_norm
        )

        actual_shift_norm = (
            torch.linalg.vector_norm(
                delta_h,
                dim=-1,
                keepdim=True,
            )
        )

        rho = (
            projected_norm
            / h_norm
        )

        denominator = torch.clamp(
            target_shift_norm,
            min=1e-12,
        )

        strength_relative_error = (
            torch.abs(
                actual_shift_norm
                -
                target_shift_norm
            )
            / denominator
        )

        self.last_h_norm = float(
            h_norm[
                0,
                0
            ].item()
        )

        self.last_projected_norm = (
            float(
                projected_norm[
                    0,
                    0
                ].item()
            )
        )

        self.last_rho = float(
            rho[
                0,
                0
            ].item()
        )

        self.last_shift_norm = float(
            actual_shift_norm[
                0,
                0
            ].item()
        )

        self.last_target_shift_norm = (
            float(
                target_shift_norm[
                    0,
                    0
                ].item()
            )
        )

        self.last_strength_relative_error = (
            float(
                strength_relative_error[
                    0,
                    0
                ].item()
            )
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

        if len(
            inputs
        ) == 1:

            return (
                x_new,
            )

        return (
            x_new,
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


class SameAlphaCSAModifier(
    BaseSubspaceModifier
):

    def calculate_delta(
        self,
        h,
        projected,
        h_norm,
        projected_norm,
    ):

        return (
            self.alpha
            - 1.0
        ) * projected


class MagnitudeNormalizedCSAModifier(
    BaseSubspaceModifier
):

    def calculate_delta(
        self,
        h,
        projected,
        h_norm,
        projected_norm,
    ):

        direction = (
            projected
            / projected_norm
        )

        return (
            (
                self.alpha
                - 1.0
            )
            * h_norm
            * direction
        )


# ============================================================
# LOAD POPULATION
# ============================================================

print("=" * 96)
print(
    "AROMA 2.0 — FORMAL "
    "MN-CSA GATE B2 v1"
)
print("=" * 96)


for required in [
    V3_RESULTS,
    U4_PATH,
    STRENGTH_CSV,
    FREEZE_DOC,
]:

    if not required.exists():

        raise RuntimeError(
            f"Missing required "
            f"artifact: {required}"
        )


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


if len(
    data
) != EXPECTED_N:

    raise RuntimeError(
        f"Expected {EXPECTED_N} "
        f"non-NOOP samples, "
        f"got {len(data)}"
    )


if (
    data[
        "sample_id"
    ]
    .astype(str)
    .nunique()
    != EXPECTED_N
):

    raise RuntimeError(
        "Non-NOOP sample IDs "
        "are not unique."
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


if counts != (
    EXPECTED_ACTION_COUNTS
):

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

    resolve_image(
        sid
    )


print(
    "PASS: frozen N=653 "
    "non-NOOP population"
)

print(
    "PASS: frozen action counts:",
    counts
)


# ============================================================
# STRENGTH AUDIT PROVENANCE
# ============================================================

strength = pd.read_csv(
    STRENGTH_CSV
)


if len(
    strength
) != EXPECTED_N:

    raise RuntimeError(
        "Strength audit N mismatch."
    )


strength_median_rho = float(
    strength[
        "csa_to_whole_shift_ratio"
    ].median()
)


strength_median_energy = float(
    strength[
        "projected_activation_energy_share"
    ].median()
)


print(
    "Strength-audit median rho:",
    strength_median_rho
)

print(
    "Strength-audit median "
    "energy share:",
    strength_median_energy
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
        f"Bad U4 shape: {U4.shape}"
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


if orth_error > 1e-4:

    raise RuntimeError(
        "U4 orthogonality "
        "audit failed."
    )


print(
    "PASS: U4 orthogonality"
)

print(
    "U4 max orth error:",
    orth_error
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

    "strength_csv_sha256":
        sha256_file(
            STRENGTH_CSV
        ),

    "model_revision":
        MODEL_REVISION,

    "n":
        EXPECTED_N,

    "rank":
        4,

    "actions":
        sorted(
            EXPECTED_ACTION_COUNTS
            .keys()
        ),

    "mn_geometry":
        (
            "delta=(alpha-1)"
            "*||h||"
            "*normalize(P_U h)"
        ),

    "bootstrap_reps":
        BOOTSTRAP_REPS,

    "bootstrap_seed":
        BOOTSTRAP_SEED,
}


METHOD_FINGERPRINT = (
    hashlib.sha256(
        json.dumps(
            fingerprint_metadata,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
        .encode(
            "utf-8"
        )
    )
    .hexdigest()
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
        "Expanded numeral "
        "support != 0..15"
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
        device_map="auto",
        attn_implementation=
            "eager",
    )
)


print(
    "model.training after load:",
    model.training
)


print(
    "text dropout config:",
    getattr(
        model.config.text_config,
        "dropout",
        None,
    )
)


model.eval()


print(
    "model.training after eval:",
    model.training
)


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
            resolve_image(
                sid
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
# ALPHA=1 IDENTITY TESTS
# ============================================================

print()
print("=" * 96)
print(
    "ALPHA=1 IDENTITY TESTS"
)
print("=" * 96)


test_sid = str(
    data.iloc[
        0
    ][
        "sample_id"
    ]
)


test_inputs = get_inputs(
    test_sid
)


with torch.inference_mode():

    baseline_out = model(
        **test_inputs,
        use_cache=False,
        return_dict=True,
    )


for name, cls in [
    (
        "same_alpha_CSA",
        SameAlphaCSAModifier,
    ),
    (
        "MN_CSA",
        MagnitudeNormalizedCSAModifier,
    ),
]:

    modifier = cls(
        model=model,
        alpha=1.0,
        U=U4,
    )

    modifier.register()

    try:

        with torch.inference_mode():

            out = model(
                **test_inputs,
                use_cache=False,
                return_dict=True,
            )

    finally:

        modifier.remove()


    identity_diff = float(
        (
            baseline_out
            .logits
            .float()
            -
            out
            .logits
            .float()
        )
        .abs()
        .max()
        .item()
    )


    print(
        name,
        "max logit diff:",
        identity_diff
    )


    if identity_diff != 0.0:

        raise RuntimeError(
            f"{name} alpha=1 "
            f"is not identity."
        )


print(
    "PASS: both CSA variants "
    "are exact identity at alpha=1"
)


del baseline_out
del out
del test_inputs

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# RESUME
# ============================================================

records = []


if PARTIAL_CSV.exists():

    if not (
        PARTIAL_META.exists()
    ):

        raise RuntimeError(
            "Partial CSV exists "
            "without metadata."
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
        !=
        METHOD_FINGERPRINT
    ):

        raise RuntimeError(
            "\nRESUME REFUSED\n"
            "Method fingerprint "
            "does not match."
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


    expected_ids = (
        data[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )


    if observed_ids != (
        expected_ids[
            :len(
                observed_ids
            )
        ]
    ):

        raise RuntimeError(
            "Resume sample order "
            "mismatch."
        )


    print(
        "RESUME VALIDATED:",
        len(records),
        "/",
        EXPECTED_N,
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
                    EXPECTED_N,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# FORMAL SAME-RUN EVALUATION
# ============================================================

print()
print("=" * 96)
print(
    "FORMAL SAME-RUN "
    "WHOLE / CSA / MN-CSA"
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


    alpha = float(
        row[
            "selected_alpha"
        ]
    )


    gt = int(
        row[
            "ground_truth"
        ]
    )


    inputs = get_inputs(
        sid
    )


    # --------------------------------------------------------
    # CURRENT BASELINE
    # --------------------------------------------------------

    baseline_state = (
        score_state(
            model,
            inputs,
            numeral_ids,
            DUMMY_GT,
        )
    )


    baseline_prediction = int(
        baseline_state[
            "best_numeral"
        ]
    )


    # --------------------------------------------------------
    # CURRENT WHOLE-HEAD
    # --------------------------------------------------------

    whole = HeadGainModifier(
        model=model,
        layer_idx=LAYER,
        head_idx=HEAD,
        alpha=alpha,
    )

    whole.register()

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

        whole.remove()


    if whole.calls <= 0:

        raise RuntimeError(
            f"Whole-head hook "
            f"not called: {sid}"
        )


    whole_prediction = int(
        whole_state[
            "best_numeral"
        ]
    )


    # --------------------------------------------------------
    # CURRENT SAME-ALPHA CSA
    # --------------------------------------------------------

    same = SameAlphaCSAModifier(
        model=model,
        alpha=alpha,
        U=U4,
    )

    same.register()

    try:

        same_state = (
            score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        )

    finally:

        same.remove()


    if same.calls <= 0:

        raise RuntimeError(
            f"Same-alpha CSA "
            f"hook not called: {sid}"
        )


    same_prediction = int(
        same_state[
            "best_numeral"
        ]
    )


    # --------------------------------------------------------
    # CURRENT MAGNITUDE-NORMALIZED CSA
    # --------------------------------------------------------

    mn = (
        MagnitudeNormalizedCSAModifier(
            model=model,
            alpha=alpha,
            U=U4,
        )
    )

    mn.register()

    try:

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


    if mn.calls <= 0:

        raise RuntimeError(
            f"MN-CSA hook "
            f"not called: {sid}"
        )


    mn_prediction = int(
        mn_state[
            "best_numeral"
        ]
    )


    # --------------------------------------------------------
    # HARD MAGNITUDE ASSERTION
    # --------------------------------------------------------

    if (
        mn.last_strength_relative_error
        is None
    ):

        raise RuntimeError(
            "MN strength metadata "
            "was not captured."
        )


    if (
        mn.last_strength_relative_error
        > 1e-5
    ):

        raise RuntimeError(
            f"MN shift magnitude "
            f"mismatch for {sid}: "
            f"{mn.last_strength_relative_error}"
        )


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

            "baseline_prediction":
                baseline_prediction,

            "whole_prediction":
                whole_prediction,

            "same_csa_prediction":
                same_prediction,

            "mn_csa_prediction":
                mn_prediction,

            "baseline_expected_numeral":
                float(
                    baseline_state[
                        "expected_numeral"
                    ]
                ),

            "whole_expected_numeral":
                float(
                    whole_state[
                        "expected_numeral"
                    ]
                ),

            "same_csa_expected_numeral":
                float(
                    same_state[
                        "expected_numeral"
                    ]
                ),

            "mn_csa_expected_numeral":
                float(
                    mn_state[
                        "expected_numeral"
                    ]
                ),

            "same_rho":
                float(
                    same.last_rho
                ),

            "mn_rho":
                float(
                    mn.last_rho
                ),

            "mn_h_norm":
                float(
                    mn.last_h_norm
                ),

            "mn_projected_norm":
                float(
                    mn.last_projected_norm
                ),

            "mn_shift_norm":
                float(
                    mn.last_shift_norm
                ),

            "mn_target_shift_norm":
                float(
                    mn.last_target_shift_norm
                ),

            "mn_strength_relative_error":
                float(
                    mn.last_strength_relative_error
                ),
        }
    )


    if (
        len(
            records
        )
        % 20
        == 0
        or
        len(
            records
        )
        == EXPECTED_N
    ):

        save_partial()


# ============================================================
# FINALIZE TABLE
# ============================================================

result = pd.DataFrame(
    records
)


if len(
    result
) != EXPECTED_N:

    raise RuntimeError(
        "Incomplete MN-CSA run."
    )


gt = (
    result[
        "ground_truth"
    ]
    .astype(int)
)


for method in [
    "baseline",
    "whole",
    "same_csa",
    "mn_csa",
]:

    result[
        f"{method}_correct"
    ] = (
        result[
            f"{method}_prediction"
        ]
        .astype(int)
        ==
        gt
    )


for method in [
    "whole",
    "same_csa",
    "mn_csa",
]:

    result[
        f"{method}_repair"
    ] = (
        (~result[
            "baseline_correct"
        ])
        &
        result[
            f"{method}_correct"
        ]
    )


    result[
        f"{method}_break"
    ] = (
        result[
            "baseline_correct"
        ]
        &
        (~result[
            f"{method}_correct"
        ])
    )


result.to_csv(
    FINAL_CSV,
    index=False,
)


# ============================================================
# METHOD SUMMARY
# ============================================================

def method_summary(
    method,
):

    correct = int(
        result[
            f"{method}_correct"
        ].sum()
    )

    if method == "baseline":

        return {
            "correct":
                correct,

            "accuracy":
                correct
                / EXPECTED_N,
        }


    repairs = int(
        result[
            f"{method}_repair"
        ].sum()
    )

    breaks = int(
        result[
            f"{method}_break"
        ].sum()
    )


    return {
        "correct":
            correct,

        "accuracy":
            correct
            / EXPECTED_N,

        "repairs":
            repairs,

        "breaks":
            breaks,

        "net":
            repairs
            -
            breaks,
    }


baseline_summary = (
    method_summary(
        "baseline"
    )
)

whole_summary = (
    method_summary(
        "whole"
    )
)

same_summary = (
    method_summary(
        "same_csa"
    )
)

mn_summary = (
    method_summary(
        "mn_csa"
    )
)


# ============================================================
# PAIRWISE TEST
# ============================================================

def pairwise_test(
    reference,
    candidate,
):

    ref = (
        result[
            f"{reference}_correct"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    cand = (
        result[
            f"{candidate}_correct"
        ]
        .to_numpy(
            dtype=bool
        )
    )


    difference = (
        cand.astype(int)
        -
        ref.astype(int)
    )


    effect = float(
        difference.mean()
    )


    ci_low, ci_high = (
        paired_bootstrap_ci(
            difference
        )
    )


    mc = exact_mcnemar(
        ref,
        cand,
    )


    return {
        "reference":
            reference,

        "candidate":
            candidate,

        "difference":
            effect,

        "difference_pp":
            100.0
            * effect,

        "bootstrap_ci":
            [
                ci_low,
                ci_high,
            ],

        "bootstrap_ci_pp":
            [
                100.0
                * ci_low,

                100.0
                * ci_high,
            ],

        "mcnemar":
            mc,
    }


mn_vs_whole = (
    pairwise_test(
        "whole",
        "mn_csa",
    )
)


mn_vs_same = (
    pairwise_test(
        "same_csa",
        "mn_csa",
    )
)


same_vs_whole = (
    pairwise_test(
        "whole",
        "same_csa",
    )
)


# ============================================================
# RECOVERY FRACTION
# ============================================================

denominator = (
    whole_summary[
        "accuracy"
    ]
    -
    same_summary[
        "accuracy"
    ]
)


numerator = (
    mn_summary[
        "accuracy"
    ]
    -
    same_summary[
        "accuracy"
    ]
)


if abs(
    denominator
) < 1e-12:

    recovery_fraction = None

else:

    recovery_fraction = float(
        numerator
        / denominator
    )


# ============================================================
# PRE-SPECIFIED STATUS
# ============================================================

whole_ci_low = (
    mn_vs_whole[
        "bootstrap_ci"
    ][0]
)

whole_ci_high = (
    mn_vs_whole[
        "bootstrap_ci"
    ][1]
)

same_ci_low = (
    mn_vs_same[
        "bootstrap_ci"
    ][0]
)

same_ci_high = (
    mn_vs_same[
        "bootstrap_ci"
    ][1]
)


if whole_ci_low > 0:

    status = (
        "MN_OUTPERFORMS_WHOLE"
    )

elif (
    same_ci_low > 0
    and
    whole_ci_low <= 0
    and
    whole_ci_high >= 0
):

    status = (
        "MN_COMPETITIVE_RECOVERY"
    )

elif (
    same_ci_low > 0
    and
    whole_ci_high < 0
):

    status = (
        "MN_PARTIAL_RECOVERY"
    )

elif same_ci_high < 0:

    status = (
        "MN_WORSE"
    )

else:

    status = (
        "MN_NO_CLEAR_RECOVERY"
    )


# ============================================================
# GROUP TABLE
# ============================================================

def group_summary(
    group,
):

    out = {
        "n":
            len(
                group
            )
    }


    for method in [
        "baseline",
        "whole",
        "same_csa",
        "mn_csa",
    ]:

        correct = int(
            group[
                f"{method}_correct"
            ].sum()
        )

        out[
            f"{method}_accuracy"
        ] = (
            correct
            / len(
                group
            )
        )


        if method != "baseline":

            repairs = int(
                group[
                    f"{method}_repair"
                ].sum()
            )

            breaks = int(
                group[
                    f"{method}_break"
                ].sum()
            )

            out[
                f"{method}_repairs"
            ] = repairs

            out[
                f"{method}_breaks"
            ] = breaks

            out[
                f"{method}_net"
            ] = (
                repairs
                -
                breaks
            )


    out[
        "mn_minus_whole_pp"
    ] = (
        100.0
        * (
            out[
                "mn_csa_accuracy"
            ]
            -
            out[
                "whole_accuracy"
            ]
        )
    )


    out[
        "mn_minus_same_pp"
    ] = (
        100.0
        * (
            out[
                "mn_csa_accuracy"
            ]
            -
            out[
                "same_csa_accuracy"
            ]
        )
    )


    return out


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
            float(
                alpha
            )
    }

    row.update(
        group_summary(
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
            str(
                condition
            )
    }

    row.update(
        group_summary(
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
# FULL-POLICY DIFFERENCE
#
# The remaining 1347 alpha=1 samples are identical under
# whole-head, same-CSA, and MN-CSA. Therefore count
# differences on the 653 intervention subset translate
# exactly to full-policy percentage-point differences by
# dividing by 2000.
# ============================================================

mn_minus_whole_count = (
    mn_summary[
        "correct"
    ]
    -
    whole_summary[
        "correct"
    ]
)


mn_minus_same_count = (
    mn_summary[
        "correct"
    ]
    -
    same_summary[
        "correct"
    ]
)


full_policy_mn_minus_whole_pp = (
    100.0
    * mn_minus_whole_count
    / 2000.0
)


full_policy_mn_minus_same_pp = (
    100.0
    * mn_minus_same_count
    / 2000.0
)


# ============================================================
# FINAL RESULT
# ============================================================

final = {
    "stage":
        "AROMA2_CSA_GATE_B2_MN_V1",

    "status":
        status,

    "git_head":
        git_head(),

    "method_fingerprint":
        METHOD_FINGERPRINT,

    "n_intervention_samples":
        EXPECTED_N,

    "baseline":
        baseline_summary,

    "whole":
        whole_summary,

    "same_alpha_csa":
        same_summary,

    "mn_csa":
        mn_summary,

    "mn_vs_whole":
        mn_vs_whole,

    "mn_vs_same_alpha_csa":
        mn_vs_same,

    "same_alpha_csa_vs_whole":
        same_vs_whole,

    "recovery_fraction":
        recovery_fraction,

    "full_policy_mn_minus_whole_pp":
        full_policy_mn_minus_whole_pp,

    "full_policy_mn_minus_same_pp":
        full_policy_mn_minus_same_pp,

    "median_rho_current_run":
        float(
            result[
                "mn_rho"
            ].median()
        ),

    "max_mn_strength_relative_error":
        float(
            result[
                "mn_strength_relative_error"
            ].max()
        ),

    "gate_a_u4_sha256":
        sha256_file(
            U4_PATH
        ),

    "strength_audit_sha256":
        sha256_file(
            STRENGTH_CSV
        ),
}


RESULT_JSON.write_text(
    json.dumps(
        final,
        indent=2,
    ),
    encoding="utf-8",
)


pd.DataFrame(
    [
        {
            "status":
                status,

            "baseline_accuracy":
                baseline_summary[
                    "accuracy"
                ],

            "whole_accuracy":
                whole_summary[
                    "accuracy"
                ],

            "same_csa_accuracy":
                same_summary[
                    "accuracy"
                ],

            "mn_csa_accuracy":
                mn_summary[
                    "accuracy"
                ],

            "mn_minus_whole_pp":
                mn_vs_whole[
                    "difference_pp"
                ],

            "mn_minus_same_pp":
                mn_vs_same[
                    "difference_pp"
                ],

            "recovery_fraction":
                recovery_fraction,

            "full_policy_mn_minus_whole_pp":
                full_policy_mn_minus_whole_pp,

            "whole_repairs":
                whole_summary[
                    "repairs"
                ],

            "whole_breaks":
                whole_summary[
                    "breaks"
                ],

            "same_repairs":
                same_summary[
                    "repairs"
                ],

            "same_breaks":
                same_summary[
                    "breaks"
                ],

            "mn_repairs":
                mn_summary[
                    "repairs"
                ],

            "mn_breaks":
                mn_summary[
                    "breaks"
                ],
        }
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
    "FORMAL MN-CSA GATE B2 RESULT"
)
print("=" * 96)


print(
    "N intervention samples:",
    EXPECTED_N
)


print()

print(
    "Baseline:",
    baseline_summary
)

print(
    "Whole-head:",
    whole_summary
)

print(
    "Same-alpha CSA:",
    same_summary
)

print(
    "MN-CSA:",
    mn_summary
)


print()
print(
    "MN - whole:"
)

print(
    f"  {mn_vs_whole['difference_pp']:+.3f} pp"
)

print(
    "  bootstrap 95% CI:",
    [
        round(
            x,
            3
        )
        for x in (
            mn_vs_whole[
                "bootstrap_ci_pp"
            ]
        )
    ],
)

print(
    "  McNemar p:",
    mn_vs_whole[
        "mcnemar"
    ][
        "p"
    ]
)


print()
print(
    "MN - same-alpha CSA:"
)

print(
    f"  {mn_vs_same['difference_pp']:+.3f} pp"
)

print(
    "  bootstrap 95% CI:",
    [
        round(
            x,
            3
        )
        for x in (
            mn_vs_same[
                "bootstrap_ci_pp"
            ]
        )
    ],
)

print(
    "  McNemar p:",
    mn_vs_same[
        "mcnemar"
    ][
        "p"
    ]
)


print()
print(
    "Recovery fraction:",
    recovery_fraction
)


print(
    "Full-policy MN - whole:",
    f"{full_policy_mn_minus_whole_pp:+.3f} pp"
)


print(
    "Full-policy MN - same CSA:",
    f"{full_policy_mn_minus_same_pp:+.3f} pp"
)


print()
print(
    "Whole repairs / breaks / net:",
    whole_summary[
        "repairs"
    ],
    "/",
    whole_summary[
        "breaks"
    ],
    "/",
    whole_summary[
        "net"
    ],
)


print(
    "Same repairs / breaks / net:",
    same_summary[
        "repairs"
    ],
    "/",
    same_summary[
        "breaks"
    ],
    "/",
    same_summary[
        "net"
    ],
)


print(
    "MN repairs / breaks / net:",
    mn_summary[
        "repairs"
    ],
    "/",
    mn_summary[
        "breaks"
    ],
    "/",
    mn_summary[
        "net"
    ],
)


print()
print(
    "Median rho current run:",
    final[
        "median_rho_current_run"
    ]
)


print(
    "Max MN magnitude "
    "relative error:",
    final[
        "max_mn_strength_relative_error"
    ]
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
    )
    .to_string(
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
    )
    .to_string(
        index=False
    )
)


print()
print(
    "Result:",
    RESULT_JSON
)

print("=" * 96)

