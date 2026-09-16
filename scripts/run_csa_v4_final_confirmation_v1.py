import gc
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
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


# ============================================================
# CANONICAL IMPLEMENTATIONS
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
    numeral_token_ids,
    score_state,
)

from run_proc_count_causal_v3_final import (
    state_to_features,
    controller_decision,
    exact_mcnemar_p,
)


# ============================================================
# FROZEN CONFIG
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

V4_META = Path(
    "data/proc_count_causal_v4/"
    "metadata.jsonl"
)

V4_MANIFEST = Path(
    "outputs/proc_count_causal_v4/"
    "v4_generation_manifest.json"
)

V4_PROTOCOL = Path(
    "configs/"
    "proc_count_causal_v4_"
    "csa_final_confirmation.json"
)

FREEZE_DOC = Path(
    "docs/"
    "AROMA2_CSA_V4_CONFIRMATION_FREEZE_v1.md"
)

RUNNER_FREEZE_DOC = Path(
    "docs/"
    "AROMA2_CSA_V4_RUNNER_FREEZE_v1.md"
)

CONTROLLER_BUNDLE = Path(
    "outputs/proc_count_causal_v2/"
    "controller/final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

CONTROLLER_MANIFEST = Path(
    "configs/"
    "aroma_cardinality_controller_frozen.json"
)

CONTROLLER_FREEZE_RECORD = Path(
    "configs/"
    "aroma_controller_freeze_record.json"
)

U4_PATH = Path(
    "outputs/aroma2/"
    "csa_gate_a_v3/"
    "U4_primary.npy"
)

RENDERER_PATH = Path(
    "scripts/"
    "generate_proc_count_causal_v3.py"
)

OUT_DIR = Path(
    "outputs/aroma2/"
    "csa_v4_final_confirmation"
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

RESULT_CSV = (
    OUT_DIR
    / "v4_final_results.csv"
)

SUMMARY_CSV = (
    OUT_DIR
    / "v4_final_summary.csv"
)

ACTION_CSV = (
    OUT_DIR
    / "v4_action_distribution.csv"
)

CONDITION_CSV = (
    OUT_DIR
    / "v4_by_condition.csv"
)

COUNT_CSV = (
    OUT_DIR
    / "v4_by_count.csv"
)

RUN_META_JSON = (
    OUT_DIR
    / "v4_run_metadata.json"
)

RESULT_JSON = (
    OUT_DIR
    / "v4_final_result.json"
)

STARTED_JSON = (
    OUT_DIR
    / "v4_run_started.json"
)


EXPECTED_N = 2000

EXPECTED_ACTIONS = {
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
}

DUMMY_GT = 1

START = 1664
END = 1792

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260916

CONTROLLER_FREEZE_COMMIT = (
    "c8c5ede1601b1ada4087e141d61247903a4458a7"
)


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


def git_is_ancestor(
    commit,
):

    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
        ],
        stdout=
            subprocess.DEVNULL,
        stderr=
            subprocess.DEVNULL,
    )

    return (
        result.returncode
        == 0
    )


def load_json(
    path,
):

    return json.loads(
        Path(path)
        .read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(
    path,
):

    return [
        json.loads(line)
        for line in (
            Path(path)
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    ]


def paired_bootstrap_ci(
    differences,
):

    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    if len(
        differences
    ) != EXPECTED_N:

        raise RuntimeError(
            "Bootstrap expected "
            "2000 paired values."
        )

    rng = (
        np.random
        .default_rng(
            BOOTSTRAP_SEED
        )
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

        current = min(
            batch_size,
            BOOTSTRAP_REPS
            -
            cursor,
        )

        indices = rng.integers(
            0,
            EXPECTED_N,
            size=(
                current,
                EXPECTED_N,
            ),
        )

        boot[
            cursor:
            cursor
            +
            current
        ] = (
            differences[
                indices
            ]
            .mean(
                axis=1
            )
        )

        cursor += current

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


def pairwise_stats(
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

    if (
        len(
            reference_correct
        )
        != EXPECTED_N
        or
        len(
            candidate_correct
        )
        != EXPECTED_N
    ):

        raise RuntimeError(
            "Pairwise stats require "
            "2000 examples."
        )

    reference_wrong_candidate_correct = (
        int(
            (
                (~reference_correct)
                &
                candidate_correct
            )
            .sum()
        )
    )

    reference_correct_candidate_wrong = (
        int(
            (
                reference_correct
                &
                (~candidate_correct)
            )
            .sum()
        )
    )

    difference = (
        candidate_correct
        .astype(int)
        -
        reference_correct
        .astype(int)
    )

    effect = float(
        difference.mean()
    )

    ci_low, ci_high = (
        paired_bootstrap_ci(
            difference
        )
    )

    p_value = (
        exact_mcnemar_p(
            reference_wrong_candidate_correct,
            reference_correct_candidate_wrong,
        )
    )

    return {
        "difference":
            effect,

        "difference_pp":
            100.0
            * effect,

        "bootstrap_ci95":
            [
                ci_low,
                ci_high,
            ],

        "bootstrap_ci95_pp":
            [
                100.0
                * ci_low,

                100.0
                * ci_high,
            ],

        "reference_wrong_candidate_correct":
            reference_wrong_candidate_correct,

        "reference_correct_candidate_wrong":
            reference_correct_candidate_wrong,

        "mcnemar_exact_p":
            float(
                p_value
            ),
    }


# ============================================================
# FROZEN MN-CSA
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

        if hidden_size != 4096:

            raise RuntimeError(
                f"Unexpected "
                f"hidden size: "
                f"{hidden_size}"
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
                f"Expected U "
                f"(128,4), "
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

        self.last_strength_error = (
            float(
                relative_error[
                    0,
                    0
                ]
                .item()
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
# PRE-INFERENCE ARTIFACT AUDIT
# ============================================================

print(
    "=" * 100
)

print(
    "AROMA 2.0 — FINAL UNTOUCHED "
    "V4 MN-CSA CONFIRMATION"
)

print(
    "=" * 100
)


required_paths = [
    V4_META,
    V4_MANIFEST,
    V4_PROTOCOL,
    FREEZE_DOC,
    RUNNER_FREEZE_DOC,
    CONTROLLER_BUNDLE,
    CONTROLLER_MANIFEST,
    CONTROLLER_FREEZE_RECORD,
    U4_PATH,
    RENDERER_PATH,
]


for path in (
    required_paths
):

    if not path.exists():

        raise RuntimeError(
            f"Missing required "
            f"artifact: {path}"
        )


if not git_is_ancestor(
    CONTROLLER_FREEZE_COMMIT
):

    raise RuntimeError(
        "Frozen controller commit "
        "is not an ancestor of HEAD."
    )


current_head = git_head()


print(
    "Git HEAD:",
    current_head
)


v4_manifest = load_json(
    V4_MANIFEST
)


current_metadata_sha = (
    sha256_file(
        V4_META
    )
)


current_renderer_sha = (
    sha256_file(
        RENDERER_PATH
    )
)


if (
    current_metadata_sha
    !=
    v4_manifest[
        "metadata_sha256"
    ]
):

    raise RuntimeError(
        "V4 metadata changed "
        "after generation."
    )


if (
    current_renderer_sha
    !=
    v4_manifest[
        "renderer_sha256"
    ]
):

    raise RuntimeError(
        "Renderer changed after "
        "V4 generation."
    )


if (
    int(
        v4_manifest[
            "n"
        ]
    )
    !=
    EXPECTED_N
):

    raise RuntimeError(
        "V4 manifest N mismatch."
    )


for key in [
    "v1",
    "v2",
    "v3",
]:

    if (
        int(
            v4_manifest[
                "seed_overlap"
            ][
                key
            ]
        )
        != 0
    ):

        raise RuntimeError(
            f"V4 seed overlap "
            f"with {key}."
        )


    if (
        int(
            v4_manifest[
                "image_hash_overlap"
            ][
                key
            ]
        )
        != 0
    ):

        raise RuntimeError(
            f"V4 image overlap "
            f"with {key}."
        )


print(
    "PASS: V4 manifest "
    "integrity"
)

print(
    "PASS: metadata SHA256"
)

print(
    "PASS: renderer SHA256"
)

print(
    "PASS: historical "
    "independence record"
)


# ============================================================
# LOAD V4 METADATA
# ============================================================

samples = load_jsonl(
    V4_META
)


if len(
    samples
) != EXPECTED_N:

    raise RuntimeError(
        f"Expected V4 N=2000, "
        f"got {len(samples)}"
    )


sample_ids = [
    str(
        x[
            "sample_id"
        ]
    )
    for x in samples
]


if len(
    set(
        sample_ids
    )
) != EXPECTED_N:

    raise RuntimeError(
        "V4 sample IDs "
        "not unique."
    )


for sample in samples:

    image_path = Path(
        sample[
            "image_path"
        ]
    )

    if not (
        image_path.exists()
    ):

        raise RuntimeError(
            f"Missing V4 image: "
            f"{image_path}"
        )


print(
    "PASS: exact V4 "
    "N=2000 population"
)


# ============================================================
# LOAD CONTROLLER
# ============================================================

controller_bundle_sha = (
    sha256_file(
        CONTROLLER_BUNDLE
    )
)


controller_manifest_sha = (
    sha256_file(
        CONTROLLER_MANIFEST
    )
)


controller_freeze_record_sha = (
    sha256_file(
        CONTROLLER_FREEZE_RECORD
    )
)


bundle = joblib.load(
    CONTROLLER_BUNDLE
)


if not isinstance(
    bundle,
    dict,
):

    raise RuntimeError(
        "Controller bundle "
        "is not a dict."
    )


if (
    "feature_names"
    not in bundle
):

    raise RuntimeError(
        "Controller bundle "
        "missing feature_names."
    )


feature_names = list(
    bundle[
        "feature_names"
    ]
)


if len(
    feature_names
) != 39:

    raise RuntimeError(
        f"Expected 39 controller "
        f"features, got "
        f"{len(feature_names)}"
    )


print(
    "PASS: frozen controller "
    "bundle loaded"
)

print(
    "Controller bundle SHA256:",
    controller_bundle_sha
)

print(
    "Feature count:",
    len(
        feature_names
    )
)


# ============================================================
# LOAD U4
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
            U4.T
            @ U4
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
    "PASS: U4 shape "
    "and orthogonality"
)

print(
    "U4 orthogonality "
    "max error:",
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

    "v4_metadata_sha256":
        current_metadata_sha,

    "v4_manifest_sha256":
        sha256_file(
            V4_MANIFEST
        ),

    "v4_protocol_sha256":
        sha256_file(
            V4_PROTOCOL
        ),

    "freeze_doc_sha256":
        sha256_file(
            FREEZE_DOC
        ),

    "runner_freeze_doc_sha256":
        sha256_file(
            RUNNER_FREEZE_DOC
        ),

    "controller_bundle_sha256":
        controller_bundle_sha,

    "controller_manifest_sha256":
        controller_manifest_sha,

    "controller_freeze_record_sha256":
        controller_freeze_record_sha,

    "u4_sha256":
        sha256_file(
            U4_PATH
        ),

    "renderer_sha256":
        current_renderer_sha,

    "model":
        MODEL_ID,

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

    "rank":
        4,

    "actions":
        sorted(
            EXPECTED_ACTIONS
        ),

    "bootstrap_reps":
        BOOTSTRAP_REPS,

    "bootstrap_seed":
        BOOTSTRAP_SEED,
}


METHOD_FINGERPRINT = (
    hashlib
    .sha256(
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
    "Method fingerprint:",
    METHOD_FINGERPRINT
)


# ============================================================
# RESUME AUDIT
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


    partial_meta = load_json(
        PARTIAL_META
    )


    if (
        partial_meta[
            "method_fingerprint"
        ]
        !=
        METHOD_FINGERPRINT
    ):

        raise RuntimeError(
            "\nRESUME REFUSED\n"
            "Method fingerprint "
            "changed."
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
        sample_ids[
            :len(
                observed_ids
            )
        ]
    ):

        raise RuntimeError(
            "Partial result "
            "sample order mismatch."
        )


    print(
        "RESUME VALIDATED:",
        len(
            records
        ),
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
# RECORD START BEFORE MODEL INFERENCE
# ============================================================

if not (
    STARTED_JSON.exists()
):

    STARTED_JSON.write_text(
        json.dumps(
            {
                "stage":
                    "AROMA2_CSA_V4_FINAL_CONFIRMATION",

                "started_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "git_head":
                    current_head,

                "method_fingerprint":
                    METHOD_FINGERPRINT,

                "v4_metadata_sha256":
                    current_metadata_sha,

                "controller_bundle_sha256":
                    controller_bundle_sha,

                "u4_sha256":
                    sha256_file(
                        U4_PATH
                    ),

                "model_revision":
                    MODEL_REVISION,

                "already_completed":
                    len(
                        records
                    ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# LOAD PROCESSOR + MODEL
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
        "Expanded numeral-token "
        "audit failed."
    )


print(
    "Numeral-token audit: PASS"
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
    "model.training "
    "after load:",
    model.training
)


model.eval()


print(
    "model.training "
    "after eval:",
    model.training
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )


# ============================================================
# ALPHA=1 IDENTITY PRECHECK
# ============================================================

print()
print(
    "=" * 100
)

print(
    "WHOLE + MN-CSA "
    "ALPHA=1 IDENTITY PRECHECK"
)

print(
    "=" * 100
)


identity_sample = samples[
    0
]


identity_image = (
    Image.open(
        identity_sample[
            "image_path"
        ]
    )
    .convert(
        "RGB"
    )
)


identity_inputs = (
    prepare_inputs(
        processor,
        identity_image,
    )
)


identity_inputs = (
    move_inputs(
        identity_inputs,
        model,
    )
)


with torch.inference_mode():

    baseline_identity = model(
        **identity_inputs,
        use_cache=False,
        return_dict=True,
    )


# Whole-head alpha=1
whole_identity = (
    HeadGainModifier(
        model=model,
        layer_idx=LAYER,
        head_idx=HEAD,
        alpha=1.0,
    )
)

whole_identity.register()


try:

    with torch.inference_mode():

        whole_identity_out = model(
            **identity_inputs,
            use_cache=False,
            return_dict=True,
        )

finally:

    whole_identity.remove()


whole_identity_diff = float(
    (
        baseline_identity
        .logits
        .float()
        -
        whole_identity_out
        .logits
        .float()
    )
    .abs()
    .max()
    .item()
)


# MN alpha=1
mn_identity = (
    MagnitudeNormalizedCSAModifier(
        model=model,
        alpha=1.0,
        U=U4,
    )
)

mn_identity.register()


try:

    with torch.inference_mode():

        mn_identity_out = model(
            **identity_inputs,
            use_cache=False,
            return_dict=True,
        )

finally:

    mn_identity.remove()


mn_identity_diff = float(
    (
        baseline_identity
        .logits
        .float()
        -
        mn_identity_out
        .logits
        .float()
    )
    .abs()
    .max()
    .item()
)


print(
    "Whole alpha=1 "
    "max logit diff:",
    whole_identity_diff
)

print(
    "MN alpha=1 "
    "max logit diff:",
    mn_identity_diff
)


if (
    whole_identity_diff
    != 0.0
):

    raise RuntimeError(
        "Whole-head alpha=1 "
        "is not exact identity."
    )


if (
    mn_identity_diff
    != 0.0
):

    raise RuntimeError(
        "MN-CSA alpha=1 "
        "is not exact identity."
    )


print(
    "PASS: both alpha=1 "
    "exact identity"
)


del baseline_identity
del whole_identity_out
del mn_identity_out
del identity_inputs
del identity_image

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# FINAL CONFIRMATION LOOP
# ============================================================

print()
print(
    "=" * 100
)

print(
    "FORMAL UNTOUCHED V4 "
    "BEHAVIORAL CONFIRMATION"
)

print(
    "=" * 100
)


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
    desc=
        "V4 final confirmation",
):

    sample = samples[
        idx
    ]


    sid = str(
        sample[
            "sample_id"
        ]
    )


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


    inputs = (
        prepare_inputs(
            processor,
            image,
        )
    )


    inputs = (
        move_inputs(
            inputs,
            model,
        )
    )


    # ========================================================
    # BASELINE
    #
    # Ground truth has NOT been read yet.
    # ========================================================

    with torch.inference_mode():

        baseline_state = (
            score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        )


    (
        X,
        feature_values,
    ) = state_to_features(
        baseline_state,
        feature_names,
    )


    (
        selected_alpha,
        selected_score,
        score_map,
    ) = controller_decision(
        bundle,
        X,
    )


    selected_alpha = float(
        selected_alpha
    )


    if selected_alpha not in (
        EXPECTED_ACTIONS
    ):

        raise RuntimeError(
            f"Unexpected controller "
            f"action {selected_alpha} "
            f"for {sid}"
        )


    baseline_prediction = int(
        baseline_state[
            "best_numeral"
        ]
    )


    # ========================================================
    # SAME FROZEN ACTION FOR BOTH ACTUATORS
    # ========================================================

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


    else:

        # ----------------------------------------------------
        # WHOLE-HEAD
        # ----------------------------------------------------

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
                f"Whole-head hook "
                f"not called for {sid}"
            )


        whole_prediction = int(
            whole_state[
                "best_numeral"
            ]
        )


        # ----------------------------------------------------
        # FROZEN MN-CSA
        # ----------------------------------------------------

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
                f"MN-CSA hook "
                f"not called for {sid}"
            )


        if (
            mn.last_strength_error
            >
            1e-5
        ):

            raise RuntimeError(
                f"MN-CSA magnitude "
                f"mismatch for {sid}: "
                f"{mn.last_strength_error}"
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


    # ========================================================
    # ONLY NOW READ GROUND TRUTH
    # ========================================================

    gt = int(
        sample[
            "ground_truth"
        ]
    )


    baseline_correct = bool(
        baseline_prediction
        ==
        gt
    )


    whole_correct = bool(
        whole_prediction
        ==
        gt
    )


    mn_correct = bool(
        mn_prediction
        ==
        gt
    )


    whole_repair = bool(
        (
            not baseline_correct
        )
        and
        whole_correct
    )


    whole_break = bool(
        baseline_correct
        and
        (
            not whole_correct
        )
    )


    mn_repair = bool(
        (
            not baseline_correct
        )
        and
        mn_correct
    )


    mn_break = bool(
        baseline_correct
        and
        (
            not mn_correct
        )
    )


    row = {
        "sample_id":
            sid,

        "condition":
            str(
                sample[
                    "condition"
                ]
            ),

        "replicate":
            int(
                sample[
                    "replicate"
                ]
            ),

        "seed":
            int(
                sample[
                    "seed"
                ]
            ),

        "ground_truth":
            gt,

        "baseline_prediction":
            baseline_prediction,

        "selected_alpha":
            selected_alpha,

        "selected_score":
            float(
                selected_score
            ),

        "whole_prediction":
            whole_prediction,

        "mn_prediction":
            mn_prediction,

        "baseline_correct":
            baseline_correct,

        "whole_correct":
            whole_correct,

        "mn_correct":
            mn_correct,

        "whole_repair":
            whole_repair,

        "whole_break":
            whole_break,

        "mn_repair":
            mn_repair,

        "mn_break":
            mn_break,

        "whole_hook_calls":
            whole_hook_calls,

        "mn_hook_calls":
            mn_hook_calls,

        "mn_rho":
            mn_rho,

        "mn_strength_error":
            mn_strength_error,

        "score_alpha_0":
            float(
                score_map[
                    0.0
                ]
            ),

        "score_alpha_1p5":
            float(
                score_map[
                    1.5
                ]
            ),

        "score_alpha_2":
            float(
                score_map[
                    2.0
                ]
            ),

        "score_alpha_4":
            float(
                score_map[
                    4.0
                ]
            ),
    }


    records.append(
        row
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
# FINAL TABLE
# ============================================================

result = pd.DataFrame(
    records
)


if len(
    result
) != EXPECTED_N:

    raise RuntimeError(
        "Final V4 result "
        "does not contain "
        "2000 examples."
    )


if (
    result[
        "sample_id"
    ]
    .astype(str)
    .tolist()
    !=
    sample_ids
):

    raise RuntimeError(
        "Final V4 sample "
        "ordering mismatch."
    )


result.to_csv(
    RESULT_CSV,
    index=False,
)


# ============================================================
# OVERALL METRICS
# ============================================================

baseline_correct = (
    result[
        "baseline_correct"
    ]
    .astype(bool)
    .to_numpy()
)


whole_correct = (
    result[
        "whole_correct"
    ]
    .astype(bool)
    .to_numpy()
)


mn_correct = (
    result[
        "mn_correct"
    ]
    .astype(bool)
    .to_numpy()
)


baseline_n = int(
    baseline_correct.sum()
)

whole_n = int(
    whole_correct.sum()
)

mn_n = int(
    mn_correct.sum()
)


baseline_accuracy = (
    baseline_n
    /
    EXPECTED_N
)

whole_accuracy = (
    whole_n
    /
    EXPECTED_N
)

mn_accuracy = (
    mn_n
    /
    EXPECTED_N
)


whole_repairs = int(
    result[
        "whole_repair"
    ]
    .astype(bool)
    .sum()
)


whole_breaks = int(
    result[
        "whole_break"
    ]
    .astype(bool)
    .sum()
)


mn_repairs = int(
    result[
        "mn_repair"
    ]
    .astype(bool)
    .sum()
)


mn_breaks = int(
    result[
        "mn_break"
    ]
    .astype(bool)
    .sum()
)


whole_net = (
    whole_repairs
    -
    whole_breaks
)


mn_net = (
    mn_repairs
    -
    mn_breaks
)


# ============================================================
# PAIRED COMPARISONS
# ============================================================

whole_vs_baseline = (
    pairwise_stats(
        baseline_correct,
        whole_correct,
    )
)


mn_vs_baseline = (
    pairwise_stats(
        baseline_correct,
        mn_correct,
    )
)


mn_vs_whole = (
    pairwise_stats(
        whole_correct,
        mn_correct,
    )
)


primary_success = bool(
    (
        mn_vs_baseline[
            "difference"
        ]
        >
        0
    )
    and
    (
        mn_vs_baseline[
            "mcnemar_exact_p"
        ]
        <
        0.05
    )
)


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

action_distribution = (
    result[
        "selected_alpha"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "selected_alpha"
    )
    .reset_index(
        name="count"
    )
)


action_distribution[
    "fraction"
] = (
    action_distribution[
        "count"
    ]
    /
    EXPECTED_N
)


action_distribution.to_csv(
    ACTION_CSV,
    index=False,
)


# ============================================================
# GROUP SUMMARIES
# ============================================================

def summarize_group(
    group,
):

    n = len(
        group
    )


    b = (
        group[
            "baseline_correct"
        ]
        .astype(bool)
    )


    w = (
        group[
            "whole_correct"
        ]
        .astype(bool)
    )


    m = (
        group[
            "mn_correct"
        ]
        .astype(bool)
    )


    wr = int(
        (
            (~b)
            &
            w
        ).sum()
    )


    wb = int(
        (
            b
            &
            (~w)
        ).sum()
    )


    mr = int(
        (
            (~b)
            &
            m
        ).sum()
    )


    mb = int(
        (
            b
            &
            (~m)
        ).sum()
    )


    ba = float(
        b.mean()
    )

    wa = float(
        w.mean()
    )

    ma = float(
        m.mean()
    )


    return {
        "n":
            n,

        "baseline_accuracy":
            ba,

        "whole_accuracy":
            wa,

        "mn_accuracy":
            ma,

        "whole_minus_baseline_pp":
            100.0
            *
            (
                wa - ba
            ),

        "mn_minus_baseline_pp":
            100.0
            *
            (
                ma - ba
            ),

        "mn_minus_whole_pp":
            100.0
            *
            (
                ma - wa
            ),

        "whole_repairs":
            wr,

        "whole_breaks":
            wb,

        "whole_net":
            wr - wb,

        "mn_repairs":
            mr,

        "mn_breaks":
            mb,

        "mn_net":
            mr - mb,
    }


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
    CONDITION_CSV,
    index=False,
)


count_rows = []


for count, group in (
    result
    .groupby(
        "ground_truth",
        sort=True,
    )
):

    row = {
        "ground_truth":
            int(
                count
            )
    }

    row.update(
        summarize_group(
            group
        )
    )

    count_rows.append(
        row
    )


pd.DataFrame(
    count_rows
).to_csv(
    COUNT_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_row = {
    "n":
        EXPECTED_N,

    "baseline_correct":
        baseline_n,

    "baseline_accuracy":
        baseline_accuracy,

    "whole_correct":
        whole_n,

    "whole_accuracy":
        whole_accuracy,

    "mn_correct":
        mn_n,

    "mn_accuracy":
        mn_accuracy,

    "whole_minus_baseline_pp":
        whole_vs_baseline[
            "difference_pp"
        ],

    "mn_minus_baseline_pp":
        mn_vs_baseline[
            "difference_pp"
        ],

    "mn_minus_whole_pp":
        mn_vs_whole[
            "difference_pp"
        ],

    "whole_repairs":
        whole_repairs,

    "whole_breaks":
        whole_breaks,

    "whole_net":
        whole_net,

    "mn_repairs":
        mn_repairs,

    "mn_breaks":
        mn_breaks,

    "mn_net":
        mn_net,

    "primary_success":
        primary_success,

    "mn_baseline_mcnemar_p":
        mn_vs_baseline[
            "mcnemar_exact_p"
        ],

    "whole_baseline_mcnemar_p":
        whole_vs_baseline[
            "mcnemar_exact_p"
        ],

    "mn_whole_mcnemar_p":
        mn_vs_whole[
            "mcnemar_exact_p"
        ],
}


pd.DataFrame([
    summary_row
]).to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================
# RUN METADATA
# ============================================================

run_metadata = {
    "stage":
        "AROMA2_CSA_V4_FINAL_CONFIRMATION",

    "completed_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "git_head":
        git_head(),

    "method_fingerprint":
        METHOD_FINGERPRINT,

    "model":
        MODEL_ID,

    "model_revision":
        MODEL_REVISION,

    "n":
        EXPECTED_N,

    "controller_bundle_sha256":
        controller_bundle_sha,

    "u4_sha256":
        sha256_file(
            U4_PATH
        ),

    "v4_metadata_sha256":
        current_metadata_sha,

    "renderer_sha256":
        current_renderer_sha,

    "whole_alpha1_identity_max_logit_diff":
        whole_identity_diff,

    "mn_alpha1_identity_max_logit_diff":
        mn_identity_diff,

    "bootstrap_reps":
        BOOTSTRAP_REPS,

    "bootstrap_seed":
        BOOTSTRAP_SEED,
}


RUN_META_JSON.write_text(
    json.dumps(
        run_metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL JSON
# ============================================================

final = {
    "stage":
        "AROMA2_CSA_V4_FINAL_CONFIRMATION",

    "primary_success":
        primary_success,

    "n":
        EXPECTED_N,

    "baseline": {
        "correct":
            baseline_n,

        "accuracy":
            baseline_accuracy,
    },

    "whole_head": {
        "correct":
            whole_n,

        "accuracy":
            whole_accuracy,

        "repairs":
            whole_repairs,

        "breaks":
            whole_breaks,

        "net":
            whole_net,
    },

    "mn_csa": {
        "correct":
            mn_n,

        "accuracy":
            mn_accuracy,

        "repairs":
            mn_repairs,

        "breaks":
            mn_breaks,

        "net":
            mn_net,
    },

    "whole_vs_baseline":
        whole_vs_baseline,

    "mn_vs_baseline":
        mn_vs_baseline,

    "mn_vs_whole":
        mn_vs_whole,

    "action_distribution":
        (
            action_distribution
            .to_dict(
                "records"
            )
        ),

    "median_nonnoop_mn_rho":
        (
            float(
                result.loc[
                    ~np.isclose(
                        result[
                            "selected_alpha"
                        ]
                        .astype(float),
                        1.0,
                    ),
                    "mn_rho",
                ]
                .median()
            )
            if (
                ~np.isclose(
                    result[
                        "selected_alpha"
                    ]
                    .astype(float),
                    1.0,
                )
            ).any()
            else None
        ),

    "max_mn_strength_error":
        float(
            result[
                "mn_strength_error"
            ]
            .fillna(
                0.0
            )
            .max()
        ),

    "method_fingerprint":
        METHOD_FINGERPRINT,

    "git_head":
        git_head(),
}


RESULT_JSON.write_text(
    json.dumps(
        final,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()
print(
    "=" * 100
)

print(
    "FINAL UNTOUCHED V4 "
    "CSA CONFIRMATION RESULT"
)

print(
    "=" * 100
)


print(
    "Baseline:",
    baseline_n,
    "/",
    EXPECTED_N,
    f"= "
    f"{100*baseline_accuracy:.3f}%"
)


print(
    "Whole-head:",
    whole_n,
    "/",
    EXPECTED_N,
    f"= "
    f"{100*whole_accuracy:.3f}%"
)


print(
    "MN-CSA:",
    mn_n,
    "/",
    EXPECTED_N,
    f"= "
    f"{100*mn_accuracy:.3f}%"
)


print()
print(
    "WHOLE - BASELINE:",
    f"{whole_vs_baseline['difference_pp']:+.3f} pp"
)

print(
    "  bootstrap 95% CI:",
    [
        round(
            x,
            3
        )
        for x in (
            whole_vs_baseline[
                "bootstrap_ci95_pp"
            ]
        )
    ],
)

print(
    "  McNemar p:",
    whole_vs_baseline[
        "mcnemar_exact_p"
    ]
)


print()
print(
    "MN - BASELINE:",
    f"{mn_vs_baseline['difference_pp']:+.3f} pp"
)

print(
    "  bootstrap 95% CI:",
    [
        round(
            x,
            3
        )
        for x in (
            mn_vs_baseline[
                "bootstrap_ci95_pp"
            ]
        )
    ],
)

print(
    "  McNemar p:",
    mn_vs_baseline[
        "mcnemar_exact_p"
    ]
)


print()
print(
    "MN - WHOLE:",
    f"{mn_vs_whole['difference_pp']:+.3f} pp"
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
                "bootstrap_ci95_pp"
            ]
        )
    ],
)

print(
    "  McNemar p:",
    mn_vs_whole[
        "mcnemar_exact_p"
    ]
)


print()
print(
    "Whole repairs / breaks / net:",
    whole_repairs,
    "/",
    whole_breaks,
    "/",
    whole_net
)


print(
    "MN repairs / breaks / net:",
    mn_repairs,
    "/",
    mn_breaks,
    "/",
    mn_net
)


print()
print(
    "PRIMARY CONFIRMATION SUCCESS:",
    primary_success
)


print()
print(
    "ACTION DISTRIBUTION"
)

print(
    action_distribution
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
        CONDITION_CSV
    )
    .to_string(
        index=False
    )
)


print()
print(
    "BY COUNT"
)

print(
    pd.read_csv(
        COUNT_CSV
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


print(
    "=" * 100
)

