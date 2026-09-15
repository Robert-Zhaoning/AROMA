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

B3A_RESULTS = Path(
    "outputs/aroma2/"
    "csa_gate_b3a_bottom_v1/"
    "gate_b3a_results.csv"
)

TOP4_PATH = Path(
    "outputs/aroma2/"
    "csa_gate_a_v3/"
    "U4_primary.npy"
)

IMAGE_DIR = Path(
    "data/proc_count_causal_v3/images"
)

FREEZE_DOC = Path(
    "docs/"
    "AROMA2_CSA_GATEB3B_RANDOM_FREEZE_v1.md"
)

OUT_DIR = Path(
    "outputs/aroma2/"
    "csa_gate_b3b_random_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_BASES_PATH = (
    OUT_DIR
    / "random_bases.npy"
)

RANDOM_META_PATH = (
    OUT_DIR
    / "random_bases_metadata.json"
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
    / "gate_b3b_results.csv"
)

RANDOM_SUMMARY_CSV = (
    OUT_DIR
    / "random_subspace_summary.csv"
)

BY_ACTION_CSV = (
    OUT_DIR
    / "gate_b3b_by_action.csv"
)

BY_CONDITION_CSV = (
    OUT_DIR
    / "gate_b3b_by_condition.csv"
)

RESULT_JSON = (
    OUT_DIR
    / "gate_b3b_result.json"
)


EXPECTED_N = 653

K_RANDOM = 20

RANDOM_MASTER_SEED = 20260915

BOOTSTRAP_REPS = 20000

BOOTSTRAP_SEED = 20260915

DUMMY_GT = 1

START = 1664
END = 1792


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
            sample_id
            + ".*"
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
            f"{sample_id}: "
            f"expected exactly "
            f"one image, got "
            f"{candidates}"
        )

    return candidates[
        0
    ]


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

    batch = 500
    cursor = 0

    while cursor < (
        BOOTSTRAP_REPS
    ):

        b = min(
            batch,
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
            difference[
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
# MAGNITUDE-NORMALIZED MODIFIER
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
            self.o_proj
            .in_features
        ) != 4096:

            raise RuntimeError(
                "Unexpected "
                "hidden size."
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
                f"Bad U shape: "
                f"{U.shape}"
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

        p_norm = (
            torch.linalg
            .vector_norm(
                projection,
                dim=-1,
                keepdim=True,
            )
        )

        if float(
            p_norm.min()
            .item()
        ) <= 1e-12:

            raise RuntimeError(
                "Projection norm "
                "too small."
            )

        direction = (
            projection
            /
            p_norm
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
            torch.linalg
            .vector_norm(
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

        error = (
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
            ]
            .item()
        )

        self.last_strength_error = (
            float(
                error[
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
# START
# ============================================================

print(
    "=" * 100
)

print(
    "AROMA 2.0 — CSA "
    "GATE B3b RANDOM-SUBSPACE "
    "SPECIFICITY"
)

print(
    "=" * 100
)


for path in [
    V3_RESULTS,
    B3A_RESULTS,
    TOP4_PATH,
    FREEZE_DOC,
]:

    if not path.exists():

        raise RuntimeError(
            f"Missing required "
            f"artifact: {path}"
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


if len(
    data
) != EXPECTED_N:

    raise RuntimeError(
        f"Expected "
        f"{EXPECTED_N}, "
        f"got "
        f"{len(data)}"
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
        f"Action-count "
        f"mismatch: "
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
    "PASS: frozen N=653"
)

print(
    "PASS: action counts:",
    counts
)


# ============================================================
# TOP-4 + B3a REFERENCE
# ============================================================

U_top = np.load(
    TOP4_PATH
).astype(
    np.float32
)


if U_top.shape != (
    128,
    4,
):

    raise RuntimeError(
        f"Bad Top-4 shape: "
        f"{U_top.shape}"
    )


top_orth_error = float(
    np.max(
        np.abs(
            U_top.T
            @ U_top
            -
            np.eye(
                4,
                dtype=np.float32,
            )
        )
    )
)


if top_orth_error > 1e-4:

    raise RuntimeError(
        "Top-4 "
        "orthogonality failed."
    )


b3a = pd.read_csv(
    B3A_RESULTS
)


if len(
    b3a
) != EXPECTED_N:

    raise RuntimeError(
        "B3a N mismatch."
    )


b3a_top_map = dict(
    zip(
        b3a[
            "sample_id"
        ].astype(str),
        b3a[
            "top4_prediction"
        ].astype(int),
    )
)


# ============================================================
# GENERATE ALL RANDOM BASES BEFORE MODEL INFERENCE
# ============================================================

if RANDOM_BASES_PATH.exists():

    random_bases = np.load(
        RANDOM_BASES_PATH
    ).astype(
        np.float32
    )

else:

    rng = (
        np.random
        .default_rng(
            RANDOM_MASTER_SEED
        )
    )

    bases = []

    for k in range(
        K_RANDOM
    ):

        Z = rng.standard_normal(
            size=(
                128,
                4,
            )
        )

        Q, R = np.linalg.qr(
            Z,
            mode="reduced",
        )

        Q = Q.astype(
            np.float32
        )

        bases.append(
            Q
        )

    random_bases = (
        np.stack(
            bases,
            axis=0,
        )
    )

    np.save(
        RANDOM_BASES_PATH,
        random_bases,
    )


if random_bases.shape != (
    K_RANDOM,
    128,
    4,
):

    raise RuntimeError(
        f"Bad random-basis "
        f"shape: "
        f"{random_bases.shape}"
    )


random_meta = []


for k in range(
    K_RANDOM
):

    Q = random_bases[
        k
    ]

    orth_error = float(
        np.max(
            np.abs(
                Q.T
                @ Q
                -
                np.eye(
                    4,
                    dtype=np.float32,
                )
            )
        )
    )

    overlap_fro_sq = float(
        np.linalg.norm(
            U_top.T
            @ Q,
            ord="fro",
        ) ** 2
    )

    overlap_fraction = (
        overlap_fro_sq
        / 4.0
    )

    if orth_error > 1e-4:

        raise RuntimeError(
            f"Random basis {k} "
            f"not orthonormal."
        )

    random_meta.append(
        {
            "random_index":
                k,

            "orth_error":
                orth_error,

            "top4_overlap_fraction":
                overlap_fraction,
        }
    )


RANDOM_META_PATH.write_text(
    json.dumps(
        {
            "K":
                K_RANDOM,

            "master_seed":
                RANDOM_MASTER_SEED,

            "bases_sha256":
                sha256_file(
                    RANDOM_BASES_PATH
                ),

            "bases":
                random_meta,
        },
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "PASS: generated/found "
    "20 frozen random bases"
)

print(
    "Random-bases SHA256:",
    sha256_file(
        RANDOM_BASES_PATH
    )
)

print(
    "Random overlap range:",
    min(
        x[
            "top4_overlap_fraction"
        ]
        for x
        in random_meta
    ),
    max(
        x[
            "top4_overlap_fraction"
        ]
        for x
        in random_meta
    ),
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

    "b3a_results_sha256":
        sha256_file(
            B3A_RESULTS
        ),

    "top4_sha256":
        sha256_file(
            TOP4_PATH
        ),

    "random_bases_sha256":
        sha256_file(
            RANDOM_BASES_PATH
        ),

    "model_revision":
        MODEL_REVISION,

    "n":
        EXPECTED_N,

    "rank":
        4,

    "K_random":
        K_RANDOM,

    "random_master_seed":
        RANDOM_MASTER_SEED,

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
        "Numeral support "
        "!= 0..15"
    )


print(
    "Numeral audit: PASS"
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


print(
    "model.training:",
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
# IDENTITY TEST
# ============================================================

print()
print(
    "=" * 100
)

print(
    "ALPHA=1 IDENTITY "
    "PRECHECK"
)

print(
    "=" * 100
)


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

    baseline_logits = (
        model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
        .logits
        .float()
        .cpu()
    )


for name, U in [
    (
        "top4",
        U_top,
    ),
    (
        "random0",
        random_bases[
            0
        ],
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

            logits = (
                model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
                .logits
                .float()
                .cpu()
            )

    finally:

        modifier.remove()


    diff = float(
        (
            logits
            -
            baseline_logits
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
    "PASS: identity precheck"
)


del logits
del baseline_logits
del inputs

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
            "Partial CSV without "
            "metadata."
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
            "Resume order mismatch."
        )


    print(
        "RESUME VALIDATED:",
        len(
            records
        ),
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
# FORMAL EVALUATION
# ============================================================

print()
print(
    "=" * 100
)

print(
    "FORMAL TOP-4 vs "
    "20 RANDOM RANK-4 SUBSPACES"
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


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    with torch.inference_mode():

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
    # TOP-4
    # --------------------------------------------------------

    top_modifier = (
        MNSubspaceModifier(
            model=model,
            alpha=alpha,
            U=U_top,
        )
    )

    top_modifier.register()

    try:

        with torch.inference_mode():

            top_state = (
                score_state(
                    model,
                    inputs,
                    numeral_ids,
                    DUMMY_GT,
                )
            )

    finally:

        top_modifier.remove()


    if (
        top_modifier.calls
        <= 0
    ):

        raise RuntimeError(
            f"Top hook not called: "
            f"{sid}"
        )


    if (
        top_modifier
        .last_strength_error
        >
        1e-5
    ):

        raise RuntimeError(
            f"Top magnitude error: "
            f"{sid}"
        )


    top_prediction = int(
        top_state[
            "best_numeral"
        ]
    )


    b3a_prediction = int(
        b3a_top_map[
            sid
        ]
    )


    if (
        top_prediction
        != b3a_prediction
    ):

        raise RuntimeError(
            f"Top-4 replay mismatch "
            f"for {sid}: "
            f"{top_prediction} vs "
            f"{b3a_prediction}"
        )


    # --------------------------------------------------------
    # RANDOM-4 x20
    # --------------------------------------------------------

    random_predictions = []
    random_rhos = []


    for k in range(
        K_RANDOM
    ):

        modifier = (
            MNSubspaceModifier(
                model=model,
                alpha=alpha,
                U=random_bases[
                    k
                ],
            )
        )

        modifier.register()

        try:

            with torch.inference_mode():

                state = (
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
                f"Random {k} hook "
                f"not called: {sid}"
            )


        if (
            modifier
            .last_strength_error
            >
            1e-5
        ):

            raise RuntimeError(
                f"Random {k} "
                f"magnitude error: "
                f"{sid}"
            )


        random_predictions.append(
            int(
                state[
                    "best_numeral"
                ]
            )
        )


        random_rhos.append(
            float(
                modifier
                .last_rho
            )
        )


    rec = {
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

        "top4_prediction":
            top_prediction,

        "top4_rho":
            float(
                top_modifier
                .last_rho
            ),
    }


    for k in range(
        K_RANDOM
    ):

        rec[
            f"random_{k:02d}_prediction"
        ] = (
            random_predictions[
                k
            ]
        )

        rec[
            f"random_{k:02d}_rho"
        ] = (
            random_rhos[
                k
            ]
        )


    records.append(
        rec
    )


    # Save after every completed sample.
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
        "Incomplete B3b result."
    )


gt = (
    result[
        "ground_truth"
    ]
    .astype(int)
)


result[
    "baseline_correct"
] = (
    result[
        "baseline_prediction"
    ]
    .astype(int)
    ==
    gt
)


result[
    "top4_correct"
] = (
    result[
        "top4_prediction"
    ]
    .astype(int)
    ==
    gt
)


random_correct_matrix = np.zeros(
    (
        EXPECTED_N,
        K_RANDOM,
    ),
    dtype=np.float64,
)


for k in range(
    K_RANDOM
):

    correct = (
        result[
            f"random_{k:02d}_prediction"
        ]
        .astype(int)
        .to_numpy()
        ==
        gt.to_numpy()
    )


    result[
        f"random_{k:02d}_correct"
    ] = correct


    random_correct_matrix[
        :,
        k
    ] = (
        correct
        .astype(
            np.float64
        )
    )


result[
    "random_mean_correct"
] = (
    random_correct_matrix
    .mean(
        axis=1
    )
)


result[
    "top_minus_random_mean_correct"
] = (
    result[
        "top4_correct"
    ]
    .astype(float)
    -
    result[
        "random_mean_correct"
    ]
)


result.to_csv(
    FINAL_CSV,
    index=False,
)


# ============================================================
# RANDOM-SUBSPACE SUMMARY
# ============================================================

top_correct = int(
    result[
        "top4_correct"
    ].sum()
)


top_accuracy = (
    top_correct
    /
    EXPECTED_N
)


baseline_correct = int(
    result[
        "baseline_correct"
    ].sum()
)


random_rows = []


for k in range(
    K_RANDOM
):

    correct = int(
        result[
            f"random_{k:02d}_correct"
        ].sum()
    )


    accuracy = (
        correct
        /
        EXPECTED_N
    )


    baseline_wrong = (
        ~result[
            "baseline_correct"
        ]
    )


    baseline_right = (
        result[
            "baseline_correct"
        ]
    )


    random_correct = (
        result[
            f"random_{k:02d}_correct"
        ]
    )


    repairs = int(
        (
            baseline_wrong
            &
            random_correct
        ).sum()
    )


    breaks = int(
        (
            baseline_right
            &
            (~random_correct)
        ).sum()
    )


    random_rows.append(
        {
            "random_index":
                k,

            "correct":
                correct,

            "accuracy":
                accuracy,

            "repairs":
                repairs,

            "breaks":
                breaks,

            "net":
                repairs
                -
                breaks,

            "median_rho":
                float(
                    result[
                        f"random_{k:02d}_rho"
                    ]
                    .median()
                ),

            "top4_overlap_fraction":
                random_meta[
                    k
                ][
                    "top4_overlap_fraction"
                ],
        }
    )


random_summary = pd.DataFrame(
    random_rows
)


random_summary.to_csv(
    RANDOM_SUMMARY_CSV,
    index=False,
)


random_accuracies = (
    random_summary[
        "accuracy"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


random_mean_accuracy = float(
    random_accuracies.mean()
)


random_median_accuracy = float(
    np.median(
        random_accuracies
    )
)


random_std_accuracy = float(
    random_accuracies.std(
        ddof=1
    )
)


random_min_accuracy = float(
    random_accuracies.min()
)


random_max_accuracy = float(
    random_accuracies.max()
)


# ============================================================
# PRIMARY SAMPLE-PAIRED TEST
# ============================================================

difference = (
    result[
        "top_minus_random_mean_correct"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


effect = float(
    difference.mean()
)


ci_low, ci_high = (
    paired_bootstrap_ci(
        difference
    )
)


# ============================================================
# RANDOM-SUBSPACE EMPIRICAL RANK
# ============================================================

n_random_ge_top = int(
    np.sum(
        random_accuracies
        >= top_accuracy
    )
)


p_rank = float(
    (
        1
        +
        n_random_ge_top
    )
    /
    (
        K_RANDOM
        +
        1
    )
)


if ci_low > 0:

    status = (
        "RANDOM_SPECIFICITY_POSITIVE"
    )

elif ci_high < 0:

    status = (
        "RANDOM_CONTROL_OUTPERFORMS_TOP4"
    )

else:

    status = (
        "RANDOM_SPECIFICITY_NOT_ESTABLISHED"
    )


# ============================================================
# TOP REPAIR / BREAK
# ============================================================

top_repairs = int(
    (
        (~result[
            "baseline_correct"
        ])
        &
        result[
            "top4_correct"
        ]
    ).sum()
)


top_breaks = int(
    (
        result[
            "baseline_correct"
        ]
        &
        (~result[
            "top4_correct"
        ])
    ).sum()
)


# ============================================================
# ACTION / CONDITION TABLES
# ============================================================

def group_summary(
    group,
):

    idx = (
        group.index
        .to_numpy()
    )


    top_acc = float(
        group[
            "top4_correct"
        ].mean()
    )


    rand_acc = float(
        random_correct_matrix[
            idx,
            :
        ].mean()
    )


    per_random_acc = (
        random_correct_matrix[
            idx,
            :
        ]
        .mean(
            axis=0
        )
    )


    return {
        "n":
            len(
                group
            ),

        "top4_accuracy":
            top_acc,

        "random_mean_accuracy":
            rand_acc,

        "top4_minus_random_mean_pp":
            100.0
            * (
                top_acc
                -
                rand_acc
            ),

        "random_min_accuracy":
            float(
                per_random_acc
                .min()
            ),

        "random_max_accuracy":
            float(
                per_random_acc
                .max()
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
# FINAL RESULT
# ============================================================

final = {
    "stage":
        "AROMA2_CSA_GATE_B3B_RANDOM_V1",

    "status":
        status,

    "git_head":
        git_head(),

    "method_fingerprint":
        METHOD_FINGERPRINT,

    "n":
        EXPECTED_N,

    "K_random":
        K_RANDOM,

    "random_master_seed":
        RANDOM_MASTER_SEED,

    "baseline_correct":
        baseline_correct,

    "top4_correct":
        top_correct,

    "top4_accuracy":
        top_accuracy,

    "random_mean_accuracy":
        random_mean_accuracy,

    "random_median_accuracy":
        random_median_accuracy,

    "random_std_accuracy":
        random_std_accuracy,

    "random_min_accuracy":
        random_min_accuracy,

    "random_max_accuracy":
        random_max_accuracy,

    "top4_minus_random_mean_pp":
        100.0
        * effect,

    "top4_minus_random_max_pp":
        100.0
        * (
            top_accuracy
            -
            random_max_accuracy
        ),

    "paired_bootstrap_95ci_pp":
        [
            100.0
            * ci_low,

            100.0
            * ci_high,
        ],

    "random_ge_top_count":
        n_random_ge_top,

    "random_rank_empirical_p":
        p_rank,

    "top4_repairs":
        top_repairs,

    "top4_breaks":
        top_breaks,

    "top4_net":
        top_repairs
        -
        top_breaks,

    "random_bases_sha256":
        sha256_file(
            RANDOM_BASES_PATH
        ),
}


RESULT_JSON.write_text(
    json.dumps(
        final,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print(
    "=" * 100
)

print(
    "FORMAL CSA GATE B3b RESULT"
)

print(
    "=" * 100
)


print(
    "Top-4:",
    top_correct,
    "/",
    EXPECTED_N,
    f"= "
    f"{100*top_accuracy:.3f}%"
)


print()


print(
    "Random mean:",
    f"{100*random_mean_accuracy:.3f}%"
)

print(
    "Random median:",
    f"{100*random_median_accuracy:.3f}%"
)

print(
    "Random std:",
    f"{100*random_std_accuracy:.3f} pp"
)

print(
    "Random min:",
    f"{100*random_min_accuracy:.3f}%"
)

print(
    "Random max:",
    f"{100*random_max_accuracy:.3f}%"
)


print()


print(
    "Top-4 - random mean:",
    f"{100*effect:+.3f} pp"
)

print(
    "Paired bootstrap 95% CI:",
    f"[{100*ci_low:+.3f}, "
    f"{100*ci_high:+.3f}] pp"
)


print(
    "Top-4 - best random:",
    f"{100*(top_accuracy-random_max_accuracy):+.3f} pp"
)


print()


print(
    "Random subspaces >= Top-4:",
    n_random_ge_top,
    "/",
    K_RANDOM
)


print(
    "Empirical rank p:",
    p_rank
)


print()


print(
    "Top-4 repairs / breaks / net:",
    top_repairs,
    "/",
    top_breaks,
    "/",
    top_repairs
    -
    top_breaks
)


print()


print(
    "PRIMARY STATUS:",
    status
)


print(
    "=" * 100
)


print()
print(
    "RANDOM SUBSPACE SUMMARY"
)

print(
    random_summary
    .sort_values(
        "accuracy",
        ascending=False,
    )
    .to_string(
        index=False
    )
)


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

print(
    "=" * 100
)

