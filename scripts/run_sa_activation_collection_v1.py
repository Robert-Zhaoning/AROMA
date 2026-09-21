import gc
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration


# ============================================================
# FROZEN CONFIG
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

CANONICAL_RUNNER = Path(
    "scripts/run_l18h13_gain_all300.py"
)

SOURCE_CSV = Path(
    "outputs/proc_count_causal_v1/router/"
    "l18h13_gain_all300/"
    "l18h13_gain_all300_results.csv"
)

IMAGE_DIR = Path(
    "data/proc_count_causal_v1/images"
)

GATE_A_DIR = Path(
    "outputs/aroma2/csa_gate_a_v3"
)

REFERENCE_PARTIAL = (
    GATE_A_DIR
    / "partial_collection.npz"
)

REFERENCE_GRADIENTS = (
    GATE_A_DIR
    / "gradient_matrix.npy"
)

OUT_DIR = Path(
    "outputs/aroma2/"
    "sa_direction_diagnostic_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL_PATH = (
    OUT_DIR
    / "activation_partial.npz"
)

LAYER = 18
HEAD = 13

HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128

START = HEAD * HEAD_DIM
END = START + HEAD_DIM

EXPECTED_N = 300


# ============================================================
# UTILITIES
# ============================================================

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
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def load_canonical():
    spec = importlib.util.spec_from_file_location(
        "canonical_l18h13",
        CANONICAL_RUNNER,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


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
            f"{sample_id}: expected exactly one image; "
            f"got {candidates}"
        )

    return candidates[0]


canonical = load_canonical()


# ============================================================
# METHOD FINGERPRINT
# ============================================================

fingerprint_metadata = {
    "canonical_runner_sha256":
        sha256_file(CANONICAL_RUNNER),

    "source_csv_sha256":
        sha256_file(SOURCE_CSV),

    "reference_partial_sha256":
        sha256_file(REFERENCE_PARTIAL),

    "reference_gradient_sha256":
        sha256_file(REFERENCE_GRADIENTS),

    "model_revision":
        MODEL_REVISION,

    "layer":
        LAYER,

    "head":
        HEAD,

    "head_slice":
        [START, END],

    "activation_coordinate":
        "mean_pool_L18H13_o_proj_input",

    "canonical_prompt":
        canonical.PROMPT,
}

fingerprint_json = json.dumps(
    fingerprint_metadata,
    sort_keys=True,
    separators=(",", ":"),
)

METHOD_FINGERPRINT = hashlib.sha256(
    fingerprint_json.encode("utf-8")
).hexdigest()


# ============================================================
# START
# ============================================================

print("=" * 92)
print(
    "AROMA — SA DIRECTION DIAGNOSTIC "
    "ACTIVATION COLLECTION v1"
)
print("=" * 92)

print("Git HEAD:", git_head())
print("Method fingerprint:", METHOD_FINGERPRINT)
print("Model:", canonical.MODEL_ID)
print("Revision:", MODEL_REVISION)
print("Layer/head:", LAYER, HEAD)
print("Head slice:", START, END)
print("Canonical prompt:", repr(canonical.PROMPT))


# ============================================================
# REFERENCE GATE-A ARTIFACTS
# ============================================================

if not REFERENCE_PARTIAL.exists():
    raise RuntimeError(
        f"Missing reference partial: "
        f"{REFERENCE_PARTIAL}"
    )

if not REFERENCE_GRADIENTS.exists():
    raise RuntimeError(
        f"Missing reference gradients: "
        f"{REFERENCE_GRADIENTS}"
    )

reference = np.load(
    REFERENCE_PARTIAL,
    allow_pickle=True,
)

reference_ids = (
    reference["sample_ids"]
    .astype(str)
    .tolist()
)

reference_token_lengths = (
    reference["token_lengths"]
    .astype(int)
)

if len(reference_ids) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} reference IDs; "
        f"got {len(reference_ids)}"
    )

G = np.load(
    REFERENCE_GRADIENTS
)

if G.shape != (
    EXPECTED_N,
    HEAD_DIM,
):
    raise RuntimeError(
        f"Bad gradient matrix shape: "
        f"{G.shape}"
    )

del G


# ============================================================
# EXACT SOURCE POPULATION
# ============================================================

df = pd.read_csv(
    SOURCE_CSV
)

identity = df[
    np.isclose(
        df["alpha"].astype(float),
        1.0,
    )
].copy()

if len(identity) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} identity rows; "
        f"got {len(identity)}"
    )

if (
    identity["sample_id"].nunique()
    != EXPECTED_N
):
    raise RuntimeError(
        "Expected 300 unique sample IDs"
    )

identity = (
    identity
    .sort_values("sample_id")
    .reset_index(drop=True)
)

sample_ids = (
    identity["sample_id"]
    .astype(str)
    .tolist()
)

if sample_ids != reference_ids:
    raise RuntimeError(
        "Current Gate-A population/order does not "
        "exactly match archived gradient sample IDs"
    )

print()
print(
    "PASS: current N=300 sample IDs exactly match "
    "archived Gate-A gradient IDs"
)

identity["image_path"] = [
    str(
        resolve_image(
            str(sample_id)
        )
    )
    for sample_id
    in identity["sample_id"]
]


# ============================================================
# MODEL
# ============================================================

print()
print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    canonical.MODEL_ID,
    revision=MODEL_REVISION,
)

print("Loading model...")

model = (
    MllamaForConditionalGeneration
    .from_pretrained(
        canonical.MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
)

model.eval()

for parameter in model.parameters():
    parameter.requires_grad_(False)

print("MODEL LOADED")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


layer = (
    model
    .model
    .language_model
    .layers[LAYER]
)

o_proj = layer.cross_attn.o_proj


# ============================================================
# CANONICAL INPUT
# ============================================================

def get_inputs(image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = canonical.prepare_inputs(
        processor,
        image,
    )

    return canonical.move_inputs(
        inputs,
        model,
    )


# ============================================================
# ACTIVATION CAPTURE HOOK
# ============================================================

def make_capture_hook(saved):

    def hook(
        module,
        hook_inputs,
    ):
        if not hook_inputs:
            raise RuntimeError(
                "o_proj hook received no inputs"
            )

        x = hook_inputs[0]

        if x.shape[-1] != HIDDEN_SIZE:
            raise RuntimeError(
                f"Unexpected o_proj input: "
                f"{tuple(x.shape)}"
            )

        H = x[
            ...,
            START:END
        ]

        if H.ndim != 3:
            raise RuntimeError(
                f"Expected [B,T,128]; "
                f"got {tuple(H.shape)}"
            )

        if H.shape[0] != 1:
            raise RuntimeError(
                f"Expected batch size 1; "
                f"got {H.shape[0]}"
            )

        h_base = (
            H
            .detach()
            .float()
            .mean(dim=1)
        )

        h = (
            h_base[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if h.shape != (HEAD_DIM,):
            raise RuntimeError(
                f"Bad pooled activation shape: "
                f"{h.shape}"
            )

        saved["h"] = h
        saved["T"] = int(
            H.shape[1]
        )

        saved["calls"] = (
            saved.get(
                "calls",
                0,
            )
            + 1
        )

        # Observation-only hook:
        # return None so model inputs are unchanged.
        return None

    return hook


# ============================================================
# RESUME
# ============================================================

pooled_h = []
processed_ids = []
token_lengths = []


if PARTIAL_PATH.exists():

    partial = np.load(
        PARTIAL_PATH,
        allow_pickle=True,
    )

    stored_fingerprint = str(
        partial[
            "method_fingerprint"
        ].item()
    )

    if (
        stored_fingerprint
        != METHOD_FINGERPRINT
    ):
        raise RuntimeError(
            "\nRESUME REFUSED.\n"
            "Existing activation partial was "
            "produced by a different method "
            "fingerprint."
        )

    processed_ids = (
        partial["sample_ids"]
        .astype(str)
        .tolist()
    )

    pooled_h = list(
        partial["pooled_h"]
    )

    token_lengths = (
        partial["token_lengths"]
        .astype(int)
        .tolist()
    )

    if (
        processed_ids
        != sample_ids[
            :len(processed_ids)
        ]
    ):
        raise RuntimeError(
            "Resume sample ordering mismatch"
        )

    print(
        f"RESUME VALIDATED: "
        f"{len(processed_ids)}/"
        f"{EXPECTED_N}"
    )


def save_partial():

    if not pooled_h:
        return

    np.savez_compressed(
        PARTIAL_PATH,

        method_fingerprint=np.asarray(
            METHOD_FINGERPRINT
        ),

        sample_ids=np.asarray(
            processed_ids,
            dtype=object,
        ),

        pooled_h=np.stack(
            pooled_h
        ).astype(
            np.float32
        ),

        token_lengths=np.asarray(
            token_lengths,
            dtype=np.int32,
        ),
    )


# ============================================================
# ACTIVATION COLLECTION
# ============================================================

print()
print("=" * 92)
print(
    "FORMAL N=300 POOLED L18H13 "
    "ACTIVATION COLLECTION"
)
print("=" * 92)

start_idx = len(
    processed_ids
)

for idx in tqdm(
    range(
        start_idx,
        EXPECTED_N,
    ),
    initial=start_idx,
    total=EXPECTED_N,
):

    row = identity.iloc[idx]

    sid = str(
        row["sample_id"]
    )

    inputs = get_inputs(
        Path(
            row["image_path"]
        )
    )

    saved = {}

    handle = (
        o_proj
        .register_forward_pre_hook(
            make_capture_hook(saved)
        )
    )

    try:
        with torch.no_grad():
            _ = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()

    if "h" not in saved:
        raise RuntimeError(
            f"{sid}: activation was not captured"
        )

    if (
        saved.get(
            "calls",
            0,
        )
        <= 0
    ):
        raise RuntimeError(
            f"{sid}: hook was not called"
        )

    h = saved["h"]

    if not np.isfinite(h).all():
        raise RuntimeError(
            f"{sid}: nonfinite pooled activation"
        )

    pooled_h.append(
        h
    )

    processed_ids.append(
        sid
    )

    token_lengths.append(
        int(
            saved["T"]
        )
    )

    if (
        (idx + 1) % 10 == 0
        or idx == EXPECTED_N - 1
    ):
        save_partial()

        recent = np.stack(
            pooled_h[
                max(
                    0,
                    len(pooled_h) - 10
                ):
            ]
        )

        recent_norm = np.linalg.norm(
            recent,
            axis=1,
        )

        tqdm.write(
            f"saved {idx+1}/"
            f"{EXPECTED_N} | "
            f"recent h-norm mean="
            f"{recent_norm.mean():.6e}"
        )

    del inputs

    if (
        (idx + 1) % 25 == 0
        and torch.cuda.is_available()
    ):
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# FINAL MATRIX
# ============================================================

H = np.stack(
    pooled_h
).astype(
    np.float32
)

if H.shape != (
    EXPECTED_N,
    HEAD_DIM,
):
    raise RuntimeError(
        f"Expected (300,128); "
        f"got {H.shape}"
    )

current_lengths = np.asarray(
    token_lengths,
    dtype=np.int32,
)

if not np.array_equal(
    current_lengths,
    reference_token_lengths,
):
    mismatch = np.where(
        current_lengths
        != reference_token_lengths
    )[0]

    raise RuntimeError(
        "Token lengths do not reproduce "
        "Gate-A reference. "
        f"Mismatches: {len(mismatch)}"
    )

print()
print(
    "PASS: token lengths match archived "
    "Gate-A reference 300/300"
)


# ============================================================
# SAVE FINAL ARTIFACTS
# ============================================================

np.save(
    OUT_DIR
    / "pooled_h_matrix.npy",
    H,
)

np.save(
    OUT_DIR
    / "sample_ids.npy",
    np.asarray(
        processed_ids,
        dtype=object,
    ),
)

np.save(
    OUT_DIR
    / "token_lengths.npy",
    current_lengths,
)

h_norms = np.linalg.norm(
    H.astype(np.float64),
    axis=1,
)

pd.DataFrame(
    {
        "sample_id":
            processed_ids,

        "token_length":
            current_lengths,

        "h_norm":
            h_norms,
    }
).to_csv(
    OUT_DIR
    / "activation_collection_summary.csv",
    index=False,
)


with open(
    OUT_DIR
    / "metadata.json",
    "w",
) as f:
    json.dump(
        {
            "method_fingerprint":
                METHOD_FINGERPRINT,

            "git_head":
                git_head(),

            "model_revision":
                MODEL_REVISION,

            "layer":
                LAYER,

            "head":
                HEAD,

            "head_slice":
                [START, END],

            "N":
                EXPECTED_N,

            "activation_definition":
                "mean over token dimension "
                "of L18H13 o_proj input",

            "gradient_reference":
                str(
                    REFERENCE_GRADIENTS
                ),

            "sample_id_exact_match":
                True,

            "token_length_exact_match":
                True,
        },
        f,
        indent=2,
    )


print()
print("=" * 92)
print("FINAL ACTIVATION MATRIX")
print("=" * 92)

print("Shape:", H.shape)

print(
    "h norm min/median/max:",
    float(
        h_norms.min()
    ),
    float(
        np.median(
            h_norms
        )
    ),
    float(
        h_norms.max()
    ),
)

print()
print(
    "Saved:",
    OUT_DIR
    / "pooled_h_matrix.npy"
)

print(
    "Saved:",
    OUT_DIR
    / "sample_ids.npy"
)

print(
    "Saved:",
    OUT_DIR
    / "activation_collection_summary.csv"
)

print(
    "Saved:",
    OUT_DIR
    / "metadata.json"
)

print()
print(
    "SA ACTIVATION COLLECTION COMPLETE"
)
