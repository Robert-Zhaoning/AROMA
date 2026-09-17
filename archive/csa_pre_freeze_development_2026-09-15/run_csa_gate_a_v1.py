import gc
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration


# ============================================================
# FROZEN GATE-A CONFIG
# ============================================================

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL_REVISION = "9eb2daaa8597bf192a8b0e73f848f3a102794df5"

LOCAL_MODEL_PATH = Path(
    "/workspace/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-11B-Vision-Instruct/"
    "snapshots/"
    + MODEL_REVISION
)

SOURCE_CSV = Path(
    "outputs/proc_count_causal_v1/router/"
    "l18h13_gain_all300/l18h13_gain_all300_results.csv"
)

CANONICAL_RUNNER = Path(
    "scripts/run_l18h13_gain_all300.py"
)

DATA_ROOT = Path("data")

OUT_DIR = Path(
    "outputs/aroma2/csa_gate_a_v1"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PARTIAL_PATH = OUT_DIR / "partial_collection.npz"

LAYER = 18
HEAD = 13
NUM_HEADS = 32
HIDDEN_SIZE = 4096
HEAD_DIM = 128

START = HEAD * HEAD_DIM
END = START + HEAD_DIM

# Frozen primary rank.
RANK = 4

# Frozen null.
NULL_B = 10_000
NULL_SEED = 20260915
NULL_BATCH = 100

# Original directional characterization numeral support.
NUMERAL_VALUES = list(range(1, 11))

# This must match the canonical runner.
QUESTION = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
"""

# Pipeline-only archival consistency tolerance.
# This is NOT a Gate-A success threshold.
ARCHIVE_MAX_MU_DIFF_TOL = 0.05


# ============================================================
# UTILITIES
# ============================================================

def get_git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def conditional_expected_numeral(
    logits,
    numeral_ids,
    numeral_values_tensor,
):
    """
    Frozen AROMA numeral proxy:
    conditional expectation over numerals 1..10.
    """

    last_logits = logits[:, -1, :].float()

    selected = last_logits[:, numeral_ids]

    probs = torch.softmax(
        selected,
        dim=-1,
    )

    mu = (
        probs
        * numeral_values_tensor.unsqueeze(0)
    ).sum(dim=-1)

    return mu.mean(), probs


def prepare_inputs(
    processor,
    model,
    image_path,
):
    """
    EXACTLY aligned to the canonical
    run_l18h13_gain_all300.py input protocol.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": QUESTION,
                },
            ],
        }
    ]

    # IMPORTANT:
    # canonical runner does NOT set tokenize=False here.
    formatted = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    # IMPORTANT:
    # canonical runner explicitly disables additional
    # special-token insertion.
    inputs = processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )

    return {
        key: value.to(model.device)
        for key, value in inputs.items()
    }


# ============================================================
# HEADER
# ============================================================

print("=" * 88)
print("AROMA 2.0 — FORMAL CSA GATE A v1")
print("=" * 88)

print("Git HEAD:", get_git_head())
print("Model revision:", MODEL_REVISION)
print("Layer/head:", LAYER, HEAD)
print("Head dimension:", HEAD_DIM)
print("Head slice:", START, END)
print("Primary rank:", RANK)
print("Null replicates:", NULL_B)
print("Null seed:", NULL_SEED)

if not SOURCE_CSV.exists():
    raise FileNotFoundError(
        SOURCE_CSV
    )

if not CANONICAL_RUNNER.exists():
    raise FileNotFoundError(
        CANONICAL_RUNNER
    )

if not LOCAL_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Frozen model snapshot missing: "
        f"{LOCAL_MODEL_PATH}"
    )


# ============================================================
# CANONICAL PROMPT GUARD
# ============================================================

runner_text = CANONICAL_RUNNER.read_text(
    errors="replace"
)

if QUESTION not in runner_text:

    print()
    print("FROZEN QUESTION STRING NOT FOUND VERBATIM")
    print("Possible question-like strings in canonical runner:")

    candidates = sorted(
        set(
            re.findall(
                r'["\']([^"\']*(?:How many|how many|count)[^"\']*)["\']',
                runner_text,
            )
        )
    )

    for c in candidates[:30]:
        print("  ", repr(c))

    raise RuntimeError(
        "Gate A stopped before model loading: "
        "QUESTION does not match canonical runner."
    )

print("PASS: canonical question matched")


# ============================================================
# EXACT 300-SAMPLE POPULATION
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

if len(identity) != 300:
    raise RuntimeError(
        f"Expected 300 alpha=1 rows, got {len(identity)}"
    )

if identity["sample_id"].nunique() != 300:
    raise RuntimeError(
        "alpha=1 population does not contain "
        "300 unique sample IDs"
    )

identity = (
    identity
    .sort_values("sample_id")
    .reset_index(drop=True)
)

print(
    "PASS: exact alpha=1 source population =",
    len(identity),
)

print(
    "GT distribution:",
    identity["ground_truth"]
    .value_counts()
    .sort_index()
    .to_dict(),
)

print(
    "Condition distribution:",
    identity["condition"]
    .value_counts()
    .sort_index()
    .to_dict(),
)


# ============================================================
# RESOLVE EXACT IMAGE FILES
# ============================================================

print()
print("Indexing images...")

image_exts = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

all_images = [
    p
    for p in DATA_ROOT.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower() in image_exts
    )
]

by_stem = {}

for p in all_images:
    by_stem.setdefault(
        p.stem,
        [],
    ).append(p)


def resolve_image(sample_id):
    exact = by_stem.get(
        sample_id,
        [],
    )

    if len(exact) == 1:
        return exact[0]

    if len(exact) > 1:
        hashes = {
            sha256_file(p)
            for p in exact
        }

        if len(hashes) == 1:
            return sorted(exact)[0]

        raise RuntimeError(
            f"Multiple non-identical exact images "
            f"for {sample_id}: {exact}"
        )

    contains = [
        p
        for p in all_images
        if sample_id in p.stem
    ]

    if len(contains) == 1:
        return contains[0]

    if len(contains) > 1:
        hashes = {
            sha256_file(p)
            for p in contains
        }

        if len(hashes) == 1:
            return sorted(contains)[0]

    raise RuntimeError(
        f"Could not uniquely resolve image for "
        f"{sample_id}; candidates={contains[:10]}"
    )


resolved_paths = []

for sample_id in identity["sample_id"]:
    resolved_paths.append(
        resolve_image(sample_id)
    )

identity["image_path"] = [
    str(p)
    for p in resolved_paths
]

if len(set(identity["image_path"])) != 300:
    raise RuntimeError(
        "Resolved population does not contain "
        "300 unique image paths"
    )

print(
    "PASS: 300 unique source images resolved"
)

print(
    "Example:",
    identity.iloc[0]["sample_id"],
    "->",
    identity.iloc[0]["image_path"],
)


population_manifest = identity[
    [
        "sample_id",
        "ground_truth",
        "condition",
        "replicate",
        "baseline_expected_numeral",
        "image_path",
    ]
].copy()

population_manifest.to_csv(
    OUT_DIR / "population_manifest.csv",
    index=False,
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading frozen model...")

processor = AutoProcessor.from_pretrained(
    str(LOCAL_MODEL_PATH),
    local_files_only=True,
)

model = MllamaForConditionalGeneration.from_pretrained(
    str(LOCAL_MODEL_PATH),
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)

model.eval()

# Parameter gradients are unnecessary.
for p in model.parameters():
    p.requires_grad_(False)

layer = (
    model
    .model
    .language_model
    .layers[LAYER]
)

o_proj = layer.cross_attn.o_proj

if o_proj.in_features != HIDDEN_SIZE:
    raise RuntimeError(
        f"Unexpected o_proj input dimension: "
        f"{o_proj.in_features}"
    )

print("MODEL LOADED")
print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# NUMERAL TOKENS
# ============================================================

numeral_ids = []

for n in NUMERAL_VALUES:
    ids = processor.tokenizer.encode(
        str(n),
        add_special_tokens=False,
    )

    if len(ids) != 1:
        raise RuntimeError(
            f"Numeral {n} not single-token: {ids}"
        )

    numeral_ids.append(
        ids[0]
    )

print(
    "Numeral IDs:",
    numeral_ids,
)

numeral_values_tensor = torch.tensor(
    NUMERAL_VALUES,
    dtype=torch.float32,
    device=model.device,
)


# ============================================================
# ARCHIVED-BASELINE CONSISTENCY PREFLIGHT
# one sample from each ground-truth count
# ============================================================

print()
print("=" * 88)
print("ARCHIVE CONSISTENCY PREFLIGHT")
print("=" * 88)

preflight_rows = (
    identity
    .groupby(
        "ground_truth",
        sort=True,
    )
    .head(1)
)

preflight_diffs = []

with torch.no_grad():

    for _, row in preflight_rows.iterrows():

        inputs = prepare_inputs(
            processor,
            model,
            Path(row["image_path"]),
        )

        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        mu, _ = conditional_expected_numeral(
            out.logits,
            numeral_ids,
            numeral_values_tensor,
        )

        observed = float(
            mu.item()
        )

        archived = float(
            row["baseline_expected_numeral"]
        )

        diff = abs(
            observed - archived
        )

        preflight_diffs.append(
            diff
        )

        print(
            row["sample_id"],
            f"archive={archived:.8f}",
            f"current={observed:.8f}",
            f"|diff|={diff:.8g}",
        )


preflight_max = float(
    np.max(preflight_diffs)
)

preflight_mean = float(
    np.mean(preflight_diffs)
)

print(
    "Preflight max |mu diff|:",
    preflight_max,
)

print(
    "Preflight mean |mu diff|:",
    preflight_mean,
)

if preflight_max > ARCHIVE_MAX_MU_DIFF_TOL:

    raise RuntimeError(
        "Archive consistency check failed. "
        "Gate A stopped before formal collection."
    )

print("PASS: archived baseline consistency")


# ============================================================
# IDENTITY TEST FOR POOLED-LEAF PARAMETERIZATION
# ============================================================

print()
print("=" * 88)
print("POOLED-LEAF IDENTITY TEST")
print("=" * 88)


def make_pooled_leaf_hook(saved):
    """
    Protocol-exact local parameterization.

    H in R^{T x 128}
    h = mean_t H_t

    Introduce a local leaf h_leaf.

    H'_t = H_t + (h_leaf - h_base)

    At initialization h_leaf == h_base,
    therefore the hooked forward pass is exactly identity.

    dH'_t / dh_leaf = I for every t,
    hence autograd returns

        d mu / d h

    directly, rather than requiring post-hoc token pooling.
    """

    def hook(module, hook_inputs):

        if not hook_inputs:
            raise RuntimeError(
                "o_proj pre-hook received no inputs"
            )

        x = hook_inputs[0]

        if x.shape[-1] != HIDDEN_SIZE:
            raise RuntimeError(
                f"Unexpected o_proj input: "
                f"{tuple(x.shape)}"
            )

        H = x[
            ...,
            START:END,
        ]

        h_base = (
            H
            .detach()
            .float()
            .mean(
                dim=1
            )
        )

        h_leaf = (
            h_base
            .clone()
            .requires_grad_(True)
        )

        # Exact zero at the initial point.
        delta = (
            h_leaf
            - h_base.detach()
        )

        H_new = (
            H.detach()
            + delta
            .to(H.dtype)
            .unsqueeze(1)
        )

        x_new = torch.cat(
            [
                x[..., :START].detach(),
                H_new,
                x[..., END:].detach(),
            ],
            dim=-1,
        )

        saved["h_leaf"] = h_leaf
        saved["h_base"] = h_base
        saved["T"] = int(
            H.shape[1]
        )

        if len(hook_inputs) == 1:
            return (
                x_new,
            )

        return (
            x_new,
            *hook_inputs[1:],
        )

    return hook


test_row = identity.iloc[0]

test_inputs = prepare_inputs(
    processor,
    model,
    Path(test_row["image_path"]),
)

with torch.no_grad():

    baseline_out = model(
        **test_inputs,
        use_cache=False,
        return_dict=True,
    )

baseline_logits = (
    baseline_out
    .logits
    .detach()
    .float()
)

saved_test = {}

handle = o_proj.register_forward_pre_hook(
    make_pooled_leaf_hook(
        saved_test
    )
)

try:

    hooked_out = model(
        **test_inputs,
        use_cache=False,
        return_dict=True,
    )

finally:

    handle.remove()

hooked_logits = (
    hooked_out
    .logits
    .detach()
    .float()
)

identity_max_diff = float(
    (
        baseline_logits
        - hooked_logits
    )
    .abs()
    .max()
    .item()
)

print(
    "Identity max logit abs diff:",
    identity_max_diff,
)

if identity_max_diff != 0.0:

    raise RuntimeError(
        "Pooled-leaf parameterization "
        "is not exact identity."
    )

print("PASS: pooled-leaf identity")


# ============================================================
# RESUME SUPPORT
# ============================================================

sample_ids = (
    identity["sample_id"]
    .astype(str)
    .tolist()
)

gradients = []
processed_ids = []
mu_values = []
archive_diffs = []
token_lengths = []

if PARTIAL_PATH.exists():

    partial = np.load(
        PARTIAL_PATH,
        allow_pickle=True,
    )

    processed_ids = (
        partial["sample_ids"]
        .astype(str)
        .tolist()
    )

    gradients = [
        x
        for x in partial["gradients"]
    ]

    mu_values = (
        partial["mu_values"]
        .astype(float)
        .tolist()
    )

    archive_diffs = (
        partial["archive_diffs"]
        .astype(float)
        .tolist()
    )

    token_lengths = (
        partial["token_lengths"]
        .astype(int)
        .tolist()
    )

    expected_prefix = sample_ids[
        :len(processed_ids)
    ]

    if processed_ids != expected_prefix:
        raise RuntimeError(
            "Partial collection does not match "
            "current frozen sample ordering."
        )

    print(
        f"RESUME: {len(processed_ids)}/300 "
        f"samples already collected"
    )


def save_partial():

    if not gradients:
        return

    np.savez_compressed(
        PARTIAL_PATH,
        sample_ids=np.asarray(
            processed_ids,
            dtype=object,
        ),
        gradients=np.stack(
            gradients
        ).astype(
            np.float32
        ),
        mu_values=np.asarray(
            mu_values,
            dtype=np.float64,
        ),
        archive_diffs=np.asarray(
            archive_diffs,
            dtype=np.float64,
        ),
        token_lengths=np.asarray(
            token_lengths,
            dtype=np.int32,
        ),
    )


# ============================================================
# FORMAL GRADIENT COLLECTION
# ============================================================

print()
print("=" * 88)
print("FORMAL N=300 HEAD-ALIGNED GRADIENT COLLECTION")
print("=" * 88)

start_idx = len(
    processed_ids
)

for idx in tqdm(
    range(
        start_idx,
        len(identity),
    ),
    initial=start_idx,
    total=len(identity),
):

    row = identity.iloc[idx]

    inputs = prepare_inputs(
        processor,
        model,
        Path(row["image_path"]),
    )

    saved = {}

    handle = o_proj.register_forward_pre_hook(
        make_pooled_leaf_hook(
            saved
        )
    )

    try:

        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        mu, _ = conditional_expected_numeral(
            out.logits,
            numeral_ids,
            numeral_values_tensor,
        )

        if "h_leaf" not in saved:
            raise RuntimeError(
                "Hook failed to expose pooled h leaf"
            )

        grad = torch.autograd.grad(
            outputs=mu,
            inputs=saved["h_leaf"],
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

    finally:

        handle.remove()

    g = (
        grad[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    if g.shape != (
        HEAD_DIM,
    ):
        raise RuntimeError(
            f"Bad gradient shape: {g.shape}"
        )

    if not np.isfinite(
        g
    ).all():
        raise RuntimeError(
            f"Non-finite gradient "
            f"for {row['sample_id']}"
        )

    if np.linalg.norm(
        g
    ) == 0:
        raise RuntimeError(
            f"Zero gradient "
            f"for {row['sample_id']}"
        )

    observed_mu = float(
        mu.detach().float().item()
    )

    archived_mu = float(
        row[
            "baseline_expected_numeral"
        ]
    )

    archived_diff = abs(
        observed_mu
        - archived_mu
    )

    gradients.append(
        g
    )

    processed_ids.append(
        str(
            row["sample_id"]
        )
    )

    mu_values.append(
        observed_mu
    )

    archive_diffs.append(
        archived_diff
    )

    token_lengths.append(
        int(
            saved["T"]
        )
    )

    if (
        (idx + 1) % 10 == 0
        or idx + 1 == len(identity)
    ):
        save_partial()

        recent = np.stack(
            gradients[
                max(
                    0,
                    len(gradients) - 10,
                ):
            ]
        )

        recent_norms = np.linalg.norm(
            recent,
            axis=1,
        )

        tqdm.write(
            f"saved {idx+1}/300 | "
            f"recent norm mean="
            f"{recent_norms.mean():.6e}"
        )


gradient_matrix = np.stack(
    gradients
).astype(
    np.float32
)

if gradient_matrix.shape != (
    300,
    128,
):
    raise RuntimeError(
        f"Expected (300,128), "
        f"got {gradient_matrix.shape}"
    )

np.save(
    OUT_DIR / "gradient_matrix.npy",
    gradient_matrix,
)

np.save(
    OUT_DIR / "gradient_norms.npy",
    np.linalg.norm(
        gradient_matrix,
        axis=1,
    ),
)

collection_df = pd.DataFrame(
    {
        "sample_id": processed_ids,
        "mu_current": mu_values,
        "mu_archive_diff": archive_diffs,
        "token_length": token_lengths,
        "gradient_norm": np.linalg.norm(
            gradient_matrix,
            axis=1,
        ),
    }
)

collection_df.to_csv(
    OUT_DIR
    / "gradient_collection_summary.csv",
    index=False,
)

print()
print(
    "GRADIENT MATRIX:",
    gradient_matrix.shape,
)

print(
    "Gradient norm min/median/max:",
    float(
        np.min(
            np.linalg.norm(
                gradient_matrix,
                axis=1,
            )
        )
    ),
    float(
        np.median(
            np.linalg.norm(
                gradient_matrix,
                axis=1,
            )
        )
    ),
    float(
        np.max(
            np.linalg.norm(
                gradient_matrix,
                axis=1,
            )
        )
    ),
)

print(
    "Archive mu diff mean/max:",
    float(
        np.mean(
            archive_diffs
        )
    ),
    float(
        np.max(
            archive_diffs
        )
    ),
)


# ============================================================
# FREE MODEL BEFORE NULL ANALYSIS
# ============================================================

del model
del processor
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# OBSERVED UN-CENTERED SECOND MOMENT
# ============================================================

print()
print("=" * 88)
print("OBSERVED CSA SPECTRUM")
print("=" * 88)

N, D = gradient_matrix.shape

G = (
    gradient_matrix.T
    @ gradient_matrix
) / float(N)

G = (
    G
    + G.T
) / 2.0

eigvals, eigvecs = np.linalg.eigh(
    G.astype(
        np.float64
    )
)

order = np.argsort(
    eigvals
)[::-1]

eigvals = eigvals[
    order
]

eigvecs = eigvecs[
    :,
    order
]

eigvals = np.maximum(
    eigvals,
    0.0,
)

trace = float(
    eigvals.sum()
)

if trace <= 0:
    raise RuntimeError(
        "Sensitivity second moment has zero trace"
    )

observed_e4 = float(
    eigvals[
        :RANK
    ].sum()
    / trace
)

U4 = eigvecs[
    :,
    :RANK
].astype(
    np.float32
)

np.save(
    OUT_DIR / "sensitivity_second_moment.npy",
    G.astype(
        np.float64
    ),
)

np.save(
    OUT_DIR / "eigenvalues.npy",
    eigvals,
)

np.save(
    OUT_DIR / "eigenvectors.npy",
    eigvecs,
)

np.save(
    OUT_DIR / "U4.npy",
    U4,
)

print(
    "Observed E4:",
    observed_e4,
)

for r in [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
]:

    er = float(
        eigvals[:r].sum()
        / trace
    )

    print(
        f"E{r}: {er:.8f}"
    )


# ============================================================
# 10,000-REPLICATE NORM-MATCHED EMPIRICAL NULL
# ============================================================

print()
print("=" * 88)
print("EMPIRICAL FINITE-SAMPLE NULL")
print("=" * 88)

norms = np.linalg.norm(
    gradient_matrix,
    axis=1,
).astype(
    np.float32
)

rng = np.random.default_rng(
    NULL_SEED
)

null_e4_parts = []

null_device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

norms_t = torch.from_numpy(
    norms
).to(
    null_device
)

for start in tqdm(
    range(
        0,
        NULL_B,
        NULL_BATCH,
    ),
    desc="null batches",
):

    b = min(
        NULL_BATCH,
        NULL_B - start,
    )

    # Random draws generated by NumPy RNG so the
    # directions are seed-reproducible independent
    # of CUDA RNG implementation.
    z = rng.standard_normal(
        size=(
            b,
            N,
            D,
        ),
        dtype=np.float32,
    )

    z_norm = np.linalg.norm(
        z,
        axis=2,
        keepdims=True,
    )

    z = (
        z
        / np.maximum(
            z_norm,
            1e-12,
        )
    )

    z_t = torch.from_numpy(
        z
    ).to(
        null_device
    )

    g_null = (
        z_t
        * norms_t[
            None,
            :,
            None,
        ]
    )

    M = torch.matmul(
        g_null.transpose(
            1,
            2,
        ),
        g_null,
    ) / float(N)

    M = (
        M
        + M.transpose(
            1,
            2,
        )
    ) / 2.0

    null_eigvals = torch.linalg.eigvalsh(
        M
    )

    top4 = null_eigvals[
        :,
        -RANK:
    ].sum(
        dim=1
    )

    null_trace = null_eigvals.sum(
        dim=1
    )

    batch_e4 = (
        top4
        / null_trace
    )

    null_e4_parts.append(
        batch_e4
        .detach()
        .cpu()
        .numpy()
    )

    del z_t
    del g_null
    del M
    del null_eigvals
    del top4
    del null_trace
    del batch_e4


null_e4 = np.concatenate(
    null_e4_parts
).astype(
    np.float64
)

if len(null_e4) != NULL_B:
    raise RuntimeError(
        f"Expected {NULL_B} null values, "
        f"got {len(null_e4)}"
    )

np.save(
    OUT_DIR / "null_E4.npy",
    null_e4,
)

null_median = float(
    np.median(
        null_e4
    )
)

null_q95 = float(
    np.quantile(
        null_e4,
        0.95,
    )
)

null_q99 = float(
    np.quantile(
        null_e4,
        0.99,
    )
)

enrichment = float(
    observed_e4
    / null_median
)

p_emp = float(
    (
        1
        + np.sum(
            null_e4
            >= observed_e4
        )
    )
    / (
        NULL_B
        + 1
    )
)

criterion_q99 = bool(
    observed_e4
    > null_q99
)

criterion_5x = bool(
    enrichment
    >= 5.0
)

gate_a_go = bool(
    criterion_q99
    and criterion_5x
)


# ============================================================
# FINAL RESULT
# ============================================================

decision = (
    "GATE_A_GO"
    if gate_a_go
    else "GATE_A_NO_GO"
)

result = {
    "stage": "AROMA2_CSA_GATE_A_V1",
    "status": "formal_frozen_gate",
    "decision": decision,

    "git_head": get_git_head(),

    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,

    "source_csv": str(
        SOURCE_CSV
    ),

    "n_samples": int(N),
    "gradient_dimension": int(D),

    "layer": LAYER,
    "head": HEAD,

    "head_slice": [
        START,
        END,
    ],

    "pooling": "mean_pool_head_activation",

    "gradient_definition": (
        "direct autograd gradient of conditional "
        "expected numeral with respect to pooled "
        "L18H13 activation"
    ),

    "numeral_values": NUMERAL_VALUES,

    "covariance": (
        "uncentered second moment "
        "(1/N) sum_i g_i g_i^T"
    ),

    "primary_rank": RANK,

    "observed_E4": observed_e4,

    "null": {
        "replicates": NULL_B,
        "seed": NULL_SEED,
        "construction": (
            "per-sample norm-preserving random "
            "directions in R^128"
        ),
        "median_E4": null_median,
        "q95_E4": null_q95,
        "q99_E4": null_q99,
    },

    "E4_over_null_median": enrichment,

    "empirical_upper_tail_p": p_emp,

    "criteria": {
        "observed_E4_gt_null_q99":
            criterion_q99,
        "E4_over_null_median_ge_5":
            criterion_5x,
    },

    "archive_consistency": {
        "preflight_max_mu_abs_diff":
            preflight_max,
        "preflight_mean_mu_abs_diff":
            preflight_mean,
        "full_collection_mean_mu_abs_diff":
            float(
                np.mean(
                    archive_diffs
                )
            ),
        "full_collection_max_mu_abs_diff":
            float(
                np.max(
                    archive_diffs
                )
            ),
    },

    "identity_max_logit_abs_diff":
        identity_max_diff,
}

with open(
    OUT_DIR / "gate_a_result.json",
    "w",
) as f:

    json.dump(
        result,
        f,
        indent=2,
    )


print()
print("=" * 88)
print("FORMAL CSA GATE A RESULT")
print("=" * 88)

print(
    "Observed E4:",
    observed_e4,
)

print(
    "Null median E4:",
    null_median,
)

print(
    "Null q95 E4:",
    null_q95,
)

print(
    "Null q99 E4:",
    null_q99,
)

print(
    "E4 / null median:",
    enrichment,
)

print(
    "Empirical upper-tail p:",
    p_emp,
)

print()
print(
    "criterion E4 > null q99:",
    criterion_q99,
)

print(
    "criterion enrichment >= 5:",
    criterion_5x,
)

print()
print(
    "FINAL DECISION:",
    decision,
)

print("=" * 88)

print(
    "U4:",
    OUT_DIR / "U4.npy",
)

print(
    "Gate result:",
    OUT_DIR / "gate_a_result.json",
)

print("=" * 88)

