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

MODEL_REVISION = "9eb2daaa8597bf192a8b0e73f848f3a102794df5"

CANONICAL_RUNNER = Path(
    "scripts/run_l18h13_gain_all300.py"
)

FREEZE_DOC = Path(
    "docs/AROMA2_CSA_GATEA_FREEZE_v2.md"
)

SOURCE_CSV = Path(
    "outputs/proc_count_causal_v1/router/"
    "l18h13_gain_all300/"
    "l18h13_gain_all300_results.csv"
)

IMAGE_DIR = Path(
    "data/proc_count_causal_v1/images"
)

OUT_DIR = Path(
    "outputs/aroma2/csa_gate_a_v3"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PARTIAL_PATH = OUT_DIR / "partial_collection.npz"

LAYER = 18
HEAD = 13

HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128

START = HEAD * HEAD_DIM
END = START + HEAD_DIM

RANK = 4

NULL_B = 10000
NULL_SEED = 20260915
NULL_BATCH = 100

NUMERAL_VALUES = list(range(1, 11))

CURRENT_CANONICAL_TOL = 1e-5


# ============================================================
# BASIC UTILITIES
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

SCRIPT_PATH = Path(__file__).resolve()

fingerprint_metadata = {
    "script_sha256":
        sha256_file(SCRIPT_PATH),

    "canonical_runner_sha256":
        sha256_file(CANONICAL_RUNNER),

    "freeze_doc_sha256":
        sha256_file(FREEZE_DOC),

    "source_csv_sha256":
        sha256_file(SOURCE_CSV),

    "model_revision":
        MODEL_REVISION,

    "layer":
        LAYER,

    "head":
        HEAD,

    "head_slice":
        [START, END],

    "pooling":
        "mean_pool_L18H13",

    "numeral_values":
        NUMERAL_VALUES,

    "rank":
        RANK,

    "null_B":
        NULL_B,

    "null_seed":
        NULL_SEED,

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
# DIFFERENTIABLE CANONICAL MU
# ============================================================

def differentiable_mu(
    logits,
    numeral_ids,
):
    last = logits[
        0,
        -1,
        :
    ].float()

    numeral_logits = torch.stack(
        [
            last[
                numeral_ids[n]
            ]
            for n in range(
                1,
                11
            )
        ]
    ).to(
        torch.float64
    )

    probs = torch.softmax(
        numeral_logits,
        dim=0,
    )

    values = torch.arange(
        1,
        11,
        dtype=torch.float64,
        device=probs.device,
    )

    mu = (
        values
        * probs
    ).sum()

    return mu, probs


# ============================================================
# POOLED L18H13 AUTOGRAD COORDINATE
# ============================================================

def make_pooled_leaf_hook(saved):

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
                f"Unexpected o_proj input: {tuple(x.shape)}"
            )

        H = x[
            ...,
            START:END
        ]

        h_base = (
            H
            .detach()
            .float()
            .mean(dim=1)
        )

        h_leaf = (
            h_base
            .clone()
            .requires_grad_(True)
        )

        delta = (
            h_leaf
            - h_base.detach()
        )

        H_new = (
            H.detach()
            +
            delta
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
        saved["T"] = int(
            H.shape[1]
        )

        if len(hook_inputs) == 1:
            return (x_new,)

        return (
            x_new,
            *hook_inputs[1:]
        )

    return hook


# ============================================================
# START
# ============================================================

print("=" * 92)
print("AROMA 2.0 — FINAL FORMAL CSA GATE A v3")
print("=" * 92)

print("Git HEAD:", git_head())
print("Method fingerprint:", METHOD_FINGERPRINT)

print("Model:", canonical.MODEL_ID)
print("Revision:", MODEL_REVISION)

print("Layer/head:", LAYER, HEAD)
print("Head slice:", START, END)
print("Head dimension:", HEAD_DIM)

print("Rank:", RANK)
print("Null B:", NULL_B)
print("Null seed:", NULL_SEED)

print("Canonical prompt:", repr(canonical.PROMPT))


with open(
    OUT_DIR / "method_fingerprint.json",
    "w",
) as f:
    json.dump(
        {
            "fingerprint":
                METHOD_FINGERPRINT,

            "metadata":
                fingerprint_metadata,
        },
        f,
        indent=2,
    )


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

if len(identity) != 300:
    raise RuntimeError(
        f"Expected 300 identity rows; got {len(identity)}"
    )

if identity["sample_id"].nunique() != 300:
    raise RuntimeError(
        "Expected 300 unique sample IDs"
    )

identity = (
    identity
    .sort_values("sample_id")
    .reset_index(drop=True)
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

print("PASS: exact N=300 population")


# ============================================================
# MODEL
# ============================================================

print()
print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    canonical.MODEL_ID,
    revision=MODEL_REVISION,
)

numeral_ids = canonical.numeral_token_ids(
    processor
)

print("Numeral IDs:", numeral_ids)


print()
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
# CURRENT IMPLEMENTATION ALIGNMENT
# ============================================================

print()
print("=" * 92)
print("CURRENT CANONICAL ALIGNMENT PREFLIGHT")
print("=" * 92)

preflight = (
    identity
    .groupby(
        "ground_truth",
        sort=True,
    )
    .head(1)
)

alignment_diffs = []


for _, row in preflight.iterrows():

    inputs = get_inputs(
        Path(row["image_path"])
    )

    gt = int(
        row["ground_truth"]
    )

    state = canonical.score_state(
        model,
        inputs,
        numeral_ids,
        gt,
    )

    canonical_mu = float(
        state["expected_numeral"]
    )

    with torch.no_grad():
        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

    diff_mu, _ = differentiable_mu(
        out.logits,
        numeral_ids,
    )

    diff_value = float(
        diff_mu.item()
    )

    delta = abs(
        diff_value
        - canonical_mu
    )

    alignment_diffs.append(
        delta
    )

    print(
        row["sample_id"],
        f"canonical={canonical_mu:.10f}",
        f"differentiable={diff_value:.10f}",
        f"|diff|={delta:.3e}",
    )


alignment_max = float(
    max(alignment_diffs)
)

print(
    "MAX CURRENT IMPLEMENTATION |mu diff|:",
    alignment_max
)

if alignment_max > CURRENT_CANONICAL_TOL:
    raise RuntimeError(
        "Differentiable statistic does not match "
        "current canonical implementation."
    )

print(
    "PASS: differentiable statistic matches canonical"
)


# ============================================================
# EXACT IDENTITY TEST
# ============================================================

print()
print("=" * 92)
print("POOLED-LEAF IDENTITY TEST")
print("=" * 92)

test_row = identity.iloc[0]

test_inputs = get_inputs(
    Path(
        test_row["image_path"]
    )
)

with torch.no_grad():
    baseline_out = model(
        **test_inputs,
        use_cache=False,
        return_dict=True,
    )

baseline_logits = (
    baseline_out.logits
    .detach()
    .float()
)

saved = {}

handle = o_proj.register_forward_pre_hook(
    make_pooled_leaf_hook(saved)
)

try:
    # Identity test is purely numerical.
    # Do not retain an autograd graph here.
    with torch.no_grad():
        hooked_out = model(
            **test_inputs,
            use_cache=False,
            return_dict=True,
        )
finally:
    handle.remove()

hooked_logits = (
    hooked_out.logits
    .detach()
    .float()
)

identity_max_diff = float(
    (
        hooked_logits
        - baseline_logits
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
        "Pooled-leaf hook is not exact identity."
    )

print("PASS: exact identity")


# ============================================================
# RELEASE IDENTITY/PREFLIGHT TEMPORARIES
# ============================================================

# These objects can otherwise retain large multimodal tensors.
for _name in [
    "baseline_out",
    "hooked_out",
    "baseline_logits",
    "hooked_logits",
    "test_inputs",
    "saved",
    "out",
    "diff_mu",
]:
    if _name in globals():
        del globals()[_name]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(
    "CUDA allocated after identity cleanup (GiB):",
    torch.cuda.memory_allocated() / (1024 ** 3)
    if torch.cuda.is_available()
    else 0.0
)

print(
    "CUDA reserved after identity cleanup (GiB):",
    torch.cuda.memory_reserved() / (1024 ** 3)
    if torch.cuda.is_available()
    else 0.0
)


# ============================================================
# FREEZE PARAMETERS
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad_(False)

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# RESUME WITH METHOD FINGERPRINT
# ============================================================

sample_ids = (
    identity["sample_id"]
    .astype(str)
    .tolist()
)

gradients = []
processed_ids = []
mu_values = []
token_lengths = []


if PARTIAL_PATH.exists():

    partial = np.load(
        PARTIAL_PATH,
        allow_pickle=True,
    )

    stored_fingerprint = str(
        partial["method_fingerprint"].item()
    )

    if stored_fingerprint != METHOD_FINGERPRINT:
        raise RuntimeError(
            "\nRESUME REFUSED.\n"
            "Existing partial gradient file was produced "
            "by a different method fingerprint.\n"
            f"stored : {stored_fingerprint}\n"
            f"current: {METHOD_FINGERPRINT}\n"
            "Delete the old partial only after verifying "
            "that it belongs to an obsolete run."
        )

    processed_ids = (
        partial["sample_ids"]
        .astype(str)
        .tolist()
    )

    gradients = list(
        partial["gradients"]
    )

    mu_values = (
        partial["mu_values"]
        .astype(float)
        .tolist()
    )

    token_lengths = (
        partial["token_lengths"]
        .astype(int)
        .tolist()
    )

    if processed_ids != sample_ids[
        :len(processed_ids)
    ]:
        raise RuntimeError(
            "Resume sample ordering mismatch"
        )

    print(
        f"RESUME VALIDATED: "
        f"{len(processed_ids)}/300"
    )


def save_partial():

    if not gradients:
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

        gradients=np.stack(
            gradients
        ).astype(
            np.float32
        ),

        mu_values=np.asarray(
            mu_values,
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
print("=" * 92)
print("FORMAL N=300 L18H13 GRADIENT COLLECTION")
print("=" * 92)

start_idx = len(
    processed_ids
)

for idx in tqdm(
    range(start_idx, 300),
    initial=start_idx,
    total=300,
):

    row = identity.iloc[idx]

    inputs = get_inputs(
        Path(row["image_path"])
    )

    saved = {}

    handle = o_proj.register_forward_pre_hook(
        make_pooled_leaf_hook(saved)
    )

    try:

        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        mu, _ = differentiable_mu(
            out.logits,
            numeral_ids,
        )

        if "h_leaf" not in saved:
            raise RuntimeError(
                "Pooled h leaf not captured"
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
        .astype(np.float32)
    )

    if g.shape != (128,):
        raise RuntimeError(
            f"Bad gradient shape: {g.shape}"
        )

    if not np.isfinite(g).all():
        raise RuntimeError(
            f"Nonfinite gradient for "
            f"{row['sample_id']}"
        )

    norm = float(
        np.linalg.norm(g)
    )

    if norm == 0.0:
        raise RuntimeError(
            f"Zero gradient for "
            f"{row['sample_id']}"
        )

    gradients.append(g)

    processed_ids.append(
        str(row["sample_id"])
    )

    mu_values.append(
        float(mu.detach().cpu().item())
    )

    token_lengths.append(
        int(saved["T"])
    )


    if (
        (idx + 1) % 10 == 0
        or idx == 299
    ):

        save_partial()

        recent = np.stack(
            gradients[
                max(
                    0,
                    len(gradients) - 10
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


X = np.stack(
    gradients
).astype(np.float64)

if X.shape != (300, 128):
    raise RuntimeError(
        f"Expected (300,128); got {X.shape}"
    )


np.save(
    OUT_DIR / "gradient_matrix.npy",
    X.astype(np.float32)
)

gradient_norms = np.linalg.norm(
    X,
    axis=1,
)

np.save(
    OUT_DIR / "gradient_norms.npy",
    gradient_norms
)


pd.DataFrame(
    {
        "sample_id":
            processed_ids,

        "mu":
            mu_values,

        "token_length":
            token_lengths,

        "gradient_norm":
            gradient_norms,
    }
).to_csv(
    OUT_DIR
    / "gradient_collection_summary.csv",
    index=False,
)


print()
print(
    "GRADIENT MATRIX:",
    X.shape
)

print(
    "NORM min/median/max:",
    float(gradient_norms.min()),
    float(np.median(gradient_norms)),
    float(gradient_norms.max()),
)


# ============================================================
# RELEASE MODEL
# ============================================================

del model
del processor

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# SPECTRUM HELPER
# ============================================================

def spectrum_from_matrix(M):

    M = (
        M
        + M.T
    ) / 2.0

    vals, vecs = np.linalg.eigh(M)

    order = np.argsort(
        vals
    )[::-1]

    vals = np.maximum(
        vals[order],
        0.0,
    )

    vecs = vecs[:, order]

    trace = float(
        vals.sum()
    )

    if trace <= 0:
        raise RuntimeError(
            "Non-positive spectral trace"
        )

    E4 = float(
        vals[:RANK].sum()
        / trace
    )

    return vals, vecs, trace, E4


# ============================================================
# A1: UNCENTERED TOTAL SENSITIVITY
# ============================================================

S = (
    X.T @ X
) / float(
    len(X)
)

vals_total, vecs_total, trace_total, E4_total = (
    spectrum_from_matrix(S)
)

U4_primary = (
    vecs_total[:, :RANK]
    .astype(np.float32)
)


# ============================================================
# MEAN-DIRECTION DECOMPOSITION
# ============================================================

mean_g = X.mean(
    axis=0
)

mean_energy = float(
    np.dot(
        mean_g,
        mean_g
    )
)

mean_share = float(
    mean_energy
    / trace_total
)


# ============================================================
# A2: CENTERED RESIDUAL GEOMETRY
# ============================================================

Xc = (
    X
    - mean_g[None, :]
)

C = (
    Xc.T @ Xc
) / float(
    len(Xc)
)

vals_centered, vecs_centered, trace_centered, E4_centered = (
    spectrum_from_matrix(C)
)

U4_centered = (
    vecs_centered[:, :RANK]
    .astype(np.float32)
)


# ============================================================
# SAVE OBSERVED GEOMETRY
# ============================================================

np.save(
    OUT_DIR
    / "second_moment_uncentered.npy",
    S
)

np.save(
    OUT_DIR
    / "covariance_centered.npy",
    C
)

np.save(
    OUT_DIR
    / "mean_gradient.npy",
    mean_g
)

np.save(
    OUT_DIR
    / "eigenvalues_total.npy",
    vals_total
)

np.save(
    OUT_DIR
    / "eigenvalues_centered.npy",
    vals_centered
)

np.save(
    OUT_DIR
    / "U4_primary.npy",
    U4_primary
)

np.save(
    OUT_DIR
    / "U4_centered_diagnostic.npy",
    U4_centered
)


print()
print("=" * 92)
print("OBSERVED SENSITIVITY GEOMETRY")
print("=" * 92)

print(
    "Mean-direction share:",
    mean_share
)

for r in [
    1, 2, 4, 8, 16, 32, 64, 128
]:

    Et = float(
        vals_total[:r].sum()
        / trace_total
    )

    Ec = float(
        vals_centered[:r].sum()
        / trace_centered
    )

    print(
        f"r={r:3d} "
        f"total={Et:.8f} "
        f"centered={Ec:.8f}"
    )


# ============================================================
# EMPIRICAL NULL — A1 + A2 TOGETHER
# ============================================================

print()
print("=" * 92)
print("10,000-REPLICATE EMPIRICAL NULL")
print("=" * 92)

rng = np.random.default_rng(
    NULL_SEED
)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

norms_t = torch.tensor(
    gradient_norms,
    dtype=torch.float32,
    device=device,
)

null_total_parts = []
null_centered_parts = []


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

    z = rng.standard_normal(
        size=(
            b,
            300,
            128,
        )
    ).astype(
        np.float32
    )

    z /= np.maximum(
        np.linalg.norm(
            z,
            axis=2,
            keepdims=True,
        ),
        1e-12,
    )

    z_t = torch.from_numpy(
        z
    ).to(device)

    Gnull = (
        z_t
        * norms_t[
            None,
            :,
            None,
        ]
    )

    # ------------------------------------------
    # A1 null
    # ------------------------------------------

    M_total = (
        torch.matmul(
            Gnull.transpose(1, 2),
            Gnull,
        )
        / 300.0
    )

    M_total = (
        M_total
        + M_total.transpose(1, 2)
    ) / 2.0


    # ------------------------------------------
    # A2 null
    #
    # Cov = E[gg^T] - mean mean^T
    # ------------------------------------------

    null_mean = Gnull.mean(
        dim=1
    )

    outer_mean = (
        null_mean.unsqueeze(2)
        * null_mean.unsqueeze(1)
    )

    M_centered = (
        M_total
        - outer_mean
    )

    M_centered = (
        M_centered
        + M_centered.transpose(1, 2)
    ) / 2.0


    vals_t = torch.linalg.eigvalsh(
        M_total
    )

    vals_c = torch.linalg.eigvalsh(
        M_centered
    )


    E4_t = (
        vals_t[:, -RANK:].sum(dim=1)
        /
        vals_t.sum(dim=1)
    )

    E4_c = (
        vals_c[:, -RANK:].sum(dim=1)
        /
        vals_c.sum(dim=1)
    )


    null_total_parts.append(
        E4_t
        .detach()
        .cpu()
        .numpy()
    )

    null_centered_parts.append(
        E4_c
        .detach()
        .cpu()
        .numpy()
    )


null_total = np.concatenate(
    null_total_parts
).astype(
    np.float64
)

null_centered = np.concatenate(
    null_centered_parts
).astype(
    np.float64
)


if (
    len(null_total) != NULL_B
    or len(null_centered) != NULL_B
):
    raise RuntimeError(
        "Incorrect null replicate count"
    )


np.save(
    OUT_DIR
    / "null_E4_total.npy",
    null_total
)

np.save(
    OUT_DIR
    / "null_E4_centered.npy",
    null_centered
)


# ============================================================
# TEST HELPER
# ============================================================

def evaluate_gate(
    observed,
    null_values,
):

    median = float(
        np.median(
            null_values
        )
    )

    q95 = float(
        np.quantile(
            null_values,
            0.95,
        )
    )

    q99 = float(
        np.quantile(
            null_values,
            0.99,
        )
    )

    enrichment = float(
        observed
        / median
    )

    p_emp = float(
        (
            1
            + np.sum(
                null_values
                >= observed
            )
        )
        /
        (
            len(null_values)
            + 1
        )
    )

    pass_q99 = bool(
        observed
        > q99
    )

    pass_5x = bool(
        enrichment
        >= 5.0
    )

    passed = bool(
        pass_q99
        and pass_5x
    )

    return {
        "observed_E4":
            float(observed),

        "null_median":
            median,

        "null_q95":
            q95,

        "null_q99":
            q99,

        "enrichment":
            enrichment,

        "empirical_p":
            p_emp,

        "criterion_q99":
            pass_q99,

        "criterion_5x":
            pass_5x,

        "passed":
            passed,
    }


A1 = evaluate_gate(
    E4_total,
    null_total,
)

A2 = evaluate_gate(
    E4_centered,
    null_centered,
)


# ============================================================
# FINAL DECISION
# ============================================================

if not A1["passed"]:

    decision = (
        "GATE_A_NO_GO"
    )

elif A2["passed"]:

    decision = (
        "GATE_A_STRONG_GO"
    )

else:

    decision = (
        "GATE_A_DIRECTIONAL_GO"
    )


result = {
    "stage":
        "AROMA2_CSA_GATE_A_V3",

    "decision":
        decision,

    "method_fingerprint":
        METHOD_FINGERPRINT,

    "git_head":
        git_head(),

    "model_id":
        canonical.MODEL_ID,

    "model_revision":
        MODEL_REVISION,

    "n_samples":
        300,

    "gradient_dimension":
        128,

    "layer":
        LAYER,

    "head":
        HEAD,

    "head_slice":
        [START, END],

    "pooling":
        "mean_pool_L18H13",

    "primary_rank":
        RANK,

    "mean_direction_share":
        mean_share,

    "A1_total_sensitivity":
        A1,

    "A2_centered_geometry":
        A2,

    "identity_max_logit_abs_diff":
        identity_max_diff,

    "canonical_alignment_max_mu_diff":
        alignment_max,
}


with open(
    OUT_DIR
    / "gate_a_result.json",
    "w",
) as f:

    json.dump(
        result,
        f,
        indent=2,
    )


print()
print("=" * 92)
print("FORMAL CSA GATE A v3 RESULT")
print("=" * 92)

print(
    "Mean-direction share:",
    mean_share
)

print()
print("A1 — TOTAL SENSITIVITY")

for k, v in A1.items():
    print(
        f"{k}: {v}"
    )

print()
print("A2 — CENTERED RESIDUAL GEOMETRY")

for k, v in A2.items():
    print(
        f"{k}: {v}"
    )

print()
print(
    "FINAL DECISION:",
    decision
)

print("=" * 92)

print(
    "Primary U4:",
    OUT_DIR
    / "U4_primary.npy"
)

print(
    "Result:",
    OUT_DIR
    / "gate_a_result.json"
)

print("=" * 92)

