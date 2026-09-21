import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


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

GATE = Path(
    "outputs/aroma2/csa_gate_a_v3"
)

SA = Path(
    "outputs/aroma2/sa_direction_diagnostic_v1"
)

LAYER = 18
HEAD = 13
HEAD_DIM = 128
HIDDEN_SIZE = 4096

START = HEAD * HEAD_DIM
END = START + HEAD_DIM


def load_canonical():
    spec = importlib.util.spec_from_file_location(
        "canonical_l18h13",
        CANONICAL_RUNNER,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


canonical = load_canonical()


def resolve_image(sample_id):
    candidates = [
        p
        for p in IMAGE_DIR.glob(sample_id + ".*")
        if p.suffix.lower()
        in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"{sample_id}: expected one image; got {candidates}"
        )

    return candidates[0]


def differentiable_mu(logits, numeral_ids):
    last = logits[0, -1, :].float()

    numeral_logits = torch.stack(
        [
            last[numeral_ids[n]]
            for n in range(1, 11)
        ]
    ).to(torch.float64)

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

    return (values * probs).sum()


# ============================================================
# ARCHIVED DATA
# ============================================================

G_archive = np.load(
    GATE / "gradient_matrix.npy"
).astype(np.float64)

partial = np.load(
    GATE / "partial_collection.npz",
    allow_pickle=True,
)

sample_ids = (
    partial["sample_ids"]
    .astype(str)
)

diag = pd.read_csv(
    SA / "sa_direction_per_sample.csv"
)

# Deterministic 5 samples spread across cosine distribution.
diag = diag.sort_values(
    "cos_u4g_u4h"
).reset_index(drop=True)

positions = [
    int(round(q * (len(diag) - 1)))
    for q in [0.05, 0.25, 0.50, 0.75, 0.95]
]

selected_ids = (
    diag.iloc[positions]["sample_id"]
    .astype(str)
    .tolist()
)


# ============================================================
# SOURCE POPULATION
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

identity = (
    identity
    .sort_values("sample_id")
    .reset_index(drop=True)
)

identity_ids = (
    identity["sample_id"]
    .astype(str)
    .to_numpy()
)

if not np.array_equal(
    identity_ids,
    sample_ids,
):
    raise RuntimeError(
        "Source IDs do not match archived Gate-A IDs"
    )


# ============================================================
# MODEL
# ============================================================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    canonical.MODEL_ID,
    revision=MODEL_REVISION,
)

numeral_ids = canonical.numeral_token_ids(
    processor
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

for p in model.parameters():
    p.requires_grad_(False)

layer = (
    model
    .model
    .language_model
    .layers[LAYER]
)

o_proj = layer.cross_attn.o_proj

print("MODEL LOADED")


def get_inputs(sample_id):
    image = Image.open(
        resolve_image(sample_id)
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
# SHARED-DELTA HOOK
# Exact semantics of Gate-A.
# ============================================================

def make_shared_hook(saved):

    def hook(module, hook_inputs):
        x = hook_inputs[0]

        H = x[..., START:END]

        h_base = (
            H.detach()
            .float()
            .mean(dim=1)
        )

        h_leaf = (
            h_base.clone()
            .requires_grad_(True)
        )

        delta = (
            h_leaf
            - h_base.detach()
        )

        H_new = (
            H.detach()
            + delta.to(H.dtype).unsqueeze(1)
        )

        x_new = torch.cat(
            [
                x[..., :START].detach(),
                H_new,
                x[..., END:].detach(),
            ],
            dim=-1,
        )

        saved["leaf"] = h_leaf

        if len(hook_inputs) == 1:
            return (x_new,)

        return (
            x_new,
            *hook_inputs[1:]
        )

    return hook


# ============================================================
# PER-TOKEN LEAF HOOK
# ============================================================

def make_token_hook(saved):

    def hook(module, hook_inputs):
        x = hook_inputs[0]

        H = x[..., START:END]

        token_leaf = (
            H.detach()
            .float()
            .clone()
            .requires_grad_(True)
        )

        H_new = token_leaf.to(H.dtype)

        x_new = torch.cat(
            [
                x[..., :START].detach(),
                H_new,
                x[..., END:].detach(),
            ],
            dim=-1,
        )

        saved["leaf"] = token_leaf

        if len(hook_inputs) == 1:
            return (x_new,)

        return (
            x_new,
            *hook_inputs[1:]
        )

    return hook


def cosine(a, b):
    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    return float(
        np.dot(a, b)
        / max(denom, 1e-30)
    )


def rel_l2(a, b):
    return float(
        np.linalg.norm(a - b)
        / max(
            np.linalg.norm(b),
            1e-30,
        )
    )


records = []

print()
print("=" * 96)
print("AUTOGRAD CHAIN-RULE AUDIT")
print("=" * 96)


for sid in selected_ids:

    idx = int(
        np.where(
            sample_ids == sid
        )[0][0]
    )

    inputs = get_inputs(sid)

    # --------------------------------------------------------
    # A. Shared perturbation gradient
    # --------------------------------------------------------

    saved = {}

    handle = o_proj.register_forward_pre_hook(
        make_shared_hook(saved)
    )

    try:
        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        mu = differentiable_mu(
            out.logits,
            numeral_ids,
        )

        g_shared_t = torch.autograd.grad(
            outputs=mu,
            inputs=saved["leaf"],
            retain_graph=False,
            create_graph=False,
        )[0]

    finally:
        handle.remove()

    g_shared = (
        g_shared_t[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    # --------------------------------------------------------
    # B. Independent token gradients
    # --------------------------------------------------------

    saved = {}

    handle = o_proj.register_forward_pre_hook(
        make_token_hook(saved)
    )

    try:
        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        mu = differentiable_mu(
            out.logits,
            numeral_ids,
        )

        g_token_t = torch.autograd.grad(
            outputs=mu,
            inputs=saved["leaf"],
            retain_graph=False,
            create_graph=False,
        )[0]

    finally:
        handle.remove()

    g_token_sum = (
        g_token_t[0]
        .sum(dim=0)
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    g_arch = G_archive[idx]

    row = {
        "sample_id":
            sid,

        "shared_vs_token_cos":
            cosine(
                g_shared,
                g_token_sum,
            ),

        "shared_vs_token_rel_l2":
            rel_l2(
                g_shared,
                g_token_sum,
            ),

        "shared_vs_archive_cos":
            cosine(
                g_shared,
                g_arch,
            ),

        "shared_vs_archive_rel_l2":
            rel_l2(
                g_shared,
                g_arch,
            ),

        "shared_norm":
            float(
                np.linalg.norm(
                    g_shared
                )
            ),

        "token_sum_norm":
            float(
                np.linalg.norm(
                    g_token_sum
                )
            ),

        "archive_norm":
            float(
                np.linalg.norm(
                    g_arch
                )
            ),
    }

    records.append(row)

    print()
    print(sid)

    print(
        "  shared vs token-sum:"
        f" cos={row['shared_vs_token_cos']:.10f}"
        f" relL2={row['shared_vs_token_rel_l2']:.3e}"
    )

    print(
        "  shared vs archive:  "
        f" cos={row['shared_vs_archive_cos']:.10f}"
        f" relL2={row['shared_vs_archive_rel_l2']:.3e}"
    )


res = pd.DataFrame(records)

print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)

print(
    res.to_string(
        index=False
    )
)

OUT = (
    SA
    / "autograd_chainrule_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

res.to_csv(
    OUT / "chainrule_results.csv",
    index=False,
)

print()
print(
    "Median shared-vs-token cosine:",
    float(
        res[
            "shared_vs_token_cos"
        ].median()
    )
)

print(
    "Median shared-vs-token relL2:",
    float(
        res[
            "shared_vs_token_rel_l2"
        ].median()
    )
)

print(
    "Median shared-vs-archive cosine:",
    float(
        res[
            "shared_vs_archive_cos"
        ].median()
    )
)

print(
    "Median shared-vs-archive relL2:",
    float(
        res[
            "shared_vs_archive_rel_l2"
        ].median()
    )
)

print()
print(
    "AUTOGRAD CHAIN-RULE AUDIT COMPLETE"
)
