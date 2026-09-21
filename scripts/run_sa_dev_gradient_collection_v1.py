import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


# ============================================================
# FROZEN CONFIG
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

PROTOCOL = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_vs_mn_protocol_v1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "39e61d47edbe9625e93e5f18b2c6f5a9"
    "fd9b25e1b19a731b50908d8eb2c61869"
)

META = Path(
    "data/proc_count_sa_dev_v1/metadata.jsonl"
)

CANONICAL_RUNNER = Path(
    "scripts/run_l18h13_gain_all300.py"
)

OUT = Path(
    "outputs/proc_count_sa_dev_v1/"
    "sa_gradient_collection_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = OUT / "partial_collection.npz"

LAYER = 18
HEAD = 13
HEAD_DIM = 128
HIDDEN_SIZE = 4096

START = HEAD * HEAD_DIM
END = START + HEAD_DIM


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def load_canonical():
    spec = importlib.util.spec_from_file_location(
        "canonical_l18h13",
        CANONICAL_RUNNER,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


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

    return (
        values
        * probs
    ).sum()


# ============================================================
# PROTOCOL INTEGRITY
# ============================================================

actual_protocol_hash = sha256(
    PROTOCOL
)

if (
    actual_protocol_hash
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Protocol hash mismatch:\n"
        f"expected={EXPECTED_PROTOCOL_SHA256}\n"
        f"actual={actual_protocol_hash}"
    )


protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

actual_meta_hash = sha256(
    META
)

if (
    actual_meta_hash
    != protocol[
        "dataset_metadata_sha256"
    ]
):
    raise RuntimeError(
        "Dataset metadata hash does not match "
        "frozen protocol."
    )


# ============================================================
# DATASET
# ============================================================

records = load_jsonl(
    META
)

if len(records) != 500:
    raise RuntimeError(
        f"Expected 500 samples, got {len(records)}"
    )

records = sorted(
    records,
    key=lambda r: str(r["sample_id"]),
)

sample_ids = np.array(
    [
        str(r["sample_id"])
        for r in records
    ],
    dtype=object,
)

if len(set(sample_ids.tolist())) != 500:
    raise RuntimeError(
        "Sample IDs are not unique."
    )

for r in records:
    p = Path(r["image_path"])
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# RESUME STATE
# ============================================================

done_ids = []
gradients = []
pooled_h = []
mus = []
token_lengths = []

if PARTIAL.exists():

    p = np.load(
        PARTIAL,
        allow_pickle=True,
    )

    done_ids = (
        p["sample_ids"]
        .astype(str)
        .tolist()
    )

    gradients = [
        x.astype(np.float64)
        for x in p["gradients"]
    ]

    pooled_h = [
        x.astype(np.float64)
        for x in p["pooled_h"]
    ]

    mus = (
        p["mu"]
        .astype(np.float64)
        .tolist()
    )

    token_lengths = (
        p["token_lengths"]
        .astype(int)
        .tolist()
    )

    expected_prefix = (
        sample_ids[
            :len(done_ids)
        ]
        .astype(str)
        .tolist()
    )

    if done_ids != expected_prefix:
        raise RuntimeError(
            "Partial checkpoint IDs do not match "
            "current dataset prefix."
        )

    print(
        f"Resuming from {len(done_ids)}/500"
    )


# ============================================================
# MODEL
# ============================================================

canonical = load_canonical()

print(
    "Loading processor..."
)

processor = AutoProcessor.from_pretrained(
    canonical.MODEL_ID,
    revision=MODEL_REVISION,
)

numeral_ids = canonical.numeral_token_ids(
    processor
)

print(
    "Loading model..."
)

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

for param in model.parameters():
    param.requires_grad_(False)

layer = (
    model
    .model
    .language_model
    .layers[LAYER]
)

o_proj = layer.cross_attn.o_proj

print(
    "MODEL LOADED"
)


# ============================================================
# GATE-A SHARED-PERTURBATION HOOK
# ============================================================

def make_hook(saved):

    def hook(
        module,
        hook_inputs,
    ):
        if not hook_inputs:
            raise RuntimeError(
                "o_proj hook received no inputs."
            )

        x = hook_inputs[0]

        if x.shape[-1] != HIDDEN_SIZE:
            raise RuntimeError(
                f"Unexpected o_proj input shape: "
                f"{tuple(x.shape)}"
            )

        H = x[
            ...,
            START:END
        ]

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

        saved["h_base"] = h_base
        saved["h_leaf"] = h_leaf
        saved["token_length"] = int(
            H.shape[1]
        )
        saved["calls"] = (
            saved.get("calls", 0)
            + 1
        )

        if len(hook_inputs) == 1:
            return (x_new,)

        return (
            x_new,
            *hook_inputs[1:]
        )

    return hook


# ============================================================
# COLLECTION
# ============================================================

start_idx = len(
    done_ids
)

for idx in range(
    start_idx,
    len(records),
):

    record = records[idx]
    sid = str(
        record["sample_id"]
    )

    image = Image.open(
        record["image_path"]
    ).convert("RGB")

    inputs = canonical.prepare_inputs(
        processor,
        image,
    )

    inputs = canonical.move_inputs(
        inputs,
        model,
    )

    saved = {}

    handle = (
        o_proj
        .register_forward_pre_hook(
            make_hook(saved)
        )
    )

    try:

        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        if saved.get(
            "calls",
            0,
        ) <= 0:
            raise RuntimeError(
                f"{sid}: hook was never called."
            )

        mu = differentiable_mu(
            out.logits,
            numeral_ids,
        )

        g = torch.autograd.grad(
            outputs=mu,
            inputs=saved["h_leaf"],
            retain_graph=False,
            create_graph=False,
        )[0]

    finally:
        handle.remove()

    g_np = (
        g[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    h_np = (
        saved["h_base"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    if g_np.shape != (128,):
        raise RuntimeError(
            f"{sid}: bad gradient shape {g_np.shape}"
        )

    if h_np.shape != (128,):
        raise RuntimeError(
            f"{sid}: bad h shape {h_np.shape}"
        )

    if not np.all(
        np.isfinite(g_np)
    ):
        raise RuntimeError(
            f"{sid}: non-finite gradient."
        )

    if not np.all(
        np.isfinite(h_np)
    ):
        raise RuntimeError(
            f"{sid}: non-finite activation."
        )

    done_ids.append(
        sid
    )

    gradients.append(
        g_np
    )

    pooled_h.append(
        h_np
    )

    mus.append(
        float(
            mu.detach()
            .cpu()
            .item()
        )
    )

    token_lengths.append(
        int(
            saved["token_length"]
        )
    )

    np.savez_compressed(
        PARTIAL,
        sample_ids=np.array(
            done_ids,
            dtype=object,
        ),
        gradients=np.stack(
            gradients,
        ),
        pooled_h=np.stack(
            pooled_h,
        ),
        mu=np.array(
            mus,
            dtype=np.float64,
        ),
        token_lengths=np.array(
            token_lengths,
            dtype=np.int64,
        ),
    )

    print(
        f"[{idx + 1:03d}/500] "
        f"{sid} "
        f"mu={mus[-1]:.6f} "
        f"|g|={np.linalg.norm(g_np):.6e} "
        f"|h|={np.linalg.norm(h_np):.6f}",
        flush=True,
    )


# ============================================================
# FINALIZE
# ============================================================

G = np.stack(
    gradients
).astype(np.float64)

H = np.stack(
    pooled_h
).astype(np.float64)

TOK = np.array(
    token_lengths,
    dtype=np.int64,
)

MU = np.array(
    mus,
    dtype=np.float64,
)

IDS = np.array(
    done_ids,
    dtype=object,
)

if G.shape != (500, 128):
    raise RuntimeError(
        f"Bad final G shape: {G.shape}"
    )

if H.shape != (500, 128):
    raise RuntimeError(
        f"Bad final H shape: {H.shape}"
    )

if not np.array_equal(
    IDS.astype(str),
    sample_ids.astype(str),
):
    raise RuntimeError(
        "Final sample ordering mismatch."
    )

np.save(
    OUT / "gradient_matrix.npy",
    G,
)

np.save(
    OUT / "pooled_h_matrix.npy",
    H,
)

np.save(
    OUT / "sample_ids.npy",
    IDS,
)

np.save(
    OUT / "token_lengths.npy",
    TOK,
)

np.save(
    OUT / "mu.npy",
    MU,
)

metadata = {
    "experiment":
        "sa_dev_gradient_collection_v1",

    "n":
        500,

    "model_id":
        canonical.MODEL_ID,

    "model_revision":
        MODEL_REVISION,

    "layer":
        LAYER,

    "head":
        HEAD,

    "head_dim":
        HEAD_DIM,

    "hidden_slice":
        [
            START,
            END,
        ],

    "gradient_semantics":
        "shared additive perturbation across all "
        "token positions at L18H13 o_proj input",

    "protocol_sha256":
        actual_protocol_hash,

    "dataset_metadata_sha256":
        actual_meta_hash,

    "gradient_matrix_sha256":
        sha256(
            OUT / "gradient_matrix.npy"
        ),

    "pooled_h_matrix_sha256":
        sha256(
            OUT / "pooled_h_matrix.npy"
        ),
}

(
    OUT / "metadata.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 88)
print("SA-DEV GRADIENT COLLECTION COMPLETE")
print("=" * 88)

print(
    "Gradient matrix:",
    G.shape,
)

print(
    "Pooled h matrix:",
    H.shape,
)

print(
    "Gradient norm min/median/max:",
    float(
        np.linalg.norm(
            G,
            axis=1,
        ).min()
    ),
    float(
        np.median(
            np.linalg.norm(
                G,
                axis=1,
            )
        )
    ),
    float(
        np.linalg.norm(
            G,
            axis=1,
        ).max()
    ),
)

print(
    "h norm min/median/max:",
    float(
        np.linalg.norm(
            H,
            axis=1,
        ).min()
    ),
    float(
        np.median(
            np.linalg.norm(
                H,
                axis=1,
            )
        )
    ),
    float(
        np.linalg.norm(
            H,
            axis=1,
        ).max()
    ),
)

print(
    "Output:",
    OUT,
)

print("=" * 88)
