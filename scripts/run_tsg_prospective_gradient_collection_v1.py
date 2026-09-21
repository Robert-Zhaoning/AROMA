import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


# ============================================================
# FROZEN CONFIG
# ============================================================

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

FREEZE_RECORD = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "freeze_v1/"
    "tsg_prospective_cohort_freeze_v1.json"
)

EXPECTED_FREEZE_SHA256 = (
    "8e863fd29561ca03862e00c482ecdb541eb1e15382"
    "d366f218208eea81b50b22"
)

PROTOCOL = Path(
    "outputs/proc_count_sa_dev_v1/"
    "tsg_protocol_v1/"
    "tsg_sa_protocol_v1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "ed04264d7b2ac2c03b998d2f128924b997a1e1b06"
    "b841c510c814d469c5d8938"
)

META = Path(
    "data/proc_count_tsg_prospective_v1/"
    "metadata.jsonl"
)

EXPECTED_META_SHA256 = (
    "b7e7cb1f2c58cffdfea06aa77a4fceeab64de7102"
    "d70b2fc99c3e3d731d72a0a"
)

CANONICAL_RUNNER = Path(
    "scripts/run_l18h13_gain_all300.py"
)

OUT = Path(
    "outputs/proc_count_tsg_prospective_v1/"
    "gradient_collection_v1"
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
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]


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
# FROZEN ARTIFACT AUDIT
# ============================================================

if sha256(FREEZE_RECORD) != EXPECTED_FREEZE_SHA256:
    raise RuntimeError(
        "Freeze-record SHA256 mismatch."
    )

if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError(
        "TSG protocol SHA256 mismatch."
    )

if sha256(META) != EXPECTED_META_SHA256:
    raise RuntimeError(
        "Prospective metadata SHA256 mismatch."
    )

freeze = json.loads(
    FREEZE_RECORD.read_text(
        encoding="utf-8"
    )
)

if (
    freeze[
        "dataset_metadata_sha256"
    ]
    != EXPECTED_META_SHA256
):
    raise RuntimeError(
        "Freeze record contains unexpected "
        "dataset metadata hash."
    )

if (
    freeze[
        "tsg_protocol_sha256"
    ]
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Freeze record contains unexpected "
        "protocol hash."
    )

frozen_algorithm = freeze[
    "frozen_algorithm"
]

if not np.isclose(
    float(
        frozen_algorithm["tau"]
    ),
    1.0 / 3.0,
):
    raise RuntimeError(
        "Frozen tau mismatch."
    )

if not np.isclose(
    float(
        frozen_algorithm["s_ref"]
    ),
    0.06780448298801545,
):
    raise RuntimeError(
        "Frozen s_ref mismatch."
    )

print(
    "PASS: prospective freeze record verified"
)


# ============================================================
# DATASET
# ============================================================

records = load_jsonl(
    META
)

if len(records) != 500:
    raise RuntimeError(
        f"Expected N=500, got {len(records)}"
    )

records = sorted(
    records,
    key=lambda r: str(
        r["sample_id"]
    ),
)

sample_ids = np.array(
    [
        str(r["sample_id"])
        for r in records
    ],
    dtype=object,
)

if len(
    set(
        sample_ids.tolist()
    )
) != 500:
    raise RuntimeError(
        "Sample IDs are not unique."
    )

for r in records:

    path = Path(
        r["image_path"]
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

print(
    "PASS: exact prospective N=500 population"
)


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
            "Partial checkpoint IDs do not "
            "match prospective dataset prefix."
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

o_proj = (
    layer
    .cross_attn
    .o_proj
)

print(
    "MODEL LOADED"
)


# ============================================================
# EXACT GATE-A SHARED-PERTURBATION SEMANTICS
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
            -
            h_base.detach()
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

        saved[
            "h_base"
        ] = h_base

        saved[
            "h_leaf"
        ] = h_leaf

        saved[
            "token_length"
        ] = int(
            H.shape[1]
        )

        saved[
            "calls"
        ] = (
            saved.get(
                "calls",
                0,
            )
            + 1
        )

        if len(hook_inputs) == 1:
            return (x_new,)

        return (
            x_new,
            *hook_inputs[1:],
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

    image = (
        Image.open(
            record["image_path"]
        )
        .convert("RGB")
    )

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
        saved[
            "h_base"
        ][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    if g_np.shape != (128,):
        raise RuntimeError(
            f"{sid}: bad gradient shape "
            f"{g_np.shape}"
        )

    if h_np.shape != (128,):
        raise RuntimeError(
            f"{sid}: bad h shape "
            f"{h_np.shape}"
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
            saved[
                "token_length"
            ]
        )
    )

    np.savez_compressed(
        PARTIAL,
        sample_ids=np.array(
            done_ids,
            dtype=object,
        ),
        gradients=np.stack(
            gradients
        ),
        pooled_h=np.stack(
            pooled_h
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
        "tsg_prospective_gradient_collection_v1",

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

    "hidden_slice": [
        START,
        END,
    ],

    "gradient_semantics":
        "shared additive perturbation across all "
        "token positions at L18H13 o_proj input",

    "freeze_record_sha256":
        sha256(FREEZE_RECORD),

    "protocol_sha256":
        sha256(PROTOCOL),

    "dataset_metadata_sha256":
        sha256(META),

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
print(
    "TSG PROSPECTIVE GRADIENT COLLECTION COMPLETE"
)
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
    "Gradient SHA256:",
    sha256(
        OUT / "gradient_matrix.npy"
    ),
)

print(
    "Pooled-h SHA256:",
    sha256(
        OUT / "pooled_h_matrix.npy"
    ),
)

print("=" * 88)
