import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
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

META = Path(
    "data/proc_count_utility_final_v1/"
    "metadata.jsonl"
)

EXPECTED_META_SHA256 = (
    "700b20fe99944c9d25cb4b80873156b796b4bba90f4af"
    "743427a4195bb040231"
)

ACTION_MANIFEST = Path(
    "outputs/proc_count_utility_final_v1/"
    "action_manifest_v1/"
    "utility_final_action_manifest_v1.csv"
)

EXPECTED_ACTION_SHA256 = (
    "3123d61cfca130376a2b911707ea20927872b1a270ac1"
    "fbd5a852d3e82fc3040"
)

ACTION_FREEZE = Path(
    "outputs/proc_count_utility_final_v1/"
    "action_manifest_v1/"
    "freeze/"
    "utility_final_action_manifest_freeze_v1.json"
)

EXPECTED_ACTION_FREEZE_SHA256 = (
    "4fb05b5d1c91ab3613432ee580126dcd504dd59abf875"
    "eb5a56ec79e4745edef"
)

PROTOCOL = Path(
    "outputs/aroma2/"
    "utility_router_final_confirmation_v1/"
    "final_confirmation_protocol.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "7a37d80f53e86944631571799d55b8eac8d4873ce74cf"
    "1f9a84d11c118654922"
)

NAMESPACE = Path(
    "outputs/proc_count_utility_final_v1/"
    "namespace_v1/"
    "utility_final_namespace_v1.csv"
)

EXPECTED_NAMESPACE_SHA256 = (
    "a5a6af6dd3f5103f6586de1e79b7f76570222b202bb14"
    "c98174f9283f3017937"
)

UTILITY_ROUTER = Path(
    "outputs/proc_count_sa_dev_v1/"
    "utility_router_final_v1/"
    "aroma_utility_router_v1.joblib"
)

EXPECTED_UTILITY_ROUTER_SHA256 = (
    "4192129c525822fed7ab9d013dc3d564981f48faf81f28"
    "b3011a4be1a2168563"
)

CANONICAL_RUNNER = Path(
    "scripts/run_l18h13_gain_all300.py"
)

OUT = Path(
    "outputs/proc_count_utility_final_v1/"
    "gradient_collection_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARTIAL = (
    OUT
    / "partial_collection.npz"
)

EXPECTED_N = 294

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

            block = f.read(
                1024 * 1024
            )

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

    spec = (
        importlib.util
        .spec_from_file_location(
            "canonical_l18h13",
            CANONICAL_RUNNER,
        )
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
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
            last[
                numeral_ids[n]
            ]
            for n in range(
                1,
                11,
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

    return (
        values
        *
        probs
    ).sum()


# ============================================================
# HARD FROZEN-ARTIFACT AUDIT
# ============================================================

checks = [
    (
        META,
        EXPECTED_META_SHA256,
        "dataset metadata",
    ),
    (
        ACTION_MANIFEST,
        EXPECTED_ACTION_SHA256,
        "action manifest",
    ),
    (
        ACTION_FREEZE,
        EXPECTED_ACTION_FREEZE_SHA256,
        "action freeze",
    ),
    (
        PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        "final protocol",
    ),
    (
        NAMESPACE,
        EXPECTED_NAMESPACE_SHA256,
        "namespace",
    ),
    (
        UTILITY_ROUTER,
        EXPECTED_UTILITY_ROUTER_SHA256,
        "frozen utility router",
    ),
]

for path, expected, name in checks:

    actual = sha256(
        path
    )

    if actual != expected:

        raise RuntimeError(
            f"{name} SHA256 mismatch:\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


print(
    "PASS: all frozen final artifacts verified"
)


# ============================================================
# FINAL UPWARD POPULATION
# ============================================================

metadata_records = load_jsonl(
    META
)

if len(
    metadata_records
) != 1000:

    raise RuntimeError(
        "Expected final metadata N=1000."
    )


record_map = {
    str(
        r[
            "sample_id"
        ]
    ):
        r
    for r in metadata_records
}


actions = pd.read_csv(
    ACTION_MANIFEST
)

actions[
    "raw_sample_id"
] = (
    actions[
        "raw_sample_id"
    ].astype(str)
)

actions[
    "cohort_uid"
] = (
    actions[
        "cohort_uid"
    ].astype(str)
)


upward = actions[
    actions[
        "selected_alpha"
    ].astype(float)
    > 1.0
].copy()


upward = (
    upward
    .sort_values(
        "raw_sample_id"
    )
    .reset_index(
        drop=True
    )
)


if len(
    upward
) != EXPECTED_N:

    raise RuntimeError(
        f"Expected upward N={EXPECTED_N}; "
        f"got {len(upward)}"
    )


sample_ids = (
    upward[
        "raw_sample_id"
    ]
    .astype(str)
    .to_numpy(
        dtype=object
    )
)

cohort_uids = (
    upward[
        "cohort_uid"
    ]
    .astype(str)
    .to_numpy(
        dtype=object
    )
)


if len(
    set(
        sample_ids.tolist()
    )
) != EXPECTED_N:

    raise RuntimeError(
        "Upward sample IDs are not unique."
    )


records = []

for sid in sample_ids:

    sid = str(
        sid
    )

    if sid not in record_map:

        raise RuntimeError(
            f"{sid}: not found in metadata."
        )

    rec = record_map[
        sid
    ]

    path = Path(
        rec[
            "image_path"
        ]
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    records.append(
        rec
    )


print(
    "PASS: exact frozen upward population N=294"
)


# ============================================================
# RESUME
# ============================================================

done_ids = []
done_uids = []
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
        p[
            "sample_ids"
        ]
        .astype(str)
        .tolist()
    )

    done_uids = (
        p[
            "cohort_uids"
        ]
        .astype(str)
        .tolist()
    )

    gradients = [
        x.astype(
            np.float64
        )
        for x in p[
            "gradients"
        ]
    ]

    pooled_h = [
        x.astype(
            np.float64
        )
        for x in p[
            "pooled_h"
        ]
    ]

    mus = (
        p[
            "mu"
        ]
        .astype(
            np.float64
        )
        .tolist()
    )

    token_lengths = (
        p[
            "token_lengths"
        ]
        .astype(int)
        .tolist()
    )


    expected_ids = (
        sample_ids[
            :len(
                done_ids
            )
        ]
        .astype(str)
        .tolist()
    )

    expected_uids = (
        cohort_uids[
            :len(
                done_uids
            )
        ]
        .astype(str)
        .tolist()
    )


    if done_ids != expected_ids:

        raise RuntimeError(
            "Partial sample IDs do not match "
            "frozen upward prefix."
        )


    if done_uids != expected_uids:

        raise RuntimeError(
            "Partial cohort UIDs do not match "
            "frozen upward prefix."
        )


    print(
        f"Resuming from "
        f"{len(done_ids)}/{EXPECTED_N}"
    )


# ============================================================
# MODEL
# ============================================================

canonical = load_canonical()


print(
    "Loading processor..."
)

processor = (
    AutoProcessor
    .from_pretrained(
        canonical.MODEL_ID,
        revision=MODEL_REVISION,
    )
)


numeral_ids = (
    canonical.numeral_token_ids(
        processor
    )
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


for param in (
    model.parameters()
):

    param.requires_grad_(
        False
    )


layer = (
    model
    .model
    .language_model
    .layers[
        LAYER
    ]
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
# EXACT VALIDATED GATE-A SEMANTICS
# ============================================================

def make_hook(saved):

    def hook(
        module,
        hook_inputs,
    ):

        if not hook_inputs:

            raise RuntimeError(
                "o_proj hook received "
                "no inputs."
            )


        x = hook_inputs[
            0
        ]


        if x.shape[
            -1
        ] != HIDDEN_SIZE:

            raise RuntimeError(
                "Unexpected o_proj "
                f"input shape: "
                f"{tuple(x.shape)}"
            )


        H = x[
            ...,
            START:END
        ]


        h_base = (
            H.detach()
            .float()
            .mean(
                dim=1
            )
        )


        h_leaf = (
            h_base
            .clone()
            .requires_grad_(
                True
            )
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
            .to(
                H.dtype
            )
            .unsqueeze(
                1
            )
        )


        x_new = torch.cat(
            [
                x[
                    ...,
                    :START
                ].detach(),

                H_new,

                x[
                    ...,
                    END:
                ].detach(),
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
            H.shape[
                1
            ]
        )

        saved[
            "calls"
        ] = (
            saved.get(
                "calls",
                0,
            )
            +
            1
        )


        if len(
            hook_inputs
        ) == 1:

            return (
                x_new,
            )


        return (
            x_new,
            *hook_inputs[
                1:
            ],
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
    EXPECTED_N,
):

    record = records[
        idx
    ]

    sid = str(
        sample_ids[
            idx
        ]
    )

    uid = str(
        cohort_uids[
            idx
        ]
    )


    image = (
        Image.open(
            record[
                "image_path"
            ]
        )
        .convert(
            "RGB"
        )
    )


    inputs = (
        canonical
        .prepare_inputs(
            processor,
            image,
        )
    )


    inputs = (
        canonical
        .move_inputs(
            inputs,
            model,
        )
    )


    saved = {}


    handle = (
        o_proj
        .register_forward_pre_hook(
            make_hook(
                saved
            )
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
                f"{sid}: hook "
                "was never called."
            )


        mu = (
            differentiable_mu(
                out.logits,
                numeral_ids,
            )
        )


        g = torch.autograd.grad(
            outputs=mu,
            inputs=saved[
                "h_leaf"
            ],
            retain_graph=False,
            create_graph=False,
        )[0]


    finally:

        handle.remove()


    g_np = (
        g[
            0
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


    h_np = (
        saved[
            "h_base"
        ][
            0
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


    if g_np.shape != (
        128,
    ):

        raise RuntimeError(
            f"{sid}: bad gradient "
            f"shape {g_np.shape}"
        )


    if h_np.shape != (
        128,
    ):

        raise RuntimeError(
            f"{sid}: bad h "
            f"shape {h_np.shape}"
        )


    if not np.isfinite(
        g_np
    ).all():

        raise RuntimeError(
            f"{sid}: non-finite "
            "gradient."
        )


    if not np.isfinite(
        h_np
    ).all():

        raise RuntimeError(
            f"{sid}: non-finite "
            "activation."
        )


    done_ids.append(
        sid
    )

    done_uids.append(
        uid
    )

    gradients.append(
        g_np
    )

    pooled_h.append(
        h_np
    )

    mus.append(
        float(
            mu
            .detach()
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

        cohort_uids=np.array(
            done_uids,
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
        f"[{idx + 1:03d}/"
        f"{EXPECTED_N}] "
        f"{sid} "
        f"mu={mus[-1]:.6f} "
        f"|g|="
        f"{np.linalg.norm(g_np):.6e} "
        f"|h|="
        f"{np.linalg.norm(h_np):.6f}",
        flush=True,
    )


# ============================================================
# FINALIZE
# ============================================================

G = np.stack(
    gradients
).astype(
    np.float64
)

H = np.stack(
    pooled_h
).astype(
    np.float64
)

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

UIDS = np.array(
    done_uids,
    dtype=object,
)


if G.shape != (
    EXPECTED_N,
    128,
):

    raise RuntimeError(
        f"Bad final G shape: "
        f"{G.shape}"
    )


if H.shape != (
    EXPECTED_N,
    128,
):

    raise RuntimeError(
        f"Bad final H shape: "
        f"{H.shape}"
    )


if not np.array_equal(
    IDS.astype(str),
    sample_ids.astype(str),
):

    raise RuntimeError(
        "Final sample ordering "
        "mismatch."
    )


if not np.array_equal(
    UIDS.astype(str),
    cohort_uids.astype(str),
):

    raise RuntimeError(
        "Final cohort UID ordering "
        "mismatch."
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
    OUT / "cohort_uids.npy",
    UIDS,
)

np.save(
    OUT / "token_lengths.npy",
    TOK,
)

np.save(
    OUT / "mu.npy",
    MU,
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "experiment":
        "utility_final_gradient_collection_v1",

    "population":
        "frozen selected_alpha > 1 "
        "final-confirmation samples",

    "n":
        EXPECTED_N,

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

    "dataset_metadata_sha256":
        sha256(
            META
        ),

    "action_manifest_sha256":
        sha256(
            ACTION_MANIFEST
        ),

    "action_freeze_sha256":
        sha256(
            ACTION_FREEZE
        ),

    "final_protocol_sha256":
        sha256(
            PROTOCOL
        ),

    "namespace_sha256":
        sha256(
            NAMESPACE
        ),

    "utility_router_sha256":
        sha256(
            UTILITY_ROUTER
        ),

    "gradient_matrix_sha256":
        sha256(
            OUT
            / "gradient_matrix.npy"
        ),

    "pooled_h_matrix_sha256":
        sha256(
            OUT
            / "pooled_h_matrix.npy"
        ),
}


(
    OUT
    / "metadata.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 96)
print(
    "UTILITY FINAL GRADIENT COLLECTION COMPLETE"
)
print("=" * 96)

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
        OUT
        / "gradient_matrix.npy"
    ),
)

print(
    "Pooled-h SHA256:",
    sha256(
        OUT
        / "pooled_h_matrix.npy"
    ),
)

print(
    "Metadata SHA256:",
    sha256(
        OUT
        / "metadata.json"
    ),
)

print("=" * 96)
