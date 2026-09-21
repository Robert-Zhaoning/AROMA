import importlib.util
import json
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

GATE = Path(
    "outputs/aroma2/csa_gate_a_v3"
)

SA = Path(
    "outputs/aroma2/sa_direction_diagnostic_v1"
)

OUT = SA / "finite_difference_v1"
OUT.mkdir(parents=True, exist_ok=True)

LAYER = 18
HEAD = 13

HEAD_DIM = 128
HIDDEN_SIZE = 4096

START = HEAD * HEAD_DIM
END = START + HEAD_DIM

# Tiny perturbations expressed as a fraction of ||h||.
REL_EPS = [
    0.0025,
    0.0050,
    0.0100,
]

# Five deterministic positions spanning the observed
# MN-vs-SA directional cosine distribution.
COS_QUANTILES = [
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
]


# ============================================================
# CANONICAL RUNNER
# ============================================================

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


# ============================================================
# DIFFERENTIABLE/CANONICAL MU
# ============================================================

def numeral_mu(logits, numeral_ids):
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
# LOAD ARCHIVED G / H / U
# ============================================================

G = np.load(
    GATE / "gradient_matrix.npy"
).astype(np.float64)

H = np.load(
    SA / "pooled_h_matrix.npy"
).astype(np.float64)

U = np.load(
    GATE / "U4_primary.npy"
).astype(np.float64)

gate_partial = np.load(
    GATE / "partial_collection.npz",
    allow_pickle=True,
)

sample_ids = (
    gate_partial["sample_ids"]
    .astype(str)
)

h_ids = (
    np.load(
        SA / "sample_ids.npy",
        allow_pickle=True,
    )
    .astype(str)
)

if G.shape != (300, 128):
    raise RuntimeError(
        f"Bad G shape: {G.shape}"
    )

if H.shape != (300, 128):
    raise RuntimeError(
        f"Bad H shape: {H.shape}"
    )

if U.shape != (128, 4):
    raise RuntimeError(
        f"Bad U shape: {U.shape}"
    )

if not np.array_equal(
    sample_ids,
    h_ids,
):
    raise RuntimeError(
        "G/H sample IDs mismatch"
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
        "Source population does not match Gate-A IDs"
    )


# ============================================================
# PROJECT / COSINE
# ============================================================

P = U @ U.T

Ug = G @ P
Uh = H @ P

g_norm = np.linalg.norm(
    Ug,
    axis=1,
)

h_u_norm = np.linalg.norm(
    Uh,
    axis=1,
)

cos = np.sum(
    Ug * Uh,
    axis=1,
) / (
    np.maximum(
        g_norm * h_u_norm,
        1e-12,
    )
)


# ============================================================
# DETERMINISTIC REPRESENTATIVE SAMPLE SELECTION
# ============================================================

order = np.argsort(cos)

selected = []

for q in COS_QUANTILES:

    pos = int(
        round(
            q
            * (len(order) - 1)
        )
    )

    idx = int(
        order[pos]
    )

    selected.append(
        (
            q,
            idx,
        )
    )

# Hard unique check.
indices = [
    x[1]
    for x in selected
]

if len(set(indices)) != len(indices):
    raise RuntimeError(
        "Representative sample selection duplicated indices"
    )

print("=" * 92)
print("SA FINITE-DIFFERENCE GRADIENT CHECK v1")
print("=" * 92)

print()
print("Selected samples:")

for q, idx in selected:
    print(
        f"  cosine q≈{q:.2f} | "
        f"sample={sample_ids[idx]} | "
        f"cos={cos[idx]:+.6f}"
    )


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


# ============================================================
# INPUT PREPARATION
# ============================================================

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
# SHARED ADDITIVE INTERVENTION
# ============================================================

def make_delta_hook(
    delta_np,
    saved,
):
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
                f"Bad o_proj input shape: "
                f"{tuple(x.shape)}"
            )

        H_head = x[
            ...,
            START:END
        ]

        delta = torch.as_tensor(
            delta_np,
            device=H_head.device,
            dtype=H_head.dtype,
        ).view(
            1,
            1,
            HEAD_DIM,
        )

        H_new = (
            H_head
            + delta
        )

        x_new = torch.cat(
            [
                x[..., :START],
                H_new,
                x[..., END:],
            ],
            dim=-1,
        )

        saved["calls"] = (
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
            *hook_inputs[1:]
        )

    return hook


def score_mu(
    inputs,
    delta_np=None,
):
    if delta_np is None:
        with torch.no_grad():
            out = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

        return float(
            numeral_mu(
                out.logits,
                numeral_ids,
            ).item()
        )

    saved = {}

    handle = (
        o_proj
        .register_forward_pre_hook(
            make_delta_hook(
                delta_np,
                saved,
            )
        )
    )

    try:
        with torch.no_grad():
            out = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()

    if saved.get(
        "calls",
        0,
    ) <= 0:
        raise RuntimeError(
            "Intervention hook was not called"
        )

    return float(
        numeral_mu(
            out.logits,
            numeral_ids,
        ).item()
    )


# ============================================================
# FINITE DIFFERENCE
# ============================================================

records = []

total = (
    len(selected)
    * 2
    * len(REL_EPS)
)

progress = tqdm(
    total=total,
)

for q, idx in selected:

    sid = str(
        sample_ids[idx]
    )

    g = G[idx]
    h = H[idx]

    ug = Ug[idx]
    uh = Uh[idx]

    ug_norm = float(
        np.linalg.norm(ug)
    )

    uh_norm = float(
        np.linalg.norm(uh)
    )

    h_norm = float(
        np.linalg.norm(h)
    )

    if ug_norm <= 1e-12:
        raise RuntimeError(
            f"{sid}: zero projected gradient"
        )

    if uh_norm <= 1e-12:
        raise RuntimeError(
            f"{sid}: zero projected activation"
        )

    d_sa = (
        ug
        / ug_norm
    )

    d_mn = (
        uh
        / uh_norm
    )

    inputs = get_inputs(
        sid
    )

    mu0 = score_mu(
        inputs,
        None,
    )

    for direction_name, d in [
        (
            "SA",
            d_sa,
        ),
        (
            "MN",
            d_mn,
        ),
    ]:

        predicted = float(
            np.dot(
                g,
                d,
            )
        )

        for rel_eps in REL_EPS:

            eps = float(
                rel_eps
                * h_norm
            )

            delta_plus = (
                eps
                * d
            )

            delta_minus = (
                -eps
                * d
            )

            mu_plus = score_mu(
                inputs,
                delta_plus,
            )

            mu_minus = score_mu(
                inputs,
                delta_minus,
            )

            fd = float(
                (
                    mu_plus
                    - mu_minus
                )
                /
                (
                    2.0
                    * eps
                )
            )

            abs_error = float(
                abs(
                    fd
                    - predicted
                )
            )

            rel_error = float(
                abs_error
                /
                max(
                    abs(predicted),
                    1e-8,
                )
            )

            if (
                abs(predicted) < 1e-10
                or abs(fd) < 1e-10
            ):
                sign_match = True
            else:
                sign_match = bool(
                    np.sign(predicted)
                    ==
                    np.sign(fd)
                )

            ratio = (
                float(
                    fd / predicted
                )
                if abs(predicted) > 1e-10
                else np.nan
            )

            symmetry_error = float(
                abs(
                    (
                        mu_plus
                        + mu_minus
                    )
                    / 2.0
                    - mu0
                )
            )

            records.append(
                {
                    "sample_id":
                        sid,

                    "cos_quantile":
                        q,

                    "cos_u4g_u4h":
                        float(
                            cos[idx]
                        ),

                    "direction":
                        direction_name,

                    "rel_eps":
                        rel_eps,

                    "eps_abs":
                        eps,

                    "h_norm":
                        h_norm,

                    "mu0":
                        mu0,

                    "mu_plus":
                        mu_plus,

                    "mu_minus":
                        mu_minus,

                    "predicted_derivative":
                        predicted,

                    "finite_difference":
                        fd,

                    "fd_over_predicted":
                        ratio,

                    "absolute_error":
                        abs_error,

                    "relative_error":
                        rel_error,

                    "sign_match":
                        int(
                            sign_match
                        ),

                    "symmetry_error":
                        symmetry_error,
                }
            )

            progress.update(1)

progress.close()


# ============================================================
# SUMMARY
# ============================================================

res = pd.DataFrame(
    records
)

res.to_csv(
    OUT / "finite_difference_results.csv",
    index=False,
)

print()
print("=" * 92)
print("FINITE-DIFFERENCE RESULTS")
print("=" * 92)

print(
    res[
        [
            "sample_id",
            "cos_u4g_u4h",
            "direction",
            "rel_eps",
            "predicted_derivative",
            "finite_difference",
            "fd_over_predicted",
            "relative_error",
            "sign_match",
        ]
    ].to_string(
        index=False
    )
)


summary_rows = []

for direction, group in res.groupby(
    "direction",
    sort=True,
):

    summary_rows.append(
        {
            "direction":
                direction,

            "N_checks":
                len(group),

            "sign_match_fraction":
                float(
                    group[
                        "sign_match"
                    ].mean()
                ),

            "median_relative_error":
                float(
                    group[
                        "relative_error"
                    ].median()
                ),

            "median_fd_over_predicted":
                float(
                    np.nanmedian(
                        group[
                            "fd_over_predicted"
                        ]
                    )
                ),

            "q25_fd_over_predicted":
                float(
                    np.nanquantile(
                        group[
                            "fd_over_predicted"
                        ],
                        0.25,
                    )
                ),

            "q75_fd_over_predicted":
                float(
                    np.nanquantile(
                        group[
                            "fd_over_predicted"
                        ],
                        0.75,
                    )
                ),
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT / "finite_difference_summary.csv",
    index=False,
)

print()
print("=" * 92)
print("SUMMARY")
print("=" * 92)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    OUT / "finite_difference_results.csv"
)

print(
    "Saved:",
    OUT / "finite_difference_summary.csv"
)

print()
print(
    "SA GRADIENT FINITE-DIFFERENCE CHECK COMPLETE"
)
