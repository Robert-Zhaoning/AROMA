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

sys.path.insert(0, "scripts")

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


MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

V3_RESULTS = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller/"
    "v3_final_results.csv"
)

IMAGE_DIR = Path(
    "data/proc_count_causal_v3/images"
)

U4_PATH = Path(
    "outputs/aroma2/csa_gate_a_v3/"
    "U4_primary.npy"
)

OUT_DIR = Path(
    "outputs/aroma2/"
    "csa_actuation_strength_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DUMMY_GT = 1

START = 1664
END = 1792


df = pd.read_csv(
    V3_RESULTS
)

df["selected_alpha"] = (
    df["selected_alpha"]
    .astype(float)
)

data = (
    df[
        ~np.isclose(
            df["selected_alpha"],
            1.0,
        )
    ]
    .copy()
    .sort_values("sample_id")
    .reset_index(drop=True)
)

if len(data) != 653:
    raise RuntimeError(
        f"Expected 653 non-NOOP samples, "
        f"got {len(data)}"
    )


U = np.load(
    U4_PATH
).astype(np.float64)

if U.shape != (128, 4):
    raise RuntimeError(
        f"Bad U shape: {U.shape}"
    )

print("=" * 88)
print("CSA ACTUATION-STRENGTH AUDIT v1")
print("=" * 88)

print("N:", len(data))
print("U shape:", U.shape)

print()
print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
)

numeral_ids = numeral_token_ids(
    processor
)

print("Loading model...")

model = (
    MllamaForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
)

model.eval()

layer = (
    model
    .model
    .language_model
    .layers[LAYER]
)

o_proj = layer.cross_attn.o_proj


records = []


for _, row in tqdm(
    data.iterrows(),
    total=len(data),
):

    sid = str(
        row["sample_id"]
    )

    image_path = (
        IMAGE_DIR
        / f"{sid}.png"
    )

    image = (
        Image.open(image_path)
        .convert("RGB")
    )

    inputs = prepare_inputs(
        processor,
        image,
    )

    inputs = move_inputs(
        inputs,
        model,
    )

    saved = {}


    def hook(
        module,
        hook_inputs,
    ):

        x = hook_inputs[0]

        H = x[
            ...,
            START:END
        ]

        h = (
            H
            .detach()
            .float()
            .mean(dim=1)[0]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        saved["h"] = h


    handle = (
        o_proj
        .register_forward_pre_hook(
            hook
        )
    )

    try:

        # Canonical forward.
        score_state(
            model,
            inputs,
            numeral_ids,
            DUMMY_GT,
        )

    finally:

        handle.remove()


    if "h" not in saved:
        raise RuntimeError(
            f"Head activation not captured: {sid}"
        )


    h = saved["h"]

    projected = (
        U
        @ (
            U.T
            @ h
        )
    )

    h_norm = float(
        np.linalg.norm(h)
    )

    projected_norm = float(
        np.linalg.norm(projected)
    )


    if h_norm <= 0:
        raise RuntimeError(
            f"Zero h norm: {sid}"
        )


    ratio = (
        projected_norm
        / h_norm
    )

    energy_share = (
        ratio ** 2
    )


    alpha = float(
        row["selected_alpha"]
    )


    records.append(
        {
            "sample_id":
                sid,

            "condition":
                str(
                    row["condition"]
                ),

            "selected_alpha":
                alpha,

            "h_norm":
                h_norm,

            "projected_h_norm":
                projected_norm,

            "csa_to_whole_shift_ratio":
                ratio,

            "projected_activation_energy_share":
                energy_share,

            "whole_mean_shift_norm":
                abs(alpha - 1.0)
                * h_norm,

            "csa_mean_shift_norm":
                abs(alpha - 1.0)
                * projected_norm,
        }
    )


result = pd.DataFrame(
    records
)

result.to_csv(
    OUT_DIR
    / "actuation_strength.csv",
    index=False,
)


ratio = result[
    "csa_to_whole_shift_ratio"
].to_numpy()


print()
print("=" * 88)
print("ACTUATION-STRENGTH RESULT")
print("=" * 88)

print(
    "ratio = ||P_U h|| / ||h||"
)

for q in [
    0.0,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    1.0,
]:

    print(
        f"q{int(q*100):02d}:",
        float(
            np.quantile(
                ratio,
                q,
            )
        ),
    )


print()
print(
    "mean ratio:",
    float(
        ratio.mean()
    )
)

print(
    "median energy share:",
    float(
        np.median(
            ratio ** 2
        )
    )
)


print()
print("BY ACTION")

by_action = (
    result
    .groupby(
        "selected_alpha"
    )[
        "csa_to_whole_shift_ratio"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    by_action.to_string()
)


print()
print("BY CONDITION")

by_condition = (
    result
    .groupby(
        "condition"
    )[
        "csa_to_whole_shift_ratio"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    by_condition.to_string()
)

print()
print(
    "Saved:",
    OUT_DIR
    / "actuation_strength.csv"
)

print("=" * 88)

