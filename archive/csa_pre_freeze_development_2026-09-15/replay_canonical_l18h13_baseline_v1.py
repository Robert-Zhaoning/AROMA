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


print("=" * 88)
print("AROMA CSA — CANONICAL L18H13 BASELINE REPLAY v1")
print("=" * 88)


# ============================================================
# Paths
# ============================================================

RUNNER_PATH = Path(
    "scripts/run_l18h13_gain_all300.py"
)

ARCHIVE_CSV = Path(
    "outputs/proc_count_causal_v1/router/"
    "l18h13_gain_all300/"
    "l18h13_gain_all300_results.csv"
)

OUT_DIR = Path(
    "outputs/aroma2/csa_canonical_replay_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Import the ORIGINAL canonical runner
# ============================================================

spec = importlib.util.spec_from_file_location(
    "canonical_l18h13_runner",
    RUNNER_PATH,
)

canonical = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    canonical
)


print("CANONICAL MODULE LOADED")

print(
    "MODEL_ID:",
    canonical.MODEL_ID,
)

print(
    "PROMPT repr:",
    repr(canonical.PROMPT),
)


# ============================================================
# Archived identity rows
# ============================================================

df = pd.read_csv(
    ARCHIVE_CSV
)

identity = df[
    np.isclose(
        df["alpha"].astype(float),
        1.0,
    )
].copy()

if len(identity) != 300:
    raise RuntimeError(
        f"Expected 300 identity rows, got {len(identity)}"
    )


# One dense/r00 example for each GT count 1..10
selected = []

for gt in range(1, 11):

    sample_id = (
        f"pccv1_n{gt:02d}_dense_r00"
    )

    match = identity[
        identity["sample_id"]
        == sample_id
    ]

    if len(match) != 1:
        raise RuntimeError(
            f"Expected one archive row for {sample_id}, "
            f"got {len(match)}"
        )

    selected.append(
        match.iloc[0]
    )


# ============================================================
# Resolve exact image
# ============================================================

def resolve_image(sample_id):

    candidates = list(
        Path(
            "data/proc_count_causal_v1/images"
        ).glob(
            sample_id + ".*"
        )
    )

    candidates = [
        p
        for p in candidates
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
            f"{sample_id}: expected exactly one image, "
            f"got {candidates}"
        )

    return candidates[0]


# ============================================================
# Load EXACTLY as canonical runner
# ============================================================

print()
print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    canonical.MODEL_ID
)

numeral_ids = canonical.numeral_token_ids(
    processor
)

print(
    "Numeral IDs:",
    numeral_ids,
)


print()
print("Loading model...")

model = (
    MllamaForConditionalGeneration
    .from_pretrained(
        canonical.MODEL_ID,
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
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Replay
# ============================================================

records = []


print()
print("=" * 88)
print("CANONICAL REPLAY")
print("=" * 88)


for row in selected:

    sample_id = str(
        row["sample_id"]
    )

    gt = int(
        row["ground_truth"]
    )

    image_path = resolve_image(
        sample_id
    )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    # --------------------------------------------
    # THIS CALL IS FROM THE ORIGINAL RUNNER
    # --------------------------------------------

    inputs = canonical.prepare_inputs(
        processor,
        image,
    )

    inputs = canonical.move_inputs(
        inputs,
        model,
    )

    # --------------------------------------------
    # THIS SCORING FUNCTION IS ALSO ORIGINAL
    # --------------------------------------------

    state = canonical.score_state(
        model,
        inputs,
        numeral_ids,
        gt,
    )


    current_probs = np.asarray(
        state["conditional_probs"],
        dtype=np.float64,
    )


    archive_probs = np.asarray(
        [
            float(
                row[
                    f"baseline_numprob_{n}"
                ]
            )
            for n in range(
                1,
                11,
            )
        ],
        dtype=np.float64,
    )


    current_mu = float(
        state["expected_numeral"]
    )

    archive_mu = float(
        row[
            "baseline_expected_numeral"
        ]
    )


    current_best = int(
        state["best_numeral"]
    )

    archive_best = int(
        row[
            "baseline_best_numeral"
        ]
    )


    mu_diff = abs(
        current_mu
        - archive_mu
    )

    prob_l1 = float(
        np.abs(
            current_probs
            - archive_probs
        ).sum()
    )

    prob_max = float(
        np.abs(
            current_probs
            - archive_probs
        ).max()
    )


    print(
        sample_id,
        f"archive_mu={archive_mu:.8f}",
        f"current_mu={current_mu:.8f}",
        f"|diff|={mu_diff:.8f}",
        f"archive_best={archive_best}",
        f"current_best={current_best}",
        f"L1={prob_l1:.8f}",
    )


    rec = {
        "sample_id":
            sample_id,

        "ground_truth":
            gt,

        "image_path":
            str(image_path),

        "archive_mu":
            archive_mu,

        "current_mu":
            current_mu,

        "mu_abs_diff":
            mu_diff,

        "archive_best":
            archive_best,

        "current_best":
            current_best,

        "best_match":
            archive_best
            == current_best,

        "probability_l1":
            prob_l1,

        "probability_max_abs_diff":
            prob_max,
    }


    for n in range(
        1,
        11,
    ):

        rec[
            f"archive_p{n}"
        ] = float(
            archive_probs[
                n - 1
            ]
        )

        rec[
            f"current_p{n}"
        ] = float(
            current_probs[
                n - 1
            ]
        )


    records.append(
        rec
    )


# ============================================================
# Summary
# ============================================================

result = pd.DataFrame(
    records
)

result.to_csv(
    OUT_DIR
    / "canonical_replay10.csv",
    index=False,
)


summary = {
    "n": int(
        len(result)
    ),

    "mean_mu_abs_diff":
        float(
            result[
                "mu_abs_diff"
            ].mean()
        ),

    "max_mu_abs_diff":
        float(
            result[
                "mu_abs_diff"
            ].max()
        ),

    "median_mu_abs_diff":
        float(
            result[
                "mu_abs_diff"
            ].median()
        ),

    "best_numeral_matches":
        int(
            result[
                "best_match"
            ].sum()
        ),

    "mean_probability_l1":
        float(
            result[
                "probability_l1"
            ].mean()
        ),

    "max_probability_l1":
        float(
            result[
                "probability_l1"
            ].max()
        ),

    "torch_version":
        torch.__version__,

    "gpu":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
}


with open(
    OUT_DIR
    / "summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=" * 88)
print("CANONICAL REPLAY SUMMARY")
print("=" * 88)

for k, v in summary.items():
    print(
        f"{k}: {v}"
    )


print()
print(
    "Saved:",
    OUT_DIR
    / "canonical_replay10.csv",
)

print(
    "Saved:",
    OUT_DIR
    / "summary.json",
)

print("=" * 88)

