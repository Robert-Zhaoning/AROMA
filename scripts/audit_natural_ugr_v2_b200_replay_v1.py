from pathlib import Path
import json
import hashlib

import joblib
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

import sys
sys.path.insert(0, "scripts")

from run_l18h13_gain_all300 import (
    MODEL_ID,
    move_inputs,
)

from expanded_numeral_metrics import (
    numeral_token_ids,
    score_state,
)

from run_proc_count_causal_v3_final import (
    state_to_features,
    controller_decision,
)

from run_tallyqa_natural_confirmation_v2_final import (
    prepare_tallyqa_inputs,
)


ROOT = Path("/workspace/AromaExperiments")

POP = (
    ROOT
    / "manifests/natural_ugr_v2/"
      "final_stage2_population_v1.csv"
)

IMAGE_ROOT = (
    ROOT
    / "data/natural_ugr_final_v1/images"
)

CONTROLLER = (
    ROOT
    / "outputs/phase2_natural_controller_k500/"
      "final_frozen_controller/"
      "aroma_natural_utility_controller_k500.joblib"
)

OUT_DIR = (
    ROOT
    / "artifacts/natural_ugr_v2/final/stage2/"
      "b200_baseline_replay_v1"
)

OUT_CSV = (
    OUT_DIR
    / "b200_baseline_replay_v1.csv"
)

OUT_JSON = (
    OUT_DIR
    / "b200_baseline_replay_v1.json"
)

MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

DUMMY_GT = 1


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


df = pd.read_csv(POP)

if len(df) != 631:
    raise RuntimeError(
        f"Expected 631 samples, got {len(df)}"
    )

bundle = joblib.load(
    CONTROLLER
)

feature_names = list(
    bundle["feature_names"]
)

if len(feature_names) != 39:
    raise RuntimeError(
        f"Expected 39 controller features, got {len(feature_names)}"
    )


print("=" * 78)
print("NATURAL UGR V2 — B200 BASELINE / CONTROLLER REPLAY")
print("=" * 78)

print("N:", len(df))
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)


processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
)

numeral_ids = numeral_token_ids(
    processor
)

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


rows = []

for idx, r in df.iterrows():

    qid = int(
        r["question_id"]
    )

    image_path = (
        IMAGE_ROOT
        /
        str(r["image"])
    )

    with Image.open(
        image_path
    ) as im:

        image = im.convert(
            "RGB"
        )

        inputs = prepare_tallyqa_inputs(
            processor,
            image,
            str(r["question"]),
        )

    inputs = move_inputs(
        inputs,
        model,
    )

    with torch.inference_mode():

        state = score_state(
            model,
            inputs,
            numeral_ids,
            DUMMY_GT,
        )

    X, _ = state_to_features(
        state,
        feature_names,
    )

    (
        live_alpha,
        live_score,
        _,
    ) = controller_decision(
        bundle,
        X,
    )

    live_pred = int(
        state["best_numeral"]
    )

    archived_pred = int(
        r["baseline_prediction"]
    )

    archived_alpha = float(
        r["selected_alpha"]
    )

    pred_match = (
        live_pred
        ==
        archived_pred
    )

    alpha_match = np.isclose(
        float(live_alpha),
        archived_alpha,
        atol=1e-12,
        rtol=0.0,
    )

    rows.append({
        "stage2_index":
            int(idx),

        "question_id":
            qid,

        "archived_baseline":
            archived_pred,

        "live_b200_baseline":
            live_pred,

        "baseline_match":
            bool(pred_match),

        "archived_alpha":
            archived_alpha,

        "live_b200_alpha":
            float(live_alpha),

        "alpha_match":
            bool(alpha_match),

        "archived_score":
            float(
                r["selected_score"]
            ),

        "live_b200_score":
            float(live_score),

        "score_abs_diff":
            abs(
                float(live_score)
                -
                float(r["selected_score"])
            ),
    })

    if (
        (not pred_match)
        or
        (not alpha_match)
    ):
        print(
            f"MISMATCH [{idx+1:03d}/631] "
            f"qid={qid} "
            f"pred {archived_pred}->{live_pred} "
            f"alpha {archived_alpha}->{float(live_alpha)}"
        )

    elif (
        (idx + 1) % 50 == 0
        or
        idx == 0
    ):
        print(
            f"[{idx+1:03d}/631] replayed"
        )


out = pd.DataFrame(
    rows
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT_CSV,
    index=False,
    lineterminator="\n",
)


pred_mismatch = out[
    ~out["baseline_match"]
]

alpha_mismatch = out[
    ~out["alpha_match"]
]


summary = {
    "artifact":
        "Natural UGR v2 B200 baseline/controller replay v1",

    "n":
        int(len(out)),

    "baseline_prediction_matches":
        int(out["baseline_match"].sum()),

    "baseline_prediction_mismatches":
        int((~out["baseline_match"]).sum()),

    "controller_action_matches":
        int(out["alpha_match"].sum()),

    "controller_action_mismatches":
        int((~out["alpha_match"]).sum()),

    "max_selected_score_abs_difference":
        float(
            out["score_abs_diff"].max()
        ),

    "population_redefined":
        False,

    "router_retrained":
        False,

    "candidate_interventions_executed":
        False,

    "csv_sha256":
        None,
}


OUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

summary[
    "csv_sha256"
] = sha256(
    OUT_CSV
)

OUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)

print(
    "Baseline matches:",
    f"{int(out['baseline_match'].sum())}/631"
)

print(
    "Baseline mismatches:",
    int((~out["baseline_match"]).sum())
)

print(
    "Controller-action matches:",
    f"{int(out['alpha_match'].sum())}/631"
)

print(
    "Controller-action mismatches:",
    int((~out["alpha_match"]).sum())
)

print(
    "Max score abs diff:",
    float(
        out["score_abs_diff"].max()
    )
)

if len(pred_mismatch):

    print()
    print("BASELINE MISMATCHES:")
    print(
        pred_mismatch[
            [
                "stage2_index",
                "question_id",
                "archived_baseline",
                "live_b200_baseline",
                "archived_alpha",
                "live_b200_alpha",
            ]
        ].to_string(
            index=False
        )
    )

if len(alpha_mismatch):

    print()
    print("ACTION MISMATCHES:")
    print(
        alpha_mismatch[
            [
                "stage2_index",
                "question_id",
                "archived_alpha",
                "live_b200_alpha",
            ]
        ].to_string(
            index=False
        )
    )

print()
print(
    "B200 REPLAY AUDIT COMPLETE"
)
print(
    "No candidate intervention executed."
)
