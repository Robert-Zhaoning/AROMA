import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


sys.path.insert(
    0,
    "scripts",
)


# ============================================================
# Reuse EXACT frozen development / v3 implementations.
# ============================================================

from run_l18h13_gain_all300 import (
    MODEL_ID,
    LAYER,
    HEAD,
    move_inputs,
    HeadGainModifier,
)

from expanded_numeral_metrics import (
    NUMERALS,
    numeral_token_ids,
    score_state,
)

from run_proc_count_causal_v3_final import (
    state_to_features,
    controller_decision,
)


# ============================================================
# Frozen paths
# ============================================================

MANIFEST_PATH = Path(
    "data/tallyqa_natural_ood_v1/"
    "manifest.jsonl"
)

IMAGE_ROOT = Path(
    "data/tallyqa_natural_ood_v1/"
    "images"
)

IMAGE_INVENTORY_PATH = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "image_inventory/image_inventory.csv"
)

OOD_PROTOCOL_PATH = Path(
    "configs/tallyqa_natural_ood_v1.json"
)

IMAGE_FREEZE_PATH = Path(
    "configs/tallyqa_natural_ood_v1_image_freeze.json"
)

RUNNER_PROTOCOL_PATH = Path(
    "configs/tallyqa_natural_ood_v1_final_runner.json"
)

CONTROLLER_MANIFEST_PATH = Path(
    "configs/aroma_cardinality_controller_frozen.json"
)

CONTROLLER_BUNDLE_PATH = Path(
    "outputs/proc_count_causal_v2/"
    "controller/final_frozen_controller/"
    "aroma_cardinality_controller.joblib"
)

V3_RUN_META_PATH = Path(
    "outputs/proc_count_causal_v3/"
    "final_frozen_controller/"
    "v3_final_run_metadata.json"
)


OUT_DIR = Path(
    "outputs/tallyqa_natural_ood_v1/"
    "final_frozen_controller"
)

RESULT_PATH = (
    OUT_DIR
    / "tallyqa_final_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "tallyqa_final_summary.csv"
)

SUBSET_PATH = (
    OUT_DIR
    / "tallyqa_final_by_subset.csv"
)

COUNT_PATH = (
    OUT_DIR
    / "tallyqa_final_by_count.csv"
)

ACTION_PATH = (
    OUT_DIR
    / "tallyqa_final_action_distribution.csv"
)

RUN_META_PATH = (
    OUT_DIR
    / "tallyqa_final_run_metadata.json"
)

RUN_STARTED_PATH = (
    OUT_DIR
    / "tallyqa_final_run_started.json"
)


# ============================================================
# Frozen hashes
# ============================================================

EXPECTED_MANIFEST_SHA256 = (
    "068a489a02d499da6eb0b837a81c8817583b64b51c23f944c77645420fe2c32c"
)

EXPECTED_INVENTORY_SHA256 = (
    "976a4f209d13d32ba5a412f7a943b4c798797a2c5cb45d561f7c182812a7f635"
)


# ============================================================
# Freeze chain
# ============================================================

CONTROLLER_FREEZE_COMMIT = (
    "c8c5ede1601b1ada4087e141d61247903a4458a7"
)

V3_RESULT_ARCHIVE_COMMIT = (
    "9bc192e7cc70cf754b0446672bb64b016490a746"
)

TALLYQA_MANIFEST_FREEZE_COMMIT = (
    "03525c8b0ef33b8120cea6f32494a53cf476814a"
)

TALLYQA_FETCHER_FREEZE_COMMIT = (
    "2617765cc1487cc62421547af395508b1b3e944e"
)

TALLYQA_IMAGE_FREEZE_COMMIT = (
    "17fe2aaabc4e0ae32acef4eef64c6e6ead57833f"
)


# ============================================================
# Frozen evaluation settings
# ============================================================

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260912

DUMMY_GT = 1

EXPECTED_ACTIONS = [
    0.0,
    1.0,
    1.5,
    2.0,
    4.0,
]

EXPECTED_NONNOOP = [
    0.0,
    1.5,
    2.0,
    4.0,
]

EXPECTED_THRESHOLD = 0.1
EXPECTED_RIDGE_ALPHA = 0.01
EXPECTED_FEATURE_COUNT = 39


# ============================================================
# Utilities
# ============================================================

def sha256_file(
    path,
):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def load_json(
    path,
):

    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(
    path,
):

    return [
        json.loads(line)
        for line in Path(path)
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]


def git_head():

    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def git_is_ancestor(
    ancestor,
    descendant="HEAD",
):

    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return (
        result.returncode
        ==
        0
    )


def exact_mcnemar_p(
    repairs,
    breaks,
):

    n = int(
        repairs
        +
        breaks
    )

    if n == 0:
        return 1.0

    k = min(
        int(repairs),
        int(breaks),
    )

    tail = sum(
        math.comb(
            n,
            i,
        )
        for i in range(
            k + 1
        )
    ) / (
        2 ** n
    )

    return min(
        1.0,
        2.0 * tail,
    )


def paired_bootstrap_ci(
    baseline_correct,
    post_correct,
    reps=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):

    baseline = np.asarray(
        baseline_correct,
        dtype=np.float64,
    )

    post = np.asarray(
        post_correct,
        dtype=np.float64,
    )

    if baseline.shape != post.shape:

        raise RuntimeError(
            "Bootstrap arrays have "
            "different shapes."
        )

    n = len(
        baseline
    )

    if n <= 0:

        raise RuntimeError(
            "Cannot bootstrap empty set."
        )

    delta = (
        post
        -
        baseline
    )

    rng = np.random.default_rng(
        seed
    )

    boot = np.empty(
        reps,
        dtype=np.float64,
    )

    for i in range(
        reps
    ):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = float(
            delta[
                idx
            ].mean()
        )

    lower = float(
        np.quantile(
            boot,
            0.025,
        )
    )

    upper = float(
        np.quantile(
            boot,
            0.975,
        )
    )

    return (
        lower,
        upper,
    )


# ============================================================
# Dynamic natural-language input construction.
#
# IMPORTANT:
# The original TallyQA question is preserved verbatim.
# Only the frozen suffix is appended.
# ============================================================

def prepare_tallyqa_inputs(
    processor,
    image,
    question,
):

    question = str(
        question
    ).strip()

    if not question:

        raise RuntimeError(
            "Empty TallyQA question."
        )

    prompt = (
        question
        +
        "\n\n"
        +
        "Return the number only."
    )

    messages = [
        {
            "role":
                "user",

            "content": [
                {
                    "type":
                        "image",
                },
                {
                    "type":
                        "text",

                    "text":
                        prompt,
                },
            ],
        }
    ]

    formatted = (
        processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    )

    return processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )



# ============================================================
# Natural Confirmation v2 frozen overrides
# ============================================================

MANIFEST_PATH = Path(
    "outputs/tallyqa_natural_confirmation_v2/manifest.jsonl"
)

IMAGE_ROOT = Path(
    "data/tallyqa_natural_confirmation_v2/images"
)

IMAGE_INVENTORY_PATH = Path(
    "outputs/tallyqa_natural_confirmation_v2/"
    "image_inventory/image_inventory.csv"
)

OOD_PROTOCOL_PATH = Path(
    "configs/tallyqa_natural_confirmation_v2.json"
)

IMAGE_FREEZE_PATH = Path(
    "configs/tallyqa_natural_confirmation_v2_image_freeze.json"
)

RUNNER_PROTOCOL_PATH = Path(
    "configs/tallyqa_natural_confirmation_v2_final_runner.json"
)

CONTROLLER_MANIFEST_PATH = Path(
    "configs/aroma_natural_controller_k500_frozen.json"
)

CONTROLLER_BUNDLE_PATH = Path(
    "outputs/phase2_natural_controller_k500/"
    "final_frozen_controller/"
    "aroma_natural_utility_controller_k500.joblib"
)

NATURAL_CONTROLLER_METADATA_PATH = Path(
    "outputs/phase2_natural_controller_k500/"
    "final_frozen_controller/"
    "frozen_controller_metadata.json"
)

CALIBRATION_MANIFEST_PATH = Path(
    "outputs/phase2_natural_controller_k500/"
    "final_frozen_controller/"
    "calibration_manifest.csv"
)

OUT_DIR = Path(
    "outputs/tallyqa_natural_confirmation_v2/"
    "final_frozen_controller"
)

RESULT_PATH = (
    OUT_DIR
    / "tallyqa_final_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "tallyqa_final_summary.csv"
)

SUBSET_PATH = (
    OUT_DIR
    / "tallyqa_final_by_subset.csv"
)

COUNT_PATH = (
    OUT_DIR
    / "tallyqa_final_by_count.csv"
)

ACTION_PATH = (
    OUT_DIR
    / "tallyqa_final_action_distribution.csv"
)

RUN_META_PATH = (
    OUT_DIR
    / "tallyqa_final_run_metadata.json"
)

RUN_STARTED_PATH = (
    OUT_DIR
    / "tallyqa_final_run_started.json"
)

EXPECTED_MANIFEST_SHA256 = (
    "cf42a6c4b07f21d72c88d302815fd1419dd912431b203ff9f8cc53f3fc05ae0d"
)

EXPECTED_INVENTORY_SHA256 = (
    "475e28e0ebb7934d49d2062734b164ff04f4f04461d61de9ee4315f96182f0b2"
)

EXPECTED_CONTROLLER_SHA256 = (
    "d9d8ba3a24814bf71b3f4039d1effa864b57754a00539f1260a1d077c03449f5"
)

EXPECTED_CALIBRATION_MANIFEST_SHA256 = (
    "67dfdafa36c66a1299be7adea0d1232269e1b557607314972ed2b298312dc49a"
)

NATURAL_CONTROLLER_FREEZE_COMMIT = (
    "9bbd337a979029da1bc46396c8e266da5c9e6193"
)

TALLYQA_V2_MANIFEST_FREEZE_COMMIT = (
    "9fc7822beb9e224291591a89b46a182c7c1ffe01"
)

TALLYQA_V2_IMAGE_FREEZE_COMMIT = (
    "9dfd1e8ce62b745227aade37edb980e34016318d"
)


# ============================================================
# Protocol / freeze audit
# ============================================================

def protocol_audit():

    print("=" * 118)
    print(
        "AROMA TALLYQA NATURAL CONFIRMATION V2 "
        "FROZEN K=500 EVALUATION"
    )
    print("=" * 118)

    required = [
        MANIFEST_PATH,
        IMAGE_INVENTORY_PATH,
        OOD_PROTOCOL_PATH,
        IMAGE_FREEZE_PATH,
        RUNNER_PROTOCOL_PATH,
        CONTROLLER_MANIFEST_PATH,
        CONTROLLER_BUNDLE_PATH,
        NATURAL_CONTROLLER_METADATA_PATH,
        CALIBRATION_MANIFEST_PATH,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required frozen artifact(s):\n"
            +
            "\n".join(missing)
        )

    # --------------------------------------------------------
    # Git freeze ancestry.
    # --------------------------------------------------------

    required_commits = {
        "natural_controller":
            NATURAL_CONTROLLER_FREEZE_COMMIT,

        "confirmation_manifest":
            TALLYQA_V2_MANIFEST_FREEZE_COMMIT,

        "confirmation_images":
            TALLYQA_V2_IMAGE_FREEZE_COMMIT,
    }

    for name, commit in required_commits.items():

        if not git_is_ancestor(commit):

            raise RuntimeError(
                "Required freeze commit is "
                f"not an ancestor of HEAD: "
                f"{name}={commit}"
            )

    # --------------------------------------------------------
    # Manifest.
    # --------------------------------------------------------

    manifest_hash = sha256_file(
        MANIFEST_PATH
    )

    if manifest_hash != EXPECTED_MANIFEST_SHA256:

        raise RuntimeError(
            "Confirmation-v2 manifest "
            "hash mismatch."
        )

    records = load_jsonl(
        MANIFEST_PATH
    )

    if len(records) != 4000:
        raise RuntimeError(
            f"Expected 4000 manifest rows, "
            f"found {len(records)}."
        )

    required_fields = {
        "manifest_index",
        "question_id",
        "image",
        "image_id",
        "subset",
        "issimple",
        "data_source",
        "question",
        "answer",
    }

    for r in records:

        missing_fields = (
            required_fields
            -
            set(r)
        )

        if missing_fields:
            raise RuntimeError(
                "Manifest record missing "
                f"fields: {missing_fields}"
            )

    qids = [
        int(r["question_id"])
        for r in records
    ]

    images = [
        str(r["image"])
        for r in records
    ]

    image_ids = [
        int(r["image_id"])
        for r in records
    ]

    if len(set(qids)) != 4000:
        raise RuntimeError(
            "Question IDs are not unique."
        )

    if len(set(images)) != 4000:
        raise RuntimeError(
            "Image paths are not unique."
        )

    if len(set(image_ids)) != 4000:
        raise RuntimeError(
            "Image IDs are not unique."
        )

    subset_counts = (
        pd.Series(
            [
                str(r["subset"])
                for r in records
            ]
        )
        .value_counts()
        .to_dict()
    )

    if subset_counts != {
        "simple": 2000,
        "complex": 2000,
    }:
        raise RuntimeError(
            "Unexpected subset counts: "
            f"{subset_counts}"
        )

    for r in records:

        if (
            (
                str(r["subset"])
                ==
                "simple"
            )
            !=
            bool(r["issimple"])
        ):
            raise RuntimeError(
                "Subset / issimple mismatch."
            )

        answer = int(
            r["answer"]
        )

        if not (
            0 <= answer <= 15
        ):
            raise RuntimeError(
                "Answer outside frozen "
                "0..15 numeral support."
            )

        image_path = (
            IMAGE_ROOT
            /
            str(r["image"])
        )

        if not image_path.exists():
            raise RuntimeError(
                "Missing confirmation image: "
                f"{image_path}"
            )

    # --------------------------------------------------------
    # Image inventory.
    # --------------------------------------------------------

    inventory_hash = sha256_file(
        IMAGE_INVENTORY_PATH
    )

    if (
        inventory_hash
        !=
        EXPECTED_INVENTORY_SHA256
    ):
        raise RuntimeError(
            "Frozen image inventory "
            "hash mismatch."
        )

    inventory = pd.read_csv(
        IMAGE_INVENTORY_PATH
    )

    if len(inventory) != 4000:
        raise RuntimeError(
            "Image inventory does not "
            "contain 4000 rows."
        )

    if (
        inventory["image"].nunique()
        !=
        4000
    ):
        raise RuntimeError(
            "Inventory paths are not unique."
        )

    if (
        set(
            inventory["image"]
            .astype(str)
        )
        !=
        set(images)
    ):
        raise RuntimeError(
            "Manifest / inventory image "
            "sets differ."
        )

    # --------------------------------------------------------
    # Confirmation protocol.
    # --------------------------------------------------------

    protocol = load_json(
        OOD_PROTOCOL_PATH
    )

    if (
        protocol["source_dataset"]
        !=
        "TallyQA"
    ):
        raise RuntimeError(
            "Dataset mismatch."
        )

    if (
        protocol["evaluation_split"]
        !=
        "official_test"
    ):
        raise RuntimeError(
            "Evaluation split mismatch."
        )

    if (
        protocol[
            "natural_ood_zero_shot"
        ]
        is not False
    ):
        raise RuntimeError(
            "This confirmation must NOT "
            "be labeled zero-shot."
        )

    if (
        protocol[
            "natural_adapted_k500"
        ]
        is not True
    ):
        raise RuntimeError(
            "Natural-adapted protocol flag "
            "mismatch."
        )

    sampling = protocol[
        "sampling"
    ]

    if int(
        sampling["total_samples"]
    ) != 4000:
        raise RuntimeError(
            "Protocol sample-count mismatch."
        )

    if int(
        sampling["simple_samples"]
    ) != 2000:
        raise RuntimeError(
            "Protocol Simple count mismatch."
        )

    if int(
        sampling["complex_samples"]
    ) != 2000:
        raise RuntimeError(
            "Protocol Complex count mismatch."
        )

    if (
        sampling[
            "one_question_per_image"
        ]
        is not True
    ):
        raise RuntimeError(
            "Image-uniqueness policy mismatch."
        )

    if (
        sampling[
            "selection_uses_answer"
        ]
        is not False
    ):
        raise RuntimeError(
            "Selection leakage policy mismatch."
        )

    if (
        sampling[
            "selection_uses_model_outcomes"
        ]
        is not False
    ):
        raise RuntimeError(
            "Outcome-filtering policy mismatch."
        )

    prompt_policy = protocol[
        "prompt_policy"
    ]

    if (
        prompt_policy[
            "use_original_tallyqa_question"
        ]
        is not True
    ):
        raise RuntimeError(
            "Prompt policy mismatch."
        )

    if (
        prompt_policy[
            "append_instruction"
        ]
        !=
        "Return the number only."
    ):
        raise RuntimeError(
            "Prompt suffix mismatch."
        )

    if (
        prompt_policy[
            "question_rewriting"
        ]
        is not False
    ):
        raise RuntimeError(
            "Question-rewriting policy "
            "mismatch."
        )

    # --------------------------------------------------------
    # Natural-adapted controller.
    # --------------------------------------------------------

    controller_manifest = load_json(
        CONTROLLER_MANIFEST_PATH
    )

    controller_meta = load_json(
        NATURAL_CONTROLLER_METADATA_PATH
    )

    bundle = joblib.load(
        CONTROLLER_BUNDLE_PATH
    )

    current_bundle_hash = sha256_file(
        CONTROLLER_BUNDLE_PATH
    )

    if (
        current_bundle_hash
        !=
        EXPECTED_CONTROLLER_SHA256
    ):
        raise RuntimeError(
            "Frozen K500 controller binary "
            "hash mismatch."
        )

    if (
        controller_meta[
            "bundle_sha256"
        ]
        !=
        current_bundle_hash
    ):
        raise RuntimeError(
            "Controller metadata / bundle "
            "hash mismatch."
        )

    if (
        controller_meta["status"]
        !=
        "FROZEN_BEFORE_NEW_NATURAL_CONFIRMATION"
    ):
        raise RuntimeError(
            "Controller freeze status mismatch."
        )

    if (
        controller_meta[
            "independent_confirmation_results_seen"
        ]
        is not False
    ):
        raise RuntimeError(
            "Controller metadata indicates "
            "confirmation outcomes were seen."
        )

    if (
        bundle["model_id"]
        !=
        MODEL_ID
    ):
        raise RuntimeError(
            "Controller model mismatch."
        )

    if (
        bundle["attention_implementation"]
        !=
        "eager"
    ):
        raise RuntimeError(
            "Attention mode mismatch."
        )

    if int(
        bundle["head_layer"]
    ) != LAYER:
        raise RuntimeError(
            "Layer mismatch."
        )

    if int(
        bundle["head_index"]
    ) != HEAD:
        raise RuntimeError(
            "Head index mismatch."
        )

    if (
        bundle["head_name"]
        !=
        "L18H13"
    ):
        raise RuntimeError(
            "Head name mismatch."
        )

    if (
        bundle.get("bundle_version")
        !=
        "AROMA-natural-k500-v1"
    ):
        raise RuntimeError(
            "Natural controller version "
            "mismatch."
        )

    actions = [
        float(a)
        for a in bundle["actions"]
    ]

    if actions != EXPECTED_ACTIONS:
        raise RuntimeError(
            "Action-space mismatch."
        )

    nonnoop = [
        float(a)
        for a in
        bundle["nonnoop_actions"]
    ]

    if nonnoop != EXPECTED_NONNOOP:
        raise RuntimeError(
            "Non-NOOP action-space mismatch."
        )

    feature_names = list(
        bundle["feature_names"]
    )

    if (
        len(feature_names)
        !=
        EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Feature-count mismatch."
        )

    if (
        bundle["feature_family"]
        !=
        "full_geometry"
    ):
        raise RuntimeError(
            "Feature-family mismatch."
        )

    if not math.isclose(
        float(bundle["ridge_alpha"]),
        EXPECTED_RIDGE_ALPHA,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Ridge alpha mismatch."
        )

    if not math.isclose(
        float(bundle["threshold"]),
        EXPECTED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Utility threshold mismatch."
        )

    if int(
        bundle["natural_calibration_n"]
    ) != 500:
        raise RuntimeError(
            "Natural calibration n mismatch."
        )

    if int(
        bundle["natural_calibration_simple"]
    ) != 250:
        raise RuntimeError(
            "Natural Simple calibration "
            "count mismatch."
        )

    if int(
        bundle["natural_calibration_complex"]
    ) != 250:
        raise RuntimeError(
            "Natural Complex calibration "
            "count mismatch."
        )

    if (
        list(
            controller_manifest[
                "controller"
            ][
                "feature_names"
            ]
        )
        !=
        feature_names
    ):
        raise RuntimeError(
            "Controller manifest / bundle "
            "feature ordering mismatch."
        )

    # --------------------------------------------------------
    # Calibration-set disjointness.
    # --------------------------------------------------------

    calibration_hash = sha256_file(
        CALIBRATION_MANIFEST_PATH
    )

    if (
        calibration_hash
        !=
        EXPECTED_CALIBRATION_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Calibration manifest "
            "hash mismatch."
        )

    calibration = pd.read_csv(
        CALIBRATION_MANIFEST_PATH
    )

    if len(calibration) != 500:
        raise RuntimeError(
            "Expected 500 calibration rows."
        )

    calibration_qids = set(
        calibration["question_id"]
        .astype(int)
    )

    calibration_images = set(
        calibration["image"]
        .astype(str)
    )

    if (
        calibration_qids
        &
        set(qids)
    ):
        raise RuntimeError(
            "Calibration / confirmation "
            "question overlap detected."
        )

    if (
        calibration_images
        &
        set(images)
    ):
        raise RuntimeError(
            "Calibration / confirmation "
            "image overlap detected."
        )

    # --------------------------------------------------------
    # Image freeze config.
    # --------------------------------------------------------

    image_freeze = load_json(
        IMAGE_FREEZE_PATH
    )

    if (
        image_freeze[
            "manifest"
        ][
            "sha256"
        ]
        !=
        EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Image-freeze manifest hash "
            "mismatch."
        )

    if (
        image_freeze[
            "image_inventory"
        ][
            "sha256"
        ]
        !=
        EXPECTED_INVENTORY_SHA256
    ):
        raise RuntimeError(
            "Image-freeze inventory hash "
            "mismatch."
        )

    if (
        image_freeze[
            "natural_confirmation_v2_model_inference_started"
        ]
        is not False
    ):
        raise RuntimeError(
            "Image freeze says inference "
            "already started."
        )

    # --------------------------------------------------------
    # Runner self-hash / frozen protocol.
    # --------------------------------------------------------

    runner_protocol = load_json(
        RUNNER_PROTOCOL_PATH
    )

    source_hash = sha256_file(
        Path(__file__)
    )

    if (
        source_hash
        !=
        runner_protocol[
            "runner_source_sha256"
        ]
    ):
        raise RuntimeError(
            "Final confirmation runner "
            "source hash mismatch."
        )

    if (
        runner_protocol[
            "manifest_sha256"
        ]
        !=
        EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Runner / manifest hash mismatch."
        )

    if (
        runner_protocol[
            "image_inventory_sha256"
        ]
        !=
        EXPECTED_INVENTORY_SHA256
    ):
        raise RuntimeError(
            "Runner / inventory hash mismatch."
        )

    if (
        runner_protocol[
            "controller_bundle_sha256"
        ]
        !=
        current_bundle_hash
    ):
        raise RuntimeError(
            "Runner / controller hash mismatch."
        )

    if (
        runner_protocol[
            "calibration_manifest_sha256"
        ]
        !=
        calibration_hash
    ):
        raise RuntimeError(
            "Runner / calibration hash mismatch."
        )

    if (
        runner_protocol[
            "natural_ood_zero_shot"
        ]
        is not False
    ):
        raise RuntimeError(
            "Runner must not claim zero-shot."
        )

    if (
        runner_protocol[
            "natural_adapted_k500"
        ]
        is not True
    ):
        raise RuntimeError(
            "Runner adaptation flag mismatch."
        )

    if (
        runner_protocol[
            "gt_read_after_action_execution"
        ]
        is not True
    ):
        raise RuntimeError(
            "GT-isolation flag mismatch."
        )

    # --------------------------------------------------------
    # Feature leakage.
    # --------------------------------------------------------

    forbidden = {
        "ground_truth",
        "answer",
        "baseline_correct",
        "modulated_correct",
        "baseline_gt_logp",
        "modulated_gt_logp",
        "delta_gt_logp",
        "baseline_margin",
        "modulated_margin",
        "delta_margin",
    }

    leaked = (
        forbidden
        &
        set(feature_names)
    )

    if leaked:
        raise RuntimeError(
            "Leakage-sensitive features: "
            f"{leaked}"
        )

    print("Samples                    : 4000")
    print("Simple / Complex           : 2000 / 2000")
    print("Unique image paths         : 4000")
    print(
        "Manifest SHA256            :",
        manifest_hash,
    )
    print(
        "Image inventory SHA256     :",
        inventory_hash,
    )
    print(
        "Controller bundle SHA256   :",
        current_bundle_hash,
    )
    print(
        "Calibration manifest SHA256:",
        calibration_hash,
    )
    print("Natural calibration        : 500 = 250 Simple + 250 Complex")
    print("Frozen head                : L18H13")
    print("Frozen actions             :", actions)
    print("Frozen feature count       :", len(feature_names))
    print("Frozen Ridge alpha         :", bundle["ridge_alpha"])
    print("Frozen utility threshold   :", bundle["threshold"])
    print("Prompt suffix              : 'Return the number only.'")
    print("Zero-shot claim            : FALSE")
    print("Natural-adapted K500       : TRUE")
    print("Calibration disjointness   : PASS")
    print("Feature leakage audit      : PASS")
    print("GT-isolation policy        : PASS")
    print("Git freeze ancestry audit  : PASS")
    print("Controller identity audit  : PASS")
    print("Natural image audit        : PASS")
    print("\nPROTOCOL AUDIT: PASS")

    return (
        records,
        bundle,
        current_bundle_hash,
        source_hash,
    )




# ============================================================
# Resume
# ============================================================

def prepare_resume():

    if not RESULT_PATH.exists():
        return set()

    df = pd.read_csv(
        RESULT_PATH
    )

    if len(df) == 0:
        return set()

    required = {
        "question_id",
        "image",
        "subset",
        "baseline_prediction",
        "selected_alpha",
        "post_prediction",
        "ground_truth",
    }

    missing = (
        required
        -
        set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Existing result file is "
            "not resumable; missing "
            f"columns: {missing}"
        )

    if (
        df[
            "question_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate question IDs in "
            "existing result file."
        )

    return {
        int(x)
        for x in df[
            "question_id"
        ].tolist()
    }


# ============================================================
# Summaries
# ============================================================

def summarize_group(
    df,
    scope_name,
):

    baseline_correct = (
        df[
            "baseline_correct"
        ]
        .astype(bool)
        .to_numpy()
    )

    post_correct = (
        df[
            "post_correct"
        ]
        .astype(bool)
        .to_numpy()
    )

    repairs = int(
        (
            (~baseline_correct)
            &
            post_correct
        ).sum()
    )

    breaks = int(
        (
            baseline_correct
            &
            (~post_correct)
        ).sum()
    )

    baseline_n = int(
        baseline_correct.sum()
    )

    baseline_wrong = int(
        (
            ~baseline_correct
        ).sum()
    )

    post_n = int(
        post_correct.sum()
    )

    baseline_acc = float(
        baseline_correct.mean()
    )

    post_acc = float(
        post_correct.mean()
    )

    delta = (
        post_acc
        -
        baseline_acc
    )

    (
        ci_low,
        ci_high,
    ) = paired_bootstrap_ci(
        baseline_correct,
        post_correct,
    )

    p = exact_mcnemar_p(
        repairs,
        breaks,
    )

    intervention_rate = float(
        (
            ~np.isclose(
                df[
                    "selected_alpha"
                ]
                .to_numpy(
                    dtype=float
                ),
                1.0,
                atol=1e-12,
                rtol=0.0,
            )
        ).mean()
    )

    return {
        "scope":
            scope_name,

        "n":
            len(df),

        "baseline_correct":
            baseline_n,

        "baseline_wrong":
            baseline_wrong,

        "baseline_accuracy":
            baseline_acc,

        "post_correct":
            post_n,

        "post_accuracy":
            post_acc,

        "accuracy_change":
            delta,

        "accuracy_change_ci95_low":
            ci_low,

        "accuracy_change_ci95_high":
            ci_high,

        "repairs":
            repairs,

        "breaks":
            breaks,

        "net_repairs":
            repairs
            -
            breaks,

        "repair_rate_among_wrong":
            (
                repairs
                /
                baseline_wrong
                if baseline_wrong
                else 0.0
            ),

        "break_rate_among_correct":
            (
                breaks
                /
                baseline_n
                if baseline_n
                else 0.0
            ),

        "intervention_rate":
            intervention_rate,

        "mcnemar_exact_p":
            p,
    }


def summarize(
    df,
):

    if len(df) != 4000:

        raise RuntimeError(
            f"Expected 4000 final rows, "
            f"found {len(df)}."
        )

    if (
        df[
            "question_id"
        ]
        .nunique()
        != 4000
    ):

        raise RuntimeError(
            "Final result question IDs "
            "are not unique."
        )

    if (
        df.isna()
        .sum()
        .sum()
        != 0
    ):

        raise RuntimeError(
            "NaN detected in final "
            "result table."
        )

    overall = summarize_group(
        df,
        "ALL4000",
    )

    overall[
        "primary_success"
    ] = bool(
        (
            overall[
                "accuracy_change"
            ]
            >
            0.0
        )
        and
        (
            overall[
                "mcnemar_exact_p"
            ]
            <
            0.05
        )
    )

    summary = pd.DataFrame([
        overall
    ])

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    subset_rows = []

    for subset in [
        "simple",
        "complex",
    ]:

        g = df[
            df[
                "subset"
            ]
            ==
            subset
        ].copy()

        if len(g) != 2000:

            raise RuntimeError(
                f"Expected 2000 rows for "
                f"{subset}, found {len(g)}."
            )

        row = summarize_group(
            g,
            subset,
        )

        subset_rows.append(
            row
        )

    subset_df = pd.DataFrame(
        subset_rows
    )

    subset_df.to_csv(
        SUBSET_PATH,
        index=False,
    )

    count_rows = []

    for gt, g in (
        df.groupby(
            "ground_truth"
        )
    ):

        row = summarize_group(
            g,
            f"GT_{int(gt)}",
        )

        row[
            "ground_truth"
        ] = int(
            gt
        )

        count_rows.append(
            row
        )

    count_df = (
        pd.DataFrame(
            count_rows
        )
        .sort_values(
            "ground_truth"
        )
    )

    count_df.to_csv(
        COUNT_PATH,
        index=False,
    )

    action_distribution = (
        df[
            "selected_alpha"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "selected_alpha"
        )
        .reset_index(
            name="count"
        )
    )

    action_distribution[
        "fraction"
    ] = (
        action_distribution[
            "count"
        ]
        /
        len(df)
    )

    action_distribution.to_csv(
        ACTION_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 118
    )

    print(
        "TALLYQA NATURAL-OOD "
        "FINAL PRIMARY RESULT"
    )

    print(
        "=" * 118
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 118
    )

    print(
        "TALLYQA PRESPECIFIED "
        "SIMPLE / COMPLEX RESULTS"
    )

    print(
        "=" * 118
    )

    print(
        subset_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 118
    )

    print(
        "FROZEN CONTROLLER "
        "ACTION DISTRIBUTION"
    )

    print(
        "=" * 118
    )

    print(
        action_distribution.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    for path in [
        RESULT_PATH,
        SUMMARY_PATH,
        SUBSET_PATH,
        COUNT_PATH,
        ACTION_PATH,
    ]:

        print(
            path
        )

    return (
        summary,
        subset_df,
        count_df,
        action_distribution,
    )


# ============================================================
# Main
# ============================================================

def main(
    resume=False,
    protocol_only=False,
):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        records,
        bundle,
        bundle_hash,
        runner_hash,
    ) = protocol_audit()

    if protocol_only:

        print(
            "\nNo model was loaded."
        )

        print(
            "No TallyQA prediction "
            "was produced."
        )

        return

    # --------------------------------------------------------
    # Freeze run-start marker BEFORE model inference.
    # --------------------------------------------------------

    if not resume:

        if RESULT_PATH.exists():

            raise RuntimeError(
                "Final result file already "
                "exists. Refusing to start "
                "a new final evaluation. "
                "Use --resume only for a "
                "technical interruption."
            )

        started = {
            "started_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "git_head":
                git_head(),

            "manifest_sha256":
                EXPECTED_MANIFEST_SHA256,

            "image_inventory_sha256":
                EXPECTED_INVENTORY_SHA256,

            "controller_bundle_sha256":
                bundle_hash,

            "runner_source_sha256":
                runner_hash,

            "samples":
                4000,

            "natural_ood_zero_shot":
                False,

            "retuning_after_start":
                False,
        }

        RUN_STARTED_PATH.write_text(
            json.dumps(
                started,
                indent=2,
            )
            +
            "\n",
            encoding="utf-8",
        )

        completed = set()

    else:

        if not RUN_STARTED_PATH.exists():

            raise RuntimeError(
                "--resume requested but "
                "run-start marker does "
                "not exist."
            )

        completed = (
            prepare_resume()
        )

    print(
        "\nAlready completed:",
        len(
            completed
        ),
    )

    print(
        "Remaining:",
        4000
        -
        len(
            completed
        ),
    )

    # --------------------------------------------------------
    # Sort independently of answer / GT.
    # --------------------------------------------------------

    ordered = sorted(
        records,
        key=lambda r:
            int(
                r[
                    "manifest_index"
                ]
            ),
    )

    remaining = [
        r
        for r in ordered
        if int(
            r[
                "question_id"
            ]
        )
        not in completed
    ]

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    numeral_ids = (
        numeral_token_ids(
            processor
        )
    )

    if sorted(
        numeral_ids.keys()
    ) != list(
        range(
            16
        )
    ):

        raise RuntimeError(
            "Expanded numeral-token "
            "audit failed."
        )

    print(
        "Numeral-token audit: PASS"
    )

    print(
        "Loading model..."
    )

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    )

    model.eval()

    feature_names = list(
        bundle[
            "feature_names"
        ]
    )

    # ========================================================
    # Final natural-OOD evaluation loop.
    # ========================================================

    for sample in tqdm(
        remaining,
        desc=(
            "TallyQA natural-adapted "
            "confirmation v2"
        ),
    ):

        qid = int(
            sample[
                "question_id"
            ]
        )

        relative_image = str(
            sample[
                "image"
            ]
        )

        image_path = (
            IMAGE_ROOT
            /
            relative_image
        )

        image = (
            Image.open(
                image_path
            )
            .convert(
                "RGB"
            )
        )

        question = str(
            sample[
                "question"
            ]
        )

        inputs = (
            prepare_tallyqa_inputs(
                processor,
                image,
                question,
            )
        )

        inputs = (
            move_inputs(
                inputs,
                model,
            )
        )

        # ====================================================
        # IMPORTANT:
        #
        # sample["answer"] is NOT accessed anywhere above
        # or during controller action selection.
        #
        # score_state receives fixed DUMMY_GT=1.
        # state_to_features explicitly discards gt_logp/margin.
        # ====================================================

        baseline_state = (
            score_state(
                model,
                inputs,
                numeral_ids,
                DUMMY_GT,
            )
        )

        (
            X,
            feature_values,
        ) = (
            state_to_features(
                baseline_state,
                feature_names,
            )
        )

        (
            selected_alpha,
            selected_score,
            score_map,
        ) = (
            controller_decision(
                bundle,
                X,
            )
        )

        baseline_prediction = int(
            baseline_state[
                "best_numeral"
            ]
        )

        # ----------------------------------------------------
        # Exact NOOP identity.
        # ----------------------------------------------------

        if math.isclose(
            selected_alpha,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):

            post_state = (
                baseline_state
            )

            hook_calls = 0

        else:

            modifier = (
                HeadGainModifier(
                    model=model,
                    layer_idx=LAYER,
                    head_idx=HEAD,
                    alpha=
                        selected_alpha,
                )
            )

            modifier.register()

            try:

                post_state = (
                    score_state(
                        model,
                        inputs,
                        numeral_ids,
                        DUMMY_GT,
                    )
                )

            finally:

                modifier.remove()

            hook_calls = int(
                modifier.calls
            )

            if hook_calls <= 0:

                raise RuntimeError(
                    "Gain hook not called "
                    f"for question_id={qid}"
                )

        post_prediction = int(
            post_state[
                "best_numeral"
            ]
        )

        # ====================================================
        # ONLY NOW is the actual TallyQA answer read.
        # ====================================================

        gt = int(
            sample[
                "answer"
            ]
        )

        baseline_correct = bool(
            baseline_prediction
            ==
            gt
        )

        post_correct = bool(
            post_prediction
            ==
            gt
        )

        repair = bool(
            (
                not baseline_correct
            )
            and
            post_correct
        )

        break_case = bool(
            baseline_correct
            and
            (
                not post_correct
            )
        )

        row = {
            "manifest_index":
                int(
                    sample[
                        "manifest_index"
                    ]
                ),

            "question_id":
                qid,

            "image":
                relative_image,

            "image_id":
                int(
                    sample[
                        "image_id"
                    ]
                ),

            "subset":
                str(
                    sample[
                        "subset"
                    ]
                ),

            "issimple":
                bool(
                    sample[
                        "issimple"
                    ]
                ),

            "data_source":
                str(
                    sample[
                        "data_source"
                    ]
                ),

            "question":
                question,

            "ground_truth":
                gt,

            "baseline_prediction":
                baseline_prediction,

            "selected_alpha":
                float(
                    selected_alpha
                ),

            "selected_score":
                float(
                    selected_score
                ),

            "post_prediction":
                post_prediction,

            "baseline_correct":
                baseline_correct,

            "post_correct":
                post_correct,

            "repair":
                repair,

            "break_case":
                break_case,

            "hook_calls":
                int(
                    hook_calls
                ),

            "score_alpha_0":
                float(
                    score_map[
                        0.0
                    ]
                ),

            "score_alpha_1p5":
                float(
                    score_map[
                        1.5
                    ]
                ),

            "score_alpha_2":
                float(
                    score_map[
                        2.0
                    ]
                ),

            "score_alpha_4":
                float(
                    score_map[
                        4.0
                    ]
                ),

            "baseline_expected_numeral":
                float(
                    baseline_state[
                        "expected_numeral"
                    ]
                ),

            "post_expected_numeral":
                float(
                    post_state[
                        "expected_numeral"
                    ]
                ),

            "baseline_entropy":
                float(
                    baseline_state[
                        "entropy"
                    ]
                ),

            "post_entropy":
                float(
                    post_state[
                        "entropy"
                    ]
                ),

            "baseline_top1_prob":
                float(
                    baseline_state[
                        "top1_prob"
                    ]
                ),

            "post_top1_prob":
                float(
                    post_state[
                        "top1_prob"
                    ]
                ),

            "baseline_conditional_margin":
                float(
                    baseline_state[
                        "conditional_margin"
                    ]
                ),

            "post_conditional_margin":
                float(
                    post_state[
                        "conditional_margin"
                    ]
                ),
        }

        # Keep exact frozen controller features for audit.
        for name in feature_names:

            row[
                f"feature__{name}"
            ] = float(
                feature_values[
                    name
                ]
            )

        local_df = pd.DataFrame([
            row
        ])

        local_df.to_csv(
            RESULT_PATH,
            mode="a",
            header=(
                not RESULT_PATH.exists()
            ),
            index=False,
        )

    # --------------------------------------------------------
    # Final integrity.
    # --------------------------------------------------------

    final_df = pd.read_csv(
        RESULT_PATH
    )

    (
        summary,
        subset_df,
        count_df,
        action_df,
    ) = summarize(
        final_df
    )

    run_meta = {
        "protocol_name":
            "AROMA TallyQA Natural-OOD "
            "Frozen Zero-Shot Final Evaluation",

        "completed_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_head":
            git_head(),

        "dataset":
            "TallyQA",

        "split":
            "official_test",

        "samples":
            4000,

        "simple_samples":
            2000,

        "complex_samples":
            2000,

        "unique_images":
            4000,

        "model":
            MODEL_ID,

        "attention_implementation":
            "eager",

        "head":
            "L18H13",

        "actions":
            EXPECTED_ACTIONS,

        "feature_count":
            EXPECTED_FEATURE_COUNT,

        "ridge_alpha":
            EXPECTED_RIDGE_ALPHA,

        "utility_threshold":
            EXPECTED_THRESHOLD,

        "prompt_policy": {
            "original_question":
                True,

            "suffix":
                "Return the number only.",

            "question_rewriting":
                False,
        },

        "manifest_sha256":
            EXPECTED_MANIFEST_SHA256,

        "image_inventory_sha256":
            EXPECTED_INVENTORY_SHA256,

        "controller_bundle_sha256":
            bundle_hash,

        "runner_source_sha256":
            runner_hash,

        "gt_read_after_action_execution":
            True,

        "retuned_on_tallyqa":
            False,

        "natural_ood_zero_shot":
            False,

        "primary_success":
            bool(
                summary.iloc[0][
                    "primary_success"
                ]
            ),
    }

    RUN_META_PATH.write_text(
        json.dumps(
            run_meta,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        "\nTALLYQA NATURAL-OOD "
        "FROZEN FINAL EVALUATION COMPLETE"
    )


if __name__ == "__main__":

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only after a purely "
            "technical interruption."
        ),
    )

    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help=(
            "Run all frozen protocol, "
            "hash, controller, leakage, "
            "and Git audits without "
            "loading the model or "
            "producing TallyQA predictions."
        ),
    )

    args = parser.parse_args()

    main(
        resume=
            args.resume,

        protocol_only=
            args.protocol_only,
    )
