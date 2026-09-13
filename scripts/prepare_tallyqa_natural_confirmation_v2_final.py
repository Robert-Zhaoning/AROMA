#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import py_compile
import re
import shutil
import subprocess

import joblib
import pandas as pd


ROOT = Path("/workspace/AromaExperiments")

SOURCE_RUNNER = (
    ROOT
    / "scripts"
    / "run_tallyqa_natural_ood_v1_final.py"
)

TARGET_RUNNER = (
    ROOT
    / "scripts"
    / "run_tallyqa_natural_confirmation_v2_final.py"
)

MANIFEST = (
    ROOT
    / "outputs"
    / "tallyqa_natural_confirmation_v2"
    / "manifest.jsonl"
)

INVENTORY = (
    ROOT
    / "outputs"
    / "tallyqa_natural_confirmation_v2"
    / "image_inventory"
    / "image_inventory.csv"
)

INVENTORY_SUMMARY = (
    ROOT
    / "outputs"
    / "tallyqa_natural_confirmation_v2"
    / "image_inventory"
    / "image_inventory_summary.json"
)

CONTROLLER = (
    ROOT
    / "outputs"
    / "phase2_natural_controller_k500"
    / "final_frozen_controller"
    / "aroma_natural_utility_controller_k500.joblib"
)

CONTROLLER_METADATA = (
    ROOT
    / "outputs"
    / "phase2_natural_controller_k500"
    / "final_frozen_controller"
    / "frozen_controller_metadata.json"
)

CALIBRATION_MANIFEST = (
    ROOT
    / "outputs"
    / "phase2_natural_controller_k500"
    / "final_frozen_controller"
    / "calibration_manifest.csv"
)

MANIFEST_SUMMARY = (
    ROOT
    / "outputs"
    / "tallyqa_natural_confirmation_v2"
    / "manifest_summary.json"
)

CONFIG_DIR = (
    ROOT
    / "configs"
)

OOD_PROTOCOL = (
    CONFIG_DIR
    / "tallyqa_natural_confirmation_v2.json"
)

IMAGE_FREEZE = (
    CONFIG_DIR
    / "tallyqa_natural_confirmation_v2_image_freeze.json"
)

CONTROLLER_MANIFEST = (
    CONFIG_DIR
    / "aroma_natural_controller_k500_frozen.json"
)

RUNNER_PROTOCOL = (
    CONFIG_DIR
    / "tallyqa_natural_confirmation_v2_final_runner.json"
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

EXPECTED_CALIBRATION_SHA256 = (
    "67dfdafa36c66a1299be7adea0d1232269e1b557607314972ed2b298312dc49a"
)

EXPECTED_SOURCE_SHA256 = (
    "cd51c8a4d5a6deb0f5423c3ae2e16e71198867d41da3b662c77631b7d26678a2"
)


def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git(*args):

    return subprocess.check_output(
        [
            "git",
            *args,
        ],
        cwd=ROOT,
        text=True,
    ).strip()


for path in [
    SOURCE_RUNNER,
    MANIFEST,
    INVENTORY,
    INVENTORY_SUMMARY,
    CONTROLLER,
    CONTROLLER_METADATA,
    CALIBRATION_MANIFEST,
    MANIFEST_SUMMARY,
]:

    if not path.exists():
        raise FileNotFoundError(path)


# ============================================================
# Frozen hash verification before constructing evaluator
# ============================================================

observed = {
    "manifest":
        sha256_file(
            MANIFEST
        ),

    "inventory":
        sha256_file(
            INVENTORY
        ),

    "controller":
        sha256_file(
            CONTROLLER
        ),

    "calibration":
        sha256_file(
            CALIBRATION_MANIFEST
        ),
}

expected = {
    "manifest":
        EXPECTED_MANIFEST_SHA256,

    "inventory":
        EXPECTED_INVENTORY_SHA256,

    "controller":
        EXPECTED_CONTROLLER_SHA256,

    "calibration":
        EXPECTED_CALIBRATION_SHA256,
}

for key in expected:

    if observed[key] != expected[key]:

        raise RuntimeError(
            f"{key} hash mismatch:\n"
            f"expected={expected[key]}\n"
            f"observed={observed[key]}"
        )


# ============================================================
# Resolve actual full Git freeze commits
# ============================================================

controller_commit = git(
    "rev-parse",
    "9bbd337",
)

manifest_commit = git(
    "rev-parse",
    "9fc7822",
)

image_commit = git(
    "rev-parse",
    "HEAD",
)

print("=" * 118)
print("FREEZE COMMITS")
print("=" * 118)

print(
    "Controller:",
    controller_commit,
)

print(
    "Manifest  :",
    manifest_commit,
)

print(
    "Image     :",
    image_commit,
)


# ============================================================
# Read frozen bundle / summaries
# ============================================================

bundle = joblib.load(
    CONTROLLER
)

controller_meta = json.loads(
    CONTROLLER_METADATA.read_text()
)

manifest_summary = json.loads(
    MANIFEST_SUMMARY.read_text()
)

inventory_summary = json.loads(
    INVENTORY_SUMMARY.read_text()
)

feature_names = list(
    bundle[
        "feature_names"
    ]
)

if len(feature_names) != 39:
    raise RuntimeError(
        "Expected 39 features."
    )

if bundle.get(
    "bundle_version"
) != "AROMA-natural-k500-v1":

    raise RuntimeError(
        "Unexpected natural-controller "
        "bundle version."
    )

if int(
    bundle.get(
        "natural_calibration_n",
        -1,
    )
) != 500:

    raise RuntimeError(
        "Unexpected calibration n."
    )


# ============================================================
# Write frozen natural-controller manifest
# ============================================================

controller_manifest_obj = {
    "protocol_name":
        "AROMA Natural-Adapted K500 "
        "Controller Freeze",

    "status":
        "FROZEN_BEFORE_NATURAL_CONFIRMATION_V2",

    "controller":
        {
            "bundle_path":
                str(
                    CONTROLLER.relative_to(
                        ROOT
                    )
                ),

            "bundle_sha256":
                EXPECTED_CONTROLLER_SHA256,

            "bundle_version":
                bundle[
                    "bundle_version"
                ],

            "model":
                bundle[
                    "model_id"
                ],

            "attention_implementation":
                bundle[
                    "attention_implementation"
                ],

            "head":
                bundle[
                    "head_name"
                ],

            "layer":
                int(
                    bundle[
                        "head_layer"
                    ]
                ),

            "head_index":
                int(
                    bundle[
                        "head_index"
                    ]
                ),

            "actions":
                [
                    float(x)
                    for x in
                    bundle[
                        "actions"
                    ]
                ],

            "nonnoop_actions":
                [
                    float(x)
                    for x in
                    bundle[
                        "nonnoop_actions"
                    ]
                ],

            "feature_family":
                bundle[
                    "feature_family"
                ],

            "feature_names":
                feature_names,

            "feature_count":
                len(
                    feature_names
                ),

            "ridge_alpha":
                float(
                    bundle[
                        "ridge_alpha"
                    ]
                ),

            "utility_threshold":
                float(
                    bundle[
                        "threshold"
                    ]
                ),
        },

    "natural_calibration":
        {
            "n":
                500,

            "simple":
                250,

            "complex":
                250,

            "manifest_path":
                str(
                    CALIBRATION_MANIFEST.relative_to(
                        ROOT
                    )
                ),

            "manifest_sha256":
                EXPECTED_CALIBRATION_SHA256,

            "selection_uses_outcomes":
                False,
        },

    "freeze_commit":
        controller_commit,
}


CONTROLLER_MANIFEST.write_text(
    json.dumps(
        controller_manifest_obj,
        indent=2,
        sort_keys=True,
    )
    +
    "\n"
)


# ============================================================
# Write Natural Confirmation-v2 protocol
# ============================================================

ood_protocol_obj = {
    "protocol_name":
        "AROMA TallyQA Natural "
        "Confirmation v2",

    "source_dataset":
        "TallyQA",

    "evaluation_split":
        "official_test",

    "source_metadata":
        {
            "path":
                "external/tallyqa/qa/test.json",

            "sha256":
                EXPECTED_SOURCE_SHA256,

            "rows":
                38589,
        },

    "sampling":
        {
            "total_samples":
                4000,

            "simple_samples":
                2000,

            "complex_samples":
                2000,

            "one_question_per_image":
                True,

            "global_image_uniqueness":
                True,

            "old_new_question_overlap":
                0,

            "old_new_image_overlap":
                0,

            "old_new_image_id_overlap":
                0,

            "calibration_new_question_overlap":
                0,

            "selection_uses_answer":
                False,

            "selection_uses_model_outcomes":
                False,

            "selection_rule":
                manifest_summary[
                    "selection_rule"
                ],
        },

    "model":
        "meta-llama/Llama-3.2-11B-Vision-Instruct",

    "attention_implementation":
        "eager",

    "prompt_policy":
        {
            "use_original_tallyqa_question":
                True,

            "append_instruction":
                "Return the number only.",

            "question_rewriting":
                False,
        },

    "controller":
        {
            "type":
                "natural_adapted_k500",

            "natural_calibration_n":
                500,

            "head":
                "L18H13",

            "actions":
                [
                    0.0,
                    1.0,
                    1.5,
                    2.0,
                    4.0,
                ],

            "feature_family":
                "full_geometry",

            "feature_count":
                39,

            "ridge_alpha":
                0.01,

            "utility_threshold":
                0.1,

            "retuning_allowed":
                False,
        },

    "primary_endpoint":
        {
            "population":
                "all_4000",

            "metric":
                "paired_accuracy_change",

            "paired_test":
                "two-sided exact McNemar",

            "success_rule":
                (
                    "accuracy_change > 0 and "
                    "exact_McNemar_p < 0.05"
                ),
        },

    "prespecified_secondary_endpoints":
        [
            "simple_accuracy_change",
            "complex_accuracy_change",
            "repairs",
            "breaks",
            "net_repairs",
            "repair_rate_among_baseline_wrong",
            "break_rate_among_baseline_correct",
            "intervention_rate",
        ],

    "bootstrap":
        {
            "repetitions":
                20000,

            "seed":
                20260912,
        },

    "natural_ood_zero_shot":
        False,

    "natural_adapted_k500":
        True,

    "prohibited_after_outcomes_are_seen":
        [
            "controller_retraining",
            "calibration_set_change",
            "head_retuning",
            "action_set_retuning",
            "feature_retuning",
            "ridge_retuning",
            "threshold_retuning",
            "prompt_rewriting",
            "sample_reselection",
            "primary_endpoint_change",
        ],

    "freeze_chain":
        {
            "controller_freeze_commit":
                controller_commit,

            "confirmation_manifest_freeze_commit":
                manifest_commit,

            "confirmation_image_freeze_commit":
                image_commit,
        },
}


OOD_PROTOCOL.write_text(
    json.dumps(
        ood_protocol_obj,
        indent=2,
        sort_keys=True,
    )
    +
    "\n"
)


# ============================================================
# Image-freeze config
# ============================================================

image_freeze_obj = {
    "protocol_name":
        "AROMA TallyQA Natural "
        "Confirmation v2 Image Freeze",

    "manifest":
        {
            "path":
                str(
                    MANIFEST.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                EXPECTED_MANIFEST_SHA256,

            "n_questions":
                4000,

            "n_unique_image_paths":
                4000,
        },

    "image_inventory":
        {
            "path":
                str(
                    INVENTORY.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                EXPECTED_INVENTORY_SHA256,

            "n_images":
                4000,

            "valid_images":
                4000,

            "missing_images":
                0,

            "unexpected_images":
                0,

            "corrupt_images":
                0,

            "unique_file_sha256":
                4000,

            "byte_duplicate_groups":
                0,

            "unique_pixel_sha256":
                4000,

            "pixel_duplicate_groups":
                0,

            "vg_100k":
                2341,

            "vg_100k_2":
                1659,

            "total_bytes":
                int(
                    inventory_summary[
                        "total_bytes"
                    ]
                ),
        },

    "data_integrity_policy":
        {
            "resampling_after_image_audit":
                False,

            "duplicate_removal_after_freeze":
                False,

            "image_replacement_after_freeze":
                False,
        },

    "natural_confirmation_v2_model_inference_started":
        False,

    "freeze_commit":
        image_commit,
}


IMAGE_FREEZE.write_text(
    json.dumps(
        image_freeze_obj,
        indent=2,
        sort_keys=True,
    )
    +
    "\n"
)


# ============================================================
# Copy old runner and insert V2 overrides
# ============================================================

text = SOURCE_RUNNER.read_text()

protocol_marker = (
    "# ============================================================\n"
    "# Protocol / freeze audit\n"
    "# ============================================================\n"
)

if protocol_marker not in text:
    raise RuntimeError(
        "Protocol marker not found."
    )


override = f'''
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
    "{EXPECTED_MANIFEST_SHA256}"
)

EXPECTED_INVENTORY_SHA256 = (
    "{EXPECTED_INVENTORY_SHA256}"
)

EXPECTED_CONTROLLER_SHA256 = (
    "{EXPECTED_CONTROLLER_SHA256}"
)

EXPECTED_CALIBRATION_MANIFEST_SHA256 = (
    "{EXPECTED_CALIBRATION_SHA256}"
)

NATURAL_CONTROLLER_FREEZE_COMMIT = (
    "{controller_commit}"
)

TALLYQA_V2_MANIFEST_FREEZE_COMMIT = (
    "{manifest_commit}"
)

TALLYQA_V2_IMAGE_FREEZE_COMMIT = (
    "{image_commit}"
)

'''

text = text.replace(
    protocol_marker,
    override
    +
    "\n"
    +
    protocol_marker,
    1,
)


# ============================================================
# Replace old protocol_audit wholesale
# ============================================================

audit_start = text.index(
    "def protocol_audit():"
)

resume_marker = (
    "# ============================================================\n"
    "# Resume\n"
    "# ============================================================"
)

audit_end = text.index(
    resume_marker,
    audit_start,
)


new_audit = r'''def protocol_audit():

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


'''

text = (
    text[
        :audit_start
    ]
    +
    new_audit
    +
    "\n\n"
    +
    text[
        audit_end:
    ]
)


# ============================================================
# Remove false zero-shot labeling from run metadata / prose
# ============================================================

text = re.sub(
    r'("natural_ood_zero_shot"\s*:\s*)True',
    r'\1False',
    text,
)

text = text.replace(
    "FROZEN ZERO-SHOT FINAL EVALUATION",
    "FROZEN K500 NATURAL-ADAPTED CONFIRMATION",
)

text = text.replace(
    "TallyQA frozen "
    '"\n            "zero-shot evaluation',
    "TallyQA natural-adapted "
    '"\n            "confirmation v2',
)


TARGET_RUNNER.write_text(
    text
)


# ============================================================
# Compile before creating runner protocol
# ============================================================

py_compile.compile(
    str(TARGET_RUNNER),
    doraise=True,
)

runner_sha = sha256_file(
    TARGET_RUNNER
)


# ============================================================
# Write final runner protocol
# ============================================================

runner_protocol_obj = {
    "protocol_name":
        "AROMA TallyQA Natural "
        "Confirmation v2 Frozen Final Runner",

    "runner_path":
        str(
            TARGET_RUNNER.relative_to(
                ROOT
            )
        ),

    "runner_source_sha256":
        runner_sha,

    "manifest_sha256":
        EXPECTED_MANIFEST_SHA256,

    "image_inventory_sha256":
        EXPECTED_INVENTORY_SHA256,

    "controller_bundle_sha256":
        EXPECTED_CONTROLLER_SHA256,

    "calibration_manifest_sha256":
        EXPECTED_CALIBRATION_SHA256,

    "model":
        "meta-llama/Llama-3.2-11B-Vision-Instruct",

    "attention_implementation":
        "eager",

    "frozen_head":
        "L18H13",

    "actions":
        [
            0.0,
            1.0,
            1.5,
            2.0,
            4.0,
        ],

    "feature_count":
        39,

    "ridge_alpha":
        0.01,

    "utility_threshold":
        0.1,

    "natural_calibration_n":
        500,

    "prompt_policy":
        {
            "use_original_tallyqa_question":
                True,

            "append_instruction":
                "Return the number only.",

            "question_rewriting":
                False,
        },

    "primary_endpoint":
        {
            "population":
                "all_4000",

            "metric":
                "paired_accuracy_change",

            "paired_test":
                "two-sided exact McNemar",

            "success_rule":
                (
                    "accuracy_change > 0 and "
                    "exact_McNemar_p < 0.05"
                ),
        },

    "bootstrap":
        {
            "repetitions":
                20000,

            "seed":
                20260912,
        },

    "gt_read_after_action_execution":
        True,

    "controller_retuning_allowed":
        False,

    "sample_reselection_allowed":
        False,

    "prompt_retuning_allowed":
        False,

    "natural_ood_zero_shot":
        False,

    "natural_adapted_k500":
        True,

    "frozen_before_confirmation_v2_model_inference":
        True,

    "freeze_commits":
        {
            "controller":
                controller_commit,

            "manifest":
                manifest_commit,

            "images":
                image_commit,
        },
}


RUNNER_PROTOCOL.write_text(
    json.dumps(
        runner_protocol_obj,
        indent=2,
        sort_keys=True,
    )
    +
    "\n"
)


print("\n" + "=" * 118)
print("CONFIRMATION-V2 FINAL EVALUATOR PREPARED")
print("=" * 118)

print(
    "Runner:",
    TARGET_RUNNER.relative_to(
        ROOT
    ),
)

print(
    "Runner SHA256:",
    runner_sha,
)

print(
    "Manifest SHA256:",
    EXPECTED_MANIFEST_SHA256,
)

print(
    "Inventory SHA256:",
    EXPECTED_INVENTORY_SHA256,
)

print(
    "Controller SHA256:",
    EXPECTED_CONTROLLER_SHA256,
)

print(
    "Calibration SHA256:",
    EXPECTED_CALIBRATION_SHA256,
)

print("\nCreated configs:")

for path in [
    OOD_PROTOCOL,
    IMAGE_FREEZE,
    CONTROLLER_MANIFEST,
    RUNNER_PROTOCOL,
]:
    print(
        path.relative_to(
            ROOT
        )
    )

print("\nPython compile: PASS")
print("No VLM inference performed.")
print("No confirmation prediction produced.")

