#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


ROOT = Path("/workspace/AromaExperiments")
os.chdir(ROOT)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "aroma_principled"))

import natural_ugr_stage2_common_v1 as C
import run_tallyqa_natural_confirmation_v2_final as nat


# ============================================================
# FROZEN PATHS
# ============================================================

POP = (
    ROOT
    / "manifests/prospective_mea_v2_confirmation_v1/"
      "final_stage2_population_v1.csv"
)

IMAGE_ROOT = (
    ROOT
    / "data/prospective_mea_v2_confirmation_v1/images"
)

POLICY_PLAN = (
    ROOT
    / "artifacts/prospective_mea_v2_confirmation_v1/"
      "stage2/policy_plan_v1/"
      "prospective_mea_v2_plan_v1.csv"
)

POLICY_FREEZE = (
    ROOT
    / "artifacts/prospective_mea_v2_confirmation_v1/"
      "stage2/policy_plan_v1/"
      "prospective_mea_v2_plan_freeze_v1.json"
)

REFERENCE_PLAN = (
    ROOT
    / "artifacts/prospective_mea_v2_confirmation_v1/"
      "stage2/reference_plan_v1/"
      "prospective_unified_reference_plan_v1.csv"
)

REFERENCE_FREEZE = (
    ROOT
    / "artifacts/prospective_mea_v2_confirmation_v1/"
      "stage2/reference_plan_v1/"
      "prospective_unified_reference_freeze_v1.json"
)

PROTOCOL = (
    ROOT
    / "aroma_principled/protocols/"
      "deployable_mea_v2_untouched_confirmation_v1.json"
)

U4_PATH = (
    ROOT
    / "outputs/aroma2/csa_gate_a_v3/U4_primary.npy"
)

SA_PATH = (
    ROOT
    / "outputs/prospective_mea_v2_confirmation_v1/"
      "stage2/mechanistic_collection_v1/"
      "sa_directions.npy"
)

MECH_SUMMARY = (
    ROOT
    / "outputs/prospective_mea_v2_confirmation_v1/"
      "stage2/mechanistic_collection_v1/"
      "mechanistic_summary_v1.csv"
)

E4_PATH = (
    ROOT
    / "aroma_principled/e4_deployable_mea_v2.py"
)

COMMON_PATH = (
    ROOT
    / "scripts/natural_ugr_stage2_common_v1.py"
)

NAT_PATH = (
    ROOT
    / "scripts/run_tallyqa_natural_confirmation_v2_final.py"
)

OUT_DIR = (
    ROOT
    / "artifacts/prospective_mea_v2_confirmation_v1/"
      "stage2/outcomes_v1"
)

EXEC_FREEZE = (
    OUT_DIR
    / "prospective_mea_v2_execution_freeze_v1.json"
)

RAW_OUT = (
    OUT_DIR
    / "prospective_apply63_raw_v1.jsonl"
)

FULL_PAIRED = (
    OUT_DIR
    / "prospective_paired_outcomes_v1.csv"
)

SUMMARY = (
    OUT_DIR
    / "prospective_summary_v1.json"
)

EXPECTED_N = 630
EXPECTED_APPLY = 63

EXPECTED_U4_SHA = (
    "af50da2cf49268cc55dc33d44dad50c0cd5a4fa29cda6cad3c01cc0b4ae63f14"
)

EXPECTED_PROTOCOL_SHA = (
    "9411fdfbc27506cf7e8d1299a35ccf26a1ca2170121609883e7ffaf084b5d269"
)

EXPECTED_MODEL_REVISION = (
    "9eb2daaa8597bf192a8b0e73f848f3a102794df5"
)

EXPECTED_POLICY_PLAN_SHA = (
    "9fb39d184f754d21c976a2b6ee5e5a6d0638f4b3d366ead9b5f3f6c892dd47b0"
)


# ============================================================
# LOAD FROZEN E4 FUNCTIONS WITHOUT RUNNING E4.main()
# ============================================================

spec = importlib.util.spec_from_file_location(
    "frozen_e4_mea_v2",
    E4_PATH,
)

e4 = importlib.util.module_from_spec(spec)
sys.modules["frozen_e4_mea_v2"] = e4
spec.loader.exec_module(e4)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git_head():
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        )
        .strip()
    )


def normalize_id(x):
    s = str(x).strip()

    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    return s


def as_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    s = str(x).strip().lower()

    if s in ("true", "1"):
        return True

    if s in ("false", "0"):
        return False

    raise RuntimeError(
        f"Cannot parse boolean: {x!r}"
    )


def load_jsonl(path):
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ============================================================
# HARD INPUT VALIDATION
# ============================================================

def validate_inputs():

    required = [
        POP,
        POLICY_PLAN,
        POLICY_FREEZE,
        REFERENCE_PLAN,
        REFERENCE_FREEZE,
        PROTOCOL,
        U4_PATH,
        SA_PATH,
        MECH_SUMMARY,
        E4_PATH,
        COMMON_PATH,
        NAT_PATH,
    ]

    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)

    if sha256(U4_PATH) != EXPECTED_U4_SHA:
        raise RuntimeError(
            "U4 SHA mismatch."
        )

    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError(
            "Prospective protocol SHA mismatch."
        )

    if sha256(POLICY_PLAN) != EXPECTED_POLICY_PLAN_SHA:
        raise RuntimeError(
            "Frozen MEA policy plan SHA mismatch."
        )

    protocol = json.loads(
        PROTOCOL.read_text()
    )

    if (
        protocol["frozen_model"]["revision"]
        != EXPECTED_MODEL_REVISION
    ):
        raise RuntimeError(
            "Model revision mismatch."
        )

    mea_spec = (
        protocol[
            "frozen_deployment"
        ]["mea"]
    )

    if int(mea_spec["max_iterations"]) != 20:
        raise RuntimeError(
            "max_iterations drift."
        )

    if not np.isclose(
        float(mea_spec["trust_fraction"]),
        0.10,
    ):
        raise RuntimeError(
            "trust_fraction drift."
        )

    if not np.isclose(
        float(mea_spec["gradient_floor"]),
        1e-12,
    ):
        raise RuntimeError(
            "gradient_floor drift."
        )

    if not bool(
        mea_spec["bf16_boundary_tie_step"]
    ):
        raise RuntimeError(
            "BF16 tie handling drift."
        )

    if C.MODEL_REVISION != EXPECTED_MODEL_REVISION:
        raise RuntimeError(
            "Common module model revision drift."
        )

    if not np.isclose(
        float(e4.BETA_TRUST),
        0.10,
    ):
        raise RuntimeError(
            "E4 trust fraction drift."
        )

    if not np.isclose(
        float(e4.PG_FLOOR),
        1e-12,
    ):
        raise RuntimeError(
            "E4 gradient floor drift."
        )

    pop = pd.read_csv(
        POP,
        low_memory=False,
    ).reset_index(drop=True)

    policy = pd.read_csv(
        POLICY_PLAN,
        low_memory=False,
    ).reset_index(drop=True)

    reference = pd.read_csv(
        REFERENCE_PLAN,
        low_memory=False,
    ).reset_index(drop=True)

    mech = pd.read_csv(
        MECH_SUMMARY,
        low_memory=False,
    ).reset_index(drop=True)

    SA = np.load(
        SA_PATH
    ).astype(np.float64)

    if len(pop) != EXPECTED_N:
        raise RuntimeError(
            f"Population N={len(pop)}"
        )

    if len(policy) != EXPECTED_N:
        raise RuntimeError(
            "Policy plan N mismatch."
        )

    if len(reference) != EXPECTED_N:
        raise RuntimeError(
            "Reference plan N mismatch."
        )

    if len(mech) != EXPECTED_N:
        raise RuntimeError(
            "Mechanistic summary N mismatch."
        )

    if SA.shape != (EXPECTED_N, C.HEAD_DIM):
        raise RuntimeError(
            f"Bad SA shape: {SA.shape}"
        )

    expected_index = np.arange(
        EXPECTED_N,
        dtype=int,
    )

    if not np.array_equal(
        policy["stage2_index"]
        .astype(int)
        .to_numpy(),
        expected_index,
    ):
        raise RuntimeError(
            "Policy stage2_index mismatch."
        )

    if not np.array_equal(
        reference["stage2_index"]
        .astype(int)
        .to_numpy(),
        expected_index,
    ):
        raise RuntimeError(
            "Reference stage2_index mismatch."
        )

    pop_uid = (
        pop["natural_ugr_uid"]
        .astype(str)
        .to_numpy()
    )

    policy_uid = (
        policy["natural_ugr_uid"]
        .astype(str)
        .to_numpy()
    )

    ref_uid = (
        reference["natural_ugr_uid"]
        .astype(str)
        .to_numpy()
    )

    mech_uid = (
        mech["cohort_uid"]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        pop_uid,
        policy_uid,
    ):
        raise RuntimeError(
            "Population/policy UID mismatch."
        )

    if not np.array_equal(
        pop_uid,
        ref_uid,
    ):
        raise RuntimeError(
            "Population/reference UID mismatch."
        )

    if not np.array_equal(
        pop_uid,
        mech_uid,
    ):
        raise RuntimeError(
            "Population/mechanistic UID mismatch."
        )

    pop_qid = np.asarray(
        [
            normalize_id(x)
            for x in pop["question_id"]
        ],
        dtype=str,
    )

    mech_qid = np.asarray(
        [
            normalize_id(x)
            for x in mech["raw_sample_id"]
        ],
        dtype=str,
    )

    if not np.array_equal(
        pop_qid,
        mech_qid,
    ):
        raise RuntimeError(
            "Population/mechanistic QID mismatch."
        )

    policy_apply = np.asarray(
        [
            as_bool(x)
            for x in policy["gate_apply"]
        ],
        dtype=bool,
    )

    ref_apply = np.asarray(
        [
            as_bool(x)
            for x in reference["gate_apply"]
        ],
        dtype=bool,
    )

    if not np.array_equal(
        policy_apply,
        ref_apply,
    ):
        raise RuntimeError(
            "MEA/reference gate decisions differ."
        )

    if int(policy_apply.sum()) != EXPECTED_APPLY:
        raise RuntimeError(
            f"Expected APPLY=63, got "
            f"{int(policy_apply.sum())}"
        )

    # --------------------------------------------------------
    # Re-derive frozen target rule from Stage-1 probabilities.
    # No GT is used here.
    # --------------------------------------------------------

    for i, row in pop.iterrows():

        b = int(
            row["baseline_prediction"]
        )

        candidates = [
            (
                float(
                    row[
                        f"feature__baseline_numprob_{n}"
                    ]
                ),
                n,
            )
            for n in range(16)
            if n != b
        ]

        t = max(
            candidates,
            key=lambda z: (
                z[0],
                -z[1],
            ),
        )[1]

        frozen_t = int(
            policy.iloc[i][
                "runner_up_target"
            ]
        )

        if t != frozen_t:
            raise RuntimeError(
                f"Target replay mismatch row={i}"
            )

    # APPLY rows must have non-identity reference action.
    for i in np.flatnonzero(
        policy_apply
    ):
        if (
            str(
                reference.iloc[i][
                    "reference_method"
                ]
            )
            ==
            "IDENTITY"
        ):
            raise RuntimeError(
                f"APPLY row {i} has IDENTITY reference."
            )

    # Non-APPLY rows must be reference identity.
    for i in np.flatnonzero(
        ~policy_apply
    ):
        if (
            str(
                reference.iloc[i][
                    "reference_method"
                ]
            )
            !=
            "IDENTITY"
        ):
            raise RuntimeError(
                f"NOOP row {i} has nonidentity reference."
            )

    for i in np.flatnonzero(
        policy_apply
    ):
        image_path = (
            IMAGE_ROOT
            / str(
                pop.iloc[i]["image"]
            )
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                image_path
            )

    policy_freeze = json.loads(
        POLICY_FREEZE.read_text()
    )

    reference_freeze = json.loads(
        REFERENCE_FREEZE.read_text()
    )

    if (
        policy_freeze["plan_sha256"]
        != sha256(POLICY_PLAN)
    ):
        raise RuntimeError(
            "Policy freeze does not match policy plan."
        )

    if (
        reference_freeze[
            "reference_plan_sha256"
        ]
        != sha256(REFERENCE_PLAN)
    ):
        raise RuntimeError(
            "Reference freeze does not match reference plan."
        )

    return (
        pop,
        policy,
        reference,
        mech,
        SA,
        policy_apply,
    )


# ============================================================
# PRE-INFERENCE EXECUTION FREEZE
# ============================================================

def preflight():

    (
        pop,
        policy,
        reference,
        mech,
        SA,
        apply_mask,
    ) = validate_inputs()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RAW_OUT.exists():
        raise RuntimeError(
            "Raw outcome file already exists. "
            "Preflight must occur before outcome execution."
        )

    if SUMMARY.exists():
        raise RuntimeError(
            "Summary already exists."
        )

    if EXEC_FREEZE.exists():
        raise RuntimeError(
            "Execution freeze already exists; "
            "refusing overwrite."
        )

    freeze = {
        "experiment":
            "prospective_mea_v2_untouched_confirmation_execution_v1",

        "status":
            "FROZEN_BEFORE_REFERENCE_OR_MEA_OUTCOME_EXECUTION",

        "git_head":
            git_head(),

        "runner_path":
            str(
                Path(__file__)
                .resolve()
                .relative_to(ROOT)
            ),

        "runner_sha256":
            sha256(
                Path(__file__).resolve()
            ),

        "population_sha256":
            sha256(POP),

        "policy_plan_sha256":
            sha256(POLICY_PLAN),

        "policy_freeze_sha256":
            sha256(POLICY_FREEZE),

        "reference_plan_sha256":
            sha256(REFERENCE_PLAN),

        "reference_freeze_sha256":
            sha256(REFERENCE_FREEZE),

        "protocol_sha256":
            sha256(PROTOCOL),

        "u4_sha256":
            sha256(U4_PATH),

        "sa_directions_sha256":
            sha256(SA_PATH),

        "mechanistic_summary_sha256":
            sha256(MECH_SUMMARY),

        "e4_source_sha256":
            sha256(E4_PATH),

        "common_source_sha256":
            sha256(COMMON_PATH),

        "natural_input_source_sha256":
            sha256(NAT_PATH),

        "population_n":
            EXPECTED_N,

        "apply_n":
            int(apply_mask.sum()),

        "identity_n":
            int((~apply_mask).sum()),

        "model_revision":
            EXPECTED_MODEL_REVISION,

        "dtype":
            "bfloat16",

        "attention_impl":
            "eager",

        "mea_max_iterations":
            20,

        "mea_trust_fraction":
            0.10,

        "mea_gradient_floor":
            1e-12,

        "bf16_boundary_tie_step":
            True,

        "reference_execution":
            (
                "execute only the frozen Unified selected "
                "method/gain on each APPLY row"
            ),

        "alternative_execution":
            (
                "execute frozen runner-up target + frozen MEA v2 "
                "on the identical APPLY rows"
            ),

        "candidate_sweep_executed":
            False,

        "ground_truth_used_for_policy":
            False,

        "outcomes_observed":
            False,
    }

    EXEC_FREEZE.write_text(
        json.dumps(
            freeze,
            indent=2,
        )
        + "\n"
    )

    print("=" * 88)
    print("PROSPECTIVE MEA-v2 EXECUTION PREFLIGHT: PASS")
    print("=" * 88)
    print("N:", EXPECTED_N)
    print("APPLY:", int(apply_mask.sum()))
    print("IDENTITY:", int((~apply_mask).sum()))
    print("runner SHA:", freeze["runner_sha256"])
    print("git HEAD:", freeze["git_head"])
    print("execution freeze SHA:", sha256(EXEC_FREEZE))
    print()
    print("NO REFERENCE OUTCOME EXECUTED.")
    print("NO MEA OUTCOME EXECUTED.")


# ============================================================
# REFERENCE ACTION
# ============================================================

def execute_reference(
    *,
    model,
    inputs,
    numeral_ids,
    method,
    gain,
    alpha_prop,
    U,
    sa_direction,
):

    method = str(method)
    gain = float(gain)
    alpha_prop = float(alpha_prop)

    if method in ("MN", "SA"):

        modifier = C.DirectionModifier(
            model,
            alpha_prop,
            gain,
            method,
            U,
            sa_direction,
        )

        effective_alpha = None

    elif method == "WHOLE":

        effective_alpha = (
            1.0
            +
            gain
            *
            (alpha_prop - 1.0)
        )

        modifier = C.HeadGainModifier(
            model,
            C.LAYER,
            C.HEAD,
            effective_alpha,
        )

    else:
        raise RuntimeError(
            f"Unexpected reference method: {method}"
        )

    modifier.register()

    try:
        with torch.inference_mode():
            state = C.score_no_intervention(
                model,
                inputs,
                numeral_ids,
            )
    finally:
        modifier.remove()

    calls = int(
        modifier.calls
    )

    if calls <= 0:
        raise RuntimeError(
            f"{method}: intervention hook not called."
        )

    pred = int(
        state["best_numeral"]
    )

    return {
        "prediction":
            pred,

        "hook_calls":
            calls,

        "matched_effective_alpha":
            (
                float(effective_alpha)
                if effective_alpha is not None
                else None
            ),
    }


# ============================================================
# EXACT FROZEN MEA-v2 EXECUTION
# ============================================================

def execute_mea(
    *,
    model,
    inputs,
    numeral_ids,
    target,
    alpha_prop,
    P,
):

    target = int(target)
    alpha_prop = float(alpha_prop)

    cumulative = np.zeros(
        C.HEAD_DIM,
        dtype=np.float64,
    )

    baseline = e4.evaluate_state(
        model,
        inputs,
        numeral_ids,
        cumulative,
        target,
    )

    live_baseline = int(
        baseline["prediction"]
    )

    initial_margin = float(
        baseline["margin"]
    )

    success = (
        live_baseline == target
    )

    pg_abstain = False
    path_length = 0.0
    trajectory = []

    for iteration in range(
        1,
        21,
    ):

        if success:
            break

        state = e4.evaluate_state(
            model,
            inputs,
            numeral_ids,
            cumulative,
            target,
        )

        if int(
            state["prediction"]
        ) == target:
            success = True
            break

        h = np.asarray(
            state["h"],
            dtype=np.float64,
        )

        g = np.asarray(
            state["g"],
            dtype=np.float64,
        )

        h_norm = float(
            np.linalg.norm(h)
        )

        PG = g @ P

        pg_norm = float(
            np.linalg.norm(PG)
        )

        if not np.isfinite(pg_norm):
            raise RuntimeError(
                "Non-finite projected gradient."
            )

        if pg_norm <= e4.PG_FLOOR:
            pg_abstain = True
            break

        margin_before = float(
            state["margin"]
        )

        epsilon_star = (
            abs(margin_before)
            / pg_norm
        )

        B_i = (
            abs(
                alpha_prop - 1.0
            )
            * h_norm
        )

        tau = (
            e4.BETA_TRUST
            * B_i
        )

        unconstrained = (
            -margin_before
            * PG
            / (pg_norm ** 2)
        )

        unconstrained_norm = float(
            np.linalg.norm(
                unconstrained
            )
        )

        if (
            margin_before == 0.0
            and
            int(
                state["prediction"]
            ) != target
        ):
            step = (
                -tau
                * PG
                / pg_norm
            )

            clipped = True
            bf16_tie_step = True

        elif unconstrained_norm <= tau:

            step = unconstrained
            clipped = False
            bf16_tie_step = False

        else:

            step = (
                -tau
                * PG
                / pg_norm
            )

            clipped = True
            bf16_tie_step = False

        step_norm = float(
            np.linalg.norm(step)
        )

        if step_norm > tau + 1e-5:
            raise RuntimeError(
                f"MEA trust budget violation: "
                f"{step_norm} > {tau}"
            )

        cumulative = (
            cumulative
            + step
        )

        path_length += step_norm

        after = e4.evaluate_state(
            model,
            inputs,
            numeral_ids,
            cumulative,
            target,
        )

        trajectory.append({
            "iteration":
                int(iteration),

            "competitor_before":
                int(
                    state["competitor"]
                ),

            "margin_before":
                margin_before,

            "margin_after":
                float(
                    after["margin"]
                ),

            "epsilon_star":
                float(
                    epsilon_star
                ),

            "tau":
                float(tau),

            "step_norm":
                float(step_norm),

            "clipped":
                bool(clipped),

            "bf16_tie_step":
                bool(
                    bf16_tie_step
                ),

            "prediction_after":
                int(
                    after["prediction"]
                ),
        })

        if int(
            after["prediction"]
        ) == target:
            success = True
            break

    final_state = e4.evaluate_state(
        model,
        inputs,
        numeral_ids,
        cumulative,
        target,
    )

    final_prediction = int(
        final_state["prediction"]
    )

    return {
        "live_baseline_prediction":
            live_baseline,

        "initial_margin":
            initial_margin,

        "final_margin":
            float(
                final_state["margin"]
            ),

        "final_prediction":
            final_prediction,

        "target_reached":
            bool(
                final_prediction
                ==
                target
            ),

        "success":
            bool(success),

        "pg_abstain":
            bool(
                pg_abstain
            ),

        "iterations":
            len(
                trajectory
            ),

        "path_length":
            float(
                path_length
            ),

        "trajectory":
            trajectory,
    }


# ============================================================
# FINAL PROSPECTIVE RUN
# ============================================================

def run():

    (
        pop,
        policy,
        reference,
        mech,
        SA,
        apply_mask,
    ) = validate_inputs()

    if not EXEC_FREEZE.is_file():
        raise RuntimeError(
            "Execution freeze missing."
        )

    freeze = json.loads(
        EXEC_FREEZE.read_text()
    )

    current_runner_sha = sha256(
        Path(__file__).resolve()
    )

    if (
        current_runner_sha
        !=
        freeze["runner_sha256"]
    ):
        raise RuntimeError(
            "Runner changed after execution freeze."
        )

    frozen_inputs = {
        "population_sha256":
            POP,

        "policy_plan_sha256":
            POLICY_PLAN,

        "policy_freeze_sha256":
            POLICY_FREEZE,

        "reference_plan_sha256":
            REFERENCE_PLAN,

        "reference_freeze_sha256":
            REFERENCE_FREEZE,

        "protocol_sha256":
            PROTOCOL,

        "u4_sha256":
            U4_PATH,

        "sa_directions_sha256":
            SA_PATH,

        "mechanistic_summary_sha256":
            MECH_SUMMARY,

        "e4_source_sha256":
            E4_PATH,

        "common_source_sha256":
            COMMON_PATH,

        "natural_input_source_sha256":
            NAT_PATH,
    }

    for key, path in frozen_inputs.items():
        if sha256(path) != freeze[key]:
            raise RuntimeError(
                f"Frozen input changed: {key}"
            )

    apply_indices = np.flatnonzero(
        apply_mask
    )

    if len(
        apply_indices
    ) != EXPECTED_APPLY:
        raise RuntimeError(
            "APPLY count changed."
        )

    expected_uids = [
        str(
            pop.iloc[i][
                "natural_ugr_uid"
            ]
        )
        for i in apply_indices
    ]

    existing = load_jsonl(
        RAW_OUT
    )

    existing_uids = [
        str(
            r[
                "natural_ugr_uid"
            ]
        )
        for r in existing
    ]

    if (
        existing_uids
        !=
        expected_uids[
            :len(existing_uids)
        ]
    ):
        raise RuntimeError(
            "Outcome resume is not exact APPLY prefix."
        )

    if len(existing) > EXPECTED_APPLY:
        raise RuntimeError(
            "Too many existing outcome rows."
        )

    print("=" * 88)
    print("PROSPECTIVE MEA-v2 FINAL OUTCOME EXECUTION")
    print("=" * 88)
    print("APPLY N:", EXPECTED_APPLY)
    print("Resume completed:", len(existing))
    print("Model revision:", C.MODEL_REVISION)
    print("Reference: frozen Unified method/gain")
    print("Alternative: frozen runner-up + MEA-v2")
    print("=" * 88)

    if len(existing) == EXPECTED_APPLY:
        print(
            "All APPLY outcomes already complete."
        )
        return

    U = np.load(
        U4_PATH
    ).astype(np.float64)

    if U.shape != (128, 4):
        raise RuntimeError(
            f"Unexpected U4 shape: {U.shape}"
        )

    P = U @ U.T

    print("Loading processor...")

    processor = (
        AutoProcessor
        .from_pretrained(
            C.MODEL_ID,
            revision=
                C.MODEL_REVISION,
        )
    )

    numeral_ids = (
        C.numeral_token_ids(
            processor
        )
    )

    if sorted(
        numeral_ids.keys()
    ) != list(range(16)):
        raise RuntimeError(
            "Numeral universe != 0..15."
        )

    print("Loading BF16 model...")

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            C.MODEL_ID,
            revision=
                C.MODEL_REVISION,
            torch_dtype=
                torch.bfloat16,
            device_map="auto",
            attn_implementation=
                "eager",
        )
        .eval()
    )

    model.requires_grad_(
        False
    )

    start = len(existing)

    for apply_pos in range(
        start,
        EXPECTED_APPLY,
    ):

        i = int(
            apply_indices[
                apply_pos
            ]
        )

        row = pop.iloc[i]
        ppol = policy.iloc[i]
        rpol = reference.iloc[i]

        uid = str(
            row[
                "natural_ugr_uid"
            ]
        )

        archived_baseline = int(
            row[
                "baseline_prediction"
            ]
        )

        alpha_prop = float(
            row[
                "selected_alpha"
            ]
        )

        target = int(
            ppol[
                "runner_up_target"
            ]
        )

        ref_method = str(
            rpol[
                "reference_method"
            ]
        )

        ref_gain = float(
            rpol[
                "reference_gain"
            ]
        )

        image_path = (
            IMAGE_ROOT
            /
            str(
                row[
                    "image"
                ]
            )
        )

        image = (
            Image.open(
                image_path
            )
            .convert(
                "RGB"
            )
        )

        inputs = (
            nat.prepare_tallyqa_inputs(
                processor,
                image,
                str(
                    row[
                        "question"
                    ]
                ),
            )
        )

        inputs = C.move_inputs(
            inputs,
            model,
        )

        # ----------------------------------------------------
        # Live no-intervention reproduction.
        # This never changes frozen target or frozen policy.
        # ----------------------------------------------------

        with torch.inference_mode():

            baseline_state = (
                C.score_no_intervention(
                    model,
                    inputs,
                    numeral_ids,
                )
            )

        live_baseline = int(
            baseline_state[
                "best_numeral"
            ]
        )

        # ----------------------------------------------------
        # Frozen old-Unified selected action.
        # ----------------------------------------------------

        ref_result = (
            execute_reference(
                model=model,
                inputs=inputs,
                numeral_ids=
                    numeral_ids,
                method=
                    ref_method,
                gain=
                    ref_gain,
                alpha_prop=
                    alpha_prop,
                U=U,
                sa_direction=
                    SA[i],
            )
        )

        # ----------------------------------------------------
        # Frozen runner-up target + MEA-v2.
        # ----------------------------------------------------

        mea_result = (
            execute_mea(
                model=model,
                inputs=inputs,
                numeral_ids=
                    numeral_ids,
                target=
                    target,
                alpha_prop=
                    alpha_prop,
                P=P,
            )
        )

        if (
            int(
                mea_result[
                    "live_baseline_prediction"
                ]
            )
            !=
            live_baseline
        ):
            raise RuntimeError(
                f"{uid}: repeated live baseline "
                "is not deterministic."
            )

        # Ground truth enters only after both frozen
        # policy executions are complete.
        gt = int(
            row[
                "ground_truth"
            ]
        )

        rec = {
            "apply_position":
                int(
                    apply_pos
                ),

            "stage2_index":
                i,

            "natural_ugr_uid":
                uid,

            "question_id":
                normalize_id(
                    row[
                        "question_id"
                    ]
                ),

            "subset":
                str(
                    row[
                        "subset"
                    ]
                ),

            "ground_truth":
                gt,

            "archived_baseline_prediction":
                archived_baseline,

            "live_baseline_prediction":
                live_baseline,

            "live_baseline_drift":
                bool(
                    live_baseline
                    !=
                    archived_baseline
                ),

            "selected_alpha":
                alpha_prop,

            "gate_probability":
                float(
                    ppol[
                        "unified_gate_probability"
                    ]
                ),

            "reference_method":
                ref_method,

            "reference_gain":
                ref_gain,

            "reference_prediction":
                int(
                    ref_result[
                        "prediction"
                    ]
                ),

            "reference_hook_calls":
                int(
                    ref_result[
                        "hook_calls"
                    ]
                ),

            "reference_matched_effective_alpha":
                ref_result[
                    "matched_effective_alpha"
                ],

            "runner_up_target":
                target,

            "target_correct":
                bool(
                    target == gt
                ),

            "mea_live_baseline_prediction":
                int(
                    mea_result[
                        "live_baseline_prediction"
                    ]
                ),

            "mea_initial_margin":
                float(
                    mea_result[
                        "initial_margin"
                    ]
                ),

            "mea_final_margin":
                float(
                    mea_result[
                        "final_margin"
                    ]
                ),

            "mea_final_prediction":
                int(
                    mea_result[
                        "final_prediction"
                    ]
                ),

            "mea_target_reached":
                bool(
                    mea_result[
                        "target_reached"
                    ]
                ),

            "mea_success":
                bool(
                    mea_result[
                        "success"
                    ]
                ),

            "mea_pg_abstain":
                bool(
                    mea_result[
                        "pg_abstain"
                    ]
                ),

            "mea_iterations":
                int(
                    mea_result[
                        "iterations"
                    ]
                ),

            "mea_path_length":
                float(
                    mea_result[
                        "path_length"
                    ]
                ),

            "mea_trajectory":
                mea_result[
                    "trajectory"
                ],
        }

        with RAW_OUT.open(
            "a"
        ) as f:
            f.write(
                json.dumps(
                    rec
                )
                + "\n"
            )

        print(
            f"[{apply_pos + 1:02d}/{EXPECTED_APPLY}] "
            f"{uid} "
            f"live={live_baseline} "
            f"ref={ref_method}@{ref_gain:g}"
            f"->{rec['reference_prediction']} "
            f"target={target} "
            f"mea->{rec['mea_final_prediction']} "
            f"iters={rec['mea_iterations']} "
            f"reached={rec['mea_target_reached']}"
        )

    print()
    print("=" * 88)
    print("PROSPECTIVE APPLY OUTCOMES COMPLETE")
    print("=" * 88)
    print("rows:", len(load_jsonl(RAW_OUT)))
    print("raw SHA256:", sha256(RAW_OUT))
    print()
    print("Run --analyze next.")


# ============================================================
# FROZEN PRIMARY / SECONDARY ANALYSIS
# ============================================================

def analyze():

    from scipy.stats import binomtest

    (
        pop,
        policy,
        reference,
        mech,
        SA,
        apply_mask,
    ) = validate_inputs()

    records = load_jsonl(
        RAW_OUT
    )

    if len(records) != EXPECTED_APPLY:
        raise RuntimeError(
            f"Need exactly {EXPECTED_APPLY} "
            f"raw APPLY outcomes; "
            f"found {len(records)}."
        )

    expected_indices = np.flatnonzero(
        apply_mask
    )

    expected_uids = [
        str(
            pop.iloc[i][
                "natural_ugr_uid"
            ]
        )
        for i in expected_indices
    ]

    actual_uids = [
        str(
            r[
                "natural_ugr_uid"
            ]
        )
        for r in records
    ]

    if actual_uids != expected_uids:
        raise RuntimeError(
            "Raw APPLY outcome ordering mismatch."
        )

    result_by_uid = {
        str(
            r[
                "natural_ugr_uid"
            ]
        ):
        r
        for r in records
    }

    rows = []

    for i, row in pop.iterrows():

        uid = str(
            row[
                "natural_ugr_uid"
            ]
        )

        gt = int(
            row[
                "ground_truth"
            ]
        )

        identity_pred = int(
            row[
                "baseline_prediction"
            ]
        )

        apply = bool(
            apply_mask[i]
        )

        if apply:

            rec = result_by_uid[
                uid
            ]

            reference_pred = int(
                rec[
                    "reference_prediction"
                ]
            )

            mea_pred = int(
                rec[
                    "mea_final_prediction"
                ]
            )

            target = int(
                rec[
                    "runner_up_target"
                ]
            )

        else:

            reference_pred = (
                identity_pred
            )

            mea_pred = (
                identity_pred
            )

            target = None

        rows.append({
            "stage2_index":
                i,

            "natural_ugr_uid":
                uid,

            "question_id":
                normalize_id(
                    row[
                        "question_id"
                    ]
                ),

            "subset":
                str(
                    row[
                        "subset"
                    ]
                ),

            "ground_truth":
                gt,

            "gate_apply":
                apply,

            "identity_prediction":
                identity_pred,

            "identity_correct":
                bool(
                    identity_pred
                    ==
                    gt
                ),

            "reference_prediction":
                reference_pred,

            "reference_correct":
                bool(
                    reference_pred
                    ==
                    gt
                ),

            "mea_prediction":
                mea_pred,

            "mea_correct":
                bool(
                    mea_pred
                    ==
                    gt
                ),

            "runner_up_target":
                target,
        })

    paired = pd.DataFrame(
        rows
    )

    idc = (
        paired[
            "identity_correct"
        ]
        .astype(bool)
        .to_numpy()
    )

    refc = (
        paired[
            "reference_correct"
        ]
        .astype(bool)
        .to_numpy()
    )

    meac = (
        paired[
            "mea_correct"
        ]
        .astype(bool)
        .to_numpy()
    )

    n_id = int(
        idc.sum()
    )

    n_ref = int(
        refc.sum()
    )

    n_mea = int(
        meac.sum()
    )

    mea_only = int(
        (
            meac
            &
            (~refc)
        ).sum()
    )

    ref_only = int(
        (
            refc
            &
            (~meac)
        ).sum()
    )

    discord = (
        mea_only
        +
        ref_only
    )

    if discord == 0:
        p_mcnemar = 1.0
    else:
        p_mcnemar = float(
            binomtest(
                mea_only,
                discord,
                p=0.5,
                alternative=
                    "two-sided",
            ).pvalue
        )

    ref_repairs = int(
        (
            (~idc)
            &
            refc
        ).sum()
    )

    ref_breaks = int(
        (
            idc
            &
            (~refc)
        ).sum()
    )

    mea_repairs = int(
        (
            (~idc)
            &
            meac
        ).sum()
    )

    mea_breaks = int(
        (
            idc
            &
            (~meac)
        ).sum()
    )

    apply_records = records

    target_correct = np.asarray(
        [
            bool(
                r[
                    "runner_up_target"
                ]
                ==
                r[
                    "ground_truth"
                ]
            )
            for r in apply_records
        ],
        dtype=bool,
    )

    reached = np.asarray(
        [
            bool(
                r[
                    "mea_target_reached"
                ]
            )
            for r in apply_records
        ],
        dtype=bool,
    )

    iterations = np.asarray(
        [
            int(
                r[
                    "mea_iterations"
                ]
            )
            for r in apply_records
        ],
        dtype=np.float64,
    )

    pg_abstain = np.asarray(
        [
            bool(
                r[
                    "mea_pg_abstain"
                ]
            )
            for r in apply_records
        ],
        dtype=bool,
    )

    live_drift = np.asarray(
        [
            bool(
                r[
                    "live_baseline_drift"
                ]
            )
            for r in apply_records
        ],
        dtype=bool,
    )

    correct_target_n = int(
        target_correct.sum()
    )

    correct_target_reached = int(
        (
            target_correct
            &
            reached
        ).sum()
    )

    summary = {
        "experiment":
            "prospective_mea_v2_untouched_confirmation_v1",

        "evidence_class":
            "prospective_untouched_confirmation",

        "population_n":
            EXPECTED_N,

        "apply_n":
            EXPECTED_APPLY,

        "identity_n":
            EXPECTED_N
            -
            EXPECTED_APPLY,

        "identity": {
            "correct":
                n_id,

            "accuracy":
                n_id
                /
                EXPECTED_N,
        },

        "old_frozen_unified": {
            "correct":
                n_ref,

            "accuracy":
                n_ref
                /
                EXPECTED_N,

            "gain_vs_identity_pp":
                100.0
                *
                (
                    n_ref
                    -
                    n_id
                )
                /
                EXPECTED_N,

            "repairs_vs_identity":
                ref_repairs,

            "breaks_vs_identity":
                ref_breaks,
        },

        "runner_up_plus_mea_v2": {
            "correct":
                n_mea,

            "accuracy":
                n_mea
                /
                EXPECTED_N,

            "gain_vs_identity_pp":
                100.0
                *
                (
                    n_mea
                    -
                    n_id
                )
                /
                EXPECTED_N,

            "repairs_vs_identity":
                mea_repairs,

            "breaks_vs_identity":
                mea_breaks,
        },

        "primary_endpoint_mea_vs_old_unified": {
            "difference_correct":
                n_mea
                -
                n_ref,

            "difference_pp":
                100.0
                *
                (
                    n_mea
                    -
                    n_ref
                )
                /
                EXPECTED_N,

            "mea_only_correct":
                mea_only,

            "old_unified_only_correct":
                ref_only,

            "discordant_n":
                discord,

            "two_sided_exact_mcnemar_p":
                p_mcnemar,
        },

        "secondary": {
            "runner_up_target_correct_n":
                correct_target_n,

            "runner_up_target_accuracy":
                correct_target_n
                /
                EXPECTED_APPLY,

            "assigned_target_reached_n":
                int(
                    reached.sum()
                ),

            "assigned_target_reachability":
                float(
                    reached.mean()
                ),

            "correct_target_reached_n":
                correct_target_reached,

            "correct_target_reachability":
                (
                    correct_target_reached
                    /
                    correct_target_n
                    if correct_target_n > 0
                    else None
                ),

            "median_mea_iterations":
                float(
                    np.median(
                        iterations
                    )
                ),

            "mean_mea_iterations":
                float(
                    np.mean(
                        iterations
                    )
                ),

            "projected_gradient_abstentions":
                int(
                    pg_abstain.sum()
                ),

            "live_baseline_drift_on_apply":
                int(
                    live_drift.sum()
                ),
        },

        "provenance": {
            "raw_outcome_sha256":
                sha256(RAW_OUT),

            "execution_freeze_sha256":
                sha256(EXEC_FREEZE),

            "policy_plan_sha256":
                sha256(POLICY_PLAN),

            "reference_plan_sha256":
                sha256(REFERENCE_PLAN),

            "protocol_sha256":
                sha256(PROTOCOL),
        },
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired.to_csv(
        FULL_PAIRED,
        index=False,
    )

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print("=" * 88)
    print("PROSPECTIVE MEA-v2 FINAL ANALYSIS")
    print("=" * 88)

    print(
        f"Identity: "
        f"{n_id}/{EXPECTED_N} "
        f"= {100*n_id/EXPECTED_N:.3f}%"
    )

    print(
        f"Old Unified: "
        f"{n_ref}/{EXPECTED_N} "
        f"= {100*n_ref/EXPECTED_N:.3f}%"
    )

    print(
        f"Runner-up + MEA-v2: "
        f"{n_mea}/{EXPECTED_N} "
        f"= {100*n_mea/EXPECTED_N:.3f}%"
    )

    print()

    print(
        "MEA-v2 minus Old Unified:",
        f"{100*(n_mea-n_ref)/EXPECTED_N:+.3f} pp",
    )

    print(
        "paired discordance:",
        f"{mea_only}:{ref_only}",
        "(MEA-only : Unified-only)",
    )

    print(
        "two-sided exact McNemar p:",
        p_mcnemar,
    )

    print()

    print(
        "Reference repairs/breaks:",
        ref_repairs,
        "/",
        ref_breaks,
    )

    print(
        "MEA repairs/breaks:",
        mea_repairs,
        "/",
        mea_breaks,
    )

    print()

    print(
        "Runner-up target correct:",
        f"{correct_target_n}/{EXPECTED_APPLY}",
        f"= {100*correct_target_n/EXPECTED_APPLY:.2f}%",
    )

    print(
        "Assigned targets reached:",
        f"{int(reached.sum())}/{EXPECTED_APPLY}",
        f"= {100*reached.mean():.2f}%",
    )

    if correct_target_n > 0:
        print(
            "Correct targets reached:",
            f"{correct_target_reached}/{correct_target_n}",
            f"= {100*correct_target_reached/correct_target_n:.2f}%",
        )

    print(
        "MEA iterations median/mean:",
        f"{np.median(iterations):.3f}",
        "/",
        f"{np.mean(iterations):.3f}",
    )

    print(
        "PG abstentions:",
        int(
            pg_abstain.sum()
        ),
    )

    print(
        "Live baseline drift on APPLY:",
        int(
            live_drift.sum()
        ),
    )

    print()
    print("summary:", SUMMARY)
    print("paired:", FULL_PAIRED)
    print("summary SHA256:", sha256(SUMMARY))


def main():

    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--preflight",
        action="store_true",
    )

    mode.add_argument(
        "--run",
        action="store_true",
    )

    mode.add_argument(
        "--analyze",
        action="store_true",
    )

    args = parser.parse_args()

    if args.preflight:
        preflight()

    elif args.run:
        run()

    else:
        analyze()


if __name__ == "__main__":
    main()
