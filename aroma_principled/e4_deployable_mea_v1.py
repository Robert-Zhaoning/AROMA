import argparse
import hashlib
import json
import os
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
import run_natural_ugr_v2_final_candidate_sweep_v1 as final_sweep


POP_PATH = ROOT / "manifests/natural_ugr_v2/final_stage2_population_v1.csv"
U4_PATH = ROOT / "outputs/aroma2/csa_gate_a_v3/U4_primary.npy"

OUT_DIR = ROOT / "aroma_principled/results/deployable_mea_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_U4_SHA = (
    "af50da2cf49268cc55dc33d44dad50c0cd5a4fa29cda6cad3c01cc0b4ae63f14"
)

NUMERALS = list(range(16))
BETA_TRUST = 0.10
PG_FLOOR = 1e-12


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deploy_target(row):
    b = int(row["baseline_prediction"])

    candidates = [
        (float(row[f"feature__baseline_numprob_{n}"]), n)
        for n in range(b + 1, 16)
    ]

    if not candidates:
        return None

    best_prob = max(p for p, _ in candidates)

    # frozen tie rule: nearest/smallest larger numeral
    return min(
        n for p, n in candidates
        if p == best_prob
    )


class ShiftedLocalMargin:
    def __init__(self, model, cumulative_delta):
        layer = (
            model.model.language_model.layers[C.LAYER]
        )

        self.o_proj = layer.cross_attn.o_proj

        self.cumulative_delta = torch.as_tensor(
            cumulative_delta,
            dtype=torch.float32,
        ).reshape(1, 1, C.HEAD_DIM)

        self.handle = None
        self.h_leaf = None
        self.h_base = None
        self.calls = 0

    def hook(self, module, inputs):
        x = inputs[0]

        H = x[..., C.START:C.END]

        base_delta = self.cumulative_delta.to(
            device=H.device,
            dtype=H.dtype,
        )

        H_shifted = H.detach() + base_delta

        h_base = H_shifted.float().mean(dim=1)

        h_leaf = (
            h_base.detach()
            .clone()
            .requires_grad_(True)
        )

        local_delta = h_leaf - h_base.detach()

        H_new = (
            H_shifted
            + local_delta.to(H.dtype).unsqueeze(1)
        )

        x_new = x.detach().clone()
        x_new[..., C.START:C.END] = H_new

        self.h_base = h_base
        self.h_leaf = h_leaf
        self.calls += 1

        return (x_new, *inputs[1:])

    def register(self):
        self.handle = self.o_proj.register_forward_pre_hook(
            self.hook
        )

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def evaluate_state(
    model,
    inputs,
    numeral_ids,
    cumulative_delta,
    target,
):
    cap = ShiftedLocalMargin(
        model,
        cumulative_delta,
    )
    cap.register()

    try:
        out = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

        token_ids = [
            numeral_ids[n]
            for n in NUMERALS
        ]

        z = (
            out.logits[0, -1, token_ids]
            .float()
            .to(torch.float64)
        )

        prediction = int(
            torch.argmax(z).item()
        )

        target_idx = int(target)

        mask = torch.ones(
            len(NUMERALS),
            dtype=torch.bool,
            device=z.device,
        )
        mask[target_idx] = False

        competitor_indices = (
            torch.arange(
                len(NUMERALS),
                device=z.device,
            )[mask]
        )

        competitor_local = int(
            torch.argmax(z[mask]).item()
        )

        competitor_idx = int(
            competitor_indices[
                competitor_local
            ].item()
        )

        competitor = int(
            NUMERALS[competitor_idx]
        )

        margin = (
            z[competitor_idx]
            - z[target_idx]
        )

        g = torch.autograd.grad(
            outputs=margin,
            inputs=cap.h_leaf,
            retain_graph=False,
            create_graph=False,
        )[0][0]

        h = (
            cap.h_base[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        g = (
            g.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        return {
            "prediction": prediction,
            "competitor": competitor,
            "margin": float(margin.detach().item()),
            "h": h,
            "g": g,
            "hook_calls": int(cap.calls),
        }

    finally:
        cap.remove()


def choose_smoke(df):
    work = df.copy()

    work["deploy_target"] = work.apply(
        deploy_target,
        axis=1,
    )

    work["target_shift"] = (
        work["deploy_target"]
        - work["baseline_prediction"]
    )

    chosen = []

    def take(mask):
        for idx in work[mask].index:
            if idx not in chosen:
                chosen.append(idx)
                return

    # ordinary local target
    take(
        work["deploy_target"].notna()
        & (work["target_shift"] == 1)
    )

    # expanded 11..15 target
    take(
        work["deploy_target"].notna()
        & (work["deploy_target"] >= 11)
    )

    # nontrivial longer jump
    take(
        work["deploy_target"].notna()
        & (work["target_shift"] >= 4)
    )

    # one natural abstention
    take(work["deploy_target"].isna())

    return work.loc[chosen].copy()


def run(args):
    df = pd.read_csv(POP_PATH)

    if len(df) != 631:
        raise RuntimeError(
            f"Expected N=631, got {len(df)}"
        )

    df["deploy_target"] = df.apply(
        deploy_target,
        axis=1,
    )

    if args.smoke:
        run_df = choose_smoke(df)
        max_iters = 3
        out_path = OUT_DIR / "smoke_v1.jsonl"
        manifest_path = OUT_DIR / "smoke_manifest_v1.csv"

        run_df[
            [
                "natural_ugr_uid",
                "manifest_index",
                "question_id",
                "image",
                "baseline_prediction",
                "selected_alpha",
                "deploy_target",
                "target_shift",
            ]
        ].to_csv(
            manifest_path,
            index=False,
        )

        out_path.unlink(missing_ok=True)

    else:
        run_df = df
        max_iters = 20
        out_path = OUT_DIR / "full_v1.jsonl"

    print("=" * 88)
    print("DEPLOYABLE MEA v1")
    print("mode:", "SMOKE" if args.smoke else "FULL")
    print("N rows:", len(run_df))
    print("max_iters:", max_iters)
    print("=" * 88)

    u4_sha = sha256_file(U4_PATH)

    if u4_sha != EXPECTED_U4_SHA:
        raise RuntimeError(
            f"U4 SHA mismatch: {u4_sha}"
        )

    U = np.load(U4_PATH).astype(np.float64)

    if U.shape != (128, 4):
        raise RuntimeError(
            f"Unexpected U4 shape: {U.shape}"
        )

    P = U @ U.T

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REVISION,
    )

    numeral_ids = C.numeral_token_ids(
        processor
    )

    if sorted(numeral_ids.keys()) != NUMERALS:
        raise RuntimeError(
            "Numeral universe is not 0..15"
        )

    print("Loading BF16 model...")

    model = (
        MllamaForConditionalGeneration
        .from_pretrained(
            C.MODEL_ID,
            revision=C.MODEL_REVISION,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        .eval()
    )

    model.requires_grad_(False)

    completed = set()

    if (not args.smoke) and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                completed.add(
                    json.loads(line)[
                        "natural_ugr_uid"
                    ]
                )

        print("Resume completed:", len(completed))

    image_root = Path(final_sweep.IMAGE_ROOT)

    if not image_root.exists():
        raise RuntimeError(
            f"IMAGE_ROOT missing: {image_root}"
        )

    processed = 0

    for _, row in run_df.iterrows():
        uid = str(row["natural_ugr_uid"])

        if uid in completed:
            continue

        target_raw = row["deploy_target"]

        if pd.isna(target_raw):
            rec = {
                "natural_ugr_uid": uid,
                "archived_baseline_prediction":
                    int(row["baseline_prediction"]),
                "deploy_target": None,
                "status": "natural_abstain",
                "iterations": 0,
            }

            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

            print(
                f"[ABSTAIN] {uid} "
                f"baseline={row['baseline_prediction']}"
            )
            continue

        target = int(target_raw)
        alpha_prop = float(
            row["selected_alpha"]
        )

        image_path = (
            image_root
            / str(row["image"])
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = nat.prepare_tallyqa_inputs(
            processor,
            image,
            str(row["question"]),
        )

        inputs = C.move_inputs(
            inputs,
            model,
        )

        cumulative = np.zeros(
            C.HEAD_DIM,
            dtype=np.float64,
        )

        baseline = evaluate_state(
            model,
            inputs,
            numeral_ids,
            cumulative,
            target,
        )

        live_baseline = int(
            baseline["prediction"]
        )

        archived_baseline = int(
            row["baseline_prediction"]
        )

        print()
        print(
            f"[{uid}] "
            f"arch={archived_baseline} "
            f"live={live_baseline} "
            f"target={target} "
            f"alpha={alpha_prop:g} "
            f"m0={baseline['margin']:+.6f}"
        )

        initial_margin = float(
            baseline["margin"]
        )

        path_length = 0.0
        iteration_rows = []
        success = (
            live_baseline == target
        )
        pg_abstain = False

        for iteration in range(
            1,
            max_iters + 1,
        ):
            if success:
                break

            state = evaluate_state(
                model,
                inputs,
                numeral_ids,
                cumulative,
                target,
            )

            if state["prediction"] == target:
                success = True
                break

            h = state["h"]
            g = state["g"]

            h_norm = float(
                np.linalg.norm(h)
            )

            PG = g @ P

            pg_norm = float(
                np.linalg.norm(PG)
            )

            if not np.isfinite(pg_norm):
                raise RuntimeError(
                    "Non-finite projected gradient"
                )

            if pg_norm <= PG_FLOOR:
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
                abs(alpha_prop - 1.0)
                * h_norm
            )

            tau = (
                BETA_TRUST
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

            # BF16 boundary-tie amendment:
            # an exact zero margin can still leave deterministic
            # argmax on the non-target numeral. Cross that
            # quantized boundary with one legal trust-region step.
            if (
                margin_before == 0.0
                and state["prediction"] != target
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
                    f"Budget violation: "
                    f"{step_norm} > {tau}"
                )

            cumulative = (
                cumulative + step
            )

            path_length += step_norm

            after = evaluate_state(
                model,
                inputs,
                numeral_ids,
                cumulative,
                target,
            )

            iteration_rows.append({
                "iteration": iteration,
                "competitor_before":
                    int(state["competitor"]),
                "margin_before":
                    margin_before,
                "margin_after":
                    float(after["margin"]),
                "epsilon_star":
                    float(epsilon_star),
                "tau":
                    float(tau),
                "step_norm":
                    float(step_norm),
                "clipped":
                    bool(clipped),
                "bf16_tie_step":
                    bool(bf16_tie_step),
                "prediction_after":
                    int(after["prediction"]),
            })

            print(
                f"  iter={iteration:02d} "
                f"c={state['competitor']:2d} "
                f"m {margin_before:+.5f}"
                f" -> {after['margin']:+.5f} "
                f"eps*={epsilon_star:.3f} "
                f"tau={tau:.3f} "
                f"pred={after['prediction']}"
            )

            if after["prediction"] == target:
                success = True
                break

        final_state = evaluate_state(
            model,
            inputs,
            numeral_ids,
            cumulative,
            target,
        )

        final_prediction = int(
            final_state["prediction"]
        )

        gt = int(row["ground_truth"])

        rec = {
            "natural_ugr_uid": uid,
            "question_id":
                str(row["question_id"]),
            "ground_truth": gt,

            "archived_baseline_prediction":
                archived_baseline,
            "live_baseline_prediction":
                live_baseline,

            "deploy_target": target,
            "selected_alpha":
                alpha_prop,

            "initial_margin":
                initial_margin,
            "final_margin":
                float(final_state["margin"]),

            "final_prediction":
                final_prediction,

            "target_reached":
                bool(final_prediction == target),
            "success":
                bool(success),

            "pg_abstain":
                bool(pg_abstain),

            "iterations":
                len(iteration_rows),

            "path_length":
                float(path_length),

            "archived_baseline_correct":
                bool(archived_baseline == gt),
            "live_baseline_correct":
                bool(live_baseline == gt),
            "final_correct":
                bool(final_prediction == gt),

            "repair_vs_archived":
                bool(
                    archived_baseline != gt
                    and final_prediction == gt
                ),

            "break_vs_archived":
                bool(
                    archived_baseline == gt
                    and final_prediction != gt
                ),

            "trajectory":
                iteration_rows,
        }

        with out_path.open("a") as f:
            f.write(
                json.dumps(rec) + "\n"
            )

        processed += 1

        print(
            f"  FINAL pred={final_prediction} "
            f"target_reached={final_prediction == target} "
            f"GT={gt}"
        )

    print()
    print("=" * 88)
    print("DONE")
    print("output:", out_path)
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    args = parser.parse_args()
    run(args)
