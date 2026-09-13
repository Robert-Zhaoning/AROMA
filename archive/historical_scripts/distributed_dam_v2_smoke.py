import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

from oracle_dam_v2_smoke import (
    OracleDAM,
    build_object_token_sets,
    generate_answer,
    load_metadata,
    move_inputs,
    numeral_token_ids,
    prepare_inputs,
    score_state,
)


MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

FROZEN_HEADS_PATH = Path(
    "outputs/proc_count_causal_v1/"
    "frozen_heads.json"
)


def load_candidate_heads():
    config = json.loads(
        FROZEN_HEADS_PATH.read_text(
            encoding="utf-8"
        )
    )

    heads = [
        (
            int(x["layer"]),
            int(x["head"]),
        )
        for x in config["candidate_heads"]
    ]

    return heads


def tensor_cv(x):
    if x is None:
        return None

    x = x.float()

    if len(x) <= 1:
        return 0.0

    mean = float(
        x.mean().item()
    )

    std = float(
        x.std(
            unbiased=False
        ).item()
    )

    if abs(mean) < 1e-12:
        return None

    return std / mean


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-id",
        required=True,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--query-scope",
        choices=[
            "answer",
            "q_many",
            "count_phrase",
        ],
        default="answer",
    )

    args = parser.parse_args()

    metadata = load_metadata()

    if args.sample_id not in metadata:
        raise KeyError(
            f"Unknown sample: "
            f"{args.sample_id}"
        )

    sample = metadata[
        args.sample_id
    ]

    gt = int(
        sample["ground_truth"]
    )

    candidate_heads = (
        load_candidate_heads()
    )

    print("=" * 100)
    print(
        "AROMA Distributed DAM-v2 "
        "Smoke Test"
    )
    print("=" * 100)

    print(
        "Sample      :",
        args.sample_id,
    )

    print(
        "GT          :",
        gt,
    )

    print(
        "Condition   :",
        sample["condition"],
    )

    print(
        "Beta        :",
        args.beta,
    )

    print(
        "Gamma       :",
        args.gamma,
    )

    print(
        "Query scope :",
        args.query_scope,
    )

    print(
        "Heads       :",
        len(candidate_heads),
    )

    for layer, head in candidate_heads:
        print(
            f"  L{layer}H{head}"
        )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
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

    image = (
        Image.open(
            sample["image_path"]
        )
        .convert("RGB")
    )

    (
        geometry,
        object_token_sets,
    ) = build_object_token_sets(
        sample,
        processor,
        image,
    )

    print(
        "\nVision geometry:"
    )

    print(
        json.dumps(
            geometry,
            indent=2,
        )
    )

    print(
        "\nGT object visual-token sets:"
    )

    for idx, tokens in enumerate(
        object_token_sets
    ):
        print(
            f"  object {idx}: "
            f"{len(tokens)} patches "
            f"[{tokens[0]} ... "
            f"{tokens[-1]}]"
        )

    (
        _formatted,
        inputs,
    ) = prepare_inputs(
        processor,
        image,
    )

    inputs = move_inputs(
        inputs,
        model,
    )

    numeral_ids = (
        numeral_token_ids(
            processor
        )
    )

    # ======================================================
    # Baseline
    # ======================================================

    baseline_score = score_state(
        model,
        inputs,
        numeral_ids,
        gt,
    )

    (
        baseline_text,
        baseline_pred,
    ) = generate_answer(
        model,
        processor,
        inputs,
    )

    print(
        "\n" + "=" * 100
    )
    print("BASELINE")
    print("=" * 100)

    print(
        "Prediction :",
        baseline_pred,
    )

    print(
        "Text       :",
        repr(baseline_text),
    )

    print(
        "GT logp    :",
        baseline_score[
            "gt_logp"
        ],
    )

    print(
        "GT margin  :",
        baseline_score[
            "margin"
        ],
    )

    # ======================================================
    # Distributed DAM
    # ======================================================

    interventions = []

    for layer, head in candidate_heads:
        dam = OracleDAM(
            model=model,
            layer_idx=layer,
            head_idx=head,
            object_token_sets=object_token_sets,
            gamma=args.gamma,
            beta=args.beta,
            query_scope=args.query_scope,
        )

        interventions.append(
            dam
        )

    for dam in interventions:
        dam.register()

    try:
        dam_score = score_state(
            model,
            inputs,
            numeral_ids,
            gt,
        )

        (
            dam_text,
            dam_pred,
        ) = generate_answer(
            model,
            processor,
            inputs,
        )

    finally:
        for dam in reversed(
            interventions
        ):
            dam.remove()

    print(
        "\n" + "=" * 100
    )
    print(
        "DISTRIBUTED ORACLE DAM-v2"
    )
    print("=" * 100)

    print(
        "Prediction :",
        dam_pred,
    )

    print(
        "Text       :",
        repr(dam_text),
    )

    print(
        "GT logp    :",
        dam_score[
            "gt_logp"
        ],
    )

    print(
        "GT margin  :",
        dam_score[
            "margin"
        ],
    )

    delta_logp = (
        dam_score["gt_logp"]
        -
        baseline_score["gt_logp"]
    )

    delta_margin = (
        dam_score["margin"]
        -
        baseline_score["margin"]
    )

    print(
        "\nDelta GT logp   :",
        delta_logp,
    )

    print(
        "Delta GT margin :",
        delta_margin,
    )

    print(
        "\nPrediction changed:",
        baseline_pred
        != dam_pred,
    )

    print(
        "Repaired:",
        (
            baseline_pred != gt
            and
            dam_pred == gt
        ),
    )

    # ======================================================
    # Per-head diagnostics
    # ======================================================

    print(
        "\n" + "=" * 100
    )
    print(
        "PER-HEAD DAM DIAGNOSTICS"
    )
    print("=" * 100)

    row_errors = []
    cv_before_all = []
    cv_after_all = []

    for (
        layer,
        head,
    ), dam in zip(
        candidate_heads,
        interventions,
    ):

        before = (
            dam.last_before_masses
        )

        after = (
            dam.last_after_masses
        )

        before_cv = tensor_cv(
            before
        )

        after_cv = tensor_cv(
            after
        )

        if before_cv is not None:
            cv_before_all.append(
                before_cv
            )

        if after_cv is not None:
            cv_after_all.append(
                after_cv
            )

        if (
            dam.last_row_sum_error
            is not None
        ):
            row_errors.append(
                float(
                    dam.last_row_sum_error
                )
            )

        print(
            f"\nL{layer}H{head}"
        )

        print(
            "  calls:",
            dam.calls,
        )

        print(
            "  row-sum error:",
            dam.last_row_sum_error,
        )

        print(
            "  object masses before:",
            before,
        )

        print(
            "  object masses after :",
            after,
        )

        print(
            "  CV before:",
            before_cv,
        )

        print(
            "  CV after :",
            after_cv,
        )

    print(
        "\n" + "=" * 100
    )
    print(
        "DISTRIBUTED SANITY SUMMARY"
    )
    print("=" * 100)

    if row_errors:
        print(
            "Max row-sum error:",
            max(row_errors),
        )

        print(
            "Mean row-sum error:",
            statistics.mean(
                row_errors
            ),
        )

    if cv_before_all:
        print(
            "Mean object CV before:",
            statistics.mean(
                cv_before_all
            ),
        )

    if cv_after_all:
        print(
            "Mean object CV after :",
            statistics.mean(
                cv_after_all
            ),
        )

    if (
        row_errors
        and
        max(row_errors) > 5e-3
    ):
        raise RuntimeError(
            "Attention normalization "
            "sanity check failed."
        )

    print(
        "\nDISTRIBUTED DAM-v2 "
        "SMOKE TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
