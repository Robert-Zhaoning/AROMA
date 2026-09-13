import argparse
import json
import statistics
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

from transformers.models.mllama.modeling_mllama import repeat_kv

from oracle_dam_v2_smoke import (
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
    cfg = json.loads(
        FROZEN_HEADS_PATH.read_text(
            encoding="utf-8"
        )
    )

    heads = [
        (
            int(x["layer"]),
            int(x["head"]),
        )
        for x in cfg["candidate_heads"]
    ]

    return heads


def tensor_cv(x):
    if x is None:
        return None

    x = x.float()

    if x.numel() <= 1:
        return 0.0

    mean = float(
        x.mean().item()
    )

    if abs(mean) < 1e-12:
        return None

    std = float(
        x.std(
            unbiased=False
        ).item()
    )

    return std / mean


class DistributedLayerDAM:
    def __init__(
        self,
        model,
        layer_idx,
        head_indices,
        object_token_sets,
        beta,
        gamma,
        query_scope,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)

        self.head_indices = [
            int(h)
            for h in head_indices
        ]

        self.object_token_sets = (
            object_token_sets
        )

        self.beta = float(beta)
        self.gamma = float(gamma)
        self.query_scope = query_scope

        self.module = (
            model
            .model
            .language_model
            .layers[
                self.layer_idx
            ]
            .cross_attn
        )

        self.original_forward = None

        self.layer_calls = 0

        self.head_calls = {
            h: 0
            for h in self.head_indices
        }

        self.last_before = {
            h: None
            for h in self.head_indices
        }

        self.last_after = {
            h: None
            for h in self.head_indices
        }

        self.last_row_sum_error = {
            h: None
            for h in self.head_indices
        }

    def select_queries(
        self,
        q_len,
        device,
    ):
        if self.query_scope == "answer":
            return torch.tensor(
                [q_len - 1],
                dtype=torch.long,
                device=device,
            )

        if self.query_scope == "q_many":
            if q_len > 12:
                return torch.tensor(
                    [12],
                    dtype=torch.long,
                    device=device,
                )

            return torch.tensor(
                [0],
                dtype=torch.long,
                device=device,
            )

        if self.query_scope == "count_phrase":
            valid = [
                idx
                for idx in [
                    11,
                    12,
                    13,
                    14,
                ]
                if idx < q_len
            ]

            if not valid:
                valid = [
                    q_len - 1
                ]

            return torch.tensor(
                valid,
                dtype=torch.long,
                device=device,
            )

        raise ValueError(
            self.query_scope
        )

    def register(self):
        module = self.module
        dam = self

        self.original_forward = (
            module.forward
        )

        def patched_forward(
            module_self,
            hidden_states,
            cross_attention_states=None,
            past_key_values=None,
            attention_mask=None,
            use_cache=None,
            **kwargs,
        ):
            bsz, q_len, _ = (
                hidden_states.size()
            )

            query_states = (
                module_self.q_proj(
                    hidden_states
                )
            )

            query_states = (
                query_states
                .view(
                    bsz,
                    q_len,
                    module_self.num_heads,
                    module_self.head_dim,
                )
                .transpose(
                    1,
                    2,
                )
            )

            query_states = (
                module_self.q_norm(
                    query_states
                )
            )

            if (
                cross_attention_states
                is not None
            ):
                key_states = (
                    module_self.k_proj(
                        cross_attention_states
                    )
                )

                value_states = (
                    module_self.v_proj(
                        cross_attention_states
                    )
                )

                key_states = (
                    key_states
                    .view(
                        bsz,
                        -1,
                        module_self.num_key_value_heads,
                        module_self.head_dim,
                    )
                    .transpose(
                        1,
                        2,
                    )
                )

                value_states = (
                    value_states
                    .view(
                        bsz,
                        -1,
                        module_self.num_key_value_heads,
                        module_self.head_dim,
                    )
                    .transpose(
                        1,
                        2,
                    )
                )

                key_states = (
                    module_self.k_norm(
                        key_states
                    )
                )

                if (
                    past_key_values
                    is not None
                ):
                    (
                        key_states,
                        value_states,
                    ) = (
                        past_key_values.update(
                            key_states,
                            value_states,
                            module_self.layer_idx,
                        )
                    )

            elif (
                past_key_values
                is not None
                and
                past_key_values.get_seq_length()
                > 0
            ):
                layer_cache = (
                    past_key_values
                    .layers[
                        module_self.layer_idx
                    ]
                )

                key_states = (
                    layer_cache.keys
                )

                value_states = (
                    layer_cache.values
                )

            else:
                raise RuntimeError(
                    "Missing cross-attention KV states."
                )

            key_states = repeat_kv(
                key_states,
                module_self.num_key_value_groups,
            )

            value_states = repeat_kv(
                value_states,
                module_self.num_key_value_groups,
            )

            attn_logits = (
                torch.matmul(
                    query_states,
                    key_states.transpose(
                        2,
                        3,
                    ),
                )
                * module_self.scaling
            )

            if attention_mask is not None:
                attn_logits = (
                    attn_logits
                    + attention_mask
                )

            attn_weights = (
                F.softmax(
                    attn_logits,
                    dim=-1,
                    dtype=torch.float32,
                )
            )

            query_indices = (
                dam.select_queries(
                    q_len,
                    attn_weights.device,
                )
            )

            eps = 1e-8

            for head in dam.head_indices:
                for q_idx in (
                    query_indices.tolist()
                ):
                    row = (
                        attn_weights[
                            :,
                            head,
                            q_idx,
                            :
                        ]
                        .clone()
                    )

                    before_masses = []

                    for token_set in (
                        dam.object_token_sets
                    ):
                        valid_tokens = [
                            t
                            for t in token_set
                            if t < row.shape[-1]
                        ]

                        if not valid_tokens:
                            before_masses.append(
                                torch.tensor(
                                    0.0,
                                    device=row.device,
                                )
                            )
                            continue

                        idx = torch.tensor(
                            valid_tokens,
                            dtype=torch.long,
                            device=row.device,
                        )

                        mass = (
                            row[
                                :,
                                idx
                            ]
                            .sum(
                                dim=-1
                            )
                            .mean()
                        )

                        before_masses.append(
                            mass
                        )

                    before_tensor = (
                        torch.stack(
                            before_masses
                        )
                    )

                    dam.last_before[
                        head
                    ] = (
                        before_tensor
                        .detach()
                        .cpu()
                    )

                    positive = (
                        before_tensor
                        > eps
                    )

                    if positive.any():
                        mean_mass = (
                            before_tensor[
                                positive
                            ]
                            .mean()
                        )

                        balance = (
                            (
                                mean_mass
                                + eps
                            )
                            /
                            (
                                before_tensor
                                + eps
                            )
                        ) ** dam.gamma

                        balance = torch.clamp(
                            balance,
                            min=0.25,
                            max=4.0,
                        )

                        multiplier_map = (
                            torch.ones(
                                row.shape[-1],
                                dtype=torch.float32,
                                device=row.device,
                            )
                        )

                        for obj_idx, token_set in enumerate(
                            dam.object_token_sets
                        ):
                            valid_tokens = [
                                t
                                for t in token_set
                                if t < row.shape[-1]
                            ]

                            if not valid_tokens:
                                continue

                            multiplier_map[
                                valid_tokens
                            ] = (
                                dam.beta
                                * balance[
                                    obj_idx
                                ]
                            )

                        row = (
                            row
                            * multiplier_map[
                                None,
                                :
                            ]
                        )

                        row = (
                            row
                            /
                            row.sum(
                                dim=-1,
                                keepdim=True,
                            )
                            .clamp_min(
                                eps
                            )
                        )

                        attn_weights[
                            :,
                            head,
                            q_idx,
                            :
                        ] = row

                    updated = (
                        attn_weights[
                            :,
                            head,
                            q_idx,
                            :
                        ]
                    )

                    after_masses = []

                    for token_set in (
                        dam.object_token_sets
                    ):
                        valid_tokens = [
                            t
                            for t in token_set
                            if t < updated.shape[-1]
                        ]

                        if not valid_tokens:
                            after_masses.append(
                                torch.tensor(
                                    0.0,
                                    device=updated.device,
                                )
                            )
                            continue

                        idx = torch.tensor(
                            valid_tokens,
                            dtype=torch.long,
                            device=updated.device,
                        )

                        mass = (
                            updated[
                                :,
                                idx
                            ]
                            .sum(
                                dim=-1
                            )
                            .mean()
                        )

                        after_masses.append(
                            mass
                        )

                    dam.last_after[
                        head
                    ] = (
                        torch.stack(
                            after_masses
                        )
                        .detach()
                        .cpu()
                    )

                    row_error = (
                        updated.sum(
                            dim=-1
                        )
                        .sub(1.0)
                        .abs()
                        .max()
                        .item()
                    )

                    dam.last_row_sum_error[
                        head
                    ] = row_error

                    dam.head_calls[
                        head
                    ] += 1

            dam.layer_calls += 1

            attn_weights = (
                attn_weights.to(
                    value_states.dtype
                )
            )

            attn_output = (
                torch.matmul(
                    attn_weights,
                    value_states,
                )
            )

            attn_output = (
                attn_output
                .transpose(
                    1,
                    2,
                )
                .contiguous()
                .reshape(
                    bsz,
                    q_len,
                    -1,
                )
            )

            attn_output = (
                module_self.o_proj(
                    attn_output
                )
            )

            return (
                attn_output,
                attn_weights,
            )

        module.forward = (
            types.MethodType(
                patched_forward,
                module,
            )
        )

    def remove(self):
        if self.original_forward is not None:
            self.module.forward = (
                self.original_forward
            )
            self.original_forward = None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-id",
        required=True,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.25,
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
            args.sample_id
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

    grouped = defaultdict(
        list
    )

    for layer, head in candidate_heads:
        grouped[
            layer
        ].append(
            head
        )

    grouped = {
        layer: sorted(
            heads
        )
        for layer, heads
        in grouped.items()
    }

    print("=" * 100)
    print(
        "AROMA Distributed DAM-v2 "
        "Smoke Test — FIXED"
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
        "Active layers:",
        len(grouped),
    )

    print(
        "Active heads :",
        len(candidate_heads),
    )

    print(
        "\nGrouped heads:"
    )

    for layer, heads in grouped.items():
        print(
            f"  Layer {layer}: "
            f"{heads}"
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
        repr(
            baseline_text
        ),
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

    layer_dams = []

    for layer, heads in grouped.items():
        layer_dams.append(
            DistributedLayerDAM(
                model=model,
                layer_idx=layer,
                head_indices=heads,
                object_token_sets=object_token_sets,
                beta=args.beta,
                gamma=args.gamma,
                query_scope=args.query_scope,
            )
        )

    for dam in layer_dams:
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
            layer_dams
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
        repr(
            dam_text
        ),
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
        "Prediction changed:",
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

    print(
        "\n" + "=" * 100
    )

    print(
        "PER-HEAD DIAGNOSTICS"
    )

    print("=" * 100)

    all_calls_ok = True

    all_row_errors = []

    cv_before = []
    cv_after = []

    total_active_heads = 0

    for dam in layer_dams:
        for head in dam.head_indices:
            total_active_heads += 1

            calls = (
                dam.head_calls[
                    head
                ]
            )

            before = (
                dam.last_before[
                    head
                ]
            )

            after = (
                dam.last_after[
                    head
                ]
            )

            err = (
                dam.last_row_sum_error[
                    head
                ]
            )

            before_cv = (
                tensor_cv(
                    before
                )
            )

            after_cv = (
                tensor_cv(
                    after
                )
            )

            print(
                f"\nL{dam.layer_idx}H{head}"
            )

            print(
                "  calls:",
                calls,
            )

            print(
                "  row-sum error:",
                err,
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

            if calls <= 0:
                all_calls_ok = False

            if err is not None:
                all_row_errors.append(
                    float(err)
                )

            if before_cv is not None:
                cv_before.append(
                    before_cv
                )

            if after_cv is not None:
                cv_after.append(
                    after_cv
                )

    print(
        "\n" + "=" * 100
    )

    print(
        "DISTRIBUTED SANITY SUMMARY"
    )

    print("=" * 100)

    print(
        "Expected active heads:",
        len(candidate_heads),
    )

    print(
        "Observed active heads:",
        total_active_heads,
    )

    print(
        "All frozen heads intervened:",
        all_calls_ok,
    )

    if all_row_errors:
        print(
            "Max row-sum error:",
            max(
                all_row_errors
            ),
        )

        print(
            "Mean row-sum error:",
            statistics.mean(
                all_row_errors
            ),
        )

    if cv_before:
        print(
            "Mean object CV before:",
            statistics.mean(
                cv_before
            ),
        )

    if cv_after:
        print(
            "Mean object CV after :",
            statistics.mean(
                cv_after
            ),
        )

    if not all_calls_ok:
        raise RuntimeError(
            "At least one frozen head "
            "received zero DAM calls."
        )

    if (
        all_row_errors
        and
        max(
            all_row_errors
        )
        > 5e-3
    ):
        raise RuntimeError(
            "Attention normalization "
            "sanity check failed."
        )

    if total_active_heads != 7:
        raise RuntimeError(
            f"Expected 7 active heads, "
            f"got {total_active_heads}."
        )

    print(
        "\nDISTRIBUTED DAM-v2 "
        "SMOKE TEST PASS"
    )


if __name__ == "__main__":
    main()
