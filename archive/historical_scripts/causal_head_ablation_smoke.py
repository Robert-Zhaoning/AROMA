import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

METADATA_PATH = Path(
    "data/calibration50/metadata.jsonl"
)

CROSS_LAYERS = [
    3,
    8,
    13,
    18,
    23,
    28,
    33,
    38,
]

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


def load_sample(sample_id):
    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record["sample_id"] == sample_id:
                return record

    raise ValueError(
        f"Unknown sample_id: {sample_id}"
    )


def extract_integer(text):
    match = re.search(
        r"(?<![\w.])-?\d+(?![\w.])",
        text,
    )

    if match is None:
        return None

    return int(
        match.group()
    )


def prepare_inputs(
    processor,
    image,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": PROMPT,
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

    inputs = processor(
        images=image,
        text=formatted,
        add_special_tokens=False,
        return_tensors="pt",
    )

    return (
        formatted,
        inputs,
    )


def move_to_model_device(
    inputs,
    model,
):
    result = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[key] = value.to(
                model.device
            )
        else:
            result[key] = value

    return result


def generate_answer(
    model,
    processor,
    device_inputs,
):
    input_length = (
        device_inputs[
            "input_ids"
        ].shape[-1]
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **device_inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
        )

    generated_ids = (
        output_ids[
            :,
            input_length:
        ]
    )

    text = (
        processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .strip()
    )

    return (
        text,
        extract_integer(text),
    )


def get_gt_token_info(
    processor,
    gt,
):
    """
    For Calibration-50 counts 1..10 we expect the numeral
    normally to tokenize as one token, but verify rather than assume.
    """

    candidate_forms = [
        str(gt),
        " " + str(gt),
    ]

    results = []

    for form in candidate_forms:
        ids = processor.tokenizer.encode(
            form,
            add_special_tokens=False,
        )

        results.append(
            {
                "form": form,
                "ids": ids,
            }
        )

    return results


def score_first_token(
    model,
    device_inputs,
    token_id,
):
    """
    Probability/logit of a candidate first assistant token at
    the current generation boundary.

    This is a diagnostic score, not yet a final paper metric.
    """

    with torch.inference_mode():
        outputs = model(
            **device_inputs,
            use_cache=False,
            return_dict=True,
        )

    logits = (
        outputs.logits[
            0,
            -1,
            :
        ]
        .float()
    )

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    prob = float(
        torch.exp(
            log_probs[token_id]
        ).item()
    )

    log_prob = float(
        log_probs[token_id].item()
    )

    logit = float(
        logits[token_id].item()
    )

    top_values, top_indices = (
        torch.topk(
            log_probs,
            k=10,
        )
    )

    top10 = [
        (
            int(idx.item()),
            float(lp.item()),
        )
        for idx, lp
        in zip(
            top_indices,
            top_values,
        )
    ]

    return {
        "prob": prob,
        "log_prob": log_prob,
        "logit": logit,
        "top10": top10,
    }


class HeadAblator:
    """
    Zero one attention head at the input of cross_attn.o_proj.

    In standard multi-head attention, the per-head context vectors
    are concatenated before the output projection:

        [head_0 | head_1 | ... | head_H-1] -> o_proj

    Zeroing the selected slice therefore removes that head's
    contribution before the shared output projection.
    """

    def __init__(
        self,
        model,
        layer_idx,
        head_idx,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx

        self.handle = None
        self.calls = []

        layer = (
            model.model.language_model.layers[
                layer_idx
            ]
        )

        if not hasattr(
            layer,
            "cross_attn",
        ):
            raise ValueError(
                f"Layer {layer_idx} has no cross_attn"
            )

        self.cross_attn = (
            layer.cross_attn
        )

        self.num_heads = getattr(
            self.cross_attn,
            "num_heads",
            None,
        )

        if self.num_heads is None:
            self.num_heads = getattr(
                model.config.text_config,
                "num_attention_heads",
                None,
            )

        if self.num_heads is None:
            raise RuntimeError(
                "Could not determine number of attention heads."
            )

        if (
            head_idx < 0
            or head_idx >= self.num_heads
        ):
            raise ValueError(
                f"Invalid head {head_idx}; "
                f"num_heads={self.num_heads}"
            )

        self.o_proj = (
            self.cross_attn.o_proj
        )

        self.hidden_size = (
            self.o_proj.in_features
        )

        if (
            self.hidden_size
            % self.num_heads
            != 0
        ):
            raise RuntimeError(
                "o_proj input size is not divisible "
                "by num_heads."
            )

        self.head_dim = (
            self.hidden_size
            // self.num_heads
        )

        self.start = (
            self.head_idx
            * self.head_dim
        )

        self.end = (
            self.start
            + self.head_dim
        )

    def _hook(
        self,
        module,
        inputs,
    ):
        if not inputs:
            return inputs

        x = inputs[0]

        if not torch.is_tensor(x):
            return inputs

        if (
            x.shape[-1]
            != self.hidden_size
        ):
            raise RuntimeError(
                "Unexpected o_proj input shape: "
                f"{tuple(x.shape)}; "
                f"expected last dim "
                f"{self.hidden_size}"
            )

        # Clone so we do not mutate a tensor that another
        # part of the graph may still reference.
        x_mod = x.clone()

        target_before = (
            x_mod[
                ...,
                self.start:self.end
            ]
        )

        target_norm_before = float(
            target_before
            .float()
            .norm()
            .item()
        )

        full_norm_before = float(
            x_mod
            .float()
            .norm()
            .item()
        )

        x_mod[
            ...,
            self.start:self.end
        ] = 0

        target_norm_after = float(
            x_mod[
                ...,
                self.start:self.end
            ]
            .float()
            .norm()
            .item()
        )

        self.calls.append(
            {
                "shape":
                    tuple(x.shape),

                "target_norm_before":
                    target_norm_before,

                "target_norm_after":
                    target_norm_after,

                "full_norm_before":
                    full_norm_before,
            }
        )

        if len(inputs) == 1:
            return (
                x_mod,
            )

        return (
            x_mod,
            *inputs[1:],
        )

    def register(self):
        self.calls = []

        self.handle = (
            self.o_proj
            .register_forward_pre_hook(
                self._hook
            )
        )

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def print_top_tokens(
    processor,
    result,
    title,
):
    print(
        "\n" + title
    )

    for rank, (
        token_id,
        log_prob,
    ) in enumerate(
        result["top10"],
        start=1,
    ):
        token = (
            processor.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

        print(
            f"  {rank:2d}. "
            f"id={token_id:6d} "
            f"logp={log_prob:9.4f} "
            f"token={repr(token)}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-id",
        type=str,
        default="cal50_n05_row",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=33,
    )

    parser.add_argument(
        "--head",
        type=int,
        default=21,
    )

    args = parser.parse_args()

    if args.layer not in CROSS_LAYERS:
        raise ValueError(
            f"Layer {args.layer} is not one of "
            f"verified cross-attention layers "
            f"{CROSS_LAYERS}"
        )

    sample = load_sample(
        args.sample_id
    )

    gt = int(
        sample["ground_truth"]
    )

    print("=" * 80)
    print("AROMA Causal Head Ablation — Smoke Test")
    print("=" * 80)

    print(
        "Sample     :",
        sample["sample_id"],
    )

    print(
        "GT         :",
        gt,
    )

    print(
        "Condition  :",
        sample["condition"],
    )

    print(
        "Ablation   :",
        f"Layer {args.layer}, Head {args.head}",
    )

    print(
        "Image      :",
        sample["image_path"],
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

    _, inputs = prepare_inputs(
        processor,
        image,
    )

    device_inputs = (
        move_to_model_device(
            inputs,
            model,
        )
    )

    # ---------------------------------------------------------
    # Determine GT tokenization
    # ---------------------------------------------------------

    gt_token_forms = (
        get_gt_token_info(
            processor,
            gt,
        )
    )

    print(
        "\nGT numeral tokenization:"
    )

    for item in gt_token_forms:
        decoded = [
            processor.tokenizer.decode(
                [tid],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for tid in item["ids"]
        ]

        print(
            f"  form={repr(item['form'])} "
            f"ids={item['ids']} "
            f"decoded={decoded}"
        )

    single_token_candidates = [
        item
        for item in gt_token_forms
        if len(item["ids"]) == 1
    ]

    if not single_token_candidates:
        raise RuntimeError(
            "GT numeral is not single-token under tested forms. "
            "Use sequence log-prob scoring instead."
        )

    # Prefer bare numeral if it is single-token.
    gt_token_id = (
        single_token_candidates[0][
            "ids"
        ][0]
    )

    print(
        "\nSelected GT token ID:",
        gt_token_id,
    )

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "BASELINE"
    )

    print(
        "-" * 80
    )

    baseline_text, baseline_pred = (
        generate_answer(
            model,
            processor,
            device_inputs,
        )
    )

    baseline_score = (
        score_first_token(
            model,
            device_inputs,
            gt_token_id,
        )
    )

    print(
        "Generated text :",
        repr(
            baseline_text
        ),
    )

    print(
        "Parsed count   :",
        baseline_pred,
    )

    print(
        "Correct        :",
        baseline_pred == gt,
    )

    print(
        "GT token prob  :",
        f"{baseline_score['prob']:.8f}",
    )

    print(
        "GT token logp  :",
        f"{baseline_score['log_prob']:.6f}",
    )

    print_top_tokens(
        processor,
        baseline_score,
        "Baseline top-10 first-token distribution:",
    )

    # ---------------------------------------------------------
    # Register head ablation
    # ---------------------------------------------------------

    ablator = HeadAblator(
        model=model,
        layer_idx=args.layer,
        head_idx=args.head,
    )

    print(
        "\nHead geometry:"
    )

    print(
        "  num_heads   =",
        ablator.num_heads,
    )

    print(
        "  hidden_size =",
        ablator.hidden_size,
    )

    print(
        "  head_dim    =",
        ablator.head_dim,
    )

    print(
        "  slice       =",
        f"[{ablator.start}:{ablator.end}]",
    )

    ablator.register()

    try:

        print(
            "\n" + "-" * 80
        )

        print(
            "ABLATION"
        )

        print(
            "-" * 80
        )

        ablated_text, ablated_pred = (
            generate_answer(
                model,
                processor,
                device_inputs,
            )
        )

        ablated_score = (
            score_first_token(
                model,
                device_inputs,
                gt_token_id,
            )
        )

    finally:
        ablator.remove()

    print(
        "Generated text :",
        repr(
            ablated_text
        ),
    )

    print(
        "Parsed count   :",
        ablated_pred,
    )

    print(
        "Correct        :",
        ablated_pred == gt,
    )

    print(
        "GT token prob  :",
        f"{ablated_score['prob']:.8f}",
    )

    print(
        "GT token logp  :",
        f"{ablated_score['log_prob']:.6f}",
    )

    print_top_tokens(
        processor,
        ablated_score,
        "Ablated top-10 first-token distribution:",
    )

    # ---------------------------------------------------------
    # Hook sanity check
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        "HOOK SANITY CHECK"
    )

    print(
        "-" * 80
    )

    print(
        "Hook calls:",
        len(ablator.calls),
    )

    if not ablator.calls:
        raise RuntimeError(
            "Ablation hook was never called."
        )

    nonzero_before = 0
    zero_after = 0

    for i, call in enumerate(
        ablator.calls[:10]
    ):
        print(
            f"  call {i:02d}: "
            f"shape={call['shape']} "
            f"target_norm_before="
            f"{call['target_norm_before']:.6f} "
            f"target_norm_after="
            f"{call['target_norm_after']:.6f}"
        )

        if (
            call[
                "target_norm_before"
            ] > 0
        ):
            nonzero_before += 1

        if (
            call[
                "target_norm_after"
            ] == 0
        ):
            zero_after += 1

    all_zero_after = all(
        call[
            "target_norm_after"
        ] == 0
        for call in ablator.calls
    )

    any_nonzero_before = any(
        call[
            "target_norm_before"
        ] > 0
        for call in ablator.calls
    )

    print(
        "\nAny non-zero target slice before ablation:",
        any_nonzero_before,
    )

    print(
        "All target slices zero after ablation:",
        all_zero_after,
    )

    if not any_nonzero_before:
        raise RuntimeError(
            "Target head slice was already zero before ablation."
        )

    if not all_zero_after:
        raise RuntimeError(
            "Target head slice was not completely zeroed."
        )

    # ---------------------------------------------------------
    # Effect summary
    # ---------------------------------------------------------

    delta_logp = (
        ablated_score[
            "log_prob"
        ]
        - baseline_score[
            "log_prob"
        ]
    )

    delta_prob = (
        ablated_score[
            "prob"
        ]
        - baseline_score[
            "prob"
        ]
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "CAUSAL EFFECT SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "Baseline prediction :",
        baseline_pred,
    )

    print(
        "Ablated prediction  :",
        ablated_pred,
    )

    print(
        "Prediction changed  :",
        baseline_pred
        != ablated_pred,
    )

    print(
        "Baseline GT prob    :",
        f"{baseline_score['prob']:.8f}",
    )

    print(
        "Ablated GT prob     :",
        f"{ablated_score['prob']:.8f}",
    )

    print(
        "Delta GT prob       :",
        f"{delta_prob:+.8f}",
    )

    print(
        "Baseline GT logp    :",
        f"{baseline_score['log_prob']:.6f}",
    )

    print(
        "Ablated GT logp     :",
        f"{ablated_score['log_prob']:.6f}",
    )

    print(
        "Delta GT logp       :",
        f"{delta_logp:+.6f}",
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a causal feasibility probe only."
    )

    print(
        "A single changed/unchanged answer does NOT establish "
        "that this is a counting head."
    )

    print(
        "Next step is candidate-vs-random-control intervention "
        "across matched samples."
    )

    print(
        "\nSMOKE TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
