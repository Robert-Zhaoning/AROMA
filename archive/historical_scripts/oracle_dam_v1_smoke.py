import argparse
import json
import math
import re
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

from transformers.models.mllama.modeling_mllama import (
    repeat_kv,
)


MODEL_ID = (
    "meta-llama/"
    "Llama-3.2-11B-Vision-Instruct"
)

META_PATH = Path(
    "data/proc_count_causal_v1/metadata.jsonl"
)

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


# ============================================================
# Vision geometry
# ============================================================

TILE_SIZE = 560
PATCH_SIZE = 14

PATCHES_PER_SIDE = (
    TILE_SIZE // PATCH_SIZE
)

PATCHES_PER_TILE = (
    PATCHES_PER_SIDE
    * PATCHES_PER_SIDE
)

TOKENS_PER_TILE = (
    PATCHES_PER_TILE + 1
)

MAX_TILES = 4


def extract_integer(text):

    match = re.search(
        r"(?<![\w.])\d+(?![\w.])",
        text,
    )

    if match is None:
        return None

    return int(
        match.group()
    )


def load_metadata():

    records = {}

    with META_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(
                line
            )

            records[
                record["sample_id"]
            ] = record

    return records


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


def move_inputs(
    inputs,
    model,
):

    return {
        key: (
            value.to(model.device)
            if torch.is_tensor(value)
            else value
        )
        for key, value
        in inputs.items()
    }


# ============================================================
# Original-image -> tiled-canvas patch mapping
# ============================================================

def get_active_tile_geometry(
    processor,
    image,
):

    image_processor = (
        processor.image_processor
    )

    width, height = image.size

    # For the current PCC-v1 images this is expected to
    # resolve to a 1x2 active tiled canvas.
    #
    # We intentionally derive it from processor output rather
    # than hard-code active tile count.

    probe = image_processor(
        images=image,
        return_tensors="pt",
    )

    aspect_ratio_mask = (
        probe[
            "aspect_ratio_mask"
        ][0, 0]
    )

    active_tiles = [
        int(i)
        for i, value
        in enumerate(
            aspect_ratio_mask.tolist()
        )
        if int(value) == 1
    ]

    n_active = len(
        active_tiles
    )

    if n_active == 1:

        rows = 1
        cols = 1

    elif n_active == 2:

        # PCC-v1 = landscape 768x512,
        # processor audit established 1x2.
        rows = 1
        cols = 2

    elif n_active == 4:

        rows = 2
        cols = 2

    else:

        raise RuntimeError(
            f"Unsupported active tile "
            f"count: {n_active}"
        )

    canvas_width = (
        cols * TILE_SIZE
    )

    canvas_height = (
        rows * TILE_SIZE
    )

    # Our previous processor audit established:
    #
    # input 768x512 -> resized 768x512
    # then pad to 1120x560.
    #
    # Keep general scale computation here.

    scale = min(
        canvas_width / width,
        canvas_height / height,
        1.0,
    )

    resized_width = int(
        round(
            width * scale
        )
    )

    resized_height = int(
        round(
            height * scale
        )
    )

    return {
        "active_tiles":
            active_tiles,

        "rows":
            rows,

        "cols":
            cols,

        "canvas_width":
            canvas_width,

        "canvas_height":
            canvas_height,

        "resized_width":
            resized_width,

        "resized_height":
            resized_height,

        "scale_x":
            resized_width / width,

        "scale_y":
            resized_height / height,
    }


def bbox_to_visual_tokens(
    bbox_px,
    geometry,
):

    x1, y1, x2, y2 = [
        float(x)
        for x in bbox_px
    ]

    sx = geometry[
        "scale_x"
    ]

    sy = geometry[
        "scale_y"
    ]

    x1 *= sx
    x2 *= sx
    y1 *= sy
    y2 *= sy

    # Patch inclusion criterion:
    # patch center lies inside GT bbox.
    #
    # If this selects none, fallback to
    # patch/bbox intersection.

    selected = []

    for tile_row in range(
        geometry["rows"]
    ):

        for tile_col in range(
            geometry["cols"]
        ):

            tile_idx = (
                tile_row
                * geometry["cols"]
                + tile_col
            )

            tile_origin_x = (
                tile_col
                * TILE_SIZE
            )

            tile_origin_y = (
                tile_row
                * TILE_SIZE
            )

            for patch_row in range(
                PATCHES_PER_SIDE
            ):

                for patch_col in range(
                    PATCHES_PER_SIDE
                ):

                    px1 = (
                        tile_origin_x
                        + patch_col
                        * PATCH_SIZE
                    )

                    py1 = (
                        tile_origin_y
                        + patch_row
                        * PATCH_SIZE
                    )

                    px2 = (
                        px1
                        + PATCH_SIZE
                    )

                    py2 = (
                        py1
                        + PATCH_SIZE
                    )

                    cx = (
                        px1 + px2
                    ) / 2

                    cy = (
                        py1 + py2
                    ) / 2

                    if (
                        x1 <= cx <= x2
                        and
                        y1 <= cy <= y2
                    ):

                        token_index = (
                            tile_idx
                            * TOKENS_PER_TILE
                            + 1
                            + patch_row
                            * PATCHES_PER_SIDE
                            + patch_col
                        )

                        selected.append(
                            token_index
                        )

    if selected:
        return sorted(
            set(selected)
        )

    # Fallback: patch intersection.

    for tile_row in range(
        geometry["rows"]
    ):

        for tile_col in range(
            geometry["cols"]
        ):

            tile_idx = (
                tile_row
                * geometry["cols"]
                + tile_col
            )

            tile_origin_x = (
                tile_col
                * TILE_SIZE
            )

            tile_origin_y = (
                tile_row
                * TILE_SIZE
            )

            for patch_row in range(
                PATCHES_PER_SIDE
            ):

                for patch_col in range(
                    PATCHES_PER_SIDE
                ):

                    px1 = (
                        tile_origin_x
                        + patch_col
                        * PATCH_SIZE
                    )

                    py1 = (
                        tile_origin_y
                        + patch_row
                        * PATCH_SIZE
                    )

                    px2 = (
                        px1
                        + PATCH_SIZE
                    )

                    py2 = (
                        py1
                        + PATCH_SIZE
                    )

                    overlap = not (
                        px2 <= x1
                        or x2 <= px1
                        or py2 <= y1
                        or y2 <= py1
                    )

                    if overlap:

                        token_index = (
                            tile_idx
                            * TOKENS_PER_TILE
                            + 1
                            + patch_row
                            * PATCHES_PER_SIDE
                            + patch_col
                        )

                        selected.append(
                            token_index
                        )

    return sorted(
        set(selected)
    )


def build_object_token_sets(
    sample,
    processor,
    image,
):

    geometry = (
        get_active_tile_geometry(
            processor,
            image,
        )
    )

    object_sets = []

    for obj in sample[
        "target_objects"
    ]:

        tokens = (
            bbox_to_visual_tokens(
                obj["bbox_px"],
                geometry,
            )
        )

        if not tokens:

            raise RuntimeError(
                "Object mapped to zero "
                "visual patches."
            )

        object_sets.append(
            tokens
        )

    return (
        geometry,
        object_sets,
    )


# ============================================================
# Numeral scoring
# ============================================================

def numeral_token_ids(
    processor,
):

    result = {}

    for n in range(
        1,
        11,
    ):

        ids = (
            processor.tokenizer.encode(
                str(n),
                add_special_tokens=False,
            )
        )

        if len(ids) != 1:

            raise RuntimeError(
                f"{n} not single token: {ids}"
            )

        result[n] = ids[0]

    return result


def score_state(
    model,
    inputs,
    numeral_ids,
    gt,
):

    with torch.inference_mode():

        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

    logits = (
        outputs.logits[
            0,
            -1
        ]
        .float()
    )

    log_probs = (
        torch.log_softmax(
            logits,
            dim=-1,
        )
    )

    gt_logp = float(
        log_probs[
            numeral_ids[gt]
        ].item()
    )

    numeral_logits = {
        n: float(
            logits[
                token_id
            ].item()
        )
        for n, token_id
        in numeral_ids.items()
    }

    best = max(
        numeral_logits,
        key=numeral_logits.get,
    )

    best_wrong = max(
        (
            n
            for n in numeral_logits
            if n != gt
        ),
        key=lambda n:
            numeral_logits[n],
    )

    margin = (
        numeral_logits[gt]
        - numeral_logits[
            best_wrong
        ]
    )

    return {
        "gt_logp":
            gt_logp,

        "margin":
            margin,

        "best_numeral":
            int(best),
    }


def generate_answer(
    model,
    processor,
    inputs,
):

    prompt_len = (
        inputs[
            "input_ids"
        ].shape[-1]
    )

    with torch.inference_mode():

        generated = (
            model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
            )
        )

    new_ids = (
        generated[
            :,
            prompt_len:
        ]
    )

    text = (
        processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .strip()
    )

    return (
        text,
        extract_integer(text),
    )


# ============================================================
# DAM
# ============================================================

class OracleDAM:

    def __init__(
        self,
        model,
        layer_idx,
        head_idx,
        object_token_sets,
        gamma,
        query_scope,
    ):

        self.model = model

        self.layer_idx = int(
            layer_idx
        )

        self.head_idx = int(
            head_idx
        )

        self.object_token_sets = (
            object_token_sets
        )

        self.gamma = float(
            gamma
        )

        self.query_scope = (
            query_scope
        )

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

        self.calls = 0

        self.last_before_masses = None
        self.last_after_masses = None
        self.last_row_sum_error = None

    def select_queries(
        self,
        q_len,
        device,
    ):

        if self.query_scope == "answer":

            return torch.tensor(
                [
                    q_len - 1
                ],
                device=device,
                dtype=torch.long,
            )

        if self.query_scope == "q_many":

            # From our audited prompt tokenization:
            # token 12 = " many".
            if q_len > 12:

                return torch.tensor(
                    [12],
                    device=device,
                    dtype=torch.long,
                )

            # During cached generation q_len=1:
            # operate on the current answer token.
            return torch.tensor(
                [0],
                device=device,
                dtype=torch.long,
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

            if valid:

                return torch.tensor(
                    valid,
                    device=device,
                    dtype=torch.long,
                )

            return torch.tensor(
                [q_len - 1],
                device=device,
                dtype=torch.long,
            )

        raise ValueError(
            f"Unknown query scope: "
            f"{self.query_scope}"
        )

    def register(self):

        module = self.module

        self.original_forward = (
            module.forward
        )

        dam = self

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

                (
                    key_states,
                    value_states,
                ) = (
                    past_key_values
                    .layers[
                        module_self.layer_idx
                    ]
                    .keys,
                    past_key_values
                    .layers[
                        module_self.layer_idx
                    ]
                    .values,
                )

            else:

                raise ValueError(
                    "Cross-attention has no "
                    "KV states."
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
                .to(
                    query_states.dtype
                )
            )

            # -----------------------------------------------
            # Oracle object-aware redistribution
            # -----------------------------------------------

            head = dam.head_idx

            query_indices = (
                dam.select_queries(
                    q_len,
                    attn_weights.device,
                )
            )

            before_records = []
            after_records = []

            eps = 1e-8

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
                    .float()
                )

                object_masses = []

                for token_set in (
                    dam.object_token_sets
                ):

                    valid_tokens = [
                        t
                        for t in token_set
                        if t < row.shape[-1]
                    ]

                    if not valid_tokens:

                        object_masses.append(
                            torch.tensor(
                                0.0,
                                device=row.device,
                            )
                        )

                        continue

                    token_tensor = (
                        torch.tensor(
                            valid_tokens,
                            device=row.device,
                            dtype=torch.long,
                        )
                    )

                    mass = (
                        row[
                            :,
                            token_tensor
                        ]
                        .sum(
                            dim=-1
                        )
                        .mean()
                    )

                    object_masses.append(
                        mass
                    )

                mass_tensor = (
                    torch.stack(
                        object_masses
                    )
                )

                before_records.append(
                    mass_tensor.detach()
                    .cpu()
                )

                positive = (
                    mass_tensor
                    > eps
                )

                if positive.any():

                    mean_mass = (
                        mass_tensor[
                            positive
                        ]
                        .mean()
                    )

                    multipliers = (
                        (
                            mean_mass
                            + eps
                        )
                        /
                        (
                            mass_tensor
                            + eps
                        )
                    ) ** dam.gamma

                    multipliers = torch.clamp(
                        multipliers,
                        min=0.25,
                        max=4.0,
                    )

                    multiplier_map = (
                        torch.ones(
                            row.shape[-1],
                            device=row.device,
                            dtype=torch.float32,
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
                            multipliers[
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

                    row_sum = (
                        row.sum(
                            dim=-1,
                            keepdim=True,
                        )
                    )

                    row = (
                        row
                        / row_sum.clamp_min(
                            eps
                        )
                    )

                    attn_weights[
                        :,
                        head,
                        q_idx,
                        :
                    ] = row.to(
                        attn_weights.dtype
                    )

                updated_row = (
                    attn_weights[
                        :,
                        head,
                        q_idx,
                        :
                    ]
                    .float()
                )

                new_masses = []

                for token_set in (
                    dam.object_token_sets
                ):

                    valid_tokens = [
                        t
                        for t in token_set
                        if t
                        < updated_row.shape[-1]
                    ]

                    if not valid_tokens:

                        new_masses.append(
                            torch.tensor(
                                0.0
                            )
                        )

                        continue

                    token_tensor = (
                        torch.tensor(
                            valid_tokens,
                            device=updated_row.device,
                            dtype=torch.long,
                        )
                    )

                    mass = (
                        updated_row[
                            :,
                            token_tensor
                        ]
                        .sum(
                            dim=-1
                        )
                        .mean()
                    )

                    new_masses.append(
                        mass.detach().cpu()
                    )

                after_records.append(
                    torch.stack(
                        new_masses
                    )
                )

                row_error = (
                    updated_row
                    .sum(
                        dim=-1
                    )
                    .sub(1.0)
                    .abs()
                    .max()
                    .item()
                )

                dam.last_row_sum_error = (
                    row_error
                )

            if before_records:

                dam.last_before_masses = (
                    before_records[-1]
                )

                dam.last_after_masses = (
                    after_records[-1]
                )

            dam.calls += 1

            attn_output = torch.matmul(
                attn_weights,
                value_states,
            )

            attn_output = (
                attn_output
                .transpose(
                    1,
                    2,
                )
                .contiguous()
            )

            attn_output = (
                attn_output.reshape(
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

        if (
            self.original_forward
            is not None
        ):

            self.module.forward = (
                self.original_forward
            )

            self.original_forward = None


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-id",
        required=True,
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--head",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
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

    if (
        args.sample_id
        not in metadata
    ):

        raise KeyError(
            args.sample_id
        )

    sample = metadata[
        args.sample_id
    ]

    gt = int(
        sample[
            "ground_truth"
        ]
    )

    print("=" * 100)
    print(
        "AROMA Oracle DAM-v1 "
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
        sample[
            "condition"
        ],
    )

    print(
        "Layer/head  :",
        f"L{args.layer}H{args.head}",
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
            sample[
                "image_path"
            ]
        )
        .convert("RGB")
    )

    (
        geometry,
        object_token_sets,
    ) = (
        build_object_token_sets(
            sample,
            processor,
            image,
        )
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

    for i, tokens in enumerate(
        object_token_sets
    ):

        print(
            f"  object {i}: "
            f"{len(tokens)} patches "
            f"[{tokens[0]} ... "
            f"{tokens[-1]}]"
        )

    formatted, inputs = (
        prepare_inputs(
            processor,
            image,
        )
    )

    inputs = move_inputs(
        inputs,
        model,
    )

    ids = numeral_token_ids(
        processor
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_score = score_state(
        model,
        inputs,
        ids,
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

    # --------------------------------------------------------
    # DAM
    # --------------------------------------------------------

    dam = OracleDAM(
        model=model,
        layer_idx=args.layer,
        head_idx=args.head,
        object_token_sets=object_token_sets,
        gamma=args.gamma,
        query_scope=args.query_scope,
    )

    dam.register()

    try:

        dam_score = score_state(
            model,
            inputs,
            ids,
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

        dam.remove()

    print(
        "\n" + "=" * 100
    )

    print("ORACLE DAM")
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

    print(
        "\nDelta GT logp  :",
        (
            dam_score[
                "gt_logp"
            ]
            -
            baseline_score[
                "gt_logp"
            ]
        ),
    )

    print(
        "Delta GT margin:",
        (
            dam_score[
                "margin"
            ]
            -
            baseline_score[
                "margin"
            ]
        ),
    )

    print(
        "\nDAM calls:",
        dam.calls,
    )

    print(
        "Max row-sum error:",
        dam.last_row_sum_error,
    )

    print(
        "\nObject masses before:"
    )

    print(
        dam.last_before_masses
    )

    print(
        "\nObject masses after:"
    )

    print(
        dam.last_after_masses
    )

    if (
        dam.last_row_sum_error
        is None
        or
        dam.last_row_sum_error
        > 5e-3
    ):

        raise RuntimeError(
            "Attention normalization "
            "sanity check failed."
        )

    print(
        "\n" + "=" * 100
    )

    print(
        "ORACLE DAM-v1 "
        "SMOKE TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
