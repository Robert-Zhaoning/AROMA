import inspect
import json

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

IMAGE_PATHS = [
    "data/calibration50/images/cal50_n01_row.png",
    "data/calibration50/images/cal50_n05_row.png",
    "data/calibration50/images/cal50_n10_row.png",
]

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


def describe_value(name, value):
    print(f"\n{name}:")

    if torch.is_tensor(value):
        print("  shape :", tuple(value.shape))
        print("  dtype :", value.dtype)
        print("  value :", value.detach().cpu().tolist())
    else:
        print("  type  :", type(value))
        print("  value :", value)


def main():
    print("=" * 80)
    print("AROMA — Llama Attention Geometry Audit")
    print("=" * 80)

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    image_processor = processor.image_processor

    print("\n" + "=" * 80)
    print("IMAGE PROCESSOR")
    print("=" * 80)

    print("Class:")
    print(type(image_processor))

    print("\nConfig / repr:")
    print(image_processor)

    interesting_attrs = [
        "size",
        "max_image_tiles",
        "resample",
        "image_mean",
        "image_std",
        "do_resize",
        "do_rescale",
        "do_normalize",
        "do_pad",
    ]

    print("\nSelected attributes:")

    for attr in interesting_attrs:
        if hasattr(image_processor, attr):
            print(
                f"{attr:20s} = "
                f"{getattr(image_processor, attr)}"
            )

    print("\n" + "=" * 80)
    print("PROCESSOR SOURCE HINTS")
    print("=" * 80)

    source_targets = [
        "preprocess",
        "_preprocess",
    ]

    for method_name in source_targets:
        if hasattr(image_processor, method_name):
            method = getattr(
                image_processor,
                method_name,
            )

            print(
                f"\n--- {method_name} source ---"
            )

            try:
                src = inspect.getsource(method)

                lines = src.splitlines()

                for line in lines[:120]:
                    print(line)

            except Exception as e:
                print(
                    "Could not inspect source:",
                    repr(e),
                )

    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    model.eval()

    cross_layers = []

    for layer_idx, layer in enumerate(
        model.model.language_model.layers
    ):
        if hasattr(layer, "cross_attn"):
            cross_layers.append(layer_idx)

    print(
        "\nDetected cross-attention layers:",
        cross_layers,
    )

    print(
        "Number of cross-attention layers:",
        len(cross_layers),
    )

    for image_path in IMAGE_PATHS:
        print("\n" + "=" * 80)
        print("IMAGE:", image_path)
        print("=" * 80)

        image = Image.open(
            image_path
        ).convert("RGB")

        print(
            "Original image size:",
            image.size,
        )

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

        formatted = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = processor(
            images=image,
            text=formatted,
            add_special_tokens=False,
            return_tensors="pt",
        )

        print("\nProcessor outputs:")

        for key, value in inputs.items():
            if torch.is_tensor(value):
                print(
                    f"{key:24s} "
                    f"shape={tuple(value.shape)} "
                    f"dtype={value.dtype}"
                )

                if key in {
                    "aspect_ratio_ids",
                    "aspect_ratio_mask",
                    "cross_attention_mask",
                }:
                    print(
                        "  values:",
                        value.cpu().tolist(),
                    )

        pixel_values = inputs["pixel_values"]

        print("\nPixel-value interpretation:")

        print(
            "batch        =",
            pixel_values.shape[0],
        )

        print(
            "media/images =",
            pixel_values.shape[1],
        )

        print(
            "tiles        =",
            pixel_values.shape[2],
        )

        print(
            "channels     =",
            pixel_values.shape[3],
        )

        print(
            "tile height  =",
            pixel_values.shape[4],
        )

        print(
            "tile width   =",
            pixel_values.shape[5],
        )

        device_inputs = {
            key: (
                value.to(model.device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(
                **device_inputs,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

        print("\nAttention tensors:")

        for layer_idx, attn in enumerate(
            outputs.attentions
        ):
            if attn is None:
                print(
                    f"Layer {layer_idx:02d}: None"
                )
                continue

            shape = tuple(attn.shape)

            attention_type = (
                "CROSS"
                if shape[-1]
                != shape[-2]
                else "SELF"
            )

            print(
                f"Layer {layer_idx:02d}: "
                f"{attention_type:5s} "
                f"{shape}"
            )

        print("\nCross-attention decomposition:")

        for layer_idx in cross_layers:
            attn = outputs.attentions[
                layer_idx
            ]

            if attn is None:
                print(
                    f"Layer {layer_idx}: None"
                )
                continue

            visual_tokens = attn.shape[-1]
            num_tiles = pixel_values.shape[2]

            print(
                f"\nLayer {layer_idx}:"
            )

            print(
                "  visual tokens =",
                visual_tokens,
            )

            print(
                "  num tiles     =",
                num_tiles,
            )

            if (
                visual_tokens
                % num_tiles
                == 0
            ):
                per_tile = (
                    visual_tokens
                    // num_tiles
                )

                print(
                    "  tokens/tile   =",
                    per_tile,
                )

                spatial_candidate = (
                    per_tile - 1
                )

                side = int(
                    spatial_candidate ** 0.5
                )

                print(
                    "  minus special =",
                    spatial_candidate,
                )

                print(
                    "  sqrt(candidate)=",
                    side,
                )

                if (
                    side * side
                    == spatial_candidate
                ):
                    print(
                        "  -> consistent with "
                        f"{side}x{side} patches "
                        "+ 1 special token/tile"
                    )
                else:
                    print(
                        "  -> NOT a simple square "
                        "patch grid + 1 token"
                    )

        del outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
