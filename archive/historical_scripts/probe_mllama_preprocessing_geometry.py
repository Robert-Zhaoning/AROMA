import inspect

import torch
from PIL import Image
from transformers import AutoProcessor


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

IMAGE_PATH = (
    "data/calibration50/images/cal50_n05_row.png"
)


def print_source(obj, name):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    try:
        source = inspect.getsource(obj)
        print(source)
    except Exception as e:
        print("Could not inspect:")
        print(repr(e))


def main():
    print("=" * 80)
    print("AROMA — Mllama Preprocessing Geometry Probe")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    ip = processor.image_processor

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print("\nOriginal image:")
    print("  size =", image.size)

    print("\nProcessor:")
    print("  tile size       =", ip.size)
    print("  max image tiles =", ip.max_image_tiles)

    print_source(
        ip.resize,
        "MllamaImageProcessor.resize",
    )

    print_source(
        ip.pad,
        "MllamaImageProcessor.pad",
    )

    # ----------------------------------------------------------
    # Normal processor output
    # ----------------------------------------------------------

    outputs = ip(
        images=image,
        return_tensors="pt",
    )

    print("\n" + "=" * 80)
    print("NORMAL PREPROCESSOR OUTPUT")
    print("=" * 80)

    for key, value in outputs.items():
        print(f"\n{key}")

        if torch.is_tensor(value):
            print("  shape =", tuple(value.shape))
            print("  dtype =", value.dtype)

            if key in {
                "aspect_ratio_ids",
                "aspect_ratio_mask",
                "num_tiles",
            }:
                print(
                    "  value =",
                    value.cpu().tolist(),
                )

        else:
            print("  type  =", type(value))
            print("  value =", value)

    print("\nInterpretation:")

    mask = outputs[
        "aspect_ratio_mask"
    ][0, 0]

    active_tiles = [
        i
        for i, x in enumerate(mask.tolist())
        if x == 1
    ]

    print(
        "  active tile indices =",
        active_tiles,
    )

    print(
        "  active tile count   =",
        len(active_tiles),
    )

    print(
        "  padded tile count   =",
        len(mask) - len(active_tiles),
    )

    # ----------------------------------------------------------
    # Attempt to call resize directly
    # ----------------------------------------------------------

    print("\n" + "=" * 80)
    print("DIRECT RESIZE PROBE")
    print("=" * 80)

    try:
        import numpy as np

        arr = np.array(image)

        print(
            "Input numpy shape:",
            arr.shape,
        )

        # Mllama image processor expects batched tensors internally.
        tensor = torch.from_numpy(
            arr
        ).permute(
            2, 0, 1
        ).unsqueeze(0)

        print(
            "Input tensor shape:",
            tuple(tensor.shape),
        )

        resized, aspect_ratio = ip.resize(
            image=tensor,
            size=ip.size,
            resample=ip.resample,
            max_image_tiles=ip.max_image_tiles,
        )

        print(
            "Resize output shape:",
            tuple(resized.shape),
        )

        print(
            "Returned aspect_ratio:",
            aspect_ratio,
        )

        if (
            isinstance(aspect_ratio, (list, tuple))
            and len(aspect_ratio) == 2
        ):
            h_tiles, w_tiles = aspect_ratio

            print(
                "num_tiles_height =",
                h_tiles,
            )

            print(
                "num_tiles_width  =",
                w_tiles,
            )

            print(
                "active tiles     =",
                h_tiles * w_tiles,
            )

            print(
                "canvas size      =",
                (
                    w_tiles * ip.size.width,
                    h_tiles * ip.size.height,
                ),
            )

    except Exception as e:
        print(
            "Direct resize probe failed:"
        )
        print(
            repr(e)
        )

        print(
            "\nThis is okay; the source code above "
            "is still useful for resolving geometry."
        )


if __name__ == "__main__":
    main()
