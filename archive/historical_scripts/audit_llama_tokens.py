from PIL import Image
from transformers import AutoProcessor


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

IMAGE_PATH = (
    "data/calibration50/images/cal50_n05_row.png"
)

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
""".strip()


def main():
    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

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

    ids = inputs[
        "input_ids"
    ][0].tolist()

    mask = inputs[
        "cross_attention_mask"
    ][0, :, 0, :]

    print("=" * 80)
    print("AROMA — Token / Cross-Attention Audit")
    print("=" * 80)

    print("\nFormatted prompt:")
    print(repr(formatted))

    print("\nSequence length:")
    print(len(ids))

    print("\nTokens:")

    for i, token_id in enumerate(ids):

        token = (
            processor.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )
        )

        cross_mask = (
            mask[i]
            .cpu()
            .tolist()
        )

        print(
            f"{i:02d} "
            f"id={token_id:6d} "
            f"cross={cross_mask} "
            f"token={repr(token)}"
        )


if __name__ == "__main__":
    main()
