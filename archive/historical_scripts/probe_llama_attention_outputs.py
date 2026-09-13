import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)


MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

IMAGE_PATH = "data/calibration50/images/cal50_n05_row.png"

PROMPT = """
Inspect the image carefully.

How many red circles are visible?

Return the number only.
"""


processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)

model.eval()

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
                "text": PROMPT.strip(),
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

inputs = {
    key: (
        value.to(model.device)
        if isinstance(value, torch.Tensor)
        else value
    )
    for key, value in inputs.items()
}

print("Input keys:")

for key, value in inputs.items():
    if torch.is_tensor(value):
        print(
            key,
            tuple(value.shape),
            value.dtype,
        )
    else:
        print(
            key,
            type(value),
        )

with torch.inference_mode():
    outputs = model(
        **inputs,
        output_attentions=True,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )

print("\n=== OUTPUT TYPE ===")
print(type(outputs))

print("\n=== OUTPUT KEYS ===")
print(outputs.keys())

for key in outputs.keys():
    value = getattr(
        outputs,
        key,
    )

    print(
        "\nFIELD:",
        key,
    )

    if isinstance(
        value,
        (tuple, list),
    ):
        print(
            "Length:",
            len(value),
        )

        for i, item in enumerate(
            value[:10]
        ):
            if torch.is_tensor(item):
                print(
                    f"  [{i}] "
                    f"shape={tuple(item.shape)} "
                    f"dtype={item.dtype}"
                )
            else:
                print(
                    f"  [{i}] type={type(item)}"
                )

    elif torch.is_tensor(value):
        print(
            "Shape:",
            tuple(value.shape),
            "dtype:",
            value.dtype,
        )

    else:
        print(
            "Type:",
            type(value),
        )
