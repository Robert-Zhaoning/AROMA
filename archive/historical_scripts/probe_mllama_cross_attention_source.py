import inspect

from transformers.models.mllama.modeling_mllama import (
    MllamaTextCrossAttention,
)

print("=" * 100)
print("AROMA — Mllama Cross-Attention Source Audit")
print("=" * 100)

print("\nCLASS:")
print(MllamaTextCrossAttention)

print("\nFORWARD SIGNATURE:")
print(
    inspect.signature(
        MllamaTextCrossAttention.forward
    )
)

print("\nFORWARD SOURCE:")
print("=" * 100)

print(
    inspect.getsource(
        MllamaTextCrossAttention.forward
    )
)

print("\n" + "=" * 100)
print("RELATED MODULE SOURCE")
print("=" * 100)

module = inspect.getmodule(
    MllamaTextCrossAttention
)

source = inspect.getsource(
    module
)

keywords = [
    "eager_attention_forward",
    "ALL_ATTENTION_FUNCTIONS",
    "attention_interface",
    "softmax",
    "scaled_dot_product",
]

for keyword in keywords:

    print(
        f"\n--- occurrences of {keyword!r} ---"
    )

    lines = source.splitlines()

    found = False

    for i, line in enumerate(lines):

        if keyword in line:

            found = True

            start = max(
                0,
                i - 12,
            )

            end = min(
                len(lines),
                i + 25,
            )

            for j in range(
                start,
                end,
            ):

                print(
                    f"{j + 1:05d}: "
                    f"{lines[j]}"
                )

            print()

    if not found:
        print("NOT FOUND")
