import json
import random
from pathlib import Path


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

NUM_HEADS = 32

CANDIDATES = [
    (33, 1),
    (3, 4),
    (18, 13),
    (8, 30),
    (3, 11),
    (13, 11),
    (33, 21),
]

SEED = 20260910

OUTPUT = Path(
    "outputs/proc_count_causal_v1/"
    "frozen_heads.json"
)


def main():
    universe = [
        (layer, head)
        for layer in CROSS_LAYERS
        for head in range(
            NUM_HEADS
        )
        if (
            layer,
            head,
        ) not in CANDIDATES
    ]

    rng = random.Random(
        SEED
    )

    controls = rng.sample(
        universe,
        len(CANDIDATES),
    )

    config = {
        "selection_status":
            "frozen_before_validation",

        "selection_source":
            "Calibration-50 discovery",

        "random_control_seed":
            SEED,

        "candidate_heads": [
            {
                "layer": l,
                "head": h,
            }
            for l, h
            in CANDIDATES
        ],

        "random_control_heads": [
            {
                "layer": l,
                "head": h,
            }
            for l, h
            in controls
        ],
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Frozen validation heads")
    print("=" * 80)

    print("\nCandidates:")

    for l, h in CANDIDATES:
        print(
            f"  L{l}H{h}"
        )

    print("\nRandom controls:")

    for l, h in controls:
        print(
            f"  L{l}H{h}"
        )

    print(
        "\nSaved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
