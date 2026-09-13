import json
from collections import Counter
from pathlib import Path

from PIL import Image


META = Path(
    "data/proc_count_causal_v1/metadata.jsonl"
)


def main():
    records = [
        json.loads(line)
        for line in META.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(records) == 300

    ids = [
        r["sample_id"]
        for r in records
    ]

    assert (
        len(ids)
        == len(set(ids))
    )

    cell_counts = Counter()

    for r in records:
        path = Path(
            r["image_path"]
        )

        assert path.exists(), (
            f"Missing image: {path}"
        )

        image = Image.open(path)

        assert image.size == (
            768,
            512,
        )

        assert (
            len(
                r["target_objects"]
            )
            == r["ground_truth"]
        )

        if (
            r["condition"]
            == "distractors"
        ):
            assert (
                len(
                    r["distractor_objects"]
                )
                == 8
            )
        else:
            assert (
                len(
                    r["distractor_objects"]
                )
                == 0
            )

        cell_counts[
            (
                r["ground_truth"],
                r["condition"],
            )
        ] += 1

    assert (
        len(cell_counts)
        == 50
    )

    assert all(
        n == 6
        for n in cell_counts.values()
    )

    print("=" * 80)
    print(
        "Proc-Count-Causal v1 Audit"
    )
    print("=" * 80)

    print(
        "Total samples:",
        len(records),
    )

    print(
        "Unique IDs   :",
        len(set(ids)),
    )

    print(
        "Count-condition cells:",
        len(cell_counts),
    )

    print(
        "Replicates per cell:",
        sorted(
            set(
                cell_counts.values()
            )
        ),
    )

    print()
    print(
        "PROC-COUNT-CAUSAL V1 AUDIT PASS"
    )


if __name__ == "__main__":
    main()
