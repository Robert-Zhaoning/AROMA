import json
from pathlib import Path

from PIL import Image


META = Path("data/calibration50/metadata.jsonl")


def main():
    records = [
        json.loads(line)
        for line in META.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) == 50, f"Expected 50 records, got {len(records)}"

    sample_ids = set()

    count_hist = {}
    condition_hist = {}

    for r in records:
        sid = r["sample_id"]

        assert sid not in sample_ids, f"Duplicate sample_id: {sid}"
        sample_ids.add(sid)

        image_path = Path(r["image_path"])
        assert image_path.exists(), f"Missing image: {image_path}"

        image = Image.open(image_path)
        assert image.size == (
            r["image_width"],
            r["image_height"],
        )

        gt = r["ground_truth"]

        assert len(r["target_objects"]) == gt, (
            f"{sid}: GT={gt} but target_objects="
            f"{len(r['target_objects'])}"
        )

        for obj in r["target_objects"]:
            box = obj["bbox_norm"]

            assert len(box) == 4
            assert all(0 <= x <= 1 for x in box)
            assert box[0] < box[2]
            assert box[1] < box[3]

        count_hist[gt] = count_hist.get(gt, 0) + 1

        condition = r["condition"]
        condition_hist[condition] = condition_hist.get(condition, 0) + 1

    assert set(count_hist.keys()) == set(range(1, 11))

    for count in range(1, 11):
        assert count_hist[count] == 5, (
            f"Count {count} has {count_hist[count]} samples"
        )

    for condition, n in condition_hist.items():
        assert n == 10, (
            f"Condition {condition} has {n} samples"
        )

    print("=" * 72)
    print("Calibration-50 Dataset Audit")
    print("=" * 72)
    print("Total samples:", len(records))
    print("Unique IDs   :", len(sample_ids))
    print("Counts       :", count_hist)
    print("Conditions   :", condition_hist)
    print()
    print("CALIBRATION-50 DATASET AUDIT PASS")


if __name__ == "__main__":
    main()
