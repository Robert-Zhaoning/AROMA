# HoloCount External Evaluation Manifest

This directory freezes the primary HoloCount evaluation population for the
AROMA generalization extension.

Dataset:

```text
MM-MVR/HoloCount
```

Exact Hugging Face revision:

```text
f44cfe591e8f7e64b63a2fb080b98bffa10b45aa
```

Primary evaluation population:

```text
All 2480 records from data/metadata.jsonl at the frozen revision.
```

No outcome-dependent filtering is permitted.

The existing TallyQA-adapted AROMA controller remains frozen before the
first HoloCount model evaluation.

Primary evaluation uses the full 2480-example population.

The analysis restricted to ground-truth counts <= 15 is prespecified as a
secondary analysis only because the existing AROMA numeral-probability
feature family uses the audited 0..15 proxy.

Manifest SHA256:

```text
07e9672295230dfdad9e356102165b0190117101f865418efd1d3a6be9d403cd
```

At manifest-freeze time, no HoloCount VLM prediction, controller action,
accuracy, repair/break statistic, or action oracle had been evaluated.
