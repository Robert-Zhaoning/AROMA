# AROMA

**Adaptive Reasoning with Object-grounded Mechanistic Analysis for Diagnosing and Repairing Stage-Specific Failures in Vision-Language Counting**

AROMA studies where visual counting fails inside multimodal large language models and whether failure-specific interventions outperform uniform repair.

## Failure Types

- P — Perception / Individuation
- R — Routing / Aggregation
- M — Magnitude-to-Symbol Mapping
- D — External Diagnostic Failure
- U — Unresolved
- C — Reliable / Correct

## Core Pipeline

Image + Question
-> Native Object Ledger
-> External + Internal Diagnostics
-> Failure Localization
-> Stage-Specific Repair
-> Verify / Abstain
-> Final Count
