# Utility-Aware AROMA v1 — Frozen Archive

This archive contains the compact, review-relevant artifacts for the
Utility-Aware AROMA routing experiments.

## Evidence chain

### 1. Development: grouped 5-fold out-of-fold evaluation

Population: frozen upward-action development samples.

- Identity: 44/160 = 27.50%
- Fold-selected global candidate: 100/160 = 62.50%
- Utility Router OOF: 108/160 = 67.50%
- Utility Router - global: +5.00 pp
- Paired bootstrap 95% CI: [+1.25, +9.375] pp
- Exact McNemar p: 0.03857421875
- Router repairs / breaks / net: 72 / 8 / 64
- Global repairs / breaks / net: 72 / 16 / 56

This stage was used for model development and cross-validation.

## 2. Frozen external validation

The trained Utility Router was frozen before evaluation on the independent
TSG prospective cohort.

Population: N=156 frozen upward-action samples.

- Identity: 45/156 = 28.85%
- Frozen global MN beta=2.0: 104/156 = 66.67%
- Frozen Utility Router: 111/156 = 71.15%
- Difference: +4.487 pp
- Paired bootstrap 95% CI: [+0.641, +8.974] pp
- Exact McNemar p: 0.0654296875
- Router repairs / breaks / net: 71 / 5 / 66
- Global repairs / breaks / net: 70 / 11 / 59

No router retraining was performed on this cohort.

## 3. Untouched final confirmation

The final confirmation cohort was generated and frozen after the Utility
Router, candidate family, gain grid, comparator, and evaluation protocol
were fixed.

Primary population: N=294 frozen upward-action samples.

- Frozen global MN beta=2.0: 178/294 = 60.54%
- Frozen Utility Router: 195/294 = 66.33%
- Difference: +5.782 pp
- Paired bootstrap 95% CI: [+3.401, +8.503] pp
- Exact McNemar p: 1.52587890625e-05
- Router-only / global-only: 17 / 0

Mechanistic behavior:

- Utility Router repairs / breaks / net: 138 / 16 / 122
- Global MN repairs / breaks / net: 132 / 27 / 105

Selection distribution:

- IDENTITY: 60
- MN: 134
- SA: 100

Full-policy N=1000:

- Baseline: 480/1000 = 48.0%
- Frozen global MN beta=2.0: 585/1000 = 58.5%
- Frozen Utility Router: 602/1000 = 60.2%

## Framework interpretation

Utility-Aware AROMA decomposes inference-time mechanistic control into:

1. causal geometry — where to intervene;
2. actuator direction — MN or sensitivity-aligned SA;
3. gain — how strongly to intervene;
4. utility-aware routing — whether and which intervention to apply.

The final result is a confirmatory comparison against the frozen global
MN beta=2.0 comparator. Development, external validation, and final
confirmation are intentionally kept separate.

## Reproducibility

`SHA256SUMS.txt` records hashes for every archived file.

Large intermediate artifacts such as gradient matrices, activation matrices,
partial checkpoints, model logs, and generated image corpora are intentionally
not duplicated in this compact archive. Their frozen hashes and provenance
are recorded by the included freeze/protocol artifacts and the experiment
repository.
