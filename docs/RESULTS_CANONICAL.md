# AROMA Canonical Results

This file contains the canonical numerical results that may be cited in the AROMA paper, README, figures, tables, and supplementary material.

All numbers below are grounded in frozen or archived experiment artifacts documented in `docs/EXPERIMENT_LEDGER.md`.

---

## 1. L18H13 Bidirectional Cardinality Steering

Early 300-sample directional characterization using the original `1,...,10` numeral proxy:

- `alpha = 0.5`: expected numeral shifted downward on `274/300 = 91.33%`
- `alpha = 1.5`: expected numeral shifted upward on `272/300 = 90.67%`
- strict ordering `E[N]_{0.5} < E[N]_{1} < E[N]_{1.5}` on `266/300 = 88.67%`
- downward/upward response magnitude Spearman:
  - `rho = 0.951442127`
  - `p ≈ 1.59e-154`

Interpretation:

L18H13 acts as a bidirectional, dose-responsive causal cardinality actuator.

---

## 2. Proc-Count-Causal v2 Action-Space Headroom

Baseline accuracy:

`50.3%`

Full multistrength oracle:

- repairable baseline-wrong examples: `139/497 = 27.97%`
- oracle accuracy: `64.2%`
- oracle gain: `+13.9 pp`

Compressed five-action oracle over:

`{0, 1, 1.5, 2, 4}`

- repairable baseline-wrong examples: `138/497 = 27.77%`
- oracle accuracy: `64.1%`
- oracle gain: `+13.8 pp`

---

## 3. Frozen Synthetic Controller Development

OOF development result on Proc-Count-Causal v2:

- baseline accuracy: `50.3%`
- post-controller accuracy: `58.6%`
- gain: `+8.3 pp`
- repairs: `91`
- breaks: `8`
- net repairs: `+83`
- intervention rate: `33.3%`

Controller:

- features: `39`
- model family: per-action `StandardScaler -> Ridge`
- ridge alpha: `0.01`
- threshold: `0.1`
- action set: `{0, 1, 1.5, 2, 4}`
- actuator: `L18H13`

---

## 4. Frozen Proc-Count-Causal v3 Confirmation

Independent procedural confirmation on `N = 2000`:

- baseline correct: `999`
- baseline accuracy: `49.95%`
- post-controller correct: `1161`
- post-controller accuracy: `58.05%`
- gain: `+8.10 pp`
- 95% CI: `[+6.85, +9.40] pp`
- repairs: `173`
- breaks: `11`
- net repairs: `+162`
- intervention rate: `32.65%`
- exact McNemar `p ≈ 1.32e-38`
- primary success: `TRUE`

By condition:

- dense: `+10.25 pp`
- distractors: `+7.25 pp`
- grid: `+12.00 pp`
- random_sparse: `+7.00 pp`
- row: `+4.00 pp`

---

## 5. Frozen TallyQA Zero-Shot Natural OOD Result

Frozen synthetic controller on `N = 4000` natural TallyQA examples:

- baseline accuracy: `64.85%`
- post-controller accuracy: `61.875%`
- change: `-2.975 pp`
- 95% CI: `[-3.70, -2.25] pp`
- repairs: `51`
- breaks: `170`
- net repairs: `-119`
- intervention rate: `65.275%`
- exact McNemar `p ≈ 3.78e-16`
- primary success: `FALSE`

This is a frozen negative-transfer result and must remain visible.

---

## 6. Natural Action-Oracle Headroom

Post-hoc natural action-oracle analysis:

- baseline accuracy: `64.85%`
- oracle accuracy: `71.575%`
- oracle gain: `+6.725 pp`
- unique repairable wrong examples: `269/1406 = 19.13%`

Controller repair capture:

- controller repairs: `51`
- captured oracle-repairable examples: `51/269 = 18.96%`

Fixed `alpha = 0`:

- post accuracy: `67.20%`
- gain: `+2.35 pp`

Interpretation:

The actuator retains natural-domain repair potential even when the synthetic-trained routing policy fails.

---

## 7. Natural Error-Direction Shift

Among `1406` natural baseline errors:

- undercount: `297 = 21.12%`
- overcount: `1109 = 78.88%`
- absolute error `1`: `1017 = 72.33%`
- mean signed error among wrong: `+0.834993`
- median signed error: `+1`
- mean absolute error: `1.554765`

Interpretation:

Natural error composition differs strongly from synthetic controller-development data.

---

## 8. Controller Domain Shift

Selected-score / realized-utility calibration:

- synthetic v2/v3: approximately `rho ≈ +0.41`
- natural TallyQA: approximately `rho ≈ -0.067`

Natural frozen-controller intervention rate:

`65.275%`

Interpretation:

Synthetic-to-natural failure is primarily a routing/calibration failure rather than disappearance of the cardinality actuator.

---

## 9. Exact Ridge OOD Attribution

The frozen per-action Ridge scores were exactly reproduced to machine precision with zero decision mismatches.

One key pathological feature:

`baseline_numprob_15`

Synthetic scaler statistics:

- mean: approximately `5.09e-07`
- scale: approximately `6.97e-07`

An extreme natural value around:

`0.351970`

induces a standardized value on the order of:

`5e5`

causing very large linear action scores.

Interpretation:

`StandardScaler + Ridge` exhibits severe numerical OOD extrapolation on natural-domain features.

---

## 10. SPUC-v1 Support Projection

Support projection removed catastrophic score explosion but did not improve final natural accuracy.

TallyQA:

- original post accuracy: `61.875%`
- SPUC post accuracy: `61.875%`
- original intervention rate: `65.275%`
- SPUC intervention rate: `63.275%`
- policy agreement: `82.95%`
- repairs: `32`
- breaks: `151`
- net repairs: `-119`

Natural samples with at least one clipped feature:

`1593/4000 = 39.825%`

Interpretation:

Numerical support violation explains score explosion but not the deeper utility mismatch.

---

## 11. Natural Utility Learnability

Post-hoc 5-fold OOF natural-domain development:

- baseline accuracy: `64.85%`
- OOF post accuracy: `67.15%`
- gain: `+2.30 pp`
- repairs: `126`
- breaks: `34`
- net repairs: `+92`
- intervention rate: `16.45%`

Interpretation:

The same 39-dimensional representation still contains useful natural-domain intervention-utility signal.

---

## 12. Natural Adaptation Sample Efficiency

Mean accuracy gain by natural calibration size:

| K | Mean gain |
|---:|---:|
| 50 | `+0.330 pp` |
| 100 | `+1.180 pp` |
| 250 | `+1.870 pp` |
| 500 | `+2.120 pp` |
| 1000 | `+2.240 pp` |
| 2000 | `+2.325 pp` |
| 3200 | `+2.300 pp` |

At `K = 500`:

- `25/25` predefined development runs had positive gain
- mean gain: `+2.12 pp`
- mean intervention rate: `21.305%`

`K = 500` was selected as a sample-efficiency elbow, not as the maximum-performing K.

---

## 13. Frozen K=500 Natural-Adapted Controller

Calibration set:

- total: `500`
- Simple: `250`
- Complex: `250`

Frozen controller:

- features: `39`
- action set: `{0, 1, 1.5, 2, 4}`
- ridge alpha: `0.01`
- threshold: `0.1`
- actuator: `L18H13`

Controller bundle SHA256:

`d9d8ba3a24814bf71b3f4039d1effa864b57754a00539f1260a1d077c03449f5`

Calibration manifest SHA256:

`67dfdafa36c66a1299be7adea0d1232269e1b557607314972ed2b298312dc49a`

Freeze commit:

`9bbd337a979029da1bc46396c8e266da5c9e6193`

---

## 14. Untouched TallyQA Natural Confirmation v2

Independent image-disjoint confirmation on `N = 4000`:

- baseline correct: `2597`
- baseline accuracy: `64.925%`
- post-controller correct: `2682`
- post-controller accuracy: `67.050%`
- gain: `+2.125 pp`
- 95% CI: `[+1.50, +2.775] pp`
- repairs: `130`
- breaks: `45`
- net repairs: `+85`
- intervention rate: `20.0%`
- exact McNemar `p = 9.176009e-11`
- primary success: `TRUE`

Prespecified Simple subset:

- baseline: `78.05%`
- post: `79.15%`
- gain: `+1.10 pp`
- 95% CI: `[+0.25, +1.95] pp`
- repairs / breaks: `49 / 27`
- intervention rate: `11.8%`

Prespecified Complex subset:

- baseline: `51.80%`
- post: `54.95%`
- gain: `+3.15 pp`
- 95% CI: `[+2.20, +4.15] pp`
- repairs / breaks: `81 / 18`
- intervention rate: `28.2%`

Action distribution:

| Alpha | Count | Fraction |
|---:|---:|---:|
| 0 | 508 | `12.70%` |
| 1 | 3200 | `80.00%` |
| 1.5 | 32 | `0.80%` |
| 2 | 122 | `3.05%` |
| 4 | 138 | `3.45%` |

The development OOF gain was `+2.30 pp`; the untouched confirmation gain was `+2.125 pp`.

The confirmation set is image-, image-ID-, and question-disjoint from the earlier TallyQA development/evaluation set, but comes from the same official TallyQA benchmark distribution.

Do not describe this as cross-dataset generalization.

---

## 15. Canonical Experimental Story

The publication-level result chain is:

`causal discovery`
→ `bidirectional cardinality steering`
→ `adaptive synthetic repair`
→ `independent procedural confirmation`
→ `frozen natural zero-shot failure`
→ `mechanism-vs-router separation`
→ `domain-shift diagnosis`
→ `sample-efficient natural adaptation`
→ `successful untouched natural confirmation`

The strongest supported conclusion is:

A causally controllable cardinality mechanism can be used as an adaptive repair primitive, but intervention utility is domain dependent; lightweight natural-domain calibration can restore useful routing and generalize to a new image-disjoint natural confirmation sample.

