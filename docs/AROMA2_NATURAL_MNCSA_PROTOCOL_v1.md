# AROMA Natural-Domain MN-CSA Validation v1

## Status

**FROZEN BEFORE NATURAL-DOMAIN MN-CSA INFERENCE**

This protocol defines the natural-domain validation of the already-developed
Magnitude-Normalized Causal Subspace Actuation (MN-CSA) operator.

No natural-domain MN-CSA outcome was inspected before freezing this protocol.

---

## 1. Scientific question

The experiment asks:

> Does the synthetic-derived rank-4 MN-CSA actuation geometry preserve or
> improve the natural-domain repair behavior of the already-validated
> whole-head L18H13 actuator?

This experiment does **not** relearn the actuator, the subspace, the router,
the action set, or the magnitude-normalization rule.

---

## 2. Experimental interpretation

The experiment isolates **actuation geometry**.

For every example:

1. obtain the baseline model state;
2. run the frozen K=500 natural controller exactly once;
3. obtain one frozen selected action alpha;
4. apply that exact same alpha to:
   - whole-head L18H13 actuation;
   - frozen rank-4 MN-CSA actuation.

Therefore:

```text
router(x) -> alpha_i

                +--> whole-head(alpha_i)
alpha_i --------|
                +--> MN-CSA(alpha_i, U4)
```

There is no independent routing decision for MN-CSA.

---

## 3. Model

Model:

```text
meta-llama/Llama-3.2-11B-Vision-Instruct
```

Frozen revision:

```text
9eb2daaa8597bf192a8b0e73f848f3a102794df5
```

Attention implementation:

```text
eager
```

Primary actuator:

```text
Layer 18
Head 13
hidden size = 4096
head dimension = 128
head slice = [1664:1792]
```

---

## 4. Frozen MN-CSA basis

Basis:

```text
outputs/aroma2/csa_gate_a_v3/U4_primary.npy
```

Rank:

```text
r = 4
```

SHA256:

```text
af50da2cf49268cc55dc33d44dad50c0cd5a4fa29cda6cad3c01cc0b4ae63f14
```

No natural-domain basis fitting is allowed.

No rank selection is allowed.

No basis rotation or replacement is allowed.

---

## 5. Frozen MN-CSA operator

For pooled L18H13 state h:

```text
p = U4 U4^T h
d_U = p / ||p||
```

The frozen intervention is:

```text
Delta h_MN = (alpha - 1) ||h|| d_U
```

The same pooled displacement magnitude as whole-head scaling is therefore:

```text
||Delta h_MN|| = |alpha - 1| ||h||
```

No modification of this formula is allowed after natural outcomes are seen.

---

## 6. Frozen natural controller

Controller:

```text
outputs/phase2_natural_controller_k500/
final_frozen_controller/
aroma_natural_utility_controller_k500.joblib
```

SHA256:

```text
d9d8ba3a24814bf71b3f4039d1effa864b57754a00539f1260a1d077c03449f5
```

Calibration size:

```text
K = 500
Simple = 250
Complex = 250
```

Feature count:

```text
39
```

Utility model:

```text
StandardScaler -> Ridge
ridge alpha = 0.01
```

Action set:

```text
{0, 1, 1.5, 2, 4}
```

Utility threshold:

```text
0.1
```

No controller retraining or recalibration is allowed.

---

## 7. Natural evaluation population

Dataset:

```text
TallyQA official test
```

Evaluation cohort:

```text
AROMA TallyQA Natural Confirmation v2
```

Manifest:

```text
outputs/tallyqa_natural_confirmation_v2/manifest.jsonl
```

Manifest SHA256:

```text
cf42a6c4b07f21d72c88d302815fd1419dd912431b203ff9f8cc53f3fc05ae0d
```

Population:

```text
N = 4000
Simple = 2000
Complex = 2000
one question per image
4000 unique image paths
4000 unique image IDs
```

Image inventory SHA256:

```text
475e28e0ebb7934d49d2062734b164ff04f4f04461d61de9ee4315f96182f0b2
```

The cohort was frozen before MN-CSA was developed and was not used to learn:

```text
U4
rank
MN normalization
Bottom-4 controls
random rank-4 controls
```

However, this cohort has previously been used to evaluate the whole-head
natural controller.

Therefore it must be described as a:

> pre-existing held-out natural cohort for MN-CSA evaluation

and **not** as a newly generated untouched natural confirmation cohort.

---

## 8. Prompt and numeral policy

Use the original TallyQA question without rewriting.

Append:

```text
Return the number only.
```

Use the separately audited expanded numeral proxy:

```text
0, ..., 15
```

Ground truth must not be read before:

```text
baseline inference
router action selection
whole-head action execution
MN-CSA action execution
```

---

## 9. Primary endpoint

The primary comparison is:

```text
MN-CSA vs same-run whole-head actuation
```

on all 4000 examples.

Primary estimand:

```text
Delta = Accuracy_MN - Accuracy_Whole
```

Report:

```text
point estimate
paired 95% bootstrap CI
two-sided exact McNemar p-value
Whole-wrong -> MN-correct count
Whole-correct -> MN-wrong count
```

There is no additional post-hoc GO/NO-GO threshold.

The effect estimate, confidence interval, and exact paired test are reported
regardless of direction.

---

## 10. Secondary endpoints

Pre-specified secondary summaries are:

```text
Baseline accuracy
Whole-head accuracy
MN-CSA accuracy

Baseline -> Whole change
Baseline -> MN change

Whole repairs
Whole breaks

MN repairs
MN breaks

intervention rate

Simple / Complex results
by-count results
by-selected-action results
```

Mechanistic diagnostics include:

```text
rho = ||U4 U4^T h|| / ||h||

MN displacement-strength relative error

MN subspace residual
```

These diagnostics may explain results but may not be used to retune the method.

---

## 11. Required implementation invariants

Before the full run:

### Identity

For alpha = 1:

```text
whole-head logits == baseline logits exactly
MN-CSA logits == baseline logits exactly
```

### Magnitude

For non-identity MN-CSA:

```text
||Delta h_MN||
/
(|alpha - 1| ||h||)
approximately equals 1
```

Required relative error:

```text
<= 1e-5
```

### Geometry

The MN-CSA displacement must remain inside the frozen U4 subspace.

Required relative residual:

```text
<= 1e-5
```

### Routing

Exactly one selected alpha is produced per example and reused for both
whole-head and MN-CSA execution.

---

## 12. Same-run evaluation requirement

Baseline, whole-head, and MN-CSA predictions must be recomputed in the same
runtime and model instance.

Archived whole-head results may be used only as a reproducibility comparison.

The primary MN-CSA versus whole-head result uses same-run predictions.

This prevents runtime or model-environment drift from being misinterpreted as
an actuation-geometry effect.

---

## 13. Prohibited changes after inference begins

After the first natural MN-CSA experimental inference begins, do not change:

```text
model revision
head
rank
U4
MN formula
action set
router
controller calibration
feature set
ridge alpha
utility threshold
prompt
numeral proxy
evaluation cohort
primary endpoint
```

No Simple/Complex-specific method selection is allowed.

No natural-specific subspace learning is allowed within this validation.

---

## 14. Interpretation policy

Possible outcomes are interpreted without rescue tuning.

If MN-CSA exceeds whole-head actuation, this supports transfer of the
synthetic-derived actuation geometry to natural images.

If MN-CSA approximately matches whole-head actuation, this supports
preservation of natural actuator utility using a low-rank sensitivity-aligned
intervention.

If MN-CSA underperforms whole-head but remains above baseline, this supports
partial geometry transfer.

If MN-CSA fails to improve over baseline while whole-head remains useful, the
supported conclusion is that actuator capacity transfers more robustly than
the learned actuation geometry.

A negative result must not trigger post-hoc rank, basis, alpha, router, or
normalization tuning within this experiment.
