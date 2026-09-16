# AROMA 2.0 — CSA v4 Final Behavioral Runner Freeze v1

Status:

FROZEN BEFORE ANY PROC-COUNT-CAUSAL v4 MODEL INFERENCE.

This document supplements:

docs/AROMA2_CSA_V4_CONFIRMATION_FREEZE_v1.md

## Dataset

Proc-Count-Causal v4

N = 2000

The dataset has already passed the independent pre-inference
audit:

- 2000 unique IDs
- 2000 unique seeds
- 2000 unique image hashes
- zero seed overlap with v1/v2/v3
- zero image-hash overlap with v1/v2/v3
- exact count balance
- exact condition balance
- exact count-condition balance

No v4 image is selected, removed, or replaced based on model
behavior.

## Model

meta-llama/Llama-3.2-11B-Vision-Instruct

Exact revision:

9eb2daaa8597bf192a8b0e73f848f3a102794df5

Inference dtype:

bfloat16

Attention implementation:

eager

Model is explicitly placed in eval mode.

## Frozen controller

Use exactly:

outputs/proc_count_causal_v2/controller/
final_frozen_controller/
aroma_cardinality_controller.joblib

Feature construction:

Reuse state_to_features() from:

scripts/run_proc_count_causal_v3_final.py

Decision rule:

Reuse controller_decision() from:

scripts/run_proc_count_causal_v3_final.py

Feature count:

39

Action space:

{0, 1, 1.5, 2, 4}

No retraining or retuning.

## Action-selection discipline

For each v4 example:

1. Run the baseline model.
2. Construct GT-free controller features.
3. Freeze the controller-selected alpha.
4. Apply the SAME selected alpha to:
   - whole-head L18H13
   - MN-CSA L18H13
5. Only after all predictions are produced is ground truth
   read for evaluation.

## Whole-head actuator

At L18H13:

H'_t = alpha H_t.

## Frozen MN-CSA actuator

Use exactly:

outputs/aroma2/csa_gate_a_v3/U4_primary.npy

Rank:

4

For:

h = mean_t H_t

p = U4 U4^T h

d = p / ||p||

apply:

Delta h =
(alpha - 1)
||h||
d

and:

H'_t =
H_t + Delta h.

No rank, basis, gain, normalization, controller, or action
changes are allowed.

## NOOP

For alpha = 1:

baseline
=
whole-head
=
MN-CSA

without additional model forwards.

## Primary endpoint

MN-CSA versus baseline over all 2000 v4 examples.

Primary success requires BOTH:

1. MN-CSA accuracy change > 0
2. two-sided exact McNemar p < 0.05

This rule was frozen before v4 behavioral inference.

## Secondary comparisons

Whole-head versus baseline.

MN-CSA versus whole-head.

Report for each paired comparison:

- accuracy difference
- paired bootstrap 95% CI
- two-sided exact McNemar p

## Bootstrap

Replicates:

20000

Seed:

20260916

No bootstrap threshold beyond the frozen primary endpoint is
introduced.

## Repair / break outcomes

Relative to the same current baseline report:

- repairs
- breaks
- net repairs

for both whole-head and MN-CSA.

## Prespecified stratification

Report:

- selected-action distribution
- condition-wise outcomes
- count-wise outcomes

No subgroup-specific policy adaptation is allowed.

## Interpretation

This is confirmation-only.

Regardless of result, no additional v4-based tuning is
permitted.

After this run, the synthetic CSA development/confirmation
line is closed.
