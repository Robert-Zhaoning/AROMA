# AROMA 2.0 — Magnitude-Normalized CSA Gate B2 Freeze v1

Status:

FROZEN BEFORE ANY MN-CSA BEHAVIORAL RESULT IS OBSERVED.

## Motivation

Gate B1 compared whole-head scaling against same-alpha
rank-4 CSA.

Gate B1 result:

whole-head accuracy = 57.95%
same-alpha CSA accuracy = 52.15%

whole-head repairs / breaks / net:
169 / 11 / 158

same-alpha CSA repairs / breaks / net:
49 / 7 / 42

The subsequent label-free actuation-strength audit found:

median
||P_U h|| / ||h||
=
0.3086

and median projected activation energy share:
approximately 9.52%.

Therefore same alpha did not imply matched intervention
magnitude.

## Population

Use exactly the 653 examples from the frozen Proc-Count-
Causal v3 controller for which:

selected_alpha != 1.

Frozen action assignments are not modified.

Action support:

alpha = 0
alpha = 1.5
alpha = 2
alpha = 4

Counts:

alpha=0   : 24
alpha=1.5 : 32
alpha=2   : 63
alpha=4   : 534

No controller is retrained.

No alpha is re-selected.

## Model and actuator

Model:

meta-llama/Llama-3.2-11B-Vision-Instruct

Revision:

9eb2daaa8597bf192a8b0e73f848f3a102794df5

Actuator:

L18H13

Head dimension:

128

CSA rank:

4

Basis:

outputs/aroma2/csa_gate_a_v3/U4_primary.npy

No rank search is permitted.

## Pooled coordinate

For token-level head output:

H in R^(T x 128)

define:

h =
mean_t H_t.

Let:

p =
U4 U4^T h.

## Same-alpha CSA

For reference:

Delta h_CSA =
(alpha - 1) p.

H'_t =
H_t + Delta h_CSA.

## Magnitude-Normalized CSA

Define:

d_U =
p / ||p||.

Then:

Delta h_MN =
(alpha - 1)
||h||
d_U.

Therefore:

||Delta h_MN||
=
|alpha - 1|
||h||

which exactly matches the pooled displacement magnitude of
whole-head scaling:

Delta h_whole =
(alpha - 1) h.

The MN-CSA displacement direction remains entirely within
the learned rank-4 subspace.

Token residuals remain unchanged because the same pooled
displacement is added to every token.

## Important semantic distinction

MN-CSA matches intervention displacement magnitude.

It does NOT preserve the interpretation that alpha directly
scales the projected activation component.

For alpha < 1, magnitude matching can move the projected
component beyond zero.

This is intentional and is part of the pre-specified
magnitude-matched comparison.

## Same-run evaluation

For every one of the 653 non-NOOP samples, recompute in the
same current model process:

1. current baseline
2. current whole-head intervention
3. current same-alpha CSA
4. current MN-CSA

Ground truth is used only after all predictions have been
produced.

## Primary comparisons

Primary comparison A:

MN-CSA vs whole-head.

Primary comparison B:

MN-CSA vs same-alpha CSA.

For each comparison report:

- paired accuracy difference
- paired bootstrap 95% CI
- exact McNemar p-value

Bootstrap replicates:

20000

Seed:

20260915

## Repair / break outcomes

Relative to the current same-run baseline, report for:

- whole-head
- same-alpha CSA
- MN-CSA

the numbers of:

- repairs
- breaks
- net repairs

## Recovery fraction

Report descriptively:

Recovery =
(Acc_MN - Acc_sameCSA)
/
(Acc_whole - Acc_sameCSA).

This is an effect-size description only.

No success threshold is attached to Recovery.

## Pre-specified interpretation

### MN_OUTPERFORMS_WHOLE

MN-CSA vs whole-head paired bootstrap lower 95% bound > 0.

### MN_COMPETITIVE_RECOVERY

MN-CSA is significantly better than same-alpha CSA, while
its paired difference versus whole-head is not clearly
negative.

### MN_PARTIAL_RECOVERY

MN-CSA is significantly better than same-alpha CSA, but
still significantly worse than whole-head.

### MN_NO_CLEAR_RECOVERY

MN-CSA is not clearly better than same-alpha CSA.

### MN_WORSE

MN-CSA is clearly worse than same-alpha CSA.

No rank change, gain search, alpha search, routing change,
or threshold tuning is permitted after this experiment
starts.

If magnitude normalization does not provide meaningful
recovery, CSA remains a mechanistic result rather than the
primary actuator algorithm.
