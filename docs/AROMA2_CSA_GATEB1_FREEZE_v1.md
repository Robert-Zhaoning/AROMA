# AROMA 2.0 — CSA Gate B1 Freeze v1

Status:
FROZEN BEFORE CSA BEHAVIORAL RESULTS ARE OBSERVED.

## Purpose

Gate B1 tests whether the rank-4 cardinality-sensitive
subspace discovered inside L18H13 can replace the original
whole-head actuator under the already-frozen Proc-Count-
Causal v3 routing policy.

Only the actuator changes.

The following remain frozen:

- evaluation population
- model
- prompt
- expanded numeral proxy
- controller
- selected alpha
- L18H13
- action space
- output scoring
- ground truth evaluation

## Evaluation population

Exact frozen Proc-Count-Causal v3 confirmation:

N = 2000

Archive:

outputs/proc_count_causal_v3/final_frozen_controller/
v3_final_results.csv

Frozen original result:

baseline correct = 999 / 2000
whole-head post correct = 1161 / 2000
repairs = 173
breaks = 11
net repairs = 162

## Frozen router actions

{0, 1, 1.5, 2, 4}

Observed frozen action counts:

alpha=0   : 24
alpha=1   : 1347
alpha=1.5 : 32
alpha=2   : 63
alpha=4   : 534

No router is retrained.

For every sample, CSA receives exactly the alpha already
selected by the frozen original AROMA controller.

## Original actuator

For each token-level L18H13 vector H_t:

H'_t = alpha H_t.

This scales all 128 dimensions.

## CSA basis

Use exactly:

outputs/aroma2/csa_gate_a_v3/U4_primary.npy

Shape:

128 x 4

No rank search is permitted.

## CSA coordinate

Gate A defined the sample-level actuator coordinate as:

h = mean_t H_t.

The Gate-A gradient was taken with respect to h while
preserving token-level residuals.

Gate B therefore uses the matching intervention geometry.

Define:

P_U h = U4 U4^T h

and

delta_h = (alpha - 1) P_U h.

Then every token is updated by the same displacement:

H'_t = H_t + delta_h.

Equivalently:

h' =
h + (alpha - 1) U4 U4^T h

while:

H'_t - h' = H_t - h.

Thus CSA changes only the pooled rank-4 cardinality
coordinate and preserves the orthogonal/token-residual
structure.

alpha=1 must be exact identity.

## Gate-A context

Formal Gate A observed:

total E4 = 0.8982293582
centered E4 = 0.8677753028

Both exceeded their norm-matched empirical null q99 with
empirical p approximately 1e-4.

The original internal 5x enrichment promotion heuristic
remains preserved in the Gate-A record. It is not treated
as a statistical-validity requirement for this behavioral
follow-up.

## Gate B1 primary comparison

Compare:

same router + same alpha + whole-head actuator

against

same router + same alpha + rank-4 CSA actuator.

Primary paired effect:

Delta =
accuracy_CSA - accuracy_whole.

Report:

- whole-head accuracy
- CSA accuracy
- paired difference
- paired bootstrap 95% CI
- exact McNemar p-value

Bootstrap:

20000 replicates

Seed:

20260915

No arbitrary multiplicative behavioral threshold is used.

## Secondary outcomes

Report:

- repairs
- breaks
- net repairs
- whole-head -> CSA discordant outcomes
- results by selected alpha
- results by procedural condition

Because both methods share the same baseline:

Delta accuracy

and

Delta(net repairs) / N

are mathematically equivalent.

The repair/break decomposition is reported to explain the
source of any behavioral difference.

## Interpretation

A positive paired effect provides direct evidence that CSA
is a better actuator under the existing frozen policy.

Similar overall accuracy with fewer breaks provides evidence
for improved intervention specificity.

A negative B1 result does NOT by itself invalidate CSA,
because the frozen controller was trained to predict the
utility of whole-head interventions.

If B1 is negative or ambiguous, Gate B2 will compare
whole-head and CSA complete fixed-action matrices without
the router.

