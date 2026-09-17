# AROMA 2.0 — CSA Gate B1 Freeze v2

Status:
FROZEN BEFORE ANY CSA BEHAVIORAL RESULT WAS OBSERVED.

This protocol supersedes Gate B1 v1.

Gate B1 v1 terminated during historical-replay preflight,
before CSA behavioral evaluation began.

## Reason for revision

The original v1 preflight required exact equality between
historically archived V3 baseline predictions and current
baseline replay.

A borderline example showed numerical execution drift:

archived baseline != current baseline

while the current whole-head intervention still matched the
archived whole-head result.

Therefore exact historical prediction replay is too strict
for the primary actuator comparison.

Gate B1 v2 removes historical outcomes from the primary
comparison.

Historical V3 results remain provenance diagnostics only.

## Primary design

All behavioral quantities used in the CSA-vs-whole-head
comparison are recomputed in the SAME current execution
environment.

The frozen quantities remain:

- exact V3 N=2000 population
- exact frozen selected alpha for every sample
- model family
- L18H13
- action space
- rank-4 U4
- numeral proxy
- prompt
- scoring implementation

The router is NOT retrained or rerun.

Its already-frozen selected_alpha assignment is treated as a
fixed policy assignment.

## Current baseline

For every one of the 2000 V3 examples, recompute the current
baseline prediction.

## Current whole-head actuator

For samples with selected_alpha != 1:

H'_t = alpha H_t.

For selected_alpha = 1:

whole-head prediction equals current baseline.

## Current CSA actuator

Gate-A-aligned pooled rank-4 intervention:

h = mean_t H_t

delta_h =
(alpha - 1) U4 U4^T h

H'_t =
H_t + delta_h.

For selected_alpha = 1:

CSA prediction equals current baseline.

## Primary comparison

Compare current whole-head and current CSA predictions
sample-by-sample.

Primary effect:

Delta accuracy =
accuracy_CSA_current
-
accuracy_whole_current.

Report:

- current baseline accuracy
- current whole-head accuracy
- current CSA accuracy
- paired accuracy difference
- paired bootstrap 95% CI
- exact McNemar p-value

Bootstrap:
20000 replicates

Seed:
20260915

## Repair/break comparison

Repairs and breaks are defined relative to the recomputed
CURRENT baseline, not the historical archived baseline.

Thus:

repair =
current baseline wrong AND post-intervention correct

break =
current baseline correct AND post-intervention wrong.

## Historical archive

The original frozen V3 result remains reported only as a
reproducibility diagnostic:

baseline 999 / 2000
whole-head 1161 / 2000
repairs 173
breaks 11

Current-vs-archive agreement rates will be reported but will
not define Gate-B1 success.

## Interpretation

This design isolates actuator quality from numerical
execution drift because current whole-head and current CSA
are evaluated within the same model process, on the same
samples, using the same frozen action assignments.

No rank, alpha, router, or threshold tuning is permitted.
