# AROMA 2.0 CSA Gate B1 v2 — Result

## Design

Same current execution environment.
Same frozen V3 population.
Same frozen router-selected alpha.
Same L18H13.
Only the actuator differs.

Whole-head:

Delta h = (alpha - 1) h

CSA:

Delta h = (alpha - 1) U4 U4^T h

## Result

Current baseline:

1001 / 2000 = 50.05%

Current whole-head:

1159 / 2000 = 57.95%

Current CSA:

1043 / 2000 = 52.15%

CSA minus whole-head:

-5.80 percentage points

Paired bootstrap 95% CI:

[-6.90, -4.75] pp

Exact McNemar p:

9.121014256730856e-31

## Repair / break decomposition

Whole-head:

repairs = 169
breaks = 11
net = 158

CSA:

repairs = 49
breaks = 7
net = 42

Delta net:

-116

## Interpretation

The same-alpha pooled rank-4 CSA intervention is a clear
negative drop-in replacement for whole-head scaling.

However, the pattern is dominated by lost repairs rather
than additional breaks.

This motivates a label-free actuation-strength audit before
concluding that the low-rank sensitivity subspace lacks
behavioral control capacity.

In particular, same alpha does not imply matched
intervention magnitude because:

||Delta h_CSA|| / ||Delta h_whole||
=
||U4 U4^T h|| / ||h||.

No behavioral threshold or action is changed on the basis of
this result.
