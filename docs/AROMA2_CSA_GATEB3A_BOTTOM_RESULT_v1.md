# AROMA 2.0 — CSA Gate B3a Bottom-Subspace Control Result

## Population

653 frozen non-NOOP Proc-Count-Causal v3 examples.

Both Top-4 and Bottom-4 interventions used the same
magnitude-normalized rank-4 actuation rule.

## Result

Top-4:

343 / 653 = 52.527%

Bottom-4:

175 / 653 = 26.799%

Difference:

+25.727 percentage points

Paired bootstrap 95% CI:

[+22.052, +29.556] pp

Exact McNemar p:

7.473538051385848e-37

Primary status:

TOP4_CLEAR_SPECIFICITY

## Reproducibility

Gate-B2 Top-4 prediction replay match:

1.0

Median Top-4 projection ratio:

0.30857235193252563

Median Bottom-4 projection ratio:

0.10029305517673492

Because both interventions were magnitude normalized,
the behavioral difference cannot be attributed to different
pooled displacement magnitudes.

## Interpretation

High-cardinality-sensitivity directions are dramatically more
effective for finite cardinality control than the lowest-
sensitivity directions from the same L18H13 head.

This establishes sensitivity-specific control relative to a
strong low-sensitivity negative control.

It does not yet establish superiority to a typical arbitrary
rank-4 subspace.

Gate B3b therefore evaluates pre-frozen Haar-random rank-4
subspaces.
