# AROMA 2.0 — MN-CSA Gate B2 v1 Result

## Population

653 frozen non-NOOP Proc-Count-Causal v3 examples.

## Results

Baseline:

175 / 653 = 26.80%

Whole-head:

333 / 653 = 50.995%

Same-alpha rank-4 CSA:

217 / 653 = 33.23%

Magnitude-Normalized rank-4 CSA:

343 / 653 = 52.53%

## MN-CSA vs whole-head

Difference:

+1.531 percentage points

Paired bootstrap 95% CI:

[-0.766, +3.828] pp

Exact McNemar p:

0.237

The difference is not statistically distinguishable from
zero.

## MN-CSA vs same-alpha CSA

Difference:

+19.296 percentage points

Paired bootstrap 95% CI:

[+16.08, +22.665] pp

Exact McNemar p:

2.642343883518912e-28

## Recovery

Recovery fraction:

1.0862

Thus magnitude normalization recovered approximately 108.6%
of the behavioral gap between same-alpha CSA and whole-head
actuation.

## Repair / break decomposition

Whole-head:

repairs = 169
breaks = 11
net = 158

Same-alpha CSA:

repairs = 49
breaks = 7
net = 42

MN-CSA:

repairs = 185
breaks = 17
net = 168

MN-CSA therefore produced 10 more net repairs than
whole-head actuation on the intervention subset.

When embedded in the full 2000-example frozen policy, this
corresponds to a +0.50 percentage-point point estimate over
whole-head actuation.

## Interpretation

The same-alpha failure of low-rank CSA was predominantly a
magnitude-calibration failure.

After matching pooled intervention magnitude, a rank-4
subspace recovered the behavioral effectiveness of the
original 128-dimensional whole-head actuator.

MN-CSA was statistically indistinguishable from whole-head
actuation overall, with a slightly higher point estimate.

This establishes low-rank causal-subspace actuation as a
behaviorally viable actuator, but does not yet establish that
the learned U4 is superior to arbitrary rank-4 directions.

Random-subspace and low-sensitivity-subspace controls are
therefore required next.

## Important heterogeneity

The strongest action group, alpha=4, favored MN-CSA by
approximately +2.43 pp.

Intermediate alpha values did not show the same advantage.

Performance also varied substantially by procedural
condition, with positive effects on grid and row examples and
a strong negative effect on distractor examples.

No post-hoc action or condition-specific tuning is performed
on this result.
