# AROMA 2.0 — Final Untouched v4 CSA Confirmation Result

## Status

PRIMARY CONFIRMATION SUCCESS

The synthetic CSA development and confirmation line is now
closed.

## Dataset

Proc-Count-Causal v4

N = 2000

Generated only after the MN-CSA algorithm was frozen.

Pre-inference audit verified:

- 2000 unique sample IDs
- 2000 unique seeds
- 2000 unique image hashes
- zero seed overlap with v1/v2/v3
- zero pixel-hash overlap with v1/v2/v3
- exact count balance
- exact condition balance
- exact count-condition balance

No sample was selected or replaced based on model behavior.

## Overall accuracy

Baseline:

1005 / 2000 = 50.250%

Whole-head AROMA:

1145 / 2000 = 57.250%

Frozen MN-CSA AROMA:

1165 / 2000 = 58.250%

## Whole-head versus baseline

Difference:

+7.000 percentage points

Paired bootstrap 95% CI:

[+5.8, +8.2] pp

Exact McNemar p:

3.330893983425689e-33

## MN-CSA versus baseline

Difference:

+8.000 percentage points

Paired bootstrap 95% CI:

[+6.7, +9.3] pp

Exact McNemar p:

2.6651344544768162e-36

This satisfies the frozen primary confirmation criterion.

## MN-CSA versus whole-head

Difference:

+1.000 percentage point

Paired bootstrap 95% CI:

[+0.3, +1.7] pp

Exact McNemar p:

0.00778774363057435225

Thus MN-CSA shows a small but statistically significant
accuracy advantage over the original whole-head actuator on
the untouched confirmation set.

## Repair / break decomposition

Whole-head:

repairs = 150
breaks = 10
net repairs = 140

MN-CSA:

repairs = 174
breaks = 14
net repairs = 160

MN-CSA therefore yields:

+24 additional repairs
+4 additional breaks
+20 additional net repairs

relative to whole-head actuation.

The +20 net repair difference over 2000 examples equals the
observed +1.00 percentage-point accuracy difference.

## Frozen action distribution

alpha=0:
27 / 2000 = 1.35%

alpha=1:
1396 / 2000 = 69.80%

alpha=1.5:
32 / 2000 = 1.60%

alpha=2:
47 / 2000 = 2.35%

alpha=4:
498 / 2000 = 24.90%

## Interpretation

The final untouched confirmation establishes that the
gradient-derived rank-4 cardinality-sensitive subspace is
not merely a mechanistic correlate.

After magnitude calibration, the frozen low-rank actuator:

1. substantially improves over baseline,
2. reproduces the repair capability of whole-head actuation,
3. and achieves a small statistically significant advantage
   over the original 128-dimensional whole-head actuator.

Together with the earlier Bottom-4 and 20-random-subspace
controls, the evidence supports a specific role for the
learned high-sensitivity directions in finite cardinality
control.

No additional synthetic CSA tuning will be performed after
this result.
