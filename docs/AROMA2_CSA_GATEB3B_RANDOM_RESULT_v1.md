# AROMA 2.0 — CSA Gate B3b Random-Subspace Specificity Result

## Population

653 frozen non-NOOP Proc-Count-Causal v3 examples.

All interventions used the same magnitude-normalized
rank-4 actuation rule.

## Learned Top-4

343 / 653 = 52.527%

## Twenty pre-frozen Haar-random rank-4 controls

Random mean accuracy:

27.175%

Random median accuracy:

26.876%

Random standard deviation:

1.864 percentage points

Random minimum:

23.890%

Random maximum:

30.781%

## Learned Top-4 versus random controls

Top-4 minus random mean:

+25.352 percentage points

Paired bootstrap 95% CI:

[+21.792, +28.936] pp

Top-4 minus best random:

+21.746 pp

Random subspaces with accuracy >= Top-4:

0 / 20

Empirical random-subspace rank p:

1 / 21
=
0.047619047619047616

## Top-4 repair / break decomposition

repairs = 185

breaks = 17

net repairs = 168

## Primary result

RANDOM_SPECIFICITY_POSITIVE

## Interpretation

Magnitude normalization alone does not explain the
behavioral effectiveness of CSA.

Typical random four-dimensional subspaces perform near the
baseline intervention-subset accuracy, whereas the learned
Gate-A Top-4 cardinality-sensitivity subspace reaches 52.53%.

Together with the Bottom-4 control, this establishes that
finite cardinality control is highly specific to the
gradient-derived high-sensitivity directions.

The CSA development phase is now closed.

No further changes to:

- rank
- learned basis
- magnitude normalization
- action space
- controller
- subspace construction

will be made before the final untouched confirmation.
