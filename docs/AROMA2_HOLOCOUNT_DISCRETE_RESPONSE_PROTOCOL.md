# AROMA 2.0 HoloCount Discrete-Response Rescue Screen

## Status

Exploratory retrospective natural-domain diagnostic.

This experiment does not alter the prior conclusions:

- Stage 1A Gate 0: NO_GO_WEAK_SIGNAL
- Residual-response rescue: RESCUE_NO_GO

HoloCount has already been used for development and diagnosis.
Therefore this experiment is not confirmatory evidence.

## Question

Can already-observed actuator responses expose HoloCount action utility
that Static-39 numeral geometry fails to recover?

## Population

Primary retrospective population:

    all HoloCount examples with ground_truth <= 15

Expected N:

    2292

This consists of:

    1134 baseline-wrong supported examples
    1158 baseline-correct examples

GT > 15 examples are excluded because the 0..15 numeral proxy cannot
represent their correct answer and fixed-action response data were not
collected for a comparable supported-control analysis.

## Probe actions

Only two previously computed intervention responses may be used as
router inputs:

    alpha = 0.0
    alpha = 1.5

Predictions from alpha=2 or alpha=4 may NOT be used as features.

Ground truth may NOT be used as a feature.

## Discrete response signature

Let

    y1   = baseline prediction
    y0   = prediction at alpha=0
    y15  = prediction at alpha=1.5

Define:

    d_down = y0 - y1
    d_up   = y15 - y1

The fixed seven-dimensional response representation is:

1. d_down
2. d_up
3. abs(d_down)
4. abs(d_up)
5. d_up - d_down
6. 0.5 * (y0 + y15) - y1
7. I[d_down <= 0 and d_up >= 0]

No feature search is permitted in this screen.

## Compared representations

1. Static39
2. DiscreteResponse7
3. Static39_Response7

## Utility targets

For each non-NOOP action:

    {0.0, 1.5, 2.0, 4.0}

utility is:

    U(x,a) = correctness(x,a) - correctness(x,1)

Therefore:

    +1 = repair
     0 = neutral
    -1 = break

## Model

For every action and representation:

    StandardScaler -> Ridge(alpha=0.01)

No hyperparameter search.

## Evaluation

Primary cross-validation:

    5-fold StratifiedGroupKFold

Stratification:

    official HoloCount split/subset

Group:

    image_sha256

Random seed:

    20260914

Policy threshold:

    0.1

No threshold tuning.

## Primary rescue comparison

    Static39_Response7 - Static39

## RESCUE-GO criteria

All must hold:

1. Static39_Response7 improves OOF net accuracy gain over Static39
   by at least +1.0 percentage point.

2. Image-cluster paired bootstrap 95% CI lower bound for that policy
   gain difference is > 0.

3. Mean per-action Spearman improves by at least +0.05.

4. Image-cluster bootstrap 95% CI lower bound for the mean-Spearman
   improvement is > 0.

5. Static39_Response7 is non-inferior to Static39 in per-action
   Spearman for at least 3 of 4 actions.

If all hold:

    HOLO_RESPONSE_RESCUE_GO

If both point estimates are positive but not all criteria hold:

    HOLO_RESPONSE_RESCUE_WEAK_NO_GO

Otherwise:

    HOLO_RESPONSE_RESCUE_NO_GO

A weak or negative result does not justify GPU collection of new
probability-level response probes.

A positive result only motivates a separately frozen next-stage test
with true local probability-level probes.
