# Natural-Domain Unified Geometry Router v1

## Status

**Frozen before Natural-UGR Stage-2 candidate outcomes.**

Historical UGR source:

- ref: `unified-geometry-router-v1`
- commit: `d2b5c2bd1b2de022302ca648bf423660c93a33b3`

Natural development source:

- `outputs/tallyqa_natural_confirmation_v2/final_frozen_controller/tallyqa_final_results.csv`
- full source N = 4000
- realized upward Stage-2 N = 292
- complex = 222
- simple = 70
- alpha 1.5 = 32
- alpha 2.0 = 122
- alpha 4.0 = 138

## Scientific question

Does example-conditional selection among identity, MN, SA, and
WHOLE geometry provide positive utility on natural TallyQA counting
examples beyond the strongest development-selected geometry-restricted
router?

## Candidate family

For each Stage-2 sample:

`{IDENTITY} union ({MN, SA, WHOLE} x G)`

where

`G = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0]`

Total = 43 candidates per sample.

Candidate intervention semantics are inherited exactly from historical
UGR-v1 at `d2b5c2bd1b2de022302ca648bf423660c93a33b3`.

## Sample representation

Exactly 47 ground-truth-free sample features:

- 39 frozen natural Proposal Controller features
- selected_alpha
- selected_score
- six mechanistic summaries

Candidate-conditioned design:

- 47 sample features
- 9 candidate channels
- 47 x 9 interactions
- 479 total design dimensions

## Router

All three router families use:

- StandardScaler
- L2 LogisticRegression
- C = 1.0
- solver = lbfgs
- max_iter = 10000
- target = candidate correctness

Families:

1. low-rank restricted: IDENTITY + MN + SA
2. whole restricted: IDENTITY + WHOLE
3. unified: IDENTITY + MN + SA + WHOLE

## Development evaluation

Grouped 5-fold OOF evaluation.

- seed = 20260920
- all candidate rows from one sample remain in the same fold
- selection = highest predicted probability of candidate correctness

Exact tie-break:

1. IDENTITY
2. smaller gain
3. MN
4. SA
5. WHOLE

Restricted comparator selection:

1. higher OOF exact-count accuracy
2. fewer OOF breaks
3. smaller mean selected nonzero gain
4. low-rank restricted

## Fresh natural final

The final cohort is not generated at F1.

It may be generated only after:

- all development candidate outcomes are frozen
- 47-feature archive is frozen
- all three final routers are fitted and frozen
- restricted comparator identity is frozen
- final analysis implementation is frozen

Target final cohort:

- total N = 8000
- simple N = 4000
- complex N = 4000
- one question per image
- selection seed = 20260926

Required freshness:

- zero question-ID overlap with all pre-F1 formal TallyQA cohorts
- zero image-ID overlap
- zero pixel-SHA256 overlap

The realized final Stage-2 population is whatever subset receives
`selected_alpha > 1` from the unchanged frozen natural Stage-1
controller. There is no upward top-up or post-Stage-1 resampling.

## Primary endpoint

Population:

`fresh final samples with selected_alpha > 1`

Comparison:

`unified router - development-selected strongest restricted router`

Endpoint:

paired exact-count accuracy difference in percentage points.

Statistics:

- exact paired McNemar test, two-sided
- 20,000 paired bootstrap replicates
- bootstrap seed = 20260922
- 95% percentile confidence interval

## Evidence interpretation

**Statistically supported natural transfer**

- effect > 0
- bootstrap CI lower bound > 0
- exact McNemar p < 0.05

**Positive but uncertain natural transfer**

- effect > 0
- unified-only > restricted-only
- one or both paired uncertainty criteria do not satisfy the
  conventional threshold

**No positive transfer**

- effect <= 0

No geometry is required to be universally superior.

## Frozen after F1

After this protocol commit, the following may not change based on
Natural-UGR candidate outcomes:

- geometry set
- gain grid
- 47-feature representation
- candidate encoding
- router model class / C
- CV folds / seed
- tie-breaking
- comparator-selection rule
- primary endpoint
- statistical tests
- target final cohort size
