# AROMA 2.0 — Final MN-CSA v4 Untouched Confirmation Freeze v1

Status:

FROZEN BEFORE PROC-COUNT-CAUSAL v4 IS GENERATED OR EVALUATED.

## Purpose

This is the final untouched synthetic confirmation of the
completed Causal Subspace Actuation (CSA) development line.

No further CSA algorithm development is permitted using v4.

## Frozen algorithm

Actuator:

L18H13

Head dimension:

128

Subspace rank:

4

Subspace basis:

outputs/aroma2/csa_gate_a_v3/U4_primary.npy

For token-level head state:

H in R^(T x 128)

define:

h = mean_t H_t

p = U4 U4^T h

d = p / ||p||

Magnitude-Normalized CSA:

Delta h =
(alpha - 1)
||h||
d

H'_t =
H_t + Delta h.

No rank tuning, basis tuning, normalization tuning,
gain tuning, or formula change is permitted.

## Frozen policy

Use the exact already-frozen synthetic AROMA controller.

Actions:

{0, 1, 1.5, 2, 4}

No controller retraining.

No feature change.

No threshold change.

No action-set change.

No decision-rule change.

## Dataset

Name:

Proc-Count-Causal v4

N:

2000

Counts:

1 through 10

Conditions:

- row
- grid
- random_sparse
- dense
- distractors

Replicates per count-condition:

40

Total:

10 x 5 x 40 = 2000

## Rendering distribution

Reuse exactly the Proc-Count-Causal v3 scene-generation
functions.

Frozen geometry:

image width = 768

image height = 512

target class = red_circle

target radius = 24 pixels

distractor class = blue_square

distractor half-size = 22 pixels

distractors in distractor condition = 8

No rendering-family modification is permitted.

## v4 seed family

Base seed:

271828182

Base seed formula:

base_seed
+
count * 10000
+
condition_index * 100
+
replicate

Collision retry:

candidate_seed
=
base_seed_for_sample
+
attempt * 1_000_000_000

Attempt 0 is always used unless an objective collision is
detected.

Resampling is allowed ONLY for:

- seed collision with v1/v2/v3/v4
- exact pixel hash collision with v1/v2/v3/v4

Resampling based on model outputs is prohibited.

## Required independence

Before model inference is allowed:

- 2000 unique v4 sample IDs
- 2000 unique v4 seeds
- 2000 unique v4 image SHA256 hashes
- zero seed overlap with v1
- zero seed overlap with v2
- zero seed overlap with v3
- zero image-hash overlap with v1
- zero image-hash overlap with v2
- zero image-hash overlap with v3
- exactly 200 samples per count
- exactly 400 samples per condition
- exactly 40 samples per count-condition combination

Failure of any condition aborts the confirmation.

## Final behavioral comparison

The final evaluation will compare on all 2000 samples:

1. baseline
2. frozen whole-head AROMA
3. frozen MN-CSA AROMA

All three are evaluated in the same current model process.

## Primary MN-CSA endpoint

Primary endpoint:

MN-CSA post accuracy minus baseline accuracy.

Primary confirmation criterion:

- paired accuracy change > 0
- two-sided exact McNemar p < 0.05

## Whole-head comparison

MN-CSA versus whole-head is a pre-specified secondary paired
comparison.

Report:

- accuracy difference
- paired bootstrap 95% CI
- exact McNemar p
- repairs
- breaks
- net repairs

No post-hoc non-inferiority margin will be introduced.

## Prohibited after v4 generation begins

- rank search
- alternative basis
- alpha search
- action-set changes
- router retraining
- feature changes
- utility-threshold changes
- normalization changes
- subgroup-dependent policy changes
- sample removal based on model outcome
- sample replacement based on model outcome
- primary endpoint changes

v4 is confirmation-only.
