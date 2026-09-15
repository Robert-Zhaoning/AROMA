# AROMA 2.0 — CSA Gate B3a Bottom-Subspace Specificity Freeze v1

Status:

FROZEN BEFORE BOTTOM-SUBSPACE BEHAVIORAL RESULTS ARE OBSERVED.

## Motivation

Gate A discovered strong low-rank cardinality sensitivity
inside L18H13.

Top-4 sensitivity concentration:

total E4 = 0.8982293582

centered E4 = 0.8677753028

Gate B2 then showed that magnitude-normalized Top-4 CSA
recovered the full behavioral effectiveness of whole-head
actuation:

Whole-head:
333 / 653 = 50.995%

Same-alpha CSA:
217 / 653 = 33.231%

Magnitude-normalized Top-4 CSA:
343 / 653 = 52.527%

MN Top-4 vs whole:
+1.531 pp
95% CI [-0.766, +3.828]

MN Top-4 vs same-alpha CSA:
+19.296 pp
95% CI [+16.08, +22.665]

The next question is whether this behavioral recovery is
specific to the learned high-sensitivity subspace.

## Population

Use exactly the same 653 frozen non-NOOP Proc-Count-Causal
v3 examples used in Gate B2.

Frozen selected actions:

alpha=0   : 24
alpha=1.5 : 32
alpha=2   : 63
alpha=4   : 534

No action is changed.

No router is retrained.

## Top-4 subspace

Use the exact Gate-A primary basis:

outputs/aroma2/csa_gate_a_v3/U4_primary.npy

This consists of the four highest-eigenvalue directions of
the uncentered sensitivity second moment.

## Bottom-4 control subspace

Reconstruct the eigendecomposition of:

outputs/aroma2/csa_gate_a_v3/second_moment_uncentered.npy

Let:

lambda_1 >= ... >= lambda_128.

The Bottom-4 control consists of the four eigenvectors with
the smallest eigenvalues:

lambda_125 ... lambda_128.

The reconstructed Top-4 projector must numerically match the
saved U4_primary projector before Bottom-4 evaluation is
allowed.

No bottom-rank search is permitted.

## Intervention

Both Top-4 and Bottom-4 use EXACTLY the same
magnitude-normalized intervention.

For subspace U:

h = mean_t H_t

p_U = U U^T h

d_U = p_U / ||p_U||

Delta h =
(alpha - 1)
||h||
d_U

H'_t =
H_t + Delta h.

Therefore both interventions have:

- rank = 4
- identical selected alpha
- identical pooled displacement magnitude
- identical token-residual preservation

and differ only in the subspace direction.

## Primary comparison

Top-4 MN-CSA versus Bottom-4 MN-CSA.

Report:

- Top-4 accuracy
- Bottom-4 accuracy
- paired accuracy difference
- paired bootstrap 95% CI
- exact McNemar p-value

Bootstrap:

20000 replicates

Seed:

20260915

## Secondary outcomes

Report:

- results by selected alpha
- results by procedural condition
- Top-4 and Bottom-4 repair/break/net values
- Top-4 replay agreement with the previously completed
  Gate-B2 Top-4 predictions
- projection ratios for each subspace

## Pre-specified interpretation

TOP4_CLEAR_SPECIFICITY:

paired bootstrap lower 95% bound for
Top-4 minus Bottom-4 > 0.

TOP4_NO_CLEAR_SPECIFICITY:

95% interval includes zero.

TOP4_WORSE_THAN_BOTTOM4:

paired bootstrap upper 95% bound < 0.

No rank change, action change, controller change, gain
change, or alternative Bottom-4 definition may be introduced
after the run begins.

If Top-4 shows clear specificity over Bottom-4, proceed to
Gate B3b with pre-frozen random rank-4 subspaces.
