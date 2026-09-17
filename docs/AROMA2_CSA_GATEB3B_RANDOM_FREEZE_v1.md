# AROMA 2.0 — CSA Gate B3b Random-Subspace Specificity Freeze v1

Status:

FROZEN BEFORE RANDOM-SUBSPACE BEHAVIORAL RESULTS ARE OBSERVED.

## Scientific question

Does the learned Top-4 cardinality-sensitivity subspace
provide better finite control than typical arbitrary
four-dimensional directions inside L18H13?

## Population

Use exactly the same 653 frozen non-NOOP Proc-Count-Causal
v3 examples used in Gate B2 and Gate B3a.

Frozen selected actions:

alpha=0   : 24
alpha=1.5 : 32
alpha=2   : 63
alpha=4   : 534

No action, router, rank, or threshold is changed.

## Learned Top-4

Use exactly:

outputs/aroma2/csa_gate_a_v3/U4_primary.npy

Rank:

4

## Random controls

Number of random subspaces:

K = 20

Master seed:

20260915

For each control k:

1. Draw Z_k in R^(128 x 4) with iid N(0,1) entries.
2. Compute reduced QR:
   Z_k = Q_k R_k.
3. Use span(Q_k) as the random four-dimensional subspace.

No rejection sampling is allowed.

No random subspace is removed because of overlap with Top-4,
poor performance, unusual projection norm, or any behavioral
outcome.

The twenty bases are generated and saved before behavioral
inference begins.

## Intervention

Top-4 and every random control use the exact same
magnitude-normalized actuation:

h = mean_t H_t

p_U = U U^T h

d_U = p_U / ||p_U||

Delta h =
(alpha - 1) ||h|| d_U

H'_t =
H_t + Delta h.

Thus all subspaces have:

- rank = 4
- identical frozen alpha
- identical pooled displacement magnitude
- identical token-residual preservation

and differ only in intervention direction.

## Same-run evaluation

For all 653 examples recompute, in the same current model
process:

- current baseline
- current Top-4 MN-CSA
- all 20 random MN-CSA predictions

The Top-4 prediction must also match the previously completed
Gate-B3a Top-4 prediction for every example.

## Primary specificity statistic

Let:

C_top(i)

be Top-4 correctness on sample i.

Let:

C_rand_bar(i)
=
(1/K) sum_k C_rand_k(i).

Define:

D_i =
C_top(i) - C_rand_bar(i).

Primary effect:

Delta_average_random
=
mean_i D_i.

Report a paired bootstrap 95% confidence interval over the
653 evaluation examples.

Bootstrap replicates:

20000

Bootstrap seed:

20260915

Primary positive specificity criterion:

lower 95% bootstrap bound > 0.

No minimum effect-size multiplier is imposed.

## Random-subspace rank statistic

Also report the twenty random-subspace accuracies.

Define:

p_rank
=
(1 + number of random subspaces with accuracy >= Top-4)
/
(K + 1).

With K=20, the minimum possible p_rank is:

1/21 = 0.047619...

This is reported as an empirical random-subspace rank test.

## Secondary outcomes

Report:

- random accuracy mean
- random accuracy median
- random accuracy standard deviation
- random minimum
- random maximum
- Top-4 minus random mean
- Top-4 minus random maximum
- each random basis's overlap with Top-4
- each random basis's median projection ratio
- repairs / breaks / net relative to current baseline
- results by frozen action
- results by procedural condition

## Interpretation

RANDOM_SPECIFICITY_POSITIVE:

paired bootstrap lower 95% bound for
Top-4 minus average-random > 0.

RANDOM_SPECIFICITY_NOT_ESTABLISHED:

the paired interval includes zero.

RANDOM_CONTROL_OUTPERFORMS_TOP4:

paired bootstrap upper bound < 0.

The empirical rank statistic is reported separately and does
not replace the paired primary test.

No random basis, rank, alpha, normalization rule, or router
may be changed after this run starts.
