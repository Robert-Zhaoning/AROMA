# AROMA 2.0 CSA — Gate A Freeze v2

Status: FINAL FROZEN GATE-A PROTOCOL BEFORE FORMAL GRADIENT COLLECTION

This document supersedes `AROMA2_CSA_GATEA_FREEZE_v1.md`
before any formal Gate-A gradient matrix, eigenspectrum,
E4 statistic, or empirical-null result was observed.

The revision was motivated by a pre-run statistical review:
the uncentered second moment contains both common-direction
energy and centered covariance structure. Gate A v2 therefore
preserves the original uncentered primary test and adds a
centered companion test plus an explicit mean-direction
decomposition.

No previously frozen rank, null size, random seed, or
success threshold is changed.

---

## 1. Source population

Exact N = 300 Proc-Count-Causal v1 examples used in the
original L18H13 directional characterization.

The alpha = 1.0 rows of:

`outputs/proc_count_causal_v1/router/l18h13_gain_all300/l18h13_gain_all300_results.csv`

define the population.

---

## 2. Base model

Model:

`meta-llama/Llama-3.2-11B-Vision-Instruct`

Revision:

`9eb2daaa8597bf192a8b0e73f848f3a102794df5`

Canonical input/scoring behavior is imported from:

`scripts/run_l18h13_gain_all300.py`

---

## 3. Intervention coordinate

Layer: 18

Head: 13

Location:

`model.model.language_model.layers[18].cross_attn.o_proj`

Hidden size: 4096

Number of heads: 32

Head dimension: 128

L18H13 slice:

[1664:1792]

---

## 4. Sample-level representation

For the pre-o_proj L18H13 output

H_i in R^(T_i x 128),

define

h_i = (1/T_i) sum_t H_{i,t}.

The implementation exposes h_i itself as the local
autograd coordinate while preserving the original
token-level residuals:

H'_{i,t} = (H_{i,t} - h_i) + h_i_leaf.

At the evaluation point:

h_i_leaf = h_i,

so the hooked forward pass must be exact identity.

---

## 5. Cardinality sensitivity

The numeral support is frozen as:

{1,2,3,4,5,6,7,8,9,10}.

The output statistic is the canonical conditional expected
numeral:

mu_i = sum_n n P(n | x_i, numeral support).

For each sample:

g_i = gradient_{h_i} mu_i in R^128.

---

## 6. A1 — total sensitivity concentration

Define the uncentered second moment:

S = (1/N) sum_i g_i g_i^T.

Let its eigenvalues satisfy:

lambda_1 >= ... >= lambda_128.

Primary rank:

r = 4.

Primary concentration statistic:

E4_total =
(lambda_1 + ... + lambda_4) / trace(S).

The primary CSA basis U4 is the top four eigenvectors of S.

---

## 7. Mean-direction decomposition

Define:

g_bar = (1/N) sum_i g_i.

Mean-direction share:

M =
||g_bar||^2 / trace(S).

Because:

trace(S)
=
trace(C) + ||g_bar||^2,

this quantity reports the fraction of total sensitivity
energy attributable to the shared mean direction.

---

## 8. A2 — residual geometry concentration

Define centered gradients:

g_i^c = g_i - g_bar.

Define:

C =
(1/N) sum_i g_i^c (g_i^c)^T.

The companion statistic is:

E4_centered =
sum of top-4 eigenvalues of C / trace(C).

A2 is diagnostic of low-dimensional structure remaining
after removing the common gradient direction.

---

## 9. Empirical finite-sample null

Number of replicates:

B = 10000.

Seed:

20260915.

For each replicate b and observed sample i:

1. Preserve exactly ||g_i||_2.
2. Draw z_i ~ N(0, I_128).
3. Normalize u_i = z_i / ||z_i||.
4. Set g_i^(b) = ||g_i|| u_i.

Thus every null replicate preserves:

- N = 300
- d = 128
- the complete observed sample-wise norm distribution

while removing cross-sample directional alignment.

A1 null:

compute the uncentered second moment of g_i^(b)
and its E4_total^(b).

A2 null:

within each replicate, subtract that replicate's own
sample mean before computing its centered covariance and
E4_centered^(b).

---

## 10. Frozen thresholds

For BOTH A1 and A2, a test passes only if:

1. observed E4 > empirical null 99th percentile

AND

2. observed E4 / median(null E4) >= 5.0.

Also report:

p_emp =
(1 + number of null values >= observed value) / (B + 1).

No rank search, threshold search, pooling search, or
alternate null construction may replace these primary
definitions after formal collection begins.

---

## 11. Gate-A decisions

### GATE_A_STRONG_GO

A1 passes AND A2 passes.

Interpretation:

The actuator contains both strong total directional
concentration and residual low-dimensional sensitivity
geometry.

### GATE_A_DIRECTIONAL_GO

A1 passes AND A2 fails.

Interpretation:

Sensitivity is strongly concentrated, but the evidence is
primarily consistent with a dominant shared direction rather
than rich centered low-dimensional geometry.

Proceed to Gate B using the primary U4 from the uncentered
second moment, but phrase the representation claim
conservatively.

### GATE_A_NO_GO

A1 fails.

Do not develop CSA as the primary AROMA 2.0 actuator.

---

## 12. Gate B

If A1 passes, Gate B compares:

whole-head:

h' = alpha h

against CSA:

h' =
(I - U4 U4^T) h
+
alpha U4 U4^T h.

Primary comparison:

- repairs
- breaks
- net gain

on both synthetic and natural domains.

---

## 13. Resume integrity

Any partial gradient collection must be bound to a
method fingerprint containing at minimum:

- SHA256 of Gate-A runner
- SHA256 of canonical runner
- SHA256 of this freeze document
- SHA256 of source CSV
- model revision
- layer
- head
- head slice
- pooling rule
- numeral support
- primary rank
- null replicate count
- null seed
- canonical prompt

Resume is forbidden when the stored fingerprint differs
from the current fingerprint.

