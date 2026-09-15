# AROMA 2.0 CSA — Gate A Freeze v1

Status: FROZEN BEFORE FORMAL GATE-A GRADIENT COLLECTION

This document supplements `AROMA2_CSA_PROTOCOL_v1_1.md`.

The preceding engineering pilot inspected only:
- intervention-site identity,
- tensor dimensions,
- gradient finiteness,
- gradient nonzero status.

No eigenspectrum, E_r statistic, empirical-null statistic, or
rank-dependent Gate-A result was inspected before this freeze.

---

## 1. Source population

Gate A uses the exact N = 300 Proc-Count-Causal v1 examples
used for the original L18H13 directional mechanism characterization.

No target-domain examples are used to construct the CSA basis.

---

## 2. Model and intervention site

Base model:

`meta-llama/Llama-3.2-11B-Vision-Instruct`

Frozen model revision:

`9eb2daaa8597bf192a8b0e73f848f3a102794df5`

Layer:

`18`

Head:

`13`

Intervention location:

`model.model.language_model.layers[18].cross_attn.o_proj`

The L18H13 slice is:

- hidden size = 4096
- number of heads = 32
- head dimension = 128
- start = 1664
- end = 1792

---

## 3. Sample-level activation

Let

H_i in R^{T_i x 128}

denote the L18H13 head output immediately before `o_proj`.

The sample-level representation is fixed as mean pooling:

h_i = (1 / T_i) sum_t H_{i,t}.

For gradient extraction, the implementation must make this pooled
128-dimensional h_i itself the local autograd leaf.

The original token-level activation must be reconstructed exactly as

H_{i,t}' = (H_{i,t} - h_i) + h_i_leaf,

so that the hooked forward pass is numerically identical to baseline
while the returned gradient is directly

g_i = d mu_i / d h_i.

Post-hoc choice among mean-token, last-token, max-token, or other
token aggregation rules is not permitted for Gate A.

---

## 4. Cardinality statistic

Gate A uses the original directional-characterization numeral set:

V_N = {1,2,3,4,5,6,7,8,9,10}.

Each numeral must be verified as a single tokenizer token.

Let q_i(n) be the conditional probability over V_N obtained by
softmaxing the corresponding numeral logits.

The statistic is

mu_i = sum_{n in V_N} n q_i(n).

---

## 5. Sensitivity matrix

For every sample,

g_i = grad_{h_i} mu_i in R^{128}.

The primary sensitivity second-moment matrix is uncentered:

G = (1/N) sum_i g_i g_i^T.

No sample-mean subtraction is applied.

Eigenvalues are ordered

lambda_1 >= ... >= lambda_128 >= 0.

---

## 6. Frozen primary rank

The primary CSA rank is fixed before Gate-A analysis as

r = 4.

Other ranks may be reported descriptively after Gate A,
but they may not replace r = 4 for the primary Gate-A decision.

---

## 7. Primary concentration statistic

The primary statistic is

E4 = (lambda_1 + lambda_2 + lambda_3 + lambda_4)
     / trace(G).

---

## 8. Empirical finite-sample null

Number of null replicates:

B = 10000.

Random seed:

20260915.

For every null replicate b and every real sample i:

1. preserve the exact observed norm ||g_i||_2;
2. draw z_i ~ N(0, I_128);
3. normalize u_i = z_i / ||z_i||_2;
4. construct g_i^(b) = ||g_i||_2 u_i.

Thus every replicate exactly matches:

- N = 300,
- d = 128,
- the complete empirical per-sample norm distribution,

while destroying shared directional structure.

For each replicate compute G^(b) and E4^(b) identically to the
observed statistic.

---

## 9. Gate-A decision

Gate A is GO only if BOTH conditions hold:

1. observed E4 is strictly greater than the empirical 99th percentile
   of the 10,000-replicate null distribution;

2. observed E4 / median(null E4) >= 5.0.

Report additionally the empirical upper-tail p-value

p_emp = (1 + #{b : E4^(b) >= E4_obs}) / (B + 1).

If either primary condition fails:

Gate A = NO-GO.

No post-hoc change to rank, pooling rule, null construction,
threshold, or sample population may be used to rescue Gate A.

---

## 10. Consequence of Gate A

GO:
proceed to protocol-defined CSA intervention

h' = (I - U_4 U_4^T)h + alpha U_4 U_4^T h

and Gate B comparison against whole-head L18H13 actuation.

NO-GO:
do not develop CSA as the primary AROMA 2.0 actuator.
The prior 4096-dimensional cross-attention-output analysis remains
exploratory only.

