# AROMA 2.0 Causal Subspace Actuation (CSA) Protocol v1.1

## Objective

This protocol defines the pre-registered methodology for replacing
whole-head intervention with a lower-dimensional causal subspace actuator.

No model results are used to determine protocol choices.

---

# 1. Intervention Site

Model:

meta-llama/Llama-3.2-11B-Vision-Instruct

Layer:

18

Attention module:

cross_attn

Head:

13


The canonical activation site is the input of:


model.model.language_model.layers[18].cross_attn.o_proj



The original whole-head intervention is:

\[
h \rightarrow \alpha h
\]


where h denotes the L18H13 head slice.

---

# 2. Activation Representation

The head output before output projection is:

\[
H^{18,13}(x)\in R^{T\times d_h}
\]


where:

- T is the query token dimension
- d_h is the head dimension


The sample-level representation is defined by mean pooling:

\[
h(x)=\frac{1}{T}\sum_t H_t^{18,13}(x)
\]


---

# 3. Causal Sensitivity Signal

The model output statistic is:

\[
\mu(x)=\sum_n nP(n|x)
\]


The local causal sensitivity vector is:

\[
g(x)=\nabla_h \mu(x)
\]


The sensitivity covariance is:

\[
G=
\frac1N\sum_i g_i g_i^T
\]


---

# 4. Subspace Construction

The causal subspace is defined by the top-r eigenvectors:

\[
U_r = eig_r(G)
\]


The null comparison must use empirical random rotations
matched by:

- sample number
- vector dimension
- vector norm distribution


---

# 5. CSA Intervention

Whole-head:

\[
h'=\alpha h
\]


CSA:

\[
h'=(I-UU^T)h+\alpha UU^Th
\]


Only the learned causal subspace is scaled.

The orthogonal component remains unchanged.

---

# 6. Evaluation Gates


## Gate A: Causal concentration

The learned subspace must exceed empirical random-subspace null distribution.


## Gate B: Control preservation

CSA must be compared against whole-head intervention:

- repair count
- break count
- net gain


Both synthetic and natural evaluation domains must be considered.


## Gate C: Collateral preservation

CSA must preserve non-counting behavior better than whole-head intervention.


---

# 7. Reproducibility Requirements

The following must be frozen before GPU experiments:

- model revision
- layer
- head index
- activation location
- pooling rule
- subspace rank
- null model
- evaluation metrics

