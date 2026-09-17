# AROMA

## Adaptive Routing of Mechanistic Actuators with Causal Subspace Control for Vision-Language Counting

**AROMA** is a mechanistic-control framework for studying and repairing
vision-language counting errors in frozen multimodal language models.

Rather than treating counting repair as a single prompting problem, AROMA
separates three questions:

```text
Actuator Capacity
       ↓
Actuation Geometry
       ↓
Routing Policy
````

- **Actuator capacity:** where can counting behavior be causally changed?
- **Actuation geometry:** along which internal directions should the intervention
  be applied, and with what finite magnitude?
- **Routing policy:** when is an intervention useful for a particular example?

The current experiments use:

```text
meta-llama/Llama-3.2-11B-Vision-Instruct
revision:
9eb2daaa8597bf192a8b0e73f848f3a102794df5
```

The repository preserves development experiments, frozen protocols, negative
results, confirmation studies, and reproducibility artifacts.

---

# Key Results

| StudyResult                                       |                            |
| ------------------------------------------------- | -------------------------- |
| L18H13 attenuation (`α = 0.5`)                    | **91.33% downward shifts** |
| L18H13 amplification (`α = 1.5`)                  | **90.67% upward shifts**   |
| Strict `0.5 < 1 < 1.5` ordering                   | **88.67%**                 |
| Frozen Proc-Count-Causal v3 controller            | **+8.10 pp**               |
| Frozen zero-shot TallyQA transfer                 | **−2.975 pp**              |
| TallyQA natural action oracle                     | **+6.725 pp**              |
| Frozen K=500 TallyQA confirmation                 | **+2.125 pp**              |
| HoloCount transferred router                      | **−0.0403 pp**             |
| HoloCount action oracle                           | **+12.2581 pp**            |
| Top-4 total cardinality-sensitivity second moment | **89.82%**                 |
| Median activation energy in Top-4 subspace        | **9.52%**                  |
| Untouched v4 whole-head AROMA                     | **+7.00 pp**               |
| Untouched v4 MN-CSA AROMA                         | **+8.00 pp**               |
| MN-CSA vs. whole-head on v4                       | **+1.00 pp, p = 0.0078**   |

These numbers come from different evidence classes. Development results,
post-hoc diagnostics, frozen confirmations, and external stress tests are kept
separate throughout the repository.

---

# 1. Actuator Capacity

AROMA identifies **Layer 18, Head 13 (L18H13)** as a bidirectional
cardinality-sensitive cross-attention actuator.

On the original 300-example directional characterization:

```text
alpha = 0.5
274 / 300 = 91.33% downward shift

alpha = 1.5
272 / 300 = 90.67% upward shift

strict ordering:
266 / 300 = 88.67%

down/up response-magnitude Spearman:
rho = 0.951442127
```

The identity intervention reproduces the baseline.

The original directional characterization used numeral support:

```text
1 ... 10
```

Later controller and CSA stages used a separately audited expanded numeral
proxy:

```text
0 ... 15
```

These stages are intentionally kept distinct.

---

# 2. Adaptive Routing

A fixed intervention strength repairs some examples but breaks others.

AROMA therefore uses a lightweight per-example utility controller with:

```text
39 ground-truth-free features

per-action predictor:
StandardScaler -> Ridge

ridge alpha:
0.01

action set:
{0, 1, 1.5, 2, 4}

decision threshold:
0.1
```

## Frozen procedural confirmation

On Proc-Count-Causal v3:

```text
baseline:
49.95%

post-controller:
58.05%

gain:
+8.10 pp

repairs:
173

breaks:
11
```

This confirms adaptive repair within the procedural distribution.

---

# 3. Natural Transfer Boundary

The frozen synthetic controller does **not** transfer zero-shot to TallyQA.

```text
baseline:
64.85%

post-controller:
61.875%

change:
-2.975 pp
```

This failure is retained as a primary scientific result.

However, the natural action oracle reaches:

```text
71.575%

oracle gain:
+6.725 pp
```

Therefore actuator capacity remains present even though the learned deployment
policy fails.

This motivates one of AROMA's main distinctions:

> **An actuator is not a policy.**

---

# 4. Natural-Domain Recalibration

The actuator, feature family, action set, Ridge model class, regularization,
and decision threshold were retained while the utility mapping was
re-estimated on natural examples.

A controller calibrated with `K = 500` natural examples was frozen before a
new confirmation evaluation.

On the new 4,000-example TallyQA confirmation sample:

```text
baseline:
64.925%

post-controller:
67.050%

gain:
+2.125 pp

95% CI:
[+1.50, +2.775] pp

repairs:
130

breaks:
45

exact McNemar:
p = 9.176009e-11
```

The confirmation sample is disjoint from the earlier TallyQA sample in
question IDs, image paths, and image IDs, while remaining within the same
benchmark distribution.

This is **not** claimed as cross-dataset generalization.

---

# 5. External HoloCount Stress Test

The TallyQA-adapted controller was evaluated on HoloCount as an independent
natural-domain stress test.

```text
baseline:
46.6935%

transferred router:
46.6532%

change:
-0.0403 pp
```

The transferred router is therefore effectively null.

However, the same actuator/action family retains substantial conditional
headroom:

```text
action oracle:
58.9516%

oracle gain:
+12.2581 pp
```

The same 39-feature linear Ridge utility family did not recover this headroom.

The supported conclusion is therefore narrow:

> L18H13 retains substantial repair capacity on HoloCount, but that capacity is
> not recoverable by the existing 39-dimensional numeral-geometry
> representation with the same linear utility family.

---

# 6. Causal Subspace Actuation

Whole-head AROMA scales all 128 dimensions of L18H13.

CSA asks whether cardinality-relevant sensitivity is concentrated in a
lower-dimensional geometry.

For pooled head state `h(x)`:

```text
g(x) = grad_h mu(x)

G = (1/N) sum_i g_i g_i^T
```

where `mu(x)` is the expected numeral.

The primary rank-4 eigenspace explains:

```text
uncentered total second moment:
89.82%

centered:
86.78%
```

The rank-4 spectrum exceeds the 10,000-sample norm-matched random null.

The pre-specified Gate-A rule additionally required at least 5× enrichment over
the null median.

Observed enrichment was:

```text
uncentered:
4.2486x

centered:
4.1031x
```

Therefore the formal frozen result remains:

```text
GATE_A_NO_GO
```

This means **NO-GO for promotion under the frozen criterion**. The threshold
was not changed after observing the result.

---

# 7. Sensitivity Is Not Activation Energy

Naively applying the same intervention coefficient only inside the learned
rank-4 subspace performs poorly.

A strength audit showed:

```text
Top-4 sensitivity second moment:
89.82%

median Top-4 activation energy:
9.52%
```

The learned subspace therefore contains most local cardinality sensitivity but
only a small fraction of pooled activation energy.

This motivates the second central distinction:

> **Sensitivity is not activation energy.**

---

# 8. Magnitude-Normalized CSA

Let

```text
p = U U^T h

d_U = p / ||p||
```

AROMA uses:

```text
Delta h_MN = (alpha - 1) ||h|| d_U
```

so that:

```text
||Delta h_MN||
=
|alpha - 1| ||h||
```

The intervention remains inside the learned sensitivity subspace while
matching the pooled displacement magnitude of whole-head scaling.

## Development evidence

On the frozen v3 intervention subset:

```text
whole-head:
50.995%

naive rank-4 CSA:
33.23%

MN-CSA:
52.53%
```

Rank-matched specificity controls:

```text
Bottom-4:
26.80%

20 random rank-4 controls:
mean = 27.18%
best = 30.78%

random controls matching Top-4:
0 / 20
```

---

# 9. Final Untouched v4 Confirmation

Proc-Count-Causal v4 was generated only after the CSA rank, basis, formula,
action policy, and endpoint were closed.

The audit verified:

```text
N = 2000

unique seeds = 2000
unique rendered-image hashes = 2000

seed overlap with v1/v2/v3 = 0
rendered-image-hash overlap with v1/v2/v3 = 0
```

Final results:

| MethodAccuracyGain vs. baseline |            |              |
| ------------------------------- | ---------- | ------------ |
| Baseline                        | 50.25%     | —            |
| Whole-head AROMA                | 57.25%     | +7.00 pp     |
| MN-CSA AROMA                    | **58.25%** | **+8.00 pp** |

MN-CSA versus whole-head:

```text
difference:
+1.00 pp

95% CI:
[+0.3, +1.7] pp

exact McNemar:
p = 0.0077877
```

No rank, basis, formula, action policy, or condition-specific tuning was
performed on v4 after observing these results.

---

# Scientific Scope

AROMA currently supports evidence for:

```text
causal actuator discovery
        ↓
bidirectional cardinality control
        ↓
adaptive per-example routing
        ↓
low-rank sensitivity geometry
        ↓
magnitude-normalized subspace actuation
        ↓
untouched procedural confirmation
```

The current evidence does **not** establish:

```text
universal counting repair

successful zero-shot natural routing

cross-model mechanistic generalization

HoloCount router transfer

universal utility learnability from the 39-feature representation

natural-image confirmation of MN-CSA
```

These boundaries are intentional parts of the research record.

---

# Repository Guide

```text
configs/      frozen experiment and controller configuration
data/         local/generated data; new data are ignored by default
docs/         protocols, freeze records, result records, and provenance
environment/  environment metadata
external/     third-party resources; ignored by default
manifests/    dataset and experiment manifests
outputs/      selected tracked research artifacts plus local outputs
scripts/      canonical experiment and analysis runners
src/          reusable AROMA package code
tests/        lightweight automated tests
archive/      historical and superseded research code
```

Start here:

- [`docs/README.md`](https://chatgpt.com/g/g-p-69f2ff8daf748191a12f33965728c1be/c/docs/README.md) — documentation map
- [`docs/PROJECT_STATUS.md`](https://chatgpt.com/g/g-p-69f2ff8daf748191a12f33965728c1be/c/docs/PROJECT_STATUS.md) — current scientific status
- [`docs/EXPERIMENT_LEDGER.md`](https://chatgpt.com/g/g-p-69f2ff8daf748191a12f33965728c1be/c/docs/EXPERIMENT_LEDGER.md) — experiment history
- [`docs/RESULTS_CANONICAL.md`](https://chatgpt.com/g/g-p-69f2ff8daf748191a12f33965728c1be/c/docs/RESULTS_CANONICAL.md) — canonical registry
- [`docs/ARTIFACT_MANIFEST.md`](https://chatgpt.com/g/g-p-69f2ff8daf748191a12f33965728c1be/c/docs/ARTIFACT_MANIFEST.md) — artifact inventory
- [`docs/REPRODUCIBILITY.md`](https://chatgpt.com/g/g-p-69f2ff8daf748191a12f33965728c1be/c/docs/REPRODUCIBILITY.md) — reproduction notes
- [`docs/AROMA2_CSA_PROTOCOL_v1_1.md`](https://chatgpt.com/g/g-p-69f2ff8daf748191a12f33965728c1be/c/docs/AROMA2_CSA_PROTOCOL_v1_1.md) — CSA protocol
- [`docs/AROMA2_CSA_V4_FINAL_RESULT_v1.md`](https://chatgpt.com/g/g-p-69f2ff8daf748191a12f33965728c1be/c/docs/AROMA2_CSA_V4_FINAL_RESULT_v1.md) — final v4 confirmation

---

# Installation

The lightweight package can be installed with:

```bash
python -m pip install -e .
```

Model-facing experiments additionally require the frozen research environment
documented in:

```text
docs/ENVIRONMENT.md
```

Large model weights and third-party datasets are not distributed through this
repository.

---

# Repository Integrity

Run:

```bash
make check
```

This performs repository-integrity and source-compilation checks without
loading the 11B vision-language model.

Full model experiments require the appropriate GPU environment and dataset
access.

---

# Research Status

The current single-model scientific milestone is frozen.

Further work should be treated as a new experimental phase rather than as
post-hoc tuning of the completed Llama-3.2-Vision experiments.

The highest-priority open direction is cross-model mechanistic replication.

See:

```text
docs/PROJECT_STATUS.md
```

for the exact boundary between closed and open research questions.

---

# Citation

A manuscript is currently in preparation.

Repository citation metadata are provided in:

```text
CITATION.cff
```

Please use the accompanying manuscript citation once the final author list and
venue information are available.

---

# License

A public reuse license has **not yet been finalized**.

Until a license is selected, please contact the authors before redistributing
or reusing substantial portions of the code.
