# AROMA Generalization Extension Protocol

## Status

This document freezes the experimental protocol for the post-v1
generalization extension of AROMA:

**Adaptive Routing of Mechanistic Actuators for Vision-Language Counting**

The protocol is created before any HoloCount evaluation and before any
Qwen2.5-VL causal-screen result is observed.

The purpose of this extension is not to replace or rewrite the completed
AROMA v1 study. The completed v1 study remains frozen at:

```text
tag:
aroma-experiments-v1-complete

commit:
ae6e08cdef84a512426268fa0e508dfe53237f14
```

Publication packaging before this extension is frozen at:

```text
3d96180fbc39f777fa4339466798382e1eb145cc
```

The new experiments provide additional evidence concerning:

1. cross-dataset natural generalization; and
2. cross-architecture replication of mechanistic cardinality control.

Existing v1 experimental artifacts must not be modified.

---

# 1. Scientific Questions

The extension addresses two primary questions.

## RQ1 — Cross-dataset natural generalization

Does the already-frozen TallyQA-adapted AROMA controller provide useful
counting repair on a genuinely independent natural counting benchmark?

The primary external benchmark will be HoloCount.

The controller, actuator, feature representation, action set, Ridge
parameters, and routing threshold must remain frozen before the first
HoloCount controlled evaluation.

## RQ2 — Cross-architecture mechanistic replication

Can a bidirectional cardinality-sensitive causal actuator be identified in
a substantially different open VLM architecture?

The primary second VLM will be:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

The objective is first to replicate the mechanistic actuator phenomenon.

A full Qwen AROMA utility router will be attempted only if a prespecified
actuator-utility gate is satisfied.

---

# 2. Existing Frozen AROMA Components

The original AROMA study remains immutable.

The frozen natural controller is:

```text
outputs/phase2_natural_controller_k500/final_frozen_controller/
aroma_natural_utility_controller_k500.joblib
```

Expected SHA256:

```text
d9d8ba3a24814bf71b3f4039d1effa864b57754a00539f1260a1d077c03449f5
```

The existing actuator is:

```text
L18H13
```

in:

```text
meta-llama/Llama-3.2-11B-Vision-Instruct
```

The frozen deployable action set is:

```text
{0, 1, 1.5, 2, 4}
```

where alpha = 1 is the identity / NoOp action.

The frozen routing threshold is:

```text
tau = 0.1
```

The existing natural router uses the same 39-dimensional GT-free feature
family documented in the AROMA v1 repository.

None of these components may be modified for the primary HoloCount
cross-dataset experiment.

---

# 3. Evidence Classes

All extension evidence must be classified as one of:

1. PROTOCOL
2. EXPLORATORY ENGINEERING
3. DEVELOPMENT
4. FROZEN CONFIRMATION
5. POST-HOC FORENSIC
6. SUPPORTING ABLATION

No exploratory or post-hoc result may later be described as prespecified
confirmation.

---

# 4. Repository Isolation

All extension outputs must live under:

```text
outputs/generalization_extension/
```

with primary subdirectories:

```text
outputs/generalization_extension/holocount/
outputs/generalization_extension/qwen25vl/
```

New scripts should use descriptive names such as:

```text
scripts/build_holocount_manifest.py
scripts/run_holocount_frozen_external.py
scripts/run_holocount_action_oracle.py

scripts/audit_qwen25vl_intervention.py
scripts/run_qwen25vl_coarse_causal_scan.py
scripts/run_qwen25vl_candidate_validation.py
scripts/run_qwen25vl_directional_confirmation.py
scripts/run_qwen25vl_dose_response.py
```

Existing AROMA v1 output directories must not be overwritten.

---

# 5. Global No-Peeking Rules

The following rules apply to both extension tracks.

1. Dataset manifests must be frozen before controlled evaluation.
2. Ground truth may be used only:
   - to compute evaluation metrics;
   - to create development utility labels when the phase is explicitly
     classified as development;
   - to construct post-hoc action oracles after a frozen evaluation is
     complete.
3. Ground truth must never enter the deployed feature vector.
4. Hyperparameters may not be modified after viewing a frozen confirmation
   result and then reevaluated on the same confirmation data as if the new
   result were confirmatory.
5. Failed frozen results must be preserved.
6. Resume mechanisms may recover technical interruption only. They may not
   support outcome-dependent selective reruns.

---

# 6. Track H — HoloCount External Natural Evaluation

## H1. Dataset acquisition and provenance audit

Before any model inference:

- record the exact dataset source;
- record dataset revision / commit where available;
- inspect the dataset schema only;
- record available official splits;
- record the number of examples;
- record answer ranges;
- record category / subset fields;
- record image availability;
- create a deterministic manifest;
- compute manifest SHA256;
- compute image identity information where feasible.

No model accuracy or intervention result may be inspected before this
manifest is frozen.

The primary HoloCount evaluation set will be the complete public evaluation
split selected from the official dataset structure.

The exact official split name will be recorded after metadata/schema
inspection and before inference.

No outcome-dependent example filtering is allowed.

---

## H2. HoloCount baseline audit

The first inference pass is baseline-only:

```text
alpha = 1
```

This phase verifies:

- prompt compatibility;
- output parser compatibility;
- numeric-answer format;
- answer range;
- baseline accuracy;
- subset distribution.

The baseline pass must not be used to modify the frozen K=500 controller.

If the benchmark contains target counts outside the controller's numeral
proxy range, the full benchmark remains the primary evaluation.

A count-range-restricted analysis may be reported only as a prespecified
secondary analysis.

---

## H3. Primary cross-dataset experiment

The primary cross-dataset test applies the already-frozen K=500
TallyQA-adapted AROMA controller to the frozen HoloCount evaluation
manifest.

The following remain unchanged:

- base VLM;
- L18H13 actuator;
- 39-feature family;
- StandardScaler parameters;
- Ridge coefficients;
- action set;
- threshold tau = 0.1;
- inference prompt;
- answer parser, except only dataset-format adaptation required to read
  HoloCount questions/images.

Primary metric:

```text
paired exact-count accuracy change
```

Secondary metrics:

- baseline accuracy;
- post-controller accuracy;
- repairs;
- breaks;
- net repairs;
- intervention rate;
- paired 95% confidence interval;
- exact McNemar test;
- subset/category breakdowns.

---

## H4. HoloCount primary success criterion

A successful cross-dataset result requires:

```text
post_accuracy > baseline_accuracy
```

AND:

```text
exact McNemar p < 0.05
```

The paired bootstrap confidence interval will also be reported.

If the primary result satisfies the above rule, it may support a
cross-dataset natural-generalization claim.

If it does not, the result must be retained as a frozen negative result.

---

## H5. If frozen HoloCount transfer fails

Only after the primary frozen evaluation has been archived may the
following post-hoc forensic analyses be performed:

1. fixed-action evaluation for:

```text
alpha in {0, 1, 1.5, 2, 4}
```

2. per-example action oracle;
3. error-direction analysis;
4. subset-specific repairability;
5. controller-score / realized-utility alignment.

These analyses are POST-HOC FORENSIC evidence.

They may not retroactively alter the primary frozen result.

---

## H6. HoloCount stopping rule

If the frozen controller fails and the post-hoc action oracle provides:

```text
<= +1.0 percentage point
```

oracle headroom over baseline, no HoloCount-specific router adaptation will
be pursued.

If oracle headroom is larger than +1.0 pp, adaptation may be considered as
a separate post-hoc development project, but it will not be treated as
independent confirmation on the same HoloCount evaluation data.

---

# 7. Track Q — Qwen2.5-VL Cross-Architecture Replication

## Q1. Model freeze

Primary second model:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Before causal screening:

- record exact Hugging Face revision;
- record configuration;
- record attention implementation;
- record hidden size;
- record number of decoder layers;
- record number of attention heads;
- record tokenizer revision;
- record processor revision.

The model revision must be frozen before any head-screening outcome is
inspected.

---

## Q2. Baseline compatibility audit

Before screening heads, verify:

1. deterministic image-question generation;
2. counting output compatibility;
3. numeral-probability extraction;
4. tokenizer behavior for numerals 0 through 15;
5. visual-token handling;
6. exact attention-module location;
7. exact per-head slice boundaries;
8. intervention identity.

The identity requirement is:

```text
alpha = 1
```

must reproduce the unintervened computation to numerical tolerance with
zero generation-decision mismatch on the audit sample.

If identity equivalence fails, causal screening must not begin.

---

## Q3. Numeral proxy

The preferred proxy is:

```text
{0, 1, ..., 15}
```

for comparability with the later AROMA v1 controller pipeline.

If Qwen tokenization prevents direct single-token probability extraction,
a deterministic sequence-probability alternative may be implemented.

That alternative must be:

- documented;
- audited;
- frozen before head-screen results are inspected.

No proxy may be selected based on which one yields a stronger causal
effect.

---

## Q4. Qwen actuator definition

The Qwen actuator intervention is defined at an individual decoder
attention-head contribution immediately before the relevant attention
output projection.

For head j:

```text
[h_1; ...; h_j; ...; h_H]
```

is replaced by:

```text
[h_1; ...; alpha * h_j; ...; h_H]
```

before output projection.

The intervention must not modify model weights.

---

# 8. Qwen Discovery Schedule

## Q5. Coarse causal scan

Use a deterministic metadata-selected subset of the existing procedural
discovery data.

Target sample size:

```text
40 examples
```

Screen all accessible attention heads with:

```text
alpha = 0.5
alpha = 1.0
alpha = 1.5
```

Record for every head:

- mean expected-numeral shift;
- downward consistency at alpha = 0.5;
- upward consistency at alpha = 1.5;
- strict directional ordering;
- identity error at alpha = 1.

Select the top:

```text
24 candidate heads
```

using a frozen ranking criterion based on bidirectional directional effect.

---

## Q6. Candidate validation

Evaluate the 24 candidates on a deterministic fresh:

```text
100-example
```

procedural validation subset not used in Q5.

Retain the top:

```text
5 candidate heads
```

for final candidate comparison.

The selected final actuator must be determined before the frozen
directional confirmation set is evaluated.

---

## Q7. Frozen directional confirmation

After one final Qwen actuator is selected and frozen, evaluate it on:

```text
300 deterministic examples
```

drawn from procedural v3 and not used in Q5 or Q6.

Primary metrics:

```text
P(mu_0.5 < mu_1)
P(mu_1.5 > mu_1)
P(mu_0.5 < mu_1 < mu_1.5)
```

Also report:

- downward response magnitude;
- upward response magnitude;
- Spearman association between response magnitudes;
- alpha = 1 identity agreement.

---

# 9. Qwen Replication Criteria

## Strong replication

A selected Qwen actuator is classified as a strong replication if:

```text
downward consistency >= 80%
upward consistency   >= 80%
strict ordering       >= 70%
```

on the frozen 300-example confirmation.

## Minimum acceptable replication

A result may still support weaker cross-architecture replication if:

```text
downward consistency >= 65%
upward consistency   >= 65%
```

and both directional effects are statistically distinguishable from a
50% directional null.

Claims must match the achieved evidence tier.

---

# 10. Qwen Specificity Controls

After actuator freeze, perform supporting controls.

At minimum:

1. random attention heads;
2. neighboring heads where meaningful;
3. image-dependence control;
4. non-counting numeric-prompt control.

The purpose is to distinguish:

```text
visual/cardinality-sensitive intervention
```

from a generic numeral-logit steering mechanism.

If specificity is weak, the paper must use the more conservative phrase:

```text
cardinality / numeral actuator
```

rather than claiming a uniquely visual-counting-specific mechanism.

---

# 11. Qwen Dose Response

If Q7 reaches the minimum replication criterion, evaluate the frozen
actuator using:

```text
alpha in {0, 0.5, 1, 1.5, 2, 4}
```

Report the expected-numeral response curve.

A monotonic or near-monotonic dose response strengthens the
cross-architecture actuator claim.

---

# 12. Gate for a Full Qwen AROMA Router

A full Qwen utility-routing experiment will be attempted only after the
actuator has been frozen.

Evaluate the compressed deployable action set:

```text
{0, 1, 1.5, 2, 4}
```

on procedural development data and compute per-example oracle headroom.

Proceed to full router training only if BOTH hold:

```text
compressed-action oracle gain >= +3.0 pp
```

AND:

```text
at least 10% of baseline-wrong examples are repairable
```

If either criterion fails, stop the Qwen experiment after mechanistic
replication and report the absence of sufficient deployable headroom.

---

# 13. Full Qwen Router, If Gate Passes

If Section 12 passes:

- use the same conceptual 39-feature GT-free family;
- retain StandardScaler -> Ridge;
- retain lambda = 0.01;
- retain tau = 0.1;
- retain action set {0,1,1.5,2,4};
- train only on procedural development data.

The router must be frozen before a new confirmation set is evaluated.

Because procedural v3 is used for the frozen directional confirmation, a
new deterministic procedural confirmation dataset must be created for a
full Qwen router confirmation.

That new dataset must have:

- frozen generation protocol;
- new base seed;
- zero seed overlap with v1/v2/v3;
- zero image hash overlap with v1/v2/v3;
- balanced count/condition structure.

---

# 14. Interpretation Rules

Possible extension outcomes are interpreted as follows.

## Case A

Qwen actuator succeeds AND HoloCount external controller succeeds.

Interpretation:

```text
cross-architecture actuator replication
+
cross-dataset natural routing generalization
```

This is the strongest extension outcome.

## Case B

Qwen actuator succeeds; HoloCount frozen controller fails but the
HoloCount action oracle retains meaningful repair headroom.

Interpretation:

```text
actuator-like mechanistic control generalizes across architectures,
while deployment policy remains domain dependent across natural datasets.
```

This still strongly supports the central AROMA thesis.

## Case C

Qwen actuator succeeds but Qwen oracle headroom is insufficient for a
full router.

Interpretation:

```text
mechanistic cardinality control replicates,
but deployable repair capacity differs across architectures.
```

## Case D

Qwen replication fails.

The negative result must be retained.

It may not be replaced by unbounded model shopping or unrestricted
head-selection criteria.

A third model may only be introduced later as a separately declared
extension, not as a hidden replacement for Qwen.

---

# 15. Publication Integration

The completed AROMA v1 paper remains the baseline manuscript.

Extension experiments should modify the paper only after their evidence is
frozen and audited.

Successful extension evidence may add:

```text
Cross-Architecture and Cross-Dataset Validation
```

to the Results section.

The original AROMA evidence chain remains unchanged:

```text
causal actuator discovery
-> adaptive procedural routing
-> frozen procedural confirmation
-> frozen natural zero-shot failure
-> mechanism/router separation
-> natural calibration
-> untouched TallyQA confirmation
```

The extension may append:

```text
-> independent natural benchmark evaluation
-> cross-architecture actuator replication
```

It must not rewrite the chronology of the original study.

---

# 16. Immutable Claims Policy

Until supported by extension results, the paper must continue to avoid
claiming:

- cross-model mechanistic generalization;
- cross-dataset natural generalization;
- universal counting repair;
- architecture-independent localization;
- successful zero-shot natural transfer in all domains.

Claims may be expanded only after corresponding frozen evidence exists.

---

# 17. Execution Order

The execution order is frozen as:

```text
1. protocol freeze
2. HoloCount dataset provenance + manifest freeze
3. HoloCount baseline-only audit
4. frozen K=500 HoloCount external evaluation
5. if needed: HoloCount post-hoc action oracle / forensics
6. archive HoloCount result
7. Qwen2.5-VL model/revision freeze
8. Qwen compatibility + alpha=1 identity audit
9. Qwen coarse causal scan
10. candidate validation
11. actuator freeze
12. 300-example directional confirmation
13. specificity controls
14. dose response
15. oracle-headroom gate
16. full Qwen router only if gate passes
```

This ordering must not be changed based on outcome desirability.

---

# 18. Reproducibility Principle

Every frozen phase must record:

- Git commit;
- model revision;
- dataset revision;
- configuration;
- random / deterministic seed;
- input manifest SHA256;
- controller/model artifact SHA256 where relevant;
- output artifact SHA256;
- protocol classification.

No result should depend solely on terminal logs.
