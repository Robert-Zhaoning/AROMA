# AROMA Reproducibility Guide

This document is the canonical reproducibility guide for the AROMA v1
experimental study.

It describes the reference environment, data layout, frozen experimental
protocols, reproduction commands, artifact dependencies, evidence classes,
and provenance rules used for the reported results.

For complementary records, see:

- `docs/EXPERIMENT_LEDGER.md` — full scientific experiment ledger;
- `docs/RESULTS_CANONICAL.md` — canonical paper-level numerical results;
- `docs/ARTIFACT_MANIFEST.md` — publication artifact paths and SHA256 hashes;
- `docs/HISTORICAL_CODE.md` — historical exploratory-code archive.

---

## 1. Scientific Scope

The completed AROMA v1 study focuses on:

1. causal discovery of a cardinality-sensitive VLM mechanism;
2. bidirectional cardinality steering through L18H13;
3. adaptive intervention routing on procedural counting;
4. frozen procedural confirmation;
5. frozen zero-shot natural-domain failure;
6. post-hoc diagnosis of the synthetic-to-natural controller shift;
7. sample-efficient natural-domain adaptation;
8. frozen confirmation on a new image-disjoint TallyQA sample.

The final study is intentionally narrower than the original broad
stage-specific P/R/M repair proposal.

The current experiments do **not** establish a complete implementation of all
originally proposed stage-specific repair modules.

---

## 2. Evidence Classes

AROMA distinguishes the following evidence classes:

- `PILOT`
- `EXPLORATORY`
- `DEVELOPMENT`
- `POST-HOC FORENSIC`
- `FROZEN CONFIRMATORY`
- `INDEPENDENT CONFIRMATION`

These categories must not be conflated.

Experiment-by-experiment classifications are recorded in:

`docs/EXPERIMENT_LEDGER.md`

In particular:

- post-hoc forensic analyses must not be described as prespecified;
- development OOF results must not be described as independent confirmation;
- the frozen natural zero-shot failure must remain visible;
- the final TallyQA confirmation must not be described as cross-dataset
  generalization.

---

## 3. Canonical Experimental Milestone

The completed experimental milestone is tagged:

~~~text
aroma-experiments-v1-complete
~~~

The tag resolves to:

~~~text
ae6e08cdef84a512426268fa0e508dfe53237f14
~~~

Publication consolidation is performed separately on:

~~~text
publication-v1
~~~

The experimental history before and including the milestone should not be
rewritten, rebased, amended, or force-pushed.

---

## 4. Primary Model

The primary VLM is:

~~~text
meta-llama/Llama-3.2-11B-Vision-Instruct
~~~

Model weights are not redistributed in this repository.

Access may require:

- accepting the corresponding Meta / Hugging Face model license;
- authenticating with Hugging Face;
- configuring a local Hugging Face cache.

The mechanistic experiments intervene on multimodal cross-attention.

Eager attention is used where required by the intervention pipeline.

---

## 5. Reference Experimental Environment

The final reported experiments were successfully executed with:

~~~text
Python            3.11.16
PyTorch           2.11.0+cu128
Transformers      5.17.0
Accelerate        1.15.0
NumPy             2.4.6
Pandas            3.0.5
SciPy             1.17.1
scikit-learn      1.9.0
Pillow            12.3.0
PyYAML            6.0.3
tqdm              4.70.0
matplotlib        3.11.1
joblib            1.6.0
~~~

Reference accelerator stack:

~~~text
CUDA              12.8
GPU               NVIDIA B200
~~~

The same reference information is stored in:

~~~text
environment/reference_environment.txt
~~~

These versions document the successful final environment. They are not a claim
that every exact version is strictly necessary on every compatible machine.

---

## 6. Recommended Repository Checkout

Several development scripts preserve the exact absolute path used during the
reported experiments:

~~~text
/workspace/AromaExperiments
~~~

For the closest reproduction of the original RunPod environment:

~~~bash
mkdir -p /workspace
cd /workspace

git clone \
  --branch publication-v1 \
  --single-branch \
  https://github.com/Robert-Zhaoning/AROMA.git \
  AromaExperiments

cd AromaExperiments
~~~

The explicit branch selection is important because the publication and
reproducibility material lives on `publication-v1`.

Some formal scripts use repository-relative paths.

Some archived development scripts intentionally preserve the original
`/workspace/AromaExperiments` absolute path rather than being retrospectively
rewritten after the experiments.

---

## 7. Installation

Create a Python 3.11 environment.

Example:

~~~bash
conda create -n aroma python=3.11
conda activate aroma
~~~

Install the local package:

~~~bash
pip install -e .
~~~

The package metadata contains the lightweight analysis dependencies.

Full VLM inference additionally requires a compatible installation of:

~~~text
torch
transformers
accelerate
~~~

The reference versions used for the final experiments are documented in:

~~~text
environment/reference_environment.txt
~~~

PyTorch installation is GPU- and CUDA-dependent. Install an appropriate build
for the target system rather than blindly copying a CUDA-specific wheel from
the reference machine.

---

## 8. Data Policy

Raw third-party datasets are not redistributed by this repository.

Third-party local material should be stored under:

~~~text
external/
~~~

`external/` is excluded from Git.

The TallyQA metadata used during the experiments was stored locally as:

~~~text
external/tallyqa/qa/train.json
external/tallyqa/qa/test.json
~~~

Natural-image experiments use:

- TallyQA questions;
- Visual Genome images.

The repository selectively archives:

- frozen manifests;
- image inventories;
- controller bundles;
- evaluation metadata;
- canonical result tables;
- forensic summaries.

Raw third-party image collections are intentionally excluded.

---

## 9. Canonical Evidence Sources

Use:

~~~text
docs/RESULTS_CANONICAL.md
~~~

for final paper-level numerical results.

Use:

~~~text
docs/EXPERIMENT_LEDGER.md
~~~

for:

- scientific questions;
- dataset roles;
- evidence classifications;
- protocol status;
- interpretations;
- allowed claims;
- freeze provenance.

Use:

~~~text
docs/ARTIFACT_MANIFEST.md
~~~

for publication-critical artifact hashes.

Do not reconstruct canonical numbers from terminal screenshots when a canonical
artifact exists.

---

## 10. L18H13 Bidirectional Cardinality Steering

The primary cardinality actuator identified by AROMA is:

~~~text
Layer 18, Head 13
L18H13
~~~

Run the 300-sample intervention experiment with:

~~~bash
python scripts/run_l18h13_gain_all300.py
~~~

A protocol-only check is available:

~~~bash
python scripts/run_l18h13_gain_all300.py --protocol-only
~~~

Canonical outputs are archived under:

~~~text
outputs/proc_count_causal_v1/router/l18h13_gain_all300/
~~~

Canonical directional statistics:

~~~text
alpha = 0.5:
274 / 300 = 91.33% shift expected numeral downward

alpha = 1.5:
272 / 300 = 90.67% shift expected numeral upward

strict ordering:
266 / 300 = 88.67%

down/up response magnitude Spearman:
rho = 0.951442127
~~~

Important provenance distinction:

The original 300-sample directional experiment used the earlier numeral proxy:

~~~text
1, ..., 10
~~~

The later controller-development pipeline used the separately audited expanded
proxy:

~~~text
0, ..., 15
~~~

These stages should not be conflated.

---

## 11. Proc-Count-Causal v2 Action-Space Development

Synthetic v2 baseline accuracy:

~~~text
50.3%
~~~

Full multistrength oracle:

~~~text
139 / 497 baseline-wrong examples repairable
oracle accuracy = 64.2%
oracle gain = +13.9 pp
~~~

Final compressed intervention action set:

~~~text
{0, 1, 1.5, 2, 4}
~~~

Compressed five-action oracle:

~~~text
138 / 497 baseline-wrong examples repairable
oracle accuracy = 64.1%
oracle gain = +13.8 pp
~~~

Canonical action-space artifacts are archived under:

~~~text
outputs/proc_count_causal_v2/signed_steering/
~~~

---

## 12. Frozen Synthetic Controller

The final synthetic controller uses:

~~~text
Actuator:
L18H13

Feature family:
39 GT-free baseline / numeral-geometry features

Model:
per-action StandardScaler -> Ridge

Ridge alpha:
0.01

Intervention threshold:
0.1

Actions:
{0, 1, 1.5, 2, 4}

NOOP:
alpha = 1
~~~

Development OOF result:

~~~text
50.3% -> 58.6%
+8.3 pp

repairs:
91

breaks:
8

net repairs:
+83

intervention rate:
33.3%
~~~

Frozen bundle:

~~~text
outputs/proc_count_causal_v2/controller/final_frozen_controller/
aroma_cardinality_controller.joblib
~~~

Freeze commit:

~~~text
c8c5ede1601b1ada4087e141d61247903a4458a7
~~~

Canonical tag:

~~~text
aroma-controller-pre-v3-freeze
~~~

---

## 13. Frozen Proc-Count-Causal v3 Confirmation

Before expensive inference, run:

~~~bash
python scripts/run_proc_count_causal_v3_final.py --protocol-only
~~~

Execute the frozen evaluation:

~~~bash
python scripts/run_proc_count_causal_v3_final.py
~~~

Resume only after a technical or infrastructure interruption:

~~~bash
python scripts/run_proc_count_causal_v3_final.py --resume
~~~

Do not use `--resume` for outcome-dependent reruns.

Canonical result:

~~~text
N = 2000

baseline:
999 / 2000 = 49.95%

post-controller:
1161 / 2000 = 58.05%

gain:
+8.10 pp

95% CI:
[+6.85, +9.40] pp

repairs:
173

breaks:
11

net repairs:
+162

intervention rate:
32.65%
~~~

By condition:

~~~text
dense          +10.25 pp
distractors     +7.25 pp
grid           +12.00 pp
random_sparse   +7.00 pp
row             +4.00 pp
~~~

Canonical result directory:

~~~text
outputs/proc_count_causal_v3/final_frozen_controller/
~~~

Freeze / result chain:

~~~text
c8c5ede  synthetic controller freeze
582bb53  v3 dataset-protocol freeze
9085d09  v3 final runner freeze
9bc192e  v3 result archive
~~~

Full commit IDs:

~~~text
c8c5ede1601b1ada4087e141d61247903a4458a7
582bb53fe51d8b8bbeb45b90b7d073253d4df2a1
9085d093a5d2a46993e5dc8576de9e50a9c21c55
9bc192e7cc70cf754b0446672bb64b016490a746
~~~

Canonical result tag:

~~~text
aroma-v3-final-confirmation
~~~

---

## 14. Frozen TallyQA Zero-Shot Natural Evaluation

The first frozen natural evaluation used:

~~~text
4000 TallyQA examples
2000 Simple
2000 Complex
4000 unique images
~~~

Download the frozen Visual Genome image set:

~~~bash
python scripts/download_tallyqa_frozen_vg_images.py --workers 8
~~~

Audit the frozen image inventory:

~~~bash
python scripts/audit_tallyqa_frozen_images.py
~~~

Audit the evaluation protocol without loading the VLM:

~~~bash
python scripts/run_tallyqa_natural_ood_v1_final.py --protocol-only
~~~

Execute the frozen evaluation:

~~~bash
python scripts/run_tallyqa_natural_ood_v1_final.py
~~~

Resume only after a purely technical interruption:

~~~bash
python scripts/run_tallyqa_natural_ood_v1_final.py --resume
~~~

Canonical result:

~~~text
baseline:
64.85%

post-controller:
61.875%

change:
-2.975 pp

95% CI:
[-3.70, -2.25] pp

repairs:
51

breaks:
170

net repairs:
-119

intervention rate:
65.275%

exact McNemar:
p ≈ 3.78e-16
~~~

This is a frozen negative-transfer result.

It is intentionally retained and must not be hidden or relabeled as successful
zero-shot natural transfer.

Freeze chain:

~~~text
4cedd72  natural OOD manifest freeze
17fe2aa  natural OOD image inventory freeze
6f04635  natural OOD final evaluator freeze
bdc740b  frozen zero-shot negative result
~~~

Full relevant commit IDs:

~~~text
4cedd7297280b65f80d7a97c333d2e31b55aeeec
17fe2aaabc4e0ae32acef4eef64c6e6ead57833f
6f04635c2aa53bfdf0867fc1d6ac35c39c03c3d4
bdc740bda8482b5a8dfa25e9e110ac2c819359a8
~~~

Canonical result tag:

~~~text
aroma-tallyqa-ood-zero-shot-result
~~~

---

## 15. Natural Action Oracle

The action oracle is a post-hoc forensic experiment.

Run:

~~~bash
python scripts/run_tallyqa_natural_ood_action_oracle.py
~~~

Protocol-only mode:

~~~bash
python scripts/run_tallyqa_natural_ood_action_oracle.py --protocol-only
~~~

Resume after infrastructure interruption only:

~~~bash
python scripts/run_tallyqa_natural_ood_action_oracle.py --resume
~~~

Canonical oracle result:

~~~text
baseline accuracy:
64.85%

oracle accuracy:
71.575%

oracle gain:
+6.725 pp

unique repairable baseline-wrong examples:
269 / 1406 = 19.13%
~~~

Useful fixed-action result:

~~~text
alpha = 0

64.85% -> 67.20%

gain:
+2.35 pp
~~~

Interpretation:

The L18H13 actuator retains substantial natural-domain repair opportunity even
though the synthetic-trained routing policy fails.

This supports separating:

~~~text
actuator transfer
~~~

from:

~~~text
controller / conditional-utility transfer
~~~

### Important downstream dependency

The action-oracle runner produces:

~~~text
outputs/tallyqa_natural_ood_v1/forensics/action_oracle/action_matrix.csv
~~~

This matrix is required by several downstream natural-domain forensic and
adaptation scripts.

Therefore, before reproducing those analyses, run:

~~~bash
python scripts/run_tallyqa_natural_ood_action_oracle.py
~~~

The matrix is an intermediate dependency and is not itself treated as an
independent confirmatory result.

---

## 16. Natural Error-Direction Forensics

Run:

~~~bash
python scripts/analyze_tallyqa_natural_ood_error_direction.py
~~~

Among the 1,406 natural baseline errors:

~~~text
undercount:
297 = 21.12%

overcount:
1109 = 78.88%

absolute error = 1:
1017 = 72.33%

mean signed error:
+0.834993

median signed error:
+1

mean absolute error:
1.554765
~~~

The natural error distribution is substantially more overcount-heavy than the
synthetic development distribution.

This is post-hoc forensic evidence.

---

## 17. Controller Domain-Shift Analysis

Run:

~~~bash
python scripts/analyze_aroma_controller_domain_shift.py
~~~

The script compares:

~~~text
v2 development OOF
-> v3 procedural confirmation
-> natural TallyQA
~~~

It analyzes:

- controller score calibration;
- intervention-rate shift;
- per-action score shift;
- realized intervention utility;
- feature-distribution shift.

Approximate selected-score / realized-utility Spearman behavior:

~~~text
synthetic v2 / v3:
approximately +0.41

natural TallyQA:
approximately -0.067
~~~

Natural frozen-controller intervention rate:

~~~text
65.275%
~~~

This analysis performs:

- no VLM inference;
- no controller fitting;
- no hyperparameter search;
- no modification of frozen artifacts.

It is post-hoc forensic evidence.

---

## 18. Exact Ridge OOD Attribution

Run:

~~~bash
python scripts/analyze_aroma_ridge_feature_attribution.py
~~~

The frozen per-action models are:

~~~text
StandardScaler -> Ridge
~~~

The analysis performs:

1. exact stored-score reproduction;
2. exact frozen-controller decision reproduction;
3. mean score-drift attribution;
4. standardized feature OOD analysis;
5. extreme-score sample attribution;
6. baseline-correct / baseline-wrong attribution.

One important pathological feature is:

~~~text
baseline_numprob_15
~~~

Synthetic scaler statistics are approximately:

~~~text
mean  ≈ 5.09e-07
scale ≈ 6.97e-07
~~~

An extreme natural raw value around:

~~~text
0.351970
~~~

can create a standardized value on the order of:

~~~text
5e5
~~~

which can produce extremely large linear Ridge action scores.

This is post-hoc forensic evidence.

---

## 19. Exact v2 Training-Support Reconstruction

Run:

~~~bash
python scripts/reconstruct_v2_controller_training_support.py
~~~

This reconstructs the exact 39-dimensional feature matrix used to characterize
the synthetic controller's empirical training support.

The reconstructed scaler statistics match the scaler stored in the frozen
controller.

Canonical artifacts are stored under:

~~~text
outputs/proc_count_causal_v2/controller/training_support_reconstruction/
~~~

This analysis enables exact investigation of natural-domain support violation.

---

## 20. SPUC-v1 Support Projection

Run:

~~~bash
python scripts/evaluate_support_projected_controller_v1.py
~~~

SPUC-v1 clips standardized feature values to the exact empirical min/max
support reconstructed from v2 while preserving:

- feature representation;
- StandardScaler parameters;
- Ridge coefficients;
- Ridge intercepts;
- action space;
- threshold;
- L18H13 actuator.

No VLM inference is performed.

No controller parameters are fitted.

Canonical natural-domain result:

~~~text
original frozen-controller accuracy:
61.875%

SPUC accuracy:
61.875%

original intervention rate:
65.275%

SPUC intervention rate:
63.275%

policy agreement:
82.95%

repairs:
32

breaks:
151

net repairs:
-119
~~~

Natural samples with at least one clipped feature:

~~~text
1593 / 4000 = 39.825%
~~~

Interpretation:

Support violation explains catastrophic score explosion but does not explain
the deeper conditional-utility mismatch.

SPUC is a diagnostic / safety ablation, not a successful natural repair method.

---

## 21. Natural Utility Learnability

Prerequisite:

~~~text
outputs/tallyqa_natural_ood_v1/forensics/action_oracle/action_matrix.csv
~~~

Generate it first with:

~~~bash
python scripts/run_tallyqa_natural_ood_action_oracle.py
~~~

Then run:

~~~bash
python scripts/run_tallyqa_natural_utility_learnability_probe.py
~~~

Protocol:

~~~text
Dataset:
already-used TallyQA-4000 development / forensic set

Evaluation:
5-fold OOF

Split:
fixed stratification by Simple / Complex

Features:
same 39-dimensional GT-free representation

Model:
StandardScaler -> Ridge

ridge alpha:
0.01

threshold:
0.1

actions:
{0, 1, 1.5, 2, 4}

VLM inference:
none

hyperparameter search:
none
~~~

Canonical result:

~~~text
64.85% -> 67.15%

gain:
+2.30 pp

repairs:
126

breaks:
34

net repairs:
+92

intervention rate:
16.45%
~~~

This is post-hoc development evidence.

It must not be described as an independent natural confirmation.

---

## 22. Natural Adaptation Sample-Efficiency Curve

Prerequisite:

~~~text
outputs/tallyqa_natural_ood_v1/forensics/action_oracle/action_matrix.csv
~~~

Run:

~~~bash
python scripts/run_tallyqa_natural_adaptation_curve.py
~~~

Calibration sizes:

~~~text
50
100
250
500
1000
2000
3200
~~~

Canonical mean gains:

~~~text
K=50     +0.330 pp
K=100    +1.180 pp
K=250    +1.870 pp
K=500    +2.120 pp
K=1000   +2.240 pp
K=2000   +2.325 pp
K=3200   +2.300 pp
~~~

At K=500:

~~~text
25 / 25 predefined development runs had positive gain
~~~

The runs share the same TallyQA development pool and fold structure.

Therefore `25/25` must not be interpreted as an independent probability
estimate.

K=500 was selected as a sample-efficiency elbow, not because it produced the
maximum development gain.

---

## 23. Frozen K=500 Natural Controller

Prerequisites include:

~~~text
outputs/proc_count_causal_v2/controller/final_frozen_controller/
aroma_cardinality_controller.joblib

outputs/tallyqa_natural_ood_v1/final_frozen_controller/
tallyqa_final_results.csv

outputs/tallyqa_natural_ood_v1/forensics/action_oracle/
action_matrix.csv
~~~

Freeze script:

~~~bash
python scripts/freeze_aroma_natural_controller_k500.py
~~~

Calibration subset:

~~~text
500 total
250 Simple
250 Complex
~~~

Selection uses deterministic SHA256 ranking based on question metadata and a
fixed salt.

Selection does not use:

- correctness;
- repairability;
- model utility;
- action outcome;
- final confirmation results.

Frozen specification:

~~~text
features:
39

model:
StandardScaler -> Ridge

ridge alpha:
0.01

threshold:
0.1

actions:
{0, 1, 1.5, 2, 4}

actuator:
L18H13
~~~

Frozen controller:

~~~text
outputs/phase2_natural_controller_k500/final_frozen_controller/
aroma_natural_utility_controller_k500.joblib
~~~

Controller SHA256:

~~~text
d9d8ba3a24814bf71b3f4039d1effa864b57754a00539f1260a1d077c03449f5
~~~

Calibration-manifest SHA256:

~~~text
67dfdafa36c66a1299be7adea0d1232269e1b557607314972ed2b298312dc49a
~~~

Freeze commit:

~~~text
9bbd337a979029da1bc46396c8e266da5c9e6193
~~~

The controller was frozen before inference on Natural Confirmation v2.

---

## 24. Untouched Natural Confirmation v2

The final confirmation set contains:

~~~text
4000 total questions
2000 Simple
2000 Complex

4000 unique question IDs
4000 unique image paths
4000 unique image IDs
~~~

There is zero overlap with the earlier TallyQA development/evaluation set in:

- question ID;
- image path;
- image ID.

The new sample is drawn from the same official TallyQA benchmark distribution.

It is **not** a separate dataset.

It must therefore not be described as cross-dataset generalization.

Download frozen confirmation images:

~~~bash
python scripts/download_tallyqa_natural_confirmation_v2_images.py --workers 8
~~~

Audit the frozen image set:

~~~bash
python scripts/audit_tallyqa_natural_confirmation_v2_images.py
~~~

Audit the frozen evaluation protocol without loading the VLM:

~~~bash
python scripts/run_tallyqa_natural_confirmation_v2_final.py --protocol-only
~~~

Execute the final frozen evaluation:

~~~bash
python scripts/run_tallyqa_natural_confirmation_v2_final.py
~~~

Resume only after a purely technical interruption:

~~~bash
python scripts/run_tallyqa_natural_confirmation_v2_final.py --resume
~~~

Prespecified primary success criterion:

~~~text
accuracy change > 0
AND
exact McNemar p < 0.05
~~~

Canonical final result:

~~~text
N = 4000

baseline correct:
2597

baseline accuracy:
64.925%

post-controller correct:
2682

post-controller accuracy:
67.050%

gain:
+2.125 pp

95% CI:
[+1.50, +2.775] pp

repairs:
130

breaks:
45

net repairs:
+85

intervention rate:
20.0%

exact McNemar:
p = 9.176009e-11

primary_success:
TRUE
~~~

Prespecified Simple subset:

~~~text
78.05% -> 79.15%

gain:
+1.10 pp

95% CI:
[+0.25, +1.95] pp

repairs:
49

breaks:
27

intervention rate:
11.8%
~~~

Prespecified Complex subset:

~~~text
51.80% -> 54.95%

gain:
+3.15 pp

95% CI:
[+2.20, +4.15] pp

repairs:
81

breaks:
18

intervention rate:
28.2%
~~~

Final action distribution:

~~~text
alpha=0      508   12.70%
alpha=1     3200   80.00%
alpha=1.5     32    0.80%
alpha=2      122    3.05%
alpha=4      138    3.45%
~~~

Freeze chain:

~~~text
9bbd337  K=500 natural controller freeze
9fc7822  Natural Confirmation v2 manifest freeze
9dfd1e8  Natural Confirmation v2 image inventory freeze
ca0fb42  final evaluator freeze
ae6e08c  final result archive
~~~

Full relevant commit IDs:

~~~text
9bbd337a979029da1bc46396c8e266da5c9e6193
9fc7822beb9e224291591a89b46a182c7c1ffe01
9dfd1e8ce62b745227aade37edb980e34016318d
ca0fb4274d147b8e70e641fecfdd9961ba9da3db
ae6e08cdef84a512426268fa0e508dfe53237f14
~~~

Experimental milestone:

~~~text
aroma-experiments-v1-complete
~~~

---

## 25. Protocol-Only Audits

Frozen final runners expose:

~~~text
--protocol-only
~~~

where available.

Protocol-only mode checks combinations of:

- manifest identity;
- controller identity;
- image inventory;
- Git ancestry;
- feature leakage;
- ground-truth isolation;
- frozen configuration;
- protocol consistency.

Protocol-only mode does not load the VLM or generate evaluation predictions.

Important examples:

~~~bash
python scripts/run_proc_count_causal_v3_final.py --protocol-only

python scripts/run_tallyqa_natural_ood_v1_final.py --protocol-only

python scripts/run_tallyqa_natural_confirmation_v2_final.py --protocol-only
~~~

Running the protocol-only audit before expensive inference is recommended.

---

## 26. Resume Policy

Frozen runners expose:

~~~text
--resume
~~~

only to recover from technical or infrastructure interruption.

It must not be used to:

- rerun because a result was undesirable;
- search for a better outcome;
- modify a frozen controller;
- change a threshold after evaluation;
- change the evaluation sample;
- alter intervention strengths;
- choose a better seed.

This restriction is part of the confirmatory provenance.

---

## 27. Canonical Publication Artifacts

Publication-critical outputs are selectively tracked under:

~~~text
outputs/
~~~

The canonical publication artifact registry is:

~~~text
docs/ARTIFACT_MANIFEST.md
~~~

The manifest records:

- artifact path;
- evidence class;
- archival status;
- scientific purpose;
- file size;
- SHA256.

Not every local intermediate file under `outputs/` should be interpreted as a
canonical paper artifact.

---

## 28. Historical Research Code

Early exploratory, superseded, smoke-test, and research-development scripts are
preserved under:

~~~text
archive/historical_scripts/
~~~

The corresponding manifest is:

~~~text
archive/historical_scripts/MANIFEST.tsv
~~~

It records:

- archived path;
- original local path;
- category;
- line count;
- SHA256.

See also:

~~~text
docs/HISTORICAL_CODE.md
~~~

These scripts are retained for transparency and methodological history.

They are not the recommended publication reproduction interface.

---

## 29. Core Reproduction Paths

There are three useful levels of reproduction.

### Level A — Artifact verification

No VLM inference is required.

Inspect:

~~~text
docs/RESULTS_CANONICAL.md
docs/EXPERIMENT_LEDGER.md
docs/ARTIFACT_MANIFEST.md
~~~

and the selectively tracked canonical outputs.

### Level B — Post-hoc analysis reproduction

Use the already-generated frozen result artifacts to reproduce analyses such as:

~~~bash
python scripts/analyze_tallyqa_natural_ood_error_direction.py

python scripts/analyze_aroma_controller_domain_shift.py

python scripts/analyze_aroma_ridge_feature_attribution.py

python scripts/reconstruct_v2_controller_training_support.py

python scripts/evaluate_support_projected_controller_v1.py

python scripts/analyze_spuc_v1_residual_miscalibration.py
~~~

Some downstream analyses require the action-oracle matrix generated by:

~~~bash
python scripts/run_tallyqa_natural_ood_action_oracle.py
~~~

### Level C — Full VLM inference reproduction

Run the frozen protocol audits and then execute:

~~~bash
python scripts/run_proc_count_causal_v3_final.py --protocol-only
python scripts/run_proc_count_causal_v3_final.py

python scripts/run_tallyqa_natural_ood_v1_final.py --protocol-only
python scripts/run_tallyqa_natural_ood_v1_final.py

python scripts/run_tallyqa_natural_confirmation_v2_final.py --protocol-only
python scripts/run_tallyqa_natural_confirmation_v2_final.py
~~~

This level requires the VLM, compatible GPU infrastructure, and the relevant
third-party image data.

---

## 30. Canonical Scientific Result Sequence

The supported scientific sequence is:

~~~text
causal mechanism discovery
-> bidirectional cardinality steering
-> adaptive procedural repair
-> successful frozen procedural confirmation
-> frozen zero-shot natural failure
-> actuator-vs-router separation
-> natural-domain utility shift diagnosis
-> sample-efficient natural adaptation
-> successful untouched natural confirmation
~~~

Key numerical landmarks:

~~~text
Procedural confirmation:
49.95% -> 58.05%
+8.10 pp

Frozen zero-shot natural transfer:
64.85% -> 61.875%
-2.975 pp

Natural action oracle:
64.85% -> 71.575%
+6.725 pp

Natural utility OOF development:
64.85% -> 67.15%
+2.30 pp

Untouched natural confirmation:
64.925% -> 67.050%
+2.125 pp
~~~

The strongest supported interpretation is:

A causally controllable cardinality mechanism can serve as an adaptive repair
primitive, but intervention utility is domain dependent. A controller trained
only on procedural data does not transfer successfully zero-shot to natural
counting. The underlying actuator nevertheless retains useful natural repair
opportunity, and lightweight natural-domain calibration can restore routing
utility that generalizes to a new image-disjoint held-out TallyQA sample.

---

## 31. Claims That Must Not Be Made

The current evidence does not establish:

- complete implementation of all originally proposed P/R/M repair modules;
- universal VLM counting repair;
- cross-model generalization;
- cross-dataset natural generalization;
- successful zero-shot natural transfer;
- that observational attention alone identifies the causal mechanism;
- that K=500 was chosen using the final confirmation set;
- that post-hoc analyses were prespecified confirmatory experiments.

The final TallyQA confirmation is:

~~~text
question-disjoint
image-disjoint
image-ID-disjoint
~~~

from the earlier TallyQA sample, but remains within the same underlying
official TallyQA benchmark distribution.

---

## 32. Reproduction Ethics and Provenance

Do not rewrite frozen experimental history.

In particular:

- do not rebase frozen experiment commits;
- do not amend frozen result commits;
- do not force-push frozen experiment history;
- do not hide the frozen natural zero-shot failure;
- do not relabel post-hoc forensic work as prespecified;
- do not modify the K=500 controller after seeing confirmation results;
- do not change the untouched confirmation manifest after evaluation;
- do not describe the final confirmation as cross-dataset generalization.

The inclusion of both successful and failed transfer experiments is an
intentional part of the AROMA scientific record.

