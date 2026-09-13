# AROMA

## Adaptive Routing of Mechanistic Actuators for Vision-Language Counting

**AROMA** studies whether vision-language counting can be **causally controlled
and selectively repaired** through internal mechanistic interventions.

We identify a bidirectional cardinality-sensitive cross-attention actuator,
build an adaptive controller that chooses intervention strengths per example,
confirm substantial gains on an untouched procedural benchmark, observe a
frozen zero-shot failure on natural images, diagnose that failure as a
controller/utility-transfer problem rather than disappearance of the actuator,
and show that lightweight natural-domain calibration restores significant gains
on a new image-disjoint TallyQA confirmation sample.

> **AROMA = Adaptive Routing of Mechanistic Actuators**

---

## Key Results

| Experiment | Baseline | Post / Oracle | Change |
|---|---:|---:|---:|
| Frozen Proc-Count-Causal v3 confirmation | 49.95% | 58.05% | **+8.10 pp** |
| Frozen zero-shot TallyQA transfer | 64.85% | 61.875% | **-2.975 pp** |
| Natural action oracle | 64.85% | 71.575% | **+6.725 pp** |
| Natural-domain OOF utility learning | 64.85% | 67.15% | **+2.30 pp** |
| Untouched TallyQA natural confirmation v2 | 64.925% | 67.050% | **+2.125 pp** |

Final natural confirmation:

~~~text
N = 4000

baseline:
2597 / 4000 = 64.925%

post-controller:
2682 / 4000 = 67.050%

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
~~~

The confirmation sample is question-, image-, and image-ID-disjoint from the
earlier TallyQA development/evaluation sample, but it comes from the same
official TallyQA benchmark distribution.

It is **not** claimed as cross-dataset generalization.

---

## Scientific Story

AROMA's central experimental sequence is:

~~~text
causal mechanism discovery
        |
        v
bidirectional cardinality steering
        |
        v
adaptive procedural repair
        |
        v
frozen procedural confirmation
        |
        |  +8.10 pp
        v
frozen zero-shot natural transfer
        |
        |  -2.975 pp
        v
actuator-vs-router separation
        |
        v
domain / utility shift diagnosis
        |
        v
sample-efficient natural adaptation
        |
        v
untouched natural confirmation
           +2.125 pp
~~~

The negative zero-shot result is part of the main scientific story rather than
an experiment that is hidden or discarded.

The evidence supports a more specific conclusion than universal mechanistic
transfer:

> A causal cardinality actuator can remain useful across domains even when the
> policy that decides when and how to use it does not.

---

## 1. Causal Cardinality Actuator

The primary actuator identified in the study is:

~~~text
Layer 18, Head 13
L18H13
~~~

Scaling this cross-attention head changes the model's expected numeral in both
directions.

On the 300-sample directional characterization:

~~~text
alpha = 0.5:
274 / 300 = 91.33% shift downward

alpha = 1.5:
272 / 300 = 90.67% shift upward

strict ordering:
266 / 300 = 88.67%

down/up response magnitude Spearman:
rho = 0.951442127
~~~

This supports interpreting L18H13 as a **bidirectional, dose-responsive causal
cardinality actuator**.

### Numeral-proxy provenance

The original 300-sample directional experiment used the earlier proxy:

~~~text
1, ..., 10
~~~

Later controller development used the separately audited expanded proxy:

~~~text
0, ..., 15
~~~

These stages are kept distinct in the experimental record.

---

## 2. Adaptive Routing

The final intervention action set is:

~~~text
{0, 1, 1.5, 2, 4}
~~~

where:

~~~text
alpha = 1
~~~

is the identity / NOOP action.

The synthetic controller uses:

~~~text
39 GT-free baseline / numeral-geometry features

per-action model:
StandardScaler -> Ridge

ridge alpha:
0.01

decision threshold:
0.1

actuator:
L18H13
~~~

Proc-Count-Causal v2 OOF development:

~~~text
50.3% -> 58.6%

gain:
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

A compressed five-action oracle reached:

~~~text
64.1%
+13.8 pp over baseline

138 / 497 baseline-wrong samples repairable
~~~

showing substantial remaining routing headroom.

---

## 3. Frozen Procedural Confirmation

The synthetic controller was frozen before evaluation on a newly generated
Proc-Count-Causal v3 set.

~~~text
N = 2000

baseline:
49.95%

post-controller:
58.05%

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

| Condition | Gain |
|---|---:|
| dense | +10.25 pp |
| distractors | +7.25 pp |
| grid | +12.00 pp |
| random_sparse | +7.00 pp |
| row | +4.00 pp |

Canonical tag:

~~~text
aroma-v3-final-confirmation
~~~

---

## 4. Frozen Zero-Shot Natural Failure

The same frozen synthetic controller was evaluated without natural adaptation on
4,000 TallyQA examples.

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

Canonical tag:

~~~text
aroma-tallyqa-ood-zero-shot-result
~~~

---

## 5. Mechanism Transfer vs. Routing Transfer

The zero-shot failure does **not** imply that the cardinality actuator becomes
useless on natural images.

### Natural action oracle

~~~text
baseline:
64.85%

oracle:
71.575%

gain:
+6.725 pp

repairable baseline-wrong samples:
269 / 1406 = 19.13%
~~~

A single suppressive action (`alpha = 0`) already produced:

~~~text
64.85% -> 67.20%
+2.35 pp
~~~

This shows that natural-domain repair opportunity remains.

### Error-direction shift

Among natural baseline errors:

~~~text
undercount:
297 / 1406 = 21.12%

overcount:
1109 / 1406 = 78.88%

absolute error = 1:
1017 / 1406 = 72.33%
~~~

The natural error mixture therefore differs strongly from the synthetic
controller-development distribution.

### Controller calibration shift

Approximate selected-score / realized-utility Spearman:

~~~text
synthetic v2 / v3:
~ +0.41

natural TallyQA:
~ -0.067
~~~

Exact Ridge attribution also revealed severe OOD score extrapolation under
`StandardScaler -> Ridge`.

Support projection removed the numerical explosion but did not recover useful
routing, indicating a deeper conditional-utility shift.

---

## 6. Natural-Domain Utility Learning

Using the **same 39-dimensional GT-free representation** and the same Ridge
model family, natural-domain action utility becomes learnable again.

Five-fold OOF development:

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

This result is **post-hoc development**, not an independent confirmation.

---

## 7. Sample-Efficient Adaptation

Natural calibration sizes:

| K | Mean gain |
|---:|---:|
| 50 | +0.330 pp |
| 100 | +1.180 pp |
| 250 | +1.870 pp |
| 500 | **+2.120 pp** |
| 1000 | +2.240 pp |
| 2000 | +2.325 pp |
| 3200 | +2.300 pp |

`K=500` was selected as a **sample-efficiency elbow**, not because it produced
the maximum observed development gain.

The selected calibration set contains:

~~~text
500 examples
250 Simple
250 Complex
~~~

selected deterministically by metadata-only SHA256 ranking.

No final-confirmation outcomes were used for selection.

---

## 8. Untouched Natural Confirmation

The frozen K=500 controller was evaluated on a new TallyQA sample with:

~~~text
4000 total questions

2000 Simple
2000 Complex

4000 unique question IDs
4000 unique image paths
4000 unique image IDs
~~~

There is zero question/image/image-ID overlap with the earlier TallyQA sample.

Final result:

~~~text
64.925% -> 67.050%

gain:
+2.125 pp

95% CI:
[+1.50, +2.775] pp

130 repairs
45 breaks

exact McNemar:
p = 9.176009e-11
~~~

Prespecified subset results:

| Subset | Baseline | Post | Gain |
|---|---:|---:|---:|
| Simple | 78.05% | 79.15% | +1.10 pp |
| Complex | 51.80% | 54.95% | **+3.15 pp** |

The prespecified primary criterion was:

~~~text
accuracy change > 0
AND
exact McNemar p < 0.05
~~~

and was satisfied.

---

## Quick Start

### Clone the publication branch

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

### Create the environment

~~~bash
conda create -n aroma python=3.11
conda activate aroma

pip install -e .
~~~

Full VLM inference additionally requires compatible installations of:

~~~text
torch
transformers
accelerate
~~~

The successful reference environment is recorded in:

[`environment/reference_environment.txt`](environment/reference_environment.txt)

---

## Reproduce the Main Frozen Evaluations

### Proc-Count-Causal v3

Audit first:

~~~bash
python scripts/run_proc_count_causal_v3_final.py --protocol-only
~~~

Then execute:

~~~bash
python scripts/run_proc_count_causal_v3_final.py
~~~

### Frozen TallyQA zero-shot evaluation

Audit first:

~~~bash
python scripts/run_tallyqa_natural_ood_v1_final.py --protocol-only
~~~

Then execute:

~~~bash
python scripts/run_tallyqa_natural_ood_v1_final.py
~~~

### Natural Confirmation v2

Audit first:

~~~bash
python scripts/run_tallyqa_natural_confirmation_v2_final.py --protocol-only
~~~

Then execute:

~~~bash
python scripts/run_tallyqa_natural_confirmation_v2_final.py
~~~

`--resume` is reserved for technical or infrastructure interruption and must
not be used for outcome-dependent reruns.

For complete setup, prerequisites, data layout, and dependency ordering, see:

**[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**

---

## Data

Third-party raw data is not redistributed.

Local third-party material is expected under:

~~~text
external/
~~~

TallyQA metadata used in the experiments was stored as:

~~~text
external/tallyqa/qa/train.json
external/tallyqa/qa/test.json
~~~

Visual Genome images for the frozen natural evaluations can be downloaded with:

~~~bash
python scripts/download_tallyqa_frozen_vg_images.py --workers 8

python scripts/download_tallyqa_natural_confirmation_v2_images.py --workers 8
~~~

See the reproducibility guide for the complete data workflow.

---

## Repository Structure

~~~text
AROMA/
|
|-- README.md
|-- pyproject.toml
|
|-- src/aroma/
|   `-- core package code
|
|-- scripts/
|   `-- formal publication / reproduction scripts
|
|-- configs/
|   `-- frozen experiment configurations
|
|-- outputs/
|   `-- selectively tracked canonical artifacts
|
|-- docs/
|   |-- EXPERIMENT_LEDGER.md
|   |-- RESULTS_CANONICAL.md
|   |-- ARTIFACT_MANIFEST.md
|   |-- REPRODUCIBILITY.md
|   `-- HISTORICAL_CODE.md
|
|-- environment/
|   `-- reference_environment.txt
|
|-- archive/historical_scripts/
|   |-- historical research scripts
|   `-- MANIFEST.tsv
|
`-- tests/
~~~

---

## Canonical Documentation

**[Experiment Ledger](docs/EXPERIMENT_LEDGER.md)**  
Full scientific record: protocol, evidence class, interpretation, allowed claim,
and provenance.

**[Canonical Results](docs/RESULTS_CANONICAL.md)**  
Authoritative source for paper-level numerical values.

**[Artifact Manifest](docs/ARTIFACT_MANIFEST.md)**  
Publication-critical paths, evidence roles, sizes, and SHA256 hashes.

**[Reproducibility Guide](docs/REPRODUCIBILITY.md)**  
Complete environment, dataset, dependency, protocol, and execution guide.

**[Historical Code](docs/HISTORICAL_CODE.md)**  
Explains the archive of exploratory and superseded research scripts.

---

## Frozen Provenance

Important milestones include:

~~~text
c8c5ede  freeze synthetic cardinality controller

582bb53  freeze Proc-Count-Causal v3 generation protocol
9085d09  freeze v3 final evaluator
9bc192e  archive v3 confirmation result

6f04635  freeze natural zero-shot evaluator
bdc740b  archive frozen zero-shot negative result

9bbd337  freeze K=500 natural-adapted controller
9fc7822  freeze natural confirmation v2 manifest
9dfd1e8  freeze confirmation image inventory
ca0fb42  freeze final natural evaluator
ae6e08c  archive final confirmation result
~~~

Completed experimental milestone:

~~~text
aroma-experiments-v1-complete
~~~

Publication consolidation branch:

~~~text
publication-v1
~~~

---

## Limitations

Current AROMA v1 evidence does **not** establish:

- universal VLM counting repair;
- complete implementation of all originally proposed P/R/M repair modules;
- cross-model generalization;
- cross-dataset natural generalization;
- successful zero-shot natural transfer;
- that observational attention alone identifies the causal mechanism.

The final natural confirmation is held out at the question and image level, but
remains within the same official TallyQA benchmark distribution.

Natural next steps include:

- replication on a second VLM;
- evaluation on an independent natural counting benchmark;
- broader mechanistic actuator discovery;
- uncertainty-aware routing;
- additional stage-specific repair mechanisms.

---

## Research Integrity

AROMA intentionally preserves:

- the frozen natural zero-shot failure;
- post-hoc vs. confirmatory distinctions;
- controller freeze chronology;
- confirmation-set disjointness;
- artifact SHA256 hashes;
- historical exploratory code.

Frozen experimental history should not be rewritten to make later outcomes
appear prespecified.

---

## Citation

Manuscript in preparation:

~~~text
AROMA: Adaptive Routing of Mechanistic Actuators
for Vision-Language Counting
~~~

Formal citation information will be added when the manuscript is released.

---

## Status

**AROMA v1 experimental development is complete.**

Current work focuses on:

~~~text
publication documentation
figures and tables
manuscript preparation
reproducibility packaging
~~~
