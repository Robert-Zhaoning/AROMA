# AROMA HoloCount Extension Checkpoint

Status: HoloCount main experimental track completed through
post-hoc utility-learnability analysis.

## Frozen external evaluation

Dataset:
MM-MVR/HoloCount

Dataset revision:
f44cfe591e8f7e64b63a2fb080b98bffa10b45aa

Model:
meta-llama/Llama-3.2-11B-Vision-Instruct

Model revision:
9eb2daaa8597bf192a8b0e73f848f3a102794df5

Actuator:
L18H13

Frozen TallyQA-adapted controller:
K=500 natural controller

Primary population:
2480 HoloCount questions / 2056 unique image hashes

## Frozen cross-dataset controller result

Baseline:
46.6935%

Post-controller:
46.6532%

Change:
-0.0403 pp

Repairs:
72

Breaks:
73

Net:
-1

Intervention rate:
35.48%

Exact McNemar:
p = 1.0

Primary external-transfer success:
FALSE

Image-cluster robustness:
FALSE

Strong external-generalization success:
FALSE

Frozen full-result SHA256:
82929de79583d0c0641a80e13c8978d4ee4c00cebe7deb8f132697d3b6295466

## Post-hoc action oracle

Repairable wrong:
304 / 1322

Repairable supported wrong:
304 / 1134 = 26.81%

Oracle accuracy:
58.9516%

Oracle headroom:
+12.2581 pp

Action repair counts:
alpha=0.0: 99
alpha=1.5: 59
alpha=2.0: 95
alpha=4.0: 176

Oracle-result SHA256:
1c5f367c43a0af43e88432cc7c577c23e349f8d4ce16d754bda32cf7e44aa6ce

Interpretation:
Substantial actuator capacity transfers to HoloCount even though the
frozen TallyQA routing policy does not.

## Router-vs-actuator forensics

Oracle-repairable supported wrong:
304

Controller captured:
72 / 304 = 23.68%

Missed repairable:
232

Missed due to NOOP:
172

Missed due to wrong non-NOOP action:
60

Supported-wrong intervention precision:
15.69%

Frozen action-score / true-repair Spearman:
alpha=0.0:  +0.182
alpha=1.5:  -0.245
alpha=2.0:  -0.250
alpha=4.0:  -0.236

Error regime:
overcount: 558
undercount: 576

Oracle repairability:
overcount: 18.82%
undercount: 34.55%

Frozen-controller intervention rate:
overcount: 52.33%
undercount: 29.00%

Frozen-controller oracle capture:
overcount: 54.29%
undercount: 7.54%

Interpretation:
Cross-dataset routing failure reflects strong conditional-utility /
error-regime shift.

## Fixed-action baselines

alpha=0.0:
repairs=99
breaks=120
gain=-0.8468 pp

alpha=1.5:
repairs=59
breaks=58
gain=+0.0403 pp

alpha=2.0:
repairs=95
breaks=111
gain=-0.6452 pp

alpha=4.0:
repairs=176
breaks=265
gain=-3.5887 pp

Baseline-correct action-matrix SHA256:
fae0cfe8a7d13405cdee9c79cb3706fce76632df53b6747f9309d2c08c5e2556

Interpretation:
No fixed gain captures the large per-example oracle capacity.
Adaptive action selection is necessary.

## Same-39-feature HoloCount utility learnability

Primary:
5-fold StratifiedGroupKFold
stratified by HoloCount subset
grouped by image_sha256

Baseline:
46.6935%

OOF post:
46.6532%

Gain:
-0.0403 pp

Repairs:
7

Breaks:
8

Intervention rate:
3.43%

Sensitivity ordinary StratifiedKFold:
+0.1613 pp

Primary OOF score-utility Spearman:
alpha=0.0:  -0.010
alpha=1.5:  -0.027
alpha=2.0:  -0.036
alpha=4.0:  +0.018

Interpretation:
The actuator transfers strongly, but the existing 39-dimensional
numeral-geometry representation with the linear Ridge utility family
does not recover HoloCount action utility.

## Current scientific conclusion

HoloCount provides an external boundary result:

1. Frozen TallyQA routing policy does not transfer.
2. Fixed actions do not solve the problem.
3. L18H13 retains large per-example repair capacity (+12.26 pp oracle).
4. Existing 39-feature linear routing representation cannot recover
   that capacity in image-disjoint HoloCount OOF evaluation.

Core conclusion:

"An actuator is not a policy."

Mechanistic actuator transfer is substantially more robust than the
current routing representation/policy.

## Next major experimental track

Do NOT continue tuning HoloCount router hyperparameters.

Next priority:
Qwen/Qwen2.5-VL-7B-Instruct cross-architecture cardinality-actuator
replication.

Planned sequence:
compatibility audit -> coarse all-head scan -> fresh validation ->
frozen actuator selection -> untouched directional confirmation ->
specificity controls -> dose response.

