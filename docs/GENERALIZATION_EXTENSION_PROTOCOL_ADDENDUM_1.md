# AROMA Generalization Extension Protocol — Addendum 1

## Status

This addendum is created after the HoloCount dataset provenance/schema
audit and immutable manifest construction, but before any HoloCount VLM
prediction, AROMA controller action, accuracy measurement, repair/break
analysis, or action-oracle evaluation.

The original protocol in:

```text
docs/GENERALIZATION_EXTENSION_PROTOCOL.md
```

remains unchanged.

This addendum records structural properties of the frozen HoloCount
evaluation population that were discovered solely through dataset
provenance and manifest auditing.

---

# 1. Frozen HoloCount Population

Dataset:

```text
MM-MVR/HoloCount
```

Exact Hugging Face revision:

```text
f44cfe591e8f7e64b63a2fb080b98bffa10b45aa
```

Primary evaluation population:

```text
all 2480 records in data/metadata.jsonl
```

Primary manifest SHA256:

```text
07e9672295230dfdad9e356102165b0190117101f865418efd1d3a6be9d403cd
```

No example is removed on the basis of question content, answer value,
image duplication, anticipated model behavior, or controller compatibility.

---

# 2. Image-Level Dependence Discovered During Manifest Audit

The frozen 2480-question population contains:

```text
2480 question/sample IDs
2480 file names
2056 unique image-content SHA256 hashes
270 duplicate-image hash groups
694 question records belonging to duplicate-image groups
```

Thus some HoloCount questions are associated with identical image content.

This structure was discovered before any model inference.

The question-level paired evaluation remains the primary benchmark
evaluation because the official benchmark population consists of question
records.

However, image-level dependence may make a purely question-level
uncertainty estimate optimistic.

Therefore, in addition to the prespecified question-level paired bootstrap
and exact McNemar test, all final HoloCount controller evaluations will
include a mandatory image-cluster sensitivity analysis.

---

# 3. Image-Cluster Bootstrap Sensitivity Analysis

The clustering variable is:

```text
image_sha256
```

Each unique image hash defines one cluster.

The frozen HoloCount population contains:

```text
2056 image clusters
```

For the cluster bootstrap:

1. sample 2056 image clusters with replacement;
2. include all question records belonging to every sampled cluster;
3. compute baseline accuracy, post-controller accuracy, and paired
   accuracy difference on the resulting bootstrap sample;
4. repeat for exactly 20,000 bootstrap replicates;
5. use deterministic seed:

```text
20260914
```

6. report the percentile 95% confidence interval of the paired accuracy
   difference.

This analysis is fixed before any HoloCount model outcome is observed.

For a strong publication-level cross-dataset generalization claim, the
result should satisfy both:

```text
original question-level primary success criterion
```

and:

```text
image-cluster bootstrap 95% CI lower bound > 0
```

Failure of the cluster-bootstrap robustness condition must be reported even
if the original question-level significance criterion is met.

The exact McNemar result remains reported as the prespecified question-level
paired test.

---

# 4. Ground-Truth Count Range

The frozen manifest contains:

```text
answer minimum: 0
answer maximum: 149
answers <= 15: 2292 / 2480 = 92.4194%
answers > 15: 188 / 2480 = 7.5806%
```

The existing AROMA controller's audited numeral-probability feature proxy
covers:

```text
0..15
```

This fact does not alter the primary evaluation population.

Primary analysis:

```text
all 2480 HoloCount examples
```

Prespecified secondary count-range analyses:

```text
answer <= 15
answer > 15
```

The secondary analyses are diagnostic only and may not replace the
full-population primary result.

No threshold, feature extractor, controller coefficient, action set, or
actuator may be changed on the basis of these answer-range statistics.

---

# 5. Image Dimension Metadata

The manifest audit found:

```text
200 examples with missing metadata image width/height
0 missing image files
0 mismatches between provided metadata dimensions and actual image dimensions
```

Actual image dimensions can be deterministically obtained from the frozen
image files for every example.

Feature extraction must use the same geometry definition as the existing
AROMA controller pipeline.

No example will be removed due to missing metadata dimensions.

---

# 6. Statistical Reporting

For the frozen HoloCount external evaluation, report at minimum:

- N = 2480 question records;
- 2056 unique image-content clusters;
- baseline accuracy;
- post-controller accuracy;
- absolute percentage-point difference;
- repairs;
- breaks;
- net repairs;
- intervention rate;
- question-level paired bootstrap 95% CI;
- exact McNemar p-value;
- image-cluster bootstrap 95% CI;
- results for answer <= 15;
- results for answer > 15;
- results by the 20 official fine-grained subsets.

All analyses above are specified before first model inference.

---

# 7. Evidence Discipline

This addendum does not alter the frozen controller, actuator, model,
threshold, action set, feature representation, or primary population.

It only strengthens the statistical analysis in response to dataset
structure discovered through pre-inference provenance auditing.

No HoloCount model outcome was observed before this addendum was frozen.
