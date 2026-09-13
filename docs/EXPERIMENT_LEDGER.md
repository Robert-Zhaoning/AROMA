# AROMA Final Experiment Ledger

**Project:** AROMA  
**Experimental milestone:** `aroma-experiments-v1-complete`  
**Milestone target:** `ae6e08cdef84a512426268fa0e508dfe53237f14`  
**Status:** Experimental development complete; publication consolidation in progress.

---

## 0. Purpose and Evidence Taxonomy

This ledger is the canonical record of the experiments supporting the AROMA paper.

The project evolved from broad stage-aware vision-language counting repair into a narrower and more deeply validated study of **causal cardinality control, adaptive repair, domain shift, and sample-efficient natural-domain adaptation**.

Every experiment is assigned one evidence class:

| Evidence class | Meaning |
|---|---|
| `PILOT` | Instrumentation, sanity checks, or early feasibility experiments |
| `EXPLORATORY` | Hypothesis-generating analyses not frozen before evaluation |
| `DEVELOPMENT` | Used to select features, actions, thresholds, controllers, or adaptation choices |
| `POST-HOC FORENSIC` | Analysis performed after observing a result to diagnose mechanism or failure |
| `FROZEN CONFIRMATORY` | Protocol/model/controller fixed before inference on the target evaluation set |
| `INDEPENDENT CONFIRMATION` | Frozen evaluation on a previously untouched held-out sample |

Important terminology:

- "Independent natural confirmation" refers to an **image-, image-ID-, and question-disjoint held-out TallyQA sample from the same official benchmark distribution**.
- It is **not** claimed as cross-dataset generalization.
- Early historical scripts that were not committed before their corresponding experiments are preserved for transparency but are not described as preregistered or frozen executables.

---

## 1. Core Model and Experimental Setting

Primary model:

`meta-llama/Llama-3.2-11B-Vision-Instruct`

Primary mechanistic intervention:

- Cross-attention head gain manipulation.
- Final cardinality actuator: **Layer 18, Head 13 (L18H13)**.
- Identity action: `alpha = 1`.

Numeral-proxy provenance:

- Early L18H13 300-sample directional characterization used the original numeral proxy over `1, ..., 10`.
- The later expanded proxy used by the v2 controller-development and subsequent evaluation pipeline covers `0, ..., 15`.
- Expanded-proxy tokenization and generation concordance were separately audited before downstream controller use.

For the expanded proxy, expected cardinality is:

\[
\mathbb{E}[N] = \sum_{n=0}^{15} n\,p(n).
\]

Final controller action set:

\[
\mathcal{A} = \{0,\ 1,\ 1.5,\ 2,\ 4\}.
\]

Final GT-free controller representation:

- 39 baseline/numeral-geometry features.
- Per-action `StandardScaler -> Ridge`.
- Ridge coefficient: `0.01`.
- Intervention threshold: `0.1`.
- `alpha = 1` is NOOP.

---

## 2. Sanity Checks and Instrumentation

**Evidence class:** `PILOT`

### Scientific question

Can the model, tokenizer, multimodal preprocessing stack, cross-attention implementation, and counting prompt be instrumented reliably enough for causal experimentation?

### Main activities

- Model loading and numeral-token audits.
- Prompt calibration.
- Enumeration/counting sanity tests.
- Cross-attention source inspection.
- Attention geometry inspection.
- Synthetic calibration-set generation.

### Historical scripts

Examples include:

- `scripts/test_llama_load.py`
- `scripts/test_llama_count.py`
- `scripts/test_llama_enumeration.py`
- `scripts/test_prompt_calibration.py`
- `scripts/audit_llama_tokens.py`
- `scripts/audit_llama_attention_geometry.py`
- `scripts/probe_mllama_cross_attention_source.py`
- `scripts/probe_mllama_preprocessing_geometry.py`
- `scripts/generate_calibration50.py`
- `scripts/run_calibration50.py`

### Archival status

These scripts currently exist as historical research snapshots and were not all committed before execution.

### Paper placement

Mostly omitted from the main paper. Relevant implementation checks belong in the reproducibility appendix.

---

## 3. Observational Attention Pilot

**Evidence class:** `EXPLORATORY`

### Scientific question

Do observational attention statistics separate successful and failed counting examples strongly enough to identify a repair mechanism?

### Main activities

- Attention pilot.
- GT patch masks.
- Matched correct/wrong attention pairs.
- Attention metrics and behavioral labels.

### Main conclusion

Observational attention alone was insufficiently reliable for identifying a robust repair mechanism.

This motivated a shift from correlational attention analysis to **causal intervention screening**.

### Historical scripts

Examples:

- `scripts/run_attention_pilot.py`
- `scripts/analyze_attention_pilot.py`
- `scripts/build_gt_patch_masks.py`
- `scripts/build_matched_attention_pairs.py`
- `scripts/run_matched_attention.py`
- `scripts/analyze_matched_attention.py`
- `scripts/analyze_matched_attention_by_label.py`
- `scripts/extract_gt_attention_metrics.py`

### Paper placement

Motivation / appendix. Do not present observational attention as the primary mechanistic evidence.

---

## 4. Causal Cross-Attention Head Discovery

**Evidence class:** `EXPLORATORY MECHANISTIC DISCOVERY`

### Scientific question

Which cross-attention heads causally affect counting behavior, especially in failure states?

### Dataset

Proc-Count-Causal v1 / calibration-based matched examples.

### Candidate heads discovered

The exhaustive causal screen identified the following high-value candidates:

- L33H1
- L3H4
- L18H13
- L8H30
- L3H11
- L13H11
- L33H21

### Key result

Held-out candidate-vs-random failure-state interaction:

\[
\text{mean interaction} \approx -0.285932
\]

with bootstrap 95% CI approximately:

\[
[-0.453263,\ -0.108896].
\]

This supported failure-state-dependent causal sensitivity rather than purely observational association.

### Historical scripts

- `scripts/run_causal_head_batch.py`
- `scripts/analyze_causal_head_batch.py`
- `scripts/run_exhaustive_causal_screen.py`
- `scripts/analyze_exhaustive_causal_screen.py`
- `scripts/causal_head_ablation_smoke.py`
- `scripts/freeze_proc_count_heads.py`

### Archival status

Early mechanistic-discovery scripts are historical snapshots and were not all committed before execution.

### Allowed claim

A causal screen identified a small set of cross-attention heads whose intervention effects were enriched in counting-failure states.

### Paper placement

Main mechanistic-discovery section, with exhaustive details in appendix.

---

## 5. Multi-Head Heterogeneity and Oracle Analysis

**Evidence class:** `EXPLORATORY / POST-HOC MECHANISTIC`

### Scientific question

Does one fixed head explain all repairable failures, or is useful intervention heterogeneous across samples?

### Key finding

For 27 held-out matched pairs, a per-sample best-head oracle produced approximately:

\[
\Delta \log p_{\text{wrong}} = +0.093235
\]

with 95% CI:

\[
[+0.061318,\ +0.126934].
\]

For correct samples:

\[
\Delta \log p_{\text{correct}} \approx +0.012637.
\]

Failure specificity:

\[
\approx +0.080598
\]

with 95% CI approximately:

\[
[+0.045740,\ +0.116334].
\]

All seven candidate heads were optimal for at least some samples.

### Interpretation

Useful causal intervention is heterogeneous across samples. This provided early evidence that **routing** is necessary.

### Historical scripts

- `scripts/analyze_dam_head_profiles.py`
- `scripts/profile_dam_head_responses.py`
- `scripts/oracle_dam_v1_smoke.py`
- `scripts/oracle_dam_v2_smoke.py`
- `scripts/run_oracle_dam_v1_batch.py`
- `scripts/analyze_oracle_dam_v1_batch.py`
- `scripts/map_dam_pairwise_interactions.py`
- `scripts/search_dam_compatible_subsets.py`

### Paper placement

Appendix / mechanistic motivation.

---

## 6. Bidirectional Cardinality Steering with L18H13

**Evidence class:** `CORE MECHANISTIC EVIDENCE`

### Scientific question

Does L18H13 behave as a directional cardinality-control actuator rather than as a generic performance-sensitive head?

### Intervention

Scale the contribution of L18H13 with gain `alpha`.

### Core result on 300 samples

These directional statistics were computed from the original `1,...,10` numeral-proxy artifact. The later `0,...,15` expanded proxy was introduced and audited separately before controller development.

For `alpha = 0.5`:

\[
274/300 = 91.33\%
\]

of samples shifted expected numeral downward.

For `alpha = 1.5`:

\[
272/300 = 90.67\%
\]

shifted expected numeral upward.

Strict ordering:

\[
\mathbb{E}[N]_{0.5}
<
\mathbb{E}[N]_{1}
<
\mathbb{E}[N]_{1.5}
\]

held on:

\[
266/300 = 88.67\%.
\]

Downward/upward response magnitudes had Spearman correlation approximately:

\[
\rho \approx 0.9514.
\]

### Interpretation

L18H13 provides a **bidirectional, dose-responsive causal cardinality actuator**.

### Related tracked script

- `scripts/run_l18h13_gain_all300.py`

### Historical supporting scripts

- `scripts/run_proc_count_gain_dose_response.py`
- `scripts/analyze_proc_count_gain_dose_response.py`
- `scripts/run_proc_count_circuit_gain_generation.py`
- `scripts/analyze_proc_count_circuit_gain_generation.py`
- expanded-numeral audit scripts

### Allowed claim

Intervening on L18H13 causally and directionally shifts the model's internal numeral distribution.

### Paper placement

Main paper; central mechanistic result.

---

## 7. Proc-Count-Causal v2 Action-Space Development

**Evidence class:** `DEVELOPMENT`

### Scientific question

How much repair headroom is available from multiple intervention strengths, and what compact action space should the adaptive controller use?

### Baseline

Proc-Count-Causal v2:

\[
\text{accuracy} = 0.503.
\]

There were 497 baseline-wrong examples.

Error spectrum among wrong examples:

- `-3`: 12
- `-2`: 153
- `-1`: 301
- `+1`: 31

The development distribution was therefore strongly undercount-dominated.

### Multistrength oracle

Actions evaluated included:

\[
\alpha \in
\{0,\ 0.25,\ 0.5,\ 0.75,\ 1,\ 1.25,\ 1.5,\ 2,\ 3,\ 4\}.
\]

Unique repairable wrong examples:

\[
139/497 = 27.97\%.
\]

Oracle accuracy:

\[
0.642,
\]

for a gain of:

\[
+13.9 \text{ pp}.
\]

### Final compressed action set

\[
\boxed{\{0,\ 1,\ 1.5,\ 2,\ 4\}}.
\]

Compressed oracle:

\[
0.503 \rightarrow 0.641,
\]

a gain of:

\[
+13.8 \text{ pp}.
\]

### Core scripts

Tracked:

- `scripts/build_v2_compressed_action_matrix.py`
- `scripts/run_v2_baseline_utility_controller.py`
- `scripts/generate_proc_count_causal_v2.py`
- `scripts/generate_proc_count_causal_v2_unique.py`

Historical:

- `scripts/run_v2_multistrength_wrong_sweep.py`
- `scripts/analyze_v2_multistrength_oracle.py`
- `scripts/analyze_v2_three_action_oracle.py`
- `scripts/run_v2_compressed_actions_correct.py`

### Paper placement

Method-development / appendix.

---

## 8. Frozen Synthetic Adaptive Controller Development

**Evidence class:** `DEVELOPMENT`

### Representation

39 GT-free baseline and numeral-geometry features.

### Model family

For each non-NOOP action:

`StandardScaler -> Ridge`

with:

- Ridge alpha: `0.01`
- threshold: `0.1`
- actions: `{0, 1, 1.5, 2, 4}`

### Development performance

Nested/grouped OOF on Proc-Count-Causal v2:

\[
0.503 \rightarrow 0.586
\]

for:

\[
\boxed{+8.3\text{ pp}}.
\]

Repairs:

\[
91
\]

Breaks:

\[
8
\]

Intervention rate:

\[
0.333.
\]

### Frozen controller

Artifact:

`outputs/proc_count_causal_v2/controller/final_frozen_controller/aroma_cardinality_controller.joblib`

Config:

`configs/aroma_cardinality_controller_frozen.json`

Freeze commit:

`c8c5ede1601b1ada4087e141d61247903a4458a7`

Canonical tag:

`aroma-controller-pre-v3-freeze`

### Allowed claim

A GT-free adaptive controller was developed on Proc-Count-Causal v2 and frozen before evaluation on Proc-Count-Causal v3.

---

## 9. Frozen Proc-Count-Causal v3 Confirmation

**Evidence class:** `INDEPENDENT CONFIRMATION`

### Scientific question

Does the frozen synthetic controller improve counting accuracy on a newly generated procedural dataset not used for controller development?

### Freeze chain

Controller freeze:

`c8c5ede1601b1ada4087e141d61247903a4458a7`

Dataset protocol freeze:

`582bb53fe51d8b8bbeb45b90b7d073253d4df2a1`

Final runner freeze:

`9085d093a5d2a46993e5dc8576de9e50a9c21c55`

Result archive:

`9bc192e7cc70cf754b0446672bb64b016490a746`

Canonical tag:

`aroma-v3-final-confirmation`

### Dataset

2,000 procedural samples.

- 400 per condition.
- 200 per count from 1 to 10.
- No v1/v2 image or seed overlap.
- 2,000 unique image hashes.
- 2,000 unique seeds.

### Primary result

Baseline:

\[
999/2000 = 49.95\%.
\]

Frozen controller:

\[
1161/2000 = 58.05\%.
\]

Gain:

\[
\boxed{+8.10\text{ pp}}.
\]

95% CI:

\[
[+6.85,\ +9.40]\text{ pp}.
\]

Repairs:

\[
173
\]

Breaks:

\[
11
\]

Net repairs:

\[
+162.
\]

Intervention rate:

\[
32.65\%.
\]

### By condition

- dense: `+10.25 pp`
- distractors: `+7.25 pp`
- grid: `+12.00 pp`
- random_sparse: `+7.00 pp`
- row: `+4.00 pp`

### Interpretation

The frozen controller generalizes strongly within the procedural counting domain.

### Paper placement

Main paper; primary procedural confirmation.

---

## 10. Frozen TallyQA Zero-Shot Natural OOD Evaluation

**Evidence class:** `FROZEN CONFIRMATORY NEGATIVE RESULT`

### Scientific question

Does the synthetic-trained controller transfer zero-shot to natural-image counting?

### Evaluation set

4,000 TallyQA samples:

- 2,000 Simple
- 2,000 Complex
- 4,000 unique images

### Freeze chain

Manifest freeze:

`4cedd7297280b65f80d7a97c333d2e31b55aeeec`

Image inventory freeze:

`17fe2aaabc4e0ae32acef4eef64c6e6ead57833f`

Evaluator freeze:

`6f04635c2aa53bfdf0867fc1d6ac35c39c03c3d4`

Result archive:

`bdc740bda8482b5a8dfa25e9e110ac2c819359a8`

Tags:

- `aroma-tallyqa-ood-final-freeze`
- `aroma-tallyqa-ood-zero-shot-result`

### Primary result

Baseline:

\[
64.85\%.
\]

Frozen synthetic controller:

\[
61.875\%.
\]

Change:

\[
\boxed{-2.975\text{ pp}}.
\]

95% CI:

\[
[-3.70,\ -2.25]\text{ pp}.
\]

Repairs:

\[
51
\]

Breaks:

\[
170
\]

Net:

\[
-119.
\]

Intervention rate:

\[
65.275\%.
\]

Exact McNemar:

\[
p \approx 3.78\times10^{-16}.
\]

### Interpretation

The frozen synthetic policy fails to transfer zero-shot to natural counting.

This result is retained as a primary falsification result and must not be hidden or relabeled.

### Paper placement

Main paper.

---

## 11. Natural Action-Oracle Forensic Analysis

**Evidence class:** `POST-HOC FORENSIC`

### Scientific question

Did the underlying cardinality actuator cease to be useful on natural images, or did the routing/controller fail?

### Result

Natural multi-action oracle:

\[
64.85\% \rightarrow 71.575\%
\]

for:

\[
\boxed{+6.725\text{ pp}}.
\]

Unique repairable baseline-wrong samples:

\[
269/1406 = 19.13\%.
\]

### Fixed-action observation

`alpha = 0` alone achieved:

\[
64.85\% \rightarrow 67.20\%
\]

for:

\[
+2.35\text{ pp}.
\]

### Interpretation

The actuator retains meaningful natural-domain repair opportunity.

Therefore:

\[
\boxed{\text{mechanism transfer} \neq \text{controller transfer}}.
\]

### Provenance

Forensic freeze:

`76c3382b41d9acabfd7df03c8957d217e833e297`

Summary archive:

`11da4742529e7d57deab4b0ce3ac0921c678bca7`

Result archive:

`d74c2572d8a5ea89ce755823536d6d78f0f47144`

### Paper placement

Main analysis section / appendix details.

---

## 12. Natural Error-Direction Shift

**Evidence class:** `POST-HOC FORENSIC`

### Result

Among 1,406 natural baseline errors:

- undercount: `297 = 21.12%`
- overcount: `1109 = 78.88%`

Absolute-error-one examples:

\[
1017/1406 = 72.33\%.
\]

The synthetic development distribution had instead been strongly undercount-dominated.

### Interpretation

Natural-domain counting errors have a substantially different direction composition from the synthetic controller-development distribution.

This helps explain why the synthetic routing policy selected harmful amplification actions too often.

### Provenance

`104c7b5b3e76d7951134c768bcb2d6b72cf057da`

### Paper placement

Main diagnostic analysis.

---

## 13. Controller Domain Shift

**Evidence class:** `POST-HOC FORENSIC`

### Key observation

Controller selected-score utility correlation remained positive on synthetic data but collapsed on TallyQA.

Approximate selected-score / realized-utility Spearman:

- v2/v3: `~ +0.41`
- TallyQA: `~ -0.067`

Intervention rate shifted from approximately one-third on synthetic data to:

\[
65.275\%
\]

on TallyQA.

### Feature shift examples

Large shifts occurred in:

- baseline best numeral
- expected numeral
- numeral-vocabulary mass

### Interpretation

The natural failure is a routing/calibration problem, not simply absence of an actuator effect.

### Provenance

`c1b946e248b3787f2a6fa27b7b7f3fe2db9cdcac`

### Paper placement

Main diagnostic analysis.

---

## 14. Exact Ridge OOD Attribution

**Evidence class:** `POST-HOC FORENSIC`

### Scientific question

Why did controller scores become extremely large and poorly calibrated on natural inputs?

### Exact reproduction

Frozen per-action controller scores were reproduced to approximately machine precision with zero decision mismatches.

### Key pathology

Extremely small synthetic training scales for rare high numerals caused massive standardized natural-domain values.

Example:

`baseline_numprob_15`

Synthetic scaler statistics:

- mean approximately `5.09e-07`
- scale approximately `6.97e-07`

An extreme natural sample had raw value approximately:

\[
0.351970,
\]

producing standardized magnitude of roughly:

\[
5.0\times10^5.
\]

This single feature could contribute tens of thousands to a Ridge action score.

### Interpretation

`StandardScaler + linear Ridge` created severe numerical OOD extrapolation.

### Provenance

`306c9b6c60c29d3cd64a6f7ea5665044d033e43f`

Exact training-support reconstruction:

`5f6abae89cf7bb0efeaef29ee19ba43a4d584908`

### Paper placement

Appendix, with concise summary in main text.

---

## 15. SPUC-v1 Support Projection

**Evidence class:** `POST-HOC FORENSIC / SAFETY ABLATION`

### Method

Clip standardized natural features to empirical v2 training-support bounds while retaining:

- original Ridge coefficients
- original intercepts
- original threshold
- original action set

No fitting or tuning was performed.

### Result

SPUC removed catastrophic score explosion but did not improve final natural accuracy:

\[
61.875\% \rightarrow 61.875\%
\]

relative to the original frozen controller outcome.

Natural samples with any clipped feature:

\[
1593/4000 = 39.825\%.
\]

However, even fully in-support natural samples still exhibited negative controller utility.

### Interpretation

Numeric support violation explains score explosion but does **not** explain the deeper conditional-utility failure.

### Provenance

Development result:

`f5b4c9dc430ffe82493afdca0bb8598e2c7c3d2b`

Residual calibration archive:

`da0f7b98722d1c1155dbedde4edc6ab036cea222`

### Paper placement

Appendix / failure-analysis section.

---

## 16. Natural Utility Learnability

**Evidence class:** `POST-HOC DEVELOPMENT`

### Scientific question

Does the same 39-dimensional GT-free representation still contain useful information for predicting intervention utility in the natural domain?

### Protocol

- Same 39 features.
- Same `StandardScaler -> Ridge` model family.
- Ridge alpha `0.01`.
- Threshold `0.1`.
- Same action set.
- 5-fold OOF by Simple/Complex.
- No hyperparameter search.

### Result

Baseline:

\[
64.85\%.
\]

OOF natural utility controller:

\[
67.15\%.
\]

Gain:

\[
\boxed{+2.30\text{ pp}}.
\]

Repairs:

\[
126
\]

Breaks:

\[
34
\]

Intervention rate:

\[
16.45\%.
\]

Per-action score/utility correlations became positive again.

### Interpretation

The representation still contains natural-domain utility signal.

The primary shift is therefore consistent with:

\[
P(u_a\mid\phi,D_{\text{synthetic}})
\neq
P(u_a\mid\phi,D_{\text{natural}}).
\]

### Provenance

Probe archive:

`dbf55c40e1ed52180d83dc6a9d188e90eb8b44a3`

Consolidated archive:

`ccba8e75a8b504fdb83fbacc745f57306fcae4f3`

### Paper placement

Main adaptation-development analysis.

---

## 17. Natural Adaptation Sample-Efficiency Curve

**Evidence class:** `POST-HOC DEVELOPMENT`

### Protocol

Natural calibration sizes:

\[
K \in
\{50,100,250,500,1000,2000,3200\}.
\]

Same features, model family, action space, ridge value, and threshold.

### Mean gain by calibration size

| K | Mean gain |
|---:|---:|
| 50 | +0.330 pp |
| 100 | +1.180 pp |
| 250 | +1.870 pp |
| 500 | +2.120 pp |
| 1000 | +2.240 pp |
| 2000 | +2.325 pp |
| 3200 | +2.300 pp |

At `K = 500`, all 25 predefined development runs were positive.

This is not interpreted as an independent probability estimate because runs share the same development pool and fold structure.

### Development decision

`K = 500` was selected as a sample-efficiency elbow rather than as the maximum-performing K.

It recovered approximately:

\[
2.12/2.30 \approx 92\%
\]

of the full-data OOF gain.

### Provenance

`ccba8e75a8b504fdb83fbacc745f57306fcae4f3`

### Paper placement

Main paper or appendix depending on space.

---

## 18. K=500 Natural-Adapted Controller Freeze

**Evidence class:** `FROZEN DEVELOPMENT ARTIFACT`

### Calibration set

500 natural development examples:

- 250 Simple
- 250 Complex

Selected deterministically using metadata-only SHA256 ranking.

No outcome-based sample selection.

### Controller

- 39 features
- `StandardScaler -> Ridge`
- ridge alpha `0.01`
- threshold `0.1`
- actions `{0,1,1.5,2,4}`
- actuator `L18H13`

### Bundle

`outputs/phase2_natural_controller_k500/final_frozen_controller/aroma_natural_utility_controller_k500.joblib`

Bundle SHA256:

`d9d8ba3a24814bf71b3f4039d1effa864b57754a00539f1260a1d077c03449f5`

Calibration-manifest SHA256:

`67dfdafa36c66a1299be7adea0d1232269e1b557607314972ed2b298312dc49a`

### Freeze commit

`9bbd337a979029da1bc46396c8e266da5c9e6193`

The controller was frozen before the new natural confirmation result was observed.

---

## 19. Untouched TallyQA Natural Confirmation v2

**Evidence class:** `INDEPENDENT CONFIRMATION`

### Scientific question

Does the K=500 natural-adapted controller generalize to a previously untouched, image-disjoint natural confirmation set?

### Confirmation set

4,000 TallyQA examples:

- 2,000 Simple
- 2,000 Complex
- 4,000 unique question IDs
- 4,000 unique image paths
- 4,000 unique image IDs

There was zero overlap with the previous TallyQA development/evaluation set in:

- question ID
- image path
- image ID

This is a held-out confirmation from the same TallyQA benchmark distribution, not a separate dataset.

### Freeze chain

K500 controller freeze:

`9bbd337a979029da1bc46396c8e266da5c9e6193`

Manifest freeze:

`9fc7822beb9e224291591a89b46a182c7c1ffe01`

Image inventory freeze:

`9dfd1e8ce62b745227aade37edb980e34016318d`

Final evaluator freeze:

`ca0fb4274d147b8e70e641fecfdd9961ba9da3db`

Result archive:

`ae6e08cdef84a512426268fa0e508dfe53237f14`

Experimental milestone tag:

`aroma-experiments-v1-complete`

### Prespecified success criterion

\[
\Delta Acc > 0
\quad\land\quad
p_{\text{exact McNemar}} < 0.05.
\]

### Primary result

Baseline:

\[
2597/4000 = 64.925\%.
\]

K500 controller:

\[
2682/4000 = 67.050\%.
\]

Gain:

\[
\boxed{+2.125\text{ pp}}.
\]

Bootstrap 95% CI:

\[
\boxed{[+1.50,\ +2.775]\text{ pp}}.
\]

Repairs:

\[
130
\]

Breaks:

\[
45
\]

Net repairs:

\[
+85.
\]

Intervention rate:

\[
20.0\%.
\]

Exact McNemar:

\[
\boxed{p = 9.176009\times10^{-11}}.
\]

Primary success:

`TRUE`

### Prespecified Simple result

\[
78.05\%
\rightarrow
79.15\%
\]

Gain:

\[
+1.10\text{ pp}
\]

95% CI:

\[
[+0.25,\ +1.95]\text{ pp}.
\]

Repairs / breaks:

\[
49/27.
\]

Intervention rate:

\[
11.8\%.
\]

### Prespecified Complex result

\[
51.80\%
\rightarrow
54.95\%
\]

Gain:

\[
\boxed{+3.15\text{ pp}}.
\]

95% CI:

\[
[+2.20,\ +4.15]\text{ pp}.
\]

Repairs / breaks:

\[
81/18.
\]

Intervention rate:

\[
28.2\%.
\]

### Action distribution

| Alpha | Count | Fraction |
|---:|---:|---:|
| 0 | 508 | 12.70% |
| 1 | 3200 | 80.00% |
| 1.5 | 32 | 0.80% |
| 2 | 122 | 3.05% |
| 4 | 138 | 3.45% |

### Interpretation

A lightweight natural-domain calibration of 500 examples restored positive intervention routing and generalized to a new image-disjoint TallyQA confirmation sample.

The development OOF gain was:

\[
+2.30\text{ pp}
\]

and the independent confirmation gain was:

\[
+2.125\text{ pp},
\]

showing close agreement between development and held-out confirmation.

### Canonical artifacts

- `outputs/tallyqa_natural_confirmation_v2/final_frozen_controller/tallyqa_final_summary.csv`
- `outputs/tallyqa_natural_confirmation_v2/final_frozen_controller/tallyqa_final_results.csv`
- `outputs/tallyqa_natural_confirmation_v2/final_frozen_controller/tallyqa_final_by_subset.csv`
- `outputs/tallyqa_natural_confirmation_v2/final_frozen_controller/tallyqa_final_action_distribution.csv`

Canonical summary SHA256:

`1cd7d40d03a8be036d9c0447d0c3c1c10c16008978b2f2046a59c932bd331700`

Canonical per-sample result SHA256:

`8a03c6d888f1b0a3e664e86f0a541e5a9d77b23a983e7383000773b4cba574d9`

### Allowed claim

The frozen K=500 natural-adapted controller significantly improves counting accuracy on an untouched image-disjoint TallyQA confirmation sample.

Do not describe this as cross-dataset generalization.

### Paper placement

Main paper; final confirmatory result.

---

## 20. Canonical Scientific Narrative

The evidence supports the following sequence:

\[
\text{Causal mechanism discovery}
\rightarrow
\text{bidirectional cardinality steering}
\rightarrow
\text{adaptive procedural repair}
\rightarrow
\text{successful frozen procedural confirmation}
\]

followed by:

\[
\text{zero-shot natural failure}
\rightarrow
\text{actuator-vs-router separation}
\rightarrow
\text{domain-shift diagnosis}
\rightarrow
\text{natural utility adaptation}
\rightarrow
\text{successful untouched natural confirmation}.
\]

The strongest interpretation is not that a universal counting controller transfers unchanged across domains.

Instead:

1. A causally controllable cardinality actuator exists.
2. Its intervention utility is sample dependent.
3. A synthetic-trained utility mapping can fail under natural-domain conditional shift.
4. The underlying actuator retains repair potential.
5. A small amount of natural-domain calibration can restore useful routing.
6. The adapted policy generalizes to a new image-disjoint natural confirmation sample.

---

## 21. Claims That Must NOT Be Made

The current experiments do **not** establish:

- full implementation of all originally proposed P/R/M stage-specific repair modules;
- universal VLM counting repair;
- cross-model generalization;
- cross-dataset natural generalization;
- zero-shot natural success;
- that observational attention alone identifies the causal mechanism;
- that K=500 was chosen using the final confirmation set;
- that post-hoc forensic analyses were prespecified confirmatory experiments.

These limitations must remain explicit in the paper.

---

## 22. Publication Provenance Policy

The following frozen / confirmatory history must not be rewritten, rebased, amended, or force-pushed:

- `c8c5ede...` — synthetic controller freeze
- `582bb53...` — v3 dataset protocol freeze
- `9085d09...` — v3 evaluator freeze
- `9bc192e...` — v3 result archive
- `4cedd72...` — natural OOD manifest freeze
- `17fe2aa...` — natural OOD image inventory freeze
- `6f04635...` — natural OOD evaluator freeze
- `bdc740b...` — natural OOD negative result
- `9bbd337...` — K500 natural controller freeze
- `9fc7822...` — natural confirmation v2 manifest freeze
- `9dfd1e8...` — natural confirmation v2 image inventory freeze
- `ca0fb42...` — natural confirmation v2 evaluator freeze
- `ae6e08c...` — final natural confirmation archive

Canonical experimental milestone:

`aroma-experiments-v1-complete`

Historical exploratory scripts archived later must be clearly distinguished from this frozen provenance chain.

---

## 23. Current Project State

Experimental method development for AROMA v1 is complete.

Future work may include:

- second-model replication;
- second-dataset natural generalization;
- broader stage-specific repair modules;
- richer nonlinear or uncertainty-aware routing.

These are extensions rather than requirements for the integrity of the current experimental story.

The current publication effort should prioritize:

- canonical result archival;
- reproducibility;
- figures and tables;
- paper narrative;
- appendix;
- consistency checking.

