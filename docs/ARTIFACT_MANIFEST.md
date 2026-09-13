# AROMA Canonical Artifact Manifest

This manifest records the publication-critical experimental artifacts supporting the AROMA v1 paper.

The manifest intentionally excludes raw datasets, model caches, debug logs, smoke-test outputs, and nonessential intermediate files.

Early exploratory artifacts archived retrospectively are explicitly distinguished from later frozen confirmatory artifacts.

## Archive Summary

- Canonical artifacts: `23`
- Total archived size: `3.28 MB`
- Experimental milestone: `aroma-experiments-v1-complete`
- Experimental milestone commit: `ae6e08cdef84a512426268fa0e508dfe53237f14`

## Artifact Registry

### A01. `outputs/calibration50/causal_head_batch/causal_head_results.csv`

- **Evidence class:** EXPLORATORY
- **Archival status:** Retrospective historical archive
- **Purpose:** Initial causal-head intervention results.
- **Size:** `12.48 KB`
- **SHA256:** `c9d21b23f004d253179f2a5107aefd5a11c2a468d6add9765e838e107db512e8`

### A02. `outputs/calibration50/exhaustive_causal/causal_head_summary.csv`

- **Evidence class:** EXPLORATORY
- **Archival status:** Retrospective historical archive
- **Purpose:** Exhaustive causal-head screening summary.
- **Size:** `24.63 KB`
- **SHA256:** `84fce900e8b90ec66d0e59ef08576e75e679f5ba2fe4e5ce179ee44145a92d6e`

### A03. `outputs/calibration50/exhaustive_causal/stable_causal_head_summary.csv`

- **Evidence class:** EXPLORATORY
- **Archival status:** Retrospective historical archive
- **Purpose:** Stable causal-head candidates used to motivate downstream validation.
- **Size:** `30.71 KB`
- **SHA256:** `dae7f3c97c7d0b55d82297885fa04a70e5b7806da3b0ae1c0b03672bdd3cf2fb`

### A04. `outputs/proc_count_causal_v1/frozen_heads.json`

- **Evidence class:** MECHANISTIC DEVELOPMENT
- **Archival status:** Retrospective historical archive
- **Purpose:** Frozen candidate-head list following early causal discovery.
- **Size:** `870 B`
- **SHA256:** `7e42f7c1bcd5dfa10f57676bb0bbe43c1bd64dec86fc25a175746491621b66a6`

### A05. `outputs/proc_count_causal_v1/causal_validation/frozen_head_validation_summary.csv`

- **Evidence class:** MECHANISTIC VALIDATION
- **Archival status:** Retrospective historical archive
- **Purpose:** Held-out validation summary for frozen candidate heads.
- **Size:** `2.85 KB`
- **SHA256:** `152c2f3518dd186bbd86eb086b49af3604d20735b2b2045fe467907960660007`

### A06. `outputs/proc_count_causal_v1/causal_validation/paired_validation_interactions.csv`

- **Evidence class:** MECHANISTIC VALIDATION
- **Archival status:** Retrospective historical archive
- **Purpose:** Paired candidate-vs-random causal interaction evidence.
- **Size:** `58.36 KB`
- **SHA256:** `3a19020caecbb57aaf8ee6301d47defabfd8f2ecd0a914ce73d26d8b36a353ba`

### A07. `outputs/proc_count_causal_v1/router/l18h13_gain_all300/l18h13_gain_all300_summary.csv`

- **Evidence class:** CORE MECHANISTIC EVIDENCE
- **Archival status:** Retrospective historical archive
- **Purpose:** Summary of L18H13 gain intervention on 300 samples.
- **Size:** `203 B`
- **SHA256:** `46a200867592661fc5578f39443cd009f2b8294102fef071e3a4789a91762f0f`

### A08. `outputs/proc_count_causal_v1/router/l18h13_gain_all300/l18h13_gain_all300_results.csv`

- **Evidence class:** CORE MECHANISTIC EVIDENCE
- **Archival status:** Retrospective historical archive
- **Purpose:** Per-sample evidence for bidirectional cardinality steering.
- **Size:** `972.16 KB`
- **SHA256:** `a8364ddfc54b06e63b3d9de8e5734968fe0e35d56f8bc97ab5e5c7794389c2d4`

### A09. `outputs/proc_count_causal_v1/dam_head_profiles/oracle_summary.csv`

- **Evidence class:** EXPLORATORY / POST-HOC
- **Archival status:** Retrospective historical archive
- **Purpose:** Per-sample best-head oracle summary.
- **Size:** `586 B`
- **SHA256:** `03b1d184f5e948c35951b184bc9191cfe66b77f4c64a2235a50bcba70559baa1`

### A10. `outputs/proc_count_causal_v1/dam_head_profiles/oracle_head_win_counts.csv`

- **Evidence class:** EXPLORATORY / POST-HOC
- **Archival status:** Retrospective historical archive
- **Purpose:** Counts showing heterogeneous best-head selection across samples.
- **Size:** `691 B`
- **SHA256:** `f856934fb69958118da44d2796cd1b0a576a8a639a7cee6515d094acfc0e143b`

### A11. `outputs/proc_count_causal_v1/dam_head_profiles/oracle_failure_specificity_summary.csv`

- **Evidence class:** EXPLORATORY / POST-HOC
- **Archival status:** Retrospective historical archive
- **Purpose:** Failure-specificity statistics for oracle head intervention.
- **Size:** `327 B`
- **SHA256:** `3fef08ca996e7d8b4eae6be3c100fd80c391bc22020d6fb8ef87128884ac6959`

### A12. `outputs/proc_count_causal_v2/signed_steering/multistrength_oracle/multistrength_oracle_summary.csv`

- **Evidence class:** DEVELOPMENT
- **Archival status:** Retrospective publication archive
- **Purpose:** Full multistrength action-oracle headroom.
- **Size:** `206 B`
- **SHA256:** `d664595587bef5f85850e660977030e35315ac34d26be76b064f6fea79ec675d`

### A13. `outputs/proc_count_causal_v2/signed_steering/multistrength_oracle/multistrength_oracle_per_sample.csv`

- **Evidence class:** DEVELOPMENT
- **Archival status:** Retrospective publication archive
- **Purpose:** Per-sample full multistrength oracle evidence.
- **Size:** `29.21 KB`
- **SHA256:** `dbacf13e50aec4f1fec06cb32a69e98b2fc1eb48e438185e9dcc716cd7ae194f`

### A14. `outputs/proc_count_causal_v2/signed_steering/compressed_action_matrix/compressed_action_summary.csv`

- **Evidence class:** DEVELOPMENT
- **Archival status:** Retrospective publication archive
- **Purpose:** Fixed-action statistics for the final compressed action set.
- **Size:** `395 B`
- **SHA256:** `02267eb54e39087e699049a3b915dc954c8352606556232c35f6080f150bbc79`

### A15. `outputs/proc_count_causal_v2/signed_steering/compressed_action_matrix/compressed_oracle_per_sample.csv`

- **Evidence class:** DEVELOPMENT
- **Archival status:** Retrospective publication archive
- **Purpose:** Per-sample compressed five-action oracle evidence.
- **Size:** `52.25 KB`
- **SHA256:** `1d555c4251f85f2170006f987fe67eda18758a331165bbeb34bd9709a375e7fd`

### A16. `outputs/proc_count_causal_v2/controller/final_frozen_controller/aroma_cardinality_controller.joblib`

- **Evidence class:** FROZEN DEVELOPMENT ARTIFACT
- **Archival status:** Frozen before v3 confirmation
- **Purpose:** Synthetic adaptive cardinality controller evaluated on Proc-Count-Causal v3.
- **Size:** `8.24 KB`
- **SHA256:** `16a57993540eb9598e3305f079447679f1a72777847d8f9b3fe7558c9290d606`

### A17. `outputs/proc_count_causal_v2/controller/final_frozen_controller/final_hyperparameter_grid.csv`

- **Evidence class:** DEVELOPMENT
- **Archival status:** Frozen controller archive
- **Purpose:** Controller hyperparameter-development record.
- **Size:** `1.78 KB`
- **SHA256:** `3889485ed665aaa11568820168c3cdf01e571caad7ac8b86142ca768008b7f96`

### A18. `outputs/proc_count_causal_v2/controller/final_frozen_controller/selected_hyperparameter_oof.csv`

- **Evidence class:** DEVELOPMENT
- **Archival status:** Frozen controller archive
- **Purpose:** OOF predictions supporting 50.3% to 58.6% development performance.
- **Size:** `63.54 KB`
- **SHA256:** `5ccb9f645cf90fd54c5dc1839d09619cd11db8fd8c0351d9afbeb3862e1b26af`

### A19. `outputs/proc_count_causal_v3/final_frozen_controller/v3_final_summary.csv`

- **Evidence class:** INDEPENDENT CONFIRMATION
- **Archival status:** Archived after frozen evaluation
- **Purpose:** Primary Proc-Count-Causal v3 confirmatory result.
- **Size:** `438 B`
- **SHA256:** `9a1c914ad9ed8e0e144bd39b337ac2b0e22f8c13239e17d639981336f364aef4`

### A20. `outputs/proc_count_causal_v3/final_frozen_controller/v3_final_by_condition.csv`

- **Evidence class:** INDEPENDENT CONFIRMATION
- **Archival status:** Archived after frozen evaluation
- **Purpose:** Prespecified procedural confirmation breakdown by condition.
- **Size:** `465 B`
- **SHA256:** `d633367c99643a61f0c65549f6e98e98fef5476bee08de7f962ff5c125dac2f2`

### A21. `outputs/proc_count_causal_v3/final_frozen_controller/v3_final_action_distribution.csv`

- **Evidence class:** INDEPENDENT CONFIRMATION
- **Archival status:** Archived after frozen evaluation
- **Purpose:** Frozen controller action distribution on v3.
- **Size:** `100 B`
- **SHA256:** `c22d9fdbf2351e1ff673fa11b6d969c5cd18e40105d37d45cc62c31c23323407`

### A22. `outputs/proc_count_causal_v3/final_frozen_controller/v3_final_results.csv`

- **Evidence class:** INDEPENDENT CONFIRMATION
- **Archival status:** Archived after frozen evaluation
- **Purpose:** Per-sample v3 confirmation results.
- **Size:** `2.05 MB`
- **SHA256:** `15bd3fb0c7b64f528aa8da7f70f95e9d6c0c64ad63920fdcad0d60b9efa6476e`

### A23. `outputs/proc_count_causal_v3/final_frozen_controller/v3_final_run_metadata.json`

- **Evidence class:** INDEPENDENT CONFIRMATION
- **Archival status:** Archived after frozen evaluation
- **Purpose:** Metadata for the final frozen v3 run.
- **Size:** `1.70 KB`
- **SHA256:** `88f9c7fc6b9cbea210c98ced1dd5feba01130d47a9b84a48043776cd09626fc2`

## Provenance Note

Artifacts from early exploratory stages were not necessarily committed before their corresponding experiments and are therefore provided as retrospective research snapshots for transparency. Later frozen controller, evaluator, and confirmation results retain commit-before-result provenance documented in `docs/EXPERIMENT_LEDGER.md`.
