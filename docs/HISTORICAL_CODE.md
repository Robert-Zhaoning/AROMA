# AROMA Historical Research Code

This directory documents early exploratory and development scripts produced during the evolution of AROMA.

## Why these scripts are separated

The canonical reproduction code for the final paper remains under `scripts/`.

Earlier research scripts are preserved under:

`archive/historical_scripts/`

to maintain transparency without presenting superseded exploratory code as the recommended reproduction interface.

## Provenance

These historical scripts were not all committed before their corresponding experiments.

They are therefore archived retrospectively as research snapshots and must not be interpreted as preregistered or frozen executables.

Exact file hashes are recorded in:

`archive/historical_scripts/MANIFEST.tsv`

## Main historical categories

The archive includes scripts related to:

- model and tokenizer sanity checks;
- multimodal preprocessing and cross-attention probing;
- observational attention pilots;
- causal-head discovery;
- early Proc-Count-Causal validation;
- cardinality dose-response experiments;
- DAM and oracle analyses;
- action-space development;
- early adaptive-routing/controller experiments;
- smoke tests and superseded implementations.

## Recommended reproduction path

For reproducing paper-level claims, use the tracked scripts in:

`scripts/`

together with:

- `docs/EXPERIMENT_LEDGER.md`
- `docs/RESULTS_CANONICAL.md`
- `docs/ARTIFACT_MANIFEST.md`

The historical archive is primarily for transparency, provenance, and methodological context.

