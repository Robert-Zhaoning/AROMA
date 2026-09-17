# Pre-Freeze CSA Development Scripts

This directory preserves three development-stage scripts created during the
AROMA Causal Subspace Actuation (CSA) investigation on 2026-09-15.

They are retained for provenance, but they are **not canonical runners** for
the final paper results.

## Files

### `run_csa_head_gradient_pilot_v1.py`

Early single-example pilot used to verify that gradients could be recovered
with respect to the L18H13 head-aligned activation slice.

This script predates the frozen Gate-A protocol and should not be used to
reproduce final CSA statistics.

---

### `run_csa_gate_a_v1.py`

Early implementation of the CSA Gate-A experiment.

The scientific procedure was subsequently revised and frozen through later
Gate-A methodology and implementation records. Final Gate-A claims must be
taken from the frozen documentation and final artifacts, not this v1 runner.

Relevant canonical documentation includes:

- `docs/AROMA2_CSA_PROTOCOL_v1_1.md`
- `docs/AROMA2_CSA_GATEA_FREEZE_v2.md`

---

### `replay_canonical_l18h13_baseline_v1.py`

Development diagnostic for replaying the original L18H13 baseline and checking
alignment with archived canonical outputs.

It was used as an implementation/provenance diagnostic rather than as a
standalone headline experiment.

---

## Status

These scripts are archived to preserve the experimental development history.

They should not be modified to generate new headline results.

For current results, use the frozen protocols, result records, and canonical
artifacts documented under `docs/`.
