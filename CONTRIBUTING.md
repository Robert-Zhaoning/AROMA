
# Contributing to AROMA

AROMA is a research repository with a frozen experimental record.

Contributions are welcome, but scientific provenance must be preserved.

## Before changing an experiment

For a new experimental line:

1. create a new protocol or protocol addendum;
2. define the evaluation population;
3. define the primary endpoint;
4. define any GO / NO-GO criterion before inspecting final results;
5. record the exact model revision and intervention location;
6. keep development and confirmation datasets separate.

## Frozen experiments

Do not silently modify runners or artifacts associated with frozen
confirmation results.

If a correction is necessary, preserve the original record and document the
correction explicitly.

## Outputs

New experimental outputs are ignored by default.

Do not force-add large model weights, caches, raw third-party datasets, or
routine intermediate outputs.

If a result artifact needs to become part of the permanent scientific record,
document why it is being tracked.

## Pull requests

A research pull request should explain:

```text
scientific question
protocol status
code changed
data population
expected outputs
whether the change affects a frozen claim
```

Run before submitting:

```bash
make check
```
