
# Repository Structure

```text
AROMA/
├── archive/
│   └── historical, superseded, and development-stage code
│
├── configs/
│   ├── experiment configuration
│   └── tracked_outputs_allowlist.txt
│
├── data/
│   └── generated/local data
│
├── docs/
│   ├── frozen protocols
│   ├── freeze records
│   ├── result records
│   ├── experiment ledger
│   ├── artifact manifest
│   └── reproducibility documentation
│
├── environment/
│   └── environment metadata
│
├── external/
│   └── third-party local resources
│
├── manifests/
│   └── dataset and experiment manifests
│
├── outputs/
│   └── selected tracked artifacts plus ignored local outputs
│
├── scripts/
│   └── canonical experiment and analysis runners
│
├── src/
│   └── reusable AROMA package code
│
└── tests/
    └── lightweight automated tests
```

## Canonical versus historical code

Canonical experiment runners belong in:

```text
scripts/
```

Superseded, diagnostic, pilot, or historical runners belong in:

```text
archive/
```

Frozen protocols and result documents define which runner/artifact pair is
authoritative for each headline result.
