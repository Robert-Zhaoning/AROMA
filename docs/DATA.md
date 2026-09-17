
# Data Policy and Dataset Structure

AROMA uses both generated procedural data and external natural-image counting
benchmarks.

## Procedural datasets

The Proc-Count-Causal sequence is used to separate experimental roles:

```text
v1  mechanism discovery / directional characterization
v2  controller development
v3  frozen procedural confirmation and later CSA development
v4  final untouched MN-CSA confirmation
```

The final v4 generation audit records:

```text
N = 2000
unique seeds = 2000
unique rendered-image hashes = 2000
zero seed overlap with v1/v2/v3
zero rendered-image-hash overlap with v1/v2/v3
```

Generation manifests and result records should be treated as the authoritative
source for dataset provenance.

## Natural datasets

Natural-domain experiments include:

```text
TallyQA
HoloCount
```

Third-party datasets are not redistributed by this repository unless their
licenses explicitly permit redistribution.

Users are responsible for obtaining external datasets from their official
sources.

## Git policy

New files under:

```text
data/
external/
outputs/
```

are ignored by default.

A limited set of historical result artifacts is intentionally tracked for
scientific provenance.

The exact existing tracked-output inventory is frozen in:

```text
configs/tracked_outputs_allowlist.txt
```

Repository-integrity checks reject accidental expansion of this inventory.
