# AROMA Natural-Domain MN-CSA Validation v1

Frozen natural-domain confirmation experiment.

## Protocol

- Protocol:
  docs/AROMA2_NATURAL_MNCSA_PROTOCOL_v1.md

- Frozen commit:
  6ac6a904df90336bf151b3170bebd5a5170ad7d1

- Model:
  meta-llama/Llama-3.2-11B-Vision-Instruct

- Dataset:
  TallyQA natural confirmation v2

- Evaluation size:
  n=4000

## Frozen components

- L18H13 actuator
- Rank-4 U4 CSA basis
- Natural K=500 controller
- Fixed action set:
  {0,1,1.5,2,4}

## Main result

Whole-head:
67.075%

MN-CSA:
66.375%

Baseline:
64.975%

MN-CSA improves over baseline while retaining most whole-head utility.

All hashes and provenance are recorded in metadata.json.
