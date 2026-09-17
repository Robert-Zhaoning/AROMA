
# Reference Environment

The primary AROMA experiments were run with the following reference software
environment.

```text
Python       3.11.16
PyTorch      2.11.0+cu128
Transformers 5.17.0
Accelerate   1.15.0
NumPy        2.4.6
Pandas       3.0.5
SciPy        1.17.1
scikit-learn 1.9.0
Pillow       12.3.0
PyYAML       6.0.3
tqdm         4.70.0
Matplotlib   3.11.1
joblib       1.6.0
CUDA         12.8
```

Primary model:

```text
meta-llama/Llama-3.2-11B-Vision-Instruct
```

Frozen revision:

```text
9eb2daaa8597bf192a8b0e73f848f3a102794df5
```

The final CSA experiments were run on a large-memory NVIDIA Blackwell-class
GPU environment.

Earlier experiments also used other large-memory RunPod GPU instances.

Exact hardware identity should not be interpreted as part of the scientific
method unless explicitly frozen in an experiment protocol.

## Local installation

The lightweight package can be installed with:

```bash
python -m pip install -e .
```

Model-facing experiment runners additionally require the model stack listed
above and sufficient GPU memory.

Large pretrained model weights are not stored in this repository.
