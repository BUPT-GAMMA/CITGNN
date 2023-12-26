Source code of our NeurIPS 2023  paper "Learning Invariant Representations of Graph Neural Networks via Cluster Generalization"

# Environment Settings

```
torch == 1.10.1
torch-geometric == 2.3.0
torch-scatter == 2.1.1
torch-sparse == 0.6.17
python == 3.8
```

# Usage
Go into the directory of the specific model, then run the command line.

```
python train.py cora --dropout=0.5 --p=0.2 --epochtimes=5 --clusters=100 --cuda_id=0
```
