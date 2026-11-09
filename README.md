# GraphNAS — Graph Neural Architecture Search

**Status: Released (v0.3.1)**

This repository is the released reference implementation accompanying the
GraphNAS paper.  It contains the complete differentiable search space, the
training loop and the evaluation scripts used to reproduce the reported
results.

## Contents

- `src/search_space.py` — message-passing operator search space
- `src/model.py` — differentiable architecture sampler
- `src/trainer.py` — training and validation loop
- `requirements.txt` — runtime dependencies

## Quick start

```bash
pip install -r requirements.txt
python -m src.trainer --dataset citeseer --epochs 100
```

## Citation

If you use this code, please cite the GraphNAS paper.
