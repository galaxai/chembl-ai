# Experiments Logbook

## 2026-04-20
- Update loader setup in `src/train/GCN.py`; keep `SMILESDataset` unshuffled because it is an `IterableDataset`.
- Revert the logger flush experiment because it hung shutdown.
- Replace `ReLU` with `SiLU` in `src/train/GCN.py` to reduce dead and sparse gradients.
- Replace `global_max_pool` with `global_mean_pool` in `src/train/GCN.py` to make graph-level gradients denser.
- Add residual skips across the hidden GCN layers in `src/train/GCN.py` to improve gradient flow.
- Log `activation`, `pooling`, and `residual_connections` in `mlruns/mlflow_gcn.py`.
