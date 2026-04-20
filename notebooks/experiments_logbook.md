# Experiments Logbook

## 2026-04-20
- Update loader setup in `src/train/GCN.py`; keep `SMILESDataset` unshuffled because it is an `IterableDataset`.
- Revert the logger flush experiment because it hung shutdown.
- Replace `ReLU` with `SiLU` in `src/train/GCN.py` to reduce dead and sparse gradients.
- Replace `global_max_pool` with `global_mean_pool` in `src/train/GCN.py` to make graph-level gradients denser.
- Add residual skips across the hidden GCN layers in `src/train/GCN.py` to improve gradient flow.
- Log `activation`, `pooling`, and `residual_connections` in `mlruns/mlflow_gcn.py`.
- Switch `src/train/GCN.py` from `MSELoss` to `HuberLoss(delta=0.5)` for noisier assay regression targets.
- Add gradient clipping with `clip_grad_norm_=1.0` in `src/train/GCN.py` to cap rare large updates.
- Log `loss`, `huber_delta`, and `grad_clip_norm` in `mlruns/mlflow_gcn.py`.
- Add epoch `MAE` and `RMSE` in `src/train/GCN.py` for both train and validation regression quality.
- Restrict `exports/training/export_graph.py` to exact `standard_relation='='` labels and `standard_type='IC50'` only.
- Drop `pIC` outliers in `exports/training/export_graph.py` with an IQR filter after computing `pIC`.
- Aggregate graph labels by `molregno` in `exports/training/export_graph.py` using median `pIC` before the train/valid split.
