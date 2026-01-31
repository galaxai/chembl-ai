# ChEMBL AI

Pipeline to export ChEMBL 36 data from SQLite into Parquet and train baseline models on activity data (pIC). The repo includes Spark-based preprocessing, RDKit feature generation (Morgan fingerprints and molecular graphs), and simple training scripts.

## What is in here
- Exporters that query the ChEMBL 36 SQLite database and write Arrow/Parquet datasets.
- Spark jobs that join activity/assay/structure data and create train/val splits.
- Baseline models: tinygrad MLP on Morgan fingerprints, torch-geometric GCN on graphs.
- Local Spark + Airflow + MLflow stack in `spark_airflow/`.

## Requirements
- Python >= 3.12
- Spark Connect endpoint (default `sc://localhost:15002`) for Spark jobs
- RDKit for fingerprint/graph exports
- Optional: Torch + torch-geometric for GCN training

## Setup
Recommended with uv:

```bash
uv sync
```

If you plan to run the GCN training, install torch + torch-geometric separately (they are not in `pyproject.toml`).

## Data layout
Expected local paths:

- `data/chembl_36/chembl_36_sqlite/chembl_36.db`
- `data/chembl_36/exports/` (Parquet exports)
- `data/chembl_36/fp_train`, `data/chembl_36/fp_val`
- `data/chembl_36/graph_train`, `data/chembl_36/graph_valid`

If you only have the archive, extract it into `data/chembl_36`:

```bash
tar -xzf data/chembl_36_sqlite.tar.gz -C data/chembl_36
```

The full ChEMBL schema reference is in `schema_documentation.txt`.

## Export base datasets (SQLite -> Parquet)
These scripts query the SQLite DB and write Parquet under `data/chembl_36/exports`:

```bash
uv run python -m exports.assay
uv run python -m exports.activity
uv run python -m exports.compound_struct
uv run python -m exports.compound_graph
```

Notes:
- `compound_struct` adds Morgan fingerprints (radius=2, 2048 bits).
- `compound_graph` exports `molregno` + `canonical_smiles` for Spark graph generation.
- `transforms.preprocess_activity` filters to nM values and computes pIC.

## Build train/val splits (Spark)
Spark jobs that join activity/assay/structure exports and write split datasets:

```bash
uv run python -m exports.training.export_morgan_fp
./exports/export_graph.sh
```

`export_graph.sh` runs Spark Connect via `uv run --no-project` with Python 3.10 and explicit deps. Set `PYTHON_VERSION` to override 3.10 and `SPARK_CONNECT_URL` to override `sc://localhost:15002`.

These scripts use `/data/chembl_36/exports` and write to `/data/chembl_36/...` by default. If you are running locally without a `/data` mount, either:

- Create a symlink: `ln -s "$(pwd)/data/chembl_36" /data/chembl_36`
- Or edit the paths in `exports/training/export_morgan_fp.py` and `exports/training/export_graph.py`.

## Train baselines
Tinygrad MLP on Morgan fingerprints:

```bash
uv run python -m src.train.tinyMLP
```

Torch-geometric GCN on graphs:

```bash
uv run python -m src.train.GCN
```

## Spark + Airflow + MLflow (local)
See `spark_airflow/README.md` for the Docker Compose stack and UI endpoints.

## Project layout
- `exports/` — SQL export definitions and dataset writers
- `src/transforms/` — cleaning and pIC computation
- `exports/training/` — Spark join + train/val split jobs
- `src/train/` — training scripts (tinygrad MLP, torch GCN)
- `notebooks/` — Jupyter notebooks (see `notebooks/README.md`)
- `data/` — local datasets and exports
