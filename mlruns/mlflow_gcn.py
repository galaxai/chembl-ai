from __future__ import annotations

import os


def log_mlflow_run() -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.train.async_logger import AsyncMetricLogger
    from src.train.GCN import (
        BS_SIZE,
        EPOCHS,
        GRAD_LOG_EPOCHS,
        HIDDEN_CHANNELS,
        LR,
        NUM_WORKERS,
        OPTIM,
        TRAIN_DIR,
        VALID_DIR,
        train_gcn,
    )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("GCN-chembl36")
    with mlflow.start_run():
        mlflow.log_param("source", "cli")
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("lr", LR)
        mlflow.log_param("batch_size", BS_SIZE)
        mlflow.log_param("hidden_channels", HIDDEN_CHANNELS)
        mlflow.log_param("optimizer", OPTIM.__name__)
        mlflow.log_param("num_workers", NUM_WORKERS)
        mlflow.log_param("train_dir", TRAIN_DIR)
        mlflow.log_param("valid_dir", VALID_DIR)
        mlflow.log_param("grad_log_epochs", GRAD_LOG_EPOCHS)

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("No active MLflow run found")

        run_id = active_run.info.run_id
        client = MlflowClient()

        class _ClientLogger:
            def log_metric(self, name: str, value: float, step: int) -> None:
                client.log_metric(run_id, name, value, step=step)

        async_logger = AsyncMetricLogger(_ClientLogger())
        try:
            train_gcn(
                epochs=EPOCHS,
                lr=LR,
                batch_size=BS_SIZE,
                hidden_channels=HIDDEN_CHANNELS,
                train_dir=TRAIN_DIR,
                valid_dir=VALID_DIR,
                logger=async_logger,
                grad_log_epochs=GRAD_LOG_EPOCHS,
            )
        finally:
            async_logger.close()


if __name__ == "__main__":
    log_mlflow_run()
