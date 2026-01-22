from __future__ import annotations

import os


def log_mlflow_run() -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.train.async_logger import AsyncMetricLogger
    from src.train.tinyMLP import (
        BATCH_SIZE,
        EPOCHS,
        LR,
        TRAIN_DIR,
        VAL_DIR,
        X_COL,
        Y_COL,
        train_tinymlp,
    )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("tinyMLP-morgan_fp")
    with mlflow.start_run():
        mlflow.log_param("source", "cli")
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("lr", LR)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("train_dir", TRAIN_DIR)
        mlflow.log_param("val_dir", VAL_DIR)
        mlflow.log_param("x_col", X_COL)
        mlflow.log_param("y_col", Y_COL)
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
            train_tinymlp(
                epochs=EPOCHS,
                lr=LR,
                batch_size=BATCH_SIZE,
                train_dir=TRAIN_DIR,
                val_dir=VAL_DIR,
                x_col=X_COL,
                y_col=Y_COL,
                logger=async_logger,
            )
        finally:
            async_logger.close()


if __name__ == "__main__":
    log_mlflow_run()
