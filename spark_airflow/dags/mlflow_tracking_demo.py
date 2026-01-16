from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.standard.operators.python import PythonOperator

MLFLOW_CALLED = Dataset("mlflowCalled")


def log_mlflow_run() -> None:
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("mlflow_tracking_demo")
    with mlflow.start_run(run_name="airflow_triggered"):
        mlflow.log_param("source", "airflow")
        mlflow.log_metric("accuracy", 0.95)


with DAG(
    dag_id="mlflow_tracking_demo",
    start_date=datetime(2024, 1, 1),
    schedule=[MLFLOW_CALLED],
    catchup=False,
    tags=["mlflow"],
) as dag:
    PythonOperator(
        task_id="log_mlflow_run",
        python_callable=log_mlflow_run,
    )
