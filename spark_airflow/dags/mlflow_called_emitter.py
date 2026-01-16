from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.standard.operators.empty import EmptyOperator

MLFLOW_CALLED = Dataset("mlflowCalled")


with DAG(
    dag_id="mlflow_called_emitter",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlflow"],
) as dag:
    EmptyOperator(
        task_id="emit_mlflow_called",
        outlets=[MLFLOW_CALLED],
    )
