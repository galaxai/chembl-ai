# Spark + Airflow (local)
Ports used:
- Spark Master UI: `8080`
- Spark master RPC: `7077`
- Spark Connect: `15002`
- Airflow UI: host `8081` -> (container `8080`)
- MLflow UI: `5000`

## Run Docker Compose directly
Use the checked-in compose file: `spark_airflow/docker-compose.yml`.

Start:

```bash
cp spark_airflow/.env.example spark_airflow/.env
# edit spark_airflow/.env (set a real password)
docker compose -f spark_airflow/docker-compose.yml up -d
```

Stop:

```bash
docker compose -f spark_airflow/docker-compose.yml down
```

UIs:
- Spark Master UI: <http://localhost:8080>
- Airflow UI: <http://localhost:8081> (credentials from `spark_airflow/.env`)
- MLflow UI: <http://localhost:5000>

DAGs:
- `spark_airflow/dags/mlflow_tracking_demo.py` logs a local run to MLflow and triggers on the `mlflowCalled` dataset.

MLflow:
- Trigger `mlflow_tracking_demo` in Airflow, then open <http://localhost:5000>.
- Local artifacts are stored in `spark_airflow/mlruns/`.

SSH Tunnel:
- `putter` is your VM server.
- Run the following to forward Spark and Airflow UIs:

ocker exec -it airflow-standalone python -c "import mlflow; print(mlflow.__version__)"


```bash
ssh -fN \
  -L 8080:127.0.0.1:8080 \
  -L 8081:127.0.0.1:8081 \
  -L 5000:127.0.0.1:5000 \
  putter
```
