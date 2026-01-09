# Spark + Airflow (local)
Ports used:
- Spark Master UI: `8080`
- Spark master RPC: `7077`
- Airflow UI: host `8081` -> (container `8080`)

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
