import sqlite3
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.dataset as ds

## PATHS
DB_PATH = "data/chembl_36/chembl_36_sqlite/chembl_36.db"
OUT_BASE = "data/chembl_36/exports"
BATCH_SIZE = 100_000
NAME = "assay"

out_dir = Path(OUT_BASE) / NAME
out_dir.mkdir(parents=True, exist_ok=True)
## FILTERS ##
organism = ("Homo sapiens", "Mus musculus", "Rattus norvegicus")
assay_tax_id = (9606, 10090, 10116)

## COLUMNS
columns = [
    "assay_id",
    "assay_type",
    "assay_organism",
    "assay_tax_id",
    "tid",
    "doc_id",
    "chembl_id",
]

## QUERY ##
SQL = f"""
SELECT
    {"ass." + ", ass.".join(columns)}
FROM assays ass
WHERE
ass.assay_organism IN {organism} AND
ass.assay_tax_id IN {assay_tax_id}
"""
print(SQL)
quit()


def rows_to_batch(rows: list[tuple], names: list[str]) -> pa.RecordBatch:
    cols = list(zip(*rows))
    arrays = [pa.array(c) for c in cols]
    return pa.record_batch(arrays, names=names)


def sql_to_arrow(
    db_path: Path | str, sql: str, batch_size: int
) -> Iterator[pa.RecordBatch]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        names = [d[0] for d in cur.description]
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield rows_to_batch(rows, names)


def infer_schema(db_path: Path | str, sql: str, batch_size: int) -> pa.Schema | None:
    it = sql_to_arrow(db_path, sql, batch_size)
    first = next(it, None)
    return None if first is None else first.schema


if __name__ == "__main__":
    schema = infer_schema(DB_PATH, SQL, BATCH_SIZE)
    if schema is None:
        raise RuntimeError("Query returned no rows; cannot infer schema.")

    ds.write_dataset(
        data=sql_to_arrow(DB_PATH, SQL, BATCH_SIZE),
        base_dir=out_dir,
        format="parquet",
        schema=schema,
        basename_template="assays-{i}.parquet",
    )
