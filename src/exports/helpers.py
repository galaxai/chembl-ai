import sqlite3
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.dataset as ds

DB_PATH = "../data/chembl_36/chembl_36_sqlite/chembl_36.db"
OUT_BASE = "../data/chembl_36/exports"


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


def export_sql_dataset(
    *,
    name: str,
    sql: str,
    batch_size: int = 100_000,
    basename_template: str | None = None,
    db_path: Path | str = DB_PATH,
    out_base: Path | str = OUT_BASE,
    format: str = "parquet",
) -> Path | None:
    out_dir = Path(out_base) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if basename_template is None:
        basename_template = f"{name}-{{i}}.parquet"

    try:
        schema = infer_schema(db_path, sql, batch_size)
        if schema is None:
            raise RuntimeError("Query returned no rows; cannot infer schema.")

        print(f"Writing to {out_dir}")
        ds.write_dataset(
            data=sql_to_arrow(db_path, sql, batch_size),
            base_dir=out_dir,
            format=format,
            schema=schema,
            basename_template=basename_template,
        )

        print(f"Written to {out_dir}")
        return out_dir

    except Exception as e:
        print(f"Error occurred: {e}")
        import shutil
        import time

        for i in range(3, 0, -1):
            print(f"Cleaning up in {i} seconds...")
            time.sleep(1)
        print("Cleaning up...")

        if out_dir.exists():
            shutil.rmtree(out_dir)

        return None
