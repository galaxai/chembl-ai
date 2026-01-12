import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pyarrow as pa
import pyarrow.dataset as ds

BatchTransform = Callable[[Sequence[tuple[Any, ...]], Sequence[str]], pa.RecordBatch]

DB_PATH = "../data/chembl_36/chembl_36_sqlite/chembl_36.db"
OUT_BASE = "../data/chembl_36/exports"


def rows_to_batch(
    rows: Sequence[tuple[Any, ...]], names: Sequence[str]
) -> pa.RecordBatch:
    """
    Convert a list of row tuples into a PyArrow RecordBatch with the given column names.

    Parameters:
        rows (list[tuple]): Sequence of row tuples where each tuple represents a record and fields are ordered to match `names`.
        names (list[str]): Column names for the resulting RecordBatch, in the same order as the tuple fields.

    Returns:
        pa.RecordBatch: A RecordBatch whose columns correspond to the transposed fields of `rows` and are named according to `names`.
    """
    cols = list(zip(*rows))
    arrays = [pa.array(c) for c in cols]
    return pa.record_batch(arrays, names=names)


def sql_to_arrow(
    db_path: Path | str,
    sql: str,
    batch_size: int,
    transform: BatchTransform = rows_to_batch,
) -> Iterator[pa.RecordBatch]:
    """
    Stream the results of an SQL query from a SQLite database as PyArrow RecordBatches.

    Parameters:
        db_path (Path | str): Path to the SQLite database file.
        sql (str): SQL query to execute.
        batch_size (int): Maximum number of rows to fetch per yielded RecordBatch.

    Returns:
        Iterator[pa.RecordBatch]: An iterator that yields RecordBatch objects, each containing up to `batch_size` rows from the query result.
    """
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        names = [d[0] for d in cur.description]
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield transform(rows, names)


def export_sql_dataset(
    *,
    name: str,
    sql: str,
    schema: pa.Schema,
    batch_size: int = 100_000,
    transform: BatchTransform = rows_to_batch,
    basename_template: str | None = None,
    db_path: Path | str = DB_PATH,
    out_base: Path | str = OUT_BASE,
    format: str = "parquet",
) -> Path | None:
    """
    Export results of an SQL query from a SQLite database into an on-disk Arrow dataset.

    Creates (or reuses) a subdirectory under `out_base` named `name` and streams query results into a PyArrow dataset written in the requested format using the provided Arrow `schema`. On error the function removes the output directory and returns None.

    Parameters:
        name (str): Subdirectory name under `out_base` where files will be written.
        sql (str): SQL query to execute against `db_path`.
        schema (pa.Schema): Arrow schema to write.
        batch_size (int): Number of rows fetched per RecordBatch when reading the query.
        transform (BatchTransform): Optional row->RecordBatch transform.
        basename_template (str | None): Filename template for output fragments containing an `{i}` placeholder (defaults to `"{name}-{i}.parquet"` when None).
        db_path (Path | str): Path to the SQLite database to query.
        out_base (Path | str): Base directory under which the `name` subdirectory will be created.
        format (str): Output dataset format supported by pyarrow.dataset (for example, `"parquet"`).

    Returns:
        Path | None: Path to the output directory on success, or `None` if an error occurred and cleanup was performed.
    """
    out_dir = Path(out_base) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if basename_template is None:
        basename_template = f"{name}-{{i}}.parquet"

    try:
        print(f"Writing to {out_dir}")
        ds.write_dataset(
            data=sql_to_arrow(db_path, sql, batch_size, transform=transform),
            base_dir=out_dir,
            format=format,
            schema=schema,
            basename_template=basename_template,
        )

        print(f"Written to {out_dir}")
        return out_dir

    except Exception as e:
        print(f"Error occurred: {e}")
        # TODO: remove it later, its good for iteration speed for now. Later it might remove user's data
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)

        return None
