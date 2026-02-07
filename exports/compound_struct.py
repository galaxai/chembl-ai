from collections.abc import Sequence

import pyarrow as pa


def rows_to_batch(rows: Sequence[tuple], names: Sequence[str]) -> pa.RecordBatch:
    cols = list(zip(*rows))
    arrays = []
    for name, col in zip(names, cols):
        arrays.append(pa.array(col, type=SCHEMA.field(name).type))
    return pa.record_batch(arrays, schema=SCHEMA)


## PATHS
BATCH_SIZE = 100_000
NAME = "compound_struct"

## COLUMNS
columns = [
    "cs.molregno AS molregno",
    "cs.canonical_smiles AS canonical_smiles",
]

## QUERY ##
SQL = f"""
SELECT
    {", ".join(columns)}
FROM compound_structures cs
"""


SCHEMA = pa.schema(
    [
        ("molregno", pa.int64()),
        ("canonical_smiles", pa.string()),
    ]
)


if __name__ == "__main__":
    from .helpers import export_sql_dataset

    export_sql_dataset(
        name=NAME,
        sql=SQL,
        batch_size=BATCH_SIZE,
        transform=rows_to_batch,
        schema=SCHEMA,
    )
