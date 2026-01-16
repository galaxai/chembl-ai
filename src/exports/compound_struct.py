from collections.abc import Sequence
from typing import Any

import pyarrow as pa
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_MORGAN_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_morgan_fp(
    smiles: str | None,
    *,
    mfpgen: Any = _MORGAN_FPGEN,
) -> list[int] | None:
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = mfpgen.GetFingerprintAsNumPy(mol)
        return fp.tolist()
    except Exception:
        return None


def rows_to_batch(rows: Sequence[tuple], names: Sequence[str]) -> pa.RecordBatch:
    cols = list(zip(*rows))
    arrays = [pa.array(c) for c in cols]

    names_list = list(names)
    data_idx = names_list.index("canonical_smiles")
    morgan_fp = pa.array(
        [smiles_to_morgan_fp(s) for s in cols[data_idx]],
        type=pa.list_(pa.uint8()),
    )

    batch_names = names_list + ["morgan_fp"]
    arrays.append(morgan_fp)
    return pa.record_batch(arrays, names=batch_names)


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
        ("morgan_fp", pa.list_(pa.uint8())),
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
