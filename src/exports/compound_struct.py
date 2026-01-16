from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

_MORGAN_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_batch_to_morgan_fps(
    smiles_list: Sequence[str | None],
    *,
    mfpgen: Any = _MORGAN_FPGEN,
) -> list[list[int] | None]:
    """Convert a batch of SMILES to Morgan fingerprints efficiently."""
    results: list[list[int] | None] = [None] * len(smiles_list)
    mols = []
    indices = []
    for i, smiles in enumerate(smiles_list):
        if not smiles:
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            mols.append(mol)
            indices.append(i)
        except Exception:
            continue

    fps = mfpgen.GetFingerprints(mols, numThreads=8)

    for idx, fp in zip(indices, fps):
        fp_array = np.zeros(fp.GetNumBits(), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        results[idx] = fp_array.tolist()
    return results


def rows_to_batch(rows: Sequence[tuple], names: Sequence[str]) -> pa.RecordBatch:
    cols = list(zip(*rows))
    arrays = [pa.array(c) for c in cols]

    names_list = list(names)
    data_idx = names_list.index("canonical_smiles")

    morgan_fp = pa.array(
        smiles_batch_to_morgan_fps(cols[data_idx]),
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
