from collections.abc import Sequence

import pyarrow as pa
from rdkit import Chem


def _smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], [[], []]
    node_features = []
    num_atoms = mol.GetAtoms()
    for atom in num_atoms:
        features = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),  # Number of bonds
            atom.GetFormalCharge(),  # Formal charge
            atom.GetHybridization().real,  # Hybridization (sp, sp2, sp3, etc.)
            int(atom.GetIsAromatic()),  # Is aromatic
            atom.GetTotalNumHs(),  # Number of hydrogens
        ]
        node_features.append(features)
    edge_indices = [[], []]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices[0].append(i)
        edge_indices[1].append(j)
        edge_indices[0].append(j)
        edge_indices[1].append(i)
    return node_features, edge_indices  # This we save to parquet


def rows_to_batch(rows: Sequence[tuple], names: Sequence[str]) -> pa.RecordBatch:
    cols = list(zip(*rows))
    names_list = list(names)
    molregno_idx = names_list.index("molregno")
    smiles_idx = names_list.index("canonical_smiles")
    node_features_list = []
    edge_index_list = []
    for smiles in cols[smiles_idx]:
        node_features, edge_indices = _smiles_to_graph(smiles)
        node_features_list.append(node_features)
        edge_index_list.append(edge_indices)
    arrays = [
        pa.array(cols[molregno_idx], type=pa.int64()),
        pa.array(node_features_list, type=pa.list_(pa.list_(pa.float64()))),
        pa.array(edge_index_list, type=pa.list_(pa.list_(pa.int64()))),
    ]
    return pa.record_batch(arrays, schema=SCHEMA)


## PATHS
NAME = "compound_graph"
BATCH_SIZE = 100_000
## COLUMNS
columns = ["cs.molregno AS molregno", "cs.canonical_smiles AS canonical_smiles"]
## QUERY
SQL = f"""
SELECT
    {", ".join(columns)}
FROM compound_structures cs
WHERE cs.canonical_smiles IS NOT NULL AND cs.canonical_smiles != ''
"""
SCHEMA = pa.schema(
    [
        pa.field("molregno", pa.int64()),
        pa.field("node_features", pa.list_(pa.list_(pa.float64()))),
        pa.field("edge_index", pa.list_(pa.list_(pa.int64()))),
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
