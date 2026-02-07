import os

import pyarrow as pa
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, FloatType, LongType, StructField, StructType

from .common import chembl, load_activity_features_df

_GRAPH_SCHEMA = StructType(
    [
        StructField("molregno", LongType(), nullable=False),
        StructField(
            "node_features",
            ArrayType(ArrayType(FloatType(), containsNull=False), containsNull=False),
            nullable=False,
        ),
        StructField(
            "edge_index",
            ArrayType(ArrayType(LongType(), containsNull=False), containsNull=False),
            nullable=False,
        ),
    ]
)


_NODE_FEATURES_TYPE = pa.list_(pa.list_(pa.float32()))
_EDGE_INDEX_TYPE = pa.list_(pa.list_(pa.int64()))
_ARROW_SCHEMA = pa.schema(
    [
        ("molregno", pa.int64()),
        ("node_features", _NODE_FEATURES_TYPE),
        ("edge_index", _EDGE_INDEX_TYPE),
    ]
)


def _smiles_to_graph_arrow_batches(batch_iter):
    from rdkit import Chem

    for batch in batch_iter:
        data = batch.to_pydict()
        molregno_out = []
        node_features_out = []
        edge_index_out = []

        for molregno, smiles in zip(data["molregno"], data["canonical_smiles"]):
            if not smiles:
                continue
            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                continue
            if mol is None or mol.GetNumAtoms() == 0:
                continue

            node_features: list[list[float]] = []
            for atom in mol.GetAtoms():
                node_features.append(
                    [
                        float(atom.GetAtomicNum()),
                        float(atom.GetDegree()),
                        float(atom.GetFormalCharge()),
                        float(atom.GetHybridization().real),
                        float(int(atom.GetIsAromatic())),
                        float(atom.GetTotalNumHs()),
                    ]
                )

            edge_indices: list[list[int]] = [[], []]
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices[0].append(i)
                edge_indices[1].append(j)
                edge_indices[0].append(j)
                edge_indices[1].append(i)

            molregno_out.append(int(molregno))
            node_features_out.append(node_features)
            edge_index_out.append(edge_indices)

        arrays = [
            pa.array(molregno_out, type=pa.int64()),
            pa.array(node_features_out, type=_NODE_FEATURES_TYPE),
            pa.array(edge_index_out, type=_EDGE_INDEX_TYPE),
        ]
        yield pa.record_batch(arrays, schema=_ARROW_SCHEMA)


def _graph_df_from_smiles(struct: DataFrame) -> DataFrame:
    return struct.select("molregno", "canonical_smiles").mapInArrow(
        _smiles_to_graph_arrow_batches,
        schema=_GRAPH_SCHEMA,
    )


def _load_base_df(spark) -> DataFrame:
    """Load and join activity, assay, and structure parquet sources."""
    return load_activity_features_df(
        spark=spark,
        struct_name="compound_struct",
        struct_to_features=_graph_df_from_smiles,
        feature_cols=["node_features", "edge_index"],
    )


if __name__ == "__main__":
    connect_url = os.environ.get("SPARK_CONNECT_URL", "sc://localhost:15002")
    spark = (
        SparkSession.builder.appName("chembl-parquet").remote(connect_url).getOrCreate()
    )
    train, val = chembl(
        spark=spark,
        load_df=_load_base_df,
        final_cols=["node_features", "edge_index", "pIC"],
        val_split=0.3,
        seed=42,
    )
    train.write.mode("overwrite").parquet("/data/chembl_36/graph_train")
    val.write.mode("overwrite").parquet("/data/chembl_36/graph_valid")
    spark.stop()
