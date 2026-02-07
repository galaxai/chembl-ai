import os

import numpy as np
import pyarrow as pa
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, ByteType, LongType, StructField, StructType

from .common import chembl, load_activity_features_df

_MORGAN_FP_SIZE = 2048
_MORGAN_RADIUS = 2

_MORGAN_SCHEMA = StructType(
    [
        StructField("molregno", LongType(), nullable=False),
        StructField(
            "morgan_fp",
            ArrayType(ByteType(), containsNull=False),
            nullable=False,
        ),
    ]
)

_ARROW_SCHEMA = pa.schema(
    [
        ("molregno", pa.int64()),
        ("morgan_fp", pa.list_(pa.int8())),
    ]
)


def _smiles_to_morgan_arrow_batches(batch_iter):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=_MORGAN_RADIUS, fpSize=_MORGAN_FP_SIZE
    )

    for batch in batch_iter:
        data = batch.to_pydict()
        molregno_out = []
        morgan_out = []
        mols = []
        indices = []

        for i, smiles in enumerate(data["canonical_smiles"]):
            if not smiles:
                continue
            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                continue
            if mol is None:
                continue
            mols.append(mol)
            indices.append(i)

        fps = mfpgen.GetFingerprints(mols, numThreads=8) if mols else []

        for idx, fp in zip(indices, fps):
            fp_array = np.zeros(fp.GetNumBits(), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            molregno_out.append(int(data["molregno"][idx]))
            morgan_out.append(fp_array.tolist())

        arrays = [
            pa.array(molregno_out, type=_ARROW_SCHEMA.field("molregno").type),
            pa.array(morgan_out, type=_ARROW_SCHEMA.field("morgan_fp").type),
        ]
        yield pa.record_batch(arrays, schema=_ARROW_SCHEMA)


def _morgan_df_from_smiles(struct: DataFrame) -> DataFrame:
    return struct.select("molregno", "canonical_smiles").mapInArrow(
        _smiles_to_morgan_arrow_batches,
        schema=_MORGAN_SCHEMA,
    )


def _load_base_df(spark) -> DataFrame:
    """Load and join activity, assay, and structure parquet sources."""
    return load_activity_features_df(
        spark=spark,
        struct_name="compound_struct",
        struct_to_features=_morgan_df_from_smiles,
        feature_cols=["morgan_fp"],
    )


if __name__ == "__main__":
    connect_url = os.environ.get("SPARK_CONNECT_URL", "sc://localhost:15002")
    spark = (
        SparkSession.builder.appName("chembl-parquet").remote(connect_url).getOrCreate()
    )
    train, val = chembl(
        spark=spark,
        load_df=_load_base_df,
        final_cols=["morgan_fp", "pIC"],
        val_split=0.2,
        seed=42,
    )
    train.write.mode("overwrite").parquet("/data/chembl_36/fp_train")
    val.write.mode("overwrite").parquet("/data/chembl_36/fp_val")
    spark.stop()
