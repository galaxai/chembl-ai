from pyspark.sql import DataFrame, SparkSession

from src.train.export.export_morgan_fp import chembl
from src.transforms import preprocess_activity


def _load_base_df(spark) -> DataFrame:
    """Load and join activity, assay, and structure parquet sources."""
    base = "/data/chembl_36/exports"
    activity = spark.read.parquet(f"{base}/activity")
    assay = spark.read.parquet(f"{base}/assay")
    struct = spark.read.parquet(f"{base}/compound_graph")
    activity = preprocess_activity(activity)

    df = (
        activity.join(assay, "assay_id", "inner")
        .join(struct, "molregno", "inner")
        .select(
            "activity_id",
            "node_features",
            "edge_index",
            "pIC",
            "assay_organism",
        )
    )
    df = df.na.drop()
    return df


if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("chembl-parquet")
        .remote("sc://localhost:15002")
        .getOrCreate()
    )
    train, val = chembl(
        spark=spark,
        load_df=_load_base_df,
        final_cols=["node_features", "edge_index", "pIC"],
        val_split=0.3,
        seed=42,
    )
    train.write.mode("overwrite").parquet("/data/chembl_36/graph_train")
    val.write.mode("overwrite").parquet("/data/chembl_36/graph_train")
