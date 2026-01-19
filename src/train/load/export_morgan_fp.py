import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from src.transforms import preprocess_activity

spark = (
    SparkSession.builder.appName("chembl-parquet")
    .remote("sc://localhost:15002")
    .getOrCreate()
)


def _load_base_df(spark) -> DataFrame:
    base = "/data/chembl_36/exports"
    activity = spark.read.parquet(f"{base}/activity")
    assay = spark.read.parquet(f"{base}/assay")
    struct = spark.read.parquet(f"{base}/compound_struct")
    activity = preprocess_activity(activity)

    df = (
        activity.join(assay, "assay_id", "inner")
        .join(struct, "molregno", "inner")
        .select(
            "activity_id",
            "morgan_fp",
            "pIC",
            "assay_organism",
        )
    )
    df = df.na.drop()
    return df


def chembl(
    *,
    fingerprint: bool = False,
    val_split: float = 0.2,
    seed: int = 42,
):
    df = _load_base_df(spark)
    human = F.col("assay_organism") == "Homo sapiens"
    human_df = df.filter(human)
    val_ids = (
        human_df.select("activity_id").sample(fraction=val_split, seed=seed).distinct()
    )
    val_df = human_df.join(val_ids, on="activity_id", how="inner")
    train_df = df.join(val_ids, on="activity_id", how="left_anti")
    return train_df, val_df


if __name__ == "__main__":
    train, val = chembl(fingerprint=True)
    train.write.mode("overwrite").parquet("/data/chembl_36/fp_train")
    val.write.mode("overwrite").parquet("/data/chembl_36/fp_val")
