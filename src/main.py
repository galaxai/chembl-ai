from pathlib import Path

from pyspark.sql import SparkSession

from transforms.cleaning import preprocess_data

spark = (
    SparkSession.builder.appName("chembl-parquet")
    .remote("sc://localhost:15002")
    .getOrCreate()
)

export_path = Path("/data/chembl_36/exports/")
df_cs = spark.read.parquet(str(export_path / "compound_struct"))
df_activity = spark.read.parquet(str(export_path / "activity"))


print(df_activity.schema)
df_activity = preprocess_data(df_activity)
df_activity = df_activity.select("molregno", "standard_value", "standard_type", "pIC")
df_inner = df_activity.join(df_cs, on="molregno", how="inner")
if __name__ == "__main__":
    df_activity.show(5)
    df_inner.show(5)
