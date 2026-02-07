from collections.abc import Callable

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from src.transforms import preprocess_activity

EXPORT_BASE = "/data/chembl_36/exports"
STD_TYPES = ("IC50", "GI50", "Ki", "EC50")
POTENTIAL_DUPLICATE = 0
ORGANISM = ("Homo sapiens", "Mus musculus", "Rattus norvegicus")
ASSAY_TAX_ID = (9606, 10090, 10116)


def load_activity_features_df(
    *,
    spark: SparkSession,
    struct_name: str,
    struct_to_features: Callable[[DataFrame], DataFrame],
    feature_cols: list[str],
    base: str = EXPORT_BASE,
) -> DataFrame:
    """Load activity/assay data and join with structure-derived features."""
    activity = spark.read.parquet(f"{base}/activity")
    assay = spark.read.parquet(f"{base}/assay")
    struct = spark.read.parquet(f"{base}/{struct_name}")
    activity = activity.filter(
        (F.col("standard_type").isin(*STD_TYPES))
        & (F.col("potential_duplicate") == POTENTIAL_DUPLICATE)
    )
    assay = assay.filter(
        (F.col("assay_organism").isin(*ORGANISM))
        & (F.col("assay_tax_id").isin(*ASSAY_TAX_ID))
    )
    if "canonical_smiles" in struct.columns:
        struct = struct.filter(
            (F.col("canonical_smiles").isNotNull()) & (F.col("canonical_smiles") != "")
        )
    features = struct_to_features(struct)
    activity = preprocess_activity(activity)

    df = (
        activity.join(assay, "assay_id", "inner")
        .join(features, "molregno", "inner")
        .select("activity_id", *feature_cols, "pIC", "assay_organism")
    )
    return df.na.drop()


def chembl(
    *,
    spark: SparkSession,
    load_df: Callable[[SparkSession], DataFrame],
    final_cols: list[str] | str,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataFrame, DataFrame]:
    """Return train/val splits of ChEMBL activity data."""
    df = load_df(spark)
    human = F.col("assay_organism") == "Homo sapiens"
    human_df = df.filter(human)
    val_ids = (
        human_df.select("activity_id").sample(fraction=val_split, seed=seed).distinct()
    )
    val_df = human_df.join(val_ids, on="activity_id", how="inner")
    ## Multi organism training data
    train_df = df.join(val_ids, on="activity_id", how="left_anti")
    train_df = train_df.select(final_cols)
    val_df = val_df.select(final_cols)
    return train_df, val_df
