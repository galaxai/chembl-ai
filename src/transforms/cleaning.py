from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_pIC(df: DataFrame) -> DataFrame:
    """Filter to valid nM activities and add a `pIC` column.

    The calculation converts `standard_value` nM to molar and
    applies the common transformation:

    `pIC = -log10(standard_value_in_molar)`

    Rows are restricted to:
    - `standard_units == "nM"`
    - `standard_value > 0`

    Returns:
        A DataFrame filtered to valid rows with an added numeric `pIC` column.
    """

    return df.filter(
        (F.col("standard_units") == "nM") & (F.col("standard_value") > 0)
    ).withColumn("pIC", -F.log10(F.col("standard_value") * F.lit(1e-9)))


def preprocess_data(df: DataFrame) -> DataFrame:
    """Apply the default preprocessing pipeline.

    Current steps:
    1. Drop rows containing nulls.
    2. Add `pIC` via :func:`add_pIC`.


    Returns:
        A cleaned DataFrame ready for downstream transformations.
    """

    df = df.na.drop()
    df = add_pIC(df)
    return df
