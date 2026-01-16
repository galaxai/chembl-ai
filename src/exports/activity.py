import pyarrow as pa

## PATHS
BATCH_SIZE = 100_000
NAME = "activity"

SCHEMA = pa.schema(
    [
        ("activity_id", pa.int64()),
        ("assay_id", pa.int64()),
        ("molregno", pa.int64()),
        ("standard_type", pa.string()),
        ("standard_value", pa.float64()),
        ("standard_units", pa.string()),
        ("standard_relation", pa.string()),
        ("standard_flag", pa.int64()),
        ("pchembl_value", pa.float64()),
        ("type", pa.string()),
        ("value", pa.float64()),
        ("units", pa.string()),
        ("potential_duplicate", pa.int64()),
    ]
)

## FILTERS ##
STD_TYPES = ("IC50", "GI50", "Ki", "EC50")
potential_duplicate = 0
organism = ("Homo sapiens", "Mus musculus", "Rattus norvegicus")
assay_tax_id = (9606, 10090, 10116)

## COLUMNS ##
columns = [
    "act.activity_id",
    "act.assay_id",
    "act.molregno",
    "act.standard_type",
    "CAST(act.standard_value AS REAL) AS standard_value",
    "act.standard_units",
    "act.standard_relation",
    "act.standard_flag",
    "act.pchembl_value",
    "act.type",
    "CAST(act.value AS REAL) AS value",
    "act.units",
    "act.potential_duplicate",
]
## QUERY ##
SQL = f"""
SELECT {", ".join(columns)}
FROM activities act

JOIN assays ass ON ass.assay_id = act.assay_id
WHERE act.standard_type IN {STD_TYPES}
  AND act.potential_duplicate = {potential_duplicate}
  AND (
    ass.assay_organism IN {organism} OR
    ass.assay_tax_id IN {assay_tax_id})
"""

if __name__ == "__main__":
    from .helpers import export_sql_dataset

    export_sql_dataset(name=NAME, sql=SQL, batch_size=BATCH_SIZE, schema=SCHEMA)
