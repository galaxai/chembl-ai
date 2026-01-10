## PATHS
BATCH_SIZE = 100_000
NAME = "assay"
## FILTERS ##
organism = ("Homo sapiens", "Mus musculus", "Rattus norvegicus")
assay_tax_id = (9606, 10090, 10116)

## COLUMNS
columns = [
    "ass.assay_id",
    "ass.assay_type",
    "ass.assay_organism",
    "ass.assay_tax_id",
    "ass.tid",
    "ass.doc_id",
    "ass.chembl_id",
]

## QUERY ##
SQL = f"""
SELECT
    {", ".join(columns)}
FROM assays ass
WHERE
ass.assay_organism IN {organism} AND
ass.assay_tax_id IN {assay_tax_id}
"""


if __name__ == "__main__":
    from .helpers import export_sql_dataset

    export_sql_dataset(
        name=NAME,
        sql=SQL,
        batch_size=BATCH_SIZE,
    )
