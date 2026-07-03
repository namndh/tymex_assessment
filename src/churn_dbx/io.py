"""Storage seam — the only place that knows where data physically lives. Resolves to
UC Delta or local parquet by mode; callers use logical table names only.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from churn_dbx.config import RAW_FILES, Config

# Date columns load as strings, then to_timestamp (errors -> null) so a malformed
# row can't change a column's type.
_SCHEMAS = {
    "customers": T.StructType([
        T.StructField("customer_id", T.IntegerType()),
        T.StructField("first_name", T.StringType()),
        T.StructField("last_name", T.StringType()),
        T.StructField("email", T.StringType()),
        T.StructField("mobile", T.StringType()),
        T.StructField("gender", T.StringType()),
        T.StructField("date_of_birth", T.StringType()),
        T.StructField("signup_date", T.StringType()),
    ]),
    "enrollments": T.StructType([
        T.StructField("product_id", T.IntegerType()),
        T.StructField("customer_id", T.IntegerType()),
        T.StructField("product_type", T.StringType()),
        T.StructField("enrollment_date", T.StringType()),
        T.StructField("limit", T.DoubleType()),
    ]),
    "transactions": T.StructType([
        T.StructField("product_id", T.IntegerType()),
        T.StructField("customer_id", T.IntegerType()),
        T.StructField("transaction_amount", T.DoubleType()),
        T.StructField("closing_balance", T.DoubleType()),
        T.StructField("transaction_date", T.StringType()),
        T.StructField("transaction_id", T.LongType()),
    ]),
    "crm": T.StructType([
        T.StructField("interaction_id", T.LongType()),
        T.StructField("customer_id", T.IntegerType()),
        T.StructField("interaction_type", T.StringType()),
        T.StructField("interaction_date", T.StringType()),
    ]),
}

_DATE_FMT = "yyyy-MM-dd"
_TS_FMT = "yyyy-MM-dd HH:mm:ss"
_DATE_COLS = {
    "customers": [("date_of_birth", _DATE_FMT), ("signup_date", _DATE_FMT)],
    "enrollments": [("enrollment_date", _DATE_FMT)],
    "transactions": [("transaction_date", _TS_FMT)],
    "crm": [("interaction_date", _DATE_FMT)],
}


def _read_csv(spark: SparkSession, cfg: Config, key: str) -> DataFrame:
    path = f"{cfg.raw_dir.rstrip('/')}/{RAW_FILES[key]}"
    df = (
        spark.read.option("sep", "|").option("header", "true")
        .schema(_SCHEMAS[key]).csv(path)
    )
    for col, fmt in _DATE_COLS[key]:
        df = df.withColumn(col, F.to_timestamp(F.col(col), fmt))
    return df


def read_raw(spark: SparkSession, cfg: Config) -> dict[str, DataFrame]:
    """Load all four raw tables; drop event rows with an unparseable primary date so
    time filters are well-defined."""
    raw = {key: _read_csv(spark, cfg, key) for key in RAW_FILES}
    raw["transactions"] = raw["transactions"].where(F.col("transaction_date").isNotNull())
    raw["crm"] = raw["crm"].where(F.col("interaction_date").isNotNull())
    return raw


def write_table(
    spark: SparkSession, cfg: Config, df: DataFrame, name: str,
    partition_by: str | None = None, overwrite_schema: bool = False,
) -> str:
    """Persist a logical table (UC Delta or local parquet). Partitioned writes use
    dynamic overwrite so a dated partition replaces only itself. `overwrite_schema`
    lets a column be added to an existing Delta table."""
    writer = df.write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(partition_by)
    if cfg.is_uc:
        if overwrite_schema:
            writer = writer.option("overwriteSchema", "true")
        writer.format("delta").saveAsTable(cfg.table(name))
        return cfg.table(name)
    path = cfg.table_path(name)
    writer.parquet(path)
    return path


def read_table(spark: SparkSession, cfg: Config, name: str) -> DataFrame:
    if cfg.is_uc:
        return spark.table(cfg.table(name))
    return spark.read.parquet(cfg.table_path(name))


def table_exists(spark: SparkSession, cfg: Config, name: str) -> bool:
    if cfg.is_uc:
        return spark.catalog.tableExists(cfg.table(name))
    import os
    return os.path.isdir(cfg.table_path(name))
