"""SparkSession factory — reuses the serverless session on Databricks, builds
`local[*]` locally. UTC + LEGACY time parser keep timestamp math identical in both.
"""
from __future__ import annotations

from pyspark.sql import SparkSession

# Static confs — only settable when building a fresh local session.
_BUILD_CONF = {
    "spark.sql.shuffle.partitions": "8",
    "spark.ui.enabled": "false",
}
# Runtime confs — applied to whichever session we return (local or serverless).
_RUNTIME_CONF = {
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.legacy.timeParserPolicy": "LEGACY",
    "spark.sql.sources.partitionOverwriteMode": "dynamic",
}


def get_spark(app_name: str = "churn_dbx") -> SparkSession:
    spark = SparkSession.getActiveSession()
    if spark is None:
        builder = SparkSession.builder.appName(app_name).master("local[*]")
        for key, val in _BUILD_CONF.items():
            builder = builder.config(key, val)
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
    for key, val in _RUNTIME_CONF.items():
        try:
            spark.conf.set(key, val)
        except Exception:
            pass  # some confs are locked on serverless
    return spark
