"""Spark-local fixtures — no Databricks workspace needed.

A session-scoped local SparkSession and a `local` Config pointing at a temp parquet
warehouse + sqlite MLflow, so the exact same job code runs here as on Databricks.
"""
from __future__ import annotations

import os
import time

# Pin the process timezone to UTC BEFORE Spark starts so naive python datetimes
# injected into test frames localize as UTC — matching the session-UTC parsing of
# the cutoff literals. Otherwise a midnight "AT cutoff" row shifts across the
# boundary and the leakage assertions become tz-dependent.
os.environ["TZ"] = "UTC"
time.tzset()

from datetime import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from churn_dbx.config import CHECKPOINT, LABEL, MODEL, MONITORING, QUALITY_GATE, Config


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = (
        SparkSession.builder.appName("churn_dbx_tests").master("local[2]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def cfg(tmp_path) -> Config:
    root = str(tmp_path / "state")
    os.makedirs(root, exist_ok=True)
    return Config(
        mode="local",
        catalog="workspace",
        schema="churn_test",
        volume="raw",
        local_root=root,
        raw_dir=str(tmp_path / "raw"),
        data_url="",
        tracking_uri=f"sqlite:///{root}/mlflow.db",
        registry_uri=f"sqlite:///{root}/mlflow.db",
        experiment="churn_test",
        model_base="churn_model",
        production_alias="champion",
        label=dict(LABEL),
        model=dict(MODEL),
        quality_gate=dict(QUALITY_GATE),
        monitoring=dict(MONITORING),
        checkpoint=dict(CHECKPOINT),
    )


CUSTOMERS_SCHEMA = T.StructType([
    T.StructField("customer_id", T.IntegerType()),
    T.StructField("gender", T.StringType()),
    T.StructField("date_of_birth", T.TimestampType()),
    T.StructField("signup_date", T.TimestampType()),
])
ENROLLMENTS_SCHEMA = T.StructType([
    T.StructField("product_id", T.IntegerType()),
    T.StructField("customer_id", T.IntegerType()),
    T.StructField("product_type", T.StringType()),
    T.StructField("enrollment_date", T.TimestampType()),
    T.StructField("limit", T.DoubleType()),
])
TRANSACTIONS_SCHEMA = T.StructType([
    T.StructField("product_id", T.IntegerType()),
    T.StructField("customer_id", T.IntegerType()),
    T.StructField("transaction_amount", T.DoubleType()),
    T.StructField("closing_balance", T.DoubleType()),
    T.StructField("transaction_date", T.TimestampType()),
    T.StructField("transaction_id", T.LongType()),
])
CRM_SCHEMA = T.StructType([
    T.StructField("interaction_id", T.LongType()),
    T.StructField("customer_id", T.IntegerType()),
    T.StructField("interaction_type", T.StringType()),
    T.StructField("interaction_date", T.TimestampType()),
])


def ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d") if len(s) == 10 else datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
