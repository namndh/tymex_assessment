"""Model store + registry (deliverable 3) — a Volume + Delta `model_registry` table
standing in for the MLflow model registry, whose writes Free Edition denies
(`s3:PutObject` AccessDenied on the serverless role). MLflow is kept for tracking only.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import joblib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from churn_dbx.config import MODEL_REGISTRY_TABLE, Config
from churn_dbx.io import read_table, table_exists, write_table

# Explicit schema: `alias` is all-null until the first promotion, so Spark can't
# infer its type.
_SCHEMA = T.StructType([
    T.StructField("version", T.IntegerType(), False),
    T.StructField("model_path", T.StringType(), False),
    T.StructField("run_id", T.StringType(), True),
    T.StructField("roc_auc", T.DoubleType(), True),
    T.StructField("average_precision", T.DoubleType(), True),
    T.StructField("ap_lift_over_base", T.DoubleType(), True),
    T.StructField("created_at", T.StringType(), True),
    T.StructField("alias", T.StringType(), True),
    # Hash of the training inputs; lets a re-run with identical inputs skip re-registering.
    T.StructField("input_hash", T.StringType(), True),
])
_COLS = [f.name for f in _SCHEMA.fields]


def _row(d: dict) -> tuple:
    """Coerce a dict to a tuple in schema order (types normalised, missing -> None)."""
    return (
        int(d["version"]),
        str(d["model_path"]),
        None if d.get("run_id") is None else str(d["run_id"]),
        None if d.get("roc_auc") is None else float(d["roc_auc"]),
        None if d.get("average_precision") is None else float(d["average_precision"]),
        None if d.get("ap_lift_over_base") is None else float(d["ap_lift_over_base"]),
        d.get("created_at"),
        d.get("alias"),
        d.get("input_hash"),
    )


def _all_rows(spark: SparkSession, cfg: Config) -> list[dict]:
    if not table_exists(spark, cfg, MODEL_REGISTRY_TABLE):
        return []
    return [r.asDict() for r in read_table(spark, cfg, MODEL_REGISTRY_TABLE).collect()]


def _overwrite(spark: SparkSession, cfg: Config, rows: list[dict]) -> None:
    # Rows were collected to the driver (in _all_rows), so the new frame has no lineage
    # to the table it overwrites — safe to replace in place.
    df = spark.createDataFrame([_row(r) for r in rows], schema=_SCHEMA)
    write_table(spark, cfg, df, MODEL_REGISTRY_TABLE, overwrite_schema=True)


def register(spark: SparkSession, cfg: Config, estimator, run_id: str, metrics: dict,
             input_hash: str | None = None) -> int:
    """Pickle the estimator to the model store and append a registry row. Returns the
    new version number (max existing + 1)."""
    rows = _all_rows(spark, cfg)
    version = (max((int(r["version"]) for r in rows), default=0)) + 1

    root = cfg.models_root()
    dest_dir = f"{root}/v{version}"
    os.makedirs(dest_dir, exist_ok=True)
    model_path = f"{dest_dir}/model.joblib"
    joblib.dump(estimator, model_path)

    rows.append({
        "version": version,
        "model_path": model_path,
        "run_id": run_id,
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "ap_lift_over_base": metrics.get("ap_lift_over_base"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alias": None,
        "input_hash": input_hash,
    })
    _overwrite(spark, cfg, rows)
    return version


def by_input_hash(spark: SparkSession, cfg: Config, input_hash: str) -> dict | None:
    """The newest version registered from identical training inputs, or None."""
    for r in sorted(_all_rows(spark, cfg), key=lambda r: int(r["version"]), reverse=True):
        if r.get("input_hash") and r["input_hash"] == input_hash:
            return r
    return None


def latest(spark: SparkSession, cfg: Config) -> dict | None:
    if not table_exists(spark, cfg, MODEL_REGISTRY_TABLE):
        return None
    rows = read_table(spark, cfg, MODEL_REGISTRY_TABLE).orderBy(F.col("version").desc()).limit(1).collect()
    return rows[0].asDict() if rows else None


def by_alias(spark: SparkSession, cfg: Config, alias: str) -> dict | None:
    if not table_exists(spark, cfg, MODEL_REGISTRY_TABLE):
        return None
    rows = (
        read_table(spark, cfg, MODEL_REGISTRY_TABLE)
        .where(F.col("alias") == alias).orderBy(F.col("version").desc()).limit(1).collect()
    )
    return rows[0].asDict() if rows else None


def set_alias(spark: SparkSession, cfg: Config, version: int, alias: str) -> None:
    """Move `alias` to `version`, clearing whichever row held it before (an alias is
    unique). Rewrites the small registry table."""
    rows = _all_rows(spark, cfg)
    for r in rows:
        if r.get("alias") == alias:
            r["alias"] = None
        if int(r["version"]) == int(version):
            r["alias"] = alias
    _overwrite(spark, cfg, rows)


def load_model(model_path: str):
    """Load a pickled estimator from the model store (Volume path or local path)."""
    return joblib.load(model_path)
