"""Batch scoring (deliverable 5) — champion scores the base via the shared
`compute_features` (anti-skew), writes one dated `predictions` partition."""
from __future__ import annotations

from datetime import date, datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from churn_dbx.config import PREDICTIONS_TABLE, Config
from churn_dbx.features import compute_features
from churn_dbx.io import read_raw, read_table, write_table
from churn_dbx.model_loader import load_scorer

PREDICTION_COLUMNS = ["customer_id", "churn_score", "model_version", "scored_at", "scored_date"]


def score_base(spark: SparkSession, cfg: Config, scored_date: date, scorer=None) -> DataFrame:
    scorer = scorer if scorer is not None else load_scorer(cfg)
    raw = read_raw(spark, cfg)
    features = compute_features(
        raw["customers"], raw["enrollments"], raw["transactions"], raw["crm"], str(scored_date)
    )
    scored_at = datetime.now(timezone.utc).isoformat()
    return (
        scorer.score(features)
        .withColumn("model_version", F.lit(scorer.model_version))
        .withColumn("scored_at", F.lit(scored_at))
        .withColumn("scored_date", F.lit(str(scored_date)))
        .select(*PREDICTION_COLUMNS)
    )


def batch_score(spark: SparkSession, cfg: Config, scored_date: date | None = None, scorer=None) -> str:
    scored_date = scored_date or datetime.now(timezone.utc).date()
    scorer = scorer if scorer is not None else load_scorer(cfg)
    predictions = score_base(spark, cfg, scored_date, scorer=scorer)
    out = write_table(spark, cfg, predictions, PREDICTIONS_TABLE, partition_by="scored_date")
    # Count from the written table (serverless forbids .cache(); don't re-run scoring).
    n = read_table(spark, cfg, PREDICTIONS_TABLE).where(F.col("scored_date") == str(scored_date)).count()
    print(f"[batch_score] scored {n} customers (model={scorer.model_version}) date={scored_date} -> {out}")
    return out
