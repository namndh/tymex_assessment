"""Behavioural churn label — no churn flag exists in the data. Population = prior
activity before cutoff; label = inactivity over `[cutoff, cutoff+window)`, absolute
by default or relative (`inactive_percentile`) for the adaptive-cutoff path.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def _cutoff_bounds(cutoff, window_days: int):
    if isinstance(cutoff, str):
        cut = datetime.strptime(cutoff[:10], "%Y-%m-%d")
    elif isinstance(cutoff, datetime):
        cut = cutoff
    else:
        cut = datetime(cutoff.year, cutoff.month, cutoff.day)
    end = cut + timedelta(days=window_days)
    fmt = "%Y-%m-%d %H:%M:%S"
    return cut.strftime(fmt), end.strftime(fmt)


def build_labels(
    transactions: DataFrame,
    cutoff,
    window_days: int,
    min_prior_txns: int = 1,
    inactive_percentile: float | None = None,
) -> DataFrame:
    """One row per eligible customer with an int `churn` column."""
    cut_str, end_str = _cutoff_bounds(cutoff, window_days)
    cut = F.to_timestamp(F.lit(cut_str))
    end = F.to_timestamp(F.lit(end_str))

    tx = transactions.where(F.col("transaction_date").isNotNull())

    eligible = (
        tx.where(F.col("transaction_date") < cut)
        .groupBy("customer_id").agg(F.count(F.lit(1)).alias("prior_txns"))
        .where(F.col("prior_txns") >= F.lit(min_prior_txns))
        .select("customer_id")
    )

    fwd = (
        tx.where((F.col("transaction_date") >= cut) & (F.col("transaction_date") < end))
        .groupBy("customer_id").agg(F.count(F.lit(1)).alias("forward_txns"))
    )

    labels = (
        eligible.join(fwd, on="customer_id", how="left")
        .withColumn("forward_txns", F.coalesce("forward_txns", F.lit(0)))
    )

    if inactive_percentile is not None:
        threshold = labels.approxQuantile("forward_txns", [inactive_percentile], 0.0)[0]
        labels = labels.withColumn("churn", (F.col("forward_txns") <= F.lit(threshold)).cast("int"))
    else:
        labels = labels.withColumn("churn", (F.col("forward_txns") == 0).cast("int"))

    return labels.select("customer_id", "churn")
