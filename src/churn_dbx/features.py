"""The single feature-computation path, shared by training and scoring (anti-skew).
Every event table is filtered strictly `< cutoff` before aggregation — the leakage guard.
"""
from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

FEATURE_COLUMNS = [
    "tenure_days",
    "age_years",
    "is_female",
    "num_products",
    "has_savings",
    "has_credit_card",
    "total_credit_limit",
    "days_since_first_enrollment",
    "txn_count",
    "txn_amount_sum",
    "txn_amount_mean",
    "txn_amount_std",
    "closing_balance_last",
    "last_txn_recency_days",
    "first_txn_age_days",
    "txn_count_30d",
    "txn_count_90d",
    "active_months",
    "crm_count",
    "crm_count_chat",
    "crm_count_email",
    "crm_count_call",
    "crm_recency_days",
    "crm_count_90d",
]

_COUNT_FILL_ZERO = [
    "num_products", "has_savings", "has_credit_card", "total_credit_limit",
    "txn_count", "txn_amount_sum", "txn_amount_mean", "closing_balance_last",
    "txn_count_30d", "txn_count_90d", "active_months",
    "crm_count", "crm_count_chat", "crm_count_email", "crm_count_call",
    "crm_count_90d",
]
_RECENCY_FILL_TENURE = [
    "last_txn_recency_days", "first_txn_age_days",
    "days_since_first_enrollment", "crm_recency_days",
]


def _as_ts_str(cutoff) -> str:
    if isinstance(cutoff, str):
        return cutoff if " " in cutoff else f"{cutoff} 00:00:00"
    if isinstance(cutoff, datetime):
        return cutoff.strftime("%Y-%m-%d %H:%M:%S")
    return f"{cutoff} 00:00:00"  # date


def _days_before(cutoff_col, ts_col):
    """Whole days from an event timestamp up to the cutoff (floor of the timedelta,
    matching a wall-clock recency in days). Works for date and timestamp columns."""
    return F.floor((cutoff_col.cast("long") - ts_col.cast("long")) / F.lit(86400)).cast("int")


def compute_features(
    customers: DataFrame,
    enrollments: DataFrame,
    transactions: DataFrame,
    crm: DataFrame,
    cutoff,
) -> DataFrame:
    """One feature row per customer as of `cutoff`, with `FEATURE_COLUMNS` doubles."""
    cutoff_str = _as_ts_str(cutoff)
    cut = F.to_timestamp(F.lit(cutoff_str))
    cut30 = F.to_timestamp(F.lit(cutoff_str)) - F.expr("INTERVAL 30 DAYS")
    cut90 = F.to_timestamp(F.lit(cutoff_str)) - F.expr("INTERVAL 90 DAYS")

    base = customers.select("customer_id").distinct()

    demo = customers.select(
        "customer_id",
        _days_before(cut, F.col("signup_date")).alias("tenure_days"),
        (_days_before(cut, F.col("date_of_birth")) / F.lit(365.25)).alias("age_years"),
        F.when(F.lower(F.col("gender")) == "female", 1).otherwise(0).alias("is_female"),
    )

    enr = enrollments.where(F.col("enrollment_date") < cut)
    enr_agg = (
        enr.groupBy("customer_id").agg(
            F.countDistinct("product_id").alias("num_products"),
            F.max(F.when(F.col("product_type") == "Savings", 1).otherwise(0)).alias("has_savings"),
            F.max(F.when(F.col("product_type") == "Credit Card", 1).otherwise(0)).alias("has_credit_card"),
            F.sum("limit").alias("total_credit_limit"),
            F.min("enrollment_date").alias("first_enrollment"),
        )
        .withColumn("days_since_first_enrollment", _days_before(cut, F.col("first_enrollment")))
        .drop("first_enrollment")
    )

    tx = transactions.where(F.col("transaction_date") < cut)
    tx_agg = (
        tx.groupBy("customer_id").agg(
            F.count(F.lit(1)).alias("txn_count"),
            F.sum("transaction_amount").alias("txn_amount_sum"),
            F.avg("transaction_amount").alias("txn_amount_mean"),
            F.stddev_samp("transaction_amount").alias("txn_amount_std"),
            F.max("transaction_date").alias("last_txn"),
            F.min("transaction_date").alias("first_txn"),
            F.countDistinct(F.date_format("transaction_date", "yyyy-MM")).alias("active_months"),
        )
        .withColumn("last_txn_recency_days", _days_before(cut, F.col("last_txn")))
        .withColumn("first_txn_age_days", _days_before(cut, F.col("first_txn")))
        .drop("last_txn", "first_txn")
    )

    # closing_balance of the last transaction in (date, id) order — deterministic.
    order = Window.partitionBy("customer_id").orderBy(
        F.col("transaction_date").desc(), F.col("transaction_id").desc()
    )
    last_bal = (
        tx.withColumn("_rn", F.row_number().over(order))
        .where(F.col("_rn") == 1)
        .select("customer_id", F.col("closing_balance").alias("closing_balance_last"))
    )

    c30 = tx.where(F.col("transaction_date") >= cut30).groupBy("customer_id").agg(
        F.count(F.lit(1)).alias("txn_count_30d"))
    c90 = tx.where(F.col("transaction_date") >= cut90).groupBy("customer_id").agg(
        F.count(F.lit(1)).alias("txn_count_90d"))

    cr = crm.where(F.col("interaction_date") < cut)
    cr_agg = (
        cr.groupBy("customer_id").agg(
            F.count(F.lit(1)).alias("crm_count"),
            F.sum(F.when(F.col("interaction_type") == "Chat", 1).otherwise(0)).alias("crm_count_chat"),
            F.sum(F.when(F.col("interaction_type") == "Email", 1).otherwise(0)).alias("crm_count_email"),
            F.sum(F.when(F.col("interaction_type") == "Call", 1).otherwise(0)).alias("crm_count_call"),
            F.max("interaction_date").alias("last_crm"),
        )
        .withColumn("crm_recency_days", _days_before(cut, F.col("last_crm")))
        .drop("last_crm")
    )
    cr90 = cr.where(F.col("interaction_date") >= cut90).groupBy("customer_id").agg(
        F.count(F.lit(1)).alias("crm_count_90d"))

    feats = base
    for part in (demo, enr_agg, tx_agg, last_bal, c30, c90, cr_agg, cr90):
        feats = feats.join(part, on="customer_id", how="left")

    feats = feats.withColumn("txn_amount_std", F.coalesce("txn_amount_std", F.lit(0.0)))
    for col in _COUNT_FILL_ZERO:
        feats = feats.withColumn(col, F.coalesce(col, F.lit(0)))
    feats = feats.withColumn("tenure_days", F.greatest(F.coalesce("tenure_days", F.lit(0)), F.lit(0)))
    for col in _RECENCY_FILL_TENURE:
        feats = feats.withColumn(
            col, F.greatest(F.coalesce(F.col(col), F.col("tenure_days")), F.lit(0))
        )

    median_age = feats.approxQuantile("age_years", [0.5], 0.01)
    fill_age = median_age[0] if median_age else 0.0
    feats = feats.withColumn("age_years", F.coalesce("age_years", F.lit(fill_age)))

    select_cols = [F.col("customer_id")] + [F.col(c).cast("double").alias(c) for c in FEATURE_COLUMNS]
    return feats.select(*select_cols)
