"""Feature logic + time-boundary tests — the parts that fail silently.

Tiny synthetic frames pin the leakage boundary exactly: an event AT or AFTER the
cutoff must never move a feature. Midnight timestamps make the pandas-reference and
Spark day-math agree, so these double as a parity check on `compute_features`.
"""
from __future__ import annotations

from conftest import (
    CRM_SCHEMA,
    CUSTOMERS_SCHEMA,
    ENROLLMENTS_SCHEMA,
    TRANSACTIONS_SCHEMA,
    ts,
)

from churn_dbx.features import FEATURE_COLUMNS, compute_features

CUTOFF = "2025-03-01"


def _frames(spark):
    customers = spark.createDataFrame([
        (1, "Female", ts("1990-01-01"), ts("2024-01-01")),
        (2, "Male", ts("2000-01-01"), ts("2024-06-01")),
    ], CUSTOMERS_SCHEMA)
    enrollments = spark.createDataFrame([
        (10, 1, "Savings", ts("2024-01-01"), 0.0),
        (11, 1, "Credit Card", ts("2024-02-01"), 5000.0),
    ], ENROLLMENTS_SCHEMA)
    transactions = spark.createDataFrame([
        (10, 1, 100.0, 100.0, ts("2025-01-15"), 1),   # before (90d window)
        (10, 1, 200.0, 300.0, ts("2025-02-20"), 2),   # before (30d window)
        (10, 1, 999.0, 1299.0, ts("2025-03-01"), 3),  # AT cutoff -> excluded
        (10, 1, 999.0, 2298.0, ts("2025-04-01"), 4),  # after -> excluded
        (11, 2, 50.0, 50.0, ts("2025-03-05"), 5),      # after; cust 2 has no prior
    ], TRANSACTIONS_SCHEMA)
    crm = spark.createDataFrame([
        (1, 1, "Chat", ts("2025-02-25")),
        (2, 1, "Email", ts("2024-12-01")),
        (3, 1, "Call", ts("2025-03-10")),  # after cutoff -> excluded
    ], CRM_SCHEMA)
    return customers, enrollments, transactions, crm


def _rows(spark):
    feats = compute_features(*_frames(spark), CUTOFF)
    return {r["customer_id"]: r for r in feats.collect()}


def test_leakage_boundary_excludes_cutoff_and_future(spark):
    r = _rows(spark)
    assert r[1]["txn_count"] == 2
    assert r[1]["txn_amount_sum"] == 300.0
    assert r[1]["closing_balance_last"] == 300.0
    assert r[1]["crm_count"] == 2
    assert r[1]["crm_count_call"] == 0


def test_rolling_windows_respect_boundaries(spark):
    r = _rows(spark)
    assert r[1]["txn_count_30d"] == 1
    assert r[1]["txn_count_90d"] == 2
    assert r[1]["last_txn_recency_days"] == 9  # cutoff - 2025-02-20


def test_customer_without_prior_events_gets_zero_fills(spark):
    feats = compute_features(*_frames(spark), CUTOFF)
    r = {row["customer_id"]: row for row in feats.collect()}
    assert r[2]["txn_count"] == 0
    assert r[2]["crm_count"] == 0
    assert r[2]["num_products"] == 0
    assert feats.columns == ["customer_id", *FEATURE_COLUMNS]
    for col in FEATURE_COLUMNS:
        assert r[2][col] is not None


def test_enrollment_and_demographics(spark):
    r = _rows(spark)
    assert r[1]["num_products"] == 2
    assert r[1]["has_savings"] == 1
    assert r[1]["has_credit_card"] == 1
    assert r[1]["total_credit_limit"] == 5000.0
    assert r[1]["is_female"] == 1
    assert r[2]["is_female"] == 0


def test_compute_features_is_deterministic(spark):
    a = {r["customer_id"]: r for r in compute_features(*_frames(spark), CUTOFF).collect()}
    b = {r["customer_id"]: r for r in compute_features(*_frames(spark), CUTOFF).collect()}
    assert a == b
