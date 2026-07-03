"""Label eligibility + window-boundary tests."""
from __future__ import annotations

from conftest import TRANSACTIONS_SCHEMA, ts

from churn_dbx.labels import build_labels

CUTOFF = "2025-03-01"
WINDOW = 90  # forward window ends 2025-05-30


def _txns(spark):
    return spark.createDataFrame([
        # cust 1: active before, active in forward window -> churn 0
        (1, 1, 10.0, 10.0, ts("2025-01-01"), 1),
        (1, 1, 10.0, 20.0, ts("2025-04-01"), 2),
        # cust 2: active before, silent in forward window -> churn 1
        (1, 2, 10.0, 10.0, ts("2025-02-01"), 3),
        # cust 3: no prior txns (only after cutoff) -> not eligible
        (1, 3, 10.0, 10.0, ts("2025-04-15"), 4),
        # cust 4: active before, only a txn AFTER the window closes -> churn 1
        (1, 4, 10.0, 10.0, ts("2025-02-15"), 5),
        (1, 4, 10.0, 20.0, ts("2025-06-15"), 6),
    ], TRANSACTIONS_SCHEMA)


def test_eligibility_and_absolute_inactivity(spark):
    labels = {r["customer_id"]: r["churn"] for r in build_labels(_txns(spark), CUTOFF, WINDOW).collect()}
    assert labels[1] == 0
    assert labels[2] == 1
    assert 3 not in labels  # no prior activity -> excluded from the population
    assert labels[4] == 1   # forward txn is outside the window


def test_window_upper_bound_is_exclusive(spark):
    # A txn exactly at cutoff+window must NOT count as forward activity.
    txns = spark.createDataFrame([
        (1, 5, 10.0, 10.0, ts("2025-02-01"), 1),
        (1, 5, 10.0, 20.0, ts("2025-05-30"), 2),  # == cutoff + 90d -> excluded
    ], TRANSACTIONS_SCHEMA)
    labels = {r["customer_id"]: r["churn"] for r in build_labels(txns, CUTOFF, WINDOW).collect()}
    assert labels[5] == 1


def test_min_prior_txns_threshold(spark):
    txns = spark.createDataFrame([
        (1, 6, 10.0, 10.0, ts("2025-01-01"), 1),  # only 1 prior txn
        (1, 7, 10.0, 10.0, ts("2025-01-01"), 2),
        (1, 7, 10.0, 10.0, ts("2025-02-01"), 3),  # 2 prior txns
    ], TRANSACTIONS_SCHEMA)
    rows = build_labels(txns, CUTOFF, WINDOW, min_prior_txns=2).collect()
    labels = {r["customer_id"]: r["churn"] for r in rows}
    assert 6 not in labels
    assert 7 in labels
