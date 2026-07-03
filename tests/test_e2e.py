"""End-to-end on Spark-local: materialize -> train -> promote -> score.

A tiny synthetic dataset with a clean recency/activity signal, written as the four
pipe-delimited CSVs, exercised through the real job code (local mode = parquet
warehouse + sqlite MLflow). Proves a champion is registered and a dated predictions
partition is written — the same wheel that ships to Databricks.
"""
from __future__ import annotations

import itertools
import os
import random
from pathlib import Path

import pandas as pd

from churn_dbx.batch_score import PREDICTION_COLUMNS, batch_score
from churn_dbx.config import FEATURES_TABLE, MODEL_REGISTRY_TABLE, PREDICTIONS_TABLE
from churn_dbx.features import FEATURE_COLUMNS
from churn_dbx.io import read_table, table_exists
from churn_dbx.materialize import materialize
from churn_dbx.promote import promote
from churn_dbx.train import train


def _write_synthetic(raw_dir: str, n: int = 600, seed: int = 7) -> None:
    random.seed(seed)
    os.makedirs(raw_dir, exist_ok=True)
    cust, enr, tx, crm = [], [], [], []
    tid, iid, pid = itertools.count(1), itertools.count(1), itertools.count(1)
    for cid in range(1, n + 1):
        active = random.random() < 0.65
        cust.append((cid, "f", "l", f"u{cid}@x.com", "0900000000",
                     "Female" if cid % 2 == 0 else "Male", "1990-01-01", "2024-01-01"))
        enr.append((next(pid), cid, "Savings", "2024-01-05", 0.0))
        if random.random() < 0.4:
            enr.append((next(pid), cid, "Credit Card", "2024-02-05", 5000.0))
        # Pre-cutoff activity (eligibility). Churners stay in early January (stale);
        # active customers add a late-February transaction (recent).
        for _ in range(random.randint(3, 6)):
            day = random.randint(1, 14)
            tx.append((1, cid, round(random.uniform(10, 500), 2), 100.0,
                       f"2025-01-{day:02d} 10:00:00", next(tid)))
        if active:
            tx.append((1, cid, 50.0, 50.0, "2025-02-26 12:00:00", next(tid)))
            # Forward-window activity -> churn = 0.
            for _ in range(random.randint(1, 3)):
                m, d = random.choice([3, 4]), random.randint(1, 27)
                tx.append((1, cid, 40.0, 40.0, f"2025-{m:02d}-{d:02d} 09:00:00", next(tid)))
        if random.random() < 0.5:
            crm.append((next(iid), cid, "Chat", "2025-02-10"))

    def dump(rows, cols, name):
        pd.DataFrame(rows, columns=cols).to_csv(Path(raw_dir) / name, sep="|", index=False)

    dump(cust, ["customer_id", "first_name", "last_name", "email", "mobile", "gender",
                "date_of_birth", "signup_date"], "customer_raw.csv")
    dump(enr, ["product_id", "customer_id", "product_type", "enrollment_date", "limit"],
         "product_enrollments.csv")
    dump(tx, ["product_id", "customer_id", "transaction_amount", "closing_balance",
              "transaction_date", "transaction_id"], "transaction_history.csv")
    dump(crm, ["interaction_id", "customer_id", "interaction_type", "interaction_date"],
         "crm_interactions.csv")


def test_end_to_end(spark, cfg):
    _write_synthetic(cfg.raw_dir)

    materialize(spark, cfg)
    assert table_exists(spark, cfg, FEATURES_TABLE)
    table = read_table(spark, cfg, FEATURES_TABLE)
    assert set(FEATURE_COLUMNS).issubset(set(table.columns))
    assert "churn" in table.columns and table.count() > 0

    result = train(spark, cfg)
    assert result["roc_auc"] > 0.58  # clean synthetic signal clears the gate floor

    decision = promote(cfg)
    assert decision["promote"] is True
    assert decision["version"] is not None

    from datetime import date
    batch_score(spark, cfg, date(2025, 3, 1))
    assert table_exists(spark, cfg, PREDICTIONS_TABLE)
    preds = read_table(spark, cfg, PREDICTIONS_TABLE)
    assert preds.columns == PREDICTION_COLUMNS or set(PREDICTION_COLUMNS).issubset(preds.columns)
    assert preds.count() == 600
    scores = [r["churn_score"] for r in preds.select("churn_score").collect()]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_training_is_idempotent(spark, cfg):
    _write_synthetic(cfg.raw_dir)
    materialize(spark, cfg)

    first = train(spark, cfg)
    assert first["skipped"] is False
    second = train(spark, cfg)  # identical inputs -> reuse, no new version
    assert second["skipped"] is True
    assert second["version"] == first["version"]
    assert read_table(spark, cfg, MODEL_REGISTRY_TABLE).count() == 1


def test_training_compares_candidates(spark, cfg):
    _write_synthetic(cfg.raw_dir)
    materialize(spark, cfg)

    result = train(spark, cfg)
    assert result["winning_model"] in cfg.model["candidates"]
