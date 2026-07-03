"""Adaptive cutoff ladder — resolve_cutoff picks/holds the cutoff from the data."""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import pandas as pd

from churn_dbx.checkpoint import resolve_cutoff


def _write_raw(raw_dir: str, txns: list[tuple[int, str]]) -> None:
    """txns = list of (customer_id, 'YYYY-MM-DD') → the four raw CSVs (only transactions
    vary; the checkpoint's label probe reads transactions only)."""
    os.makedirs(raw_dir, exist_ok=True)
    cust, enr, tx, crm = [], [], [], []
    tid = itertools.count(1)
    customers = sorted({c for c, _ in txns})
    for cid in customers:
        cust.append((cid, "f", "l", f"u{cid}@x.com", "0900000000", "Female",
                     "1990-01-01", "2024-01-01"))
        enr.append((cid, cid, "Savings", "2024-01-05", 0.0))
        crm.append((cid, cid, "Chat", "2024-02-10"))
    for cid, day in txns:
        tx.append((1, cid, 50.0, 50.0, f"{day} 10:00:00", next(tid)))

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


def test_no_new_data_keeps_default_cutoff(spark, cfg):
    # Data ends early 2025 -> newest observable cutoff is <= the default -> hold default.
    txns = [(c, "2025-01-10") for c in range(1, 21)] + [(c, "2025-02-10") for c in range(1, 21)]
    _write_raw(cfg.raw_dir, txns)

    d = resolve_cutoff(spark, cfg, execution_date="2025-03-15")
    assert d["shifted"] is False
    assert d["cutoff"] == cfg.label["cutoff_date"]
    assert d["definition"] == "absolute_inactivity"


def test_fresh_healthy_window_rolls_forward(spark, cfg):
    cfg.checkpoint["min_eligible"] = 5
    # 50 eligible (Apr txn). ~20% inactive in [May,Jul) -> base rate 0.20 (in band).
    # Aug txn on cid 1 pushes data_max to Aug so candidate = 2025-05-01 (> default).
    txns = [(c, "2025-04-10") for c in range(1, 51)]
    txns += [(c, "2025-06-15") for c in range(1, 51) if c % 5 != 0]   # active -> not churn
    txns += [(1, "2025-08-15")]                                        # sets data_max
    _write_raw(cfg.raw_dir, txns)

    d = resolve_cutoff(spark, cfg, execution_date="2025-09-01")
    assert d["shifted"] is True
    assert d["cutoff"] == "2025-05-01"
    assert d["definition"] == "relative_inactivity"
    assert 0.05 <= d["base_rate"] <= 0.40


def test_degenerate_windows_fall_back_to_default(spark, cfg):
    cfg.checkpoint["min_eligible"] = 5
    # Everyone transacts every month Feb..Aug -> every forward window above the default
    # is fully active (base rate ~0) -> no healthy cutoff -> fall back to default.
    months = ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08"]
    txns = [(c, f"{m}-12") for c in range(1, 31) for m in months]
    _write_raw(cfg.raw_dir, txns)

    d = resolve_cutoff(spark, cfg, execution_date="2025-09-01")
    assert d["shifted"] is False
    assert d["cutoff"] == cfg.label["cutoff_date"]
