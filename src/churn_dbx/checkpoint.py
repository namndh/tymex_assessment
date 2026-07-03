"""Adaptive cutoff — rolls forward to the newest cutoff whose label stays healthy
(base rate in band), else falls back to the fixed default. Pure function of
(execution_date, raw data, config), so materialize/train agree without shared state.
"""
from __future__ import annotations

from datetime import date, datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from churn_dbx.config import Config
from churn_dbx.io import read_raw
from churn_dbx.labels import build_labels


def _to_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _month_floor(d: date) -> date:
    return date(d.year, d.month, 1)


def _prev_month(d: date) -> date:
    return date(d.year - 1, 12, 1) if d.month == 1 else date(d.year, d.month - 1, 1)


def _minus_days(d: date, days: int) -> date:
    from datetime import timedelta

    return d - timedelta(days=days)


def _health(transactions, cutoff: date, window: int, min_prior: int) -> tuple[float, int]:
    """Absolute-inactivity base rate + eligible count at `cutoff` — the degeneracy signal
    (a spike-reactivated window collapses the rate toward zero)."""
    labels = build_labels(transactions, cutoff=str(cutoff), window_days=window,
                          min_prior_txns=min_prior, inactive_percentile=None)
    row = labels.agg(F.count(F.lit(1)).alias("n"), F.avg("churn").alias("rate")).collect()[0]
    eligible = int(row["n"])
    base_rate = float(row["rate"]) if eligible else 0.0
    return base_rate, eligible


def _default(cfg: Config, reason: str, base_rate: float, eligible: int) -> dict:
    return {
        "cutoff": cfg.label["cutoff_date"],
        "window_days": cfg.label["window_days"],
        "definition": cfg.label["definition"],
        "inactive_percentile": None,
        "shifted": False,
        "reason": reason,
        "base_rate": base_rate,
        "eligible": eligible,
    }


def resolve_cutoff(spark: SparkSession, cfg: Config, execution_date=None) -> dict:
    lbl = cfg.label
    chk = cfg.checkpoint
    window = lbl["window_days"]
    min_prior = lbl.get("min_prior_txns", 1)
    default_cut = _to_date(lbl["cutoff_date"])

    if not chk.get("enabled", True):
        return _default(cfg, "checkpoint disabled", float("nan"), -1)

    transactions = read_raw(spark, cfg)["transactions"]
    row = transactions.agg(F.max("transaction_date").alias("mx")).collect()[0]
    data_max = _to_date(row["mx"])
    if data_max is None:
        return _default(cfg, "no transactions", float("nan"), -1)

    anchor = min(_to_date(execution_date) or data_max, data_max)
    candidate = _month_floor(_minus_days(anchor, window))

    if candidate <= default_cut:
        base_rate, eligible = _health(transactions, default_cut, window, min_prior)
        return _default(cfg, f"no new data past baseline (candidate {candidate} <= "
                        f"default {default_cut})", base_rate, eligible)

    # Walk back month-by-month from the newest observable cutoff to the default, and take
    # the most recent cutoff whose forward window is still a healthy (non-degenerate) label.
    lo, hi = chk["base_rate_min"], chk["base_rate_max"]
    min_elig = chk["min_eligible"]
    cur = candidate
    for _ in range(chk.get("max_lookback_months", 12)):
        if cur <= default_cut:
            break
        base_rate, eligible = _health(transactions, cur, window, min_prior)
        if eligible >= min_elig and lo <= base_rate <= hi:
            return {
                "cutoff": str(cur),
                "window_days": window,
                "definition": "relative_inactivity",
                "inactive_percentile": lbl.get("inactive_percentile", 0.10),
                "shifted": True,
                "reason": (f"rolled to {cur} (base_rate={base_rate:.3f}, eligible={eligible})"
                           + ("" if cur == candidate else f"; newest {candidate} was degenerate")),
                "base_rate": base_rate,
                "eligible": eligible,
            }
        cur = _prev_month(cur)

    base_rate, eligible = _health(transactions, default_cut, window, min_prior)
    return _default(cfg, f"no healthy cutoff above default within lookback; kept {default_cut}",
                    base_rate, eligible)
