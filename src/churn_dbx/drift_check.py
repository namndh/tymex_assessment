"""Drift monitoring (deliverable 8) — score-distribution PSI vs the champion baseline
is the sole trigger (growth-robust; count/feature PSI false-fire on organic growth).
Always exits 0 — the ok/breach token, read by the caller, carries the decision.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from churn_dbx.config import DRIFT_BASELINE_TABLE, Config
from churn_dbx.features import compute_features
from churn_dbx.io import read_raw, read_table, table_exists, write_table
from churn_dbx.model_loader import load_scorer

_BASELINE_SCHEMA = T.StructType([
    T.StructField("score", T.DoubleType()),
    T.StructField("model_version", T.StringType()),
])


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate, >0.25 significant."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(expected, edges)[0] / len(expected)
    a_pct = np.histogram(actual, edges)[0] / len(actual)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _scores_at(spark: SparkSession, cfg: Config, scorer, cutoff) -> np.ndarray:
    raw = read_raw(spark, cfg)
    feats = compute_features(
        raw["customers"], raw["enrollments"], raw["transactions"], raw["crm"], str(cutoff)
    ).where("txn_count >= 1")
    return np.array([r["churn_score"] for r in scorer.score(feats).collect()])


def build_baseline(spark: SparkSession, cfg: Config, scorer, cutoff: date | None = None) -> np.ndarray:
    """Snapshot the champion's score distribution at `cutoff` (default: today) as the
    drift baseline. Called on promotion so 'normal' tracks the newly promoted model and
    current world — a baseline frozen at a fixed cutoff false-breaches on organic growth."""
    cutoff = cutoff or datetime.now(timezone.utc).date()
    scores = _scores_at(spark, cfg, scorer, cutoff)
    rows = [(float(s), scorer.model_version) for s in scores]
    df = spark.createDataFrame(rows, schema=_BASELINE_SCHEMA)
    write_table(spark, cfg, df, DRIFT_BASELINE_TABLE)
    return scores


def rebuild_baseline(spark: SparkSession, cfg: Config) -> str | None:
    """Reset the drift baseline to the current champion's scores at today. Returns the
    champion model version, or None if there is no champion yet."""
    try:
        scorer = load_scorer(cfg)
    except Exception:
        return None
    build_baseline(spark, cfg, scorer)
    return scorer.model_version


def _load_or_build_baseline(spark: SparkSession, cfg: Config, scorer) -> np.ndarray:
    if table_exists(spark, cfg, DRIFT_BASELINE_TABLE):
        return np.array([r["score"] for r in read_table(spark, cfg, DRIFT_BASELINE_TABLE).collect()])
    # No baseline yet: seed at today so the first comparison is champion-vs-itself (ok).
    return build_baseline(spark, cfg, scorer)


def emit_decision(token: str) -> None:
    """Publish the ok/breach token: a Databricks task value for the condition task,
    and stdout for Airflow XCom. dbutils only resolves in a Databricks job."""
    try:
        from databricks.sdk.runtime import dbutils

        dbutils.jobs.taskValues.set(key="decision", value=token)
    except Exception:
        pass
    print(token)


def drift_check(spark: SparkSession, cfg: Config, compare_cutoff: date | None = None) -> str:
    threshold = cfg.monitoring["psi_threshold"]
    try:
        scorer = load_scorer(cfg)
    except Exception as exc:
        print(f"[drift_check] no champion to monitor ({exc}); emitting ok")
        emit_decision("ok")
        return "ok"

    baseline = _load_or_build_baseline(spark, cfg, scorer)
    cutoff = compare_cutoff or datetime.now(timezone.utc).date()
    current = _scores_at(spark, cfg, scorer, cutoff)

    score_psi = psi(baseline, current)
    breached = score_psi > threshold
    print(f"[drift_check] threshold={threshold} SCORE_psi={score_psi:.3f} (trigger) "
          f"population={len(current)} cutoff={cutoff}")
    token = "breach" if breached else "ok"
    emit_decision(token)
    return token
