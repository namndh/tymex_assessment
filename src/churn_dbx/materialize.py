"""Materialize the offline feature+label table (deliverable 1) — features join labels
at the cutoff resolved by the checkpoint, unless one is passed explicitly.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from churn_dbx.checkpoint import resolve_cutoff
from churn_dbx.config import FEATURES_TABLE, Config
from churn_dbx.features import compute_features
from churn_dbx.io import read_raw, write_table
from churn_dbx.labels import build_labels


def resolve_decision(spark: SparkSession, cfg: Config, cutoff=None, execution_date=None) -> dict:
    """The cutoff/label decision for this run: an explicit `cutoff` is a manual override;
    otherwise the checkpoint decides."""
    if cutoff:
        lbl = cfg.label
        return {"cutoff": str(cutoff), "window_days": lbl["window_days"],
                "definition": lbl["definition"], "inactive_percentile": lbl.get("inactive_percentile"),
                "shifted": False, "reason": "manual --cutoff-date override",
                "base_rate": float("nan"), "eligible": -1}
    return resolve_cutoff(spark, cfg, execution_date)


def build_feature_table(spark: SparkSession, cfg: Config, cutoff=None, execution_date=None,
                        decision: dict | None = None) -> DataFrame:
    decision = decision or resolve_decision(spark, cfg, cutoff, execution_date)
    raw = read_raw(spark, cfg)

    features = compute_features(
        raw["customers"], raw["enrollments"], raw["transactions"], raw["crm"], decision["cutoff"]
    )
    ip = decision["inactive_percentile"] if decision["definition"] == "relative_inactivity" else None
    labels = build_labels(
        raw["transactions"],
        cutoff=decision["cutoff"],
        window_days=decision["window_days"],
        min_prior_txns=cfg.label.get("min_prior_txns", 1),
        inactive_percentile=ip,
    )
    return labels.join(features, on="customer_id", how="inner")


def materialize(spark: SparkSession, cfg: Config, cutoff=None, execution_date=None) -> str:
    decision = resolve_decision(spark, cfg, cutoff, execution_date)
    table = build_feature_table(spark, cfg, decision=decision)
    out = write_table(spark, cfg, table, FEATURES_TABLE)
    rows = table.count()
    rate = table.agg({"churn": "avg"}).collect()[0][0] if rows else float("nan")
    print(f"[materialize] cutoff={decision['cutoff']} definition={decision['definition']} "
          f"shifted={decision['shifted']} rows={rows} churn_rate={rate:.3f} -> {out}")
    print(f"[materialize] checkpoint: {decision['reason']}")
    return out
