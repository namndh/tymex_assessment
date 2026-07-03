"""Quality gate + alias move (deliverable 3). Pass -> move `champion` to the
candidate (the alias move IS the deploy). Fail -> exit non-zero, alias untouched —
failure-to-promote is the automatic rollback.
"""
from __future__ import annotations

from churn_dbx import registry
from churn_dbx.config import Config
from churn_dbx.spark import get_spark


def decide(cand_auc: float, cand_lift: float, champion_auc: float | None, gate: dict) -> tuple[bool, str]:
    """Pure gate logic (no storage) so the rules are unit-testable directly."""
    failed = []
    if cand_auc < gate["min_roc_auc"]:
        failed.append("min_roc_auc")
    if cand_lift < gate["min_ap_lift_over_base"]:
        failed.append("min_ap_lift_over_base")
    if champion_auc is not None and cand_auc < champion_auc - gate["max_roc_auc_regression"]:
        failed.append("no_regression")
    if failed:
        return False, "failed: " + ", ".join(failed)
    return True, "passed all gate checks"


def evaluate_gate(cfg: Config, spark) -> dict:
    candidate = registry.latest(spark, cfg)
    if candidate is None:
        return {"promote": False, "reason": "no registered model version found", "version": None}

    cand_auc = candidate.get("roc_auc")
    cand_lift = candidate.get("ap_lift_over_base")
    if cand_auc is None or cand_lift is None:
        return {"promote": False, "reason": "candidate missing gate metrics",
                "version": candidate["version"]}

    # Skip the regression check when the champion IS this candidate (already promoted).
    champ = registry.by_alias(spark, cfg, cfg.production_alias)
    champion_auc = None
    if champ is not None and int(champ["version"]) != int(candidate["version"]):
        champion_auc = champ.get("roc_auc")

    do_promote, reason = decide(cand_auc, cand_lift, champion_auc, cfg.quality_gate)
    return {
        "promote": do_promote,
        "reason": reason,
        "version": candidate["version"],
        "roc_auc": cand_auc,
        "ap_lift_over_base": cand_lift,
        "champion_roc_auc": champion_auc,
    }


def promote(cfg: Config, spark=None) -> dict:
    """Evaluate the gate and, on pass, move the production alias to the candidate."""
    spark = spark or get_spark("churn-promote")
    decision = evaluate_gate(cfg, spark)
    if decision["promote"] and decision["version"] is not None:
        registry.set_alias(spark, cfg, decision["version"], cfg.production_alias)
        # Reset the drift baseline to the new champion's scores, so monitoring compares
        # against the model+world we just promoted (a stale baseline false-breaches on
        # organic growth). Best-effort — the release already succeeded.
        try:
            from churn_dbx.drift_check import rebuild_baseline
            rebuild_baseline(spark, cfg)
        except Exception as exc:
            print(f"[promote] warning: drift baseline refresh failed ({exc}); "
                  f"promotion still succeeded")
    return decision
