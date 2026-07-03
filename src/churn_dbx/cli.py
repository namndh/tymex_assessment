"""Console-script entry points — the stable interface the Databricks Jobs call, so
orchestration never imports module paths."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

from churn_dbx.config import Config
from churn_dbx.spark import get_spark

_ENV_ARGS = {"catalog": "CATALOG", "schema": "SCHEMA", "volume": "VOLUME", "mode": "CHURN_DBX_MODE"}


def _load_cfg(extra=None) -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=None)
    for name in _ENV_ARGS:
        parser.add_argument(f"--{name}", default=None)
    if extra:
        extra(parser)
    args = parser.parse_args()
    for name, env_name in _ENV_ARGS.items():
        val = getattr(args, name)
        if val:
            os.environ[env_name] = val
    return Config.load(args.env_file), args


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _run_stage(app_name: str, fn):
    spark = get_spark(app_name)
    try:
        return fn(spark)
    finally:
        spark.stop()


def _cutoff_args(p) -> None:
    p.add_argument("--cutoff-date", default=None)      # manual override, wins over checkpoint
    p.add_argument("--execution-date", default=None)   # run date the checkpoint rolls to


def materialize() -> None:
    from churn_dbx.materialize import materialize as run
    cfg, args = _load_cfg(_cutoff_args)
    _run_stage("churn-materialize", lambda spark: run(spark, cfg, args.cutoff_date, args.execution_date))


def train() -> None:
    from churn_dbx.train import train as run
    cfg, args = _load_cfg(lambda p: p.add_argument("--execution-date", default=None))
    _run_stage("churn-train", lambda spark: run(spark, cfg, args.execution_date))


def promote() -> None:
    from churn_dbx.promote import promote as run
    cfg, _ = _load_cfg()
    decision = _run_stage("churn-promote", lambda spark: run(cfg, spark))
    champ = decision.get("champion_roc_auc")
    print("[promote] candidate v{v} roc_auc={a} ap_lift={l} champion={c} -> {d} ({r})".format(
        v=decision.get("version"), a=_fmt(decision.get("roc_auc")), l=_fmt(decision.get("ap_lift_over_base")),
        c=_fmt(champ), d="PROMOTED" if decision["promote"] else "REJECTED", r=decision["reason"]))
    # Non-zero exit on rejection = failed release = automatic rollback (alias unchanged).
    if not decision["promote"]:
        sys.exit(1)


def score() -> None:
    from churn_dbx.batch_score import batch_score as run
    cfg, args = _load_cfg(lambda p: p.add_argument("--scored-date", default=None))
    _run_stage("churn-score", lambda spark: run(spark, cfg, _parse_date(args.scored_date)))


def monitor() -> None:
    from churn_dbx.drift_check import drift_check as run
    cfg, args = _load_cfg(lambda p: p.add_argument("--compare-cutoff", default=None))
    _run_stage("churn-monitor", lambda spark: run(spark, cfg, _parse_date(args.compare_cutoff)))
    # ALWAYS exit 0: the ok/breach token is the signal, not the exit code.


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"
