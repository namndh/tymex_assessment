"""Training (deliverable 2) — scikit-learn on the driver (Free Edition blocks
`pyspark.ml`), MLflow-tracked. Fits every candidate, registers the winner; idempotent
on identical inputs.
"""
from __future__ import annotations

import hashlib
import json

import mlflow
import mlflow.sklearn
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from churn_dbx import registry
from churn_dbx.config import FEATURES_TABLE, Config
from churn_dbx.features import FEATURE_COLUMNS
from churn_dbx.io import read_table, table_exists
from churn_dbx.materialize import build_feature_table, resolve_decision
from churn_dbx.mlflow_utils import setup_mlflow


def build_estimator(cfg: Config, model_type: str):
    """A GBT estimator, or a scaling pipeline for logistic regression. Class imbalance
    (~20% churn) is handled with balanced weights (see `_fit_eval`)."""
    mcfg = cfg.model
    if model_type == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
    return GradientBoostingClassifier(
        n_estimators=mcfg.get("n_estimators", 100),
        max_depth=mcfg.get("max_depth", 3),
        random_state=mcfg["random_state"],
    )


def _precision_at_topk(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    n = max(1, int(len(scores) * k_frac))
    top = np.argsort(scores)[::-1][:n]
    return float(np.asarray(y_true)[top].mean())


def _fit_eval(cfg: Config, model_type: str, X_tr, y_tr, X_te, y_te) -> tuple:
    estimator = build_estimator(cfg, model_type)
    # GBT takes balanced weights at fit; the LR pipeline carries class_weight itself.
    fit_kwargs = {}
    if not isinstance(estimator, Pipeline):
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_tr)
    estimator.fit(X_tr, y_tr, **fit_kwargs)

    p = estimator.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    two_class = len(np.unique(y_te)) > 1
    base_rate = float(y_te.mean()) if len(y_te) else 0.0
    ap = float(average_precision_score(y_te, p)) if two_class else 0.0
    tp = float(((pred == 1) & (y_te == 1)).sum())
    metrics = {
        "roc_auc": float(roc_auc_score(y_te, p)) if two_class else 0.0,
        "average_precision": ap,
        "ap_lift_over_base": ap / base_rate if base_rate else 0.0,
        "precision_at_decile": _precision_at_topk(y_te, p, 0.10),
        "precision": tp / max(1.0, float((pred == 1).sum())),
        "recall": tp / max(1.0, float((y_te == 1).sum())),
        "accuracy": float((pred == y_te).mean()),
    }
    return estimator, metrics


def _load_table(spark: SparkSession, cfg: Config, decision: dict) -> DataFrame:
    if table_exists(spark, cfg, FEATURES_TABLE):
        return read_table(spark, cfg, FEATURES_TABLE)
    return build_feature_table(spark, cfg, decision=decision)


def _input_hash(decision: dict, rows: int, churn_mean: float, cfg: Config) -> str:
    payload = {
        "cutoff": decision["cutoff"], "window": decision["window_days"],
        "definition": decision["definition"], "rows": int(rows),
        "churn_mean": round(float(churn_mean), 6),
        "candidates": cfg.model["candidates"],
        "seed": cfg.model["random_state"], "test_size": cfg.model["test_size"],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def train(spark: SparkSession, cfg: Config, execution_date=None) -> dict:
    setup_mlflow(cfg)
    decision = resolve_decision(spark, cfg, None, execution_date)

    # Sort by customer_id so the train/test split is deterministic across Spark runs.
    pdf = (
        _load_table(spark, cfg, decision)
        .select("customer_id", F.col("churn").cast("int").alias("churn"), *FEATURE_COLUMNS)
        .toPandas()
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    churn_mean = float(pdf["churn"].mean()) if len(pdf) else 0.0
    input_hash = _input_hash(decision, len(pdf), churn_mean, cfg)

    # Idempotency: identical inputs -> reuse the existing version, no duplicate.
    existing = registry.by_input_hash(spark, cfg, input_hash)
    if existing is not None:
        print(f"[train] inputs unchanged (hash={input_hash[:8]}); reusing v{existing['version']}")
        return {"version": int(existing["version"]), "skipped": True, "input_hash": input_hash,
                "roc_auc": existing.get("roc_auc"), "ap_lift_over_base": existing.get("ap_lift_over_base")}

    X = pdf[FEATURE_COLUMNS]
    y = pdf["churn"].to_numpy()
    stratify = y if len(np.unique(y)) > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.model["test_size"],
        random_state=cfg.model["random_state"], stratify=stratify,
    )

    params = {
        "cutoff_date": decision["cutoff"], "window_days": decision["window_days"],
        "label_definition": decision["definition"], "cutoff_shifted": decision["shifted"],
        "n_train": len(X_tr), "n_test": len(X_te),
        "base_rate": round(float(y_te.mean()) if len(y_te) else 0.0, 4),
        "random_state": cfg.model["random_state"],
    }

    with mlflow.start_run() as parent:
        candidates = []
        for model_type in cfg.model["candidates"]:
            with mlflow.start_run(nested=True, run_name=model_type) as child:
                estimator, metrics = _fit_eval(cfg, model_type, X_tr, y_tr, X_te, y_te)
                mlflow.log_params({**params, "model_type": model_type})
                mlflow.log_metrics(metrics)
                try:  # model artifact; best-effort (serverless artifact store may deny)
                    mlflow.sklearn.log_model(estimator, artifact_path="model")
                except Exception as exc:
                    print(f"[train] log_model skipped for {model_type}: {exc}")
                candidates.append({"model_type": model_type, "estimator": estimator,
                                   "metrics": metrics, "run_id": child.info.run_id})

        # Winner = highest ROC-AUC; log the comparison so "why the winner won" is on record.
        winner = max(candidates, key=lambda c: c["metrics"]["roc_auc"])
        mlflow.log_params({**params, "model_type": winner["model_type"],
                           "winning_model": winner["model_type"]})
        mlflow.log_metrics(winner["metrics"])
        for c in candidates:
            mlflow.log_metric(f"cand_{c['model_type']}_roc_auc", c["metrics"]["roc_auc"])
        runner_up = max([c["metrics"]["roc_auc"] for c in candidates if c is not winner], default=0.0)
        mlflow.set_tag("winning_model", winner["model_type"])
        mlflow.set_tag("roc_auc_margin", round(winner["metrics"]["roc_auc"] - runner_up, 4))

        version = registry.register(spark, cfg, winner["estimator"], parent.info.run_id,
                                    winner["metrics"], input_hash=input_hash)
        mlflow.set_tag("model_version", version)
        result = {"run_id": parent.info.run_id, "version": version, "skipped": False,
                  "winning_model": winner["model_type"], "input_hash": input_hash,
                  **winner["metrics"]}

    tried = ", ".join(f"{c['model_type']}={c['metrics']['roc_auc']:.4f}" for c in candidates)
    print(f"[train] tried [{tried}] -> winner={winner['model_type']} "
          f"roc_auc={winner['metrics']['roc_auc']:.4f} registered v{version}")
    return result
