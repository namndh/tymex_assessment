"""Runtime config — `mode` switches local-Spark vs Databricks backends; ML decisions
(label, gate, model, drift) are plain dicts, identical in both modes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RAW_FILES = {
    "customers": "customer_raw.csv",
    "enrollments": "product_enrollments.csv",
    "transactions": "transaction_history.csv",
    "crm": "crm_interactions.csv",
}

# Absolute inactivity over a pre-spike window: cutoff + 90d ends before the
# June-2025 bulk reactivation (see docs/label_decision).
LABEL = {
    "definition": "absolute_inactivity",
    "cutoff_date": "2025-03-01",
    "window_days": 90,
    "min_prior_txns": 1,
    "inactive_percentile": 0.10,
}
# `candidates` are the families train fits and compares each run (winner is registered).
MODEL = {
    "type": "gradient_boosting",
    "candidates": ["gradient_boosting", "logistic_regression"],
    "test_size": 0.2,
    "random_state": 42,
}
# Baseline-grounded, not absolute: behaviour has a ~0.62 AUC ceiling on this data.
QUALITY_GATE = {"min_roc_auc": 0.58, "min_ap_lift_over_base": 1.10, "max_roc_auc_regression": 0.02}
MONITORING = {"psi_threshold": 0.25}
# Adaptive cutoff: retrain rolls forward to the newest cutoff whose label stays healthy
# (base rate in band, enough eligible), else falls back to LABEL["cutoff_date"].
CHECKPOINT = {
    "enabled": True,
    "base_rate_min": 0.05,
    "base_rate_max": 0.40,
    "min_eligible": 1000,
    "max_lookback_months": 12,
}

FEATURES_TABLE = "features"
PREDICTIONS_TABLE = "predictions"
DRIFT_BASELINE_TABLE = "drift_baseline"
MODEL_REGISTRY_TABLE = "model_registry"

# Raw CSVs promoted into UC Delta tables (scripts/create_tables.py); the analysis
# notebook reads these instead of the CSVs.
RAW_TABLES = {
    "customers": "customers",
    "enrollments": "product_enrollments",
    "transactions": "transaction_history",
    "crm": "crm_interactions",
}


def _load_dotenv(path: str | os.PathLike) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


@dataclass(frozen=True)
class Config:
    mode: str
    catalog: str
    schema: str
    volume: str
    local_root: str
    raw_dir: str
    data_url: str
    tracking_uri: str
    registry_uri: str
    experiment: str
    model_base: str
    production_alias: str
    label: dict
    model: dict
    quality_gate: dict
    monitoring: dict
    checkpoint: dict

    @property
    def is_uc(self) -> bool:
        return self.mode == "uc"

    def model_name(self) -> str:
        """3-level UC name on Databricks; bare name for the local sqlite registry."""
        return f"{self.catalog}.{self.schema}.{self.model_base}" if self.is_uc else self.model_base

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}" if self.is_uc else name

    def raw_table(self, key: str) -> str:
        name = RAW_TABLES[key]
        return f"{self.catalog}.{self.schema}.{name}" if self.is_uc else name

    def warehouse_dir(self) -> str:
        return str(Path(self.local_root) / "warehouse")

    def table_path(self, name: str) -> str:
        return str(Path(self.warehouse_dir()) / name)

    def artifact_root(self) -> str:
        return str(Path(self.local_root).resolve() / "mlartifacts")

    def models_root(self) -> str:
        """Where model pickles live: a UC Volume (local dir in local mode), not the
        MLflow registry — see registry.py."""
        if self.is_uc:
            return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}/models"
        return str(Path(self.local_root).resolve() / "models")

    @classmethod
    def load(cls, env_file: str | None = None) -> Config:
        _load_dotenv(env_file or os.environ.get("CHURN_DBX_ENV_FILE", ".env"))

        on_databricks = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))
        mode = os.environ.get("CHURN_DBX_MODE", "uc" if on_databricks else "local")
        catalog = os.environ.get("CATALOG", "workspace")
        schema = os.environ.get("SCHEMA", "churn_dev")
        volume = os.environ.get("VOLUME", "raw")
        local_root = os.environ.get("CHURN_DBX_LOCAL_ROOT", "./_local")

        if mode == "uc":
            raw_dir = os.environ.get("CHURN_DBX_RAW_DIR", f"/Volumes/{catalog}/{schema}/{volume}/raw")
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
            registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", "databricks-uc")
        else:
            raw_dir = os.environ.get("CHURN_DBX_RAW_DIR", str(Path(local_root) / "raw"))
            sqlite_uri = f"sqlite:///{Path(local_root).resolve() / 'mlflow.db'}"
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", sqlite_uri)
            registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", sqlite_uri)

        return cls(
            mode=mode,
            catalog=catalog,
            schema=schema,
            volume=volume,
            local_root=local_root,
            raw_dir=raw_dir,
            data_url=os.environ.get("DATA_URL", ""),
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            experiment=os.environ.get("CHURN_DBX_EXPERIMENT", "customer360_churn"),
            model_base=os.environ.get("CHURN_DBX_MODEL", "churn_model"),
            production_alias=os.environ.get("CHURN_DBX_ALIAS", "champion"),
            label=dict(LABEL),
            model=dict(MODEL),
            quality_gate=dict(QUALITY_GATE),
            monitoring=dict(MONITORING),
            checkpoint=dict(CHECKPOINT),
        )
