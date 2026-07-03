"""MLflow wiring — points tracking + registry at the configured backend (sqlite +
local artifacts in local mode; managed Databricks + `databricks-uc` in UC mode)."""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from churn_dbx.config import Config


def _experiment_name(cfg: Config) -> str:
    """Databricks needs an absolute workspace path, so resolve a bare name under the
    user's home. Local mode keeps the name as-is."""
    name = cfg.experiment
    if not cfg.is_uc or name.startswith("/"):
        return name
    try:
        from databricks.sdk import WorkspaceClient

        user = WorkspaceClient().current_user.me().user_name
        return f"/Users/{user}/{name}"
    except Exception:
        return f"/Shared/{name}"


def setup_mlflow(cfg: Config) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_registry_uri(cfg.registry_uri)
    if cfg.is_uc:
        mlflow.set_experiment(_experiment_name(cfg))
    else:
        exp = mlflow.get_experiment_by_name(cfg.experiment)
        if exp is None:
            mlflow.create_experiment(cfg.experiment, artifact_location=cfg.artifact_root())
        mlflow.set_experiment(cfg.experiment)


def get_client(cfg: Config) -> MlflowClient:
    return MlflowClient(tracking_uri=cfg.tracking_uri, registry_uri=cfg.registry_uri)


def latest_model_version(client: MlflowClient, name: str):
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))
