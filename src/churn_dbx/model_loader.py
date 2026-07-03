"""Resolves the production model behind a small `Scorer` interface
(`.score(spark_df) -> DataFrame[customer_id, churn_score]`, `.model_version`), so
scoring/drift/tests don't depend on how the model is stored. See registry.py.
"""
from __future__ import annotations

from pyspark.sql import DataFrame

from churn_dbx import registry
from churn_dbx.config import Config
from churn_dbx.features import FEATURE_COLUMNS
from churn_dbx.spark import get_spark


class SklearnModelScorer:
    def __init__(self, model, model_version: str) -> None:
        self._model = model
        self.model_version = model_version

    def score(self, features: DataFrame) -> DataFrame:
        # sklearn estimator lives on the driver; collect the small feature frame,
        # predict churn probability, return a Spark frame.
        spark = features.sparkSession
        pdf = features.select("customer_id", *FEATURE_COLUMNS).toPandas()
        pdf["churn_score"] = self._model.predict_proba(pdf[FEATURE_COLUMNS])[:, 1].astype(float)
        return spark.createDataFrame(pdf[["customer_id", "churn_score"]])


def load_scorer(cfg: Config) -> SklearnModelScorer:
    """Load the champion model from the registry; raise if none is promoted."""
    spark = get_spark("churn-load-model")
    champ = registry.by_alias(spark, cfg, cfg.production_alias)
    if champ is None:
        raise RuntimeError(
            f"no champion model at {cfg.model_name()}@{cfg.production_alias}; "
            f"train+promote first."
        )
    model = registry.load_model(champ["model_path"])
    return SklearnModelScorer(model, model_version=f"{cfg.model_name()} v{champ['version']}")
