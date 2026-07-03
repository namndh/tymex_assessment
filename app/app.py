"""Serving endpoint — reads pre-computed scores from `predictions`
over a SQL warehouse.
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException

CATALOG = os.environ.get("CHURN_DBX_CATALOG", "workspace")
SCHEMA = os.environ.get("CHURN_DBX_SCHEMA", "churn_dev")
TABLE = f"{CATALOG}.{SCHEMA}.predictions"

_client = None
_warehouse_cache: str | None = None


def _w():
    global _client
    if _client is None:
        from databricks.sdk import WorkspaceClient
        _client = WorkspaceClient()
    return _client


def _warehouse_id() -> str:
    global _warehouse_cache
    if _warehouse_cache:
        return _warehouse_cache
    wid = os.environ.get("CHURN_DBX_WAREHOUSE_ID", "").strip()
    if not wid:
        wid = next((wh.id for wh in _w().warehouses.list()), "")
    if not wid:
        raise HTTPException(status_code=503, detail="no SQL warehouse available to the app")
    _warehouse_cache = wid
    return wid


def _query_one(customer_id: int) -> dict | None:
    from databricks.sdk.service.sql import StatementParameterListItem, StatementState

    terminal = {StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED,
                StatementState.CLOSED}
    sql = (
        f"SELECT customer_id, churn_score, model_version, scored_at, scored_date "
        f"FROM {TABLE} WHERE customer_id = :cid ORDER BY scored_date DESC LIMIT 1"
    )
    s = _w().statement_execution.execute_statement(
        warehouse_id=_warehouse_id(), statement=sql, wait_timeout="50s",
        parameters=[StatementParameterListItem(name="cid", value=str(customer_id), type="INT")],
    )
    for _ in range(60):  # first call may wait on a cold warehouse auto-start
        if s.status.state in terminal:
            break
        time.sleep(2)
        s = _w().statement_execution.get_statement(s.statement_id)
    if s.status.state != StatementState.SUCCEEDED:
        raise HTTPException(status_code=502, detail=f"prediction query {s.status.state}")
    rows = (s.result.data_array if s.result else None) or []
    if not rows:
        return None
    r = rows[0]
    return {"customer_id": int(r[0]), "churn_score": float(r[1]),
            "model_version": r[2], "scored_at": r[3], "scored_date": r[4]}


app = FastAPI(title="Customer 360 Churn Serving (Databricks App)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "table": TABLE}


@app.get("/predict/{customer_id}")
def predict(customer_id: int) -> dict:
    try:
        row = _query_one(customer_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"no prediction for customer_id={customer_id}")
    return {**row, "mode": "batch"}


if __name__ == "__main__":
    import uvicorn  # bind DATABRICKS_APP_PORT — the Apps proxy routes here, not 8080

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", "8080")))
