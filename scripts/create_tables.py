"""Promote the uploaded raw CSVs into typed UC Delta tables (idempotent: CREATE OR
REPLACE), via a serverless SQL warehouse. Run after `deploy.py upload`.
"""
from __future__ import annotations

import os
import sys
import time

from churn_dbx.config import RAW_FILES, RAW_TABLES, _load_dotenv

# Typed projection per dataset. Casting to the target types coerces the handful of
# bad rows to NULL (non-ANSI) and drops read_files' _rescued_data column.
_SELECTS = {
    "customers": """
        CAST(customer_id AS INT) AS customer_id,
        first_name, last_name, email, mobile, gender,
        CAST(date_of_birth AS DATE) AS date_of_birth,
        CAST(signup_date AS DATE) AS signup_date""",
    "enrollments": """
        CAST(product_id AS INT) AS product_id,
        CAST(customer_id AS INT) AS customer_id,
        product_type,
        CAST(enrollment_date AS DATE) AS enrollment_date,
        CAST(`limit` AS DOUBLE) AS `limit`""",
    "transactions": """
        CAST(product_id AS INT) AS product_id,
        CAST(customer_id AS INT) AS customer_id,
        CAST(transaction_amount AS DOUBLE) AS transaction_amount,
        CAST(closing_balance AS DOUBLE) AS closing_balance,
        CAST(transaction_date AS TIMESTAMP) AS transaction_date,
        CAST(transaction_id AS BIGINT) AS transaction_id""",
    "crm": """
        CAST(interaction_id AS BIGINT) AS interaction_id,
        CAST(customer_id AS INT) AS customer_id,
        interaction_type,
        CAST(interaction_date AS DATE) AS interaction_date""",
}


def _pick_warehouse(w):
    wid = os.environ.get("CHURN_DBX_WAREHOUSE_ID")
    if wid:
        return wid
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("no SQL warehouse found; create one or set CHURN_DBX_WAREHOUSE_ID")
    running = [x for x in warehouses if str(getattr(x.state, "value", x.state)) == "RUNNING"]
    return (running[0] if running else warehouses[0]).id


def _ensure_running(w, warehouse_id: str) -> None:
    wh = w.warehouses.get(warehouse_id)
    if str(getattr(wh.state, "value", wh.state)) != "RUNNING":
        print(f"[create_tables] starting warehouse {warehouse_id} ...")
        w.warehouses.start(warehouse_id).result(timeout=__import__("datetime").timedelta(minutes=10))


def _execute(w, warehouse_id: str, catalog: str, schema: str, sql: str):
    from databricks.sdk.service.sql import StatementState

    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id, catalog=catalog, schema=schema, statement=sql, wait_timeout="50s"
    )
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"statement failed ({resp.status.state}): "
                           f"{getattr(resp.status.error, 'message', resp.status)}")
    return resp


def create_tables(env_file: str = ".env") -> int:
    _load_dotenv(env_file)
    from databricks.sdk import WorkspaceClient

    catalog = os.environ.get("CATALOG", "workspace")
    schema = os.environ.get("SCHEMA", "churn_dev")
    volume = os.environ.get("VOLUME", "raw")
    volume_root = f"/Volumes/{catalog}/{schema}/{volume}/raw"

    w = WorkspaceClient()
    try:
        w.schemas.create(name=schema, catalog_name=catalog)
    except Exception:
        pass  # already exists

    warehouse_id = _pick_warehouse(w)
    _ensure_running(w, warehouse_id)
    print(f"[create_tables] warehouse={warehouse_id} target={catalog}.{schema}")

    for key, table in RAW_TABLES.items():
        csv = f"{volume_root}/{RAW_FILES[key]}"
        sql = (
            f"CREATE OR REPLACE TABLE {catalog}.{schema}.{table} AS\n"
            f"SELECT {_SELECTS[key]}\n"
            f"FROM read_files('{csv}', format => 'csv', sep => '|', header => true, "
            f"mode => 'PERMISSIVE')"
        )
        _execute(w, warehouse_id, catalog, schema, sql)
        cnt = _execute(w, warehouse_id, catalog, schema,
                       f"SELECT COUNT(*) FROM {catalog}.{schema}.{table}")
        rows = cnt.result.data_array[0][0] if cnt.result and cnt.result.data_array else "?"
        print(f"[create_tables] {catalog}.{schema}.{table:<20} rows={rows}")
    print("[create_tables] done")
    return 0


if __name__ == "__main__":
    sys.exit(create_tables(sys.argv[1] if len(sys.argv) > 1 else ".env"))
