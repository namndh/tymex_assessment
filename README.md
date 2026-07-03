# Customer 360 Churn — Databricks Free Edition (PySpark)

Churn prediction on **Databricks Free Edition**: raw CSVs → feature store → training →
gated registry → batch scoring → serving App, brought up by one deploy script.

**This README is how to run it.** The churn definition, platform choice, and every
design decision are in **[assessment_report.md](assessment_report.md)**.

The same wheel runs two ways, set by `CHURN_DBX_MODE`:
- **local** — Spark-local + parquet + sqlite MLflow (tests, CI, rehearsal).
- **uc** — Databricks serverless + Unity Catalog + managed MLflow (real deploy).

## Layout

```
src/churn_dbx/     features.py labels.py materialize.py train.py promote.py
                   batch_score.py drift_check.py model_loader.py registry.py io.py
                   config.py spark.py mlflow_utils.py cli.py
app/               FastAPI served as a Databricks App (GET /predict/{customer_id})
resources/         jobs.yml (serverless jobs) app.yml (serving App) volume.yml
scripts/deploy.py  one-command deploy CLI (data -> upload -> deploy -> run -> serve -> report)
tests/             Spark-local: features, labels, gate, checkpoint, end-to-end
notebooks/         cutoff_metrics_analysis.ipynb (label EDA, local-only, outputs committed)
databricks.yml     Asset Bundle (targets: dev / prod)
```

## Prerequisites

- **Python 3.10+**.
- **Databricks CLI** ≥ 0.230 (`databricks version`) — for the bundle deploy/run.
- A **Databricks Free Edition** workspace (only for the real deploy, not for local tests).
- A **Personal Access Token** — workspace UI → Settings → Developer → Access tokens.

## Install

```bash
python -m pip install -e ".[dev]"     # + [local] for Spark-local dev/CI
```

## Get the data

Download the case-study zip and unpack the four pipe-delimited CSVs into `./data/`
(gitignored, not shipped in this repo):

```bash
mkdir -p data
# unzip your case-study zip into data/, or download it there first, e.g.:
# curl -L -o data/customer360.zip <your hosted zip URL>
# unzip data/customer360.zip -d data
```

`data/` must directly contain `customer_raw.csv`, `product_enrollments.csv`,
`transaction_history.csv`, `crm_interactions.csv`. Both the local check
(`CHURN_DBX_RAW_DIR`) and the deploy script's `data` stage (`DATA_URL`) read from
here by default (`.env.example`) — get the files in place **before** running either,
or `materialize`/`deploy.py data` fail on a missing path.

## Notebook

Local-only EDA behind [assessment_report.md](assessment_report.md) §1, committed with
outputs. Rerun:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/cutoff_metrics_analysis.ipynb
```

## Local check (no Databricks needed)

```bash
pytest -q                       # 19 Spark-local tests

cp .env.example .env            # local section points CHURN_DBX_RAW_DIR at ./data
churn-dbx-materialize           # -> features table
churn-dbx-train                 # sklearn run (MLflow-tracked), writes a model version
churn-dbx-promote               # quality gate -> champion alias
churn-dbx-score                 # score the base -> predictions partition
churn-dbx-monitor               # score-PSI drift check -> ok/breach token
```

`materialize`/`train` accept `--execution-date` (the Jobs pass the run date) and
`--cutoff-date` (pin a specific cutoff); how these drive the adaptive cutoff and why
training is idempotent are in [assessment_report.md](assessment_report.md) §4.1, §5.

## Deploy to Free Edition

Fill `DATABRICKS_HOST` / `DATABRICKS_TOKEN` / `CATALOG` / `SCHEMA` / `VOLUME` in
`.env` (`.env` is gitignored — never commit it). `DATA_URL` defaults to the `./data`
folder from "Get the data" above; point it at an https URL instead if you host the
zip remotely. Then:

```bash
python scripts/deploy.py all               # data -> upload -> deploy -> run -> serve -> report
python scripts/deploy.py --stage serve     # or run any single stage
python scripts/deploy.py all -t prod       # non-default target
```

### Reach the endpoint

The Apps proxy is **OAuth-only** — a PAT gets `403`, and so does the Databricks CLI's
own `apps logs` (`OAuth Token not supported for current auth type pat`). One-time
browser login, then curl with the resulting bearer token:

```bash
databricks auth login --host $DATABRICKS_HOST      # opens a browser, once

TOKEN=$(databricks auth token --host $DATABRICKS_HOST -o json | jq -r .access_token)
APP_URL=$(databricks apps get churn-serving -o json | jq -r .url)
curl -s -H "Authorization: Bearer $TOKEN" "$APP_URL/health"
curl -s -H "Authorization: Bearer $TOKEN" "$APP_URL/predict/1001"
```

For programmatic (non-browser) callers, use a service principal with `CAN_USE` on the
app and an M2M OAuth client instead of a user login.

## CI/CD

- `PR → main` / `push → main`: ruff + pytest + `bundle validate`.
- Tag push: `…-rc…` → **dev**, release tag → **prod** (protected environment). Deploy
  is gated by `DATABRICKS_DEPLOY_ENABLED=true`.
