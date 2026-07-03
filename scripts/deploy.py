"""One-command deploy CLI: `deploy.py all` (or `--stage <name>` for one stage).
Idempotent; each stage prints a status line, exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from churn_dbx.config import RAW_FILES, _load_dotenv

STAGES = ["preflight", "data", "upload", "deploy", "run", "serve", "report"]


class Ctx:
    """Carries resolved config + accumulated results across stages."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.host = os.environ.get("DATABRICKS_HOST", "")
        self.token = os.environ.get("DATABRICKS_TOKEN", "")
        self.catalog = os.environ.get("CATALOG", "workspace")
        self.schema = os.environ.get("SCHEMA", "churn_dev")
        self.volume = os.environ.get("VOLUME", "raw")
        self.data_url = os.environ.get("DATA_URL", "")
        self.raw_dir = os.environ.get("CHURN_DBX_RAW_DIR", "")
        self.pipeline_job = "churn_pipeline"
        self.app_name = "churn_serving"
        self.local_data: Path | None = None
        self.report: list[tuple[str, str]] = []
        self._client = None

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}/raw"

    def client(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient

            # Explicit .env creds if present, else the CLI profile (~/.databrickscfg).
            if self.host and self.token:
                self._client = WorkspaceClient(host=self.host, token=self.token)
            else:
                self._client = WorkspaceClient()
        return self._client

    def add(self, key: str, value: str) -> None:
        self.report.append((key, value))


def _ok(msg: str) -> tuple[bool, str]:
    return True, msg


def _fail(msg: str) -> tuple[bool, str]:
    return False, msg


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


# --- stages -----------------------------------------------------------------

def preflight(ctx: Ctx) -> tuple[bool, str]:
    # Verify auth actually works, not just that env vars are present.
    try:
        me = ctx.client().current_user.me()
    except Exception as exc:
        return _fail(f"auth failed (set DATABRICKS_HOST/TOKEN in .env or a CLI profile): {exc}")
    code, out = _run(["databricks", "bundle", "validate", "-t", ctx.target])
    if code != 0:
        return _fail(f"bundle validate failed:\n{out}")
    ctx.add("workspace", ctx.host or getattr(ctx.client().config, "host", "?"))
    ctx.add("user", getattr(me, "user_name", "?"))
    return _ok(f"authenticated as {getattr(me, 'user_name', '?')}; bundle valid")


def _resolve_source(ctx: Ctx) -> Path:
    """Return a local dir containing the four raw CSVs, downloading/unzipping DATA_URL
    if needed. Falls back to CHURN_DBX_RAW_DIR when DATA_URL is unset."""
    if not ctx.data_url:
        if ctx.raw_dir and Path(ctx.raw_dir).is_dir():
            return Path(ctx.raw_dir)
        raise RuntimeError("DATA_URL unset and CHURN_DBX_RAW_DIR is not a directory")

    parsed = urlparse(ctx.data_url)
    tmp = Path(tempfile.mkdtemp(prefix="churn_raw_"))
    if parsed.scheme in ("http", "https"):
        local_zip = tmp / "data.zip"
        urllib.request.urlretrieve(ctx.data_url, local_zip)
        src = local_zip
    else:  # file:// or bare path
        src = Path(unquote(parsed.path if parsed.scheme == "file" else ctx.data_url))
    if src.is_dir():
        return src
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
        return tmp
    raise RuntimeError(f"unsupported DATA_URL target: {src}")


def _find_csv(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def data(ctx: Ctx) -> tuple[bool, str]:
    try:
        src = _resolve_source(ctx)
    except Exception as exc:
        return _fail(str(exc))
    found = {k: _find_csv(src, fn) for k, fn in RAW_FILES.items()}
    missing = [RAW_FILES[k] for k, v in found.items() if v is None]
    if missing:
        return _fail(f"raw CSVs not found under {src}: {', '.join(missing)}")
    ctx.local_data = src
    ctx.add("raw_source", str(src))
    return _ok(f"raw data ready at {src}")


def _ensure_uc(ctx: Ctx) -> None:
    w = ctx.client()
    try:
        w.schemas.create(name=ctx.schema, catalog_name=ctx.catalog)
    except Exception:
        pass
    try:
        from databricks.sdk.service.catalog import VolumeType

        w.volumes.create(catalog_name=ctx.catalog, schema_name=ctx.schema,
                         name=ctx.volume, volume_type=VolumeType.MANAGED)
    except Exception:
        pass


def upload(ctx: Ctx) -> tuple[bool, str]:
    if ctx.local_data is None:
        ok, msg = data(ctx)
        if not ok:
            return _fail(msg)
    _ensure_uc(ctx)
    w = ctx.client()
    uploaded = 0
    for fn in RAW_FILES.values():
        local = _find_csv(ctx.local_data, fn)
        with open(local, "rb") as fh:
            w.files.upload(f"{ctx.volume_root}/{fn}", fh, overwrite=True)
        uploaded += 1
    ctx.add("uploaded_to", ctx.volume_root)
    return _ok(f"uploaded {uploaded} CSVs to {ctx.volume_root}")


def deploy(ctx: Ctx) -> tuple[bool, str]:
    code, out = _run(["databricks", "bundle", "deploy", "-t", ctx.target])
    if code != 0:
        return _fail(f"bundle deploy failed:\n{out}")
    ctx.add("bundle", f"deployed ({ctx.target})")
    return _ok(f"bundle deployed to target {ctx.target}")


def run_pipeline(ctx: Ctx) -> tuple[bool, str]:
    code, out = _run(["databricks", "bundle", "run", ctx.pipeline_job, "-t", ctx.target])
    ctx.add("pipeline_run", "SUCCESS" if code == 0 else "FAILED")
    if code != 0:
        return _fail(f"pipeline run failed:\n{out[-1500:]}")
    return _ok("churn_pipeline run reached SUCCESS")


def serve(ctx: Ctx) -> tuple[bool, str]:
    # `deploy` registered the app resource; `bundle run` deploys its code and starts
    # the app compute (the App only goes live after this, not on bundle deploy alone).
    code, out = _run(["databricks", "bundle", "run", ctx.app_name, "-t", ctx.target])
    ctx.add("app", "started" if code == 0 else "FAILED")
    if code != 0:
        return _fail(f"app deploy failed:\n{out[-1500:]}")
    return _ok("churn_serving app deployed + started")


def report(ctx: Ctx) -> tuple[bool, str]:
    print("\n" + "=" * 60)
    print(f"  DEPLOY REPORT — target={ctx.target}")
    print("=" * 60)
    for key, val in ctx.report:
        print(f"  {key:<18} {val}")
    print("=" * 60)
    return _ok("report printed")


_FUNCS = {
    "preflight": preflight, "data": data, "upload": upload, "deploy": deploy,
    "run": run_pipeline, "serve": serve, "report": report,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Databricks Free Edition deploy driver")
    parser.add_argument("action", nargs="?", default="all", help="'all' or a stage name")
    parser.add_argument("--stage", default=None, help="run a single stage")
    parser.add_argument("-t", "--target", default=os.environ.get("TARGET", "dev"))
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)

    _load_dotenv(args.env_file)
    ctx = Ctx(args.target)

    stage = args.stage or (None if args.action == "all" else args.action)
    to_run = [stage] if stage else STAGES
    if stage and stage not in _FUNCS:
        print(f"unknown stage '{stage}'; choose from {', '.join(STAGES)}")
        return 2

    failed = False
    for name in to_run:
        if name == "report":
            report(ctx)
            continue
        ok, msg = _FUNCS[name](ctx)
        marker = "OK " if ok else "ERR"
        print(f"[{marker}] {name}: {msg}")
        if not ok:
            failed = True
            break
    if not stage:
        report(ctx)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
