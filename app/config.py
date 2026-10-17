"""Runtime configuration (environment variables, all optional)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("OLFDB_DATA_DIR", ROOT / "data")).resolve()
REFERENCE_DIR = Path(os.environ.get("OLFDB_REFERENCE_DIR", ROOT / "reference_data")).resolve()
STATIC_DIR = ROOT / "app" / "static"

DB_PATH = DATA_DIR / "olfdb.sqlite3"
STAGING_DIR = DATA_DIR / "staging"
DATASETS_DIR = DATA_DIR / "datasets"

MAX_UPLOAD_MB = float(os.environ.get("OLFDB_MAX_UPLOAD_MB", "1024"))
STAGE_TTL_HOURS = float(os.environ.get("OLFDB_STAGE_TTL_HOURS", "48"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, STAGING_DIR, DATASETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
