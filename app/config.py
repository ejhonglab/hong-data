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
EXPORTS_DIR = DATA_DIR / "exports"  # downloads being sent (deleted once sent)
STRUCTURES_DIR = DATA_DIR / "odor_structures"  # PubChem structure images, by CID
ODOR_PREVIEWS_DIR = DATA_DIR / "odor_previews"  # parsed sheets awaiting confirmation, by fingerprint

MAX_UPLOAD_MB = float(os.environ.get("OLFDB_MAX_UPLOAD_MB", "1024"))
STAGE_TTL_HOURS = float(os.environ.get("OLFDB_STAGE_TTL_HOURS", "48"))

# Google sheet with odor metadata (link as copied from the browser), shared as "anyone with the link can view"
ODOR_SHEET_URL = os.environ.get("OLFDB_ODOR_SHEET_URL", "").strip()
ODOR_SHEET_CHECK_MINUTES = float(os.environ.get("OLFDB_ODOR_SHEET_CHECK_MINUTES", "10"))
ODOR_TEMPLATE = ROOT / "config" / "odor_sheet_template.csv"

# the lab NAS as mounted on the server; users' data folders are given relative to it
TENSOR_ROOT = Path(os.environ.get("OLFDB_TENSOR_ROOT", "/mnt/tank0/tensor"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, STAGING_DIR, DATASETS_DIR, EXPORTS_DIR, STRUCTURES_DIR, ODOR_PREVIEWS_DIR):
        d.mkdir(parents=True, exist_ok=True)
