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
ODOR_PREVIEWS_DIR = DATA_DIR / "odor_previews"  # parsed sheets awaiting confirmation, by fingerprint
DELETED_DIR = DATA_DIR / "deleted"  # the files of datasets deleted in the Admin tab, by id
REFERENCE_CACHE_DIR = DATA_DIR / "references"  # reference datasets added in the Admin tab, as imported, by id
TMP_DIR = DATA_DIR / "tmp"  # the server's temporary files (uploads being received), so it writes nothing outside DATA_DIR
ADMIN_FILE = DATA_DIR / "admin.json"  # folders the server may read and write (Admin tab, see sandbox.py)
CHEMUTILS_CACHE = DATA_DIR / "chemutils_cache.sqlite"  # chemutils' cache of what it looked up (vapor.py)

MAX_UPLOAD_MB = float(os.environ.get("OLFDB_MAX_UPLOAD_MB", "1024"))
STAGE_TTL_HOURS = float(os.environ.get("OLFDB_STAGE_TTL_HOURS", "48"))

# Google sheet with odor metadata (link as copied from the browser), shared as "anyone with the link can view",
# if none is set in the Admin tab
ODOR_SHEET_URL = os.environ.get("OLFDB_ODOR_SHEET_URL", "").strip()
ODOR_SHEET_CHECK_MINUTES = float(os.environ.get("OLFDB_ODOR_SHEET_CHECK_MINUTES", "10"))
ODOR_TEMPLATE = ROOT / "config" / "odor_sheet_template.csv"

# the lab NAS as mounted on the server; users' data folders are given relative to it
TENSOR_ROOT = Path(os.environ.get("OLFDB_TENSOR_ROOT", "/local/tensor"))

# the server's /etc/hostname, as docker-compose.yml mounts it into the container (whose own name is random)
HOST_HOSTNAME_FILE = Path("/etc/host_hostname")

# have the kernel (Landlock) deny the server any folder but those set in the Admin tab (and its own)
LANDLOCK = os.environ.get("OLFDB_LANDLOCK", "1").strip().lower() not in ("0", "false", "no", "off")


def ensure_dirs() -> None:
    for d in (DATA_DIR, STAGING_DIR, DATASETS_DIR, EXPORTS_DIR, ODOR_PREVIEWS_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
