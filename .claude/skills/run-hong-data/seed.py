"""Seeds the odor table of the data folder `OLFDB_DATA_DIR` points at with a few odors (those
not in it yet), as an odor sheet would, so the Headspace calculator and Odors tab have some.

Run from the repo, with it on the path: PYTHONPATH=. ./venv/bin/python <this file>
"""
import os
import sys

if not os.environ.get("OLFDB_DATA_DIR"):  # (before app.config takes data/)
    sys.exit("set OLFDB_DATA_DIR to a scratch folder (never the real data/)")

from app import config, odormeta  # noqa: E402
from app.db import Database  # noqa: E402

HEADER = "name,abbrev,aliases,cid,cas,solvent,supplier,catalog_no,catalog_url,metacyc,lotus,inventory,notes\n"
ODORS = "ethyl acetate,ea,,8857,,pfo,,,,,,,\n" "2-heptanone,2h,,8051,,pfo,,,,,,,\n" "1-octen-3-ol,1o3ol,,18827,,pfo,,,,,,,\n"

if config.DATA_DIR == (config.ROOT / "data").resolve():
    sys.exit("OLFDB_DATA_DIR is the real data/")
rows, errors, _ = odormeta.parse_sheet(HEADER + ODORS)
assert not errors, errors
config.ensure_dirs()
db = Database(config.DB_PATH)
d = odormeta.diff(rows, db.list_odors())
if d["added"]:
    db.apply_odor_sync(added=d["added"], changed=[], deleted=[], source="run-hong-data seed", fingerprint=odormeta.fingerprint(rows),
                       details={}, when="2026-10-29T00:00:00")
print("odors:", ", ".join(f"{o['name']} (id {o['id']})" for o in db.list_odors()))
