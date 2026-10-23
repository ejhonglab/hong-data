"""SQLite storage.

Tables
------
datasets        one row per upload (metadata, README, file names, summary numbers)
dataset_panels  normalised panel names per dataset (for "browse by panel")
units           one row per (date, fly_num, roi) column of a dataset (date and fly_num null
                without flies: ROIs alone)
trials          one row per trial (panel, odor, repeat, ...) of a dataset
responses       long table: one row per cell (unit_id, trial_id, value) -- queryable
matrices        the same values as a float32 blob (units x trials) for fast serving
odors           one row per chemical: names, IDs, supplier (synced from the odor sheet)
odor_syncs      log of odor syncs (what was added / changed / deleted, by whom)
users           each user's data folders on tensor (relative to its top), by name
pubchem         what was fetched from PubChem for an odor CID (its InChI, and its vapor pressure
                through chemutils: see vapor.py)
settings        site-wide settings from the Admin tab (the odor sheet's link), by key
"""
from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np

from .ingest import drop_uninformative

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    title TEXT NOT NULL,
    uploader_name TEXT NOT NULL,
    indicator TEXT NOT NULL,
    driver TEXT NOT NULL,
    coding_unit TEXT NOT NULL,
    genotype TEXT,
    age_min_days REAL,
    age_max_days REAL,
    power_mw REAL,
    power_tolerance_mw REAL,
    power_measured_same_day INTEGER,
    wavelength_nm REAL,
    source_code_url TEXT,
    raw_data_path TEXT,
    mocorr_data_path TEXT,
    panel_names TEXT,
    recording_system TEXT,
    notes TEXT,
    readme_md TEXT NOT NULL,
    readme_edited INTEGER NOT NULL DEFAULT 0,
    data_file TEXT NOT NULL,
    data_format TEXT NOT NULL,
    data_sha256 TEXT NOT NULL,
    odor_yaml_files TEXT,
    experiment_xml_file TEXT,
    experiment_xml_info TEXT,
    n_flies INTEGER NOT NULL,
    n_units INTEGER NOT NULL,
    n_trials INTEGER NOT NULL,
    n_odors INTEGER NOT NULL,
    repeats_min INTEGER,
    repeats_max INTEGER,
    first_fly_date TEXT,
    last_fly_date TEXT,
    roi_kind TEXT,
    metadata_json TEXT NOT NULL,
    summary_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_panels (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    panel TEXT NOT NULL,
    source TEXT NOT NULL,          -- 'form' or 'file'
    PRIMARY KEY (dataset_id, panel)
);
CREATE TABLE IF NOT EXISTS units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    date TEXT,
    fly_num INTEGER,
    roi TEXT NOT NULL,
    roi_kind TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS units_dataset ON units(dataset_id, idx);
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    panel TEXT,
    odor TEXT NOT NULL,
    components_json TEXT NOT NULL,
    repeat INTEGER NOT NULL,
    extra_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trials_dataset ON trials(dataset_id, idx);
CREATE TABLE IF NOT EXISTS responses (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    unit_id INTEGER NOT NULL REFERENCES units(id) ON DELETE CASCADE,
    trial_id INTEGER NOT NULL REFERENCES trials(id) ON DELETE CASCADE,
    value REAL,
    PRIMARY KEY (unit_id, trial_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS responses_dataset ON responses(dataset_id);
CREATE TABLE IF NOT EXISTS matrices (
    dataset_id INTEGER PRIMARY KEY REFERENCES datasets(id) ON DELETE CASCADE,
    n_units INTEGER NOT NULL,
    n_trials INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    values_blob BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS odors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL UNIQUE,     -- casefolded name
    abbrev TEXT,
    aliases TEXT NOT NULL DEFAULT '[]', -- JSON list
    cid INTEGER,
    cas TEXT,
    solvent TEXT,
    supplier TEXT,
    catalog_no TEXT,
    catalog_url TEXT,
    metacyc TEXT,
    lotus TEXT,
    inventory TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS odor_syncs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    applied_at TEXT NOT NULL,
    applied_by TEXT NOT NULL,  -- no longer recorded: '' (names of older syncs not shown)
    source TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    n_added INTEGER NOT NULL,
    n_changed INTEGER NOT NULL,
    n_deleted INTEGER NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pubchem (
    cid INTEGER PRIMARY KEY,
    inchi TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reference_datasets (  -- added in the Admin tab (see reference.py)
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    path TEXT NOT NULL,                -- the file on the server (never written to)
    title TEXT NOT NULL,
    description TEXT,
    kind TEXT NOT NULL,                -- 'detect' | 'sequential' | 'diverging' | 'binary'
    chemical_axis TEXT NOT NULL,       -- 'detect' | 'rows' | 'cols' | 'none'
    value_label TEXT,
    row_label TEXT,
    col_label TEXT,
    source TEXT,
    read_grant TEXT,                   -- the path added to the folders the server may read for it, if one was
    fields TEXT                        -- JSON: which of its fields make its rows, columns and values (reference.parse)
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL UNIQUE,     -- casefolded name
    raw_dirs TEXT NOT NULL,            -- JSON lists of folders
    mocorr_dirs TEXT NOT NULL,
    odor_dirs TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS file_log (  -- files the server has read or written in its own folders (see filelog.py)
    path TEXT PRIMARY KEY,             -- as the server sees it (in Docker, in the container)
    read INTEGER NOT NULL DEFAULT 0,
    wrote INTEGER NOT NULL DEFAULT 0,
    first_at TEXT                      -- null: in the data folder when the record began
);
"""
# columns added since the first release: (table, column, type)
ADDED_COLUMNS = [("datasets", "mocorr_data_path", "TEXT"), ("reference_datasets", "fields", "TEXT"),
                 # vapor pressure, and the URL of where it is from; when looked up (with both null if none was found)
                 ("pubchem", "vapor_pressure_kpa", "REAL"), ("pubchem", "vapor_pressure_source", "TEXT"),
                 ("pubchem", "vapor_pressure_at", "TEXT")]
USER_DIR_COLUMNS = ("raw_dirs", "mocorr_dirs", "odor_dirs")
ODOR_COLUMNS = ("name", "abbrev", "aliases", "cid", "cas", "solvent", "supplier", "catalog_no", "catalog_url",
                "metacyc", "lotus", "inventory", "notes")
REFERENCE_COLUMNS = ("id", "created_at", "path", "title", "description", "kind", "chemical_axis", "value_label",
                     "row_label", "col_label", "source", "read_grant", "fields")


# tables written only along with their dataset (whose row in `datasets` tells of any change)
DATASET_TABLES = ("dataset_panels", "units", "trials", "responses", "matrices")


def content_fingerprint(con: sqlite3.Connection) -> str:
    """A fingerprint of what a database holds (to tell whether a backup is current): every
    row of its tables but the record of files read and written (`file_log`), and the
    number of rows of `DATASET_TABLES`."""
    h = hashlib.sha256()
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                                        "AND name != 'file_log' ORDER BY name")]
    for t in tables:
        h.update(f"\0{t}\0".encode())
        if t in DATASET_TABLES:
            h.update(repr(tuple(con.execute(f"SELECT count(*) FROM {t}").fetchone())).encode())
            continue
        for row in con.execute(f"SELECT * FROM {t} ORDER BY rowid"):
            h.update(repr(tuple(row)).encode())
    return h.hexdigest()


def _units_nullable(con: sqlite3.Connection) -> None:
    """The units table of a database from before files without flies, remade with its date
    and fly_num nullable (SQLite cannot drop a NOT NULL: the steps of its documentation for
    other changes to a table, the ids, so the responses of each unit, kept)."""
    create = SCHEMA[SCHEMA.index("CREATE TABLE IF NOT EXISTS units ("):]
    create = create[:create.index(");") + 2].replace("IF NOT EXISTS units (", "units_new (")
    con.commit()
    con.executescript(f"""PRAGMA foreign_keys = OFF;
        BEGIN;
        {create}
        INSERT INTO units_new (id, dataset_id, idx, date, fly_num, roi, roi_kind)
            SELECT id, dataset_id, idx, date, fly_num, roi, roi_kind FROM units;
        DROP TABLE units;
        ALTER TABLE units_new RENAME TO units;
        CREATE INDEX IF NOT EXISTS units_dataset ON units(dataset_id, idx);
        COMMIT;
        PRAGMA foreign_keys = ON;""")
    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    if bad:
        raise sqlite3.IntegrityError(f"remaking the units table left {len(bad)} rows referring to nothing")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)
            for table, column, kind in ADDED_COLUMNS:
                if column not in {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            if any(r["notnull"] for r in con.execute("PRAGMA table_info(units)") if r["name"] in ("date", "fly_num")):
                _units_nullable(con)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=60)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
        con.execute("PRAGMA temp_store = MEMORY")  # no temporary files outside the data folder (see sandbox.py)
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------------ insert
    def insert_dataset(
        self,
        *,
        created_at: str,
        title: str,
        metadata: dict,
        readme_md: str,
        readme_edited: bool,
        data_file: str,
        data_format: str,
        data_sha256: str,
        odor_yaml_files: list[str],
        experiment_xml_file: Optional[str],
        experiment_xml_info: Optional[dict],
        summary: dict,
        units: list[dict],
        trials: list[dict],
        values: np.ndarray,
    ) -> int:
        panels_form = metadata.get("panel_names") or []
        panels_file = summary.get("panel_names") or []
        with self.connect() as con:
            cur = con.execute(
                """INSERT INTO datasets (created_at, title, uploader_name, indicator, driver, coding_unit,
                    genotype, age_min_days, age_max_days, power_mw, power_tolerance_mw, power_measured_same_day,
                    wavelength_nm, source_code_url, raw_data_path, mocorr_data_path, panel_names, recording_system,
                    notes, readme_md, readme_edited, data_file, data_format, data_sha256, odor_yaml_files,
                    experiment_xml_file, experiment_xml_info, n_flies, n_units, n_trials, n_odors,
                    repeats_min, repeats_max, first_fly_date, last_fly_date, roi_kind, metadata_json, summary_json)
                VALUES (?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?,?)""",
                (
                    created_at, title, metadata["uploader_name"], metadata["indicator"], metadata["driver"],
                    metadata["coding_unit"], metadata.get("genotype"), metadata.get("age_min_days"),
                    metadata.get("age_max_days"), metadata.get("power_mw"), metadata.get("power_tolerance_mw"),
                    None if metadata.get("power_measured_same_day") is None else int(metadata["power_measured_same_day"]),
                    metadata.get("wavelength_nm"), metadata.get("source_code_url"), metadata.get("raw_data_path"),
                    metadata.get("mocorr_data_path"), ", ".join(panels_form) if panels_form else None, metadata.get("recording_system"),
                    metadata.get("notes"), readme_md, int(readme_edited), data_file, data_format, data_sha256,
                    json.dumps(odor_yaml_files), experiment_xml_file,
                    json.dumps(experiment_xml_info) if experiment_xml_info else None,
                    summary["n_flies"], summary["n_units"], summary["n_trials"], summary["n_odors"],
                    summary.get("repeats_min"), summary.get("repeats_max"), summary.get("first_date"),
                    summary.get("last_date"), summary.get("roi_kind"), json.dumps(metadata), json.dumps(summary),
                ),
            )
            ds_id = int(cur.lastrowid)
            panel_rows = [(ds_id, p, "form") for p in panels_form]
            panel_rows += [(ds_id, p, "file") for p in panels_file if p not in panels_form]
            con.executemany("INSERT OR IGNORE INTO dataset_panels VALUES (?,?,?)", panel_rows)

            unit_ids = []
            for i, u in enumerate(units):
                c = con.execute(
                    "INSERT INTO units (dataset_id, idx, date, fly_num, roi, roi_kind) VALUES (?,?,?,?,?,?)",
                    (ds_id, i, u["date"], u["fly_num"], u["roi"], u["roi_kind"]),
                )
                unit_ids.append(c.lastrowid)
            trial_ids = []
            for j, t in enumerate(trials):
                c = con.execute(
                    "INSERT INTO trials (dataset_id, idx, panel, odor, components_json, repeat, extra_json) VALUES (?,?,?,?,?,?,?)",
                    (ds_id, j, t["panel"], t["odor"], json.dumps(t["components"]), t["repeat"], json.dumps(t.get("extra") or {})),
                )
                trial_ids.append(c.lastrowid)

            vals = np.asarray(values, dtype=np.float32)
            n_units, n_trials = vals.shape
            uid = np.repeat(np.asarray(unit_ids, dtype=np.int64), n_trials)
            tid = np.tile(np.asarray(trial_ids, dtype=np.int64), n_units)
            flat = vals.reshape(-1).astype(np.float64)
            vals_obj = [None if np.isnan(v) else float(v) for v in flat]
            con.executemany(
                "INSERT INTO responses (dataset_id, unit_id, trial_id, value) VALUES (?,?,?,?)",
                zip([ds_id] * len(vals_obj), uid.tolist(), tid.tolist(), vals_obj),
            )
            con.execute(
                "INSERT INTO matrices (dataset_id, n_units, n_trials, dtype, values_blob) VALUES (?,?,?,?,?)",
                (ds_id, n_units, n_trials, "<f4", sqlite3.Binary(np.ascontiguousarray(vals.astype("<f4")).tobytes())),
            )
        return ds_id

    # ------------------------------------------------------------------- query
    def list_datasets(self) -> list[dict]:
        with self.connect() as con:
            rows = con.execute(
                """SELECT id, created_at, title, uploader_name, indicator, driver, coding_unit, panel_names,
                          recording_system, n_flies, n_units, n_trials, n_odors, repeats_min, repeats_max,
                          first_fly_date, last_fly_date, roi_kind, data_file
                   FROM datasets ORDER BY last_fly_date DESC, id DESC"""
            ).fetchall()
            panels = con.execute("SELECT dataset_id, panel, source FROM dataset_panels ORDER BY rowid").fetchall()
        by_ds: dict[int, list[dict]] = {}
        for p in panels:
            by_ds.setdefault(p["dataset_id"], []).append({"panel": p["panel"], "source": p["source"]})
        out = []
        for r in rows:
            d = dict(r)
            d["panels"] = by_ds.get(r["id"], [])
            out.append(d)
        return out

    def get_dataset(self, ds_id: int) -> Optional[dict]:
        with self.connect() as con:
            r = con.execute("SELECT * FROM datasets WHERE id = ?", (ds_id,)).fetchone()
            if r is None:
                return None
            panels = con.execute(
                "SELECT panel, source FROM dataset_panels WHERE dataset_id = ? ORDER BY rowid", (ds_id,)
            ).fetchall()
        d = dict(r)
        d["metadata"] = json.loads(d.pop("metadata_json"))
        d["summary"] = json.loads(d.pop("summary_json"))
        d["odor_yaml_files"] = json.loads(d.get("odor_yaml_files") or "[]")
        d["experiment_xml_info"] = json.loads(d["experiment_xml_info"]) if d.get("experiment_xml_info") else None
        d["panels"] = [dict(p) for p in panels]
        return d

    def get_matrix_arrays(self, ds_id: int) -> Optional[dict]:
        """Units, trials and the float32 response matrix (n_units, n_trials) of a dataset.

        Trials are without row levels that say nothing (`ingest.uninformative_levels`),
        also for datasets stored before those were dropped on upload.
        """
        with self.connect() as con:
            m = con.execute("SELECT * FROM matrices WHERE dataset_id = ?", (ds_id,)).fetchone()
            if m is None:
                return None
            units = con.execute(
                "SELECT date, fly_num, roi, roi_kind FROM units WHERE dataset_id = ? ORDER BY idx", (ds_id,)
            ).fetchall()
            trials = con.execute(
                "SELECT panel, odor, components_json, repeat, extra_json FROM trials WHERE dataset_id = ? ORDER BY idx",
                (ds_id,),
            ).fetchall()
        values = np.frombuffer(bytes(m["values_blob"]), dtype="<f4").reshape(m["n_units"], m["n_trials"])
        return {
            "units": [dict(u) for u in units],
            "trials": drop_uninformative([
                {
                    "panel": t["panel"],
                    "odor": t["odor"],
                    "components": json.loads(t["components_json"]),
                    "repeat": t["repeat"],
                    "extra": json.loads(t["extra_json"]),
                }
                for t in trials
            ]),
            "values": values,
        }

    def get_trials(self, ds_id: int) -> list[dict]:
        """A dataset's trials (panel, odor, components), in order, as in `get_matrix_arrays`."""
        with self.connect() as con:
            rows = con.execute(
                "SELECT panel, odor, components_json, extra_json FROM trials WHERE dataset_id = ? ORDER BY idx", (ds_id,)
            ).fetchall()
        trials = drop_uninformative([
            {"panel": r["panel"], "odor": r["odor"], "components": json.loads(r["components_json"]),
             "extra": json.loads(r["extra_json"])}
            for r in rows
        ])
        return [{"panel": t["panel"], "odor": t["odor"], "components": t["components"]} for t in trials]

    def odor_strings(self) -> set[str]:
        """Every odor string (one odor, as in `get_trials`' components) of every dataset."""
        with self.connect() as con:
            rows = con.execute("SELECT DISTINCT components_json FROM trials").fetchall()
        return {c for r in rows for c in json.loads(r["components_json"])}

    def get_matrix(self, ds_id: int) -> Optional[dict]:
        m = self.get_matrix_arrays(ds_id)
        if m is None:
            return None
        values = m.pop("values")
        return {
            **m,
            "shape": list(values.shape),
            "dtype": "float32",
            "values_b64": base64.b64encode(values.tobytes()).decode("ascii"),
        }

    def options(self) -> dict[str, list[str]]:
        def distinct(sql: str) -> list[str]:
            return [r[0] for r in con.execute(sql).fetchall() if r[0]]

        with self.connect() as con:
            return {
                "indicators": distinct("SELECT DISTINCT indicator FROM datasets ORDER BY indicator"),
                "drivers": distinct("SELECT DISTINCT driver FROM datasets ORDER BY driver"),
                "coding_units": distinct("SELECT DISTINCT coding_unit FROM datasets ORDER BY coding_unit"),
                "panels": distinct("SELECT DISTINCT panel FROM dataset_panels ORDER BY panel"),
                "recording_systems": distinct("SELECT DISTINCT recording_system FROM datasets ORDER BY recording_system"),
                "uploaders": distinct("SELECT DISTINCT uploader_name FROM datasets ORDER BY uploader_name"),
                "genotypes": distinct("SELECT DISTINCT genotype FROM datasets ORDER BY genotype"),
            }

    def update_data_paths(self, ds_id: int, paths: dict[str, Optional[str]]) -> dict:
        """Set some of a dataset's data paths (raw_data_path, mocorr_data_path), in their
        columns and its metadata; returns the metadata."""
        assert paths and set(paths) <= {"raw_data_path", "mocorr_data_path"}
        with self.connect() as con:
            r = con.execute("SELECT metadata_json FROM datasets WHERE id = ?", (ds_id,)).fetchone()
            metadata = {**json.loads(r["metadata_json"]), **paths}
            con.execute(f"UPDATE datasets SET {', '.join(f'{k} = ?' for k in paths)}, metadata_json = ? WHERE id = ?",
                        (*paths.values(), json.dumps(metadata), ds_id))
        return metadata

    def update_readme(self, ds_id: int, readme_md: str) -> None:
        """Replace a dataset's README (marking it as edited)."""
        with self.connect() as con:
            con.execute("UPDATE datasets SET readme_md = ?, readme_edited = 1 WHERE id = ?", (readme_md, ds_id))

    def delete_dataset(self, ds_id: int, before_commit: Optional[Callable[[], None]] = None) -> bool:
        """Delete a dataset and all its rows; False if there is no such dataset.

        Args:
            before_commit: called once the rows are deleted but before that is committed;
                nothing is deleted if it raises.
        """
        with self.connect() as con:
            cur = con.execute("DELETE FROM datasets WHERE id = ?", (ds_id,))
            if cur.rowcount and before_commit:
                before_commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------- odors
    def list_odors(self) -> list[dict]:
        """Every odor, with the InChI of its CID (None until fetched from PubChem), and its
        vapor pressure (``vapor_pressure_kpa``, ``_source`` and ``_at``: see `save_vapor_pressure`)."""
        with self.connect() as con:
            rows = con.execute(
                "SELECT odors.*, pubchem.inchi, pubchem.vapor_pressure_kpa, pubchem.vapor_pressure_source, "
                "pubchem.vapor_pressure_at FROM odors LEFT JOIN pubchem USING (cid) ORDER BY name_key"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            del d["name_key"]
            d["aliases"] = json.loads(d["aliases"])
            out.append(d)
        return out

    def apply_odor_sync(self, *, added: list[dict], changed: list[tuple[int, dict]], deleted: list[int],
                        source: str, fingerprint: str, details: dict, when: str) -> int:
        """Insert, update and delete odors (dicts with `ODOR_COLUMNS`), and log it, in one transaction.

        Returns:
            id of the odor_syncs row.
        """
        def values(r: dict) -> list:
            return [json.dumps(r[c]) if c == "aliases" else r[c] for c in ODOR_COLUMNS]

        cols = ", ".join(ODOR_COLUMNS)
        with self.connect() as con:
            for r in added:
                con.execute(
                    f"INSERT INTO odors ({cols}, name_key, created_at, updated_at) VALUES ({', '.join('?' * len(ODOR_COLUMNS))}, ?, ?, ?)",
                    (*values(r), r["name"].casefold(), when, when),
                )
            for odor_id, r in changed:
                con.execute(
                    f"UPDATE odors SET {', '.join(f'{c} = ?' for c in ODOR_COLUMNS)}, name_key = ?, updated_at = ? WHERE id = ?",
                    (*values(r), r["name"].casefold(), when, odor_id),
                )
            con.executemany("DELETE FROM odors WHERE id = ?", [(i,) for i in deleted])
            cur = con.execute(
                """INSERT INTO odor_syncs (applied_at, applied_by, source, fingerprint, n_added, n_changed, n_deleted,
                   details_json) VALUES (?,'',?,?,?,?,?,?)""",
                (when, source, fingerprint, len(added), len(changed), len(deleted), json.dumps(details)),
            )
            return int(cur.lastrowid)

    def inchis(self) -> dict[int, str]:
        """CID -> InChI, for every CID one was fetched for."""
        with self.connect() as con:
            return {r["cid"]: r["inchi"] for r in con.execute("SELECT cid, inchi FROM pubchem")}

    def save_inchis(self, inchis: dict[int, str], when: str) -> None:
        with self.connect() as con:
            con.executemany(
                "INSERT INTO pubchem (cid, inchi, fetched_at) VALUES (?,?,?) "
                "ON CONFLICT (cid) DO UPDATE SET inchi = excluded.inchi, fetched_at = excluded.fetched_at",
                [(cid, inchi, when) for cid, inchi in inchis.items()],
            )

    def save_vapor_pressure(self, cid: int, found: Optional[dict], when: str) -> None:
        """Save the vapor pressure looked up for a CID with an InChI (`vapor.look_up`).

        Args:
            found: ``{"kpa", "source"}``, or None if there is none (kept, with when it was
                looked up, so that it is not looked up again).
        """
        with self.connect() as con:
            con.execute(
                "UPDATE pubchem SET vapor_pressure_kpa = ?, vapor_pressure_source = ?, vapor_pressure_at = ? WHERE cid = ?",
                (found and found["kpa"], found and found["source"], when, cid),
            )

    def odor_syncs(self, limit: int = 50, source: Optional[str] = None) -> list[dict]:
        """Most recent odor syncs first (optionally only those from `source`)."""
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM odor_syncs WHERE ? IS NULL OR source = ? ORDER BY id DESC LIMIT ?", (source, source, limit)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            del d["applied_by"]
            d["details"] = json.loads(d.pop("details_json"))
            out.append(d)
        return out

    # ---------------------------------------------------------------- settings
    def get_setting(self, key: str) -> Optional[dict]:
        """A site-wide setting (``{'value', 'updated_at'}``), or None if it is not set."""
        with self.connect() as con:
            r = con.execute("SELECT value, updated_at FROM settings WHERE key = ?", (key,)).fetchone()
        return dict(r) if r else None

    def set_setting(self, key: str, value: Optional[str], when: str) -> None:
        """Set a site-wide setting, or clear it (`value` None)."""
        with self.connect() as con:
            if value is None:
                con.execute("DELETE FROM settings WHERE key = ?", (key,))
            else:
                con.execute("INSERT INTO settings (key, value, updated_at) VALUES (?,?,?) ON CONFLICT (key) DO UPDATE "
                            "SET value = excluded.value, updated_at = excluded.updated_at", (key, value, when))

    # ---------------------------------------------------------- reference data
    def list_references(self) -> list[dict]:
        """The reference datasets added in the Admin tab, in the order they were added."""
        with self.connect() as con:
            return [dict(r) for r in con.execute("SELECT * FROM reference_datasets ORDER BY created_at, id")]

    def add_reference(self, ref: dict) -> None:
        """Add a reference dataset (a dict with `REFERENCE_COLUMNS`)."""
        with self.connect() as con:
            con.execute(f"INSERT INTO reference_datasets ({', '.join(REFERENCE_COLUMNS)}) VALUES ({', '.join('?' * len(REFERENCE_COLUMNS))})",
                        [ref.get(c) for c in REFERENCE_COLUMNS])

    def delete_reference(self, ref_id: str) -> bool:
        with self.connect() as con:
            return con.execute("DELETE FROM reference_datasets WHERE id = ?", (ref_id,)).rowcount > 0

    # ------------------------------------------------------------------- users
    @staticmethod
    def _user(r: sqlite3.Row) -> dict:
        return {"name": r["name"], "updated_at": r["updated_at"], **{k: json.loads(r[k]) for k in USER_DIR_COLUMNS}}

    def get_user(self, name: str) -> Optional[dict]:
        """A user's folders (lists of paths relative to tensor, by `USER_DIR_COLUMNS`), by name in any case."""
        with self.connect() as con:
            r = con.execute("SELECT * FROM users WHERE name_key = ?", (name.casefold(),)).fetchone()
        return self._user(r) if r else None

    def save_user(self, name: str, dirs: dict[str, list[str]], when: str) -> dict:
        """Create or replace a user's folders; returns them as `get_user` does."""
        values = [json.dumps(dirs.get(k) or []) for k in USER_DIR_COLUMNS]
        with self.connect() as con:
            con.execute(
                f"""INSERT INTO users (name, name_key, {', '.join(USER_DIR_COLUMNS)}, updated_at) VALUES (?,?,?,?,?,?)
                    ON CONFLICT (name_key) DO UPDATE SET name = excluded.name,
                    {', '.join(f'{k} = excluded.{k}' for k in USER_DIR_COLUMNS)}, updated_at = excluded.updated_at""",
                (name, name.casefold(), *values, when),
            )
            r = con.execute("SELECT * FROM users WHERE name_key = ?", (name.casefold(),)).fetchone()
        return self._user(r)

    def fingerprint(self) -> str:
        """See `content_fingerprint`."""
        with self.connect() as con:
            return content_fingerprint(con)

    # ---------------------------------------------------------------- file log
    def file_log(self) -> list[dict]:
        """The files the server has read or written in its own folders (see filelog.py)."""
        with self.connect() as con:
            return [dict(r) for r in con.execute("SELECT path, read, wrote, first_at FROM file_log ORDER BY path")]

    def record_files(self, files: dict[str, int], when: Optional[str]) -> None:
        """Note files read (bit 1) and written (bit 2), each first `when` unless noted before."""
        with self.connect() as con:
            con.executemany(
                "INSERT INTO file_log (path, read, wrote, first_at) VALUES (?,?,?,?) ON CONFLICT (path) DO UPDATE "
                "SET read = max(read, excluded.read), wrote = max(wrote, excluded.wrote)",
                [(p, bits & 1, bits >> 1 & 1, when) for p, bits in files.items()])
