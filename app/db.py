"""SQLite storage.

Tables
------
datasets        one row per upload (metadata, README, file names, summary numbers)
dataset_panels  normalised panel names per dataset (for "browse by panel")
units           one row per (date, fly_num, roi) column of a dataset
trials          one row per trial (panel, odor, repeat, ...) of a dataset
responses       long table: one row per cell (unit_id, trial_id, value) -- queryable
matrices        the same values as a float32 blob (units x trials) for fast serving
"""
from __future__ import annotations

import base64
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

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
    date TEXT NOT NULL,
    fly_num INTEGER NOT NULL,
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
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=60)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
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
                    wavelength_nm, source_code_url, raw_data_path, panel_names, recording_system, notes,
                    readme_md, readme_edited, data_file, data_format, data_sha256, odor_yaml_files,
                    experiment_xml_file, experiment_xml_info, n_flies, n_units, n_trials, n_odors,
                    repeats_min, repeats_max, first_fly_date, last_fly_date, roi_kind, metadata_json, summary_json)
                VALUES (?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?,?)""",
                (
                    created_at, title, metadata["uploader_name"], metadata["indicator"], metadata["driver"],
                    metadata["coding_unit"], metadata.get("genotype"), metadata.get("age_min_days"),
                    metadata.get("age_max_days"), metadata.get("power_mw"), metadata.get("power_tolerance_mw"),
                    None if metadata.get("power_measured_same_day") is None else int(metadata["power_measured_same_day"]),
                    metadata.get("wavelength_nm"), metadata.get("source_code_url"), metadata.get("raw_data_path"),
                    ", ".join(panels_form) if panels_form else None, metadata.get("recording_system"),
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

    def get_matrix(self, ds_id: int) -> Optional[dict]:
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
        return {
            "units": [dict(u) for u in units],
            "trials": [
                {
                    "panel": t["panel"],
                    "odor": t["odor"],
                    "components": json.loads(t["components_json"]),
                    "repeat": t["repeat"],
                    "extra": json.loads(t["extra_json"]),
                }
                for t in trials
            ],
            "shape": [m["n_units"], m["n_trials"]],
            "dtype": "float32",
            "values_b64": base64.b64encode(bytes(m["values_blob"])).decode("ascii"),
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

    def delete_dataset(self, ds_id: int) -> bool:
        with self.connect() as con:
            cur = con.execute("DELETE FROM datasets WHERE id = ?", (ds_id,))
            return cur.rowcount > 0
