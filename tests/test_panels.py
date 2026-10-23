"""Tests of panels shown under other names (the Admin tab's). Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from app import config
from app.db import Database, rename_in_text
from app.export import write_download
from app.ingest import load_file
from tests.test_export import add, parse
from tests.test_ingest import COLS, make_wide, write_csv
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)

ROWS = [(panel, False, odor, "solvent", rep) for panel, odor in (("glomeruli_diagnostics", "va @ -3"), ("megamat", "ea @ -2"))
        for rep in (0, 1)]


def test_rename_in_text():
    renames = {"glomeruli_diagnostics": "diagnostics", "diag": "d"}
    assert rename_in_text("glomeruli_diagnostics, megamat: GH146 > GCaMP6f", renames) == "diagnostics, megamat: GH146 > GCaMP6f"
    # whole names only
    assert rename_in_text("diag_2, diag: x", renames) == "diag_2, d: x" and rename_in_text(None, renames) is None


def test_renamed_on_read(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    parsed = parse(tmp_path, ROWS, COLS)
    ds_id = add((db, tmp_path / "datasets"), parsed)
    db.set_setting("panel_renames", '{"glomeruli_diagnostics": "diagnostics"}', when="2026-10-23T00:00:00")
    # wherever a dataset's panels are read: its trials, panels, title, metadata and summary
    assert {t["panel"] for t in db.get_trials(ds_id)} == {"diagnostics", "megamat"}
    assert {t["panel"] for t in db.get_matrix_arrays(ds_id)["trials"]} == {"diagnostics", "megamat"}
    with db.connect() as con:
        con.execute("UPDATE datasets SET title = 'glomeruli_diagnostics, megamat: GH146' WHERE id = ?", (ds_id,))
    (d,) = db.list_datasets()
    assert [(p["panel"], p["source"]) for p in d["panels"]] == [("p1", "form"), ("diagnostics", "file"), ("megamat", "file")]
    assert d["title"] == "diagnostics, megamat: GH146" == db.get_dataset(ds_id)["title"]
    summary = db.get_dataset(ds_id)["summary"]
    assert summary["panel_names"] == ["diagnostics", "megamat"] and [p["panel"] for p in summary["panels"]] == ["diagnostics", "megamat"]
    # kept as uploaded, the data's panels listed by their own names
    assert db.data_panels() == {"glomeruli_diagnostics": {ds_id}, "megamat": {ds_id}}
    # downloads too
    out = tmp_path / "out.csv"
    write_download(db, [ds_id], "long", out, tmp_path / "datasets")
    assert set(pd.read_csv(out)["panel"]) == {"diagnostics", "megamat"}


def test_admin(main, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    a = add((main.db, tmp_path / "datasets"), parse(tmp_path, ROWS, COLS))
    other = [("diag", False, "va @ -3", "solvent", rep) for rep in (0, 1)] + [("diagnostics", False, "ea @ -2", "solvent", rep) for rep in (0, 1)]
    b = add((main.db, tmp_path / "datasets"), parse(tmp_path, other, COLS, name="b.csv"))
    status, s = call(main.admin_panels)
    assert status == 200 and s["renames"] == []
    assert [(p["name"], p["datasets"], p["shown"]) for p in s["panels"]] == [
        ("diag", 1, "diag"), ("diagnostics", 1, "diagnostics"), ("glomeruli_diagnostics", 1, "glomeruli_diagnostics"), ("megamat", 1, "megamat")]
    save = lambda *pairs: call(main.admin_panels_save, body={"renames": [{"from": f, "to": t} for f, t in pairs]})[1]
    # refused: half a rename, to itself, twice, to a renamed one, two panels of one dataset to one name
    assert save(("x", ""))["errors"] == ["a rename needs both names (not 'x' alone)"]
    assert save(("x", "x"))["errors"] == ["'x' is renamed to itself"]
    assert save(("x", "y"), ("x", "z"))["errors"] == ["'x' is renamed twice"]
    assert "itself renamed (to 'd')" in save(("glomeruli_diagnostics", "diag"), ("diag", "d"))["errors"][0]
    r = save(("diag", "diagnostics"))
    assert not r["ok"] and r["errors"][0].startswith(f"dataset #{b} (p1: GH146-Gal4) has panels 'diag', 'diagnostics', which would all be 'diagnostics'")
    assert call(main.admin_panels)[1]["renames"] == []  # (nothing saved)
    # saved: shown so at once, everywhere; empty rows left out; none to undo
    r = save(("glomeruli_diagnostics", " diagnostics "), ("", ""))
    assert r["ok"] and r["renames"] == [{"from": "glomeruli_diagnostics", "to": "diagnostics"}]
    assert next(p for p in r["panels"] if p["name"] == "glomeruli_diagnostics")["shown"] == "diagnostics"
    assert {t["panel"] for t in main.db.get_trials(a)} == {"diagnostics", "megamat"}
    assert save()["ok"] and {t["panel"] for t in main.db.get_trials(a)} == {"glomeruli_diagnostics", "megamat"}
    assert call(main.admin_panels_save, body={"renames": "x"})[0] == 400


def test_same_data_renamed(main, tmp_path, monkeypatch):  # noqa: F811
    # an upload of a dataset's data is its data whatever its panels are shown as
    path = write_csv(make_wide(ROWS, COLS, np.arange(len(ROWS) * len(COLS), dtype=float).reshape(len(ROWS), len(COLS))), tmp_path, "a.csv")
    parsed = load_file(path)
    ds_id = add((main.db, tmp_path / "datasets"), parsed)
    main.db.set_setting("panel_renames", '{"glomeruli_diagnostics": "diagnostics"}', when="2026-10-23T00:00:00")
    m = main.db.get_matrix_arrays(ds_id)
    assert main._same_data(parsed, m, main.db.panel_renames()) and not main._same_data(parsed, m)
