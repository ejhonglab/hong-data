"""Tests of what the Compare tab shows of datasets and chemicals. Run with: python -m pytest tests/"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from app import compare
from app.db import Database
from app.ingest import load_file
from tests.test_ingest import COLS, ROWS, make_wide, write_csv
from tests.test_odormeta import HEADER, apply_all, rows_of
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)

VA = "InChI=1S/C5H10O2/c1-2-3-4-5(6)7/h2-4H2,1H3,(H,6,7)"
EA = "InChI=1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3"


def insert(db: Database, tmp_path: Path, cols=COLS, values=None) -> tuple[int, np.ndarray]:
    """A dataset of ROWS (va, ea and their mixture, twice each) by `cols`; its id and values (trials × units)."""
    values = np.arange(len(ROWS) * len(cols), dtype=float).reshape(len(ROWS), len(cols)) if values is None else values
    parsed = load_file(write_csv(make_wide(ROWS, cols, values), tmp_path))
    ds_id = db.insert_dataset(
        created_at="2026-10-21T10:00:00", title="t", readme_md="#", readme_edited=False, data_file="x.csv",
        metadata={"uploader_name": "me", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli"},
        data_format=parsed.source_format, data_sha256="0" * 64, odor_yaml_files=[], experiment_xml_file=None,
        experiment_xml_info=None, summary=parsed.summary(), units=parsed.units, trials=parsed.trials, values=parsed.values)
    return ds_id, values


def test_dataset(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    apply_all(db, rows_of(HEADER + "valeric acid,va,,7991,,,,,,,,,\nethyl acetate,ea,,8857,,,,,,,,,\n"))
    v = np.arange(len(ROWS) * len(COLS), dtype=float).reshape(len(ROWS), len(COLS))
    v[1, 0] = np.nan  # a trial missing for one ROI
    ds_id, v = insert(db, tmp_path, values=v)
    d = compare.dataset(db, ds_id, tmp_path)
    # odors as rows, each the mean of its trials; for glomeruli, of the flies with one, in the list's order
    assert d["kind"] == "glomeruli" and d["shape"] == [3, 3] and d["driver"] == "GH146" and d["coding_unit"] == "glomeruli"
    assert [(c["label"], c["n_flies"]) for c in d["cols"]] == [("DL5", 1), ("DM2", 2), ("VA2", 1)]
    va, ea = [o["id"] for o in sorted(db.list_odors(), key=lambda o: o["name"], reverse=True)]
    assert [(r["odor"], r["odor_id"], r["log10_conc"], (r["chem"] or {}).get("name")) for r in d["rows"]] == [
        ("va @ -3", va, -3.0, "valeric acid"), ("ea @ -2", ea, -2.0, "ethyl acetate"), ("ea @ -2 + va @ -3", None, None, None)]
    m = np.frombuffer(base64.b64decode(d["values_b64"]), "<f4").reshape(3, 3)
    trial_means = np.stack([np.nanmean(v[k:k + 2], axis=0) for k in (0, 2, 4)])  # odors × units (DM2, DL5, DM2, VA2)
    expected = np.stack([trial_means[:, 1], (trial_means[:, 0] + trial_means[:, 2]) / 2, trial_means[:, 3]], axis=1)
    np.testing.assert_allclose(m, expected, rtol=1e-6)

    # not glomeruli: a column per ROI of each fly
    ds_id, v = insert(db, tmp_path, cols=[("2024-01-01", 1, 0), ("2024-01-01", 1, 1), ("2024-01-02", 1, 0)])
    d = compare.dataset(db, ds_id, tmp_path)
    assert d["kind"] == "rois" and [c["label"] for c in d["cols"]] == ["2024-01-01 fly 1: 0", "2024-01-01 fly 1: 1",
                                                                       "2024-01-02 fly 1: 0"]
    m = np.frombuffer(base64.b64decode(d["values_b64"]), "<f4").reshape(3, 3)
    np.testing.assert_allclose(m[0], v[0:2].mean(axis=0))
    assert compare.dataset(db, ds_id + 1, tmp_path) is None


def test_fingerprints():
    odors = [{"id": 1, "inchi": VA}, {"id": 2, "inchi": EA}, {"id": 3, "inchi": None}, {"id": 4, "inchi": "InChI=1S/garbage"}]
    fp = compare.fingerprints(odors, 6)
    assert fp["diameter"] == 6 and fp["bits"] == compare.FP_BITS and fp["missing"] == [3, 4]
    assert set(fp["fingerprints"]) == {1, 2} and fp["fingerprints"][1] == sorted(fp["fingerprints"][1])
    assert compare.fingerprints(odors, 6) is fp  # kept
    # the larger the diameter, the more of each molecule's neighbourhoods
    sizes = [len(compare.fingerprints(odors, d)["fingerprints"][1]) for d in (0, 2, 6)]
    assert sizes[0] < sizes[1] <= sizes[2]


def test_api(main, tmp_path):  # noqa: F811
    status, r = call(main.compare_ecfp, params={"diameter": 5})
    assert status == 400 and "even" in r["error"]
    status, r = call(main.compare_ecfp, params={"diameter": 4})
    assert status == 200 and r["diameter"] == 4 and r["fingerprints"] == {}
    assert call(main.compare_dataset, path_params={"ds_id": 99})[0] == 404
    ds_id, _ = insert(main.db, tmp_path)
    status, r = call(main.compare_dataset, path_params={"ds_id": ds_id})
    assert status == 200 and r["kind"] == "glomeruli" and r["rows"][0]["odor"] == "va @ -3"
