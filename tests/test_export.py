"""Download tests (concatenating stored datasets). Run with: python -m pytest tests/"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.db import Database
from app.export import ExportError, trial_keys, write_zip
from app.ingest import ParsedData, load_file
from tests.test_ingest import COLS, ROWS, make_wide, write_csv


@pytest.fixture
def store(tmp_path):
    return Database(tmp_path / "db.sqlite3"), tmp_path / "datasets"


def add(store, parsed: ParsedData, driver="GH146-Gal4") -> int:
    db, datasets_dir = store
    metadata = {"uploader_name": "me", "indicator": "GCaMP6f", "driver": driver, "coding_unit": "glomeruli",
                "panel_names": ["p1"], "notes": "two\nlines"}
    ds_id = db.insert_dataset(
        created_at="2026-01-02T03:04:05", title=f"p1: {driver}", metadata=metadata, readme_md="# readme\n",
        readme_edited=False, data_file="data.csv", data_format=parsed.source_format, data_sha256="0" * 64,
        odor_yaml_files=[], experiment_xml_file=None, experiment_xml_info=None, summary=parsed.summary(),
        units=parsed.units, trials=parsed.trials, values=parsed.values,
    )
    d = datasets_dir / str(ds_id)
    (d / "odor_yaml").mkdir(parents=True)
    (d / "data.csv").write_text("original")
    (d / "odor_yaml" / "odors.yaml").write_text("odors: []")
    (d / "README.md").write_text("# readme\n")
    (d / "metadata.json").write_text("{}")
    return ds_id


def parse(tmp_path, rows, cols, row_names=("panel", "is_pair", "odor1", "odor2", "repeat"), name="x.csv", nan_at=None):
    df = make_wide(rows, cols, row_names=row_names)
    if nan_at:
        df.iloc[nan_at] = np.nan
    return load_file(write_csv(df, tmp_path, name))


def read_member(zip_path: Path, name: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf, zf.open(name) as f:
        return pd.read_csv(f, **kw)


@pytest.fixture
def two(store, tmp_path):
    a = parse(tmp_path, ROWS, COLS, name="a.csv", nan_at=(1, 2))
    # one odor level (padded with 'solvent'), a trial shared with `a`, a new panel, and a
    # (date, fly_num, roi) that `a` also has
    b_rows = [("p1", False, "va @ -3", 0), ("p2", False, "ms @ -3", 0)]
    b = parse(tmp_path, b_rows, [("2024-01-01", 1, "DM2"), ("2024-02-01", 1, "DL5")],
              row_names=("panel", "is_pair", "odor1", "repeat"), name="b.csv")
    return (add(store, a), a), (add(store, b, driver="pebbled"), b)


def test_wide(store, tmp_path, two):
    (ia, a), (ib, b) = two
    out = tmp_path / "out.zip"
    name = write_zip(store[0], [ib, ia, ia], "wide", out, store[1])
    assert name == f"olfdb_datasets-{ia}-{ib}_wide.zip"
    stem = name[:-4]
    df = read_member(out, f"{stem}/responses_wide.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2, 3, 4])
    assert list(df.columns.names) == ["dataset_id", "date", "fly_num", "roi"]
    assert list(df.index.names) == ["panel", "is_pair", "odor1", "odor2", "repeat"]
    # a's 6 trials, then b's 'ms @ -3' (b's 'va @ -3' trial is a's first row)
    assert df.shape == (7, 6)
    assert list(df.index.get_level_values("odor2"))[-1] == "solvent"
    vals = df.to_numpy()
    np.testing.assert_allclose(vals[:6, :4], a.values.T, rtol=1e-6)
    assert np.isnan(vals[1, 2]) and np.isnan(vals[6, :4]).all()
    np.testing.assert_allclose(vals[[0, 6], 4:], b.values.T, rtol=1e-6)
    assert np.isnan(vals[1:6, 4:]).all()
    assert df.columns[4] == (str(ib), "2024-01-01", "1", "DM2")


def test_long(store, tmp_path, two):
    (ia, a), (ib, b) = two
    out = tmp_path / "out.zip"
    stem = write_zip(store[0], [ia, ib], "long", out, store[1])[:-4]
    df = read_member(out, f"{stem}/responses_long.csv")
    assert list(df.columns) == ["dataset_id", "date", "fly_num", "roi", "panel", "is_pair", "odor1", "odor2",
                                "repeat", "value"]
    assert len(df) == np.isfinite(a.values).sum() + np.isfinite(b.values).sum()
    r = df[(df.dataset_id == ib) & (df.roi == "DL5") & (df.odor1 == "ms @ -3")]
    assert r.odor2.tolist() == ["solvent"] and r.value.tolist() == pytest.approx([b.values[1, 1]], rel=1e-6)


def test_files_and_tables(store, tmp_path, two):
    (ia, _), (ib, _) = two
    out = tmp_path / "out.zip"
    stem = write_zip(store[0], [ia, ib], "files", out, store[1])[:-4]
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        readme = zf.read(f"{stem}/README.md").decode()
    for i in (ia, ib):
        assert {f"{stem}/dataset-{i}/{p}" for p in ("data.csv", "odor_yaml/odors.yaml", "README.md", "metadata.json")} <= names
    assert f"#{ib} p1: pebbled" in readme
    meta = read_member(out, f"{stem}/datasets.csv")
    assert meta.dataset_id.tolist() == [ia, ib] and meta.driver.tolist() == ["GH146-Gal4", "pebbled"]
    assert meta.notes.tolist() == ["two\nlines"] * 2 and meta.n_units.tolist() == [4, 2]


def test_unknown_and_empty(store, tmp_path, two):
    with pytest.raises(ExportError, match="not found: 99"):
        write_zip(store[0], [two[0][0], 99], "wide", tmp_path / "x.zip", store[1])
    with pytest.raises(ExportError):
        write_zip(store[0], [], "wide", tmp_path / "x.zip", store[1])
    with pytest.raises(ExportError):
        write_zip(store[0], [two[0][0]], "xlsx", tmp_path / "x.zip", store[1])


def test_extra_levels(tmp_path):
    with_pair = parse(tmp_path, [("p1", False, "va @ -3", 0), ("p1", True, "va @ -3", 0)], COLS,
                      row_names=("panel", "is_pair", "odor1", "repeat"), name="a.csv")
    no_pair = parse(tmp_path, [("va @ -3", 0), ("ea @ -2", 0)], COLS, row_names=("odor1", "repeat"), name="b.csv")
    plain = parse(tmp_path, ROWS, COLS, name="c.csv")
    levels_of = lambda *ps: trial_keys([{"trials": p.trials} for p in ps])
    # is_pair is only kept if every dataset has it ...
    assert levels_of(no_pair, plain)[0] == ["panel", "odor1", "odor2", "repeat"]
    assert levels_of(plain)[0] == ["panel", "is_pair", "odor1", "odor2", "repeat"]
    # ... or some dataset needs it to keep its trials apart
    levels, keys = levels_of(with_pair, no_pair)
    assert levels == ["panel", "is_pair", "odor1", "repeat"]
    assert keys[1] == [(None, None, "va @ -3", 0), (None, None, "ea @ -2", 0)]
