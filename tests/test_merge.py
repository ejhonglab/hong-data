"""Tests of merged downloads (every dataset of a coding unit, groups scaled to each other). Run with: python -m pytest tests/"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from app import config, merge
from app.db import Database
from app.export import write_download
from app.ingest import load_file
from tests.test_ingest import COLS, ROWS, make_wide, write_csv
from tests.test_movies import get
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)

V = np.abs(np.random.default_rng(3).normal(size=(len(ROWS), len(COLS)))) + 0.5
# va @ -3.5 instead of va @ -3: within the default tolerance of it
ROWS_35 = [tuple("va @ -3.5" if x == "va @ -3" else x for x in r) for r in ROWS]


def add(db, tmp_path, name, factor, rows=ROWS, **md):
    parsed = load_file(write_csv(make_wide(rows, COLS, values=V * factor), tmp_path, f"{name}.csv"))
    metadata = {"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli", **md}
    return db.insert_dataset(
        created_at="2026-01-02T03:04:05", title=name, metadata=metadata, readme_md="#", readme_edited=False,
        data_file=f"{name}.csv", data_format=parsed.source_format, data_sha256="0" * 64, odor_yaml_files=[],
        experiment_xml_file=None, experiment_xml_info=None, summary=parsed.summary(), units=parsed.units,
        trials=parsed.trials, values=parsed.values)


@pytest.fixture
def four(tmp_path):
    """Ann's datasets at 1x and 2x, Bob's at 0.5x (with va @ -3.5) and 4x (no power); and one of PNs."""
    db = Database(tmp_path / "db.sqlite3")
    ids = [
        add(db, tmp_path, "a", 1, recording_system="rig1", power_mw=10, power_tolerance_mw=1),
        add(db, tmp_path, "b", 2, recording_system="rig2", power_mw=10.5, power_tolerance_mw=0.5),
        add(db, tmp_path, "c", 0.5, rows=ROWS_35, uploader_name="Bob", recording_system="rig1", power_mw=20,
            power_tolerance_mw=1),
        add(db, tmp_path, "d", 4, uploader_name=" bob ", coding_unit="Glomeruli"),
    ]
    add(db, tmp_path, "e", 1, coding_unit="PNs")
    return db, ids


def groups(p):
    return {g["key"]: (g["ids"], g["scale"]) for g in p["groups"]}


def test_match():
    avail = {(("va", -3.5),), (("va", -4.0),), (("ea", -2.0), ("va", -3.0))}
    assert merge.match(avail, (("va", -3.0),), 1) == (("va", -3.5),)
    assert merge.match(avail | {(("va", -3.0),)}, (("va", -3.0),), 1) == (("va", -3.0),)  # the exact one first
    assert merge.match(avail, (("va", -3.0),), 0.4) is None and merge.match(avail, (("ea", -3.0),), 1) is None
    assert merge.match(avail, (("ea", -2.0), ("va", -2.5)), 1) == (("ea", -2.0), ("va", -3.0))
    assert merge.odor_key(["va @ -3", "solvent"]) == (("va", -3.0),) and merge.odor_key(["solvent"]) is None


def test_plan(four):
    db, (a, b, c, d) = four
    units = merge.coding_units(db)
    assert [(u["coding_unit"], u["ids"]) for u in units] == [("glomeruli", [a, b, c, d]), ("PNs", [5])]
    p = merge.plan(db, "GLOMERULI")
    assert p["scales"] == {a: 1, b: 1, c: 1, d: 1} and p["groups"] == [] and p["coding_unit"] == "glomeruli"

    # by experimenter: Ann 1.5x on average, Bob 2.25x (c's va @ -3.5 stands for va @ -3)
    p = merge.plan(db, "glomeruli", "groups", ["uploader"])
    assert p["target"] == "average" and p["by_roi"]
    assert groups(p) == {"Ann": ([a, b], pytest.approx(1.875 / 1.5)), "Bob": ([c, d], pytest.approx(1.875 / 2.25))}
    assert p["scales"][c] == pytest.approx(1.875 / 2.25) and p["groups"][0]["r"] == pytest.approx(1)
    p = merge.plan(db, "glomeruli", "groups", ["uploader"], target="Ann")
    assert groups(p) == {"Ann": ([a, b], 1), "Bob": ([c, d], pytest.approx(1.5 / 2.25))} and p["target"] == "Ann"
    assert p["groups"][0]["note"] == "the target" and p["groups"][1]["n_stimuli"] == 3
    # a narrower tolerance: c's va @ -3.5 no longer stands for va @ -3
    s = groups(merge.plan(db, "glomeruli", "groups", ["uploader"], target="Ann", tolerance=0.4))["Bob"][1]
    assert 1.5 / 4 < s < 1.5 / 2.25 - 0.01
    assert merge.plan(db, "glomeruli", "groups", ["uploader"], target="nobody")["target"] == "average"

    # by power: d has none (left as it is); a and b are within each other's ±
    p = merge.plan(db, "glomeruli", "groups", ["power"])
    assert groups(p) == {"10–10.5 mW": ([a, b], pytest.approx(2 / 3)), "20 mW": ([c], pytest.approx(2))}
    assert p["unscaled"] == [{"id": d, "reason": "no power measured: left as it is"}] and p["scales"][d] == 1
    p = merge.plan(db, "glomeruli", "groups", ["uploader", "system"])
    assert list(groups(p)) == ["Ann · rig1", "Ann · rig2", "Bob · rig1", "bob · recording system not given"]
    p = merge.plan(db, "glomeruli", "groups", [])
    assert groups(p) == {"all datasets": ([a, b, c, d], 1)} and "only group" in p["groups"][0]["note"]

    for kw in ({"mode": "x"}, {"mode": "groups", "by": ["driver"]}, {"tolerance": -1}):
        with pytest.raises(merge.MergeError):
            merge.plan(db, "glomeruli", **kw)
    with pytest.raises(merge.MergeError, match="no datasets"):
        merge.plan(db, "KCs")


def test_download(four, tmp_path):
    db, (a, b, c, d) = four
    p = merge.plan(db, "glomeruli", "groups", ["uploader"], target="Ann")
    name, _ = write_download(db, [a, b, c, d], "long", tmp_path / "out.zip", tmp_path / "datasets", merged=p)
    assert name == "olfdb_glomeruli_4-datasets_long.zip"
    with zipfile.ZipFile(tmp_path / "out.zip") as zf:
        long = pd.read_csv(io.BytesIO(zf.read("olfdb_glomeruli_4-datasets_long/responses_long.csv")))
        scaling = pd.read_csv(io.BytesIO(zf.read("olfdb_glomeruli_4-datasets_long/scaling.csv")))
        readme = zf.read("olfdb_glomeruli_4-datasets_long/README.md").decode()
    sums = long.groupby("dataset_id").value.sum()
    assert sums[a] == pytest.approx(V.sum(), rel=1e-5) and sums[b] == pytest.approx(2 * V.sum(), rel=1e-5)
    assert sums[c] == pytest.approx(0.5 * V.sum() * 1.5 / 2.25, rel=1e-5)
    assert scaling.group.tolist() == ["Ann", "Ann", "Bob", "Bob"] and scaling.scale[2] == pytest.approx(1.5 / 2.25)
    assert "## Scaling" in readme and "the group Ann" in readme
    # one dataset is a zip too, when merged
    p = merge.plan(db, "PNs")
    assert write_download(db, [5], "wide", tmp_path / "pn.zip", tmp_path / "datasets", merged=p)[0] == "olfdb_pns_1-dataset_wide.zip"


def test_api(main, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "EXPORTS_DIR", tmp_path / "exports")
    config.EXPORTS_DIR.mkdir()
    a = add(main.db, tmp_path, "a", 1, power_mw=10)
    b = add(main.db, tmp_path, "b", 3, uploader_name="Bob", power_mw=10)
    assert call(main.merge_units)[1] == {"coding_units": [{"coding_unit": "glomeruli", "ids": [a, b]}]}
    status, p = call(main.merge_preview, body={"coding_unit": "glomeruli", "mode": "groups", "by": ["uploader"], "target": "Bob"})
    assert status == 200 and {g["key"]: g["scale"] for g in p["groups"]} == {"Ann": pytest.approx(3), "Bob": 1}
    assert call(main.merge_preview, body={"coding_unit": "glomeruli", "tolerance": "x"})[0] == 422
    res = get(main.merge_download, {"coding_unit": "glomeruli", "format": "wide", "mode": "groups", "by": "uploader",
                                    "target": "Bob"})
    assert res.status_code == 200 and res.headers["content-disposition"].endswith('olfdb_glomeruli_2-datasets_wide.zip"')
    assert get(main.merge_download, {"coding_unit": "glomeruli", "format": "files"}).status_code == 400
