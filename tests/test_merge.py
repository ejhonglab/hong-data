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
    return add_values(db, tmp_path, name, V * factor, rows, **md)


def add_values(db, tmp_path, name, values, rows, **md):
    parsed = load_file(write_csv(make_wide(rows, COLS, values=values), tmp_path, f"{name}.csv"))
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
    # exactly the tolerance apart is not close enough
    assert merge.match({(("va", -4.0),)}, (("va", -3.0),), 1) is None
    assert merge.match({(("va", -4.0),)}, (("va", -3.0),), 1.01) == (("va", -4.0),)
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


def test_power_groups():
    """Power ranges (± 0.5 mW where none was given) at most the power tolerance apart, for every two in a group."""
    ds = [{"id": i, "metadata": {"power_mw": p, "power_tolerance_mw": pm}}
          for i, (p, pm) in enumerate([(10, None), (12, None), (11.5, 0.2), (20, 2)], 1)]
    grouped = lambda tol: {g.key: g.ids for g in merge.make_groups(ds, {"power"}, tol)[0]}  # noqa: E731
    assert grouped(0) == {"10 mW": [1], "11.5–12 mW": [2, 3], "20 mW": [4]}
    assert grouped(1) == {"10–12 mW": [1, 2, 3], "20 mW": [4]}  # 10 ± 0.5 and 12 ± 0.5: 1 mW apart
    assert grouped(0.9) == {"10–11.5 mW": [1, 3], "12 mW": [2], "20 mW": [4]}


def test_scaling_per_dataset(tmp_path):
    """Each dataset is multiplied by its group's factor, fitted on the pairs of odors that
    stand for each other across datasets (here Bob's c has va @ -3.5 for Ann's va @ -3)."""
    db = Database(tmp_path / "db.sqlite3")
    rng = np.random.default_rng(7)
    data = {name: np.abs(rng.normal(size=V.shape)) + 0.2 for name in "abc"}  # not proportional to each other
    a = add_values(db, tmp_path, "a", data["a"], ROWS)
    b = add_values(db, tmp_path, "b", data["b"], ROWS, uploader_name="Bob")
    c = add_values(db, tmp_path, "c", data["c"], ROWS_35, uploader_name="Bob")

    def means(values, rows):  # mean response to each stimulus, over repeats then flies, per glomerulus
        stim = [" + ".join(sorted(o for o in r[2:4] if o != "solvent")) for r in rows]
        df = pd.DataFrame(values, columns=pd.MultiIndex.from_tuples(COLS, names=["date", "fly_num", "roi"]))
        return df.groupby(stim).mean().T.groupby(level="roi").mean().T

    target = means(data["a"], ROWS)
    pairs = {"va @ -3": "va @ -3.5", "ea @ -2": "ea @ -2", "ea @ -2 + va @ -3": "ea @ -2 + va @ -3.5"}
    mb, mc = means(data["b"], ROWS), means(data["c"], ROWS_35)
    bob = pd.DataFrame({s: (mb.loc[s] + mc.loc[pairs[s]]) / 2 for s in target.index}).T
    t, g = target.values.ravel(), bob.loc[target.index, target.columns].values.ravel()
    expected = t @ g / (g @ g)

    p = merge.plan(db, "glomeruli", "groups", ["uploader"], target="Ann")
    assert p["scales"] == {a: 1, b: pytest.approx(expected), c: pytest.approx(expected)}
    assert p["groups"][1]["n_stimuli"] == 3 and p["groups"][1]["n_points"] == 9  # 3 stimuli x 3 glomeruli
    # without the pairing (too narrow a tolerance), the factor comes from fewer stimuli of c
    assert merge.plan(db, "glomeruli", "groups", ["uploader"], target="Ann", tolerance=0.4)["scales"][c] != pytest.approx(expected)

    write_download(db, [a, b, c], "long", tmp_path / "out.zip", tmp_path / "datasets", merged=p)
    with zipfile.ZipFile(tmp_path / "out.zip") as zf:
        long = pd.read_csv(io.BytesIO(zf.read("olfdb_glomeruli_3-datasets_long/responses_long.csv")))
    for i, name in ((a, "a"), (b, "b"), (c, "c")):  # every value of each dataset, times its own factor
        got = np.sort(long[long.dataset_id == i].value.to_numpy())
        np.testing.assert_allclose(got, np.sort(data[name].ravel() * p["scales"][i]), rtol=1e-5)


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
    status, p = call(main.merge_preview, body={"coding_unit": "glomeruli", "mode": "groups", "by": ["power"], "power_tolerance": 2})
    assert status == 200 and p["power_tolerance"] == 2 and [g["key"] for g in p["groups"]] == ["10 mW"]
    res = get(main.merge_scales, {"coding_unit": "glomeruli", "mode": "groups", "by": "uploader", "target": "Bob"})
    assert res.headers["content-disposition"].endswith('olfdb_glomeruli_2-datasets_scales.csv"')
    scales = pd.read_csv(io.StringIO(res.body.decode()))
    assert scales.dataset_id.tolist() == [a, b] and scales.scale.tolist() == pytest.approx([3, 1])
    res = get(main.merge_download, {"coding_unit": "glomeruli", "format": "wide", "mode": "groups", "by": "uploader",
                                    "target": "Bob"})
    assert res.status_code == 200 and res.headers["content-disposition"].endswith('olfdb_glomeruli_2-datasets_wide.zip"')
    assert get(main.merge_download, {"coding_unit": "glomeruli", "format": "files"}).status_code == 400
