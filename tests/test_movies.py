"""Tests of finding and reading a dataset's movies on tensor. Run with: python -m pytest tests/"""
from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlencode

import numpy as np
import pytest
import roifile
import tifffile
from starlette.exceptions import HTTPException
from starlette.requests import Request

from app import movies, thorimage
from app.ingest import load_file
from tests.test_ingest import COLS, ROWS, make_wide, write_csv
from tests.test_tensor import call, main  # noqa: F401 (fixtures)

XML = """<ThorImageExperiment><LSM pixelX="8" pixelY="6" frameRate="30" averageMode="0" averageNum="1"/>
<ZStage name="ThorZPiezo" steps="3" stepSizeUM="10"/><Streaming enable="1" zFastEnable="1" flybackFrames="1"/>
<PMT enableA="1" gainA="50" enableB="0" gainB="0"/></ThorImageExperiment>"""


@pytest.fixture
def root(tmp_path):
    """A tensor with motion-corrected movies of fly 1 of 2023-11-19 (as al_analysis writes them) and its raw data."""
    r = tmp_path / "tensor"
    rng = np.random.default_rng(0)
    rec = r / "tom/analysis/2023-11-19/1/t1"
    rec.mkdir(parents=True)
    tzyx = rng.integers(0, 4000, (5, 3, 6, 8), dtype=np.uint16)
    tifffile.imwrite(rec / "mocorr.tif", tzyx, imagej=True, metadata={"axes": "TZYX", "finterval": 0.25})
    tifffile.imwrite(rec / "flipped.tif", tzyx[:, :, :, ::-1], imagej=True, metadata={"axes": "TZYX"})
    poly = roifile.ImagejRoi.frompoints(np.array([[1, 1], [6, 1], [4, 5]], "f4"), name="DM2")
    poly.z_position = 2
    rect = roifile.ImagejRoi(roitype=roifile.ROI_TYPE.RECT, left=0, top=0, right=3, bottom=2, name="DL5", position=3)
    rect.tofile(rec / "RoiSet.zip", name="DL5.roi")
    poly.tofile(rec / "RoiSet.zip", name="DM2.roi")
    # a second recording, compressed (read page by page), with the ROIs of the fly's folder
    rec2 = r / "tom/analysis/2023-11-19/1/t2"
    rec2.mkdir()
    tifffile.imwrite(rec2 / "mocorr.tif", tzyx[:2], compression="zlib", photometric="minisblack", metadata={"axes": "TZYX"})
    poly.tofile(rec2.parent / "RoiSet.zip", name="DM2.roi")
    (r / "tom/analysis/2023-11-19/2/t1").mkdir(parents=True)  # another fly, not in the dataset
    tifffile.imwrite(r / "tom/analysis/2023-11-19/2/t1/mocorr.tif", tzyx[:1], photometric="minisblack")
    raw = r / "tom/raw/2023-11-19/1/t1"
    raw.mkdir(parents=True)
    (raw / "Experiment.xml").write_text(XML)
    rng.integers(0, 4000, (4, 3 + 1, 1, 6, 8), dtype="<u2").tofile(raw / "Image_001_001.raw")
    return r


FLIES = [{"date": "2023-11-19", "fly_num": 1}]


def test_raw_layout():
    assert thorimage.raw_layout(XML) == {"x": 8, "y": 6, "z": 3, "flyback": 1, "c": 1, "fps": 30.0}
    xml = XML.replace('zFastEnable="1"', 'zFastEnable="0"').replace('averageMode="0" averageNum="1"', 'averageMode="1" averageNum="2"')
    assert thorimage.raw_layout(xml.replace('enableB="0"', 'enableB="1"')) == {"x": 8, "y": 6, "z": 3, "flyback": 0, "c": 2, "fps": 15.0}
    with pytest.raises(thorimage.ExperimentXmlError):
        thorimage.raw_layout("<ThorImageExperiment/>")


def test_find(root):
    found = movies.find(root / "tom/analysis", "mocorr", FLIES)
    assert [m["label"] for m in found] == ["2023-11-19/1/t1/mocorr.tif", "2023-11-19/1/t1/flipped.tif", "2023-11-19/1/t2/mocorr.tif"]
    assert found[0]["rois"] == str(root / "tom/analysis/2023-11-19/1/t1/RoiSet.zip") and found[0]["fly"] == FLIES[0]
    assert found[2]["rois"] == str(root / "tom/analysis/2023-11-19/1/RoiSet.zip")
    # a folder of one fly (no date / fly folders below it): everything in it
    assert [m["label"] for m in movies.find(root / "tom/analysis/2023-11-19/2", "mocorr", FLIES)] == ["t1/mocorr.tif"]
    raw = movies.find(root / "tom/raw", "raw", FLIES)
    assert [(m["label"], m["format"], m["rois"]) for m in raw] == [("2023-11-19/1/t1/Image_001_001.raw", "raw", None)]
    assert [m["recording"] for m in found] == ["t1", "t1", "t2"] and raw[0]["recording"] == "t1"


def test_unmatched(root):
    # recordings in the raw data and not motion corrected, and the other way around (not anatomical ones)
    for name in ("t3", "anat_1"):
        (root / "tom/raw/2023-11-19/1" / name).mkdir()
        (root / "tom/raw/2023-11-19/1" / name / "Experiment.xml").write_text(XML)
        (root / "tom/raw/2023-11-19/1" / name / "Image_001_001.raw").write_bytes(b"")
    raw, mocorr = movies.find(root / "tom/raw", "raw", FLIES), movies.find(root / "tom/analysis", "mocorr", FLIES)
    assert movies.unmatched(raw, mocorr) == {
        "raw": [{"date": "2023-11-19", "fly_num": 1, "of_dataset": True, "recording": "t3", "n_movies": 1}],
        "mocorr": [{"date": "2023-11-19", "fly_num": 1, "of_dataset": True, "recording": "t2", "n_movies": 1}]}
    assert [(m["recording"], m["unmatched"]) for m in raw] == [("anat_1", False), ("t1", False), ("t3", True)]
    assert [m["unmatched"] for m in mocorr] == [False, False, True]
    # all side by side, in the order they were recorded; one with no raw data after the one before it by name
    for name, t in (("t1", 2e9), ("t3", 1e9), ("anat_1", 3e9)):
        os.utime(root / "tom/raw/2023-11-19/1" / name / "Experiment.xml", (t, t))
    rows = movies.side_by_side(raw, mocorr)
    assert [(r["recording"], r["raw"], r["mocorr"], r["time"], r["matched"], r["anatomical"]) for r in rows] == [
        ("t3", ["Image_001_001.raw"], None, 1e9, False, False),
        ("t1", ["Image_001_001.raw"], ["mocorr.tif", "flipped.tif"], 2e9, True, False),
        ("t2", None, ["mocorr.tif"], None, False, False),
        ("anat_1", ["Image_001_001.raw"], None, 3e9, False, True)]
    assert {(r["date"], r["fly_num"]) for r in rows} == {("2023-11-19", 1)} and rows[0]["recorded"] is None
    # by when their Experiment.xml says they were recorded, when they all do
    for name, when in (("t1", "11/19/2023 10:00:00"), ("t3", "11/19/2023 11:00:00"), ("anat_1", "11/19/2023 9:30:00")):
        xml = root / "tom/raw/2023-11-19/1" / name / "Experiment.xml"
        t = os.path.getmtime(xml)
        xml.write_text(XML.replace("<LSM", f'<Date date="{when}" uTime="0"/><LSM'))
        os.utime(xml, (t, t))
    rows = movies.side_by_side(raw, mocorr)
    assert [(r["recording"], r["recorded"]) for r in rows] == [
        ("anat_1", "2023-11-19 09:30:00"), ("t1", "2023-11-19 10:00:00"), ("t2", None), ("t3", "2023-11-19 11:00:00")]
    assert [m["by"] for m in mocorr] == ["fly folder"] * 3

    # a folder with no fly folders of the dataset in it (a single recording's, say): its movies
    # matched to flies by the folders in their paths
    rec = movies.find(root / "tom/analysis/2023-11-19/1/t2", "mocorr", FLIES)
    assert [(m["label"], m["fly"], m["by"], m["recording"]) for m in rec] == [("mocorr.tif", FLIES[0], "path", "t2")]
    assert movies.unmatched(raw, rec)["mocorr"] == [{"date": "2023-11-19", "fly_num": 1, "of_dataset": True, "recording": "t2", "n_movies": 1}]
    rec = movies.find(root / "tom/analysis/2023-11-19/1/t1", "mocorr", FLIES)
    assert movies.unmatched(raw, rec)["mocorr"] == [] and [r["recording"] for r in movies.side_by_side(raw, rec)
                                                         if r["matched"]] == ["t1"]
    # and those of a fly not the dataset's, listed all the same: by the fly their folders name, after the dataset's
    other = movies.find(root / "tom/analysis/2023-11-19/2", "mocorr", FLIES)
    assert other[0]["fly"] is None and other[0]["by"] is None
    assert movies.unmatched(raw, other)["mocorr"] == [
        {"date": "2023-11-19", "fly_num": 2, "of_dataset": False, "recording": "t1", "n_movies": 1}]
    assert other[0]["unmatched"]
    rows = movies.side_by_side(raw, other)
    assert [(r["date"], r["fly_num"], r["of_dataset"], r["recording"], r["matched"]) for r in rows] == [
        ("2023-11-19", 1, True, "anat_1", False), ("2023-11-19", 1, True, "t1", False), ("2023-11-19", 1, True, "t3", False),
        ("2023-11-19", 2, False, "t1", False)]
    assert rows[-1]["mocorr"] == ["mocorr.tif"] and not rows[-1]["anatomical"]
    # and those in no fly's folder at all, by the folder they are in, last
    loose = root / "tom/loose/a"
    loose.mkdir(parents=True)
    tifffile.imwrite(loose / "x.tif", np.zeros((2, 2), np.uint16))
    rows = movies.side_by_side(raw, movies.find(root / "tom/loose", "mocorr", FLIES))
    assert [(r["date"], r["of_dataset"], r["recording"]) for r in rows][-1] == (None, False, "a")


def test_other_flies_left_out(root):
    # a data folder with date / fly folders, none of the dataset's flies': theirs left out, others kept
    loose = root / "tom/analysis/notes"
    loose.mkdir()
    tifffile.imwrite(loose / "overview.tif", np.zeros((2, 2), np.uint16))
    skipped: set = set()
    found = movies.find(root / "tom/analysis", "mocorr", [{"date": "2023-11-19", "fly_num": 3}], skipped)
    assert [m["label"] for m in found] == ["notes/overview.tif"] and found[0]["fly"] is None
    assert skipped == {("2023-11-19", 1), ("2023-11-19", 2)}
    # one of them in it: just its folder, as before, nothing left out to say
    skipped = set()
    assert {m["fly"]["fly_num"] for m in movies.find(root / "tom/analysis", "mocorr", FLIES, skipped)} == {1} and not skipped
    # the folder given being another fly's own: its movies are kept (they are what was given)
    assert [m["label"] for m in movies.find(root / "tom/analysis/2023-11-19/2", "mocorr", FLIES)] == ["t1/mocorr.tif"]


def test_tiff(root):
    tzyx = tifffile.imread(root / "tom/analysis/2023-11-19/1/t1/mocorr.tif")
    for name, data in (("t1/mocorr.tif", tzyx), ("t2/mocorr.tif", tzyx[:2])):  # in one piece, and compressed
        m = movies.open_movie(root / "tom/analysis/2023-11-19/1" / name)
        assert (m.data is None) == name.startswith("t2")
        assert {k: m.info()[k] for k in "tzcyx"} == {"t": len(data), "z": 3, "c": 1, "y": 6, "x": 8} and m.info()["dtype"] == "uint16"
        np.testing.assert_array_equal(m.planes(1, 5, 2, 0), data[1:, 2])
        np.testing.assert_array_equal(m.planes(0, 2, None, 0), data[:2].max(axis=1))
        with pytest.raises(movies.MovieError):
            m.planes(0, 1, 3, 0)
    assert movies.open_movie(root / "tom/analysis/2023-11-19/1/t1/mocorr.tif").fps == 4.0
    # a movie of one plane
    one = root / "tyx.tif"
    tifffile.imwrite(one, tzyx[:, 0], imagej=True, metadata={"axes": "TYX"})
    m = movies.open_movie(one)
    assert m.shape == {"t": 5, "z": 1, "c": 1, "y": 6, "x": 8}
    np.testing.assert_array_equal(m.planes(3, 1, None, 0), tzyx[3:4, 0])


def test_raw(root):
    path = root / "tom/raw/2023-11-19/1/t1/Image_001_001.raw"
    data = np.fromfile(path, "<u2").reshape(4, 4, 1, 6, 8)
    m = movies.open_movie(path)
    assert m.info() == {"path": str(path), "format": "raw", "t": 4, "z": 3, "c": 1, "y": 6, "x": 8, "dtype": "uint16",
                        "fps": 7.5, "warnings": []}
    np.testing.assert_array_equal(m.planes(2, 2, 1, 0), data[2:, 1, 0])
    np.testing.assert_array_equal(m.planes(0, 4, None, 0), data[:, :3, 0].max(axis=1))  # without the flyback frame
    with path.open("ab") as f:
        f.write(b"\0" * 10)
    assert "longer than 4 whole volumes" in movies.open_movie(path).warnings[0]


def test_rois(root):
    rois = movies.read_rois(root / "tom/analysis/2023-11-19/1/t1/RoiSet.zip")
    assert [(r["name"], r["type"], r["z"], r["position"], r["closed"]) for r in rois] == [
        ("DL5", "rect", None, 2, True), ("DM2", "freehand", 1, None, True)]
    assert rois[0]["paths"] == [[[0, 0], [0, 2], [3, 2], [3, 0], [0, 0]]] and rois[1]["paths"][0][:2] == [[1, 1], [6, 1]]


def get(endpoint, params=None, path_params=None):
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": urlencode(params or {}).encode(),
             "path_params": path_params or {}}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return asyncio.run(endpoint(Request(scope, receive)))


def test_api(main, root, tmp_path, monkeypatch):  # noqa: F811
    from app import config

    monkeypatch.setattr(config, "TENSOR_ROOT", root)
    parsed = load_file(write_csv(make_wide(ROWS, COLS), tmp_path))
    flies = [{"date": "2023-11-19", "fly_num": 1}]
    ds_id = main.db.insert_dataset(
        created_at="2026-01-02T03:04:05", title="t", readme_md="#", readme_edited=False, data_file="x.csv",
        metadata={"uploader_name": "me", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli",
                  "raw_data_path": "tom/raw", "mocorr_data_path": "tom/analysis/2023-11-19"},
        data_format=parsed.source_format, data_sha256="0" * 64, odor_yaml_files=[], experiment_xml_file=None,
        experiment_xml_info=None, summary={**parsed.summary(), "flies": flies}, units=parsed.units, trials=parsed.trials,
        values=parsed.values)
    r = json.loads(get(main.dataset_movies, path_params={"ds_id": ds_id}).body)
    assert [m["label"] for m in r["mocorr"]["movies"]] == ["1/t1/mocorr.tif", "1/t1/flipped.tif", "1/t2/mocorr.tif"]
    assert len(r["raw"]["movies"]) == 1 and r["raw"]["error"] is None
    # t2 is motion corrected, but not in the raw data: said so with its folder, and of its movies
    t2 = [{"date": "2023-11-19", "fly_num": 1, "of_dataset": True, "recording": "t2", "n_movies": 1}]
    assert r["mocorr"]["compared"] and r["mocorr"]["unmatched"] == t2 and r["raw"]["unmatched"] == []
    assert [m["unmatched"] for m in r["mocorr"]["movies"]] == [False, False, True]
    assert r["mocorr"]["found"] == {"fly folder": 3, "path": 0, "none": 0} and r["mocorr"]["skipped"] == []
    paths = json.loads(get(main.dataset_data_paths, path_params={"ds_id": ds_id}).body)
    assert paths["mocorr"]["status"] == "ok" and paths["mocorr"]["unmatched"] == t2 and paths["raw"]["compared"]
    assert [(r["recording"], r["matched"]) for r in paths["recordings"]] == [("t1", True), ("t2", False)]

    movie = r["mocorr"]["movies"][0]
    info = json.loads(get(main.movie_info, {"path": movie["path"]}).body)
    assert (info["t"], info["z"], info["fps"]) == (5, 3, 4.0)
    res = get(main.movie_planes, {"path": movie["path"], "t": 3, "n": 9, "z": "max"})
    planes = np.frombuffer(res.body, res.headers["x-dtype"]).reshape([int(x) for x in res.headers["x-shape"].split(",")])
    np.testing.assert_array_equal(planes, tifffile.imread(movie["path"])[3:].max(axis=1))
    assert json.loads(get(main.movie_planes, {"path": movie["path"], "z": "a"}).body)["error"].startswith("t, n, c and z")
    assert [x["name"] for x in json.loads(get(main.movie_rois, {"path": movie["rois"]}).body)["rois"]] == ["DL5", "DM2"]
    # only movies (and ROIs) on tensor or in a folder set in the Admin tab
    outside = tmp_path / "elsewhere.tif"
    tifffile.imwrite(outside, np.zeros((2, 2), np.uint8))
    for params, status in (({"path": str(outside)}, 403), ({"path": movie["path"][:-4] + ".csv"}, 400),
                           ({"path": "tom/analysis/x.tif"}, 400), ({"path": str(root / "nothing.tif")}, 404)):
        with pytest.raises(HTTPException) as e:
            get(main.movie_info, params)
        assert e.value.status_code == status
