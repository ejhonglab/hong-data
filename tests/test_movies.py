"""Tests of finding and reading a dataset's movies on tensor. Run with: python -m pytest tests/"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
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
    # one movie per recording folder, the most processed (t1's mocorr.tif, not its flipped.tif)
    assert [m["label"] for m in found] == ["2023-11-19/1/t1/mocorr.tif", "2023-11-19/1/t2/mocorr.tif"]
    assert found[0]["rois"] == str(root / "tom/analysis/2023-11-19/1/t1/RoiSet.zip") and found[0]["fly"] == FLIES[0]
    assert found[1]["rois"] == str(root / "tom/analysis/2023-11-19/1/RoiSet.zip")
    assert [m["recordings"] for m in found] == [None, None] and [m["trials"] for m in found] == [None, None]
    # a folder of one fly (no date / fly folders below it): everything in it
    assert [m["label"] for m in movies.find(root / "tom/analysis/2023-11-19/2", "mocorr", FLIES)] == ["t1/mocorr.tif"]
    (root / "tom/raw/2023-11-19/1/t1/RoiMask.raw").write_bytes(b"\0" * 96)  # ThorImage's, not a movie
    raw = movies.find(root / "tom/raw", "raw", FLIES)
    assert [(m["label"], m["format"], m["rois"]) for m in raw] == [("2023-11-19/1/t1/Image_001_001.raw", "raw", None)]
    assert [m["recording"] for m in found] == ["t1", "t2"] and raw[0]["recording"] == "t1"
    # a fly's folder with no movie al_analysis names (nor one of its recordings registered together): every TIFF in it
    other = root / "tom/other/2023-11-19/1"
    (other / "a").mkdir(parents=True)
    for name in ("a/x.tif", "y.tif"):
        tifffile.imwrite(other / name, np.zeros((2, 2), np.uint16))
    assert [m["label"] for m in movies.find(root / "tom/other", "mocorr", FLIES)] == ["2023-11-19/1/y.tif", "2023-11-19/1/a/x.tif"]


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
    assert [m["unmatched"] for m in mocorr] == [False, True]
    # all side by side, in the order they were recorded; one with no raw data after the one before it by name
    for name, t in (("t1", 2e9), ("t3", 1e9), ("anat_1", 3e9)):
        os.utime(root / "tom/raw/2023-11-19/1" / name / "Experiment.xml", (t, t))
    rows = movies.side_by_side(raw, mocorr)
    assert [(r["recording"], r["raw"], r["mocorr"], r["time"], r["matched"], r["anatomical"]) for r in rows] == [
        ("t3", ["Image_001_001.raw"], None, 1e9, False, False),
        ("t1", ["Image_001_001.raw"], ["mocorr.tif"], 2e9, True, False),
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
    assert [m["by"] for m in mocorr] == ["fly folder"] * 2

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


TRIALS = [{"start_frame": 0, "first_odor_frame": 1, "end_frame": 2, "odors": "ea @ -4.2", "name": "ea", "log10_conc": -4.2},
          {"start_frame": 3, "first_odor_frame": 3, "end_frame": 4, "name": "pfo", "log10_conc": 0}]


def concat_fly(root):
    """Fly 1 as al_analysis writes it once its recordings are registered together: each
    recording's motion-corrected movie and mocorr_concat.tif of them in a suite2p run's
    folder, linked from the fly's folder (and the recordings'), next to TIFFs that are not
    movies of its recordings."""
    fly = root / "tom/analysis/2023-11-19/1"
    s2p = fly / "suite2p_runs/0/suite2p"
    s2p.mkdir(parents=True)
    tzyx = tifffile.imread(fly / "t1/mocorr.tif")
    for name in ("t1", "t2"):
        tifffile.imwrite(s2p / f"{name}.tif", tzyx, imagej=True, metadata={"axes": "TZYX"})
    tifffile.imwrite(s2p / "mocorr_concat.tif", np.concatenate([tzyx, tzyx]), imagej=True, metadata={"axes": "TZYX"})
    (s2p / "trial_frames_and_odors.json").write_text(json.dumps(TRIALS * 2))
    (fly / "suite2p").symlink_to("suite2p_runs/0/suite2p")
    (fly / "mocorr_concat.tif").symlink_to("suite2p/mocorr_concat.tif")
    (fly / "trial_frames_and_odors.json").symlink_to("suite2p/trial_frames_and_odors.json")
    for name in ("raw_concat.tif", "trial_dff_concat.tif", "t1/trial_dff.tif"):
        tifffile.imwrite(fly / name, np.zeros((2, 2), np.uint16))
    (fly / "t1/trial_frames_and_odors.json").write_text(json.dumps(TRIALS))
    # a recording not registered with the others
    (fly / "t3").mkdir()
    tifffile.imwrite(fly / "t3/flipped.tif", tzyx[:1], imagej=True, metadata={"axes": "TZYX"})
    return fly


def test_concat(root):
    fly = concat_fly(root)
    found = movies.find(root / "tom/analysis", "mocorr", FLIES)
    # the fly's movie of its recordings registered together (named for the fly), then those not in it
    assert [(m["label"], m["recording"], m["recordings"]) for m in found] == [
        ("2023-11-19/1", None, ["t1", "t2"]), ("2023-11-19/1/t3/flipped.tif", "t3", None)]
    assert found[0]["path"] == str(fly / "mocorr_concat.tif") and found[0]["rois"] == str(fly / "RoiSet.zip")
    assert found[0]["trials"] == str(fly / "trial_frames_and_odors.json")
    assert found[0]["recording_trials"] == {"t1": str(fly / "t1/trial_frames_and_odors.json")}
    assert found[1]["rois"] == str(fly / "RoiSet.zip") and found[1]["trials"] is None
    # without ROIs in the fly's folder: those of its recordings, when they are all one file (al_analysis links them)
    (fly / "RoiSet.zip").unlink()
    (fly / "t2/RoiSet.zip").symlink_to("../t1/RoiSet.zip")
    assert movies.find(root / "tom/analysis", "mocorr", FLIES)[0]["rois"] == str(fly / "t1/RoiSet.zip")
    (fly / "t2/RoiSet.zip").unlink()
    roifile.ImagejRoi.frompoints(np.array([[1, 1], [2, 2], [1, 2]], "f4"), name="x").tofile(fly / "t2/RoiSet.zip")
    assert movies.find(root / "tom/analysis", "mocorr", FLIES)[0]["rois"] is None

    # paired with the raw data by each recording in it: t1 is in both, t2 and t3 not in the raw data
    raw = movies.find(root / "tom/raw", "raw", FLIES)
    mocorr = movies.find(root / "tom/analysis", "mocorr", FLIES)
    assert [(u["recording"], u["n_movies"]) for u in movies.unmatched(raw, mocorr)["mocorr"]] == [("t2", 1), ("t3", 1)]
    assert [m["unmatched"] for m in mocorr] == [False, True] and not raw[0]["unmatched"]
    rows = movies.side_by_side(raw, mocorr)
    assert [(r["recording"], r["mocorr"], r["matched"]) for r in rows] == [
        ("t1", ["mocorr_concat.tif"], True), ("t2", ["mocorr_concat.tif"], False), ("t3", ["flipped.tif"], False)]
    # the raw movie with the trials of its recording in the motion-corrected data
    movies.link_trials(raw, mocorr)
    assert raw[0]["trials"] == str(fly / "t1/trial_frames_and_odors.json")
    # the movie of the recordings registered together opens as any other
    assert movies.open_movie(Path(mocorr[0]["path"])).shape["t"] == 10


def raw_recording(root, name):
    d = root / "tom/raw/2023-11-19/1" / name
    d.mkdir()
    (d / "Experiment.xml").write_text(XML)
    (d / "Image_001_001.raw").write_bytes(b"")


def test_redone(root):
    # t2 redone: in both as t2_redo, t2 left out of both (its folder in the motion-corrected data too)
    for name in ("t2", "t2_redo"):
        raw_recording(root, name)
    redo = root / "tom/analysis/2023-11-19/1/t2_redo"
    redo.mkdir()
    tifffile.imwrite(redo / "mocorr.tif", np.zeros((2, 2), np.uint16))
    found = movies.find(root / "tom/raw", "raw", FLIES), movies.find(root / "tom/analysis", "mocorr", FLIES)
    assert [[m["recording"] for m in f] for f in found] == [["t1", "t2", "t2_redo"], ["t1", "t2", "t2_redo"]]
    raw, mocorr = movies.drop_redone(*found)
    assert [m["recording"] for m in raw] == [m["recording"] for m in mocorr] == ["t1", "t2_redo"]
    assert movies.unmatched(raw, mocorr) == {"raw": [], "mocorr": []}
    assert [(r["recording"], r["matched"]) for r in movies.side_by_side(raw, mocorr)] == [("t1", True), ("t2_redo", True)]
    # a _redo in only one of them still supersedes in both (here, t1 redone and not yet motion corrected)
    raw_recording(root, "t1_redo")
    raw, mocorr = movies.drop_redone(movies.find(root / "tom/raw", "raw", FLIES), movies.find(root / "tom/analysis", "mocorr", FLIES))
    assert [m["recording"] for m in raw] == ["t1_redo", "t2_redo"] and [m["recording"] for m in mocorr] == ["t2_redo"]
    assert movies.unmatched(raw, mocorr)["raw"] == [
        {"date": "2023-11-19", "fly_num": 1, "of_dataset": True, "recording": "t1_redo", "n_movies": 1}]
    # and taken out of a fly's movie of its recordings registered together (t1 and t2)
    fly = concat_fly(root)
    shutil.rmtree(root / "tom/raw/2023-11-19/1/t1_redo")
    raw, mocorr = movies.drop_redone(movies.find(root / "tom/raw", "raw", FLIES), movies.find(root / "tom/analysis", "mocorr", FLIES))
    assert [(m["label"], m["recording"], m["recordings"]) for m in mocorr] == [
        ("2023-11-19/1", None, ["t1"]), ("2023-11-19/1/t2_redo/mocorr.tif", "t2_redo", None), ("2023-11-19/1/t3/flipped.tif", "t3", None)]
    assert mocorr[0]["recording_trials"] == {"t1": str(fly / "t1/trial_frames_and_odors.json")}
    assert [r["recording"] for r in movies.side_by_side(raw, mocorr)] == ["t1", "t2_redo", "t3"]
    # left out, if all of them are redone
    raw_recording(root, "t1_redo")
    raw, mocorr = movies.drop_redone(movies.find(root / "tom/raw", "raw", FLIES), movies.find(root / "tom/analysis", "mocorr", FLIES))
    assert [m["recording"] for m in mocorr] == ["t2_redo", "t3"]


def test_trials(tmp_path):
    path = tmp_path / "trial_frames_and_odors.json"
    path.write_text(json.dumps(TRIALS))
    assert movies.read_trials(path) == [{"start": 0, "onset": 1, "end": 2, "odor": "ea @ -4.2"},
                                        {"start": 3, "onset": 3, "end": 4, "odor": "pfo @ 0"}]
    for bad, msg in (("{}", "not a list"), ('[{"start_frame": 0}]', "trial 1 has no"), ("[", "cannot read")):
        path.write_text(bad)
        with pytest.raises(movies.MovieError, match=msg):
            movies.read_trials(path)


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
    # big-endian, as hong2p writes them: memory-mapped in the file's byte order
    big = root / "big.tif"
    tifffile.imwrite(big, tzyx, imagej=True, byteorder=">", metadata={"axes": "TZYX"})
    m = movies.open_movie(big)
    assert m.data is not None and m.info()["dtype"] == "uint16"
    np.testing.assert_array_equal(m.planes(1, 2, 0, 0), tzyx[1:3, 0])
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
    assert [m["label"] for m in r["mocorr"]["movies"]] == ["1/t1/mocorr.tif", "1/t2/mocorr.tif"]
    assert len(r["raw"]["movies"]) == 1 and r["raw"]["error"] is None
    # t2 is motion corrected, but not in the raw data: said so with its folder, and of its movies
    t2 = [{"date": "2023-11-19", "fly_num": 1, "of_dataset": True, "recording": "t2", "n_movies": 1}]
    assert r["mocorr"]["compared"] and r["mocorr"]["unmatched"] == t2 and r["raw"]["unmatched"] == []
    assert [m["unmatched"] for m in r["mocorr"]["movies"]] == [False, True]
    assert r["mocorr"]["found"] == {"fly folder": 2, "path": 0, "none": 0} and r["mocorr"]["skipped"] == []
    paths = json.loads(get(main.dataset_data_paths, path_params={"ds_id": ds_id}).body)
    assert paths["mocorr"]["status"] == "ok" and paths["mocorr"]["unmatched"] == t2 and paths["raw"]["compared"]
    assert [(r["recording"], r["matched"]) for r in paths["recordings"]] == [("t1", True), ("t2", False)]
    # t2 redone (in the raw data only, so far): left out of both, its redo in the raw data only
    raw_recording(root, "t2_redo")
    r2 = json.loads(get(main.dataset_movies, path_params={"ds_id": ds_id}).body)
    assert [m["label"] for m in r2["mocorr"]["movies"]] == ["1/t1/mocorr.tif"] and r2["mocorr"]["found"]["fly folder"] == 1
    assert [u["recording"] for u in r2["raw"]["unmatched"]] == ["t2_redo"] and r2["mocorr"]["unmatched"] == []

    movie = r["mocorr"]["movies"][0]
    info = json.loads(get(main.movie_info, {"path": movie["path"]}).body)
    assert (info["t"], info["z"], info["fps"]) == (5, 3, 4.0)
    res = get(main.movie_planes, {"path": movie["path"], "t": 3, "n": 9, "z": "max"})
    planes = np.frombuffer(res.body, res.headers["x-dtype"]).reshape([int(x) for x in res.headers["x-shape"].split(",")])
    np.testing.assert_array_equal(planes, tifffile.imread(movie["path"])[3:].max(axis=1))
    assert json.loads(get(main.movie_planes, {"path": movie["path"], "z": "a"}).body)["error"].startswith("t, n, c and z")
    assert [x["name"] for x in json.loads(get(main.movie_rois, {"path": movie["rois"]}).body)["rois"]] == ["DL5", "DM2"]
    trials = root / "tom/analysis/2023-11-19/1/t1/trial_frames_and_odors.json"
    trials.write_text(json.dumps(TRIALS))
    assert [t["odor"] for t in json.loads(get(main.movie_trials, {"path": str(trials)}).body)["trials"]] == ["ea @ -4.2", "pfo @ 0"]
    trials.write_text("{}")
    assert get(main.movie_trials, {"path": str(trials)}).status_code == 422
    # only movies (and ROIs) on tensor or in a folder set in the Admin tab
    outside = tmp_path / "elsewhere.tif"
    tifffile.imwrite(outside, np.zeros((2, 2), np.uint8))
    for params, status in (({"path": str(outside)}, 403), ({"path": movie["path"][:-4] + ".csv"}, 400),
                           ({"path": "tom/analysis/x.tif"}, 400), ({"path": str(root / "nothing.tif")}, 404)):
        with pytest.raises(HTTPException) as e:
            get(main.movie_info, params)
        assert e.value.status_code == status
