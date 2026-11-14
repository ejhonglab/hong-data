"""Tests of the ROI training export (roi_training.py). Run with:
OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import asyncio
import io
import json
import shutil
import zipfile

import numpy as np
import pytest
import roifile
import tifffile

from app import config, odormeta, roi_training
from app.ingest import load_file
from tests.test_ingest import make_wide, write_csv
from tests.test_tensor import XML, call, main, root  # noqa: F401 (fixtures)

DIAG = [("glomeruli_diagnostics", False, o, "solvent", r) for o in ("va @ -3", "ea @ -2") for r in (0, 1)]
OTHER = [("megamat", False, "2h @ -6", "solvent", r) for r in (0, 1)]
COLS = [("2024-01-01", 1, "DM2"), ("2024-01-02", 1, "DM2"), ("2024-01-02", 2, "DM2")]
GRID = (2, 4, 5)


def trials(*odors):
    """A recording's trial_frames_and_odors.json, as al_analysis writes it (each odor twice)."""
    names = {"va": "valeric acid", "ea": "ethyl acetate", "2h": "2-heptanone"}
    return json.dumps([{"start_frame": 0, "first_odor_frame": 1, "end_frame": 2, "odors": f"{names[a]} @ {c}", "name": names[a],
                        "abbrev": a, "log10_conc": c} for a, c in odors for _ in range(2)])


# of a fly's ROIs: glomeruli (DM2 on two planes), tentative ones (DL5's only so; DA1's also not), several that could not
# be told apart, and ROIs not trained on (numbered, or ending in ? or +)
ROI_NAMES = ["DM2", "DM2", "DL5_t0", "DA1_t0", "DA1", "VA1d/VA1v", "12", "DC1?", "DC1+"]


def roiset(path, names=ROI_NAMES) -> bytes:
    """The bytes of a RoiSet.zip of ROIs of these names, as ImageJ saves them (written at `path`)."""
    for i, n in enumerate(names):
        roifile.ImagejRoi.frompoints(np.array([[1, 1], [3, 1], [2, 3]], "f4"), name=n).tofile(path, name=f"{i:04d}.roi")
    return path.read_bytes()


def recording(fly, name, *odors, roiset=None):
    d = fly / name
    d.mkdir(parents=True)
    tifffile.imwrite(d / "trialmean_dff.tif", np.ones((len(odors), *GRID), np.float32), imagej=True, metadata={"axes": "TZYX"})
    (d / "trial_frames_and_odors.json").write_text(trials(*odors))
    if roiset is not None:
        (d / "RoiSet.zip").write_bytes(roiset)


@pytest.fixture
def export(main, root, tmp_path, monkeypatch):  # noqa: F811
    """Two glomeruli datasets with a diagnostic panel (of the same flies, one uploaded by
    another) and one of KCs, and on tensor: the first fly as al_analysis lays it out (a
    diagnostic recording with its ROIs, another, its movie of them registered together, and in
    its raw data their Experiment.xml and an anatomical stack); the second's folder without
    ROIs; none of the third."""
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "ROI_TRAINING_DIR", tmp_path / "roi_training")
    rows, _, _ = odormeta.parse_sheet("name,abbrev\nvaleric acid,va\nethyl acetate,ea\n2-heptanone,2h\n")
    main.db.apply_odor_sync(added=odormeta.diff(rows, [])["added"], changed=[], deleted=[], source="test",
                            fingerprint="0" * 64, details={}, when="2026-11-04T00:00:00")
    parsed = load_file(write_csv(make_wide(DIAG + OTHER, COLS), tmp_path))
    ids = []
    for i, (who, unit) in enumerate([("Ann", "glomeruli"), ("Bob", "glomeruli"), ("Ann", "KCs")]):
        ids.append(main.db.insert_dataset(
            created_at="2026-01-02T03:04:05", title=f"t{i}", readme_md="#", readme_edited=False, data_file="x.csv",
            metadata={"uploader_name": who, "indicator": "GCaMP6f", "driver": "pebbled", "coding_unit": unit,
                      "raw_data_path": "tom/raw", "mocorr_data_path": "tom/analysis"},
            data_format=parsed.source_format, data_sha256=str(i) * 64, odor_yaml_files=[], experiment_xml_file=None,
            experiment_xml_info=None, summary=parsed.summary(), units=parsed.units, trials=parsed.trials, values=parsed.values))
    fly = root / "tom/analysis/2024-01-01/1"
    recording(fly, "diagnostics1", ("va", -3), ("ea", -2), roiset=roiset(tmp_path / "RoiSet.zip"))
    recording(fly, "megamat1", ("2h", -6))
    movie = np.arange(3 * np.prod(GRID), dtype=np.uint16).reshape(3, *GRID)
    tifffile.imwrite(fly / "mocorr_concat.tif", movie, imagej=True, metadata={"axes": "TZYX"})
    raw = root / "tom/raw/2024-01-01/1"
    for name in ("diagnostics1", "megamat1", "anat"):
        (raw / name).mkdir(parents=True)
        (raw / name / "Experiment.xml").write_text(XML.format(notes=""))
    tifffile.imwrite(raw / "anat/ChanA_0001_0001_0001_0001.tif", np.zeros((4, 4), np.uint16))
    (raw / "anat/ChanA_Preview.tif").write_bytes(b"not exported")
    recording(root / "tom/analysis/2024-01-02/1", "diagnostics1", ("va", -3), ("ea", -2))
    return {"ids": ids, "fly": fly, "movie": movie}


def test_inventory(main, root, export):  # noqa: F811
    status, r = call(main.roi_training_flies, params={})
    assert status == 200 and len(r["flies"]) == 1
    f = r["flies"][0]
    a, b, kc = export["ids"]
    # one fly of both datasets; its diagnostic recording only; its average image to be computed
    assert [d["id"] for d in f["datasets"]] == [a, b] and (f["date"], f["fly_num"]) == ("2024-01-01", 1)
    assert f["dir"] == "tom/analysis/2024-01-01/1" and f["roiset"] == "tom/analysis/2024-01-01/1/diagnostics1/RoiSet.zip"
    assert f["recordings"] == [{"name": "diagnostics1", "n_odors": 2, "xml": True}] and f["shape"] == list(GRID)
    assert f["average"] == "computed" and [s["name"] for s in f["stacks"]] == ["anat"]
    assert r["stack_bytes"] == f["stacks"][0]["bytes"] > 0
    # its glomeruli, as al-diag-seg trains on them, and its panels but diagnostic ones (of both datasets)
    assert f["glomeruli"] == ["DA1", "DL5", "DM2", "VA1d/VA1v"] and f["tentative"] == ["DL5"] and f["roi_error"] is None
    assert f["panels"] == ["megamat"]
    # the others left out, with why; the KC dataset is not one to look at
    why = {(s["dataset"]["id"], s["date"], s["fly_num"]): s["why"] for s in r["skipped"]}
    assert set(why) == {(i, "2024-01-02", n) for i in (a, b) for n in (1, 2)}
    assert why[(a, "2024-01-02", 1)].startswith("no RoiSet.zip") and why[(a, "2024-01-02", 2)].startswith("no folder of it")
    # of the datasets asked for
    assert [s["dataset"]["id"] for s in call(main.roi_training_flies, params={"datasets": str(b)})[1]["skipped"]] == [b, b]


def test_roi_glomeruli(root):  # noqa: F811
    p = root / "RoiSet.zip"
    roiset(p, ["DM2", "DM2_t1", "VM7d?"])
    assert roi_training.roi_glomeruli(p) == ({"glomeruli": ["DM2"], "tentative": []}, None)
    p.write_bytes(b"not a zip")
    found, why = roi_training.roi_glomeruli(p)
    assert found is None and why.startswith("its RoiSet.zip could not be read")


def test_recording_odors(tmp_path):
    rows, _, _ = odormeta.parse_sheet("name,abbrev\nvaleric acid,va\n")
    index = odormeta.OdorIndex([{**r, "id": i + 1} for i, r in enumerate(rows)])
    p = tmp_path / "t.json"
    # named in the odor table where it has them; as written otherwise, or only as "odors" are
    p.write_text(json.dumps([{"odors": "pentanoic acid @ -3", "name": "pentanoic acid", "abbrev": "va", "log10_conc": -3},
                             {"odors": "pentanoic acid @ -3"}, {"odors": "ea @ -2"}]))
    odors = roi_training.recording_odors(p, index)
    assert [(o["recorded"], o["odor"], o["odor_id"]) for o in odors] == [
        ("pentanoic acid @ -3", "valeric acid @ -3", 1), ("ea @ -2", "ea @ -2", None)]
    assert ("ea", "-2") in odors[1]["keys"]
    p.write_text("{}")
    assert roi_training.recording_odors(p, index) is None


def unzip(main, root, stacks):  # noqa: F811
    found, bases = main._roi_training_found(type("R", (), {"query_params": {}})())
    return zipfile.ZipFile(io.BytesIO(b"".join(roi_training.stream_zip(found, root, bases, stacks, "top"))))


def test_download(main, root, export):  # noqa: F811
    z = unzip(main, root, stacks=False)
    fly = "top/tom/analysis/2024-01-01/1"
    assert sorted(z.namelist()) == sorted([
        f"{fly}/mocorr_concat_avg.tif", f"{fly}/diagnostics1/RoiSet.zip", f"{fly}/diagnostics1/trialmean_dff.tif",
        f"{fly}/diagnostics1/trial_frames_and_odors.json", "top/tom/raw/2024-01-01/1/diagnostics1/Experiment.xml",
        "top/roi_training.json"])
    # the average image computed (and kept, to be sent again), where al_analysis writes it
    avg = tifffile.imread(io.BytesIO(z.read(f"{fly}/mocorr_concat_avg.tif")))
    assert avg.dtype == np.float32 and np.allclose(avg, export["movie"].mean(axis=0))
    assert len(list(config.ROI_TRAINING_DIR.iterdir())) == 1
    m = json.loads(z.read("top/roi_training.json"))
    f = m["flies"][0]
    assert f["average"] == "tom/analysis/2024-01-01/1/mocorr_concat_avg.tif" and f["average_from"].startswith("the mean of")
    assert f["recordings"] == [{"name": "diagnostics1", "responses": "tom/analysis/2024-01-01/1/diagnostics1/trialmean_dff.tif",
                                "trials": "tom/analysis/2024-01-01/1/diagnostics1/trial_frames_and_odors.json",
                                "experiment_xml": "tom/raw/2024-01-01/1/diagnostics1/Experiment.xml",
                                "odors": ["valeric acid @ -3", "ethyl acetate @ -2"],
                                "as_recorded": ["valeric acid @ -3", "ethyl acetate @ -2"]}]
    assert f["stacks"] == [] and f["failed"] == {} and len(m["skipped"]) == 4 and "DM2" in m["glomeruli"]
    # with its anatomical stack (as recorded), and al_analysis' own anatomy when it is there
    tifffile.imwrite(export["fly"] / "mocorr_concat_avg.tif", np.full(GRID, 7, np.float32), imagej=True, metadata={"axes": "ZYX"})
    z = unzip(main, root, stacks=True)
    assert {n for n in z.namelist() if "/anat/" in n} == {"top/tom/raw/2024-01-01/1/anat/Experiment.xml",
                                                          "top/tom/raw/2024-01-01/1/anat/ChanA_0001_0001_0001_0001.tif"}
    assert (tifffile.imread(io.BytesIO(z.read(f"{fly}/mocorr_concat_avg.tif"))) == 7).all()
    m = json.loads(z.read("top/roi_training.json"))
    assert m["flies"][0]["average_from"] == "al_analysis" and m["flies"][0]["stacks"][0]["dir"] == "tom/raw/2024-01-01/1/anat"


def test_download_failed(main, root, export):  # noqa: F811
    # a file gone once listed: left out, and said so in the manifest
    found, bases = main._roi_training_found(type("R", (), {"query_params": {}})())
    (export["fly"] / "diagnostics1/RoiSet.zip").unlink()
    z = zipfile.ZipFile(io.BytesIO(b"".join(roi_training.stream_zip(found, root, bases, False, "top"))))
    f = json.loads(z.read("top/roi_training.json"))["flies"][0]
    assert list(f["failed"]) == ["tom/analysis/2024-01-01/1/diagnostics1/RoiSet.zip"]
    assert "top/tom/analysis/2024-01-01/1/diagnostics1/RoiSet.zip" not in z.namelist()


def test_download_endpoint(main, root, export):  # noqa: F811
    from starlette.requests import Request

    def names(query: bytes) -> list[str]:
        scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": query, "path_params": {}}
        response = asyncio.run(main.roi_training_download(Request(scope)))
        assert response.media_type == "application/zip" and "olfdb_roi_training_" in response.headers["content-disposition"]

        async def body() -> bytes:
            return b"".join([chunk async for chunk in response.body_iterator])

        return zipfile.ZipFile(io.BytesIO(asyncio.run(body()))).namelist()

    # with the anatomical stacks, unless asked not to
    stack = lambda ns: any(n.endswith("/anat/ChanA_0001_0001_0001_0001.tif") for n in ns)  # noqa: E731
    assert stack(names(b"")) and not stack(names(b"stacks=0"))


def test_kept_summary(main, root, export):  # noqa: F811
    # none kept yet: made, and kept; then the one kept, though the fly's folder is gone, until asked for anew
    made = call(main.roi_training_flies, params={"kept": "1"})[1]
    assert len(made["flies"]) == 1 and "kept" not in made
    shutil.rmtree(export["fly"])
    kept = call(main.roi_training_flies, params={"kept": "1"})[1]
    assert kept.pop("kept") and kept == made
    assert call(main.roi_training_flies, params={})[1]["flies"] == []
    assert call(main.roi_training_flies, params={"kept": "1"})[1]["flies"] == []
    # one of some datasets only is not kept, nor answered from the one kept
    assert len(call(main.roi_training_flies, params={"datasets": str(export["ids"][0]), "kept": "1"})[1]["skipped"]) == 3
    assert len(call(main.roi_training_flies, params={"kept": "1"})[1]["skipped"]) == 6


def test_user_flies(main, root, export):  # noqa: F811
    assert call(main.user_flies, params={"name": "Ann"})[1] == {"flies": []}
    main.db.save_user("Ann", {"raw_dirs": ["tom/raw"], "mocorr_dirs": ["tom/analysis", "tom/mocorr"]}, "2026-01-01T00:00:00")
    # the flies with a recording's mean dF/F that are in the raw data too, with whether each has ROIs
    (root / "tom/raw/2024-01-02/1").mkdir(parents=True)
    recording(root / "tom/analysis/2024-01-03/1", "diagnostics1", ("va", -3))
    fly = lambda d, rois: {"date": d, "fly_num": 1, "dir": f"tom/analysis/{d}/1", "raw_dir": f"tom/raw/{d}/1", "rois": rois}  # noqa: E731
    assert call(main.user_flies, params={"name": "ann"})[1]["flies"] == [fly("2024-01-01", True), fly("2024-01-02", False)]
