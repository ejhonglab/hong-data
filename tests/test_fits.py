"""Tests of fits of al-diag-seg models (fits.py), their API, and the fit worker (fit_worker.py). Run
with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/. The worker is run with a fake
al-diag-seg, and (where al-diag-seg is installed: the worker image) with the real one, its
stand-in segmenter in place of Cellpose."""
from __future__ import annotations

import asyncio
import errno
import io
import json
import logging
import os
import sys
import types
import zipfile

import numpy as np
import pytest
import roifile
import tifffile
from starlette.requests import Request

from app import config, fit_worker, fits, odormeta
from app.glomeruli import load_glomeruli
from app.ingest import load_file
from tests.test_ingest import make_wide, write_csv
from tests.test_tensor import main, root  # noqa: F401 (fixtures)

GRID = (3, 24, 30)
GLOMS = ["DM2", "DL5", "VA1d", "DA1"]
ODORS = [("va", -3), ("ea", -2), ("2h", -6), ("1-6ol", -4)]
NAMES = {"va": "valeric acid", "ea": "ethyl acetate", "2h": "2-heptanone", "1-6ol": "1-hexanol"}
DIAG = [("glomeruli_diagnostics", False, f"{NAMES[a]} @ {c}", "solvent", r) for a, c in ODORS for r in (0, 1)]
FLIES = [("2024-01-01", n) for n in (1, 2, 3)]
XML = ('<ThorImageExperiment><Name name="t"/><Date date="01/01/2024 10:00:00" uTime="1704103200"/>'
       '<LSM pixelX="30" pixelY="24" widthUM="15" heightUM="12" pixelSizeUM="0.5" frameRate="30" averageMode="0" averageNum="1"/>'
       '<ZStage name="ThorZPiezo" steps="3" stepSizeUM="10" zStreamFrames="1"/>'
       '<Streaming enable="1" zFastEnable="1" flybackFrames="0" frames="30"/><PMT enableA="1" gainA="50" enableB="0" gainB="0"/>'
       '<ExperimentNotes text=""/></ThorImageExperiment>')


def trials(*odors):
    """A recording's trial_frames_and_odors.json, as al_analysis writes it (each odor twice)."""
    return json.dumps([{"start_frame": 0, "first_odor_frame": 1, "end_frame": 2, "odors": f"{NAMES[a]} @ {c}", "name": NAMES[a],
                        "abbrev": a, "log10_conc": c} for a, c in odors for _ in range(2)])


def request(method="GET", body=None, path_params=None, client=("10.1.2.3", 4321), headers=()):
    raw = json.dumps(body).encode() if body is not None else b""
    scope = {"type": "http", "method": method, "path": "/", "headers": [(b"content-type", b"application/json"), *headers],
             "query_string": b"", "path_params": path_params or {}, "client": client}

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request(scope, receive)


def call(endpoint, **kw) -> tuple[int, dict]:
    response = asyncio.run(endpoint(request(**kw)))
    return response.status_code, json.loads(response.body)


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "ROI_TRAINING_DIR", tmp_path / "roi_training")
    # the reference atlas fits add by default, as the worker's image has it (empty: the stand-in al-diag-seg doesn't read it)
    (tmp_path / "al_atlas_stub/grabe2016").mkdir(parents=True)
    (tmp_path / "al_atlas_stub/grabe2016/labels.tif").write_bytes(b"")
    monkeypatch.setattr(config, "AL_ATLAS_DIR", tmp_path / "al_atlas_stub")
    return tmp_path / "data"


@pytest.fixture
def tensor(main, root, data, tmp_path):  # noqa: F811
    """Three flies of a glomeruli dataset with a diagnostic panel, on tensor as al_analysis lays
    them out: a diagnostic recording each (four odors, each glomerulus tuned to its own), with
    its ROIs (each glomerulus a box on every plane, a little shifted between flies), its average
    image, and in its raw data the recording's Experiment.xml."""
    rows, _, _ = odormeta.parse_sheet("name,abbrev\n" + "".join(f"{n},{a}\n" for a, n in NAMES.items()))
    main.db.apply_odor_sync(added=odormeta.diff(rows, [])["added"], changed=[], deleted=[], source="test",
                            fingerprint="0" * 64, details={}, when="2026-11-04T00:00:00")
    parsed = load_file(write_csv(make_wide(DIAG, [(d, n, g) for d, n in FLIES for g in GLOMS]), tmp_path))
    main.db.insert_dataset(
        created_at="2026-01-02T03:04:05", title="t", readme_md="#", readme_edited=False, data_file="x.csv",
        metadata={"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "pebbled", "coding_unit": "glomeruli",
                  "raw_data_path": "ann/raw", "mocorr_data_path": "ann/analysis"},
        data_format=parsed.source_format, data_sha256="0" * 64, odor_yaml_files=[], experiment_xml_file=None,
        experiment_xml_info=None, summary=parsed.summary(), units=parsed.units, trials=parsed.trials, values=parsed.values)
    rng = np.random.default_rng(0)
    tuning = rng.uniform(0, 1, (len(GLOMS), len(ODORS)))
    for date, n in FLIES:
        fly = root / f"ann/analysis/{date}/{n}"
        rec = fly / "diagnostics1"
        rec.mkdir(parents=True)
        labels = np.zeros(GRID, np.uint8)
        for i in range(len(GLOMS)):
            y, x = 2 + 11 * (i // 2) + n % 2, 2 + 14 * (i % 2) + n
            labels[:, y:y + 9, x:x + 11] = i + 1
        resp = rng.normal(0, 0.05, (len(ODORS), *GRID)).astype(np.float32)
        for i in range(len(GLOMS)):
            resp[:, labels == i + 1] += tuning[i][:, None]
        resp[0, 0, 0, 0] = np.nan  # (as al_analysis' dF/F has at the odd voxel)
        tifffile.imwrite(rec / "trialmean_dff.tif", resp, imagej=True, metadata={"axes": "TZYX"})
        (rec / "trial_frames_and_odors.json").write_text(trials(*ODORS))
        for i, g in enumerate(GLOMS):
            for z in range(GRID[0]):
                ys, xs = np.nonzero(labels[z] == i + 1)
                box = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], "f4")
                roifile.ImagejRoi.frompoints(box, name=g, z=z).tofile(rec / "RoiSet.zip", name=f"{g}-{z}.roi")
        tifffile.imwrite(fly / "mocorr_concat_avg.tif", (1 + (labels > 0)).astype(np.float32), imagej=True, metadata={"axes": "ZYX"})
        raw = root / f"ann/raw/{date}/{n}/diagnostics1"
        raw.mkdir(parents=True)
        (raw / "Experiment.xml").write_text(XML)
    return root


# ------------------------------------------------------------------ the API

def test_options():
    assert fits.options({}) == fits.DEFAULTS
    assert fits.options({"held_out": "3", "epochs": 50, "atlas": False}) == {**fits.DEFAULTS, "held_out": 3, "epochs": 50,
                                                                             "atlas": False, "refs": []}
    # reference atlases: those it has, once each, only added to the atlas; Grabe et al.'s unless asked for none
    assert fits.DEFAULTS["refs"] == ["grabe2016"] and fits.options({"refs": []})["refs"] == []
    assert fits.options({"refs": ["grabe2016", "grabe2016"]})["refs"] == ["grabe2016"]
    assert fits.options({"refs": None, "atlas": False})["refs"] == []
    # how its ROIs are made (as the form sends them, strings)
    o = fits.options({"cellprob_threshold": "-1.5", "flow_threshold": "0", "dilate": "2", "join_overlap": "", "join_corr": -1})
    assert o == {**fits.DEFAULTS, "cellprob_threshold": -1.5, "flow_threshold": 0.0, "dilate": 2, "join_corr": -1.0}
    for bad in ({"epochs": 0}, {"held_out": "x"}, {"held_out": 31}, {"refs": ["hemibrain"]}, {"refs": "grabe2016"},
                {"refs": [{}]}, {"refs": ["grabe2016"], "atlas": False}, {"cellprob_threshold": "x"}, {"cellprob_threshold": 7},
                {"flow_threshold": -0.1}, {"dilate": "0.5"}, {"dilate": 11}, {"join_overlap": 1.5}, {"join_corr": "nan"}):
        with pytest.raises(ValueError):
            fits.options(bad)


def test_api(main, data):  # noqa: F811
    # a name is needed (Settings'), a first name alone
    assert call(main.fits_create, method="POST", body={"options": {}})[0] == 422
    status, r = call(main.fits_create, method="POST", body={"name": "Ann Lee"})
    assert status == 422 and "no spaces" in r["error"]
    assert call(main.fits_create, method="POST", body={"name": "Ann", "options": {"epochs": -1}})[0] == 400
    status, f = call(main.fits_create, method="POST", body={"name": "Ann", "options": {"held_out": 2}})
    assert status == 201 and (f["name"], f["ip"], f["status"], f["options"]["held_out"]) == ("Ann", "10.1.2.3", "queued", 2)
    # from behind a proxy, the address it says the request came from too
    f2 = call(main.fits_create, method="POST", body={"name": "Bob"}, headers=[(b"x-forwarded-for", b"10.9.9.9, 10.0.0.1")])[1]
    assert f2["ip"] == "10.9.9.9 (through 10.1.2.3)"
    status, r = call(main.fits_list)
    assert [x["id"] for x in r["fits"]] == [f2["id"], f["id"]] and not r["worker"]["alive"]
    # queued ones end one after the other
    eta = {x["id"]: x["eta"] for x in r["fits"]}
    assert eta[f2["id"]]["seconds"] > 0 and eta[f2["id"]]["ends_at"] > eta[f["id"]]["ends_at"] and eta[f["id"]]["from"] == 0
    # a queued one cancelled at once; a running one only asked to stop, and not deleted while it runs
    assert call(main.fits_cancel, method="POST", body={}, path_params={"id": f2["id"]})[1]["status"] == "cancelled"
    running = fits.claim(main.db)
    assert running["id"] == f["id"] and running["status"] == "running" and fits.claim(main.db) is None
    assert call(main.fits_cancel, method="POST", body={}, path_params={"id": f["id"]})[1]["cancel"] is True
    assert asyncio.run(main.fits_delete(request("DELETE", path_params={"id": f["id"]}))).status_code == 409
    assert fits.interrupt_running(main.db) == [f["id"]] and fits.get(main.db, f["id"])["status"] == "interrupted"
    fits.fit_dir(f["id"]).mkdir(parents=True)
    assert asyncio.run(main.fits_delete(request("DELETE", path_params={"id": f["id"]}))).status_code == 200
    assert fits.get(main.db, f["id"]) is None and not fits.fit_dir(f["id"]).exists()
    assert asyncio.run(main.fits_delete(request("DELETE", path_params={"id": f["id"]}))).status_code == 404
    # the worker, as it says it is there
    fits.heartbeat(main.db, {"host": "hal", "gpu": "RTX"})
    assert call(main.fits_list)[1]["worker"] | {"at": 0} == {"host": "hal", "gpu": "RTX", "at": 0, "alive": True,
                                                             "seen": main.db.get_setting(fits.WORKER_SETTING)["updated_at"]}


def test_estimates():
    done = [{"status": "done", "stages": {"load": {"seconds": 100, "size": 50}, "train": {"seconds": 900, "size": 9000},
                                          "atlas": {"seconds": 0, "size": 0}, "evaluate": {"seconds": 40, "size": 4}}}]
    r = fits.rates(done)
    assert r["from"] == 1 and r["seconds"] == {"load": 2.0, "train": 0.1, "atlas": fits.GUESSES["atlas"], "evaluate": 10.0}
    fit = {"options": {"held_out": 2, "epochs": 100, "atlas": False},
           "flies": {"train": ["a"] * 8, "held_out": ["b", "c"], "stacks": 0}}
    assert fits.sizes(fit) == {"load": 10, "train": 800, "atlas": 0, "evaluate": 2}
    # the atlas: a registration per training fly and anatomical stack, and per reference atlas
    assert fits.sizes({**fit, "options": {"refs": ["grabe2016"]}, "flies": {**fit["flies"], "stacks": 3}})["atlas"] == 12
    # (a fit asked for before refs were an option had none)
    assert fits.sizes({**fit, "options": {}, "flies": {**fit["flies"], "stacks": 3}})["atlas"] == 11
    # not started: every stage from the rates; halfway through training: what is left of it as it has gone, and the rest
    assert fits.remaining(fit, r, 1000.0) == pytest.approx(10 * 2 + 800 * 0.1 + 2 * 10)
    fit["progress"] = {"stage": "train", "done": 50, "total": 100, "stage_started": 900.0}
    assert fits.remaining(fit, r, 1000.0) == pytest.approx(100 + 2 * 10)


def test_held_out():
    def fly(ds, n, top="Ann"):
        return {"dir": f"{top}/analysis/2024-01-01/{n}", "date": "2024-01-01", "fly_num": n, "datasets": [{"id": ds}]}

    flies = [fly(1, n) for n in range(1, 7)] + [fly(2, n, "Bob") for n in range(1, 3)] + [fly(2, 1, "Bob")]
    assert fits.fly_names(flies)[-1] == "Bob_2024-01-01_1-2"
    # from each dataset in turn, spread through its flies
    assert fits.choose_held_out(flies, 3) == ["Ann_2024-01-01_2", "Ann_2024-01-01_5", "Bob_2024-01-01_1-2"]
    assert len(fits.choose_held_out(flies, 50)) == len(flies) - 1  # one left to fit to


def test_manifest(main, tensor):  # noqa: F811
    from app import roi_training

    found = roi_training.inventory(main.db, config.TENSOR_ROOT, [])
    m = fits.manifest(found, [])
    assert len(m["flies"]) == 3 and "DM2" in m["glomeruli"]
    f = m["flies"][0]
    # named from the top of tensor; every file where it is
    assert f["dir"] == "ann/analysis/2024-01-01/1" and f["roiset"] == str(tensor / "ann/analysis/2024-01-01/1/diagnostics1/RoiSet.zip")
    assert f["recordings"][0]["odors"] == [f"{NAMES[a]} @ {c}" for a, c in ODORS]
    assert f["average"] == str(tensor / "ann/analysis/2024-01-01/1/mocorr_concat_avg.tif")


# ----------------------------------------------------------------- the worker

def fake_alseg(monkeypatch, calls):
    """A stand-in for al-diag-seg's API as the worker uses it (alseg.olfdb, alseg.model)."""
    olfdb, model = types.ModuleType("alseg.olfdb"), types.ModuleType("alseg.model")

    def load_export(path, tentative=True, progress=None):
        m = json.loads(path.read_text())
        names = fits.fly_names(m["flies"])
        for i in range(len(names)):
            progress("load", i + 1, len(names))
        return {"flies": [{"name": n} for n in names], "odors": ["a"], "aliases": {}, "glomeruli": m["glomeruli"]}

    class Options:
        def __init__(self, **kw):
            calls["options"] = kw

    class Model:
        info, odors = {}, ["a"]

        def save(self, d):
            d.mkdir(parents=True)
            (d / "model.json").write_text("{}")
            (d / "segmenter").write_bytes(b"x" * 1000)

        def evaluate(self, flies, out, progress=None):
            out.mkdir(parents=True)
            (out / "summary.csv").write_text("fly\n")
            progress("evaluate", len(flies), len(flies))
            row = lambda f, kind, leak: {"fly": f["name"], "rois": kind, "n": 4, "recall": 0.5, "precision": 1.0,  # noqa: E731
                                         "dice": 0.8, "recall_2d": 0.6, "precision_2d": 0.9, "dice_2d": 0.85, "leak": leak}
            return {"final": "refined_tuning", "summary": [r for f in flies for r in (row(f, "manual", 0.03), row(f, "refined_tuning", 0.05))],
                    "identity": [{"fly": f["name"], "rois": "refined_tuning", "correct": 3, "total": 4} for f in flies],
                    "glomeruli": [{"glomerulus": "DM2", "flies": len(flies), "detected": 1, "dice": 0.8, "named": 1}],
                    "scoring": "partly drawn"}

    def fit(flies, val, options, out=None, progress=None, **kw):
        calls["held_out"] = val
        progress("train", 0, 10)
        if calls.get("cancel"):
            fits.request_cancel(calls["db"], calls["id"])
            progress("train", 5, 10)
        progress("atlas", 1, 1)
        if calls.get("fail"):
            raise RuntimeError("out of memory")
        return Model()

    olfdb.load_export, model.Options, model.fit, model.version = load_export, Options, fit, lambda: {"commit": "abc"}
    alseg = types.ModuleType("alseg")
    alseg.olfdb, alseg.model = olfdb, model
    monkeypatch.setitem(sys.modules, "alseg", alseg)
    monkeypatch.setitem(sys.modules, "alseg.olfdb", olfdb)
    monkeypatch.setitem(sys.modules, "alseg.model", model)
    monkeypatch.setattr(fit_worker, "SAVE_S", 0)


def test_worker(main, tensor, monkeypatch):  # noqa: F811
    calls = {"db": main.db}
    fake_alseg(monkeypatch, calls)
    f = fits.create(main.db, "Ann", "10.1.2.3", fits.options({"held_out": 1, "epochs": 10, "atlas": False}))
    fit_worker.run(main.db, fits.claim(main.db))
    f = fits.get(main.db, f["id"])
    assert f["status"] == "done" and f["error"] is None and calls["options"] == {"epochs": 10, "atlas": False, "tentative": True,
                                                                                 "refs": [], **fits.ROI_DEFAULTS}
    assert f["flies"]["held_out"] == calls["held_out"] == ["ann_2024-01-01_2"] and len(f["flies"]["train"]) == 2
    # each stage timed, with its size, for estimates of later fits
    assert set(f["stages"]) == set(fits.STAGES) and f["stages"]["train"]["size"] == 20 and f["stages"]["load"]["seconds"] > 0
    # (as 3D objects, and plane by plane)
    assert f["result"]["mean"] == {"recall": 0.5, "precision": 1.0, "dice": 0.8, "recall_2d": 0.6, "precision_2d": 0.9, "dice_2d": 0.85,
                                   "leak": 0.05, "manual_leak": 0.03}
    assert (f["result"]["named"], f["result"]["named_of"], f["model_bytes"]) == (3, 4, 1000 + 2 + len((fits.fit_dir(f["id"]) / "model/README.txt").read_bytes()))
    assert f["result"]["scoring"] == "partly drawn"
    readme = (fits.fit_dir(f["id"]) / "model/README.txt").read_text()
    assert not (fits.fit_dir(f["id"]) / "work").exists()  # (Cellpose's, removed once the fit is over)
    assert "fit #1" in readme and "al-diag-seg@abc" in readme and "alseg-predict model/" in readme
    assert "held out: ann_2024-01-01_2" in (fits.fit_dir(f["id"]) / "log.txt").read_text()
    # its download: how to run it, the model and the results
    z = zipfile.ZipFile(io.BytesIO(b"".join(fits.stream_zip(f["id"], "top"))))
    assert sorted(z.namelist()) == ["top/README.txt", "top/model/README.txt", "top/model/model.json", "top/model/segmenter",
                                    "top/results/summary.csv"]
    # stopped when asked; failed, with why
    calls["cancel"] = True
    g = fits.create(main.db, "Bob", None, fits.options({"held_out": 1}))
    calls["id"] = g["id"]
    fit_worker.run(main.db, fits.claim(main.db))
    assert fits.get(main.db, g["id"])["status"] == "cancelled"
    calls.update(cancel=False, fail=True)
    h = fits.create(main.db, "Bob", None, fits.options({}))
    fit_worker.run(main.db, fits.claim(main.db))
    h = fits.get(main.db, h["id"])
    assert (h["status"], h["error"], h["stages"]) == ("failed", "RuntimeError: out of memory", None)
    assert "Traceback" in (fits.fit_dir(h["id"]) / "log.txt").read_text()
    # estimates from the fit that finished
    assert fits.listing(main.db)["rates"]["from"] == 1
    # fits' work/ left from before (as the worker starts): removed
    (fits.fit_dir(h["id"]) / "work/models").mkdir(parents=True)
    (fits.fit_dir(h["id"]) / "work/models/al_glomeruli").write_bytes(b"w" * 10)
    assert fits.drop_work() == 10 and not (fits.fit_dir(h["id"]) / "work").exists() and (fits.fit_dir(h["id"]) / "log.txt").is_file()
    assert fits.drop_work() == 0


def test_worker_gpu_lost(main, tensor, monkeypatch):  # noqa: F811
    """A fit whose worker can no longer use its GPU, as training starts or once the fit failed, is put
    back in the queue (`GpuLost`), not trained on the CPU, nor failed; without a GPU, it is not checked."""
    calls, lost = {"db": main.db}, []
    fake_alseg(monkeypatch, calls)
    monkeypatch.setattr(fit_worker, "gpu_lost", lambda: lost.pop(0) if lost else None)
    f = fits.create(main.db, "Ann", None, fits.options({"held_out": 1}))
    lost.append("/dev/nvidia0: Operation not permitted")
    with pytest.raises(fit_worker.GpuLost, match="nvidia0: Operation not permitted"):
        fit_worker.run(main.db, fits.claim(main.db), gpu=True)
    f = fits.get(main.db, f["id"])
    assert (f["status"], f["started_at"], f["progress"], f["flies"], f["error"]) == ("queued", None, None, None, None)
    assert "can no longer use the GPU" in (fits.fit_dir(f["id"]) / "log.txt").read_text()
    calls["fail"] = True  # (as CUDA failing during training)
    lost[:] = [None, "/dev/nvidia0: Operation not permitted"]
    with pytest.raises(fit_worker.GpuLost):
        fit_worker.run(main.db, fits.claim(main.db), gpu=True)
    assert fits.get(main.db, f["id"])["status"] == "queued" and "Traceback" in (fits.fit_dir(f["id"]) / "log.txt").read_text()
    lost.append("/dev/nvidia0: Operation not permitted")
    fit_worker.run(main.db, fits.claim(main.db))
    f = fits.get(main.db, f["id"])
    assert (f["status"], f["error"]) == ("failed", "RuntimeError: out of memory") and len(lost) == 1


def test_gpu_lost(monkeypatch):
    """The GPU can no longer be used once one of its device files can no longer be opened."""
    assert fit_worker.gpu_lost(["/dev/null"]) is None and fit_worker.gpu_lost([]) is None
    real = os.open

    def denied(path, flags, *args):
        if path == "/dev/null":
            raise PermissionError(errno.EPERM, "Operation not permitted")
        return real(path, flags, *args)

    with monkeypatch.context() as m:
        m.setattr(os, "open", denied)
        assert fit_worker.gpu_lost(["/dev/null"]) == "/dev/null: Operation not permitted"


def test_worker_refs(main, tensor, monkeypatch, tmp_path):  # noqa: F811
    """A reference atlas asked for is given to al-diag-seg as its folder; without it there, a fit fails at once."""
    calls = {"db": main.db}
    fake_alseg(monkeypatch, calls)
    monkeypatch.setattr(config, "AL_ATLAS_DIR", tmp_path / "al_atlas")
    f = fits.create(main.db, "Ann", None, fits.options({"held_out": 1, "refs": ["grabe2016"]}))
    fit_worker.run(main.db, fits.claim(main.db))
    f = fits.get(main.db, f["id"])
    assert f["status"] == "failed" and "grabe2016: no labels.tif" in f["error"] and "options" not in calls
    d = tmp_path / "al_atlas/grabe2016"
    d.mkdir(parents=True)
    (d / "labels.tif").write_bytes(b"")
    g = fits.create(main.db, "Ann", None, fits.options({"held_out": 1, "refs": ["grabe2016"]}))
    fit_worker.run(main.db, fits.claim(main.db))
    assert fits.get(main.db, g["id"])["status"] == "done" and calls["options"]["refs"] == [str(d)]


def test_references():
    """The Grabe et al. 2016 atlas, where it is (the worker's image), as al-diag-seg loads it: either side's 54
    glomeruli, named as the database names them but VP1 (the database's VP1d, VP1l and VP1m)."""
    reference = pytest.importorskip("alseg.reference")
    d = fits.reference_dir("grabe2016")
    if not d.is_dir():
        pytest.skip(f"no {d} (the worker's image has it)")
    accepted = {g.casefold() for g in load_glomeruli()}
    assert reference.sides(d) == ["left", "right"]
    for side in ("left", "right"):
        names = reference.load_reference(d, side).names.values()
        assert len(names) == 54 and {n for n in names if n.casefold() not in accepted} == {"VP1"}


def test_free_gpu(caplog):
    """What PyTorch keeps of the GPU once done with it (its cache) given back, where there is a GPU."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    caplog.set_level(logging.INFO, logger="fit_worker")
    x = torch.ones(64 << 20, device="cuda")  # (256 MB)
    del x
    assert torch.cuda.memory_reserved() >= 256 << 20
    fit_worker._free_gpu()
    assert torch.cuda.memory_reserved() == 0 and "gave back" in caplog.text


def test_free_ram(caplog):
    """The RAM freed but kept by glibc's allocator given back, and how much the worker holds said
    (where /proc says: Linux)."""
    if fit_worker._rss_gb() is None:
        pytest.skip("no /proc/self/status")
    caplog.set_level(logging.INFO, logger="fit_worker")
    x = [bytearray(1000) for _ in range(200_000)]  # (small blocks, on the heap)
    del x
    fit_worker._free_ram()
    assert "RAM in use" in caplog.text and "after giving back what was freed" in caplog.text


def test_worker_with_alseg(main, tensor, monkeypatch):  # noqa: F811
    """The real al-diag-seg (where it is installed: the worker image), with its stand-in segmenter."""
    am = pytest.importorskip("alseg.model")

    class StandIn(am.Options):
        def __init__(self, **kw):
            super().__init__(**kw, segmenter="stand-in", gpu=False)

    monkeypatch.setattr(am, "Options", StandIn)
    f = fits.create(main.db, "Ann", "10.1.2.3", fits.options({"held_out": 1, "atlas": False}))
    fit_worker.run(main.db, fits.claim(main.db))
    f = fits.get(main.db, f["id"])
    d = fits.fit_dir(f["id"])
    assert f["status"] == "done", (d / "log.txt").read_text()[-3000:]
    held = f["flies"]["held_out"][0]
    names = {r.name for r in roifile.ImagejRoi.fromfile(d / f"results/{held}_refined_tuning_RoiSet.zip")}
    assert names and names <= set(GLOMS) | {n for n in names if n.isdigit()}, names
    assert (d / "model/model.json").is_file() and f["result"]["flies"][0]["fly"] == held
    assert f["result"]["scoring"] == "partly drawn"  # (glomeruli drawn on some of their planes only)
    # how its ROIs are made: the site's defaults are al-diag-seg's, and saved with the model
    assert fits.ROI_DEFAULTS == {k: getattr(am.Options(), k) for k in fits.ROI_DEFAULTS}
    saved = json.loads((d / "model/model.json").read_text())["options"]
    assert {k: saved[k] for k in fits.ROI_DEFAULTS} == fits.ROI_DEFAULTS
    assert "values not finite (NaN or inf), taken as 0" in (d / "log.txt").read_text()


def test_worker_with_cellpose(main, tensor):  # noqa: F811
    """The real al-diag-seg, training Cellpose for one epoch, where there is a GPU (the worker
    image, given one): what the stand-in segmenter skips. Its weights are written in the fit's
    work/ (which Cellpose does not make), saved with the model, and segment the held-out fly;
    work/ is removed once the fit is over."""
    pytest.importorskip("alseg.model")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    f = fits.create(main.db, "Ann", "10.1.2.3", fits.options({"held_out": 1, "epochs": 1, "atlas": False}))
    fit_worker.run(main.db, fits.claim(main.db))
    f = fits.get(main.db, f["id"])
    d = fits.fit_dir(f["id"])
    assert f["status"] == "done", (d / "log.txt").read_text()[-3000:]
    assert not (d / "work").exists() and (d / "model/model.json").is_file() and (d / "model/segmenter").is_file()
    assert f["result"]["flies"][0]["fly"] == f["flies"]["held_out"][0]
