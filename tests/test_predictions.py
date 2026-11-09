"""Tests of predictions of flies' glomerulus ROIs by fits' models (predictions.py), their API, and the
fit worker's making of them (fit_worker.run_prediction): with the real al-diag-seg, where it is
installed (the worker image), its stand-in segmenter in place of Cellpose. Run with:
OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/."""
from __future__ import annotations

import io
import json
import shutil
import sys
import threading
import time
import zipfile

import numpy as np
import pytest
import roifile
import tifffile
from starlette.exceptions import HTTPException

from app import config, fit_worker, fits, predictions
from tests.test_fits import GLOMS, call, data, request, tensor  # noqa: F401 (fixtures)
from tests.test_tensor import main, root  # noqa: F401 (fixtures)

FLY = "ann/analysis/2024-01-01/1"
RAW = "ann/raw/2024-01-01/1"


def done_fit(db) -> dict:
    """A fit done, with a model (as far as the web server can tell: its model.json)."""
    f = fits.create(db, "Ann", None, fits.options({"held_out": 1}))
    fits.update(db, f["id"], status="done", finished_at=fits.now(),
                flies={"train": ["ann_2024-01-01_2", "ann_2024-01-01_3"], "held_out": ["ann_2024-01-01_1"]})
    (fits.fit_dir(f["id"]) / "model").mkdir(parents=True)
    (fits.fit_dir(f["id"]) / "model" / "model.json").write_text("{}")
    return fits.get(db, f["id"])


def test_api(main, tensor):  # noqa: F811
    f = done_fit(main.db)
    body = {"name": "Bob", "fit": f["id"], "dir": FLY, "raw_dir": RAW}
    # who asks, with a model, of a fly's folders with what al-diag-seg reads
    assert call(main.predictions_create, method="POST", body={**body, "name": ""})[0] == 422
    fits.update(main.db, f["id"], status="failed")
    assert "no model" in call(main.predictions_create, method="POST", body=body)[1]["error"]
    fits.update(main.db, f["id"], status="done")
    status, r = call(main.predictions_create, method="POST", body={**body, "dir": "ann/analysis/2030-01-01/1"})
    assert status == 400 and r["error"].startswith("its motion-corrected data folder:")
    status, r = call(main.predictions_create, method="POST", body={**body, "dir": "ann/raw/2024-01-01/1"})
    assert status == 400 and "trialmean_dff.tif" in r["error"]
    status, p = call(main.predictions_create, method="POST", body=body)
    assert status == 201 and (p["fly"], p["status"], p["name"], p["ip"]) == ("ann_2024-01-01_1", "queued", "Bob", "10.1.2.3")
    assert p["fly_dir"] == str(tensor / FLY) and p["raw_dir"] == str(tensor / RAW)
    # asked for again while it waits: that one; the same fly, however entered
    status, again = call(main.predictions_create, method="POST", body={**body, "dir": str(tensor / FLY) + "/"})
    assert status == 200 and again["id"] == p["id"]
    # once made: said to be there (to view, or make again, replacing it)
    claimed = predictions.claim(main.db)
    assert claimed["id"] == p["id"] and claimed["status"] == "running" and predictions.claim(main.db) is None
    predictions.update(main.db, p["id"], status="done", finished_at=predictions.now(), result={"n": 4})
    status, r = call(main.predictions_create, method="POST", body={**body, "name": "Cy"})
    assert status == 409 and r["prediction"]["id"] == p["id"] and r["prediction"]["name"] == "Bob"
    status, r = call(main.predictions_create, method="POST", body={**body, "name": "Cy", "overwrite": True})
    assert status == 201 and r["id"] == p["id"] and (r["status"], r["name"], r["result"], r["finished_at"]) == ("queued", "Cy", None, None)
    # listed, with the fly's folder from the top of tensor, and the models to predict with
    status, r = call(main.predictions_list)
    assert [x["id"] for x in r["predictions"]] == [p["id"]] and r["predictions"][0]["tensor_dir"] == FLY
    assert [m["id"] for m in r["models"]] == [f["id"]] and r["models"][0]["held_out"] == ["ann_2024-01-01_1"]
    assert not r["predictions"][0]["fit_deleted"]
    fits.delete(main.db, f["id"])
    status, r = call(main.predictions_list)
    assert r["predictions"][0]["fit_deleted"] and r["models"] == []
    # not deleted while it runs
    predictions.claim(main.db)
    assert call(main.predictions_delete, method="DELETE", path_params={"id": p["id"]})[0] == 409
    assert predictions.interrupt_running(main.db) == [p["id"]]
    predictions.pred_dir(p["id"]).mkdir(parents=True)
    assert call(main.predictions_delete, method="DELETE", path_params={"id": p["id"]})[0] == 200
    assert predictions.get(main.db, p["id"]) is None and not predictions.pred_dir(p["id"]).exists()


def test_view(main, tensor):  # noqa: F811
    f = done_fit(main.db)
    fly = predictions.locate(FLY, RAW, config.TENSOR_ROOT, [])
    p, _ = predictions.create(main.db, f["id"], fly, "Ann", None)
    assert call(main.predictions_view, path_params={"id": p["id"]})[0] == 404  # (not done)
    d = predictions.pred_dir(p["id"])
    d.mkdir(parents=True)
    tifffile.imwrite(d / predictions.AVERAGE, np.ones((3, 4, 5), np.float32), imagej=True, metadata={"axes": "ZYX"})
    roifile.ImagejRoi.frompoints(np.array([[0, 0], [2, 0], [2, 2]], "f4"), name="DM2", z=1).tofile(d / "ann_2024-01-01_1_RoiSet.zip")
    manual = str(tensor / FLY / "diagnostics1/RoiSet.zip")
    (d / predictions.PROVENANCE).write_text(json.dumps({"fly": {"roiset": manual}}))
    predictions.update(main.db, p["id"], status="done")
    status, v = call(main.predictions_view, path_params={"id": p["id"]})
    # its average image, with the ROIs predicted and those it was scored against
    assert status == 200 and v["fly"] == "ann_2024-01-01_1" and len(v["movies"]) == 1
    m = v["movies"][0]
    assert (m["path"], m["rois"], m["predicted"]) == (str(d / predictions.AVERAGE), manual, str(d / "ann_2024-01-01_1_RoiSet.zip"))
    # which the movie player may read, as it may tensor's; not the rest of the data folder
    assert main._server_file(m["predicted"], (".zip",)) == d / "ann_2024-01-01_1_RoiSet.zip"
    (fits.fit_dir(f["id"]) / "model" / "x.zip").write_bytes(b"")
    with pytest.raises(HTTPException):
        main._server_file(str(fits.fit_dir(f["id"]) / "model" / "x.zip"), (".zip",))
    # its download: its folder
    z = zipfile.ZipFile(io.BytesIO(b"".join(predictions.stream_zip(p["id"], "top"))))
    assert sorted(z.namelist()) == ["top/ann_2024-01-01_1_RoiSet.zip", "top/average.tif", "top/provenance.json"]


def test_streams(tmp_path):
    """Each worker thread's printing to its own log, as a fit and a prediction run at once."""
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = fit_worker._Streams(sys.stdout), fit_worker._Streams(sys.stderr)
    try:
        go = threading.Barrier(2)

        def work(name):
            with open(tmp_path / f"{name}.txt", "w") as out, fit_worker._output_to(out):
                go.wait()
                for i in range(50):
                    print(name, i)
                fit_worker.log.warning("%s done", name)

        ts = [threading.Thread(target=work, args=(n,)) for n in ("fit", "prediction")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        sys.stdout, sys.stderr = saved
    for name in ("fit", "prediction"):
        lines = (tmp_path / f"{name}.txt").read_text().splitlines()
        assert lines[:50] == [f"{name} {i}" for i in range(50)] and lines[50:] == [f"{name} done"]


def test_worker_prediction(main, tensor, monkeypatch):  # noqa: F811
    """The real al-diag-seg (where it is installed: the worker image), with its stand-in segmenter:
    a fit's model predicting a fly with ROIs of its own (scored against them), and one without."""
    am = pytest.importorskip("alseg.model")

    class StandIn(am.Options):  # (also as the model is loaded, with its options saved)
        def __init__(self, **kw):
            super().__init__(**{"segmenter": "stand-in", "gpu": False, **kw})

    monkeypatch.setattr(am, "Options", StandIn)
    f = fits.create(main.db, "Ann", None, fits.options({"held_out": 1, "atlas": False}))
    fit_worker.run(main.db, fits.claim(main.db))
    assert fits.get(main.db, f["id"])["status"] == "done"
    held = fits.get(main.db, f["id"])["flies"]["held_out"][0]
    date, n = held.split("_")[1:]

    p, _ = predictions.create(main.db, f["id"], predictions.locate(f"ann/analysis/{date}/{n}", f"ann/raw/{date}/{n}",
                                                                    config.TENSOR_ROOT, []), "Bob", None)
    fit_worker.run_prediction(main.db, predictions.claim(main.db), False, "a fit is using the GPU")
    p = predictions.get(main.db, p["id"])
    d = predictions.pred_dir(p["id"])
    assert p["status"] == "done", (d / "log.txt").read_text()[-3000:]
    r = p["result"]
    assert r["final"] == "pred" and r["n"] >= 1 and r["device"] == "CPU" and set(r["named"]) <= set(GLOMS)
    # (its own ROIs grown, as the stand-in segmenter predicts: found as 3D objects and plane by plane)
    assert r["scores"]["manual"] == len(GLOMS) and r["scores"]["recall"] > 0 and r["scores"]["recall_2d"] > 0
    assert r["scores"]["manual_2d"] == len(GLOMS) * 3
    names = {x.name for x in roifile.ImagejRoi.fromfile(d / f"{held}_RoiSet.zip")}
    assert names and names <= set(GLOMS) | {x for x in names if x.isdigit()}
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["model"]["fit"] == f["id"] and prov["fly"]["recordings"][0]["used"] and prov["fly"]["roiset"].endswith("RoiSet.zip")
    assert prov["device"] == "CPU (a fit is using the GPU)" and prov["fly"]["recordings"][0]["responses"]["bytes"] > 0
    assert tifffile.imread(d / "average.tif").shape == tuple(prov["fly"]["planes"])
    assert "summary.csv" in (d / "README.txt").read_text() and (d / "summary.csv").is_file()

    # a fly with no ROIs of its own (the stand-in segmenter then needs some: boxes)
    new, raw = tensor / "ann/analysis/2024-02-02/1", tensor / "ann/raw/2024-02-02/1"
    shutil.copytree(tensor / f"ann/analysis/{date}/{n}", new)
    shutil.copytree(tensor / f"ann/raw/{date}/{n}", raw)
    (new / "diagnostics1/RoiSet.zip").unlink()

    def boxes(fly):
        lab = np.zeros(fly["average"].shape, np.int32)
        lab[:, 2:10, 2:12], lab[:, 13:21, 15:26] = 1, 2
        return lab

    monkeypatch.setattr(am, "_stand_in", boxes)
    q, _ = predictions.create(main.db, f["id"], predictions.locate(str(new), str(raw), config.TENSOR_ROOT, []), "Bob", None)
    assert q["fly"] == "ann_2024-02-02_1"
    fit_worker.run_prediction(main.db, predictions.claim(main.db), False)
    q = predictions.get(main.db, q["id"])
    d = predictions.pred_dir(q["id"])
    assert q["status"] == "done", (d / "log.txt").read_text()[-3000:]
    assert q["result"]["scores"] is None and q["result"]["n"] >= 1 and q["result"]["device"] == "CPU"
    assert (d / "ann_2024-02-02_1_identities.csv").is_file() and (d / "ann_2024-02-02_1_RoiSet.zip").is_file()
    assert json.loads((d / "provenance.json").read_text())["fly"]["roiset"] is None
    # (the viewer: its average image, with only the ROIs predicted)
    v = predictions.view(main.db, q["id"])
    assert v["movies"][0]["rois"] is None and v["movies"][0]["predicted"].endswith("ann_2024-02-02_1_RoiSet.zip")

    # a fit's model gone: said so
    fits.delete(main.db, f["id"])
    predictions.create(main.db, f["id"], predictions.locate(str(new), str(raw), config.TENSOR_ROOT, []), "Bob", None, overwrite=True)
    fit_worker.run_prediction(main.db, predictions.claim(main.db), False)
    assert "the fit was deleted" in predictions.get(main.db, q["id"])["error"]


def test_prediction_with_cellpose(main, tensor, monkeypatch):  # noqa: F811
    """The real al-diag-seg and Cellpose (where they are installed: the worker image): a model fit
    for one epoch on the CPU (leaving the GPU to what else uses it), then a prediction on the CPU
    (as while a fit uses the GPU) and, where there is one, on the GPU."""
    am = pytest.importorskip("alseg.model")
    torch = pytest.importorskip("torch")

    class OnCPU(am.Options):
        def __init__(self, **kw):
            super().__init__(**{"gpu": False, **kw})

    monkeypatch.setattr(am, "Options", OnCPU)
    f = fits.create(main.db, "Ann", None, fits.options({"held_out": 1, "epochs": 1, "atlas": False}))
    fit_worker.run(main.db, fits.claim(main.db))
    f = fits.get(main.db, f["id"])
    assert f["status"] == "done", (fits.fit_dir(f["id"]) / "log.txt").read_text()[-3000:]
    date, n = f["flies"]["held_out"][0].split("_")[1:]
    fly = predictions.locate(f"ann/analysis/{date}/{n}", f"ann/raw/{date}/{n}", config.TENSOR_ROOT, [])
    for on_gpu in (False, True) if torch.cuda.is_available() else (False,):
        p, _ = predictions.create(main.db, f["id"], fly, "Bob", None, overwrite=True)
        fit_worker.run_prediction(main.db, predictions.claim(main.db), on_gpu)
        p = predictions.get(main.db, p["id"])
        assert p["status"] == "done", (predictions.pred_dir(p["id"]) / "log.txt").read_text()[-3000:]
        assert p["result"]["device"] == ("GPU" if on_gpu else "CPU") and p["result"]["scores"]["manual"] == len(GLOMS)


def wait(cond, seconds: float = 10) -> bool:
    """Whether `cond()` comes true within `seconds`."""
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_loops(main, tensor, monkeypatch):  # noqa: F811
    """The worker's two threads: a prediction runs on the CPU while a fit holds the GPU, on the GPU
    when none does; a fit asked for waits for a prediction on the GPU to end."""
    monkeypatch.setattr(fit_worker, "POLL_S", 0.01)
    fitting, fit_go, ran, pred_go = threading.Event(), threading.Event(), [], threading.Event()

    def run(db, fit, gpu=False):
        fitting.set()
        fit_go.wait(10)
        fits.update(db, fit["id"], status="done")

    def run_prediction(db, p, on_gpu, why_cpu=None):
        ran.append((p["id"], on_gpu, why_cpu, state["fit"]))
        pred_go.wait(10)
        predictions.update(db, p["id"], status="done")

    monkeypatch.setattr(fit_worker, "run", run)
    monkeypatch.setattr(fit_worker, "run_prediction", run_prediction)
    freed = []  # (each time the GPU's memory is given back)
    monkeypatch.setattr(fit_worker, "_free_gpu", lambda: freed.append(1))
    f = done_fit(main.db)
    fly = predictions.locate(FLY, RAW, config.TENSOR_ROOT, [])
    state, lock, stop = {"fit": None, "prediction": None}, threading.Lock(), threading.Event()
    threads = [threading.Thread(target=fit_worker.fit_loop, args=(main.db, state, lock, True, stop)),
               threading.Thread(target=fit_worker.predict_loop, args=(main.db, state, lock, True, stop))]
    for t in threads:
        t.start()
    try:
        g = fits.create(main.db, "Ann", None, fits.options({}))
        assert fitting.wait(10) and lock.locked()
        p, _ = predictions.create(main.db, f["id"], fly, "Bob", None)
        pred_go.set()
        assert wait(lambda: predictions.get(main.db, p["id"])["status"] == "done")
        assert ran == [(p["id"], False, "a fit is using the GPU", g["id"])]
        fit_go.set()
        assert wait(lambda: fits.get(main.db, g["id"])["status"] == "done") and wait(lambda: not lock.locked())
        assert len(freed) == 1  # (after the fit; not after the prediction on the CPU)
        pred_go.clear()
        predictions.create(main.db, f["id"], fly, "Bob", None, overwrite=True)
        assert wait(lambda: len(ran) == 2) and ran[1][1:] == (True, None, None) and lock.locked() and len(freed) == 1
        h = fits.create(main.db, "Ann", None, fits.options({}))
        fitting.clear()
        assert not fitting.wait(0.3)  # (waits for the prediction on the GPU)
        pred_go.set()
        assert fitting.wait(10) and fits.get(main.db, h["id"])["status"] in ("running", "done") and len(freed) >= 2
    finally:
        stop.set()
        fit_go.set()
        pred_go.set()
        for t in threads:
            t.join(10)


def test_loops_gpu_lost(main, tensor, monkeypatch):  # noqa: F811
    """A fit or a prediction to run on a GPU the worker can no longer use is put back in the queue,
    unrun, and the worker told to stop (to be started again, with the GPU); a prediction on the
    CPU, beside a fit, is not held up."""
    monkeypatch.setattr(fit_worker, "POLL_S", 0.01)
    ran = []
    monkeypatch.setattr(fit_worker, "run", lambda db, fit, gpu=False: ran.append(fit["id"]))
    monkeypatch.setattr(fit_worker, "run_prediction", lambda db, p, on_gpu, why_cpu=None: ran.append((p["id"], on_gpu)))
    monkeypatch.setattr(fit_worker, "gpu_lost", lambda: "/dev/nvidia0: Operation not permitted")
    state, lock, stop = {"fit": None, "prediction": None}, threading.Lock(), threading.Event()
    g = fits.create(main.db, "Ann", None, fits.options({}))
    fit_worker.fit_loop(main.db, state, lock, True, stop)
    g = fits.get(main.db, g["id"])
    assert stop.is_set() and ran == [] and (g["status"], g["started_at"]) == ("queued", None) and not lock.locked()
    stop.clear()
    p, _ = predictions.create(main.db, done_fit(main.db)["id"], predictions.locate(FLY, RAW, config.TENSOR_ROOT, []), "Bob", None)
    fit_worker.predict_loop(main.db, state, lock, True, stop)
    p = predictions.get(main.db, p["id"])
    assert stop.is_set() and ran == [] and (p["status"], p["started_at"]) == ("queued", None) and not lock.locked()
    stop.clear()
    with lock:  # (a fit holds the GPU: the prediction runs on the CPU)
        t = threading.Thread(target=fit_worker.predict_loop, args=(main.db, state, lock, True, stop))
        t.start()
        assert wait(lambda: ran == [(p["id"], False)])
        stop.set()
        t.join(10)
