"""Tests of the look for glomeruli a prediction's ROIs miss (nmf_check.py): its state and API, the
names of ROIs with what it found merged in, what the web server asks of the fit worker, and the
worker's making of one (fit_worker.run_nmf) with the real al-diag-seg, where it is installed (the
worker image), on a fly of the stand-in segmenter's with ROIs left out. Run with:
OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/."""
from __future__ import annotations

import json
import threading
import time
import zipfile

import numpy as np
import pytest
import roifile
import tifffile
from scipy import ndimage as ndi

from app import config, fit_worker, fits, nmf_check, predictions
from tests.test_fits import GLOMS, GRID, NAMES, ODORS, call, data, request, tensor  # noqa: F401 (fixtures)
from tests.test_predictions import FLY, RAW, done_fit
from tests.test_tensor import main, root  # noqa: F401 (fixtures)

# what a look left in a prediction's folder (nmf_check.run): of ROIs 1 (DM2), 2 (DL5) and 3 (left unnamed), a new ROI (4), more
# of DL5 on a plane, and a part cut off DM2 (5)
SQUARE = [[0, 0], [4, 0], [4, 4], [0, 4]]
DOC = {
    "finished": "2026-11-15T12:00:00-08:00", "of": "x_pred.tif", "params": {"n_free": 25}, "cleaned": ["margin"],
    "predicted": {"1": "DM2", "2": "DL5", "3": None},
    "candidates": [{"id": 1, "kind": "added", "label": 4, "of": None, "of_name": None, "planes": [0, 1], "name": "DA1", "area_um2": 30.0,
                    "outlines": [{"z": 0, "points": SQUARE}, {"z": 1, "points": SQUARE}]},
                   {"id": 2, "kind": "extended", "label": 2, "of": 2, "of_name": "DL5", "planes": [2], "name": "DL5", "area_um2": 25.0,
                    "outlines": [{"z": 2, "points": SQUARE}]},
                   {"id": 3, "kind": "split", "label": 5, "of": 1, "of_name": "DM2", "planes": [0], "name": None, "area_um2": 21.0,
                    "outlines": [{"z": 0, "points": SQUARE}], "rest": [{"z": 0, "points": SQUARE}]}],
    "names": {"kept": {"1": "DM2", "2": "DL5", "3": None, "4": "DA1", "5": None},
              "joint": {"1": "VA1d", "2": "DL5", "3": None, "4": "DA1", "5": "DM2"}},
    "glomeruli": ["DA1", "DL5", "DM2", "VA1d", "not_one"],
    "files": {"atlas": None},
}


def prediction(main, status="done") -> dict:  # noqa: F811
    f = done_fit(main.db)
    p, _ = predictions.create(main.db, f["id"], predictions.locate(FLY, RAW, config.TENSOR_ROOT, []), "Ann", None)
    predictions.pred_dir(p["id"]).mkdir(parents=True)
    predictions.update(main.db, p["id"], status=status, result={"final": "pred", "n": 3, "named": ["DM2", "DL5"], "scores": None, "device": "CPU"})
    return predictions.get(main.db, p["id"])


def test_state(main, tensor):  # noqa: F811
    """A look asked for, claimed by the worker, followed, and done: as the tab and the viewer are told."""
    db = main.db
    p = prediction(main, "queued")
    nmf = lambda **kw: call(main.predictions_nmf, path_params={"id": p["id"]}, **kw)  # noqa: E731
    ask = lambda **body: call(main.predictions_nmf_ask, method="POST", path_params={"id": p["id"]}, body=body)  # noqa: E731
    assert nmf()[0] == 404 and ask(name="Bob")[0] == 404  # (not done)
    predictions.update(db, p["id"], status="done")
    assert nmf()[1] == {"nmf": None, "found": None} and predictions.get(db, p["id"])["nmf"] is None
    assert ask()[0] == 422 and ask(name="Bob Smith")[0] == 422 and call(main.predictions_nmf_ask, method="POST", path_params={"id": 99}, body={"name": "Bob"})[0] == 404
    # its settings: those of the form that are not the defaults are kept; one that is none, or out of range, is refused
    assert nmf_check.settings({}) == {} and nmf_check.settings({"settings": {"min_eta2": 0.5, "n_free": 25, "split": True}}) == {}
    assert nmf_check.settings({"settings": {"min_eta2": 0.3, "n_free": 0.0, "split": False}}) == {"min_eta2": 0.3, "n_free": 0, "split": False}
    for bad in ({"theta": 0.2}, {"min_eta2": 1.5}, {"min_novelty": "0.5"}, {"n_free": 2.5}, {"n_free": True}, {"min_area_um2": 0}):
        status, r = ask(name="Bob", settings=bad)
        assert status == 400 and r["error"].startswith("settings:"), bad
    status, r = ask(name="Bob", settings={"min_novelty": 0.4, "min_eta2": 0.5})
    assert status == 201 and r["nmf"]["status"] == "queued" and r["nmf"]["name"] == "Bob" and r["nmf"]["asked_at"] and r["nmf"]["settings"] == {"min_novelty": 0.4}
    assert ask(name="Cy")[0] == 200 and nmf()[1]["nmf"]["name"] == "Bob"  # (once: the one waiting is whose it is)
    listed = call(main.predictions_list)[1]["predictions"][0]
    assert listed["nmf"]["status"] == "queued" and call(main.predictions_view, path_params={"id": p["id"]})[1]["nmf"]["status"] == "queued"
    assert predictions.worker_jobs(db)["queued"] == 1

    got = nmf_check.claim(db)
    assert got["id"] == p["id"] and got["nmf"]["status"] == "running" and got["nmf"]["started_at"] and nmf_check.claim(db) is None
    assert ask(name="Cy")[0] == 200 and predictions.delete(db, p["id"]) is False  # (not while it runs)
    assert call(main.predictions_delete, method="DELETE", path_params={"id": p["id"]})[0] == 409
    jobs = predictions.worker_jobs(db)
    assert jobs["queued"] == 0 and len(jobs["jobs"]) == 1 and "miss being looked for, asked for by Bob" in jobs["jobs"][0]["what"]
    # how far it is: the stages it has been through, and the one it is at as far as it says
    of = {"load": 10, "register": 60, "read": 30, "factorize": 20, "name": 10}
    assert nmf_check.expected(db, "GPU") == nmf_check.GUESSES["GPU"]
    assert nmf_check.fraction(None, of) == 0.0 and nmf_check.fraction({"stage": "done"}, of) == 0.0
    assert nmf_check.fraction({"stage": "read", "done": 1, "total": 2, "stages": ["load", "read", "factorize", "name"]}, of) == pytest.approx(25 / 70)
    assert nmf_check.fraction({"stage": "factorize", "done": 5, "total": 10}, of) == pytest.approx(110 / 130)
    assert nmf_check.fraction({"stage": "register", "stage_started": 100.0}, of, t=130.0) == pytest.approx(40 / 130)
    assert nmf_check.fraction({"stage": "register", "stage_started": 100.0}, of, t=1e6) == pytest.approx((10 + 0.95 * 60) / 130)
    nmf_check.update(db, p["id"], progress={"stage": "factorize", "done": 1, "total": 4, "device": "GPU", "stages": ["load", "read", "factorize", "name"]})
    r = nmf()[1]["nmf"]
    assert r["status"] == "running" and r["progress"]["stage"] == "factorize" and r["fraction"] == pytest.approx(45 / 70) and nmf()[1]["found"] is None

    nmf_check.files(got)["json"].write_text(json.dumps(DOC))
    nmf_check.update(db, p["id"], "done", finished_at=predictions.now(), progress=None,
                     result={"candidates": 3, "kinds": {"added": 1, "extended": 1, "split": 1}, "device": "GPU", "seconds": 40.0, "stages": {"load": 4.0, "read": 9.0}})
    r = nmf()[1]
    assert r["nmf"]["status"] == "done" and r["nmf"]["result"]["candidates"] == 3 and "fraction" not in r["nmf"]
    assert nmf_check.expected(db, "GPU") == {**nmf_check.GUESSES["GPU"], "load": 4.0, "read": 9.0} and nmf_check.expected(db, "CPU") == nmf_check.GUESSES["CPU"]
    found = r["found"]
    assert found["run"] == DOC["finished"] and [c["id"] for c in found["candidates"]] == [1, 2, 3] and found["predicted"] == DOC["predicted"]
    assert found["names"] == DOC["names"] and found["candidates"][2]["rest"] == [{"z": 0, "points": SQUARE}] and found["rois"] == []
    assert found["kinds"] == {"DA1": "glomerulus", "DL5": "glomerulus", "DM2": "glomerulus", "VA1d": "glomerulus", "not_one": "other"}
    # once it is over, the prediction can be deleted, the look's files with it
    assert predictions.delete(db, p["id"]) is True and not nmf_check.files(got)["json"].exists()


def test_again(main, tensor):  # noqa: F811
    """A look left running by a worker that stopped is said so, and can be asked for again; predicted again, it is gone."""
    db = main.db
    p = prediction(main)
    nmf_check.ask(db, p["id"], "Bob", None)
    nmf_check.claim(db)
    assert nmf_check.interrupt_running(db) == [p["id"]]
    n = predictions.get(db, p["id"])["nmf"]
    assert n["status"] == "interrupted" and "stopped" in n["error"] and nmf_check.interrupt_running(db) == []
    assert nmf_check.found(predictions.get(db, p["id"]), set(GLOMS)) is None
    _, queued = nmf_check.ask(db, p["id"], "Cy", "10.0.0.1")
    n = predictions.get(db, p["id"])["nmf"]
    assert queued and n == {"status": "queued", "name": "Cy", "ip": "10.0.0.1", "asked_at": n["asked_at"], "settings": {}}
    # predicted again: the look is gone with the prediction's files, and what a worker still at it says is dropped
    nmf_check.claim(db)
    predictions.create(db, p["fit_id"], predictions.locate(FLY, RAW, config.TENSOR_ROOT, []), "Ann", None, overwrite=True)
    assert predictions.get(db, p["id"])["nmf"] is None
    nmf_check.update(db, p["id"], "done", result={"candidates": 1})
    assert predictions.get(db, p["id"])["nmf"] is None and nmf_check.claim(db) is None


def test_plan(main, tensor, monkeypatch):  # noqa: F811
    """The names of the ROIs with candidates merged in: as the look left them for all of them, and for some with the
    predicted ROIs' names kept; for some named again together, asked of the fit worker."""
    db = main.db
    p = prediction(main)
    plan = lambda **body: call(main.predictions_nmf_plan, method="POST", path_params={"id": p["id"]}, body=body)  # noqa: E731
    assert plan(chosen=[1])[0] == 404  # (no look)
    nmf_check.ask(db, p["id"], "Bob", None)
    nmf_check.files(p)["json"].write_text(json.dumps(DOC))
    nmf_check.update(db, p["id"], "done", finished_at=predictions.now())
    for bad in (None, "1", [9], [1, "2"], [True]):
        assert plan(chosen=bad)[0] == 400, bad
    status, r = plan(chosen=[3, 1, 2, 1])
    assert status == 200 and r["names"] == DOC["names"]["joint"] and r["kinds"] == {"VA1d": "glomerulus", "DL5": "glomerulus", "DA1": "glomerulus", "DM2": "glomerulus"}
    assert plan(chosen=[1, 2, 3], joint=False)[1]["names"] == DOC["names"]["kept"]
    # some of them, names kept: the others' own labels left out (an ROI another is more of, or cut from, stays)
    assert plan(chosen=[2], joint=False)[1]["names"] == {"1": "DM2", "2": "DL5", "3": None}
    assert plan(chosen=[1], joint=False)[1]["names"] == {"1": "DM2", "2": "DL5", "3": None, "4": "DA1"}
    assert plan(chosen=[], joint=False)[1]["names"] == DOC["predicted"]
    # some of them, named again together: the worker's to do, which is not there
    status, r = plan(chosen=[1, 3])
    assert status == 503 and "the fit worker is not running" in r["error"]
    fits.heartbeat(db, {"host": "h", "gpu": None})
    asked = []

    def named(db_, args):
        asked.append(args)
        if args["chosen"] == [1]:
            raise ValueError("that look is no longer there")
        return {"names": {"1": "DM2", "2": "DL5", "3": None, "4": "VA1d", "5": "DA1"}}

    monkeypatch.setitem(nmf_check.CALLS, "nmf_plan", named)
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            if not nmf_check.answer(db):
                time.sleep(0.02)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        status, r = plan(chosen=[3, 1])
        assert status == 200 and r["names"]["4"] == "VA1d" and r["kinds"]["VA1d"] == "glomerulus"
        assert asked == [{"prediction": p["id"], "chosen": [1, 3], "run": DOC["finished"]}]
        status, r = plan(chosen=[1])
        assert status == 503 and "that look is no longer there" in r["error"]
    finally:
        stop.set()
        t.join()
    with db.connect() as con:  # (nothing asked is left behind)
        assert con.execute("SELECT COUNT(*) FROM worker_calls").fetchone()[0] == 0
    # one the worker does not get to in time
    with pytest.raises(nmf_check.CallError, match="did not answer"):
        nmf_check.call(db, "nmf_plan", {}, timeout=0.3)
    assert nmf_check.answer(db) is False


def test_merged_saved(main, tensor):  # noqa: F811
    """What the viewer merged of a look is kept with the edited ROIs: which candidates, the names then, and the ROIs of them."""
    p = prediction(main)
    roifile.ImagejRoi.frompoints(np.array(SQUARE, "f4"), name="DM2", z=0).tofile(predictions.roiset(p))
    rois = [{"name": "DM2", "z": 0, "points": SQUARE}, {"name": "DA1", "z": 0, "points": SQUARE, "nmf": 1}, {"name": "DA1", "z": 1, "points": SQUARE, "nmf": 1}]
    merged = {"run": DOC["finished"], "merged": [1, 1], "joint": True, "names": {"1": "DM2", "2": "DL5", "3": None, "4": "DA1"}}
    save = lambda **body: call(main.predictions_edited_save, method="PUT", path_params={"id": p["id"]},  # noqa: E731
                               body={"rois": rois, "base": 0, "name": "Bob", "simplified": 1, **body})
    for bad in ("x", {"run": 1, "merged": [], "names": {}}, {"run": "r", "merged": ["1"], "names": {}}, {"run": "r", "merged": [], "names": {"a": "DM2"}},
                {"run": "r", "merged": [], "names": {"1": ""}}, {"run": "r", "merged": [], "names": []}):
        status, r = save(nmf=bad)
        assert status == 400 and r["error"].startswith("nmf:"), bad
    assert save(rois=[{**rois[0], "nmf": "1"}])[0] == 400 and save(rois=[{**rois[0], "nmf": 0}])[0] == 400
    status, info = save(nmf=merged)
    assert status == 200 and info["nmf"] == {**merged, "merged": [1]} and info["from_nmf"] == [None, 1, 1] and info["rois"] == 3
    v = call(main.predictions_view, path_params={"id": p["id"]})[1]
    assert v["edited"]["nmf"]["merged"] == [1] and v["edited"]["from_nmf"] == [None, 1, 1]
    # (in the RoiSet's order, as the viewer reads them back)
    assert [r.name for r in roifile.ImagejRoi.fromfile(predictions.edited_roiset(p))] == ["DM2", "DA1", "DA1"]
    read = call(main.movie_rois, query={"path": str(predictions.edited_roiset(p))})[1]["rois"]
    assert [(r["name"], r["z"]) for r in read] == [("DM2", 0), ("DA1", 0), ("DA1", 1)]
    # saved without any: nothing of it is said (as before there were looks)
    status, info = save(base=1, rois=rois[:1])
    assert status == 200 and "nmf" not in info and "from_nmf" not in info


def movie(rec, tuning, labels, seed=0):
    """A recording's movie (mocorr.tif) and its trials: each odor twice, 2 time points before it and 4 with it, each
    glomerulus of `labels` answering by `tuning` (glomeruli, odors), over a baseline, with noise."""
    rng = np.random.default_rng(seed)
    frames, trials = [], []
    for k in range(2 * len(ODORS)):
        a, c = ODORS[k // 2]
        v = np.full((6,) + GRID, 1000.0)
        for i in range(len(GLOMS)):
            v[2:, labels == i + 1] += 400 * tuning[i, k // 2]
        frames.append(v + rng.normal(0, 20, v.shape))
        trials.append({"start_frame": 6 * k, "first_odor_frame": 6 * k + 2, "end_frame": 6 * k + 5, "odors": f"{NAMES[a]} @ {c}", "name": NAMES[a],
                       "abbrev": a, "log10_conc": c})
    tifffile.imwrite(rec / "mocorr.tif", np.concatenate(frames).astype(np.uint16), imagej=True, metadata={"axes": "TZYX"})
    (rec / "trial_frames_and_odors.json").write_text(json.dumps(trials))


def test_worker_nmf(main, tensor, monkeypatch):  # noqa: F811
    """The real al-diag-seg (where it is installed: the worker image), with its stand-in segmenter: a fly whose ROIs lack
    a glomerulus, and another on one plane, both in its movie: found, named, and merged in by the names the worker gives."""
    am = pytest.importorskip("alseg.model")
    anmf = pytest.importorskip("alseg.nmf")
    pytest.importorskip("torch")

    class StandIn(am.Options):
        def __init__(self, **kw):
            super().__init__(**{"segmenter": "stand-in", "gpu": False, **kw})

    monkeypatch.setattr(am, "Options", StandIn)
    db = main.db
    f = fits.create(db, "Ann", None, fits.options({"held_out": 1, "atlas": False}))
    fit_worker.run(db, fits.claim(db))
    held = fits.get(db, f["id"])["flies"]["held_out"][0]
    date, n = held.split("_")[1:]
    rec = tensor / f"ann/analysis/{date}/{n}/diagnostics1"
    # the fly's ROIs (which the stand-in segmenter predicts, grown) without DA1, and without DL5 on the middle plane
    labels = np.zeros(GRID, np.uint8)
    kept = []
    for r in roifile.ImagejRoi.fromfile(rec / "RoiSet.zip"):
        (x0, y0), (x1, y1) = r.coordinates().min(0).astype(int), r.coordinates().max(0).astype(int)
        labels[r.z_position - 1, y0:y1 + 1, x0:x1 + 1] = GLOMS.index(r.name) + 1
        if r.name != "DA1" and not (r.name == "DL5" and r.z_position == 2):
            kept.append(r)
    (rec / "RoiSet.zip").unlink()
    for r in kept:
        r.tofile(rec / "RoiSet.zip", name=f"{r.name}-{r.z_position}.roi")
    movie(rec, np.eye(len(GLOMS), len(ODORS)) + 0.05, labels)
    # (the fixture's field is all lobe, and its glomeruli 4.5 um across)
    monkeypatch.setattr(anmf, "hulls", lambda lab: np.ones(lab.shape, bool))
    monkeypatch.setitem(anmf.DEFAULTS, "min_radius_um", 1.0)
    monkeypatch.setitem(anmf.DEFAULTS, "min_area_um2", 8.0)
    monkeypatch.setitem(anmf.DEFAULTS, "min_masks", 2)

    opts = predictions.options({"clean": ["margin", "core"], "clean_params": {"margin": {"gap_um": 0.5}}})
    p, _ = predictions.create(db, f["id"], predictions.locate(f"ann/analysis/{date}/{n}", f"ann/raw/{date}/{n}", config.TENSOR_ROOT, []), "Bob", None, opts=opts)
    fit_worker.run_prediction(db, predictions.claim(db), False)
    d = predictions.pred_dir(p["id"])
    assert predictions.get(db, p["id"])["status"] == "done", (d / "log.txt").read_text()[-3000:]
    names0 = nmf_check.predicted_names(predictions.get(db, p["id"]))
    assert sorted(v for v in names0.values() if v) == ["DL5", "DM2", "VA1d"]

    dm2, dl5 = (next(l for l, v in names0.items() if v == g) for g in ("DM2", "DL5"))

    # (which masks hold two glomeruli is al-diag-seg's to say, by the trials' maps: here, DM2's on the first plane, its right half)
    def cut(masks, fly, return_report=False, **kw):
        out, new = masks.copy(), int(masks.max()) + 1
        ys, xs = np.nonzero(masks[0] == dm2)
        out[0][(masks[0] == dm2) & (np.arange(masks.shape[2])[None, :] > xs.mean())] = new
        return out, {"split": {(dm2, 0): new}, "flagged": {}, "stats": {(dm2, 0): {"corr": 0.2}}}

    monkeypatch.setattr(__import__("alseg.clean_trim").clean_trim, "split", cut)
    assert call(main.predictions_nmf_ask, method="POST", path_params={"id": p["id"]}, body={"name": "Cy"})[0] == 201
    fit_worker.run_nmf(db, nmf_check.claim(db), False, "a fit is using the GPU")
    got = predictions.get(db, p["id"])
    log = (d / "log.txt").read_text()
    assert got["nmf"]["status"] == "done", log[-3000:]
    res = got["nmf"]["result"]
    assert res["candidates"] == 3 and res["kinds"] == {"added": 1, "extended": 1, "split": 1} and res["device"] == "CPU" and res["planes"] == 3
    assert res["time_points"] == 48 and set(res["stages"]) == {"load", "read", "factorize", "name"} and got["nmf"]["progress"] is None
    assert "glomeruli its ROIs miss, looked for in its movies: asked for by Cy" in log and "on the CPU (a fit is using the GPU)" in log
    assert "movies of 1 recordings (diagnostics1): 48 time points of 8 trials" in log and "3 candidates: 1 added, 1 more of an ROI, 1 cut off an ROI" in log
    assert "cleaned as the prediction's ROIs: margin" in log  # (not core: candidates have nothing of Cellpose's to go by)
    files = nmf_check.files(got)
    assert all(files[k].is_file() for k in ("json", "labels", "candidates", "roiset", "inputs")) and not files["prior"].exists()

    found = call(main.predictions_nmf, path_params={"id": p["id"]})[1]["found"]
    by = {c["kind"]: c for c in found["candidates"]}
    # the glomerulus with no ROI: one ROI over its three planes, named as itself (the predicted ROIs' names kept, or all named again)
    a = by["added"]
    assert a["planes"] == [0, 1, 2] and a["name"] == "DA1" and a["of"] is None and a["label"] > max(names0) and len(a["outlines"]) == 3
    assert found["names"]["kept"][str(a["label"])] == "DA1" == found["names"]["joint"][str(a["label"])] and found["kinds"]["DA1"] == "glomerulus"
    assert {l: v for l, v in found["names"]["kept"].items() if int(l) in names0} == {str(l): v for l, v in names0.items()}
    # the part cut off DM2: an ROI of its own, with what is left of DM2 on that plane; DM2's name stays its own where the ROIs' are
    # kept (the part numbered: no glomerulus is left for it), and is one of the two's where all are named again
    c = by["split"]
    assert c["planes"] == [0] and c["of"] == dm2 and c["of_name"] == "DM2" and c["label"] > max(names0) and c["name"] is None and c["corr"] == 0.2
    assert len(c["outlines"]) == 1 and len(c["rest"]) == 1 and c["rest"][0]["z"] == 0 and "rest" not in a
    assert found["names"]["kept"][str(c["label"])] is None
    assert sorted([found["names"]["joint"][str(dm2)], found["names"]["joint"][str(c["label"])]], key=str) == ["DM2", None]
    # the one with no outline on the middle plane: more of its ROI there
    e = by["extended"]
    assert e["planes"] == [1] and e["label"] == dl5 == e["of"] and e["of_name"] == "DL5" and e["name"] == "DL5" and len(e["outlines"]) == 1
    merged, cand = tifffile.imread(files["labels"]), tifffile.imread(files["candidates"])
    truth = lambda g, z: labels[z] == GLOMS.index(g) + 1  # noqa: E731
    iou = lambda x, y: (x & y).sum() / (x | y).sum()  # noqa: E731
    assert all(iou(merged[z] == a["label"], truth("DA1", z)) > 0.5 for z in range(3)) and iou(merged[1] == dl5, truth("DL5", 1)) > 0.5
    pred = tifffile.imread(d / f"{held}_clean.tif")
    assert np.array_equal(merged[cand == 0], pred[cand == 0]) and set(np.unique(cand)) == {0, a["id"], e["id"], c["id"]}
    assert not pred[np.isin(cand, [a["id"], e["id"]])].any() and (pred[cand == c["id"]] == dm2).all()
    # (the part is kept from what is left of DM2 as ROIs are from each other, by the pixels it gives up)
    part, rest = merged[0] == c["label"], merged[0] == dm2
    assert part.sum() > 8 and rest.sum() > 8 and ((cand[0] == c["id"]) & ~part).sum() > 0 and not (ndi.binary_dilation(part) & rest).any()
    # (kept 0.5 um from the predicted ROIs, as those are from each other)
    assert ndi.distance_transform_edt(pred[0] == 0, sampling=(0.5, 0.5))[merged[0] == a["label"]].min() >= 1.0
    # its outlines are the labels', in the pixels of the image; the RoiSet has the candidates alone
    pts = np.array(a["outlines"][0]["points"])
    ys, xs = np.nonzero(merged[0] == a["label"])
    assert abs(pts[:, 0].min() - (xs.min() - 0.5)) < 1e-6 and abs(pts[:, 1].max() - (ys.max() + 0.5)) < 1e-6
    assert sorted((r.name, r.z_position) for r in roifile.ImagejRoi.fromfile(files["roiset"])) == [(str(c["label"]), 1), ("DA1", 1), ("DA1", 2), ("DA1", 3), ("DL5", 2)]
    doc = json.loads(files["json"].read_text())
    assert doc["cleaned"] == ["margin"] and doc["of"] == f"{held}_clean.tif" and doc["movies"]["time_points"] == 48 and doc["asked_for"]["name"] == "Cy"
    assert doc["named"] == f"{held}_pred.tif"  # (the ROIs as the model named them: what all are named again by)
    assert doc["params"]["min_radius_um"] == 1.0 and sorted(doc["glomeruli"]) == sorted(GLOMS) and doc["planes"] == [0, 1, 2]
    with zipfile.ZipFile(__import__("io").BytesIO(b"".join(predictions.stream_zip(p["id"], "top")))) as z:
        zipped = {x.split("/", 1)[1] for x in z.namelist()}
    assert f"{held}_nmf.json" in zipped and f"{held}_nmf_RoiSet.zip" in zipped and nmf_check.INPUTS not in zipped

    # some of them, all named again together: the worker's, from what the look left
    plan = lambda **body: call(main.predictions_nmf_plan, method="POST", path_params={"id": p["id"]}, body=body)  # noqa: E731
    assert plan(chosen=[a["id"], e["id"], c["id"]])[1]["names"] == found["names"]["joint"]
    fits.heartbeat(db, {"host": "h", "gpu": None})
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            if not nmf_check.answer(db):
                time.sleep(0.02)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        status, r = plan(chosen=[e["id"]])
        assert status == 200 and r["names"] == {str(l): v for l, v in names0.items()}, r
        status, r = plan(chosen=[a["id"]])
        assert status == 200 and r["names"][str(a["label"])] == "DA1" and r["names"][str(dl5)] == "DL5" and len(r["names"]) == len(names0) + 1
        # (of a look made again since: not this one's candidates)
        assert "no longer there" in pytest.raises(nmf_check.CallError, nmf_check.call, db, "nmf_plan",
                                                  {"prediction": p["id"], "chosen": [a["id"]], "run": "before"}).value.args[0]
    finally:
        stop.set()
        t.join()

    # the predicted ROIs' own outlines by the same tests, for the viewer to mark those under them
    assert doc["rois"] == found["rois"] and {(r["name"], r["z"]) for r in found["rois"]} == {(v or str(l), z) for l, v in names0.items() for z in range(3)} - {("DL5", 1)}
    assert all(r["eta2"] > 0.9 and r["novelty"] > 0.5 and r["area_um2"] > 20 for r in found["rois"])
    # asked for with nothing factorized, and no masks cut: those tests alone
    assert call(main.predictions_nmf_ask, method="POST", path_params={"id": p["id"]}, body={"name": "Cy", "settings": {"n_free": 0, "split": False}})[0] == 201
    fit_worker.run_nmf(db, nmf_check.claim(db), False)
    got = predictions.get(db, p["id"])
    assert got["nmf"]["status"] == "done" and got["nmf"]["result"]["candidates"] == 0 and got["nmf"]["settings"] == {"n_free": 0, "split": False}
    f2 = call(main.predictions_nmf, path_params={"id": p["id"]})[1]["found"]
    assert f2["candidates"] == [] and f2["params"]["n_free"] == 0 and len(f2["rois"]) == len(found["rois"]) and not nmf_check.files(got)["roiset"].exists()
    # asked for again, it is made again; a fly with no movie fails, said in the prediction's log
    (rec / "mocorr.tif").unlink()
    nmf_check.ask(db, p["id"], "Cy", None)
    fit_worker.run_nmf(db, nmf_check.claim(db), False)
    n2 = predictions.get(db, p["id"])["nmf"]
    assert n2["status"] == "failed" and "no recording with mocorr.tif" in n2["error"] and "no recording with mocorr.tif" in (d / "log.txt").read_text()
