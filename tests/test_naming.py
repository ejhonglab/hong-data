"""Tests of the naming explorer's data (naming.py): what the web server gives of a prediction (the ROIs its names were
given to, the atlas on its fly's planes, the volumes), every pair's cost as the fit worker makes it when asked (with the
real al-diag-seg, where it is installed: the worker image), and the page's own naming under other weights and thresholds
(app/static/naming_solve.js), against SciPy's matching as al-diag-seg uses it and, in the worker image, al-diag-seg itself.
Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/.

tests/naming_solve_fixture.json (what tests/naming_solve.test.mjs checks the page's naming against) is made by
``python -m tests.test_naming [<another file>]`` (`make_fixture`)."""
from __future__ import annotations

import asyncio
import json
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.optimize import linear_sum_assignment

from app import config, fit_worker, fits, naming, nmf_check, predictions
from tests.test_fits import GLOMS, GRID, call, data, request, tensor  # noqa: F401 (fixtures)
from tests.test_predictions import FLY, RAW, done_fit
from tests.test_tensor import main, root  # noqa: F401 (fixtures)

FLYNAME = "ann_2024-01-01_1"
FIXTURE = Path(__file__).with_name("naming_solve_fixture.json")
SHAPE = (3, 6, 8)


# ------------------------------------------- the page's naming, in Python

def match(cost: np.ndarray, max_cost: float) -> np.ndarray:
    """Each row's column, or -1: al-diag-seg's `identity._solve` (SciPy's matching, with a column per row that costs ``max_cost``)."""
    n_rows, n_cols = cost.shape
    rows, cols = linear_sum_assignment(np.hstack([cost, np.full((n_rows, n_rows), max_cost)]))
    out = np.full(n_rows, -1)
    for i, j in zip(rows, cols):
        if j < n_cols:
            out[i] = j
    return out


def resolve(doc: dict, weights: dict, max_cost: float) -> dict:
    """What naming_solve.js's ``solve`` gives of a prediction's costs (`naming.make_costs`' doc) under ``weights`` and
    ``max_cost``, worked out as al-diag-seg's `identity.assign_identities` does: ``names`` (per ROI; None: unnamed),
    ``costs`` (of each name; None) and ``settled`` (whether the names stopped changing within the rounds)."""
    parts = {k: np.array([[np.nan if v is None else v for v in r] for r in doc["terms"][k]], float) for k in naming.TERMS}
    n_rows, n_cols = parts["tuning"].shape
    atlas = doc["weights"]["atlas"] is not None
    w = {k: weights.get(k) or 0.0 for k in naming.TERMS}
    active = w["tuning"] + w["position"] + w["shape"] + w["layout"] + (w["atlas"] if atlas else 0)
    base = w["tuning"] * parts["tuning"] + w["position"] * parts["position"] + w["shape"] * parts["shape"]
    if atlas and w["atlas"]:
        base = base + w["atlas"] * np.nan_to_num(parts["atlas"])
    base = base / active
    wl = w["layout"] / active
    roi, exp, sd = np.array(doc["roi_pos"], float), np.array(doc["exp_pos"], float), np.array(doc["pos_sd"], float)

    def layout(names):
        out = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            terms = []
            for k in doc["neighbors"][i]:
                if names[k] < 0:
                    continue
                scale = np.sqrt(sd**2 + sd[names[k]] ** 2) + 1e-8
                terms.append(np.minimum(np.linalg.norm(exp - exp[names[k]] - (roi[i] - roi[k]), axis=1) / scale, naming.CAP) / naming.CAP)
            if terms:
                out[i] = np.mean(terms, 0)
        return out

    names, settled = match(base, max_cost), not wl
    for _ in range(doc["rounds"] if wl else 0):
        new = match(base + wl * layout(names), max_cost)
        if np.array_equal(new, names):
            settled = True
            break
        names = new
    cost = base + (wl * layout(names) if wl else 0)
    return {"names": [doc["names"][j] if j >= 0 else None for j in names],
            "costs": [float(cost[i, j]) if j >= 0 else None for i, j in enumerate(names)], "settled": bool(settled)}


def random_doc(rng, n_rows: int, n_cols: int, n_ref: int, atlas: bool, coarse: bool) -> dict:
    """Costs as `naming.make_costs` writes them, of made-up ROIs and glomeruli: the last ``n_ref`` glomeruli of a reference
    atlas alone (tuning and shape fixed, as al-diag-seg has them), terms in steps of a quarter if ``coarse`` (so that many
    pairs cost the same, and which is matched is the algorithm's to say)."""
    step = (lambda a: np.round(a * 4) / 4) if coarse else (lambda a: a)
    roi, exp = rng.uniform(0, 40, (n_rows, 3)), rng.uniform(0, 40, (n_cols, 3))
    sd = rng.uniform(3, 6, n_cols)
    dist = np.linalg.norm(roi[:, None] - exp[None], axis=2) / (sd + 1e-8)
    parts = {"tuning": step(rng.uniform(0, 1.2, (n_rows, n_cols))), "position": np.minimum(dist, naming.CAP) / naming.CAP,
             "shape": step(rng.uniform(0, 1, (n_rows, n_cols))), "atlas": step(rng.uniform(0.5, 1, (n_rows, n_cols))),
             "layout": np.zeros((n_rows, n_cols))}
    if n_ref:
        parts["tuning"][:, -n_ref:] = parts["shape"][:, -n_ref:] = 0.5
    if not atlas:
        parts["atlas"][:] = np.nan
    elif n_cols > 2:
        parts["atlas"][:, 1] = np.nan  # (a glomerulus the atlas has no map of)
    near = np.linalg.norm(roi[:, None] - roi[None], axis=2)
    doc = {"labels": list(range(1, n_rows + 1)), "names": [f"G{j}" for j in range(n_cols)], "reference": [j >= n_cols - n_ref for j in range(n_cols)],
           "weights": {"tuning": 1.0, "position": 0.5, "shape": 0.3 if atlas else 0.0, "atlas": 1.0 if atlas else None, "layout": 0.5 if atlas else 0.0},
           "max_cost": 0.6, "rounds": 3, "terms": {k: [[None if np.isnan(v) else float(v) for v in r] for r in a] for k, a in parts.items()},
           "roi_pos": roi.tolist(), "exp_pos": exp.tolist(), "pos_sd": sd.tolist(),
           "neighbors": [[int(k) for k in np.argsort(near[i])[1:4] if near[i, k] < 25] for i in range(n_rows)]}
    doc["assigned"] = resolve(doc, doc["weights"], doc["max_cost"])["names"]
    return doc


def make_fixture() -> dict:
    """What tests/naming_solve.test.mjs checks naming_solve.js against: ``matchings`` (costs, with ties, and what SciPy
    matches) and ``docs`` (made-up costs, each with the names they give under several weights and thresholds)."""
    rng = np.random.default_rng(7)
    matchings = []
    for _ in range(60):
        n_rows, n_cols = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        cost = rng.integers(0, 4, (n_rows, n_cols)) / 4 if rng.random() < 0.7 else rng.uniform(0, 1, (n_rows, n_cols))
        max_cost = float(rng.choice([0.25, 0.5, 0.6, 0.75]))
        matchings.append({"cost": cost.tolist(), "max_cost": max_cost, "expected": match(cost, max_cost).tolist()})
    docs = []
    for n_rows, n_cols, n_ref, atlas, coarse in ((12, 10, 3, True, False), (14, 9, 4, True, True), (6, 12, 0, True, False), (9, 7, 0, False, False),
                                                 (10, 10, 5, True, True)):
        doc = random_doc(rng, n_rows, n_cols, n_ref, atlas, coarse)
        settings = [(doc["weights"], m) for m in (0.4, 0.6, 0.75, 1.0)]
        if atlas:
            settings += [({**doc["weights"], "atlas": a}, 0.6) for a in (0.0, 2.0)] + [({**doc["weights"], "layout": 0.0}, 0.6)]
        settings += [({k: (float(rng.uniform(0, 2)) if doc["weights"][k] is not None else None) for k in naming.TERMS}, float(rng.uniform(0.3, 0.9)))
                     for _ in range(6)]
        docs.append({"doc": doc, "expected": [{"weights": w, "max_cost": m, **resolve(doc, w, m)} for w, m in settings]})
    return {"matchings": matchings, "docs": docs}


def test_solver_fixture():
    """The fixture the page's naming is checked against is what SciPy's matching gives now, as al-diag-seg uses it (so a SciPy
    that breaks ties otherwise shows here); and it has what it is meant to test: ties, glomeruli of a reference alone, a
    matching that does not settle."""
    fx = json.loads(FIXTURE.read_text())
    for m in fx["matchings"]:
        assert match(np.array(m["cost"]), m["max_cost"]).tolist() == m["expected"]
    unsettled = 0
    for d in fx["docs"]:
        assert resolve(d["doc"], d["doc"]["weights"], d["doc"]["max_cost"])["names"] == d["doc"]["assigned"]
        for e in d["expected"]:
            got = resolve(d["doc"], e["weights"], e["max_cost"])
            assert got["names"] == e["names"] and got["settled"] == e["settled"]
            unsettled += not e["settled"]
    assert unsettled and any(None in e["names"] for d in fx["docs"] for e in d["expected"])


def test_solver_js():
    """tests/naming_solve.test.mjs: the page's naming (app/static/naming_solve.js), where there is a node to run it."""
    node = shutil.which("node")
    if not node:
        pytest.skip("no node")
    done = subprocess.run([node, str(Path(__file__).with_name("naming_solve.test.mjs"))], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


# ------------------------------------------------- what the web server gives

def raw(endpoint, **kw):
    response = asyncio.run(endpoint(request(**kw)))
    return response.status_code, response.body


def header(body: bytes) -> tuple[dict, int]:
    """The header of a prediction's volumes (`naming.volumes`), and where the volumes start."""
    assert body[:4] == naming.MAGIC
    n = struct.unpack("<I", body[4:8])[0]
    return json.loads(body[8:8 + n]), 8 + n


def labels() -> np.ndarray:
    """Two ROIs: 1 on the first two planes, 2 on the last."""
    lab = np.zeros(SHAPE, np.uint16)
    lab[:2, 1:4, 1:4] = 1
    lab[2, 2:5, 4:7] = 2
    return lab


def prediction(main) -> dict:  # noqa: F811
    """A prediction done, its folder with what one keeps of its ROIs' names: its labels, the atlas on its planes (DM2 where
    ROI 1 is, DL5 where ROI 2 is, VM7d of the reference atlas alone, nowhere), and what the names cost."""
    f = done_fit(main.db)
    mdir = fits.fit_dir(f["id"]) / "model"
    (mdir / "model.json").write_text(json.dumps({"atlas": True, "options": {"max_cost": 0.6}}))
    (mdir / "template.json").write_text(json.dumps({"DM2": {"n": 5, "centroid_sd": 4.5, "tunings": [[1, 2]]}, "DL5": {"n": 3, "centroid_sd": 6.0}}))
    np.savez(mdir / "atlas.npz", names=np.array(["DL5", "DM2", "VM7d"]), counts=np.array([4.0, 6.0, 1.0]), prob=np.zeros((3, 1, 1, 1), np.float16))
    p, _ = predictions.create(main.db, f["id"], predictions.locate(FLY, RAW, config.TENSOR_ROOT, []), "Ann", None)
    d = predictions.pred_dir(p["id"])
    d.mkdir(parents=True)
    lab = labels()
    tifffile.imwrite(d / f"{FLYNAME}_pred.tif", lab, imagej=True, metadata={"axes": "ZYX"})
    avg = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    tifffile.imwrite(d / predictions.AVERAGE, avg, imagej=True, metadata={"axes": "ZYX"})
    prior = np.zeros((3, *SHAPE), np.float16)
    prior[0][lab == 2], prior[1][lab == 1] = 0.25, 0.5
    np.savez_compressed(d / f"{FLYNAME}_atlas_prior.npz", prior=prior, names=np.array(["DL5", "DM2", "VM7d"]))
    terms = {"weights": {"tuning": 1.0, "position": 0.5, "shape": 0.3, "atlas": 1.0, "layout": 0.5}, "weight_sum": 3.3, "max_cost": 0.6, "unknown_cost": 0.5,
             "rois": {"1": {"assigned": "DM2", "cost": 0.3, "confidence": 0.4, "candidates": [
                 {"name": "DM2", "cost": 0.3, "tuning": 0.1, "position": 0.2, "shape": 0.3, "atlas": 0.5, "layout": 0.0, "reference": False, "taken_by": None},
                 {"name": "VM7d", "cost": 0.7, "tuning": 0.5, "position": 1.0, "shape": 0.5, "atlas": 1.0, "layout": 0.0, "reference": True, "taken_by": None}]},
                      "2": {"assigned": None, "cost": 0.6, "confidence": 0.1, "candidates": [
                          {"name": "DL5", "cost": 0.65, "tuning": 0.6, "position": 0.5, "shape": 0.4, "atlas": 0.75, "layout": 0.2, "reference": False, "taken_by": None}]}}}
    (d / f"{FLYNAME}_identity_terms.json").write_text(json.dumps(terms))
    (d / predictions.PROVENANCE).write_text(json.dumps({"fly": {"spacing_um": [10.0, 0.5, 0.5]}}))
    predictions.update(main.db, p["id"], status="done", finished_at=predictions.now(),
                       result={"final": "pred", "n": 2, "named": ["DM2"], "scores": None, "device": "CPU"})
    return predictions.get(main.db, p["id"])


def test_view_and_volumes(main, tensor):  # noqa: F811
    """What the explorer is given of a prediction: its ROIs, the atlas on its planes, the model's glomeruli, and the volumes as bytes."""
    p = prediction(main)
    d = predictions.pred_dir(p["id"])
    view = lambda i=p["id"]: call(main.predictions_naming, path_params={"id": i})  # noqa: E731
    assert view(p["id"] + 1)[0] == 404
    status, v = view()
    assert status == 200 and (v["fly"], v["kind"], v["final"], v["shape"], v["spacing"]) == (FLYNAME, "pred", "pred", list(SHAPE), [10.0, 0.5, 0.5])
    assert v["rois"] == [{"label": 1, "name": "DM2", "voxels": 18, "planes": [0, 1], "centroid": [0.5, 2.0, 2.0]},
                         {"label": 2, "name": None, "voxels": 9, "planes": [2], "centroid": [2.0, 3.0, 5.0]}]
    # the atlas on its planes: how sure each map is, on which planes, and where its middle is (none of a glomerulus that is nowhere)
    dl5, dm2, vm7d = v["maps"]
    assert (dm2["name"], dm2["peak"], dm2["planes"], dm2["centre"], dm2["at"][0]) == ("DM2", 0.5, [0.5, 0.5, 0.0], [0.5, 2.0, 2.0], 0)
    assert (dl5["name"], dl5["peak"], dl5["planes"], dl5["centre"], dl5["mass"]) == ("DL5", 0.25, [0.0, 0.0, 0.25], [2.0, 3.0, 5.0], 2.25)
    assert (vm7d["peak"], vm7d["centre"]) == (0.0, None)
    # the model's glomeruli: the flies behind each map, and its position's SD; one of the reference atlas alone has neither tuning nor SD
    assert v["glomeruli"] == {"DL5": {"flies": 4.0, "trained": 3, "sd": 6.0}, "DM2": {"flies": 6.0, "trained": 5, "sd": 4.5},
                              "VM7d": {"flies": 1.0, "trained": None, "sd": None}}
    assert v["terms"]["rois"]["2"]["candidates"][0]["name"] == "DL5" and v["terms"]["max_cost"] == 0.6
    assert v["costs"] == {"made": None} and v["model"] == {"has_atlas": True, "max_cost": 0.6}

    # the volumes: a header, then the average image (each plane to 0-255), the labels and each glomerulus' probability (255: 1)
    assert raw(main.predictions_naming_volumes, path_params={"id": p["id"] + 1})[0] == 404
    status, body = raw(main.predictions_naming_volumes, path_params={"id": p["id"]})
    (head, at), size = header(body), int(np.prod(SHAPE))
    assert status == 200 and head == {"shape": list(SHAPE), "maps": ["DL5", "DM2", "VM7d"]} and len(body) == at + size * (1 + 2 + 3)
    avg = np.frombuffer(body, np.uint8, size, at).reshape(SHAPE)
    assert avg[0].min() == 0 and avg[2].max() == 255 and (np.diff(avg[1].ravel().astype(int)) >= 0).all()
    assert np.array_equal(np.frombuffer(body, "<u2", size, at + size).reshape(SHAPE), labels())
    prior = np.frombuffer(body, np.uint8, 3 * size, at + 3 * size).reshape(3, *SHAPE)
    assert np.array_equal(prior[1], np.where(labels() == 1, 128, 0)) and np.array_equal(prior[0], np.where(labels() == 2, 64, 0)) and not prior[2].any()
    # gzipped for a browser that takes it
    gz = asyncio.run(main.predictions_naming_volumes(request(path_params={"id": p["id"]}, headers=[(b"accept-encoding", b"gzip, br")])))
    assert gz.headers["content-encoding"] == "gzip" and __import__("gzip").decompress(gz.body) == body

    # cleaned ROIs keep the names of those they were cleaned from: those are the ROIs shown (split ones, where there are some)
    predictions.update(main.db, p["id"], result={**p["result"], "final": "clean"})
    assert view()[1]["kind"] == "pred" and view()[1]["final"] == "clean"
    tifffile.imwrite(d / f"{FLYNAME}_split.tif", labels(), imagej=True, metadata={"axes": "ZYX"})
    assert view()[1]["kind"] == "split"
    predictions.update(main.db, p["id"], result=p["result"])
    # an atlas of other planes than the ROIs (or none kept): no maps; no labels: nothing to show
    np.savez_compressed(d / f"{FLYNAME}_atlas_prior.npz", prior=np.zeros((3, 2, 6, 8), np.float16), names=np.array(["DL5", "DM2", "VM7d"]))
    body = raw(main.predictions_naming_volumes, path_params={"id": p["id"]})[1]
    assert view()[1]["maps"] == [] and header(body)[0]["maps"] == [] and len(body) == header(body)[1] + size * 3
    (d / f"{FLYNAME}_identity_terms.json").unlink()
    assert view()[1]["terms"] == {"rois": None} and view()[1]["rois"][0]["name"] is None
    # (the fit deleted: what its model says of its glomeruli is not known)
    shutil.rmtree(fits.fit_dir(p["fit_id"]) / "model")
    assert view()[1]["glomeruli"] == {} and view()[1]["model"] == {"has_atlas": None, "max_cost": None}
    (d / f"{FLYNAME}_pred.tif").unlink()
    status, r = view()
    assert status == 409 and "predict it again" in r["error"] and raw(main.predictions_naming_volumes, path_params={"id": p["id"]})[0] == 409


def test_costs_api(main, tensor, monkeypatch):  # noqa: F811
    """Every pair's cost: not there until asked for, made by the fit worker when it is, kept, and out of date once the ROIs change."""
    db = main.db
    p = prediction(main)
    f = naming.files(p)
    costs = lambda method="GET", i=p["id"]: call(main.predictions_naming_costs, method=method, path_params={"id": i})  # noqa: E731
    assert costs(i=p["id"] + 1)[0] == 404
    assert costs() == (200, {"made": None})
    status, r = costs("POST")
    assert status == 503 and "the fit worker is not running" in r["error"]
    fits.heartbeat(db, {"host": "h", "gpu": None})
    asked = []

    def make(db_, args):
        asked.append(args)
        if len(asked) == 2:
            raise ValueError("the atlas on its fly's planes was not kept with it")
        f["costs"].write_text(json.dumps({"format": naming.FORMAT, "made": "2026-11-23T12:00:00-08:00", "labels": [1, 2],
                                          "of": {"labels": f["labels"].name, "stat": naming._stat(f["labels"])}}))
        return {"made": "2026-11-23T12:00:00-08:00"}

    monkeypatch.setitem(nmf_check.CALLS, naming.CALL, make)
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            if not nmf_check.answer(db):
                time.sleep(0.02)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        status, doc = costs("POST")
        assert status == 200 and doc["labels"] == [1, 2] and asked == [{"prediction": p["id"]}]
        # kept: given again without the worker, and said to be there
        assert costs() == (200, doc) and costs("POST") == (200, doc) and len(asked) == 1
        assert call(main.predictions_naming, path_params={"id": p["id"]})[1]["costs"] == {"made": "2026-11-23T12:00:00-08:00"}
        # its ROIs changed since (refined again): out of date, and made again when asked for; what the worker says, if it cannot
        time.sleep(0.01)
        tifffile.imwrite(f["labels"], np.ones(SHAPE, np.uint16), imagej=True, metadata={"axes": "ZYX"})
        assert costs()[1] == {"made": None} and call(main.predictions_naming, path_params={"id": p["id"]})[1]["costs"] == {"made": None}
        status, r = costs("POST")
        assert status == 503 and "the atlas on its fly's planes was not kept" in r["error"]
        assert costs("POST")[0] == 200 and len(asked) == 3
        # one of another format (an older version's): made again
        f["costs"].write_text(json.dumps({**json.loads(f["costs"].read_text()), "format": 0}))
        assert costs() == (200, {"made": None})
    finally:
        stop.set()
        t.join()
    # in a prediction's download; gone when it is predicted again (its folder is made anew)
    assert naming.COSTS in f["costs"].name and f["costs"].parent == predictions.pred_dir(p["id"])
    assert naming.CALL in nmf_check.CALLS and fit_worker.naming is naming


# -------------------------------------------- the fit worker, with al-diag-seg

def test_worker_costs(main, tensor, monkeypatch):  # noqa: F811
    """The real al-diag-seg (where it is installed: the worker image), with its stand-in segmenter: every pair's cost of a
    prediction, without an atlas and with one on its fly's planes; and the names they give under other weights and
    thresholds (as the page works them out) against al-diag-seg's own."""
    am = pytest.importorskip("alseg.model")

    class StandIn(am.Options):
        def __init__(self, **kw):
            super().__init__(**{"segmenter": "stand-in", "gpu": False, **kw})

    monkeypatch.setattr(am, "Options", StandIn)
    db = main.db
    f = fits.create(db, "Ann", None, fits.options({"held_out": 1, "atlas": False}))
    fit_worker.run(db, fits.claim(db))
    held = fits.get(db, f["id"])["flies"]["held_out"][0]
    date, n = held.split("_")[1:]
    p, _ = predictions.create(db, f["id"], predictions.locate(f"ann/analysis/{date}/{n}", f"ann/raw/{date}/{n}", config.TENSOR_ROOT, []), "Bob", None)
    fit_worker.run_prediction(db, predictions.claim(db), False)
    p = predictions.get(db, p["id"])
    d = predictions.pred_dir(p["id"])
    assert p["status"] == "done", (d / "log.txt").read_text()[-3000:]
    pred = tifffile.imread(d / f"{held}_pred.tif")

    # a model with no atlas: named by tuning and position alone
    made = naming.make_costs(db, {"prediction": p["id"]})
    doc = naming.costs(db, p["id"])
    assert doc["made"] == made["made"] and doc["exact"] and doc["why"] is None and doc["same"] and doc["kind"] == "pred" and doc["responses"] == "tensor"
    assert doc["weights"]["atlas"] is None and doc["weights"]["shape"] == 0 and doc["weights"]["layout"] == 0 and doc["max_cost"] == 0.6
    assert doc["labels"] == [int(l) for l in np.unique(pred) if l] and sorted(doc["names"]) == sorted(GLOMS) and not any(doc["reference"])
    assert all(v is None for r in doc["terms"]["atlas"] for v in r) and all(v is not None for r in doc["terms"]["tuning"] for v in r)
    assert doc["assigned"] == doc["predicted"] and sorted(x for x in doc["assigned"] if x) == sorted(p["result"]["named"])
    assert resolve(doc, doc["weights"], doc["max_cost"])["names"] == doc["assigned"]
    assert np.array(doc["tuning"]["rois"]).shape == (len(doc["labels"]), len(doc["odors"])) and len(doc["tuning"]["glomeruli"]) == len(GLOMS)
    assert call(main.predictions_naming_costs, path_params={"id": p["id"]})[1]["assigned"] == doc["assigned"]
    assert f"{held}_naming_costs.json" in (d / "README.txt").read_text()

    # with an atlas on its planes (made up here: each glomerulus where the ROI of that name is, half sure, and one of a
    # reference atlas alone over the first ROI): the atlas term, the layout term, and a glomerulus with no tuning
    mdir = fits.fit_dir(f["id"]) / "model"
    (mdir / "model.json").write_text(json.dumps({**json.loads((mdir / "model.json").read_text()), "atlas": True}))
    names = sorted(GLOMS) + ["VM7d"]
    prior = np.zeros((len(names), *GRID), np.float16)
    for l, name in zip(doc["labels"], doc["assigned"]):
        if name:
            prior[names.index(name)][pred == l] = 0.5
    prior[-1][pred == doc["labels"][0]] = 1.0
    np.savez_compressed(d / f"{held}_atlas_prior.npz", prior=prior, names=np.array(names))
    assert naming.costs(db, p["id"]) is not None  # (still of its ROIs: only another model would name them otherwise)
    naming.make_costs(db, {"prediction": p["id"]})
    doc = naming.costs(db, p["id"])
    assert doc["exact"], doc["why"]
    assert doc["names"][-1] == "VM7d" and doc["reference"] == [False] * len(GLOMS) + [True] and doc["weights"]["atlas"] == 1.0 and doc["weights"]["layout"] == 0.5
    ref = doc["names"].index("VM7d")
    assert all(r[ref] == doc["unknown_cost"] for r in doc["terms"]["tuning"]) and doc["tuning"]["glomeruli"][ref] is None
    assert doc["terms"]["atlas"][0][ref] == 0.0 and all(len(x) == 3 for x in doc["roi_pos"] + doc["exp_pos"]) and len(doc["pos_sd"]) == len(names)
    assert resolve(doc, doc["weights"], doc["max_cost"])["names"] == doc["assigned"]

    # other weights, and other most a name may cost: the names the page would give are al-diag-seg's own
    model = am.Model.load(mdir, atlas=False)
    fly, source = naming._fly(db, p, model)
    assert source == "tensor"
    rng = np.random.default_rng(3)
    settings = [({}, 0.3), ({}, 0.9), ({"w_atlas": 0.0}, 0.6), ({"w_atlas": 2.0, "w_layout": 0.0}, 0.5), ({"w_tuning": 0.0, "w_position": 2.0}, 0.45)]
    settings += [({f"w_{k}": float(rng.uniform(0, 2)) for k in naming.TERMS}, float(rng.uniform(0.2, 0.9))) for _ in range(8)]
    for kw, max_cost in settings:
        model.options.max_cost = max_cost
        ids, terms = model.name(fly, pred, prior.astype(np.float32), names, **kw)
        got = resolve(doc, terms["weights"], max_cost)
        assert got["names"] == [ids[l][0] for l in doc["labels"]], (kw, max_cost)
        assert all(a is None or abs(a - ids[l][1]) < 1e-9 for a, l in zip(got["costs"], doc["labels"]))

    # what it cannot name: a prediction whose atlas was not kept, or whose fit is gone
    (d / f"{held}_atlas_prior.npz").unlink()
    with pytest.raises(ValueError, match="was not kept"):
        naming.make_costs(db, {"prediction": p["id"]})
    fits.delete(db, f["id"])
    with pytest.raises(ValueError, match="the fit was deleted"):
        naming.make_costs(db, {"prediction": p["id"]})
    with pytest.raises(ValueError, match="no such prediction"):
        naming.make_costs(db, {"prediction": p["id"] + 1})


if __name__ == "__main__":  # (to another file, if one is given)
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE
    out.write_text(json.dumps(make_fixture()))
    print(f"wrote {out}")
