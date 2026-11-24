"""How a fit's model did on each glomerulus, over an experimenter's flies it predicted (by_glomerulus.py): the sums the web
server makes of the predictions' ROIs matched to their flies' own, when those are out of date, and what it asks of the
fit worker; and, where al-diag-seg is installed (the worker's image), the matches themselves.
"""
from __future__ import annotations

import csv
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from app import by_glomerulus as bg
from app import config, fit_worker, fits, nmf_check, predictions
from tests.test_fits import call, data, tensor  # noqa: F401 (fixtures)
from tests.test_predictions import done_fit
from tests.test_tensor import main, root  # noqa: F401 (fixtures)


def predicted(db, fit: dict, fly: str, tmp: Path, scored: bool = True, final: str = "pred") -> dict:
    """A prediction done of a fly (named as tensor's are: whose it is, its date and number), its folder with what the
    sums look at: its ROIs' labels and names (of each kind up to its final one) and its provenance (the fly's own ROIs)."""
    p, _ = predictions.create(db, fit["id"], {"dir": str(tmp / fly), "raw_dir": None, "name": fly}, "Ann", None)
    d = predictions.pred_dir(p["id"])
    d.mkdir(parents=True)
    roiset = tmp / f"{fly}_RoiSet.zip"
    roiset.write_bytes(b"its own ROIs")
    for kind in dict.fromkeys(("pred", final)):
        (d / f"{fly}_{kind}.tif").write_bytes(b"labels")
        (d / f"{fly}_identities_{kind}.csv").write_text("label,assigned,cost,confidence\n")
    (d / predictions.PROVENANCE).write_text(json.dumps({"fly": {"roiset": str(roiset) if scored else None}}))
    predictions.update(db, p["id"], status="done", finished_at=predictions.now(),
                       result={"final": final, "n": 3, "named": [], "scores": {"recall": 0.5} if scored else None, "device": "CPU"})
    return predictions.get(db, p["id"])


def matched(p: dict, kinds: dict, **more) -> None:
    """Keeps a prediction's ROIs matched to its fly's own, as the fit worker would: ``kinds``, {kind: (manual, predicted)},
    manual as (name, IoU, the ROI on it, its name) and predicted as (label, name, the fly's own it is on, IoU, over)."""
    doc = {"format": bg.FORMAT, "prediction": p["id"], "fly": p["fly"], "made": predictions.now(), "scoring": "partly drawn",
           "detect_iou": 0.5, "name_iou": 0.3, "of": bg._of(p), "changed": False,
           "kinds": {k: {"manual": [dict(zip(("name", "iou", "roi", "as"), m)) for m in man],
                         "predicted": [dict(zip(("label", "name", "manual", "iou", "over"), q)) for q in pre]} for k, (man, pre) in kinds.items()}, **more}
    bg.doc_path(p).write_text(json.dumps(doc))


def model(fit: dict) -> None:
    """What the fit's model says of its glomeruli: DM2 and DL5 of training flies, VM7d of the reference atlas alone."""
    mdir = fits.fit_dir(fit["id"]) / "model"
    (mdir / "model.json").write_text(json.dumps({"atlas": True, "options": {"max_cost": 0.6}}))
    (mdir / "template.json").write_text(json.dumps({"DM2": {"n": 5, "centroid_sd": 4.5}, "DL5": {"n": 3, "centroid_sd": 6.0}}))
    np.savez(mdir / "atlas.npz", names=np.array(["DL5", "DM2", "VM7d"]), counts=np.array([4.0, 6.0, 1.0]), prob=np.zeros((3, 1, 1, 1), np.float16))


def test_experimenter():
    assert bg.experimenter({"fly": "Sam_2024-05-24_1"}) == "Sam" and bg.experimenter({"fly": "Ann_B_2024-05-24_12"}) == "Ann_B"
    assert bg.experimenter({"fly": "flies_1"}) == bg.OTHER and bg.experimenter({"fly": None}) == bg.OTHER


def test_summary(main, tensor, tmp_path):  # noqa: F811
    """The sums over a fit's predictions of one experimenter's flies with ROIs of their own: per name the experimenter drew."""
    db = main.db
    assert call(main.fits_by_glomerulus)[1] == {"choices": [], "fit": None, "experimenter": None, "flies": "new", "rois": "final", "worker": False}
    fit = done_fit(db)  # (trained on ann_2024-01-01_2 and _3, ann_2024-01-01_1 held out)
    model(fit)
    held = predicted(db, fit, "ann_2024-01-01_1", tmp_path, final="clean")
    trained = predicted(db, fit, "ann_2024-01-01_2", tmp_path)
    new = predicted(db, fit, "ann_2024-02-02_1", tmp_path)
    predicted(db, fit, "ann_2024-02-02_2", tmp_path, scored=False)
    bob = predicted(db, fit, "bob_2024-03-03_1", tmp_path)
    # the fits with something to sum, and whose flies: those with ROIs of their own, and of them those not trained on
    status, s = call(main.fits_by_glomerulus)
    assert status == 200 and s["choices"] == [{"id": fit["id"], "by": "Ann", "deleted": False, "experimenters": [
        {"name": "ann", "predicted": 4, "scored": 3, "new": 2}, {"name": "bob", "predicted": 1, "scored": 1, "new": 1}]}]
    # nothing matched yet: the predictions to match (of those chosen: not the one of a fly it was trained on)
    assert (s["fit"], s["experimenter"]) == (fit["id"], "ann") and s["glomeruli"] == [] and s["detect_iou"] is None
    assert s["counts"] == {"scored": 3, "trained": 1, "held_out": 1, "chosen": 2, "ready": 0, "missing": [held["id"], new["id"]], "changed": 0, "kinds": {}}
    assert call(main.fits_by_glomerulus, query={"flies": "all"})[1]["counts"]["missing"] == [held["id"], trained["id"], new["id"]]

    # the held-out fly (its ROIs cleaned: those are summed): DM2 found and named right; DL5 found, its ROI named VM7d; DA1 with an ROI
    # barely on it, named DA1; VA1d untouched. The model's own ROIs of it (before cleaning) had DM2 unnamed.
    clean = ([("DM2", 0.8, 1, "DM2"), ("DL5", 0.6, 2, "VM7d"), ("DA1", 0.2, 3, "DA1"), ("VA1d", 0.0, None, None)],
             [(1, "DM2", "DM2", 0.8, 1.0), (2, "VM7d", "DL5", 0.6, None), (3, "DA1", "DA1", 0.2, 0.4), (4, "DL5", None, 0.0, 0.0), (5, None, None, 0.0, None)])
    own = ([("DM2", 0.7, 1, None), ("DL5", 0.6, 2, "VM7d"), ("DA1", 0.2, 3, "DA1"), ("VA1d", 0.0, None, None)],
           [(1, None, "DM2", 0.7, None), (2, "VM7d", "DL5", 0.6, None), (3, "DA1", "DA1", 0.2, 0.4), (4, "DL5", None, 0.0, 0.0)])
    matched(held, {"pred": own, "clean": clean})
    # the other fly: DM2 with an ROI on it left unnamed, its name on an ROI over it on other planes; DL5 named right; VA1d (not
    # drawn there) and DM2/DL5 (which the model has not) given nowhere
    matched(new, {"pred": ([("DM2", 0.4, 7, None), ("DL5", 0.5, 8, "DL5"), ("DM2/DL5", 0.35, 8, "DL5")],
                           [(7, None, "DM2", 0.4, None), (8, "DL5", "DL5", 0.5, 1.0), (9, "DM2", None, 0.0, 0.9), (10, "VA1d", "DL5", 0.1, None)])}, changed=True)
    s = call(main.fits_by_glomerulus, query={"fit": str(fit["id"]), "who": "ann"})[1]
    assert s["counts"] == {"scored": 3, "trained": 1, "held_out": 1, "chosen": 2, "ready": 2, "missing": [], "changed": 1, "kinds": {"clean": 1, "pred": 1}}
    assert (s["detect_iou"], s["name_iou"], s["failed"], s["model"]) == (0.5, 0.3, [], {"has_atlas": True, "max_cost": 0.6})
    rows = {r["name"]: r for r in s["glomeruli"]}
    assert list(rows) == ["DA1", "DL5", "DM2", "DM2/DL5", "VA1d"] and [r["name"] for r in s["only_model"]] == ["VM7d"]
    count = lambda r: {k: v for k, v in r.items() if k not in ("name", "flies", "model", "site", "dice") and v}  # noqa: E731
    assert count(rows["DM2"]) == {"drawn": 2, "found": 1, "named": 1, "unnamed": 1, "moved": 1, "given": 2, "given_right": 1, "given_stacked": 1}
    assert rows["DM2"]["dice"] == pytest.approx(2 * 0.8 / 1.8) and rows["DM2"]["model"] == {"trained": 5, "atlas": True}
    assert count(rows["DL5"]) == {"drawn": 2, "found": 2, "named": 1, "other": {"VM7d": 1}, "moved": 1, "given": 2, "given_right": 1, "given_elsewhere": 1}
    assert rows["DL5"]["dice"] == pytest.approx((2 * 0.6 / 1.6 + 2 * 0.5 / 1.5) / 2)
    assert count(rows["DA1"]) == {"drawn": 1, "none": 1, "given": 1, "given_weak": 1} and rows["DA1"]["model"] == {"trained": None, "atlas": False}
    assert count(rows["VA1d"]) == {"drawn": 1, "none": 1, "untouched": 1, "given": 1, "given_undrawn": 1}
    assert count(rows["DM2/DL5"]) == {"drawn": 1, "other": {"DL5": 1}}
    assert count(s["only_model"][0]) == {"given": 1, "given_wrong": {"DL5": 1}} and s["only_model"][0]["model"] == {"trained": None, "atlas": True}
    # each fly of a glomerulus: what became of it there, the ROI, and where the model gave the name instead
    assert rows["DM2"]["flies"] == [
        {"p": held["id"], "state": "named", "iou": 0.8, "roi": 1, "as": "DM2", "has": None},
        {"p": new["id"], "state": "unnamed", "iou": 0.4, "roi": 7, "as": None, "has": 9},
        {"p": new["id"], "state": "stacked", "iou": 0.0, "roi": 9, "as": "DM2", "on": None}]
    assert rows["DL5"]["flies"][:2] == [{"p": held["id"], "state": "other", "iou": 0.6, "roi": 2, "as": "VM7d", "has": 4},
                                        {"p": held["id"], "state": "elsewhere", "iou": 0.0, "roi": 4, "as": "DL5", "on": None}]
    assert rows["DA1"]["flies"] == [{"p": held["id"], "state": "none", "iou": 0.2, "roi": 3, "as": "DA1", "has": 3},
                                    {"p": held["id"], "state": "weak", "iou": 0.2, "roi": 3, "as": "DA1", "on": "DA1"}]
    assert s["total"] == {"drawn": 7, "found": 3, "named": 2, "unnamed": 1, "none": 2, "untouched": 1, "moved": 2, "given": 6, "given_right": 2,
                          "given_weak": 1, "given_stacked": 1, "given_elsewhere": 1, "given_undrawn": 1, "other": 2, "given_wrong": 0}
    # whose ROIs the naming explorer shows (cleaned ones have the labels of those they were cleaned from)
    assert s["predictions"] == {str(held["id"]): {"fly": "ann_2024-01-01_1", "kind": "clean", "set": "held_out", "changed": False, "explorer": True},
                                str(new["id"]): {"fly": "ann_2024-02-02_1", "kind": "pred", "set": "new", "changed": True, "explorer": True}}

    # the model's own ROIs instead of the predictions' refined ones
    s = call(main.fits_by_glomerulus, query={"rois": "pred"})[1]
    assert s["counts"]["kinds"] == {"pred": 2} and count({r["name"]: r for r in s["glomeruli"]}["DM2"]) == {
        "drawn": 2, "found": 1, "unnamed": 2, "moved": 1, "given": 1, "given_stacked": 1}
    # another experimenter's, and what is not a choice
    assert call(main.fits_by_glomerulus, query={"who": "bob"})[1]["counts"]["missing"] == [bob["id"]]
    assert call(main.fits_by_glomerulus, query={"who": "nobody"})[1]["experimenter"] == "ann"
    assert call(main.fits_by_glomerulus, query={"fit": "999"})[1]["fit"] is None
    assert call(main.fits_by_glomerulus, query={"flies": "some"})[0] == 400 and call(main.fits_by_glomerulus, query={"fit": "x"})[0] == 400

    # names the experimenter gave ROIs in the flies of the site's datasets (as last looked at) are listed too
    assert {r["name"]: r["site"] for r in s["glomeruli"]} == {"DA1": 0, "DL5": 0, "DM2": 0, "DM2/DL5": 0, "VA1d": 0}
    config.ROI_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    (config.ROI_TRAINING_DIR / "summary.json").write_text(json.dumps({"made_at": "2026-11-23T00:00:00", "summary": {"flies": [
        {"dir": "ann/analysis/2024-01-01/1", "glomeruli": ["DM2", "VM7d"]}, {"dir": "ann/analysis/2024-01-01/2", "glomeruli": ["DM2"]},
        {"dir": "bob/analysis/2024-03-03/1", "glomeruli": ["DC1"]}]}}))
    s = call(main.fits_by_glomerulus)[1]
    assert {r["name"]: r["site"] for r in s["glomeruli"]} == {"DA1": 0, "DL5": 0, "DM2": 2, "DM2/DL5": 0, "VA1d": 0, "VM7d": 1} and s["only_model"] == []

    # one whose ROIs cannot be matched says why, and is left out; matches of other ROIs than it now has (refined since), or of
    # another format, are made again
    matched(new, {}, error="its fly's own ROIs are no longer there")
    s = call(main.fits_by_glomerulus)[1]
    assert s["failed"] == [{"id": new["id"], "fly": "ann_2024-02-02_1", "why": "its fly's own ROIs are no longer there"}] and s["counts"]["ready"] == 1
    time.sleep(0.01)
    (predictions.pred_dir(held["id"]) / "ann_2024-01-01_1_clean.tif").write_bytes(b"other labels")
    assert bg.read(held) is None and call(main.fits_by_glomerulus)[1]["counts"]["missing"] == [held["id"]]
    matched(held, {"pred": own, "clean": clean})
    assert bg.read(held) is not None
    (tmp_path / "ann_2024-01-01_1_RoiSet.zip").write_bytes(b"its own ROIs, drawn again")
    assert bg.read(held) is None
    matched(held, {"pred": own, "clean": clean}, format=0)
    assert bg.read(held) is None
    # its refinement undone since (its ROIs the model's own again): those are summed
    matched(held, {"pred": own})
    predictions.update(db, held["id"], result={**held["result"], "final": "pred"})
    (predictions.pred_dir(held["id"]) / "ann_2024-01-01_1_clean.tif").unlink()
    matched(held, {"pred": own})
    assert call(main.fits_by_glomerulus)[1]["counts"]["kinds"] == {"pred": 1}
    # the fit deleted: still summed, without what its model has
    shutil.rmtree(fits.fit_dir(fit["id"]))
    with db.connect() as con:
        con.execute("DELETE FROM fits WHERE id = ?", (fit["id"],))
    s = call(main.fits_by_glomerulus)[1]
    assert s["choices"][0]["deleted"] and s["choices"][0]["by"] is None and s["model"] is None and s["glomeruli"][0]["model"] is None
    assert s["counts"]["trained"] == 0 and s["counts"]["chosen"] == 3


def test_ask(main, tensor, tmp_path, monkeypatch):  # noqa: F811
    """The predictions not matched yet are matched by the fit worker, as many as it gets to: asked for through the API."""
    db = main.db
    fit = done_fit(db)
    ps = [predicted(db, fit, f"ann_2024-02-0{i}_1", tmp_path) for i in (1, 2, 3)]
    ids = [p["id"] for p in ps]
    post = lambda body: call(main.fits_by_glomerulus, method="POST", body=body)  # noqa: E731
    for bad in ({}, {"predictions": []}, {"predictions": ["1"]}, {"predictions": [True]}, {"predictions": list(range(bg.MOST + 1))}, [1]):
        assert post(bad)[0] == 400
    status, r = post({"predictions": ids})
    assert status == 503 and "the fit worker is not running" in r["error"]
    fits.heartbeat(db, {"host": "h", "gpu": None})
    assert call(main.fits_by_glomerulus)[1]["worker"] is True
    asked = []

    def make_one(p):
        asked.append(p["id"])
        if p["id"] == ids[1]:
            raise ValueError("its fly's own ROIs could not be read")
        return {"format": bg.FORMAT, "of": bg._of(p), "detect_iou": 0.5, "name_iou": 0.3, "changed": False,
                "kinds": {"pred": {"manual": [{"name": "DM2", "iou": 0.9, "roi": 1, "as": "DM2"}], "predicted": []}}}

    monkeypatch.setattr(bg, "make_one", make_one)
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            if not nmf_check.answer(db):
                time.sleep(0.02)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        # (no time at all: one is done all the same, the rest left for the next time it is asked)
        monkeypatch.setattr(bg, "BUDGET_S", 0.0)
        assert post({"predictions": ids}) == (200, {"made": [ids[0]], "failed": {}, "left": ids[1:]})
        monkeypatch.setattr(bg, "BUDGET_S", 20.0)
        # one that cannot be matched is done too: why is kept with it, and it is not asked for again
        assert post({"predictions": [*ids[1:], 999]}) == (200, {"made": ids[1:], "failed": {str(ids[1]): "its fly's own ROIs could not be read"}, "left": []})
    finally:
        stop.set()
        t.join()
    s = call(main.fits_by_glomerulus)[1]
    assert s["counts"]["ready"] == 2 and s["counts"]["missing"] == [] and [f["id"] for f in s["failed"]] == [ids[1]]
    assert s["glomeruli"][0]["named"] == 2 and asked == ids
    # kept in the predictions' folders; what the worker does when asked
    assert bg.doc_path(ps[0]).parent == predictions.pred_dir(ids[0]) and bg.FILE in bg.doc_path(ps[0]).name
    assert bg.CALL in nmf_check.CALLS and fit_worker.by_glomerulus is bg


# -------------------------------------------- the fit worker, with al-diag-seg

def test_worker_matches(main, tensor, tmp_path):  # noqa: F811
    """A prediction's ROIs matched to its fly's own, by al-diag-seg's own matching: what is on what, and named what."""
    roi_io = pytest.importorskip("alseg.roi_io")
    from alseg import metrics
    from alseg import model as am

    from app import glomeruli

    assert {"DM2", "DL5", "VA1d", "DA1", "DC3", "VM7d"} <= set(glomeruli.load_glomeruli())
    db = main.db
    fit = done_fit(db)
    model(fit)
    shape = (3, 24, 30)
    box = {"A": (slice(2, 11), slice(2, 13)), "B": (slice(2, 11), slice(16, 28)), "C": (slice(13, 23), slice(2, 13)), "D": (slice(13, 23), slice(16, 28))}
    # the fly's own: DM2 drawn on its first plane only; DL5, VA1d and DA1 on all; and what is no glomerulus to the model:
    # a numbered ROI, one not certain (?), and a tentative one, taken as that glomerulus
    own = np.zeros(shape, np.uint16)
    own[0][box["A"]] = 1
    own[(slice(None), *box["B"])], own[(slice(None), *box["C"])], own[(slice(None), *box["D"])] = 2, 3, 4
    own[1, 0:2, 0:6], own[1, 0:2, 8:14], own[1, 0:2, 16:22] = 5, 6, 7
    roiset = tmp_path / "RoiSet.zip"
    roi_io.write_roiset(own, {1: "DM2", 2: "DL5", 3: "VA1d", 4: "DA1", 5: None, 6: "VC1?", 7: "DC3_t0"}, roiset)
    # predicted: 1 on DM2, unnamed; 2 over it on the last plane (where it is not drawn), named DM2; 3 on most of DL5, named VA1d;
    # 4 on DA1, named so; 5 and 6 where nothing is drawn, named DL5 (which is drawn elsewhere) and VM7d (which is not drawn)
    lab = np.zeros(shape, np.uint16)
    lab[0][box["A"]], lab[2][box["A"]] = 1, 2
    lab[:, 2:11, 20:30] = 3
    lab[(slice(None), *box["D"])] = 4
    lab[0, 11:13, 0:8], lab[2, 11:13, 16:24] = 5, 6
    names = {1: None, 2: "DM2", 3: "VA1d", 4: "DA1", 5: "DL5", 6: "VM7d"}
    p = predicted(db, fit, "ann_2024-02-02_1", tmp_path)
    d = predictions.pred_dir(p["id"])
    tifffile.imwrite(d / "ann_2024-02-02_1_pred.tif", lab, imagej=True, metadata={"axes": "ZYX"})
    with open(d / "ann_2024-02-02_1_identities_pred.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "assigned", "manual", "cost", "confidence"])
        w.writerows([l, n or "", "", "0.400", "0.100"] for l, n in names.items())
    st = roiset.stat()
    prov = {"fly": {"roiset": str(roiset), "roiset_file": fit_worker._file(roiset)}, "model": {"alseg_options": {"tentative": True}}}
    (d / predictions.PROVENANCE).write_text(json.dumps(prov))

    assert bg.make(db, {"predictions": [p["id"], 999]}) == {"made": [p["id"]], "failed": {}, "left": []}
    doc = bg.read(p)
    assert (doc["detect_iou"], doc["name_iou"], doc["scoring"], doc["changed"]) == (am.DETECT_IOU, am.NAME_IOU, "partly drawn", False)
    assert doc["of"] == {"roiset": [str(roiset), st.st_size, st.st_mtime_ns], "labels": {"pred": bg._stat(d / "ann_2024-02-02_1_pred.tif")}}
    k = doc["kinds"]["pred"]
    # as al-diag-seg matches them
    gt, lut, _ = roi_io.load_roiset(roiset, shape, keep=__import__("alseg.olfdb", fromlist=["x"]).roi_keeper(set(glomeruli.load_glomeruli())))
    assert sorted(lut.values()) == ["DA1", "DC3", "DL5", "DM2", "VA1d"]
    iou, g_ids, p_ids, _ = metrics.iou_matrix_partly_drawn(gt, lab)
    by = {lut[int(g)]: i for i, g in enumerate(g_ids)}
    man = {m["name"]: m for m in k["manual"]}
    assert set(man) == set(lut.values()) and all(m["iou"] == pytest.approx(float(iou[by[n]].max())) for n, m in man.items())
    # (2, on planes where DM2 is not drawn, counts neither for it nor against it)
    assert man["DM2"] == {"name": "DM2", "iou": pytest.approx(1.0), "roi": 1, "as": None} and man["DA1"]["roi"] == 4 and man["DA1"]["as"] == "DA1"
    assert man["DL5"]["roi"] == 3 and man["DL5"]["as"] == "VA1d" and 0.5 <= man["DL5"]["iou"] < 0.9
    assert man["VA1d"] == {"name": "VA1d", "iou": 0.0, "roi": None, "as": None} and man["DC3"]["roi"] is None
    pre = {q["label"]: q for q in k["predicted"]}
    assert pre[2] == {"label": 2, "name": "DM2", "manual": None, "iou": 0.0, "over": pytest.approx(1.0)}
    assert (pre[3]["manual"], pre[3]["over"]) == ("DL5", 0.0) and (pre[5]["manual"], pre[5]["over"]) == (None, 0.0)
    assert pre[6]["over"] is None and pre[1]["over"] is None and pre[4]["over"] == pytest.approx(1.0)

    s = bg.summary(db)
    rows = {r["name"]: r for r in s["glomeruli"]}
    state = lambda n: [(f["state"], f["roi"]) for f in rows[n]["flies"]]  # noqa: E731
    assert state("DM2") == [("unnamed", 1), ("stacked", 2)] and state("DA1") == [("named", 4)] and state("DC3") == [("none", None)]
    assert state("DL5") == [("other", 3), ("elsewhere", 5)] and state("VA1d") == [("none", None), ("wrong", 3)]
    assert [(r["name"], r["given_undrawn"]) for r in s["only_model"]] == [("VM7d", 1)]
    assert (rows["DM2"]["found"], rows["DM2"]["moved"], rows["VA1d"]["moved"], rows["VA1d"]["given_wrong"]) == (1, 1, 1, {"DL5": 1})

    # the fly's own ROIs drawn again since it was predicted: matched to those they are now, and said so; a model fit without
    # tentative ROIs does not take them as glomeruli
    time.sleep(0.01)
    roi_io.write_roiset(own, {1: "DM2", 2: "DL5", 3: "VA1d", 4: "DL1", 5: None, 6: "VC1?", 7: "DC3_t0"}, roiset)
    assert bg.read(p) is None
    was = {**prov["fly"]["roiset_file"], "bytes": prov["fly"]["roiset_file"]["bytes"] + 1}  # (as it was: of another size)
    (d / predictions.PROVENANCE).write_text(json.dumps({"fly": {"roiset": str(roiset), "roiset_file": was}, "model": {"alseg_options": {"tentative": False}}}))
    doc = bg.make_one(p)
    assert doc["changed"] is True and sorted(m["name"] for m in doc["kinds"]["pred"]["manual"]) == ["DL1", "DL5", "DM2", "VA1d"]
    # what cannot be matched: why, kept with the prediction until its files change
    roiset.unlink()
    assert bg.make(db, {"predictions": [p["id"]]})["failed"] == {str(p["id"]): f"its fly's own ROIs are no longer there ({roiset})"}
    assert bg.read(p)["error"].startswith("its fly's own ROIs are no longer there")
    (d / predictions.PROVENANCE).write_text(json.dumps({"fly": {"roiset": None}}))
    with pytest.raises(ValueError, match="had no ROIs of its own"):
        bg.make_one(p)
