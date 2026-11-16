"""Tests of refining a prediction's ROIs after it was made (refinement.py): asking for it, its state and the prediction's
result while it has one, undoing one asked for again, a prediction refined before refinements were kept apart, and the
fit worker's making of one (fit_worker.run_refinement) with the real al-diag-seg, where it is installed (the worker
image), its stand-in segmenter in place of Cellpose. Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/."""
from __future__ import annotations

import json

import numpy as np
import pytest
import roifile
import tifffile

from app import config, fit_worker, fits, nmf_check, predictions, refinement
from tests.test_fits import GLOMS, call, data, request, tensor  # noqa: F401 (fixtures)
from tests.test_predictions import FLY, RAW, done_fit
from tests.test_tensor import main, root  # noqa: F401 (fixtures)

FLYNAME = "ann_2024-01-01_1"


def rois(path, names) -> None:
    """A RoiSet of a square per name, on the first plane."""
    path.unlink(missing_ok=True)
    for i, n in enumerate(names):
        roifile.ImagejRoi.frompoints(np.array([[i * 5, 0], [i * 5 + 4, 0], [i * 5 + 4, 4]], "f4"), name=n, z=0).tofile(path)


def done(main, **result) -> dict:  # noqa: F811
    """A prediction done, its folder with the model's own ROIs (as the web server sees them)."""
    f = done_fit(main.db)
    p, _ = predictions.create(main.db, f["id"], predictions.locate(FLY, RAW, config.TENSOR_ROOT, []), "Ann", None)
    d = predictions.pred_dir(p["id"])
    d.mkdir(parents=True)
    tifffile.imwrite(d / f"{FLYNAME}_pred.tif", np.ones((3, 4, 5), np.uint16), imagej=True, metadata={"axes": "ZYX"})
    tifffile.imwrite(d / predictions.AVERAGE, np.ones((3, 4, 5), np.float32), imagej=True, metadata={"axes": "ZYX"})
    rois(d / f"{FLYNAME}_pred_RoiSet.zip", ["DM2", "DL5"])
    rois(d / f"{FLYNAME}_RoiSet.zip", ["DM2", "DL5"])
    predictions.update(main.db, p["id"], status="done", finished_at=predictions.now(),
                       result={"final": "pred", "n": 2, "named": ["DL5", "DM2"], "scores": None, "device": "CPU", **result})
    return predictions.get(main.db, p["id"])


def test_ask_and_state(main, tensor):  # noqa: F811
    """A refinement asked for, waited for and claimed, as the tab is told; what cannot be asked for; one interrupted."""
    db = main.db
    p = done(main)
    ask = lambda **body: call(main.predictions_refine, method="POST", path_params={"id": p["id"]}, body=body)  # noqa: E731
    assert ask(split=True)[0] == 422 and ask(name="Bob Smith", split=True)[0] == 422
    assert call(main.predictions_refine, method="POST", path_params={"id": 99}, body={"name": "Bob", "split": True})[0] == 404
    assert ask(name="Bob", clean=["erode"])[0] == 400
    status, r = ask(name="Bob")
    assert status == 409 and "at least one step" in r["error"]
    status, r = ask(name="Bob", clean=["core"])  # (it kept no cell probabilities)
    assert status == 409 and "core goes by Cellpose's cell probabilities" in r["error"]
    # listed: none yet, what can be done
    listed = lambda: next(x for x in call(main.predictions_list)[1]["predictions"] if x["id"] == p["id"])  # noqa: E731
    assert listed()["refine"] is None and listed()["refinable"] == {"any": True, "core": False}
    tifffile.imwrite(refinement.files(p)["cellprob"], np.zeros((3, 4, 5), np.float32), imagej=True, metadata={"axes": "ZYX"})
    assert listed()["refinable"]["core"]
    opts = {"clean": ["trim"], "clean_params": {"trim": {"max_share": 0.2}}, "split": True}
    status, r = ask(name="Bob", **opts)
    assert status == 201 and r["refine"]["status"] == "queued" and r["refine"]["options"] == opts and r["refine"]["name"] == "Bob"
    assert ask(name="Cy", split=True)[0] == 200 and listed()["refine"]["name"] == "Bob"  # (one is waiting already)
    # while it waits: no look for missed glomeruli (it would be out of date), nor a new prediction of the fly over it once it runs
    assert call(main.predictions_nmf_ask, method="POST", path_params={"id": p["id"]}, body={"name": "Cy"})[0] == 400
    q = refinement.claim(db)
    assert q["id"] == p["id"] and q["refine"]["status"] == "running" and refinement.claim(db) is None
    fly = predictions.locate(FLY, RAW, config.TENSOR_ROOT, [])
    assert predictions.create(db, p["fit_id"], fly, "Cy", None, overwrite=True)[1] is False
    assert call(main.predictions_delete, method="DELETE", path_params={"id": p["id"]})[0] == 409
    assert any("being refined" in j["what"] for j in predictions.worker_jobs(db)["jobs"])
    # the worker stopped while it ran: interrupted, its ROIs the model's own
    assert refinement.interrupt_running(db) == [p["id"]]
    r = predictions.get(db, p["id"])
    assert r["refine"]["status"] == "interrupted" and "stopped" in r["refine"]["error"] and r["result"]["final"] == "pred"
    assert listed()["refine"]["status"] == "interrupted"


def test_done_and_undone(main, tensor):  # noqa: F811
    """A prediction refined: its result that of the ROIs refined, the viewer told what was done to each and given the model's
    own too; asked for again, undone at once (and a look for missed glomeruli made of them deleted)."""
    db = main.db
    p = done(main, scores={"manual": 3, "recall": 0.5, "gap_um": 0.0, "named": 1, "named_of": 2})
    d, f = predictions.pred_dir(p["id"]), refinement.files(p)
    status, _ = call(main.predictions_refine, method="POST", path_params={"id": p["id"]}, body={"name": "Bob", "split": True, "clean": ["margin"]})
    assert status == 201
    refinement.claim(db)
    # (as the worker leaves it: refinement.run)
    for kind in ("split", "clean"):
        rois(d / f"{FLYNAME}_{kind}_RoiSet.zip", ["DM2", "DL5", "3"])
        tifffile.imwrite(d / f"{FLYNAME}_{kind}.tif", np.ones((3, 4, 5), np.uint16), imagej=True, metadata={"axes": "ZYX"})
    rois(f["roiset"], ["DM2", "DL5", "3"])
    f["json"].write_text(json.dumps({"rois": {"1": {"name": "DM2", "split": {"0": 7}, "of_names": {"0": "DM2"}, "trimmed": {}},
                                              "3": {"name": "3", "split": {"0": 7}, "of_names": {"0": "DM2"}, "trimmed": {"1": 0.8}}}}))
    kinds = {"pred": {"n": 2, "recall": 0.5, "gap_um": 0.0, "named": 1, "named_of": 2},
             "split": {"n": 3, "recall": 0.75, "gap_um": 0.0, "named": 2, "named_of": 2},
             "clean": {"n": 3, "recall": 0.7, "gap_um": 2.0, "named": 2, "named_of": 2}}
    refinement.update(db, p["id"], "done", finished_at=predictions.now(),
                      result={"steps": ["split", "margin"], "final": "clean", "n": 3, "named": ["DL5", "DM2"], "split": 2, "trimmed": 1, "kinds": kinds})
    # its result: of the ROIs refined (the model's own's kept as they were, apart)
    r = predictions.get(db, p["id"])["result"]
    assert (r["final"], r["n"]) == ("clean", 3) and r["scores"]["recall"] == 0.7 and r["scores"]["gap_um"] == 2.0 and r["scores"]["kinds"] == kinds
    assert r["scores"]["manual"] == 3 and r["device"] == "CPU"
    with db.connect() as con:
        assert json.loads(con.execute("SELECT result_json FROM predictions WHERE id = ?", (p["id"],)).fetchone()[0])["final"] == "pred"
    # the viewer: the refined ROIs, the model's own, and what was done to each refined one by plane
    v = call(main.predictions_view, path_params={"id": p["id"]})[1]
    assert v["movies"][0]["predicted"] == str(f["roiset"]) and v["movies"][0]["unrefined"] == str(f["pred_rois"])
    assert v["refined"]["rois"] == {"DM2": {"0": {"split_from": "DM2"}}, "3": {"0": {"split_from": "DM2"}, "1": {"trimmed": 0.8}}}
    assert v["refined"]["steps"] == ["split", "margin"] and not v["refined"]["legacy"]
    assert refinement.options_of(predictions.get(db, p["id"]))["split"]
    # a look for missed glomeruli made of them, then refined again: the refinement before undone at once, the look deleted
    nf = nmf_check.files(p)
    nf["json"].write_text("{}")
    with db.connect() as con:
        con.execute("UPDATE predictions SET nmf_status = 'done', nmf_json = '{}' WHERE id = ?", (p["id"],))
    status, _ = call(main.predictions_refine, method="POST", path_params={"id": p["id"]}, body={"name": "Cy", "clean": ["margin"]})
    assert status == 201
    q = predictions.get(db, p["id"])
    assert q["result"]["final"] == "pred" and q["refine"]["status"] == "queued" and q["nmf"] is None and not nf["json"].exists()
    assert f["roiset"].read_bytes() == f["pred_rois"].read_bytes() and not f["json"].exists()
    assert not any(x.exists() for k in refinement.MADE for x in refinement.kind_files(q, k)) and f["pred"].is_file()
    assert call(main.predictions_view, path_params={"id": p["id"]})[1]["refined"] is None


def test_legacy(main, tensor):  # noqa: F811
    """A prediction refined as it was made, before refinements were kept apart: one done, from its result; refined again,
    its result made the model's own's."""
    db = main.db
    kinds = {"pred": {"n": 2, "recall": 0.5}, "clean": {"n": 2, "recall": 0.4}, "clean_trim": {"n": 2, "recall": 0.45}}
    p = done(main, final="clean", scores={"recall": 0.4, "kinds": kinds})
    with db.connect() as con:
        con.execute("UPDATE predictions SET options_json = ? WHERE id = ?",
                    (json.dumps({"clean": ["trim"], "clean_params": {}, "split": True}), p["id"]))
    d = predictions.pred_dir(p["id"])
    rois(d / f"{FLYNAME}_clean_RoiSet.zip", ["DM2"])
    with open(d / f"{FLYNAME}_identities_pred.csv", "w") as fh:
        fh.write("label,assigned,manual,cost,confidence\n1,DM2,DM2,0.2,0.9\n2,DL5,,0.3,0.8\n")
    p = predictions.get(db, p["id"])
    st = refinement.state(p)
    assert st["status"] == "done" and st["legacy"] and st["result"]["steps"] == ["split", "trim"] and set(st["result"]["kinds"]) == {"pred", "clean"}
    assert refinement.details(p)["legacy"] and refinement.options_of(p)["clean"] == ["trim"]
    status, _ = call(main.predictions_refine, method="POST", path_params={"id": p["id"]}, body={"name": "Bob", "clean": ["margin"]})
    assert status == 201
    q = predictions.get(db, p["id"])
    assert (q["result"]["final"], q["result"]["n"], q["result"]["named"]) == ("pred", 2, ["DL5", "DM2"])
    assert q["result"]["scores"] == {"recall": 0.5} and not (d / f"{FLYNAME}_clean_RoiSet.zip").exists()


def test_worker_refinement(main, tensor, monkeypatch):  # noqa: F811
    """The real al-diag-seg (where it is installed: the worker image), with its stand-in segmenter: a fly's prediction
    refined after it was made, an ROI split (where al-diag-seg says) and the ROIs trimmed; then refined again by margin
    alone; then a refinement that fails, leaving the model's own ROIs."""
    am = pytest.importorskip("alseg.model")
    if not hasattr(am.Model, "refine"):
        pytest.skip("an al-diag-seg without Model.refine")

    class StandIn(am.Options):
        def __init__(self, **kw):
            super().__init__(**{"segmenter": "stand-in", "gpu": False, **kw})

    monkeypatch.setattr(am, "Options", StandIn)
    db = main.db
    f = fits.create(db, "Ann", None, fits.options({"held_out": 1, "atlas": False}))
    fit_worker.run(db, fits.claim(db))
    held = fits.get(db, f["id"])["flies"]["held_out"][0]
    date, n = held.split("_")[1:]
    p, _ = predictions.create(db, f["id"], predictions.locate(f"ann/analysis/{date}/{n}", f"ann/raw/{date}/{n}", config.TENSOR_ROOT, []), "Ann", None)
    fit_worker.run_prediction(db, predictions.claim(db), False)
    d = predictions.pred_dir(p["id"])
    p = predictions.get(db, p["id"])
    assert p["status"] == "done", (d / "log.txt").read_text()[-3000:]
    own = p["result"]
    assert own["final"] == "pred" and p["refine"] is None and (d / f"{held}_identities_pred.csv").is_file()
    names0 = refinement.pred_names(p)
    dm2 = next(l for l, v in names0.items() if v[0] == "DM2")

    # (which masks hold two glomeruli is al-diag-seg's to say, by the trials' maps: here, DM2's on the first plane, its right half)
    def cut(masks, fly, return_report=False, **kw):
        out, new = masks.copy(), int(masks.max()) + 1
        ys, xs = np.nonzero(masks[0] == dm2)
        out[0][(masks[0] == dm2) & (np.arange(masks.shape[2])[None, :] > xs.mean())] = new
        return out, {"split": {(dm2, 0): new}, "flagged": {}, "stats": {}}

    monkeypatch.setattr(__import__("alseg.clean_trim").clean_trim, "split", cut)
    status, r = call(main.predictions_refine, method="POST", path_params={"id": p["id"]}, body={"name": "Bob", "split": True, "clean": ["trim"]})
    assert status == 201
    fit_worker.run_refinement(db, refinement.claim(db))
    p = predictions.get(db, p["id"])
    log = (d / "log.txt").read_text()
    rf = p["refine"]
    assert rf["status"] == "done", log[-3000:]
    res = rf["result"]
    assert res["steps"] == ["split", "trim"] and res["final"] == "clean" and set(res["kinds"]) == {"pred", "split", "clean"}
    assert res["split"] == 2 and res["seconds"] > 0 and set(res["stages"]) <= set(refinement.STAGES) and rf["progress"] is None
    assert "its ROIs refined: asked for by Bob" in log and "masks over two glomeruli split, trim" in log
    # its ROIs: those refined (one more: DM2's first plane is two ROIs there), scored as the model's own were, which are kept
    k = res["kinds"]
    assert k["pred"]["n"] == own["n"] and k["split"]["n"] == own["n"] + 1 and k["pred"]["recall"] == own["scores"]["recall"]
    assert p["result"]["final"] == "clean" and p["result"]["n"] == res["n"] and p["result"]["scores"]["kinds"] == k
    assert (d / f"{held}_RoiSet.zip").read_bytes() == (d / f"{held}_clean_RoiSet.zip").read_bytes()
    split = tifffile.imread(d / f"{held}_split.tif")
    assert len(np.unique(split[0][tifffile.imread(d / f"{held}_pred.tif")[0] == dm2])) == 2
    # what was done to each, for the viewer: the two parts of DM2's outline on the first plane, each by its name in the RoiSet
    v = predictions.view(db, p["id"])
    parts = {name: planes for name, planes in v["refined"]["rois"].items() if "0" in planes and planes["0"].get("split_from")}
    assert len(parts) == 2 and all(x["0"]["split_from"] == "DM2" for x in parts.values())
    in_roiset = {(x.name, x.z_position - 1) for x in roifile.ImagejRoi.fromfile(d / f"{held}_RoiSet.zip")}
    assert all((name, 0) in in_roiset for name in parts) and v["movies"][0]["unrefined"].endswith(f"{held}_pred_RoiSet.zip")
    assert predictions.identity_terms(db, p["id"])["rois"]  # (what the refined ROIs' names cost)

    # refined again, by margin alone: from the model's own, the split ones gone
    status, _ = call(main.predictions_refine, method="POST", path_params={"id": p["id"]}, body={"name": "Cy", "clean": ["margin"]})
    assert status == 201 and not (d / f"{held}_split.tif").exists()
    fit_worker.run_refinement(db, refinement.claim(db))
    p = predictions.get(db, p["id"])
    assert p["refine"]["status"] == "done" and p["refine"]["result"]["steps"] == ["margin"] and set(p["refine"]["result"]["kinds"]) == {"pred", "clean"}
    assert p["result"]["n"] == own["n"] and p["refine"]["result"]["split"] == 0 and not (d / f"{held}_split.tif").exists()

    # one that fails: said so, its ROIs the model's own
    status, _ = call(main.predictions_refine, method="POST", path_params={"id": p["id"]}, body={"name": "Cy", "clean": ["trim"]})
    monkeypatch.setattr(am.Model, "refine", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no luck")))
    fit_worker.run_refinement(db, refinement.claim(db))
    p = predictions.get(db, p["id"])
    assert p["refine"]["status"] == "failed" and p["refine"]["error"] == "RuntimeError: no luck" and p["result"]["final"] == "pred"
    assert (d / f"{held}_RoiSet.zip").read_bytes() == (d / f"{held}_pred_RoiSet.zip").read_bytes()
