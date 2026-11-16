"""Glomeruli a prediction's ROIs miss, looked for in the fly's movies by non-negative matrix
factorization (al-diag-seg's `alseg.nmf`): asked for in the viewer of a prediction (the AL
segmentation tab), and run by the fit worker between predictions (`fit_worker.run_nmf`), on the
GPU when no fit is using it, else on the CPU.

A prediction has one such look (asked for again, it is made again; predicted again, it is gone
with the prediction's files). Its state is the prediction's ``nmf`` (`state`: ``status``, who
asked and when, how far it is), and what it found is in the prediction's folder:

- ``<fly>_nmf.json``: the candidates (`alseg.nmf.find`'s: glomeruli added, more of an ROI on a
  plane it had no outline on, and parts cut off ROIs that hold two), each with its outlines and
  the name it gets, the names of all ROIs with every candidate merged in (the predicted ROIs'
  kept, or all named again together: by the ROIs as the model named them, which are Cellpose's
  own where the prediction's are those cleaned, so that only what is merged changes a name),
  how it was made, of what, by whom;
- ``<fly>_nmf.tif``: the ROIs with every candidate merged in (labels, as the prediction's), and
  ``<fly>_nmf_candidates.tif``: the candidate each voxel is of;
- ``<fly>_nmf_RoiSet.zip``: the candidates alone, named (the predicted ROIs' names kept), for
  ImageJ;
- ``<fly>_atlas_prior.npz``: the model's atlas on the fly's grid (each glomerulus' probability),
  which names are given by (kept from the prediction, if it has one; else registered again);
- ``.nmf_inputs.npz``: the fly's odor responses as the model reads them (hidden: not in the
  prediction's download), with which ROIs are named again without reading the fly.

Candidates are only proposed: the viewer shows them beside the prediction's ROIs, and merges
those chosen into its edited ROIs (`predictions.save_edited`), named as here (`found`), or, for
some of them with all ROIs named again together, as the fit worker names them when asked (`plan`,
through `call`: the web server has no al-diag-seg).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from . import fits, movies, predictions, refinement

STATUSES = ("queued", "running", "done", "failed", "interrupted")
ACTIVE = ("queued", "running")
STAGES = ("load", "register", "read", "factorize", "name")  # what a look goes through (`fraction`)
# seconds each stage takes, until a look done on that device says (its result's ``stages``)
GUESSES = {"GPU": {"load": 10, "register": 60, "read": 30, "factorize": 20, "name": 10},
           "CPU": {"load": 10, "register": 60, "read": 30, "factorize": 240, "name": 10}}
# what of a look can be set when it is asked for (`settings`): al-diag-seg's `alseg.nmf.DEFAULTS` of those names, each its
# (default, least, most); n_free 0: nothing is factorized, and the look only tests the predicted ROIs themselves
SETTINGS = {"n_free": (25, 0, 100), "min_area_um2": (20.0, 1.0, 500.0), "min_radius_um": (2.5, 0.0, 20.0), "min_in_hull": (0.8, 0.0, 1.0),
            "min_eta2": (0.5, 0.0, 1.0), "min_novelty": (0.5, 0.0, 1.0)}
INPUTS = ".nmf_inputs.npz"
CALL_TIMEOUT_S = 90.0  # how long the web server waits for the fit worker to do what it asked
CALL_POLL_S = 0.2
CALL_STALE_S = 600.0  # calls left by a web server that stopped waiting are dropped after this


class CallError(Exception):
    """What the web server asked of the fit worker (`call`) was not done: why."""


def files(p: dict) -> dict[str, Path]:
    """The files of a prediction's look, in its folder."""
    d, fly = predictions.pred_dir(p["id"]), p["fly"]
    return {"json": d / f"{fly}_nmf.json", "labels": d / f"{fly}_nmf.tif", "candidates": d / f"{fly}_nmf_candidates.tif",
            "roiset": d / f"{fly}_nmf_RoiSet.zip", "prior": d / f"{fly}_atlas_prior.npz", "inputs": d / INPUTS}


def settings(body: dict) -> dict:
    """How a look asked for is made, where not as by default: of `SETTINGS`, the values given that are not the default
    (``n_free`` a whole number), and ``split`` false if the masks al-diag-seg would cut in two are not to be proposed.

    Raises:
        ValueError: for a setting that is not one, or a value out of range.
    """
    given = body.get("settings") or {}
    if not isinstance(given, dict) or any(k not in SETTINGS and k != "split" for k in given):
        raise ValueError(f"settings: of {', '.join(SETTINGS)}, split")
    out = {}
    for k, v in given.items():
        if k == "split":
            if v is False:
                out[k] = False
            continue
        default, lo, hi = SETTINGS[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi or (k == "n_free" and v != int(v)):
            raise ValueError(f"settings: {k} is a {'whole ' if k == 'n_free' else ''}number from {lo:g} to {hi:g}")
        if v != default:
            out[k] = int(v) if k == "n_free" else float(v)
    return out


# -------------------------------------------------------------------- state

def _set(con, pred_id: int, status: Optional[str], **fields) -> None:
    r = con.execute("SELECT nmf_status, nmf_json FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    if r is None or r["nmf_status"] is None:  # (predicted again meanwhile: the look is gone with its files)
        return
    d = {**(json.loads(r["nmf_json"]) if r["nmf_json"] else {}), **fields}
    if status is None:
        con.execute("UPDATE predictions SET nmf_json = ? WHERE id = ?", (json.dumps(d), pred_id))
    else:
        con.execute("UPDATE predictions SET nmf_status = ?, nmf_json = ? WHERE id = ?", (status, json.dumps(d), pred_id))


def update(db, pred_id: int, status: Optional[str] = None, **fields) -> None:
    """Sets fields of a prediction's look (kept with the others), and its status if given; nothing
    of a prediction with no look (asked for again since it was started)."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        _set(con, pred_id, status, **fields)


def ask(db, pred_id: int, name: str, ip: Optional[str], how: Optional[dict] = None) -> tuple[dict, bool]:
    """Asks for a look for glomeruli a prediction's ROIs miss (made as ``how`` says: `settings`'):
    queued, unless one is queued or running already. One made before is replaced when this one is done.

    Returns:
        (the prediction, whether it was queued now).

    Raises:
        LookupError: no such prediction done.
        ValueError: its ROIs are being refined (`refinement`), which would make the look out of date.
    """
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT status, nmf_status, refine_status FROM predictions WHERE id = ?", (pred_id,)).fetchone()
        if r is None or r["status"] != "done":
            raise LookupError("no such prediction done")
        if r["refine_status"] in ACTIVE:
            raise ValueError("its ROIs are being refined: look for glomeruli they miss once that is done")
        queued = r["nmf_status"] not in ACTIVE
        if queued:
            con.execute("UPDATE predictions SET nmf_status = 'queued', nmf_json = ? WHERE id = ?",
                        (json.dumps({"name": name, "ip": ip, "asked_at": predictions.now(), "settings": how or {}}), pred_id))
    return predictions.get(db, pred_id), queued


def claim(db) -> Optional[dict]:
    """The prediction whose look was asked for first, of those queued, its look now running (for
    the worker), or None."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        rows = con.execute("SELECT id, nmf_json FROM predictions WHERE nmf_status = 'queued' AND status = 'done'").fetchall()
        if not rows:
            return None
        first = min(rows, key=lambda r: (json.loads(r["nmf_json"] or "{}").get("asked_at") or "", r["id"]))
        _set(con, first["id"], "running", started_at=predictions.now())
    return predictions.get(db, first["id"])


def interrupt_running(db) -> list[int]:
    """Marks looks left running (by a worker that stopped) interrupted. Returns their predictions' ids."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        ids = [r["id"] for r in con.execute("SELECT id FROM predictions WHERE nmf_status = 'running'")]
        for i in ids:
            _set(con, i, "interrupted", finished_at=predictions.now(), error="the fit worker stopped (restarted, or the server did) while it ran")
    return ids


def expected(db, device: str) -> dict:
    """The seconds each of `STAGES` should take on `device` ('GPU' or 'CPU'): as the last look done
    on it took (those it went through), else `GUESSES`."""
    out = dict(GUESSES[device])
    with db.connect() as con:
        rows = [json.loads(r["nmf_json"]) for r in con.execute("SELECT nmf_json FROM predictions WHERE nmf_status = 'done' AND nmf_json IS NOT NULL")]
    done = [d for d in rows if (d.get("result") or {}).get("device") == device and d["result"].get("stages")]
    if done:
        out.update(max(done, key=lambda d: d.get("finished_at") or "")["result"]["stages"])
    return out


def fraction(progress: Optional[dict], of: dict, t: Optional[float] = None) -> float:
    """How far a look running is, from 0 to 1, for a progress bar: the time of the stages it has
    been through, and of the one it is at as far as it says (``done`` of ``total``; else as long
    as it has been at it, of what it should take), over the time of all it goes through
    (``stages``), each taking what `of` says (`expected`)."""
    if not progress or progress.get("stage") not in STAGES:
        return 0.0
    stages = [s for s in STAGES if s in (progress.get("stages") or STAGES)]
    total = sum(of[s] for s in stages) or 1.0
    at = progress["stage"]
    before = sum(of[s] for s in stages[:stages.index(at)]) if at in stages else 0.0
    if progress.get("total"):
        part = min(1.0, progress.get("done", 0) / progress["total"])
    else:
        part = min(0.95, ((time.time() if t is None else t) - progress.get("stage_started", 0)) / max(of[at], 1e-6))
    return max(0.0, min(0.99, (before + part * of[at]) / total))


def state(db, p: dict) -> Optional[dict]:
    """A prediction's look, as the tab shows it: its ``status``, who asked (``name``) and when,
    its ``progress`` and ``fraction`` (how far, of one running), ``result`` (of one done: how
    many candidates of each kind, how long it took), ``error``; None if none was ever asked for."""
    n = p.get("nmf")
    if not n:
        return None
    out = {k: n.get(k) for k in ("status", "name", "asked_at", "started_at", "finished_at", "progress", "result", "error", "settings")}
    if n["status"] == "running":
        out["fraction"] = fraction(n.get("progress"), expected(db, (n.get("progress") or {}).get("device") or "CPU"))
    return out


# ------------------------------------------------------------ what it found

def _read(p: dict) -> Optional[dict]:
    try:
        return json.loads(files(p)["json"].read_text())
    except (OSError, ValueError):
        return None


def _kinds(names, glomeruli: set) -> dict:
    return {n: movies.roi_kind(n, glomeruli) for n in set(names) if n}


def found(p: dict, glomeruli: set) -> Optional[dict]:
    """What a prediction's look found, for the viewer: its ``candidates`` (each its ``id``,
    ``kind``, ``label``, ``of`` (and that ROI's predicted name, ``of_name``), ``planes``, ``name``
    (with the predicted ROIs' names kept; None for one that got none), ``outlines`` ([{z,
    points}], in pixels) and, of a part cut off an ROI, ``rest``: what is left of that ROI on
    its plane, as outlines), ``predicted`` ({label: name or None}: the prediction's ROIs),
    ``names`` (of every ROI with all candidates merged in, by label: ``kept``, the predicted
    ROIs' as they are, and ``joint``, all named again together), ``kinds`` ({name: whether it is
    a glomerulus to the database, `movies.roi_kind`}, of every name the model can give),
    ``rois`` (the predicted ROIs' own in-plane masks by the tests a candidate is taken by: each
    its ``name``, ``z``, ``eta2``, ``novelty``, ``radius_um``, ``area_um2``), ``params`` (the
    look's: its thresholds among them) and ``run`` (when it was made: what candidates' ids are
    of). None if it has none (not done, or
    its file is gone)."""
    if (p.get("nmf") or {}).get("status") != "done":
        return None
    d = _read(p)
    if d is None:
        return None
    names = [n for m in d["names"].values() for n in m.values()] + list(d["predicted"].values()) + d.get("glomeruli", [])
    return {"run": d["finished"], "candidates": d["candidates"], "predicted": d["predicted"], "names": d["names"],
            "kinds": _kinds(names, glomeruli), "params": d.get("params"), "cleaned": d.get("cleaned"), "rois": d.get("rois", [])}


def plan(db, p: dict, chosen, joint: bool, glomeruli: set) -> dict:
    """The names of a prediction's ROIs with the candidates `chosen` (their ids) merged in: with
    `joint`, all named again together (by the atlas, as the model names a fly's ROIs); else the
    predicted ROIs' kept, and the candidates named among the glomeruli left (as when all are
    merged). All of them together, and any with names kept, are as the look left them; some of
    them named again together, the fit worker names when asked (`call`).

    Returns:
        ``names`` ({label: name or None}, of the predicted ROIs and the candidates chosen) and
        ``kinds`` (as `found`).

    Raises:
        LookupError: the prediction has no look done.
        ValueError: `chosen` is not a list of its candidates' ids.
        CallError: the fit worker did not name them (it is not running, or failed).
    """
    d = _read(p) if (p.get("nmf") or {}).get("status") == "done" else None
    if d is None:
        raise LookupError("no look for missed glomeruli done for that prediction")
    ids = {c["id"] for c in d["candidates"]}
    if not isinstance(chosen, list) or any(isinstance(c, bool) or not isinstance(c, int) or c not in ids for c in chosen):
        raise ValueError("chosen: a list of its candidates' ids")
    chosen = sorted(set(chosen))
    out = {c["label"] for c in d["candidates"] if c["id"] not in chosen and c["kind"] != "extended"}
    if set(chosen) == ids or not joint:
        names = {l: n for l, n in d["names"]["joint" if joint else "kept"].items() if int(l) not in out}
    else:
        names = call(db, "nmf_plan", {"prediction": p["id"], "chosen": chosen, "run": d["finished"]})["names"]
    return {"names": names, "kinds": _kinds(names.values(), glomeruli)}


# ------------------------------------------------------ in the fit worker

def _outlines(mask) -> list[list[list[float]]]:
    from alseg import roi_io

    return [[[round(float(x), 2), round(float(y), 2)] for x, y in o] for o in roi_io._outlines(mask)]


def predicted_names(p: dict) -> dict[int, Optional[str]]:
    """A prediction's ROIs' names by label (None: one left unnamed), as al-diag-seg wrote them
    with its final ROIs (``<fly>_identities*.csv``)."""
    import csv

    d, final = predictions.pred_dir(p["id"]), (p.get("result") or {}).get("final")
    for path in (d / f"{p['fly']}_identities_{final}.csv", d / f"{p['fly']}_identities.csv"):
        if path.is_file():
            with open(path, newline="") as fh:
                return {int(r["label"]): r["assigned"] or None for r in csv.DictReader(fh)}
    raise FileNotFoundError(f"the prediction's ROIs' names ({p['fly']}_identities.csv) are not in its folder")


def run(db, p: dict, model, fly: dict, on_gpu: bool, say=print, progress=None) -> dict:
    """Makes a prediction's look (in the fit worker, which has al-diag-seg): finds the candidates
    in the fly's movies, cleaned as the prediction's ROIs were (its options' steps that need
    nothing of Cellpose), names them, and writes the look's files (the module's docstring).

    Args:
        model: the prediction's model (`alseg.model.Model`), with the fly's odors as recorded known to it.
        fly: the fly, as `alseg.fly.load_fly` gives it.
        progress: called as ``progress(stage, done, total)``, stage one of `STAGES` (done and total
            None where it cannot say).

    Returns:
        the look's result: ``candidates`` (how many), ``kinds`` ({kind: how many}), ``planes``
        (how many were looked at) and ``time_points`` (of the movies read).
    """
    import numpy as np
    import tifffile
    from alseg import atlas as atlas_mod
    from alseg import evidence, roi_io
    from alseg import model as am
    from alseg import nmf as anmf

    f = files(p)
    final = p["result"]["final"]
    d = predictions.pred_dir(p["id"])
    labels = tifffile.imread(d / f"{p['fly']}_{final}.tif")
    # (cleaned ROIs keep the names of those they were cleaned from, Cellpose's own or those split: named again, it is by those)
    base = next((f"{p['fly']}_{k}.tif" for k in ("split", "pred") if (d / f"{p['fly']}_{k}.tif").is_file()), None)
    named = base if final.startswith("clean") else None
    names = predicted_names(p)
    tell = progress or (lambda *a: None)
    prior, prior_names = None, None
    if model.atlas is not None:
        prior_names = list(model.atlas.names)
        if f["prior"].is_file():
            z = np.load(f["prior"])
            if [str(n) for n in z["names"]] == prior_names and z["prior"].shape[1:] == labels.shape:
                prior = z["prior"].astype(np.float32)
        if prior is None:
            tell("register", None, None)
            say("registering the model's atlas to the fly (the prediction did not keep it)")
            o = model.options
            if fly.get("hires") and fly["hires"].get("mirror") is None:
                atlas_mod.orient_hires(fly, o.hires_reg)
            prior = atlas_mod.atlas_to_fly(model.atlas, fly, transform=o.reg, hires_transform=o.hires_reg)
            am.write_prior(f["prior"], prior, prior_names)
    tell("read", 0, 1)
    mv = evidence.frames(fly["dir"], labels.shape, progress=lambda a, b: tell("read", a, b))
    say(f"movies of {len(mv['recordings'])} recordings ({', '.join(mv['recordings'])}): {len(mv['dff'])} time points of {len(mv['trials'])} trials")
    po = refinement.options_of(p)  # (how its ROIs were refined, if they were)
    steps = [s for s in po.get("clean") or [] if s != "core"]  # (core goes by Cellpose's probabilities, which candidates have none of)
    res = anmf.check(model, fly, labels, names, prior=prior, steps=steps, clean_params={**model.options.clean_params, **(po.get("clean_params") or {})},
                     movies=mv, progress=lambda stage, a, b: tell("name" if stage == "name" else "factorize", a, b),
                     named=tifffile.imread(d / named) if named else None, **((p.get("nmf") or {}).get("settings") or {}))
    merged, cand = res["labels"], res["candidate"]
    out = []
    for c in res["candidates"]:
        own = (cand == c["id"]) & (merged == c["label"])
        item = {**c, "name": res["names"]["kept"].get(c["label"]), "of_name": names.get(c["of"]) if c["of"] is not None else None,
                "outlines": [{"z": int(z), "points": pts} for z in c["planes"] for pts in _outlines(own[z])]}
        if c["kind"] == "split":
            item["rest"] = [{"z": int(z), "points": pts} for z in c["planes"] for pts in _outlines(merged[z] == c["of"])]
        out.append(item)
    am.write_volume(f["labels"], merged.astype(np.uint16))
    am.write_volume(f["candidates"], cand.astype(np.uint16))
    shown = np.where(cand > 0, merged, 0)
    for c in res["candidates"]:  # (more of an ROI: under its own label there, to be an ROI of its own in the RoiSet)
        if c["kind"] == "extended":
            shown[(cand == c["id"])] = c["label"]
    if out:
        roi_io.write_roiset(shown, {int(l): n for l, n in res["names"]["kept"].items()}, f["roiset"])
    else:
        f["roiset"].unlink(missing_ok=True)
    np.savez_compressed(f["inputs"], responses=fly["responses"].astype(np.float32), spacing=np.array(fly["spacing"], float),
                        odors=np.array(list(fly.get("odors") or model.odors)))
    kinds = {k: sum(c["kind"] == k for c in out) for k in ("added", "extended", "split")}
    doc = {"prediction": p["id"], "fly": p["fly"], "of": f"{p['fly']}_{final}.tif", "named": named or f"{p['fly']}_{final}.tif", "asked_for": {k: (p.get("nmf") or {}).get(k) for k in ("name", "ip", "asked_at")},
           "finished": predictions.now(), "device": "GPU" if on_gpu else "CPU", "params": res["params"], "cleaned": res["cleaned"],
           "movies": {"recordings": mv["recordings"], "time_points": int(len(mv["dff"])), "trials": len(mv["trials"])},
           "planes": res["planes"], "candidates": out,
           "rois": [{**r, "name": names.get(r["label"]) or str(r["label"])} for r in res["rois"]], "predicted": {str(l): n for l, n in names.items()},
           "glomeruli": sorted(set(model.template) | set(prior_names or [])),
           "names": {k: {str(l): n for l, n in v.items()} for k, v in res["names"].items()},
           "files": {"labels": f["labels"].name, "candidates": f["candidates"].name, "rois": f["roiset"].name if out else None,
                     "atlas": f["prior"].name if prior is not None else None}}
    tmp = f["json"].with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(f["json"])
    say(f"{len(out)} candidates: {kinds['added']} added, {kinds['extended']} more of an ROI, {kinds['split']} cut off an ROI"
        + (f"; cleaned as the prediction's ROIs: {', '.join(res['cleaned'])}" if res["cleaned"] else ""))
    return {"candidates": len(out), "kinds": kinds, "planes": len(res["planes"]), "time_points": int(len(mv["dff"]))}


def run_plan(db, args: dict) -> dict:
    """In the fit worker: names a prediction's ROIs again, all together, with the candidates
    ``chosen`` merged in (`plan`), from what its look left in its folder. ``run``: the look the
    ids are of (when it finished), so that candidates of a look made since are not taken for them."""
    import numpy as np
    import tifffile
    from alseg import model as am
    from alseg import nmf as anmf

    p = predictions.get(db, int(args["prediction"]))
    d = _read(p) if p and (p.get("nmf") or {}).get("status") == "done" else None
    if d is None or d["finished"] != args.get("run"):
        raise ValueError("that look for missed glomeruli is no longer there (made again since): view the prediction again")
    f = files(p)
    labels = tifffile.imread(predictions.pred_dir(p["id"]) / d.get("named", d["of"]))  # (the ROIs as they were named)
    merged, cand = tifffile.imread(f["labels"]), tifffile.imread(f["candidates"])
    inputs = np.load(f["inputs"])
    model = am.Model.load(fits.fit_dir(p["fit_id"]) / "model", atlas=False)
    prior, prior_names = None, None
    if f["prior"].is_file() and d["files"].get("atlas"):
        z = np.load(f["prior"])
        prior, prior_names = z["prior"].astype(np.float32), [str(n) for n in z["names"]]
    fly = {"responses": inputs["responses"], "spacing": tuple(float(s) for s in inputs["spacing"]),
           # (those of the model's odors the fly has: looks made before a fly could lack any have them all)
           "odors": [str(x) for x in inputs["odors"]] if "odors" in inputs else None}
    ids = anmf.name(model, fly, anmf.subset(labels, merged, cand, args["chosen"]), prior, prior_names)[0]
    return {"names": {str(l): v[0] for l, v in ids.items()}}


# ------------------------------------- what the web server asks of the worker

def call(db, kind: str, args: dict, timeout: float = CALL_TIMEOUT_S) -> dict:
    """Asks the fit worker to do something quick (`CALLS`), and waits for it.

    Raises:
        CallError: the worker is not running, did not do it in `timeout` seconds, or failed (why).
    """
    if not fits.worker_status(db).get("alive"):
        raise CallError("the fit worker is not running")
    with db.connect() as con:
        con.execute("DELETE FROM worker_calls WHERE created < ?", (time.time() - CALL_STALE_S,))
        call_id = con.execute("INSERT INTO worker_calls (kind, args_json, created, status) VALUES (?,?,?,'queued')",
                              (kind, json.dumps(args), time.time())).lastrowid
    until = time.time() + timeout
    try:
        while time.time() < until:
            time.sleep(CALL_POLL_S)
            with db.connect() as con:
                r = con.execute("SELECT status, result_json, error FROM worker_calls WHERE id = ?", (call_id,)).fetchone()
            if r is None:
                break
            if r["status"] == "done":
                return json.loads(r["result_json"])
            if r["status"] == "failed":
                raise CallError(r["error"] or "the fit worker could not do it")
        raise CallError(f"the fit worker did not answer in {timeout:.0f} s")
    finally:
        with db.connect() as con:
            con.execute("DELETE FROM worker_calls WHERE id = ?", (call_id,))


CALLS = {"nmf_plan": run_plan}


def answer(db) -> bool:
    """In the fit worker: does the oldest thing asked of it (`call`), if any. Returns whether there was one."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT id, kind, args_json FROM worker_calls WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
        if r is None:
            return False
        con.execute("UPDATE worker_calls SET status = 'running' WHERE id = ?", (r["id"],))
    try:
        result, error = CALLS[r["kind"]](db, json.loads(r["args_json"])), None
    except Exception as e:  # noqa: BLE001 (said to whoever asked)
        result, error = None, str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
    with db.connect() as con:
        con.execute("UPDATE worker_calls SET status = ?, result_json = ?, error = ? WHERE id = ?",
                    ("failed" if error else "done", json.dumps(result) if result is not None else None, error, r["id"]))
    return True
