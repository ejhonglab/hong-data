"""A prediction's ROIs refined: kept from their neighbours by the steps of `predictions.CLEAN` (trim, margin, core)
and, before them, split where a mask holds two glomeruli (al-diag-seg's `Model.refine`). Asked for with the
prediction (its options), or after, from the AL segmentation tab's list of predictions; either way the fit worker
does it the same way, from what the prediction kept (`run`): the model's own ROIs and their names, the atlas on the
fly's grid (registered again if it was not kept) and Cellpose's cell probabilities (for core: kept by predictions
since 2026-11-16 alone, so older ones can be refined by the other steps only).

A prediction has one refinement. Its state is the prediction's ``refine`` (`state`: ``status``, who asked and when,
how, how far it is, what it made). Asked for again, the one before is undone at once (`undo`), and the new one made
from the model's own ROIs; predicted again, it is gone with the prediction's files. While a prediction has one done,
its ``result`` is of the refined ROIs (`predictions.with_refined`). Its files are in the prediction's folder:

- ``<fly>_split.tif``, ``<fly>_split_RoiSet.zip`` (if split was asked for): the model's own with the masks over two
  glomeruli cut in two, the planes' masks joined again and named again; ``<fly>_clean.*``: those after the other
  steps (names kept); each with its ``<fly>_identities_<kind>.csv`` and ``<fly>_identity_terms_<kind>.json``;
- ``<fly>_RoiSet.zip``: the refined ones (the last of those), which are then the prediction's ROIs to use, while
  the model's own stay as ``<fly>_pred*`` (and are its ROIs to use again if the refinement fails, or is asked
  for again);
- ``<fly>_refinement.json``: how it was asked for and made, what each kind of ROIs scores (`run`'s ``kinds``), and
  what was done to each refined ROI: on which planes it is a part of a mask cut in two (and which), and where trim
  cut its edge back (keeping what share of its area).

Refining a prediction's ROIs (or undoing that) makes a look for glomeruli they miss (`nmf_check`) out of date: it
is deleted. Its ROIs as edited in the viewer stay as they are (a copy of those it had then).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from . import predictions

STATUSES = ("queued", "running", "done", "failed", "interrupted")
ACTIVE = ("queued", "running")
STAGES = {"load": "loading the fly", "register": "registering the atlas to it", "refine": "refining its ROIs",
          "score": "scoring them"}
MADE = ("split", "clean")  # the kinds of ROIs a refinement makes (beside the model's own, "pred")
CELLPROB = "cellprob"  # in the name of a prediction's Cellpose's cell probabilities (`files`)


def files(p: dict) -> dict[str, Path]:
    """The files of a prediction that a refinement reads and writes, in its folder."""
    d, fly = predictions.pred_dir(p["id"]), p["fly"]
    return {"pred": d / f"{fly}_pred.tif", "pred_rois": d / f"{fly}_pred_RoiSet.zip", "cellprob": d / f"{fly}_{CELLPROB}.tif",
            "prior": d / f"{fly}_atlas_prior.npz", "json": d / f"{fly}_refinement.json", "roiset": predictions.roiset(p)}


def kind_files(p: dict, kind: str) -> list[Path]:
    """A kind of a prediction's ROIs' files: labels, RoiSet, names and what they cost."""
    d, fly = predictions.pred_dir(p["id"]), p["fly"]
    return [d / f"{fly}_{kind}.tif", d / f"{fly}_{kind}_RoiSet.zip", d / f"{fly}_identities_{kind}.csv", d / f"{fly}_identity_terms_{kind}.json"]


def asked(opts: Optional[dict]) -> bool:
    """Whether options (`predictions.options`) ask for any refining."""
    return bool(opts and (opts.get("split") or opts.get("clean")))


def can(p: dict) -> dict:
    """What can be refined of a prediction done, by its files: ``any`` (whether the model's own ROIs are there) and
    ``core`` (whether Cellpose's cell probabilities were kept, which core goes by)."""
    f = files(p)
    return {"any": f["pred"].is_file() and f["pred_rois"].is_file(), "core": f["cellprob"].is_file()}


def _set_roiset(p: dict, kind: str) -> None:
    """Makes a kind of a prediction's ROIs the ones to use (its ``<fly>_RoiSet.zip``)."""
    d, f = predictions.pred_dir(p["id"]), files(p)
    tmp = d / f".{p['fly']}_RoiSet.tmp.zip"
    shutil.copyfile(d / f"{p['fly']}_{kind}_RoiSet.zip", tmp)
    os.replace(tmp, f["roiset"])


def _drop_nmf(db, p: dict) -> bool:
    """Deletes a prediction's look for glomeruli its ROIs miss (`nmf_check`), if one is there and not under way:
    made of other ROIs than it now has. Returns whether there was one."""
    from . import nmf_check  # (it imports this module)

    if (p.get("nmf") or {}).get("status") in (None, *ACTIVE):
        return False
    nf = nmf_check.files(p)
    for k in ("json", "labels", "candidates", "roiset", "inputs"):  # (not the atlas on the fly's grid: the prediction's)
        nf[k].unlink(missing_ok=True)
    with db.connect() as con:
        con.execute("UPDATE predictions SET nmf_status = NULL, nmf_json = NULL WHERE id = ? AND nmf_status NOT IN ('queued', 'running')",
                    (p["id"],))
    return True


def undo(db, p: dict) -> None:
    """Makes a prediction's ROIs to use the model's own again, its refinement's files (or what one that failed left)
    deleted. One refined as it was made before refinements were kept apart (`state`'s ``legacy``) has its result made
    the model's own's too. A look for glomeruli the refined ROIs miss is deleted."""
    f = files(p)
    if not f["pred_rois"].is_file():
        return
    was = state(p)
    _set_roiset(p, "pred")
    for kind in MADE:
        for x in kind_files(p, kind):
            x.unlink(missing_ok=True)
    f["json"].unlink(missing_ok=True)
    if was and was.get("legacy"):
        res = dict(p.get("result") or {})
        own = ((res.get("scores") or {}).get("kinds") or {}).get("pred")
        names = pred_names(p) or {}
        res.update(final="pred", named=sorted({v[0] for v in names.values() if v[0]}), n=(own or {}).get("n", len(names)))
        if res.get("scores") is not None:
            res["scores"] = {k: v for k, v in {**res["scores"], **{k: v for k, v in (own or {}).items() if k != "n"}}.items() if k != "kinds"}
        predictions.update(db, p["id"], result=res)
    if was and was["status"] == "done":
        _drop_nmf(db, p)


# -------------------------------------------------------------------- state

def _set(con, pred_id: int, status: Optional[str], **fields) -> None:
    r = con.execute("SELECT refine_status, refine_json FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    if r is None or r["refine_status"] is None:  # (predicted again meanwhile: the refinement is gone with its files)
        return
    d = {**(json.loads(r["refine_json"]) if r["refine_json"] else {}), **fields}
    if status is None:
        con.execute("UPDATE predictions SET refine_json = ? WHERE id = ?", (json.dumps(d), pred_id))
    else:
        con.execute("UPDATE predictions SET refine_status = ?, refine_json = ? WHERE id = ?", (status, json.dumps(d), pred_id))


def update(db, pred_id: int, status: Optional[str] = None, **fields) -> None:
    """Sets fields of a prediction's refinement (kept with the others), and its status if given; nothing of a
    prediction with none (predicted again since it was started)."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        _set(con, pred_id, status, **fields)


def start(db, p: dict, opts: dict) -> None:
    """Records a refinement as running, as a prediction does it as it is made (its options asked for one), asked for
    by whoever asked for the prediction, when they did."""
    with db.connect() as con:
        con.execute("UPDATE predictions SET refine_status = 'running', refine_json = ? WHERE id = ?",
                    (json.dumps({"name": p["name"], "ip": p["ip"], "asked_at": p["created_at"], "started_at": predictions.now(),
                                 "options": opts, "with_prediction": True}), p["id"]))


def ask(db, pred_id: int, name: str, ip: Optional[str], opts: dict) -> tuple[dict, bool]:
    """Asks for a prediction's ROIs to be refined as ``opts`` say (`predictions.options`, at least one step): queued,
    unless a refinement of it is queued or running already. One made before is undone at once (`undo`).

    Returns:
        (the prediction, whether it was queued now).

    Raises:
        LookupError: no such prediction done.
        ValueError: nothing asked for, a prediction without the model's own ROIs, core asked of one that did not
            keep what it goes by, or glomeruli its ROIs miss being looked for.
    """
    if not asked(opts):
        raise ValueError("refine: ask for at least one step (split, or a step of clean)")
    p = predictions.get(db, pred_id)
    if p is None or p["status"] != "done":
        raise LookupError("no such prediction done")
    if (p["refine"] or {}).get("status") in ACTIVE:
        return p, False
    if (p["nmf"] or {}).get("status") in ACTIVE:
        raise ValueError("glomeruli its ROIs miss are being looked for: wait for that to end")
    have = can(p)
    if not have["any"]:
        raise ValueError("its folder has not the model's own ROIs to refine: predict it again")
    if "core" in opts["clean"] and not have["core"]:
        raise ValueError("core goes by Cellpose's cell probabilities, which this prediction did not keep (only those made "
                         "since 2026-11-16 do): predict it again, or refine it by the other steps")
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT refine_status FROM predictions WHERE id = ?", (pred_id,)).fetchone()
        if r is None:
            raise LookupError("no such prediction done")
        if r["refine_status"] in ACTIVE:  # (asked for meanwhile)
            return predictions.get(db, pred_id), False
        con.execute("UPDATE predictions SET refine_status = 'queued', refine_json = ? WHERE id = ?",
                    (json.dumps({"name": name, "ip": ip, "asked_at": predictions.now(), "options": opts}), pred_id))
    undo(db, p)
    return predictions.get(db, pred_id), True


def claim(db) -> Optional[dict]:
    """The prediction whose refinement was asked for first, of those queued, now running (for the worker), or None."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        rows = con.execute("SELECT id, refine_json FROM predictions WHERE refine_status = 'queued' AND status = 'done'").fetchall()
        if not rows:
            return None
        first = min(rows, key=lambda r: (json.loads(r["refine_json"] or "{}").get("asked_at") or "", r["id"]))
        _set(con, first["id"], "running", started_at=predictions.now(), progress=None, error=None)
    return predictions.get(db, first["id"])


def interrupt_running(db) -> list[int]:
    """Marks refinements left running (by a worker that stopped) interrupted, their predictions' ROIs the model's own.
    Returns their predictions' ids."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        rows = con.execute("SELECT id, status FROM predictions WHERE refine_status = 'running'").fetchall()
        for r in rows:
            _set(con, r["id"], "interrupted", finished_at=predictions.now(), progress=None,
                 error="the fit worker stopped (restarted, or the server did) while it ran")
    for r in rows:
        if r["status"] == "done":
            undo(db, predictions.get(db, r["id"]))
    return [r["id"] for r in rows]


def state(p: dict) -> Optional[dict]:
    """A prediction's refinement, as the tab shows it: its ``status``, who asked (``name``) and when, how
    (``options``: `predictions.options`), whether it was made with the prediction (``with_prediction``), its
    ``progress`` (of one running: ``stage``, of `STAGES`), ``result`` (of one done: ``steps`` done, ``final``,
    ``n``, ``named``, ``split`` and ``trimmed`` (how many refined ROIs were), ``kinds`` (each kind's scores:
    `run`), ``seconds``) and ``error``; None if none was asked for. A prediction refined as it was made, before
    refinements were kept apart, has one done (``legacy``: its model's own ROIs were split already, if split
    was asked for), from its result."""
    r = p.get("refine")
    if r:
        return {k: r.get(k) for k in ("status", "name", "asked_at", "started_at", "finished_at", "options", "with_prediction",
                                      "progress", "result", "error")}
    res, opts = p.get("result") or {}, p.get("options") or {}
    if p["status"] != "done" or not asked(opts) or res.get("final") != "clean":
        return None
    kinds = {k: v for k, v in ((res.get("scores") or {}).get("kinds") or {}).items() if k in ("pred", "clean")}
    return {"status": "done", "name": p["name"], "asked_at": p["created_at"], "started_at": p["started_at"], "finished_at": p["finished_at"],
            "options": opts, "with_prediction": True, "legacy": True, "progress": None, "error": None,
            "result": {"steps": [*(["split"] if opts.get("split") else []), *opts.get("clean", [])], "final": "clean", "n": res.get("n"),
                       "named": res.get("named"), "split": None, "trimmed": None, "kinds": kinds or None, "seconds": None}}


def options_of(p: dict) -> dict:
    """How a prediction's ROIs to use were made (`predictions.options`): as its refinement done asked (none: the
    model's own)."""
    st = state(p)
    return st["options"] if st and st["status"] == "done" else predictions.options({})


def details(p: dict) -> Optional[dict]:
    """What a prediction's refinement done did to its ROIs (its ``<fly>_refinement.json``), for the viewer: ``rois``
    ({name in its RoiSet: {plane (from 0, as a string): {``split_from`` (the name of the model's ROI it is a part of,
    there), ``trimmed`` (the share of its area there trim kept)}}}), ``steps``, ``legacy`` (`state`'s) and
    ``unrefined`` (the model's own ROIs: their RoiSet); None if it has none."""
    st = state(p)
    if not st or st["status"] != "done":
        return None
    f = files(p)
    try:
        doc = json.loads(f["json"].read_text())
    except (OSError, ValueError):
        doc = {}
    rois = {}
    for r in (doc.get("rois") or {}).values():
        planes = {}
        for z in r.get("split") or {}:
            planes.setdefault(z, {})["split_from"] = r["of_names"].get(z)
        for z, kept in (r.get("trimmed") or {}).items():
            planes.setdefault(z, {})["trimmed"] = kept
        if planes:
            rois[r["name"]] = planes
    return {"rois": rois, "steps": (st.get("result") or {}).get("steps") or [], "legacy": bool(st.get("legacy")),
            "unrefined": str(f["pred_rois"]) if f["pred_rois"].is_file() else None}


def listing_fields(p: dict) -> dict:
    """What the tab's list of predictions says of a prediction's refinement: ``refine`` (`state`) and, of one done,
    ``refinable`` (`can`)."""
    return {"refine": state(p), "refinable": can(p) if p["status"] == "done" else None}


# ------------------------------------------------------ in the fit worker

def _read_names(path: Path) -> dict:
    """{label: (name or None, cost, confidence)} from an identities CSV of al-diag-seg's (or `write_kind`'s)."""
    num = lambda v: float(v) if v not in ("", None) else float("nan")  # noqa: E731
    with open(path, newline="") as fh:
        return {int(r["label"]): (r["assigned"] or None, num(r.get("cost")), num(r.get("confidence"))) for r in csv.DictReader(fh)}


def pred_names(p: dict) -> Optional[dict]:
    """The names the model gave its own ROIs in a prediction ({label: (name or None, cost, confidence)}), as its
    folder has them; None if it has them not (a prediction of a fly without ROIs of its own, made before each kind's
    were written, whose ROIs to use were not the model's own)."""
    d, fly = predictions.pred_dir(p["id"]), p["fly"]
    if (d / f"{fly}_identities_pred.csv").is_file():
        return _read_names(d / f"{fly}_identities_pred.csv")
    if (d / f"{fly}_identities.csv").is_file() and (not asked(p.get("options")) or (p.get("result") or {}).get("final") == "pred"):
        return _read_names(d / f"{fly}_identities.csv")
    return None


def write_kind(d: Path, fly: str, kind: str, labels, ids: dict, terms: Optional[dict]) -> None:
    """Writes a kind of a fly's ROIs as al-diag-seg does: labels, RoiSet (named), identities and what they cost."""
    import numpy as np
    from alseg import model as am
    from alseg import roi_io

    am.write_volume(d / f"{fly}_{kind}.tif", labels.astype(np.uint16))
    roi_io.write_roiset(labels, {int(l): v[0] for l, v in ids.items()}, d / f"{fly}_{kind}_RoiSet.zip")
    with open(d / f"{fly}_identities_{kind}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "assigned", "cost", "confidence"])
        w.writerows([l, nm or "", f"{cost:.3f}", f"{conf:.3f}"] for l, (nm, cost, conf) in sorted(ids.items()))
    if terms is not None:
        am.write_identity_terms(d / f"{fly}_identity_terms_{kind}.json", terms)


def _kind_scores(fly: dict, labels, ids: dict, kind: str, ref, manual: Optional[dict]) -> dict:
    """A kind's scores, as the tab shows them (`fit_worker._scores`, with ``n``, and ``named``/``named_of`` of a fly
    with ROIs of its own)."""
    from alseg import model as am

    from . import fit_worker  # (it imports this module)

    s = am.score(fly, labels, ids, kind, ref)
    out = {"n": s["row"]["n"], **fit_worker._scores(s["row"], manual or {})}
    if s["identity"] is not None:
        out.update(named=s["identity"]["correct"], named_of=s["identity"]["total"])
    return out


def run(db, p: dict, model, fly: dict, opts: dict, say=print, progress=None) -> dict:
    """Refines a prediction's ROIs (in the fit worker, which has al-diag-seg) as ``opts`` say (`predictions.options`),
    from what the prediction kept (the module's docstring), writes the refinement's files and makes its ROIs those
    refined, deleting a look for glomeruli they miss made of the model's own. If it fails, the prediction's ROIs are
    left the model's own (`undo`, which whoever calls this does).

    Args:
        model: the prediction's model (`alseg.model.Model`), with the fly's odors as recorded known to it.
        fly: the fly, as `alseg.fly.load_fly` gives it (with its ``labels``, if it has ROIs of its own: they are
            then scored against them).
        progress: called as ``progress(stage)``, stage one of `STAGES`.

    Returns:
        the refinement's ``result`` (`state`'s, but for ``seconds``).
    """
    import numpy as np
    import tifffile
    from alseg import model as am

    tell = progress or (lambda *a: None)
    t0 = time.time()
    f, d, name = files(p), predictions.pred_dir(p["id"]), p["fly"]
    pred = tifffile.imread(f["pred"])
    ids = pred_names(p)
    prior = prior_names = None
    if model.atlas is not None:
        prior_names = list(model.atlas.names)
        if f["prior"].is_file():
            z = np.load(f["prior"])
            if [str(n) for n in z["names"]] == prior_names and z["prior"].shape[1:] == pred.shape:
                prior = z["prior"].astype(np.float32)
        if prior is None and (opts.get("split") or ids is None):
            tell("register")
            say("registering the model's atlas to the fly (the prediction did not keep it)")
            prior = model.prior_of(fly)
            am.write_prior(f["prior"], prior, prior_names)
    tell("refine")
    if ids is None:
        say("naming the model's own ROIs again (the prediction did not keep their names)")
        ids = model.name(fly, pred, prior, prior_names)[0]
    if f["cellprob"].is_file():
        fly["cellprob"] = tifffile.imread(f["cellprob"]).astype(np.float32)
    # (what the model's names cost, which cleaned ROIs keep: as al-diag-seg wrote them)
    own = p.get("result") or {}
    terms = next((json.loads(x.read_text()) for x in (d / f"{name}_identity_terms_pred.json", d / f"{name}_identity_terms.json")
                  if x.is_file() and (x.name.endswith("_pred.json") or own.get("final") == "pred")), None)
    r = model.refine(fly, pred, ids, terms, prior, prior_names, split=bool(opts.get("split")), steps=list(opts.get("clean") or []),
                     params=opts.get("clean_params") or {})
    for kind, lab in r["labels"].items():
        write_kind(d, name, kind, lab, r["identities"][kind], r["identity_terms"][kind])
    final = r["final"]
    tell("score")
    manual = am.manual_row(fly) if fly.get("labels") is not None else None
    base = r["labels"].get("split", pred)
    kinds = {"pred": _kind_scores(fly, pred, ids, "pred", None, manual)}
    for kind, lab in r["labels"].items():
        kinds[kind] = _kind_scores(fly, lab, r["identities"][kind], kind, base if kind == "clean" else None, manual)
    # what was done to each refined ROI, by plane
    fin = r["identities"][final]
    rois = {}
    for l in sorted(set(r["report"]["split"]) | set(r["report"]["trimmed"])):
        split = {str(z): of for z, of in r["report"]["split"].get(l, {}).items()}
        rois[str(l)] = {"name": fin.get(l, (None,))[0] or str(l),  # (its name in the RoiSet)
                        "split": split, "of_names": {z: ids.get(of, (None,))[0] or str(of) for z, of in split.items()},
                        "trimmed": {str(z): round(k, 4) for z, k in r["report"]["trimmed"].get(l, {}).items()}}
    named = sorted({v[0] for v in fin.values() if v[0]})
    result = {"steps": r["steps"], "final": final, "n": int(len(np.unique(r["labels"][final])) - 1), "named": named,
              "split": sum(bool(x["split"]) for x in rois.values()), "trimmed": sum(bool(x["trimmed"]) for x in rois.values()),
              "kinds": kinds}
    rf = p.get("refine") or {}
    doc = {"prediction": p["id"], "fly": name, "asked_for": {k: rf.get(k) for k in ("name", "ip", "asked_at")}, "finished": predictions.now(),
           "options": opts, "steps": r["steps"], "final": final,
           "files": {"rois": f["roiset"].name, "unrefined": f["pred_rois"].name, **{k: f"{name}_{k}_RoiSet.zip" for k in r["labels"]}},
           "kinds": kinds, "rois": rois}
    tmp = f["json"].with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(f["json"])
    _set_roiset(p, final)
    if _drop_nmf(db, predictions.get(db, p["id"]) or p):
        say("the look for glomeruli its ROIs miss (made of the model's own) deleted: out of date")
    say(f"refined ({', '.join(r['steps']) or 'nothing could run'}): {result['n']} ROIs, {len(named)} named; {result['split']} of them parts "
        f"of masks cut in two, {result['trimmed']} trimmed; {time.time() - t0:.0f} s")
    return result
