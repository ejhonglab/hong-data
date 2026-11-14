"""Glomerulus ROIs predicted for a fly by a fit's model (`fits`), asked for in the AL
segmentation tab, and run by the fit worker (`fit_worker.run_prediction`), beside fits: on
the GPU when no fit is using it, else on the CPU.

A prediction is a row of the ``predictions`` table, one per fit and fly (its motion-corrected
data folder, on tensor): asked for again, it is run again, its files replaced. Its folder
(`pred_dir`) has:

- ``<fly>_RoiSet.zip``: the ROIs to use (the model's final ones), named by glomerulus where
  one fits (numbered where none does), per plane, for ImageJ and al_analysis;
- ``<fly>_<kind>.tif`` and ``<fly>_<kind>_RoiSet.zip``: each kind of ROIs al-diag-seg makes
  (``pred``: Cellpose's; ``refined_tuning``; ``refined_prior``, with the atlas), as labels and
  RoiSets, and the names given (``<fly>_identities*.csv``);
- ``average.tif``: the fly's average image, as the model saw it (for the tab's viewer);
- where the fly has ROIs of its own, the scores of each kind against them (``summary.csv``,
  ``glomeruli.csv``);
- ``provenance.json`` (the model, its fit and the al-diag-seg it was made with; the fly's
  files read, with their sizes and times; the software that predicted; who asked, and when),
  ``README.txt`` and ``log.txt``;
- once its ROIs were edited in the tab's viewer (`save_edited`), ``<fly>_pred-edited_RoiSet.zip``:
  a copy of ``<fly>_RoiSet.zip``'s ROIs (which stay as predicted) as edited there, and
  ``<fly>_pred-edited.json``, saying when, by whom, and how they were simplified.

When one waiting or running should end is estimated from how long those done like it took
(`estimate`), which each keeps in its result.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import shutil
import statistics
import threading
import time
from pathlib import Path
from typing import Optional

from . import config, fits, movies, roi_training, sandbox, tensor

STATUSES = ("queued", "running", "done", "failed", "interrupted")
ACTIVE = ("queued", "running")
PROVENANCE = "provenance.json"
AVERAGE = "average.tif"
RECENT = 10  # the predictions done that one's time is estimated from (the last ones like it, on its device)
STAGES = ("load", "predict")  # what a prediction's time is kept of (its result's ``stages``: seconds of each)
KIND = ("stack", "scored")  # what its time depends on, beside its device (`_kind`)
GUESSES = {"GPU": 120.0, "CPU": 900.0}  # seconds one takes, until one is done on that device to estimate from


EDITED = "pred-edited"  # in the names of a prediction's ROIs as edited in the viewer (`save_edited`)
MAX_EDITED_ROIS, MAX_EDITED_POINTS = 5000, 1_000_000
_edit_lock = threading.Lock()  # (the web server is one process)


class Stale(Exception):
    """A prediction's edited ROIs were changed by someone else since the revision given: they are
    now at ``revision`` (``edited``: as `edited_info` gives them, None if deleted)."""

    def __init__(self, revision: int, edited: Optional[dict]):
        super().__init__("these ROIs were changed elsewhere since they were opened here")
        self.revision, self.edited = revision, edited


class Exists(Exception):
    """A prediction of that fit's model for that fly there is already (``prediction``)."""

    def __init__(self, prediction: dict):
        super().__init__("that model has predicted that fly's ROIs already")
        self.prediction = prediction


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def root() -> Path:
    return config.DATA_DIR / "predictions"


def pred_dir(pred_id: int) -> Path:
    return root() / str(int(pred_id))


def roiset(p: dict) -> Path:
    """A prediction's ROIs to use (its model's final ones)."""
    return pred_dir(p["id"]) / f"{p['fly']}_RoiSet.zip"


# ------------------------------------------------------------------- records

def _row(r) -> dict:
    d = dict(r)
    for k in ("progress", "result"):
        v = d.pop(f"{k}_json")
        d[k] = json.loads(v) if v else None
    return d


def get(db, pred_id: int) -> Optional[dict]:
    with db.connect() as con:
        r = con.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    return _row(r) if r else None


def all_predictions(db) -> list[dict]:
    with db.connect() as con:
        return [_row(r) for r in con.execute("SELECT * FROM predictions ORDER BY created_at DESC, id DESC")]


def update(db, pred_id: int, **fields) -> None:
    """Sets columns of a prediction (``*_json`` ones given as their names without it, as values)."""
    cols, vals = [], []
    for k, v in fields.items():
        if k in ("progress", "result"):
            k, v = f"{k}_json", None if v is None else json.dumps(v)
        cols.append(f"{k} = ?")
        vals.append(v)
    with db.connect() as con:
        con.execute(f"UPDATE predictions SET {', '.join(cols)} WHERE id = ?", (*vals, pred_id))


def create(db, fit_id: int, fly: dict, name: str, ip: Optional[str], overwrite: bool = False) -> tuple[dict, bool]:
    """Asks for the ROIs of `fly` (as `locate` gives it) by fit `fit_id`'s model. One asked
    for again (``overwrite``) is queued again, as asked for now, its files replaced when it
    runs; one queued or running is left as it is.

    Returns:
        (the prediction, whether it was queued now).

    Raises:
        Exists: if there is one already, and not ``overwrite``.
    """
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT * FROM predictions WHERE fit_id = ? AND fly_dir = ?", (fit_id, fly["dir"])).fetchone()
        if r is not None and r["status"] in ACTIVE:
            return _row(r), False
        if r is not None and not overwrite:
            raise Exists(_row(r))
        if r is None:
            pred_id = con.execute("INSERT INTO predictions (fit_id, fly_dir, raw_dir, fly, name, ip, created_at, status) "
                                  "VALUES (?,?,?,?,?,?,?,?)",
                                  (fit_id, fly["dir"], fly["raw_dir"], fly["name"], name, ip, now(), "queued")).lastrowid
        else:
            pred_id = r["id"]
            con.execute("UPDATE predictions SET raw_dir = ?, fly = ?, name = ?, ip = ?, created_at = ?, status = 'queued', "
                        "started_at = NULL, finished_at = NULL, progress_json = NULL, result_json = NULL, error = NULL "
                        "WHERE id = ?", (fly["raw_dir"], fly["name"], name, ip, now(), pred_id))
    return get(db, pred_id), True


def delete(db, pred_id: int) -> Optional[bool]:
    """Deletes a prediction and its folder. None if there is no such prediction; False if it
    is running."""
    p = get(db, pred_id)
    if p is None:
        return None
    if p["status"] == "running":
        return False
    with db.connect() as con:
        con.execute("DELETE FROM predictions WHERE id = ? AND status != 'running'", (pred_id,))
    shutil.rmtree(pred_dir(pred_id), ignore_errors=True)
    return True


def claim(db) -> Optional[dict]:
    """The oldest queued prediction, now running (for the worker), or None."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT id FROM predictions WHERE status = 'queued' ORDER BY created_at, id LIMIT 1").fetchone()
        if r is None:
            return None
        con.execute("UPDATE predictions SET status = 'running', started_at = ? WHERE id = ?", (now(), r["id"]))
    return get(db, r["id"])


def interrupt_running(db) -> list[int]:
    """Marks predictions left running (by a worker that stopped) interrupted. Returns their ids."""
    with db.connect() as con:
        ids = [r["id"] for r in con.execute("SELECT id FROM predictions WHERE status = 'running'")]
        con.execute("UPDATE predictions SET status = 'interrupted', finished_at = ?, error = ? WHERE status = 'running'",
                    (now(), "the fit worker stopped (restarted, or the server did) while it ran"))
    return ids


def _seconds(p: dict) -> Optional[float]:
    """How long a prediction done took: as its result says, else from when it started and finished."""
    s = (p.get("result") or {}).get("seconds")
    if s is None and p.get("started_at") and p.get("finished_at"):
        s = dt.datetime.fromisoformat(p["finished_at"]).timestamp() - dt.datetime.fromisoformat(p["started_at"]).timestamp()
    return s


def _kind(p: dict) -> dict:
    """What a prediction's time depends on, as far as is known of it (`KIND`, each None where not):
    of one done, from its result (``scored``: whether it has scores; ``stack``: only of those made
    since it was kept); of one running, from its progress, once its fly is loaded."""
    if p["status"] == "done":
        r = p.get("result") or {}
        return {"stack": r.get("stack"), "scored": r.get("scores") is not None}
    pr = p.get("progress") or {}
    return {k: pr.get(k) for k in KIND}


def expected(done: list[dict], device: str, kind: Optional[dict] = None) -> dict:
    """How long a prediction on `device` ('GPU' or 'CPU') should take: the median of the last
    `RECENT` done on it (of `done`, newest first) that are like it in what is known of it
    (`kind`, as `_kind` gives it: an anatomical stack to register, ROIs of its own to score
    against), else of all those on it; and of each of `STAGES`, where all of them say.

    Returns:
        ``seconds``, ``stages`` (seconds of each, or None), ``from`` (how many it is of) and
        ``device``; with none done on that device, `GUESSES`' and 0.
    """
    on = lambda q: (q.get("result") or {}).get("device") or (q.get("progress") or {}).get("device")  # noqa: E731
    pool = [q for q in done if on(q) == device and _seconds(q) is not None]
    known = {k: v for k, v in (kind or {}).items() if v is not None}
    like = [q for q in pool if all(_kind(q)[k] == v for k, v in known.items())]
    pool = (like or pool)[:RECENT]
    if not pool:
        return {"seconds": GUESSES[device], "stages": None, "from": 0, "device": device}
    stages = None
    if all(((q.get("result") or {}).get("stages") or {}).keys() >= set(STAGES) for q in pool):
        stages = {s: statistics.median(q["result"]["stages"][s] for q in pool) for s in STAGES}
    return {"seconds": statistics.median(_seconds(q) for q in pool), "stages": stages, "from": len(pool), "device": device}


def estimate(preds: list[dict], gpu: bool, fit_running: bool, t: Optional[float] = None) -> dict:
    """Gives each prediction of `preds` queued or running its ``eta``: ``seconds`` it has left to
    run, ``until`` it ends (one queued starts when those before it end: one runs at a time),
    ``ends_at``, ``device`` (the one it runs on; of one queued, the one it would get now: the
    GPU, if the worker has one, `gpu`, and no fit is running, `fit_running`) and ``from``: how many
    predictions done that is estimated from (`expected`; 0: a guess). One running has left what
    its stage and those after it take, less how long it has been at it (once they are known),
    else what it takes in all, less how long it has run; nothing, once that is past.

    Returns:
        the same for a prediction asked for now (after all of those).
    """
    t = time.time() if t is None else t
    done = sorted((p for p in preds if p["status"] == "done"), key=lambda p: p.get("finished_at") or "", reverse=True)
    queue_device = "GPU" if gpu and not fit_running else "CPU"
    at = t

    def eta(p: Optional[dict]) -> dict:
        nonlocal at
        running = p is not None and p["status"] == "running"
        pr = (p.get("progress") or {}) if running else {}
        e = expected(done, pr.get("device") or queue_device, _kind(p) if running else None)
        left = e["seconds"]
        if running:
            stage, since = pr.get("stage"), pr.get("stage_started")
            if e["stages"] and stage in STAGES and since:
                later = sum(e["stages"][s] for s in STAGES[STAGES.index(stage) + 1:])
                left = max(0.0, e["stages"][stage] - (t - since)) + later
            elif p.get("started_at"):
                left = max(0.0, e["seconds"] - (t - dt.datetime.fromisoformat(p["started_at"]).timestamp()))
        at += left
        return {"seconds": left, "until": at - t, "from": e["from"], "device": e["device"],
                "ends_at": dt.datetime.fromtimestamp(at).astimezone().isoformat(timespec="seconds")}

    for p in sorted((p for p in preds if p["status"] in ACTIVE), key=lambda p: (p["status"] != "running", p["created_at"], p["id"])):
        p["eta"] = eta(p)
    return eta(None)


def worker_jobs(db) -> dict:
    """What restarting the fit worker would stop: the fits, predictions and extractions of
    signals (`extract`) running, as the web server's `_running_jobs` gives its own (``jobs``,
    each its ``what``, ``started_at``, ``ends_at`` and ``then``), and ``queued``: how many are
    waiting (left to start after).
    Also run in the worker's container by restart-olfdb when the web server does not answer."""
    jobs = []
    all_fits = fits.listing(db)["fits"]
    for f in all_fits:
        if f["status"] == "running":
            jobs.append({"what": f"Fit #{f['id']}, asked for by {f['name']}", "started_at": f["started_at"],
                         "ends_at": f["eta"]["ends_at"],
                         "then": "It would be stopped and marked interrupted, to be asked for again (from the start)."})
    preds = all_predictions(db)
    estimate(preds, bool(fits.worker_status(db).get("gpu")), any(f["status"] == "running" for f in all_fits))
    for p in preds:
        if p["status"] == "running":
            jobs.append({"what": f"Fit #{p['fit_id']}'s model predicting {p['fly']}'s ROIs, asked for by {p['name']}",
                         # (not a guess, nor a time already past)
                         "started_at": p["started_at"], "ends_at": p["eta"]["ends_at"] if p["eta"]["from"] and p["eta"]["seconds"] else None,
                         "then": "It would be stopped and marked interrupted, to be asked for again."})
    from . import extract  # (it imports this module)

    exts = extract.all_extractions(db)
    extract.estimate(exts)
    for x in exts:
        if x["status"] == "running":
            jobs.append({"what": f"The signals of {x['fly']}'s ROIs being extracted, asked for by {x['name']}",
                         "started_at": x["started_at"], "ends_at": x["eta"]["ends_at"] if x["eta"]["seconds"] else None,
                         "then": "It would be stopped and marked interrupted, to be asked for again."})
    queued = (sum(f["status"] == "queued" for f in all_fits) + sum(p["status"] == "queued" for p in preds)
              + sum(x["status"] == "queued" for x in exts))
    return {"jobs": jobs, "queued": queued}


# --------------------------------------------------------------------- flies

def fly_name(tensor_path: str, path: Path) -> str:
    """A fly's name, as al-diag-seg names a fit's flies (`fits.fly_names`): the top folder of its
    data on tensor, its date and number (from the folders in its path), else its folder's parent's
    name and its own."""
    fly, _ = movies._path_fly(Path(tensor_path or path.as_posix()) / "x")
    top = (tensor_path or "").split("/")[0]
    if fly and top:
        return f"{top}_{fly['date']}_{fly['fly_num']}"
    return re.sub(r"[^\w.-]+", "_", f"{path.parent.name}_{path.name}")


def locate(dir_: str, raw_dir: str, root: Path, bases: list[Path]) -> dict:
    """A fly to predict, from its motion-corrected and raw data folders as entered (from the top
    of tensor, or absolute): ``dir``, ``raw_dir`` (on the server), ``tensor_dir``,
    ``tensor_raw_dir`` (from the top of tensor) and ``name`` (`fly_name`).

    Raises:
        ValueError: for a folder not found, or one the server may not read, or a fly's folder
            with no recording with its odors' mean dF/F (al_analysis' trialmean_dff.tif).
    """
    out = {}
    for key, what, v in (("dir", "its motion-corrected data folder", dir_), ("raw_dir", "its raw data folder", raw_dir)):
        if not (v or "").strip():
            raise ValueError(f"{what}: none given")
        found = tensor.find_data(v, root, bases)
        if found["status"] != "ok":
            raise ValueError(f"{what}: {found['error']}")
        if not found["is_dir"]:
            raise ValueError(f"{what}: a file, not a folder")
        out[key] = os.path.realpath(found["path"])
        out[f"tensor_{key}"] = found["tensor_path"]
    d = Path(out["dir"])
    if not any(roi_training._is_file(r / roi_training.TRIALMEAN) and roi_training._is_file(r / roi_training.TRIALS)
               for r in tensor._subdirs(d)):
        raise ValueError(f"its motion-corrected data folder: no recording in it with its odors' mean dF/F "
                         f"({roi_training.TRIALMEAN} and {roi_training.TRIALS}, as al_analysis writes them)")
    out["name"] = fly_name(out["tensor_dir"], d)
    return out


# ------------------------------------------------------------------- listing

def models(all_fits: list[dict]) -> list[dict]:
    """The models predictions may be made with: fits done (of `all_fits`, newest first) whose
    model is there, each with its flies (``train``, ``held_out``: their names) and when it finished."""
    out = []
    for f in all_fits:
        if f["status"] != "done" or not (fits.fit_dir(f["id"]) / "model" / "model.json").is_file():
            continue
        fl = f["flies"] or {}
        out.append({"id": f["id"], "name": f["name"], "created_at": f["created_at"], "finished_at": f["finished_at"],
                    "options": f["options"], "train": fl.get("train", []), "held_out": fl.get("held_out", []),
                    "mean": (f["result"] or {}).get("mean")})
    return out


def listing(db, root_: Path) -> dict:
    """Every prediction (newest first: each with ``tensor_dir`` and ``tensor_raw_dir``, its fly's
    folders from the top of tensor, ``fit_deleted``, and those queued or running with ``eta``:
    when it should end, see `estimate`), the models to predict with (`models`), the fit worker,
    where tensor is on the server, and ``expected``: the same of a prediction asked for now."""
    all_fits = fits.all_fits(db)
    have = {f["id"] for f in all_fits}
    preds = []
    for p in all_predictions(db):
        rel = lambda s: roi_training._arcname(Path(s), root_, []) if s else None  # noqa: E731
        preds.append({**p, "tensor_dir": rel(p["fly_dir"]), "tensor_raw_dir": rel(p["raw_dir"]),
                      "fit_deleted": p["fit_id"] not in have, "edited": edited_info(p) if p["status"] == "done" else None})
    worker = fits.worker_status(db)
    new = estimate(preds, bool(worker.get("gpu")), any(f["status"] == "running" for f in all_fits))
    return {"predictions": preds, "models": models(all_fits), "worker": worker, "tensor_root": str(root_), "expected": new}


def view(db, pred_id: int) -> Optional[dict]:
    """What the tab's viewer shows of a prediction done: its fly's average image (as the model saw
    it) and motion-corrected movies (`movies._fly_movies`), each with ``predicted`` (the
    prediction's ROIs) and ``rois`` (the fly's own, next to the movie or above it; for the
    average image, those the prediction was scored against); and the predicted ROIs as edited
    there (``edited``: `edited_info`, None if they were not; ``revision``: `edited_revision`).
    None if there is no such prediction done."""
    p = get(db, pred_id)
    if p is None or p["status"] != "done":
        return None
    d, top = pred_dir(pred_id), Path(p["fly_dir"])
    try:
        prov = json.loads((d / PROVENANCE).read_text())
    except (OSError, ValueError):
        prov = {}
    manual = (prov.get("fly") or {}).get("roiset")
    pred = str(roiset(p))
    out = []
    if (d / AVERAGE).is_file():
        out.append({"path": str(d / AVERAGE), "label": "Average image (as the model saw it)", "format": "tiff",
                    "size": (d / AVERAGE).stat().st_size, "recording": None, "recordings": None,
                    "rois": manual if manual and sandbox.can_read(manual) and os.path.isfile(manual) else None,
                    "trials": None})
    fly, _ = movies._path_fly(top / "x")
    try:
        found = movies._fly_movies(top.parent, top, fly or {"date": None, "fly_num": None}) or []
    except OSError:
        found = []
    for m in found:
        m["label"] = f"{top.name}/{Path(m['path']).relative_to(top).as_posix()}" if m["recording"] else f"{top.name}/{Path(m['path']).name}"
        out.append(m)
    for m in out:
        m["predicted"] = pred
    return {"id": p["id"], "fit_id": p["fit_id"], "fly": p["fly"], "movies": out, "edited": edited_info(p),
            "revision": edited_revision(p)}


# ---------------------------------------------------------- ROIs as edited

def edited_roiset(p: dict) -> Path:
    """A prediction's ROIs as edited in the tab's viewer (`save_edited`), if they were."""
    return pred_dir(p["id"]) / f"{p['fly']}_{EDITED}_RoiSet.zip"


def _edited_json(p: dict) -> Path:
    return pred_dir(p["id"]) / f"{p['fly']}_{EDITED}.json"


def _edited_meta(p: dict) -> dict:
    """What a prediction's ``<fly>_pred-edited.json`` says ({} if its ROIs were never edited)."""
    try:
        return json.loads(_edited_json(p).read_text())
    except (OSError, ValueError):
        return {}


def edited_revision(p: dict) -> int:
    """How many times a prediction's edited ROIs were saved or deleted (0: never): what the next
    change must say it was made from (`save_edited`)."""
    return int(_edited_meta(p).get("revision", 0))


def edited_info(p: dict) -> Optional[dict]:
    """What is known of a prediction's edited ROIs (`save_edited` says what), with their ``path``;
    None if it has none (never edited, or the edits deleted)."""
    path, meta = edited_roiset(p), _edited_meta(p)
    return {**meta, "path": str(path)} if meta and not meta.get("deleted") and path.is_file() else None


def _edited_rois(rois) -> list[dict]:
    """The ROIs the viewer sends to save, checked: each its ``name``, ``z`` (its plane, from 0;
    None: all) and ``points`` ([x, y] in pixels, as ImageJ's: three or more).

    Raises:
        ValueError: saying what is wrong with them.
    """
    if not isinstance(rois, list) or not rois:
        raise ValueError("rois: a list of ROIs")
    if len(rois) > MAX_EDITED_ROIS:
        raise ValueError(f"more than {MAX_EDITED_ROIS} ROIs")
    out, total = [], 0
    for i, r in enumerate(rois):
        what = f"ROI {i + 1}"
        if not isinstance(r, dict):
            raise ValueError(f"{what}: not an object")
        name, z, pts = r.get("name"), r.get("z"), r.get("points")
        if not isinstance(name, str) or not name.strip() or len(name) > 100 or not name.isprintable():
            raise ValueError(f"{what}: its name must be 1 to 100 printable characters")
        if z is not None and (isinstance(z, bool) or not isinstance(z, int) or not 0 <= z < 10_000):
            raise ValueError(f"{what} ({name}): its plane must be a whole number from 0")
        ok = isinstance(pts, list) and len(pts) >= 3 and all(
            isinstance(q, list) and len(q) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                         and math.isfinite(v) and abs(v) <= 100_000 for v in q) for q in pts)
        if not ok:
            raise ValueError(f"{what} ({name}): its points must be three or more [x, y], in pixels")
        total += len(pts)
        out.append({"name": name.strip(), "z": z, "points": [[round(float(x), 2), round(float(y), 2)] for x, y in pts]})
    if total > MAX_EDITED_POINTS:
        raise ValueError(f"more than {MAX_EDITED_POINTS} points in all")
    return out


def _write_roiset(rois: list[dict], path: Path) -> None:
    """Writes ROIs (as `_edited_rois` gives them) as an ImageJ RoiSet.zip of polygons, each on its
    plane as a hyperstack Z (as al-diag-seg writes a prediction's, and al_analysis' are), its
    file in the zip named as ImageJ names them (the ROI's name, made unique with -1, -2, ...)."""
    import numpy as np
    import roifile

    tmp = path.with_name(f".{path.stem}.tmp.zip")  # (roifile writes a zip to a name that ends so)
    tmp.unlink(missing_ok=True)
    seen: dict[str, int] = {}
    for r in rois:
        roi = roifile.ImagejRoi.frompoints(np.asarray(r["points"], np.float32), name=r["name"], z=r["z"])
        roi.roitype = roifile.ROI_TYPE.POLYGON
        stem = re.sub(r"[^\w.+?-]+", "_", r["name"])
        seen[stem] = seen.get(stem, -1) + 1
        roi.tofile(tmp, name=f"{stem}-{seen[stem]}.roi" if seen[stem] else f"{stem}.roi")
    os.replace(tmp, path)


def save_edited(db, pred_id: int, rois, base, by: Optional[str], ip: Optional[str], simplified=None,
                points_edited: bool = False) -> dict:
    """Saves a prediction's ROIs as edited in the viewer, beside those predicted (left as they
    are): ``<fly>_pred-edited_RoiSet.zip``, and ``<fly>_pred-edited.json`` saying of them
    ``revision`` (1 more at each save), ``saved_at``, ``by`` and ``ip`` (who saved them last),
    ``simplified`` (by how many pixels their outlines were let move when simplified; None if
    they were not), ``points_edited`` (whether points were moved, added or deleted since),
    ``rois`` and ``points`` (how many) and ``of`` (the predicted ROIs they are a copy of).

    Args:
        rois: as `_edited_rois` takes them.
        base: the revision they were edited from (0: none yet), so that two people's edits do
            not overwrite each other unseen.

    Returns:
        what `edited_info` then gives.

    Raises:
        LookupError: no such prediction done.
        Stale: they were saved (or deleted) by someone else since `base`.
        ValueError: for ROIs or a simplification not as they must be.
    """
    rois = _edited_rois(rois)
    if simplified is not None and (isinstance(simplified, bool) or not isinstance(simplified, (int, float))
                                   or not 0 <= simplified <= 100):
        raise ValueError("simplified: by how many pixels (0 to 100), or null")
    with _edit_lock:
        p = get(db, pred_id)
        if p is None or p["status"] != "done":
            raise LookupError("no such prediction done")
        revision = edited_revision(p)
        if revision != base:
            raise Stale(revision, edited_info(p))
        _write_roiset(rois, edited_roiset(p))
        _write_meta(p, {"revision": revision + 1, "saved_at": now(), "by": by, "ip": ip,
                        "simplified": simplified, "points_edited": bool(points_edited), "rois": len(rois),
                        "points": sum(len(r["points"]) for r in rois), "of": roiset(p).name})
    return edited_info(p)


def _write_meta(p: dict, meta: dict) -> None:
    tmp = _edited_json(p).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, _edited_json(p))


def delete_edited(db, pred_id: int, base, by: Optional[str], ip: Optional[str]) -> int:
    """Deletes a prediction's edited ROIs (those predicted are then the ones shown, and used); its
    ``<fly>_pred-edited.json`` stays, saying so (``deleted``), by whom and when. Returns the
    revision this makes. Raises as `save_edited`."""
    with _edit_lock:
        p = get(db, pred_id)
        if p is None or p["status"] != "done":
            raise LookupError("no such prediction done")
        revision = edited_revision(p)
        if revision != base:
            raise Stale(revision, edited_info(p))
        if revision:  # (never edited: nothing to delete, nor to say)
            edited_roiset(p).unlink(missing_ok=True)
            _write_meta(p, {"revision": revision + 1, "saved_at": now(), "by": by, "ip": ip, "deleted": True})
            revision += 1
    return revision


def stream_zip(pred_id: int, top: str):
    """A prediction's folder as a zip, under ``top``, as it is written."""
    return zip_folder(pred_dir(pred_id), top)


def zip_folder(d: Path, top: str):
    """A folder's files as a zip, under ``top``, as it is written (but for hidden ones: those
    being written)."""
    import zipfile

    sink = roi_training._Sink()
    when = dt.datetime.now().timetuple()[:6]
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            info = zipfile.ZipInfo(f"{top}/{p.relative_to(d).as_posix()}", when)
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w", force_zip64=True) as w, open(p, "rb") as fin:
                while chunk := fin.read(roi_training.CHUNK):
                    w.write(chunk)
                    yield sink.take()
            yield sink.take()
    yield sink.take()
