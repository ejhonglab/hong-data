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
  ``README.txt`` and ``log.txt``.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import statistics
import time
from pathlib import Path
from typing import Optional

from . import config, fits, movies, roi_training, sandbox, tensor

STATUSES = ("queued", "running", "done", "failed", "interrupted")
ACTIVE = ("queued", "running")
PROVENANCE = "provenance.json"
AVERAGE = "average.tif"
RECENT = 10  # the predictions done that one running is estimated from (the last ones, on its device)


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


def ends_at(db, p: dict, t: Optional[float] = None) -> Optional[str]:
    """When prediction `p`, running, should end: its start, plus the median time the last
    `RECENT` done on the same device (GPU or CPU) took. None without one, or once that is past."""
    device = (p.get("progress") or {}).get("device")
    took = [dt.datetime.fromisoformat(q["finished_at"]).timestamp() - dt.datetime.fromisoformat(q["started_at"]).timestamp()
            for q in all_predictions(db)
            if q["status"] == "done" and q["started_at"] and q["finished_at"] and (q["progress"] or {}).get("device") == device]
    if not took or not p.get("started_at"):
        return None
    end = dt.datetime.fromisoformat(p["started_at"]).timestamp() + statistics.median(took[:RECENT])
    if end < (time.time() if t is None else t):
        return None
    return dt.datetime.fromtimestamp(end).astimezone().isoformat(timespec="seconds")


def worker_jobs(db) -> dict:
    """What restarting the fit worker would stop: the fits and predictions running, as the
    web server's `_running_jobs` gives its own (``jobs``, each its ``what``, ``started_at``,
    ``ends_at`` and ``then``), and ``queued``: how many are waiting (left to start after).
    Also run in the worker's container by restart-olfdb when the web server does not answer."""
    jobs = []
    all_fits = fits.listing(db)["fits"]
    for f in all_fits:
        if f["status"] == "running":
            jobs.append({"what": f"Fit #{f['id']}, asked for by {f['name']}", "started_at": f["started_at"],
                         "ends_at": f["eta"]["ends_at"],
                         "then": "It would be stopped and marked interrupted, to be asked for again (from the start)."})
    preds = all_predictions(db)
    for p in preds:
        if p["status"] == "running":
            jobs.append({"what": f"Fit #{p['fit_id']}'s model predicting {p['fly']}'s ROIs, asked for by {p['name']}",
                         "started_at": p["started_at"], "ends_at": ends_at(db, p),
                         "then": "It would be stopped and marked interrupted, to be asked for again."})
    queued = sum(f["status"] == "queued" for f in all_fits) + sum(p["status"] == "queued" for p in preds)
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
    folders from the top of tensor, and ``fit_deleted``), the models to predict with
    (`models`), the fit worker, and where tensor is on the server."""
    all_fits = fits.all_fits(db)
    have = {f["id"] for f in all_fits}
    preds = []
    for p in all_predictions(db):
        rel = lambda s: roi_training._arcname(Path(s), root_, []) if s else None  # noqa: E731
        preds.append({**p, "tensor_dir": rel(p["fly_dir"]), "tensor_raw_dir": rel(p["raw_dir"]),
                      "fit_deleted": p["fit_id"] not in have})
    return {"predictions": preds, "models": models(all_fits), "worker": fits.worker_status(db), "tensor_root": str(root_)}


def view(db, pred_id: int) -> Optional[dict]:
    """What the tab's viewer shows of a prediction done: its fly's average image (as the model saw
    it) and motion-corrected movies (`movies._fly_movies`), each with ``predicted`` (the
    prediction's ROIs) and ``rois`` (the fly's own, next to the movie or above it; for the
    average image, those the prediction was scored against). None if there is no such prediction
    done."""
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
    return {"id": p["id"], "fit_id": p["fit_id"], "fly": p["fly"], "movies": out}


def stream_zip(pred_id: int, top: str):
    """A prediction's folder as a zip, under ``top``, as it is written."""
    import zipfile

    d = pred_dir(pred_id)
    sink = roi_training._Sink()
    when = dt.datetime.now().timetuple()[:6]
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            info = zipfile.ZipInfo(f"{top}/{p.relative_to(d).as_posix()}", when)
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w", force_zip64=True) as w, open(p, "rb") as fin:
                while chunk := fin.read(roi_training.CHUNK):
                    w.write(chunk)
                    yield sink.take()
            yield sink.take()
    yield sink.take()
