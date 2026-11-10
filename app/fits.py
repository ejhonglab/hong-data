"""Fits of al-diag-seg models (https://github.com/tom-f-oconnell/al-diag-seg) to the flies the
AL segmentation tab lists, asked for there, and run one at a time by the fit worker
(`fit_worker`): a process of its own (in Docker, a container of its own, with the GPU).

A fit is a row of the ``fits`` table and a folder, `fit_dir`: ``manifest.json`` (its flies, as
the tab's download describes them, but with each file where it is on the server, so they are
read in place), ``log.txt``, ``model/`` (al-diag-seg's model: what is downloaded, to run on
other flies), ``results/`` (its held-out flies' predicted ROIs, as label TIFFs and named
RoiSets, and their scores) and ``work/`` (Cellpose's own files).

How long one has left is estimated from how long each stage (`STAGES`) of earlier fits
took for its size (`sizes`: flies loaded, epochs times training flies, flies registered
into the atlas, held-out flies), and for the stage it is in, from how far it has got.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Optional

from . import config, roi_training
from .glomeruli import load_glomeruli

STATUSES = ("queued", "running", "done", "failed", "cancelled", "interrupted")
ACTIVE = ("queued", "running")
STAGES = ("load", "train", "atlas", "evaluate")
# how a fit's model makes its ROIs and joins them into 3D objects (al-diag-seg's Options of these
# names, saved with the model): al-diag-seg's defaults (tests/test_fits.py checks), picked on fit 6's
# flies (design.md)
ROI_DEFAULTS = {"cellprob_threshold": -2.0, "flow_threshold": 0.8, "dilate": 0, "join_overlap": 0.2, "join_corr": 0.7}
# 15 epochs: the held-out flies' loss had stopped improving by epoch 10 in fits 2 and 6 (design.md)
DEFAULTS = {"held_out": 4, "epochs": 15, "atlas": True, "tentative": True, "refs": ["grabe2016"], **ROI_DEFAULTS}
LIMITS = {"held_out": (1, 30), "epochs": (1, 5000), "dilate": (0, 10)}  # (whole numbers)
FLOAT_LIMITS = {"cellprob_threshold": (-6.0, 6.0), "flow_threshold": (0.0, 3.0), "join_overlap": (0.0, 1.0),
                "join_corr": (-1.0, 1.0)}
# the fixed reference atlases a fit may add to its flies' atlas (``refs``, al-diag-seg's): folders of config.AL_ATLAS_DIR.
# (Not al_atlas' hemibrain export: some of its glomeruli are fragments of them.)
REFERENCES = {"grabe2016": "Grabe et al. 2016"}
# seconds per unit of each stage (see `sizes`), until fits have finished to estimate from
GUESSES = {"load": 1.5, "train": 0.05, "atlas": 60.0, "evaluate": 45.0}
RECENT = 10  # the finished fits estimates are made from (the last ones)
DEFAULT_FLIES = 80  # flies assumed for the estimate of a fit not yet started, if none has been
WORKER_SETTING = "fit_worker"  # the worker's heartbeat (settings table)
HEARTBEAT_S = 10
STALE_S = 60  # a worker not heard from for longer is taken to be gone


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def fits_dir() -> Path:
    return config.DATA_DIR / "fits"


def fit_dir(fit_id: int) -> Path:
    return fits_dir() / str(int(fit_id))


def options(body: dict) -> dict:
    """The options of a fit asked for, `DEFAULTS` for those not given.

    Raises:
        ValueError: for one out of `LIMITS`, or not a whole number; out of `FLOAT_LIMITS`, or not
            a number; for ``refs`` not a list of `REFERENCES`, or given without the atlas (which
            they are added to). (Not given, they are `DEFAULTS`' with the atlas, none without.)
    """
    out = dict(DEFAULTS)
    for k, (lo, hi) in LIMITS.items():
        if body.get(k) not in (None, ""):
            try:
                v = int(body[k])
            except (TypeError, ValueError):
                raise ValueError(f"{k.replace('_', ' ')}: not a whole number") from None
            if not lo <= v <= hi:
                raise ValueError(f"{k.replace('_', ' ')}: must be from {lo} to {hi}")
            out[k] = v
    for k, (lo, hi) in FLOAT_LIMITS.items():
        if body.get(k) not in (None, ""):
            try:
                v = float(body[k])
            except (TypeError, ValueError):
                raise ValueError(f"{k.replace('_', ' ')}: not a number") from None
            if not lo <= v <= hi:  # (also NaN)
                raise ValueError(f"{k.replace('_', ' ')}: must be from {lo:g} to {hi:g}")
            out[k] = v
    for k in ("atlas", "tentative"):
        if k in body:
            out[k] = bool(body[k])
    refs = body.get("refs")
    if refs is None:
        refs = DEFAULTS["refs"] if out["atlas"] else []
    if not isinstance(refs, list) or not all(isinstance(r, str) and r in REFERENCES for r in refs):
        raise ValueError(f"reference atlases: a list of {', '.join(REFERENCES)}")
    out["refs"] = list(dict.fromkeys(refs))
    if out["refs"] and not out["atlas"]:
        raise ValueError("a reference atlas is added to the atlas of the flies: it needs the atlas")
    return out


def fit_options(fit: dict) -> dict:
    """A fit's options, `DEFAULTS` for those that were not options yet when it was asked for (but
    no reference atlases: fits from before ``refs`` had none)."""
    return {**DEFAULTS, "refs": [], **(fit.get("options") or {})}


def reference_dir(name: str) -> Path:
    """The folder of the reference atlas `name` (of `REFERENCES`)."""
    return config.AL_ATLAS_DIR / name


# ------------------------------------------------------------------- records

def _row(r) -> dict:
    d = dict(r)
    for k in ("options", "flies", "progress", "stages", "result"):
        v = d.pop(f"{k}_json")
        d[k] = json.loads(v) if v else None
    d["cancel"] = bool(d["cancel"])
    return d


def create(db, name: str, ip: Optional[str], opts: dict) -> dict:
    with db.connect() as con:
        cur = con.execute("INSERT INTO fits (name, ip, created_at, status, options_json) VALUES (?,?,?,?,?)",
                          (name, ip, now(), "queued", json.dumps(opts)))
        fit_id = cur.lastrowid
    return get(db, fit_id)


def get(db, fit_id: int) -> Optional[dict]:
    with db.connect() as con:
        r = con.execute("SELECT * FROM fits WHERE id = ?", (fit_id,)).fetchone()
    return _row(r) if r else None


def all_fits(db) -> list[dict]:
    with db.connect() as con:
        return [_row(r) for r in con.execute("SELECT * FROM fits ORDER BY id DESC")]


def update(db, fit_id: int, **fields) -> None:
    """Sets columns of a fit (``*_json`` ones given as their names without it, as values)."""
    cols, vals = [], []
    for k, v in fields.items():
        if k in ("options", "flies", "progress", "stages", "result"):
            k, v = f"{k}_json", None if v is None else json.dumps(v)
        cols.append(f"{k} = ?")
        vals.append(v)
    with db.connect() as con:
        con.execute(f"UPDATE fits SET {', '.join(cols)} WHERE id = ?", (*vals, fit_id))


def request_cancel(db, fit_id: int) -> Optional[dict]:
    """Asks a fit to stop: one queued is cancelled now, one running at its next step."""
    with db.connect() as con:
        con.execute("UPDATE fits SET cancel = 1 WHERE id = ? AND status IN ('queued', 'running')", (fit_id,))
        con.execute("UPDATE fits SET status = 'cancelled', finished_at = ? WHERE id = ? AND status = 'queued'", (now(), fit_id))
    return get(db, fit_id)


def delete(db, fit_id: int) -> Optional[bool]:
    """Deletes a fit and its folder. None if there is no such fit; False if it is running
    (stop it first)."""
    f = get(db, fit_id)
    if f is None:
        return None
    if f["status"] == "running":
        return False
    with db.connect() as con:
        con.execute("DELETE FROM fits WHERE id = ? AND status != 'running'", (fit_id,))
    shutil.rmtree(fit_dir(fit_id), ignore_errors=True)
    return True


def claim(db) -> Optional[dict]:
    """The oldest queued fit, now running (for the worker), or None."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT id FROM fits WHERE status = 'queued' AND cancel = 0 ORDER BY id LIMIT 1").fetchone()
        if r is None:
            return None
        con.execute("UPDATE fits SET status = 'running', started_at = ? WHERE id = ?", (now(), r["id"]))
    return get(db, r["id"])


def interrupt_running(db) -> list[int]:
    """Marks fits left running (by a worker that stopped) interrupted. Returns their ids."""
    with db.connect() as con:
        ids = [r["id"] for r in con.execute("SELECT id FROM fits WHERE status = 'running'")]
        con.execute("UPDATE fits SET status = 'interrupted', finished_at = ?, error = ? WHERE status = 'running'",
                    (now(), "the fit worker stopped (restarted, or the server did) while it ran"))
    return ids


def heartbeat(db, info: dict) -> None:
    db.set_setting(WORKER_SETTING, json.dumps({**info, "at": time.time()}), now())


def worker_status(db) -> dict:
    """The fit worker as it last said (``host``, ``gpu``, ``alseg``, ...), with ``alive``:
    whether it has been heard from in the last `STALE_S` seconds."""
    s = db.get_setting(WORKER_SETTING)
    if not s:
        return {"alive": False, "seen": None}
    w = json.loads(s["value"])
    return {**w, "alive": time.time() - w.get("at", 0) < STALE_S, "seen": s["updated_at"]}


# ------------------------------------------------------------------- flies

def fly_names(flies: list[dict]) -> list[str]:
    """The flies' names as al-diag-seg gives them (``alseg.olfdb.fly_entries``): the top folder
    of their data on tensor, date and number, made unique with -2, -3, ..."""
    out, seen = [], {}
    for f in flies:
        n = f"{f['dir'].split('/')[0]}_{f['date']}_{f['fly_num']}"
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}-{seen[n]}")
    return out


def choose_held_out(flies: list[dict], n: int) -> list[str]:
    """``n`` flies (of the manifest's) to hold out: from each dataset in turn (by its number,
    a fly's first), and of each, spread evenly through its flies (by date and number), so
    that every dataset, and uploader, is scored where there are enough."""
    names = fly_names(flies)
    by_ds: dict[int, list[tuple]] = {}
    for f, name in zip(flies, names):
        by_ds.setdefault(f["datasets"][0]["id"] if f.get("datasets") else 0, []).append((f["date"], f["fly_num"], name))
    groups = [sorted(v) for _, v in sorted(by_ds.items())]
    counts = [0] * len(groups)
    k, left = 0, min(n, len(flies) - 1)
    while left > 0:
        if counts[k % len(groups)] < len(groups[k % len(groups)]):
            counts[k % len(groups)] += 1
            left -= 1
        k += 1
    out = []
    for g, c in zip(groups, counts):
        out += [g[int((i + 0.5) * len(g) / c)][2] for i in range(c)]
    return out


def manifest(found: dict, bases: list[Path]) -> dict:
    """The manifest al-diag-seg reads (as the tab's download's, `roi_training.manifest`), of
    the flies `roi_training.inventory` found, with every file's absolute path on the server,
    so they are read in place; and ``dir`` from the top of tensor (what flies are named by).
    Average images not on tensor are computed (and kept: `roi_training.computed_average`)."""
    root = config.TENSOR_ROOT
    flies = []
    for f in found["flies"]:
        a = f["average"]
        flies.append({
            "dir": roi_training._arcname(f["dir"], root, bases), "date": f["date"], "fly_num": f["fly_num"],
            "datasets": f["datasets"], "shape": list(f["grid"]), "raw_dir": str(f["raw_dir"]) if f["raw_dir"] else None,
            "average": str(roi_training.computed_average(a["source"]) if a["computed"] else a["path"]),
            "roiset": str(f["roiset"]),
            "recordings": [{"name": r["dir"].name, "responses": str(r["dir"] / roi_training.TRIALMEAN),
                            "trials": str(r["dir"] / roi_training.TRIALS), "experiment_xml": str(r["xml"]) if r["xml"] else None,
                            "odors": [o["odor"] for o in r["odors"]], "as_recorded": [o["recorded"] for o in r["odors"]]}
                           for r in f["recordings"]],
            "stacks": [{"name": s["dir"].name, "dir": str(s["dir"]), "experiment_xml": str(s["dir"] / roi_training.XML)}
                       for s in f["stacks"]],
        })
    return {"format": "olfdb ROI training export", "version": 1, "exported_at": now(), "stacks": True,
            "glomeruli": load_glomeruli(), "flies": flies, "skipped": []}


# ----------------------------------------------------------------- estimates

def sizes(fit: dict, n_flies: Optional[int] = None) -> dict[str, float]:
    """How much work each stage of a fit is: flies loaded, epochs x training flies, flies (and
    anatomical stacks, and reference atlases) registered into the atlas, held-out flies. From
    its flies once the worker has listed them, else ``n_flies`` (a guess)."""
    o = fit_options(fit)
    fl = fit.get("flies") or {}
    n = len(fl.get("train", [])) + len(fl.get("held_out", [])) if fl else (n_flies or DEFAULT_FLIES)
    val = len(fl.get("held_out", [])) if fl else min(o["held_out"], n - 1)
    train = n - val
    return {"load": n, "train": o["epochs"] * train,
            "atlas": (train + fl.get("stacks", 0) + len(o["refs"])) if o["atlas"] else 0,
            "evaluate": val}


def rates(fits: list[dict]) -> dict:
    """Seconds per unit of work of each stage (`sizes`): the median of the last `RECENT` fits
    that finished, else `GUESSES`; with ``from``: how many fits they are from."""
    done = [f for f in fits if f["status"] == "done" and f.get("stages")][:RECENT]
    out = {}
    for s in STAGES:
        rs = [f["stages"][s]["seconds"] / f["stages"][s]["size"] for f in done
              if f["stages"].get(s) and f["stages"][s].get("size")]
        out[s] = statistics.median(rs) if rs else GUESSES[s]
    return {"seconds": out, "from": len(done)}


def remaining(fit: dict, r: dict, t: float, n_flies: Optional[int] = None) -> float:
    """Seconds a fit has left: of the stage it is at, from how far it has got (once it has got
    anywhere), else from `rates`; and of the stages after it, from `rates`."""
    sz, sec = sizes(fit, n_flies), r["seconds"]
    p = fit.get("progress") or {}
    stage = p.get("stage")
    left = 0.0
    later = STAGES if stage not in STAGES else STAGES[STAGES.index(stage) + 1:]
    for s in later:
        left += sec[s] * sz[s]
    if stage in STAGES:
        elapsed = t - p.get("stage_started", t)
        done, total = p.get("done") or 0, p.get("total") or 0
        if done and total:
            left += max(0.0, elapsed / done * (total - done))
        else:
            left += max(0.0, sec[stage] * sz[stage] - elapsed)
    return left


def listing(db, t: Optional[float] = None) -> dict:
    """Every fit (newest first), those queued or running with ``eta``: ``seconds`` it has left
    to run, ``until`` it ends and ``ends_at`` (one queued starts when those before it end),
    and ``from``: how many finished fits that is estimated from; and the worker's status."""
    fits = all_fits(db)
    t = time.time() if t is None else t
    r = rates(fits)
    n_flies = next((len(f["flies"]["train"]) + len(f["flies"]["held_out"]) for f in fits if f.get("flies")), None)
    at = t
    for f in sorted((f for f in fits if f["status"] in ACTIVE), key=lambda f: (f["status"] != "running", f["id"])):
        left = remaining(f, r, t, n_flies)
        at += left
        f["eta"] = {"seconds": left, "until": at - t, "from": r["from"],
                    "ends_at": dt.datetime.fromtimestamp(at).astimezone().isoformat(timespec="seconds")}
    return {"fits": fits, "worker": worker_status(db), "rates": r}


def stream_zip(fit_id: int, top: str):
    """A fit's model and results as a zip, under ``top``, as it is written (stored: Cellpose's
    weights hardly compress, and are over a GB): ``README.txt`` (how to run the model, from
    the model's), ``model/`` and ``results/``."""
    import zipfile

    d = fit_dir(fit_id)
    sink = roi_training._Sink()
    when = dt.datetime.now().timetuple()[:6]
    files = [(d / "model" / "README.txt", "README.txt")] if (d / "model" / "README.txt").is_file() else []
    for sub in ("model", "results"):
        files += [(p, p.relative_to(d).as_posix()) for p in sorted((d / sub).rglob("*")) if p.is_file()]
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for src, name in files:
            with zf.open(zipfile.ZipInfo(f"{top}/{name}", when), "w", force_zip64=True) as w, open(src, "rb") as fin:
                while chunk := fin.read(roi_training.CHUNK):
                    w.write(chunk)
                    yield sink.take()
            yield sink.take()
    yield sink.take()
