"""Baseline models run on ORN data of the site, asked for in the Models tab and run by the model
worker (model_worker.py: a container with olfsysm), one at a time.

A run is a row of the ``model_runs`` table: a baseline model (of those chosen in the Admin tab,
`model_data.baselines`), datasets of glomeruli (and which of their panels), and how it is
tuned (`TUNINGS`): ``fixed``, with the baseline's own threshold and APL weights (the same
model, on new input), or ``tuned``, those fitted again to the baseline's response rate on
the new odors. The datasets' responses are scaled into changes in spike rate first
(orn_input.py). How its baseline is run is worked out when it is asked for (`plan`), from the
baseline's params.csv row: pyolfsysm's `fit_and_plot_mb_model` is called with its settings
(``base_kwargs_json``, and its other columns that are settings of it), and a baseline with
anything else that is not off (another experiment's own mechanism: a gain, a breadth
exponent, ...) cannot be run. What `fit_and_plot_mb_model` takes is what the model worker
says it does (`WORKER_FILE`, written as it starts, and every `WORKER_BEAT` seconds).

Its folder (`run_dir`) has, once run:

- ``output/model/``: everything `fit_and_plot_mb_model` wrote (its plots, the spike counts
  and responses of every KC, the weights, the parameters and its own provenance);
- ``inputs/``: the glomeruli's mean responses, those scaled (the model's input), how they
  were scaled and from which responses in common with Hallem & Carlson's (and their table),
  the baseline's parameters, and the arguments of the call;
- ``run.py``: the script that ran it, to run again elsewhere; ``provenance.json`` (what was
  run, on what, by what software, for whom), ``README.txt`` and ``log.txt``;
- ``view/``: its spike counts as the Models tab reads a model (`model_data.JOB_RUN`).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Optional

from . import model_data, orn_input

STATUSES = ("queued", "running", "done", "failed", "cancelled", "interrupted")
ACTIVE = ("queued", "running")
TUNINGS = ("fixed", "tuned")
STAGES = ("inputs", "model", "outputs")  # scaling the responses; fit_and_plot_mb_model; reading and packing what it wrote
WORKER_FILE = "worker.json"
WORKER_BEAT = 20  # seconds between the worker's writes of it
WORKER_STALE = 90  # seconds after which a worker that has not written it is taken to be gone
RECENT = 10  # the runs done that one running is estimated from (the last ones tuned alike)
DEFAULT_SECONDS = {"fixed": 150.0, "tuned": 600.0}  # a run's model stage, before any is done
BASE_KWARGS = "base_kwargs_json"  # the params.csv column of the settings its models share, as JSON
# a baseline's own threshold and APL weights (pyolfsysm's `optimizable_model_params`): the
# params.csv columns of each, the first it has
FIXED = {"fixed_thr": ("fixed_thr", "fixed_thr_mean"), "wAPLKC_scale": ("wAPLKC_scale", "wAPLKC"), "wKCAPL_scale": ("wKCAPL_scale",),
         "wAPLPN_scale": ("wAPLPN_scale",), "wPNAPL_scale": ("wPNAPL_scale",)}
NEEDED = ("fixed_thr", "wAPLKC_scale")  # those without which it cannot be run as it is
TARGET = ("target_rate", "target_sparsity")  # its response rate tuned to: the columns of it, the first it has
# params.csv columns that say what a model is, or how it came out: not settings
INFO = {"source_table", "tag", "variant", "base", "part", "n_kcs", "n_glomeruli", "rate_on", "rate_off", BASE_KWARGS,
        "base_params", "init_params", "generated"}
NEUTRAL = {"gain": 1.0, "thr_factor": 1.0, "wAPLKC_scale_factor": 1.0}  # values of settings that are not ones of it, that leave it as it is (besides 0 and none)


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def root() -> Path:
    return model_data.runs_root()


def run_dir(run_id: int) -> Path:
    return model_data.job_dir(run_id)


# --------------------------------------------------------------------- the worker

def worker() -> Optional[dict]:
    """What the model worker last said of itself (`WORKER_FILE`: ``at``, ``host``, ``kwargs``:
    the arguments `fit_mb_model` takes, ``versions``), with ``alive`` (it said so less than
    `WORKER_STALE` seconds ago); None if none ever did."""
    try:
        w = json.loads((root() / WORKER_FILE).read_text())
    except (OSError, ValueError):
        return None
    return {**w, "alive": time.time() - float(w.get("at") or 0) < WORKER_STALE}


# ------------------------------------------------------------------------ plans

def _given(v: Any) -> bool:
    return v is not None and v != "" and not (isinstance(v, float) and math.isnan(v))


def _off(col: str, v: Any) -> bool:
    """Whether a column's value leaves the model as it is: 0, false, or its `NEUTRAL` one."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v == 0 or v == NEUTRAL.get(col)
    return False


def plan(params: dict, kwargs: Optional[list[str]]) -> dict:
    """How a baseline is run, from its params.csv row (`params`) and the arguments
    `fit_mb_model` takes (`kwargs`: the worker's; None if not known).

    Returns:
        ``runnable`` (None if `kwargs` is not known), ``why`` (what stops it, a list),
        ``model`` (the settings `fit_and_plot_mb_model` is called with), ``fixed`` (its own
        threshold and APL weights, as pyolfsysm's `optimizable_model_params` names them; None
        if its row lacks them: then not run ``fixed``), ``target`` (the response rate it is
        tuned to; None if its row has none: then not run ``tuned``) and ``tunings`` (those of
        `TUNINGS` it can be run with).
    """
    why: list[str] = []
    model: dict = {}
    raw = params.get(BASE_KWARGS)
    if not _given(raw):
        why.append(f"its params.csv has no {BASE_KWARGS} (the settings its model was made with)")
    else:
        try:
            model = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if not isinstance(model, dict):
                raise ValueError
        except (TypeError, ValueError):
            model = {}
            why.append(f"its {BASE_KWARGS} is not a JSON object of settings")
    fixed = {k: float(params[c]) for k, cols in FIXED.items() if (c := next((c for c in cols if _given(params.get(c))), None))
             and isinstance(params[c], (int, float)) and not isinstance(params[c], bool)}
    target = next((float(params[c]) for c in TARGET if isinstance(params.get(c), (int, float)) and _given(params[c])), None)
    used = {c for cols in FIXED.values() for c in cols} | set(TARGET) | INFO | set(model)
    other = []
    for col, v in params.items():
        if col in used or not _given(v) or _off(col, v):
            continue
        if kwargs is not None and col in kwargs:
            model[col] = v
        else:
            other.append(f"{col} = {v}")
    if kwargs is not None:
        unknown = [k for k in model if k not in kwargs]
        if unknown:
            why.append(f"{', '.join(unknown)} {'is not a setting' if len(unknown) == 1 else 'are not settings'} fit_and_plot_mb_model takes")
        if other:
            why.append(f"it has what fit_and_plot_mb_model does not take (of the experiment that made it): {', '.join(other)}")
    has_fixed = all(k in fixed for k in NEEDED)
    tunings = [t for t, ok in (("fixed", has_fixed), ("tuned", target is not None)) if ok]
    if not tunings:
        why.append("its params.csv has neither its threshold and APL weights (fixed_thr, wAPLKC_scale) nor the response rate it was tuned to (target_rate)")
    return {"runnable": None if kwargs is None else not why, "why": why, "model": model, "fixed": fixed if has_fixed else None,
            "target": target, "tunings": tunings}


def baselines(db, tensor_root: Path) -> list[dict]:
    """The baselines chosen in the Admin tab, each with how it would be run (`plan`): ``id``,
    ``name``, ``experiment``, ``ok`` and ``error`` (as `model_data.baseline_listing`'s),
    ``params`` (its row) and ``plan``."""
    w = worker()
    listing = model_data.baseline_listing(db, tensor_root)
    out = []
    for b in listing["baselines"]:
        entry = {k: b[k] for k in ("id", "name", "experiment", "row", "ok", "error")}
        if b["ok"]:
            m = listing["models"][b["model"]]
            entry["params"] = m["params"]
            entry["plan"] = plan(m["params"], w["kwargs"] if w else None)
        out.append(entry)
    return out


# ---------------------------------------------------------------------- records

def _row(r) -> dict:
    d = dict(r)
    for k in ("options", "progress", "result"):
        v = d.pop(f"{k}_json")
        d[k] = json.loads(v) if v else None
    d["cancel"] = bool(d["cancel"])
    return d


def get(db, run_id: int) -> Optional[dict]:
    with db.connect() as con:
        r = con.execute("SELECT * FROM model_runs WHERE id = ?", (run_id,)).fetchone()
    return _row(r) if r else None


def all_runs(db) -> list[dict]:
    with db.connect() as con:
        return [_row(r) for r in con.execute("SELECT * FROM model_runs ORDER BY created_at DESC, id DESC")]


def update(db, run_id: int, **fields) -> None:
    """Sets columns of a run (``*_json`` ones given as their names without it, as values)."""
    cols, vals = [], []
    for k, v in fields.items():
        if k in ("options", "progress", "result"):
            k, v = f"{k}_json", None if v is None else json.dumps(v)
        cols.append(f"{k} = ?")
        vals.append(v)
    with db.connect() as con:
        con.execute(f"UPDATE model_runs SET {', '.join(cols)} WHERE id = ?", (*vals, run_id))


class RunError(ValueError):
    """Why a run cannot be asked for, for people."""


def options(db, body: dict, tensor_root: Path) -> dict:
    """A run's options, from what the Models tab asks for (``baseline``: its id; ``datasets``:
    ids; ``panels``: names, or None for the datasets' default ones; ``tuning``), checked, with
    how its baseline is run kept (`plan`'s, and its row), so that it is run as asked even if
    the baseline is changed since.

    Raises:
        RunError: for a baseline that cannot be run (so), or datasets that are not of glomeruli.
    """
    b = next((b for b in baselines(db, tensor_root) if b["id"] == body.get("baseline")), None)
    if b is None:
        raise RunError("choose a baseline model")
    if not b["ok"]:
        raise RunError(f"its baseline cannot be read: {b['error']}")
    p = b["plan"]
    if p["runnable"] is None:
        raise RunError("no model worker has started yet, so what fit_and_plot_mb_model takes is not known")
    if not p["runnable"]:
        raise RunError(f"{b['name']} cannot be run: {'; '.join(p['why'])}")
    tuning = body.get("tuning") or p["tunings"][0]
    if tuning not in p["tunings"]:
        raise RunError(f"{b['name']} cannot be run {tuning}: its params.csv lacks what that needs")
    ids = body.get("datasets")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise RunError("choose a dataset of glomeruli")
    ids = list(dict.fromkeys(ids))
    for i in ids:
        d = db.get_dataset(i)
        if d is None or d["roi_kind"] != "glomeruli":
            raise RunError(f"#{i} is not a dataset of glomeruli")
    have = [p_["name"] for p_ in orn_input.panels(db, ids)]
    panels = body.get("panels")
    if panels is None:
        panels = [p_["name"] for p_ in orn_input.panels(db, ids) if p_["default"]]
    if not isinstance(panels, list) or not panels or any(x not in have for x in panels):
        raise RunError("choose at least one of its panels")
    return {"baseline": {k: b[k] for k in ("id", "name", "experiment", "row", "params")}, "datasets": ids,
            "panels": [x for x in have if x in panels], "tuning": tuning, "model": p["model"],
            "fixed": p["fixed"] if tuning == "fixed" else None, "target": p["target"] if tuning == "tuned" else None}


def create(db, name: str, ip: Optional[str], opts: dict) -> dict:
    """Asks for a run (`options`'s), queued."""
    with db.connect() as con:
        run_id = con.execute("INSERT INTO model_runs (name, ip, created_at, status, options_json) VALUES (?,?,?,?,?)",
                             (name, ip, now(), "queued", json.dumps(opts))).lastrowid
    return get(db, run_id)


def request_cancel(db, run_id: int) -> Optional[dict]:
    """Stops a run: one queued now, one running as soon as the worker sees it. None if there is none."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT status FROM model_runs WHERE id = ?", (run_id,)).fetchone()
        if r is None:
            return None
        if r["status"] == "queued":
            con.execute("UPDATE model_runs SET status = 'cancelled', finished_at = ? WHERE id = ?", (now(), run_id))
        elif r["status"] == "running":
            con.execute("UPDATE model_runs SET cancel = 1 WHERE id = ?", (run_id,))
    return get(db, run_id)


def delete(db, run_id: int) -> Optional[bool]:
    """Deletes a run and its folder. None if there is no such run; False if it is running."""
    r = get(db, run_id)
    if r is None:
        return None
    if r["status"] == "running":
        return False
    with db.connect() as con:
        con.execute("DELETE FROM model_runs WHERE id = ? AND status != 'running'", (run_id,))
    shutil.rmtree(run_dir(run_id), ignore_errors=True)
    return True


def claim(db) -> Optional[dict]:
    """The oldest queued run, now running (for the worker); None if there is none, or one is
    running already: one at a time, whoever asks."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        if con.execute("SELECT 1 FROM model_runs WHERE status = 'running'").fetchone():
            return None
        r = con.execute("SELECT id FROM model_runs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1").fetchone()
        if r is None:
            return None
        con.execute("UPDATE model_runs SET status = 'running', started_at = ? WHERE id = ?", (now(), r["id"]))
    return get(db, r["id"])


def interrupt_running(db) -> list[int]:
    """Marks runs left running (by a worker that stopped) interrupted. Returns their ids."""
    with db.connect() as con:
        ids = [r["id"] for r in con.execute("SELECT id FROM model_runs WHERE status = 'running'")]
        con.execute("UPDATE model_runs SET status = 'interrupted', finished_at = ?, error = ? WHERE status = 'running'",
                    (now(), "the model worker stopped (restarted, or the server did) while it ran"))
    return ids


# -------------------------------------------------------------------- estimates

def _seconds(a: str, b: str) -> float:
    return dt.datetime.fromisoformat(b).timestamp() - dt.datetime.fromisoformat(a).timestamp()


def expected_seconds(runs: list[dict], tuning: str) -> float:
    """How long a run tuned so should take: the median of the last `RECENT` done so, else `DEFAULT_SECONDS`'."""
    took = [_seconds(r["started_at"], r["finished_at"]) for r in runs
            if r["status"] == "done" and r["started_at"] and r["finished_at"] and r["options"]["tuning"] == tuning]
    return statistics.median(took[:RECENT]) if took else DEFAULT_SECONDS[tuning]


def progress(runs: list[dict], r: dict, t: Optional[float] = None) -> Optional[dict]:
    """A running run's progress: ``stage`` (of `STAGES`) and ``detail`` (what it last said), as
    the worker has them, ``fraction`` (of the time runs tuned alike took, at most 0.98) and
    ``ends_at`` (when it should end; None once that is past). None of one not running."""
    if r["status"] != "running" or not r["started_at"]:
        return None
    t = time.time() if t is None else t
    start = dt.datetime.fromisoformat(r["started_at"]).timestamp()
    expected = expected_seconds(runs, r["options"]["tuning"])
    end = start + expected
    p = r["progress"] or {}
    return {"stage": p.get("stage"), "detail": p.get("detail"), "fraction": max(0.0, min(0.98, (t - start) / expected)),
            "ends_at": dt.datetime.fromtimestamp(end).astimezone().isoformat(timespec="seconds") if end > t else None}


def worker_jobs(db) -> dict:
    """What restarting the model worker would stop (as `predictions.worker_jobs`'s): the run
    running, and how many are queued."""
    runs = all_runs(db)
    jobs = []
    for r in runs:
        if r["status"] == "running":
            p = progress(runs, r)
            jobs.append({"what": f"Model run #{r['id']} ({r['options']['baseline']['name']}), asked for by {r['name']}",
                         "started_at": r["started_at"], "ends_at": p and p["ends_at"],
                         "then": "It would be stopped and marked interrupted, to be asked for again."})
    return {"jobs": jobs, "queued": sum(r["status"] == "queued" for r in runs)}


# ----------------------------------------------------------------------- listing

def title(db, r: dict) -> str:
    """What a run is called: its baseline, on its datasets."""
    o = r["options"]
    return f"{o['baseline']['name']} on {', '.join(f'#{i}' for i in o['datasets'])} ({o['tuning']})"


def listing(db, tensor_root: Path) -> dict:
    """What the Models tab shows of runs: ``worker`` (`worker`'s, without ``kwargs``),
    ``baselines`` (`baselines`'s, without their rows), ``datasets`` (those of glomeruli:
    ``id``, ``title``, ``uploader``, ``n_flies``, ``panels``), ``runs`` (each with its
    ``title``, ``progress`` (`progress`'s), ``position`` (in the queue, from 1) and ``run``
    (the id the Models tab opens its spike counts by, once done)), and ``view``: the done
    runs' models, as `model_data.baseline_listing` lists models (``experiments``, ``models``,
    ``columns``)."""
    w = worker()
    runs = all_runs(db)
    queued = [r["id"] for r in sorted(runs, key=lambda r: (r["created_at"], r["id"])) if r["status"] == "queued"]
    view = {"experiments": {}, "models": []}
    keys: dict[str, int] = {}
    out_runs = []
    for r in runs:
        entry = {k: r[k] for k in ("id", "name", "created_at", "started_at", "finished_at", "status", "cancel", "options", "result", "error")}
        entry["options"] = {k: v for k, v in r["options"].items() if k != "baseline"} | {
            "baseline": {k: r["options"]["baseline"][k] for k in ("id", "name", "experiment")}}
        entry.update(title=title(db, r), progress=progress(runs, r), position=queued.index(r["id"]) + 1 if r["id"] in queued else None,
                     run=None)
        if r["status"] == "done":
            try:
                s = model_data.scan(model_data.job_view_dir(r["id"]))
                e, m = s["experiments"][0], s["models"][0]
            except (model_data.ScanError, IndexError):
                e = m = None
            if m and m["runs"]:
                key = f"run:{r['id']}"
                view["experiments"][key] = model_data._listed_experiment(None, f"run #{r['id']} ({entry['title']})", e)
                listed = model_data._listed_model(m, key, f"{model_data.JOB_RUN}{r['id']}")
                view["models"].append(listed)
                entry["run"] = listed["runs"][0]["id"]
                for k in [*e["params"], *e["keys"]]:
                    keys[k] = keys.get(k, 0) + 1
        out_runs.append(entry)
    view["columns"] = model_data._columns(keys, {k: [] for k in model_data.LISTS})
    datasets = [{"id": d["id"], "title": d["title"], "uploader": d["uploader_name"], "n_flies": d["n_flies"]}
                for d in sorted(db.list_datasets(), key=lambda d: d["id"]) if d["roi_kind"] == "glomeruli"]
    return {"worker": w and {k: v for k, v in w.items() if k != "kwargs"},
            "baselines": [{k: v for k, v in b.items() if k != "params"} for b in baselines(db, tensor_root)],
            "datasets": datasets, "runs": out_runs, "view": view, "tunings": list(TUNINGS)}


def stream_zip(run_id: int, top: str):
    """A run's folder (but its ``view``, a copy of what ``output`` has) as a zip, under ``top``, as it is written."""
    import zipfile

    from . import roi_training

    d = run_dir(run_id)
    sink = roi_training._Sink()
    when = dt.datetime.now().timetuple()[:6]
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for p in sorted(d.rglob("*")):
            rel = p.relative_to(d)
            if not p.is_file() or rel.parts[0] == "view":
                continue
            info = zipfile.ZipInfo(f"{top}/{rel.as_posix()}", when)
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w", force_zip64=True) as w, open(p, "rb") as fin:
                while chunk := fin.read(roi_training.CHUNK):
                    w.write(chunk)
                    yield sink.take()
            yield sink.take()
    yield sink.take()
