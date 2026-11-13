"""The model worker: runs the baseline models asked for in the Models tab (model_runs.py) on
datasets of glomeruli, one at a time, with pyolfsysm's `fit_and_plot_mb_model`. It is the same
app as the web server, in a container that also has olfsysm (the Dockerfile's ``model`` stage,
docker-compose.yml's ``model`` service); run as ``python -m app.model_worker``.

A run's model is run by the script written into its folder, ``run.py`` (`RUN_SCRIPT`), in a
process of its own (so that it can be stopped, and what it says kept: ``log.txt``): the same
script runs it again elsewhere, from the files beside it. As it starts, and every
`model_runs.WORKER_BEAT` seconds, the worker says what it is (`capabilities`:
`model_runs.WORKER_FILE`), from which the web server knows which baselines it can run.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config, model_data, model_runs, orn_input
from .db import Database
from .odormeta import OdorIndex
from .version import VERSION

POLL = 2.0  # seconds between looks for a run to start, and at one running (its log, whether to stop it)
OUTPUT = "output"  # in a run's folder: what fit_and_plot_mb_model is given to write in
MODEL = "model"  # the folder it makes there
CONDITION = "on"  # what the run's spike counts are called, as the baselines' (APL on)
LOG = "log.txt"
LOG_TAIL = 30  # lines of the log kept in a failed run's error

# the script that runs a run's model: written into its folder, and what the worker runs
RUN_SCRIPT = '''#!/usr/bin/env python3
"""Runs this folder's model: pyolfsysm's fit_and_plot_mb_model, on inputs/orn_deltas.csv (ORN
responses by glomerulus, as changes in spike rate), with the arguments of
inputs/model_kwargs.json (its ``model``: the baseline model's settings; and either ``fixed``:
its own threshold and APL weights, or ``target_sparsity``: the response rate to tune them to).
Its outputs go to output/model/ (which must not exist: move the one from olfdb away first).

Needs olfsysm (https://github.com/ejhonglab/olfsysm: `pip install -e .` in a checkout, see its
README), at the commit provenance.json gives (worker -> olfsysm) for the same numbers.
"""
import json
from pathlib import Path

from pyolfsysm import mb_model

here = Path(__file__).resolve().parent
kwargs = json.loads((here / "inputs" / "model_kwargs.json").read_text())
orn_deltas = mb_model.read_orn_deltas(here / "inputs" / "orn_deltas.csv")
model = dict(kwargs["model"])
if kwargs.get("fixed"):
    model.update(mb_model.fixed_params_to_fit_mb_model_kws(kwargs["fixed"]))
else:
    model["target_sparsity"] = kwargs["target_sparsity"]
params = mb_model.fit_and_plot_mb_model(here / "output", orn_deltas=orn_deltas, plot_dirname="model", try_cache=False, **model)
print("done:", {k: v for k, v in (params or {}).items() if isinstance(v, (str, int, float, bool))})
'''

README = """Model run #{id} of olfdb: {title}
Asked for by {name}, {created}; run {started} on {host}.

The baseline model {baseline} (of {experiment}), run by pyolfsysm's fit_and_plot_mb_model on the
mean responses of the glomeruli of {datasets} (panels: {panels}), {tuned}.

{scaled}

Files:
- output/model/: everything fit_and_plot_mb_model wrote. spike_counts.csv (and .parquet): each
  KC's spike count to each odor (KCs as rows); responses.csv: which are responses; params.csv and
  params.json: the model's parameters, as run; model_provenance.json: the olfsysm build, the
  commits of pyolfsysm and hong2p, the packages; orn_deltas.csv: its input, as it took it; the
  weights (w*.parquet); and its plots (.pdf).
- inputs/responses.csv: the glomeruli's mean responses (glomeruli x odors), in the datasets' units.
- inputs/orn_deltas.csv: those scaled into changes in spike rate: the model's input.
- inputs/scaling.json and scaling_points.csv: how they were scaled, and every response compared
  with Hallem & Carlson's; inputs/{hallem}: their table.
- inputs/baseline.json: the baseline's row of its params.csv; inputs/model_kwargs.json: the
  arguments of the call.
- run.py: the script that ran it. To run it again: move output/ away, then `python run.py`, with
  olfsysm installed (at the commit in provenance.json for the same numbers).
- provenance.json: all of the above in one place, with the datasets (their titles, files and
  checksums), who asked, and the software that ran it.
- log.txt: what the run printed.
"""


class Cancelled(Exception):
    pass


def host_name() -> str:
    try:
        return Path("/etc/host_hostname").read_text().strip() or platform.node()
    except OSError:
        return platform.node()


# what says what the model's environment is (`capabilities`): run with its Python, printing JSON
CAPS_SCRIPT = '''
import importlib.metadata, inspect, json, platform
import olfsysm
from pyolfsysm import mb_model

names = list(inspect.signature(mb_model.fit_mb_model).parameters) + list(inspect.signature(mb_model.fit_and_plot_mb_model).parameters)
skip = {"orn_deltas", "plot_dir", "model_kws", "plot_dirname", "try_cache"}
try:
    build = {k: v for k, v in olfsysm.build_info().items() if not k.endswith("diff")}
except Exception as e:  # (an older build with none)
    build = {"error": str(e)}
try:
    checkout = mb_model.model_provenance().get("pyolfsysm")
except Exception as e:
    checkout = {"error": str(e)}
versions = {"python": platform.python_version()}
for pkg in ("olfsysm", "hong2p", "natmix", "drosolf", "numpy", "pandas", "scipy", "xarray", "matplotlib", "seaborn", "statsmodels"):
    try:
        versions[pkg] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        pass
print("CAPS " + json.dumps({"kwargs": [n for n in dict.fromkeys(names) if n not in skip and not n.startswith("_")],
    "olfsysm": {"build": build, "checkout": checkout, "problems": list(mb_model.olfsysm_clean_problems())}, "versions": versions}, default=str))
'''


def model_python() -> str:
    """The Python that has olfsysm (its own environment in the worker's image; else this one)."""
    return os.environ.get("OLFDB_MODEL_PYTHON") or sys.executable


def capabilities() -> dict:
    """What this worker is: ``kwargs`` (the arguments pyolfsysm's `fit_mb_model` takes, which
    `fit_and_plot_mb_model` passes on, with its own), ``olfsysm`` (its build and checkout, as
    pyolfsysm says them, and what is wrong with them, if anything), ``versions`` (Python and
    the packages the model uses), ``host`` and ``olfdb`` (this app's version).

    Raises:
        RuntimeError: if the model's Python cannot import pyolfsysm.
    """
    p = subprocess.run([model_python(), "-c", CAPS_SCRIPT], capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"})
    line = next((x for x in p.stdout.splitlines() if x.startswith("CAPS ")), None)
    if p.returncode != 0 or line is None:
        raise RuntimeError(f"{model_python()} cannot say what pyolfsysm takes:\n{p.stderr[-3000:]}")
    return {"host": host_name(), **json.loads(line[5:]), "olfdb": VERSION}


def _say(caps: dict) -> None:
    d = model_runs.root()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (model_runs.WORKER_FILE + ".tmp")
    tmp.write_text(json.dumps({**caps, "at": time.time()}))
    tmp.replace(d / model_runs.WORKER_FILE)


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # (olfsysm's log is in color)


def _tail(path: Path, n: int, said: bool = False) -> list[str]:
    """The last `n` lines of a log that are not empty, without colors; with `said`, only those
    that say what it is doing (not warnings, nor lines continuing another)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 20_000))
            lines = [_ANSI.sub("", x) for x in f.read().decode("utf-8", "replace").splitlines()]
    except OSError:
        return []
    return [x for x in lines if x.strip() and not (said and (x[:1].isspace() or "Warning" in x or "warn(" in x))][-n:]


def _json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, default=str) + "\n")


def run(db: Database, r: dict, caps: dict) -> None:
    """Runs a run claimed (`model_runs.claim`'s): its inputs, the model, and what the Models
    tab reads of it; its row made done, failed (with why) or cancelled."""
    run_id, o = r["id"], r["options"]
    d = model_runs.run_dir(run_id)
    started = {}

    def stage(name: str, detail: Optional[str] = None) -> None:
        started.setdefault(name, model_runs.now())
        model_runs.update(db, run_id, progress={"stage": name, "detail": detail, "started": started})

    try:
        shutil.rmtree(d, ignore_errors=True)
        (d / "inputs").mkdir(parents=True)
        stage("inputs")
        index = OdorIndex(db.list_odors())
        s = orn_input.scale(db, o["datasets"], index, o["panels"])
        s["responses"].to_csv(d / "inputs" / "responses.csv")
        orn_input.model_frame(s).to_csv(d / "inputs" / "orn_deltas.csv")
        fits = [("the datasets", s["hallem"])] + ([("the combination of all datasets", s["combination"]["hallem"])] if s["combination"] else [])
        pd.DataFrame([{"of": of, **p} for of, h in fits for p in h["points"]]).to_csv(d / "inputs" / "scaling_points.csv", index=False)
        summary = orn_input.summary(s)
        _json(d / "inputs" / "scaling.json", {**summary, "said": orn_input.said(s), "hallem_shift": orn_input.HALLEM_SHIFT,
                                             "min_odors": orn_input.MIN_ODORS, "min_points": orn_input.MIN_POINTS})
        shutil.copyfile(config.REFERENCE_DIR / orn_input.HALLEM_FILE, d / "inputs" / orn_input.HALLEM_FILE)
        _json(d / "inputs" / "baseline.json", o["baseline"])
        _json(d / "inputs" / "model_kwargs.json", {"model": o["model"], "fixed": o["fixed"], "target_sparsity": o["target"]})
        (d / "run.py").write_text(RUN_SCRIPT)
        datasets = []
        for i in o["datasets"]:
            ds = db.get_dataset(i)
            datasets.append({k: ds.get(k) for k in ("id", "title", "created_at", "coding_unit", "data_file", "data_sha256")}
                            | {"uploader": ds["metadata"].get("uploader_name"), "response_calc": ds.get("response_calc")})
        how = ("with its own threshold and APL weights (fixed: the same model, on this input)" if o["fixed"]
               else f"its threshold and APL weights tuned again, to {o['target']:.4g} of KC-odor pairs responding")
        about = dict(id=run_id, title=model_runs.title(db, r), name=r["name"], created=r["created_at"], started=r["started_at"],
                     host=caps["host"], baseline=o["baseline"]["name"], experiment=o["baseline"]["experiment"],
                     datasets=", ".join(f"#{x['id']} ({x['title']})" for x in datasets), panels=", ".join(o["panels"]),
                     tuned=how, scaled=orn_input.said(s), hallem=orn_input.HALLEM_FILE)
        (d / "README.txt").write_text(README.format(**about))
        provenance = {"run": run_id, "asked_by": r["name"], "asked_at": r["created_at"], "started_at": r["started_at"],
                      "baseline": o["baseline"], "datasets": datasets, "panels": o["panels"], "tuning": o["tuning"],
                      "model_kwargs": {"model": o["model"], "fixed": o["fixed"], "target_sparsity": o["target"]},
                      "scaling": {**summary, "said": orn_input.said(s)},
                      "worker": {k: caps[k] for k in ("host", "olfsysm", "versions", "olfdb")},
                      "reproduce": "move output/ away, then `python run.py`, with olfsysm at worker.olfsysm.checkout.git_hash"}
        _json(d / "provenance.json", provenance)

        stage("model")
        t0 = time.time()
        with open(d / LOG, "wb") as log:
            child = subprocess.Popen([model_python(), "-u", "run.py"], cwd=d, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env={**os.environ, "MPLBACKEND": "Agg"})
            last = None
            try:
                while child.poll() is None:
                    time.sleep(POLL)
                    if (model_runs.get(db, run_id) or {"cancel": True})["cancel"]:
                        raise Cancelled
                    line = (_tail(d / LOG, 1, said=True) or [None])[0]
                    if line != last:
                        last = line
                        stage("model", line and line[:200])
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        if child.returncode != 0:
            raise RuntimeError("fit_and_plot_mb_model did not finish (log.txt, in its download, has all it said):\n"
                               + "\n".join(_tail(d / LOG, LOG_TAIL)))
        seconds = time.time() - t0

        stage("outputs")
        out = d / OUTPUT / MODEL
        counts, responses = pd.read_parquet(out / "spike_counts.parquet"), pd.read_parquet(out / "responses.parquet")
        params = json.loads((out / "params.json").read_text())
        view = d / model_data.JOB_VIEW
        view.mkdir()
        tag = f"run{run_id}"
        file = f"{tag}_{CONDITION}{model_data.SUFFIX}"
        shutil.copyfile(out / "spike_counts.parquet", view / file)
        rate = float(responses.to_numpy(dtype=float).mean())
        scalars = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool)) and k not in ("output_dir", "used_model_cache")}
        row = {"tag": tag, "base": o["baseline"]["name"], "baseline_experiment": o["baseline"]["experiment"],
               "datasets": ", ".join(f"#{i}" for i in o["datasets"]), "panels": ", ".join(o["panels"]), "tuning": o["tuning"],
               "scaled_by": s["method"], "scale_factor": s["factor"], "n_glomeruli_in": len(s["glomeruli"]), "n_odors": len(s["odors"]),
               "n_kcs": len(counts), "rate_on": rate, **scalars}
        pd.DataFrame([row]).to_csv(view / model_data.PARAMS, index=False)
        _json(view / model_data.PROVENANCE, {model_data.GENERATED: model_runs.now(), "script": "olfdb model run (run.py)",
                                             "host": caps["host"], "asked_by": r["name"], "scaling": orn_input.said(s),
                                             "olfsysm_commit": (caps["olfsysm"].get("checkout") or {}).get("git_hash")})
        files = [p for p in d.rglob("*") if p.is_file() and model_data.JOB_VIEW not in p.relative_to(d).parts]
        provenance.update(finished_at=model_runs.now(), seconds=round(seconds, 1), response_rate=rate, params=scalars)
        _json(d / "provenance.json", provenance)
        result = {"seconds": round(seconds, 1), "rate": rate, "n_kcs": len(counts), "n_odors": len(s["odors"]),
                  "n_glomeruli": len(s["glomeruli"]), "scaling": {**summary, "said": orn_input.said(s)},
                  "files": len(files), "bytes": sum(p.stat().st_size for p in files),
                  "plots": sum(p.suffix == ".pdf" for p in files), "fixed_thr": scalars.get("fixed_thr"),
                  "wAPLKC_scale": scalars.get("wAPLKC_scale")}
        model_runs.update(db, run_id, status="done", finished_at=model_runs.now(), result=result, progress=None)
    except Cancelled:
        shutil.rmtree(d, ignore_errors=True)
        model_runs.update(db, run_id, status="cancelled", finished_at=model_runs.now(), progress=None)
    except orn_input.InputError as e:
        shutil.rmtree(d, ignore_errors=True)
        model_runs.update(db, run_id, status="failed", finished_at=model_runs.now(), error=str(e), progress=None)
    except Exception as e:  # (kept with its folder: its log says more)
        model_runs.update(db, run_id, status="failed", finished_at=model_runs.now(), error=f"{type(e).__name__}: {e}"[:6000], progress=None)


def main() -> None:
    config.ensure_dirs()
    Path.home().mkdir(parents=True, exist_ok=True)  # (where pyolfsysm keeps the learning rates that tune a model in one step)
    db = Database(config.DB_PATH)
    caps = capabilities()
    stop = threading.Event()

    def beat() -> None:
        while not stop.is_set():
            try:
                _say(caps)
            except OSError as e:
                print(f"model worker: could not write {model_runs.WORKER_FILE}: {e}", flush=True)
            stop.wait(model_runs.WORKER_BEAT)

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    threading.Thread(target=beat, daemon=True).start()
    for i in model_runs.interrupt_running(db):
        print(f"model worker: run {i} was left running: marked interrupted", flush=True)
    problems = caps["olfsysm"]["problems"]
    print(f"model worker on {caps['host']}: olfsysm {(caps['olfsysm'].get('checkout') or {}).get('git_hash')}"
          + (f"; NOT a clean build: {problems}" if problems else ""), flush=True)
    try:
        while True:
            r = model_runs.claim(db)
            if r is None:
                time.sleep(POLL)
                continue
            print(f"model worker: run {r['id']} ({model_runs.title(db, r)}) started, {dt.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
            run(db, r, caps)
            print(f"model worker: run {r['id']} {model_runs.get(db, r['id'])['status']}", flush=True)
    finally:
        stop.set()
        model_runs.interrupt_running(db)


if __name__ == "__main__":
    main()
