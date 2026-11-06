"""The fit worker: runs the fits asked for in the AL Segmentation tab (`fits`), one at a time,
with al-diag-seg, on the GPU if there is one: ``python -m app.fit_worker``. In Docker it is
the ``worker`` service (the image's ``worker`` target: this app, al-diag-seg, PyTorch and
Cellpose-SAM's weights), the only one given the GPU.

It is held to the folders the server may use as the web server is (`sandbox.start`), and
may also write to the GPU's device files (and its own threads' /proc entries, as CUDA does); the
files it reads and writes in them are noted as
the web server's are (`filelog`), for the Admin tab. It says it is there every `fits.HEARTBEAT_S`
seconds (`fits.heartbeat`), with which GPU and al-diag-seg it has, for the tab to show; and
as it starts, marks fits a worker before it left running interrupted.
"""
from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
import shutil
import socket
import threading
import time
import traceback
from importlib import metadata
from pathlib import Path

from . import config, filelog, fits, roi_training, sandbox
from .db import Database

log = logging.getLogger("fit_worker")
POLL_S = 3
SAVE_S = 2  # progress is saved (and a request to stop looked for) at most this often


class Cancelled(Exception):
    pass


def host_name() -> str:
    try:
        return config.HOST_HOSTNAME_FILE.read_text().strip() or socket.gethostname()
    except OSError:
        return socket.gethostname()


def gpu() -> dict:
    """The GPU PyTorch sees (``gpu``: its name, None if none) and ``torch``'s version."""
    try:
        import torch
    except ImportError:
        return {"gpu": None, "torch": None}
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__}


def versions() -> dict:
    """al-diag-seg's (``alseg``: version and commit) and hong2p's (``hong2p``: the commit
    pip installed it from), as a model's README says to install them."""
    out = {"alseg": None, "hong2p": None}
    try:
        from alseg.model import version

        out["alseg"] = version()
    except ImportError:
        pass
    try:
        direct = json.loads(metadata.distribution("hong2p").read_text("direct_url.json") or "{}")
        out["hong2p"] = (direct.get("vcs_info") or {}).get("commit_id")
    except metadata.PackageNotFoundError:
        pass
    return out


class Progress:
    """al-diag-seg's progress callback for a fit, ``(stage, done, total)``: saves where it is
    (at each new stage, else at most every `SAVE_S` seconds), times each stage, and stops the
    fit (`Cancelled`) when asked to."""

    def __init__(self, db, fit_id: int):
        self.db, self.id = db, fit_id
        self.stage, self.started, self.seconds, self.last = None, {}, {}, 0.0

    def __call__(self, stage: str, done: int, total: int) -> None:
        t = time.time()
        new = stage != self.stage
        if new:
            self.finish(t)
            self.stage, self.started[stage] = stage, t
        if new or t - self.last >= SAVE_S:
            self.last = t
            fits.update(self.db, self.id, progress={"stage": stage, "done": done, "total": total,
                                                    "stage_started": self.started[stage], "seconds": self.seconds})
            if fits.get(self.db, self.id)["cancel"]:
                raise Cancelled()

    def finish(self, t=None) -> None:
        if self.stage:
            self.seconds[self.stage] = (t or time.time()) - self.started[self.stage]
            self.stage = None


def summary(res: dict) -> dict:
    """What the tab shows of a fit's scores (al-diag-seg's ``Model.evaluate``): per held-out fly,
    its final ROIs' recall, precision, Dice and leak (and its manual ROIs' leak), and how
    many were named as the manual ROI they overlap; their means; and per glomerulus."""
    final = res["final"]
    rows = {r["fly"]: r for r in res["summary"] if r["rois"] == final}
    manual = {r["fly"]: r for r in res["summary"] if r["rois"] == "manual"}
    named = {r["fly"]: r for r in res["identity"] if r["rois"] == final}
    num = lambda v: v if isinstance(v, (int, float)) else None  # noqa: E731
    flies = [{"fly": f, "n": r["n"], **{k: num(r[k]) for k in ("recall", "precision", "dice", "leak")},
              "manual_leak": num(manual.get(f, {}).get("leak")), "named": named.get(f, {}).get("correct", 0),
              "named_of": named.get(f, {}).get("total", 0)} for f, r in rows.items()]
    mean = {}
    for k in ("recall", "precision", "dice", "leak", "manual_leak"):
        vs = [f[k] for f in flies if f[k] is not None]
        mean[k] = sum(vs) / len(vs) if vs else None
    return {"final": final, "flies": flies, "mean": mean, "named": sum(f["named"] for f in flies),
            "named_of": sum(f["named_of"] for f in flies), "glomeruli": res["glomeruli"]}


README = """al-diag-seg model: fit #{id} in the odor response database's AL Segmentation tab, asked for by
{name} on {created}. Fit to {n_train} flies; scored on {n_val} held out ({held_out}).
Odors it needs (each fly must have all of them): {odors}.

To run it on your own flies (Python 3.11 to 3.13; with a GPU it is faster, but not needed):

  pip install "hong2p @ git+https://github.com/ejhonglab/hong2p{hong2p}" \\
              "al-diag-seg @ git+https://github.com/tom-f-oconnell/al-diag-seg{alseg}"
  alseg-predict model/ --fly <a fly's motion-corrected data folder> --raw <its raw data folder> --out predictions/

e.g. --fly /local/tensor/Ann/analysis/2026-03-31/1 --raw /local/tensor/Ann/raw/2026-03-31/1, as al_analysis
leaves them (its diagnostic recordings' trialmean_dff.tif and trial_frames_and_odors.json, and
mocorr_concat_avg.tif or mocorr_concat.tif; and the Experiment.xml of those recordings, and anat/ if it has
one). predictions/<fly>_RoiSet.zip then has its ROIs, named by glomerulus (numbered where none fits), per
plane, for ImageJ and al_analysis. alseg-predict -h says more.

results/ has this fit's held-out flies: each kind of predicted ROIs as labels (<fly>_<kind>.tif) and as named
RoiSets, the names given (identities CSVs), and their scores against the flies' own ROIs (summary.csv,
glomeruli.csv).
"""


def run(db, fit: dict) -> None:
    """Runs one fit (claimed: `fits.claim`) to the end: done, failed or cancelled."""
    d = fits.fit_dir(fit["id"])
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    o, prog = {**fits.DEFAULTS, **fit["options"]}, Progress(db, fit["id"])
    status, error, result = "failed", None, None
    with open(d / "log.txt", "a", buffering=1) as out, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        handler = logging.StreamHandler(out)
        logging.getLogger().addHandler(handler)
        try:
            print(f"fit {fit['id']}, asked for by {fit['name']} ({fit['ip']}) at {fit['created_at']}; options {json.dumps(o)}")
            prog("load", 0, 0)
            bases = [Path(p) for p in sandbox.folders()]
            found = roi_training.inventory(db, config.TENSOR_ROOT, bases)
            m = fits.manifest(found, bases)
            if len(m["flies"]) < 2:
                raise ValueError(f"{len(m['flies'])} flies to fit to (the AL Segmentation tab lists them): at least 2 are needed")
            (d / "manifest.json").write_text(json.dumps(m, indent=1))
            names = fits.fly_names(m["flies"])
            held = fits.choose_held_out(m["flies"], o["held_out"])
            flies = {"train": [n for n in names if n not in held], "held_out": held,
                     "stacks": sum(1 for f in m["flies"] if f["stacks"]),
                     "datasets": sorted({x["id"] for f in m["flies"] for x in f["datasets"]})}
            fits.update(db, fit["id"], flies=flies)
            print(f"{len(names)} flies ({len(found['skipped'])} left out, as the tab says why); held out: {', '.join(held)}")
            from alseg import model as am
            from alseg import olfdb as ao

            data = ao.load_export(d / "manifest.json", tentative=o["tentative"], progress=prog)
            (d / "work").mkdir()  # (Cellpose makes only its models/ in the folder it is given)
            model = am.fit(data["flies"], held, am.Options(epochs=o["epochs"], atlas=o["atlas"], tentative=o["tentative"]),
                           out=d / "work", progress=prog, odors=data["odors"], aliases=data["aliases"],
                           glomeruli=data["glomeruli"])
            model.info["olfdb"] = {"fit": fit["id"], "by": fit["name"], "datasets": flies["datasets"]}
            model.save(d / "model")
            v = versions()
            (d / "model" / "README.txt").write_text(README.format(
                id=fit["id"], name=fit["name"], created=fit["created_at"], n_train=len(flies["train"]), n_val=len(held),
                held_out=", ".join(held), odors=", ".join(model.odors),
                hong2p=f"@{v['hong2p']}" if v["hong2p"] else "",
                alseg=f"@{v['alseg']['commit']}" if v["alseg"] and v["alseg"]["commit"] else ""))
            prog("evaluate", 0, len(held))
            res = model.evaluate([f for f in data["flies"] if f["name"] in held], d / "results", progress=prog)
            prog.finish()
            result, status = summary(res), "done"
        except Cancelled:
            status = "cancelled"
            print("stopped, as asked")
        except Exception as e:  # noqa: BLE001 (any failure is the fit's, said in its log and record)
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            logging.getLogger().removeHandler(handler)
    prog.finish()
    done = fits.get(db, fit["id"]) or fit
    sz = fits.sizes(done)
    model_bytes = sum(p.stat().st_size for p in (d / "model").rglob("*") if p.is_file()) if (d / "model").is_dir() else None
    fits.update(db, fit["id"], status=status, finished_at=fits.now(), error=error, result=result, model_bytes=model_bytes,
                stages={s: {"seconds": prog.seconds.get(s, 0.0), "size": sz[s]} for s in fits.STAGES} if status == "done" else None)
    log.info("fit %s: %s%s", fit["id"], status, f" ({error})" if error else "")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    db = Database(config.DB_PATH)
    filelog.start(db)  # first, to see what it reads as it starts
    # CUDA opens the GPU's device files, and its threads' /proc entries, to write to them
    status = sandbox.start(files=[*sorted(glob.glob("/dev/nvidia*")), "/proc/self/task"])
    for i in fits.interrupt_running(db):
        log.warning("fit %s was left running by a worker before this one: marked interrupted", i)
    info = {"host": host_name(), "pid": os.getpid(), "started": fits.now(), **gpu(), **versions(),
            "landlock": status["kernel"]["enforced"]}
    log.info("fit worker on %s: GPU %s, al-diag-seg %s", info["host"], info["gpu"], info["alseg"])
    state = {"fit": None}

    def beat():
        n = 0
        while True:
            try:
                fits.heartbeat(db, {**info, "fit": state["fit"]})
                n += 1
                if n * fits.HEARTBEAT_S % filelog.SAVE_S < fits.HEARTBEAT_S:
                    filelog.flush(db)
            except Exception:  # noqa: BLE001
                log.exception("could not say the worker is there, or save the files it read and wrote")
            time.sleep(fits.HEARTBEAT_S)

    threading.Thread(target=beat, daemon=True).start()
    while True:
        fit = fits.claim(db)
        if fit is None:
            time.sleep(POLL_S)
            continue
        state["fit"] = fit["id"]
        log.info("fit %s: started (asked for by %s)", fit["id"], fit["name"])
        try:
            run(db, fit)
        finally:
            state["fit"] = None


if __name__ == "__main__":
    main()
