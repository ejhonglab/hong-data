"""The fit worker: runs the fits asked for in the AL segmentation tab (`fits`), one at a time,
with al-diag-seg, on the GPU if there is one: ``python -m app.fit_worker``. In Docker it is
the ``worker`` service (the image's ``worker`` target: this app, al-diag-seg, PyTorch and
Cellpose-SAM's weights), the only one given the GPU.

Beside them, in a thread of its own, it predicts the ROIs of flies with fits' models, as asked
in the tab (`predictions`), one at a time: on the GPU when no fit is using it (a fit waits for
one to end), else on the CPU, so that a prediction does not wait hours for a fit. Each thread's
printing goes to its own log (`_Streams`).

It is held to the folders the server may use as the web server is (`sandbox.start`), and
may also write to the GPU's device files (and its own threads' /proc entries, as CUDA does); the
files it reads and writes in them are noted as
the web server's are (`filelog`), for the Admin tab. It says it is there every `fits.HEARTBEAT_S`
seconds (`fits.heartbeat`), with which GPU and al-diag-seg it has, for the tab to show; and
as it starts, marks fits a worker before it left running interrupted.
"""
from __future__ import annotations

import contextlib
import csv
import ctypes
import datetime as dt
import gc
import glob
import json
import logging
import os
import shutil
import socket
import sys
import threading
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Optional

from . import config, filelog, fits, odormeta, predictions, roi_training, sandbox, version
from .db import Database
from .glomeruli import load_glomeruli

log = logging.getLogger("fit_worker")
POLL_S = 3
SAVE_S = 2  # progress is saved (and a request to stop looked for) at most this often


class Cancelled(Exception):
    pass


class _Streams:
    """sys.stdout or sys.stderr for the worker's threads: each thread's own (`_output_to`: its
    fit's or prediction's log), else the process's, as a fit and a prediction run at once."""

    def __init__(self, default):
        self.default, self.local = default, threading.local()

    def _out(self):
        return getattr(self.local, "out", None) or self.default

    def write(self, s):
        return self._out().write(s)

    def flush(self):
        self._out().flush()

    def __getattr__(self, name):
        return getattr(self._out(), name)


@contextlib.contextmanager
def _output_to(out):
    """This thread's printing, and logging, into `out`: through `_Streams`, where `main` has put
    them in place; else by redirecting the process's (one fit or prediction at a time, as tests
    run them)."""
    handler = logging.StreamHandler(out)
    me = threading.get_ident()
    handler.addFilter(lambda r: r.thread == me)
    logging.getLogger().addHandler(handler)
    try:
        if isinstance(sys.stdout, _Streams) and isinstance(sys.stderr, _Streams):
            prev = (getattr(sys.stdout.local, "out", None), getattr(sys.stderr.local, "out", None))
            sys.stdout.local.out = sys.stderr.local.out = out
            try:
                yield
            finally:
                sys.stdout.local.out, sys.stderr.local.out = prev
        else:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                yield
    finally:
        logging.getLogger().removeHandler(handler)


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


SCORES = ("recall", "precision", "dice", "recall_2d", "precision_2d", "dice_2d", "leak")


def summary(res: dict) -> dict:
    """What the tab shows of a fit's scores (al-diag-seg's ``Model.evaluate``): per held-out fly,
    its final ROIs' recall, precision and Dice, matched to its own as 3D objects and (``*_2d``,
    where al-diag-seg scores them so) plane by plane, their leak (and its manual ROIs'), and how
    many were named as the manual ROI they overlap; their means; and per glomerulus."""
    final = res["final"]
    rows = {r["fly"]: r for r in res["summary"] if r["rois"] == final}
    manual = {r["fly"]: r for r in res["summary"] if r["rois"] == "manual"}
    named = {r["fly"]: r for r in res["identity"] if r["rois"] == final}
    num = lambda v: v if isinstance(v, (int, float)) else None  # noqa: E731
    flies = [{"fly": f, "n": r["n"], **{k: num(r.get(k)) for k in SCORES}, "manual_leak": num(manual.get(f, {}).get("leak")),
              "named": named.get(f, {}).get("correct", 0), "named_of": named.get(f, {}).get("total", 0)} for f, r in rows.items()]
    mean = {}
    for k in (*SCORES, "manual_leak"):
        vs = [f[k] for f in flies if f[k] is not None]
        mean[k] = sum(vs) / len(vs) if vs else None
    return {"final": final, "flies": flies, "mean": mean, "named": sum(f["named"] for f in flies),
            "named_of": sum(f["named_of"] for f in flies), "glomeruli": res["glomeruli"]}


README = """al-diag-seg model: fit #{id} in the odor response database's AL segmentation tab, asked for by
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
    with open(d / "log.txt", "a", buffering=1) as out, _output_to(out):
        try:
            print(f"fit {fit['id']}, asked for by {fit['name']} ({fit['ip']}) at {fit['created_at']}; options {json.dumps(o)}")
            refs = [fits.reference_dir(r) for r in o["refs"]]
            for r, p in zip(o["refs"], refs):
                if not (p / "labels.tif").is_file():
                    raise FileNotFoundError(f"{p}: no labels.tif, so not the {fits.REFERENCES[r]} atlas, which the "
                                            "worker's image has (its Dockerfile downloads it)")
            prog("load", 0, 0)
            bases = [Path(p) for p in sandbox.folders()]
            found = roi_training.inventory(db, config.TENSOR_ROOT, bases)
            m = fits.manifest(found, bases)
            if len(m["flies"]) < 2:
                raise ValueError(f"{len(m['flies'])} flies to fit to (the AL segmentation tab lists them): at least 2 are needed")
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
            model = am.fit(data["flies"], held, am.Options(epochs=o["epochs"], atlas=o["atlas"], tentative=o["tentative"],
                                                           refs=[str(p) for p in refs]),
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
    prog.finish()
    done = fits.get(db, fit["id"]) or fit
    sz = fits.sizes(done)
    model_bytes = sum(p.stat().st_size for p in (d / "model").rglob("*") if p.is_file()) if (d / "model").is_dir() else None
    fits.update(db, fit["id"], status=status, finished_at=fits.now(), error=error, result=result, model_bytes=model_bytes,
                stages={s: {"seconds": prog.seconds.get(s, 0.0), "size": sz[s]} for s in fits.STAGES} if status == "done" else None)
    log.info("fit %s: %s%s", fit["id"], status, f" ({error})" if error else "")

# ---------------------------------------------------------------- predictions

PRED_README = """Glomerulus ROIs of {fly}, predicted by the model of fit #{fit} in the odor response database's
AL segmentation tab, asked for by {name} on {created}, on the {device}.

{fly}_RoiSet.zip    the ROIs to use ({final}: {final_what}), named by glomerulus where one fits
                    (numbered where none does), per plane (hyperstack Z), for ImageJ and al_analysis
{fly}_<kind>.tif and {fly}_<kind>_RoiSet.zip
                    each kind of ROIs al-diag-seg makes, as label volumes (Z, Y, X) and RoiSets:
                    pred (Cellpose's), refined_tuning (refined by odor tuning), refined_prior (refined
                    with the atlas, if the model has one)
{fly}_identities*.csv
                    the name given each ROI (by its label in the .tif), with its cost and confidence{manual_ids}
average.tif         the fly's average image, as the model saw it
{scores}provenance.json     the model (its fit, its flies, the al-diag-seg that made it), the fly's files read (with
                    their sizes and when they were modified), the software that predicted, and who asked
log.txt             what it printed

To make these again elsewhere, download fit #{fit}'s model, and run alseg-predict on the fly, as its README says.
"""
KINDS = {"pred": "Cellpose's own, named", "refined_tuning": "refined by odor tuning", "refined_prior": "refined with the atlas"}


def _free_gpu() -> None:
    """Gives back to the GPU the memory PyTorch keeps for reuse once a fit or prediction is done
    with it (its cache: what a fit's training peaked at, about 18 GB), so that other programs, and
    a prediction beside the next fit, have it. Only CUDA's own context (a few hundred MB) stays.
    Says how much it gave back."""
    if "torch" not in sys.modules:  # (nothing has used the GPU)
        return
    try:
        import torch

        if torch.cuda.is_available():
            gc.collect()  # (tensors only cycles of objects still hold)
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            log.info("gave back %.1f GB of the GPU's memory (%.1f GB still in use)", (before - torch.cuda.memory_reserved()) / 2**30,
                     torch.cuda.memory_allocated() / 2**30)
    except Exception:  # noqa: BLE001
        log.exception("could not free the GPU's memory")


def _rss_gb() -> Optional[float]:
    """The worker's resident memory (RAM), in GB; None where /proc does not say (not Linux)."""
    try:
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("VmRSS:")) / 2**20
    except (OSError, StopIteration, ValueError):
        return None


def _free_ram() -> None:
    """Gives back to the system the RAM a fit or prediction freed but glibc's allocator kept for
    reuse (``malloc_trim``): idle after fit 2, the worker held 6.8 GB of RAM, 5.5 GB of it its heap,
    though loading the flies alone left nothing behind. Says how much it held before and after,
    so the next fit tells whether that was it, or memory still referenced (which only a process of
    its own per fit would give back)."""
    before = _rss_gb()
    if before is None:
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # (glibc, as the worker's image has)
    except (OSError, AttributeError):
        log.exception("could not give back the RAM freed")
        return
    log.info("RAM in use: %.1f GB, %.1f GB after giving back what was freed", before, _rss_gb())


def _file(p: Path) -> Optional[dict]:
    """A file read, for a prediction's provenance: its path, size and when it was modified."""
    try:
        st = os.stat(p)
    except OSError:
        return None
    return {"path": str(p), "bytes": st.st_size,
            "modified": dt.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")}


def _fly_inputs(fly_dir: Path, raw_dir: Optional[Path], model, index) -> dict:
    """What al-diag-seg reads of a fly (`alseg.fly.load_fly`), for a prediction's provenance:
    each recording with its odors' mean dF/F (and whether it was used: more than half its
    odors the model's), the average image, the fly's own ROIs and its anatomical stack. Teaches
    the model the fly's odors as recorded, as the odor table names them, where it does not
    know them yet (as the database's export names a fit's flies' odors)."""
    from alseg import fly as af

    want, recs = set(model.odors), []
    for r in sorted(x for x in fly_dir.iterdir() if x.is_dir()):
        if not ((r / af.TRIALMEAN).is_file() and (r / af.TRIALS).is_file()):
            continue
        odors = roi_training.recording_odors(r / af.TRIALS, index) or []
        for o in odors:
            model.aliases.setdefault(o["recorded"], o["odor"])
        names = [model.aliases.get(o["recorded"], o["recorded"]) for o in odors]
        xml = raw_dir / r.name / "Experiment.xml" if raw_dir else None
        recs.append({"name": r.name, "used": sum(n in want for n in names) > len(names) / 2, "odors": names,
                     "responses": _file(r / af.TRIALMEAN), "trials": _file(r / af.TRIALS),
                     "experiment_xml": _file(xml) if xml and xml.is_file() else None})
    avg = fly_dir / af.CONCAT_AVG if (fly_dir / af.CONCAT_AVG).is_file() else fly_dir / af.CONCAT
    used = [fly_dir / r["name"] for r in recs if r["used"]]
    rois = fly_dir / af.ROISET if (fly_dir / af.ROISET).is_file() else next((r / af.ROISET for r in used if (r / af.ROISET).is_file()), None)
    stack = next((x for x in sorted(raw_dir.iterdir()) if x.is_dir() and af.ANATOMICAL.fullmatch(x.name)
                  and (x / "Experiment.xml").is_file()), None) if raw_dir and model.atlas is not None else None
    return {"recordings": recs, "average": {**(_file(avg) or {"path": str(avg)}), "computed": avg.name == af.CONCAT},
            "roiset": str(rois) if rois else None, "roiset_file": _file(rois) if rois else None,
            "anatomical_stack": str(stack) if stack else None}


def run_prediction(db, p: dict, on_gpu: bool, why_cpu: Optional[str] = None) -> None:
    """Runs one prediction (claimed: `predictions.claim`) to the end, on the GPU or the CPU:
    done or failed. Its folder is made anew (`predictions` says what is in it)."""
    d = predictions.pred_dir(p["id"])
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    t0 = time.time()
    device = f"GPU ({gpu()['gpu']})" if on_gpu else f"CPU{f' ({why_cpu})' if why_cpu else ''}"
    status, error, result = "failed", None, None
    with open(d / "log.txt", "a", buffering=1) as out, _output_to(out):
        try:
            print(f"prediction {p['id']}: fit {p['fit_id']}'s model, on fly {p['fly']} ({p['fly_dir']}; raw data "
                  f"{p['raw_dir']}), asked for by {p['name']} ({p['ip']}) at {p['created_at']}; on the {device}")
            predictions.update(db, p["id"], progress={"stage": "load", "device": "GPU" if on_gpu else "CPU"})
            fit = fits.get(db, p["fit_id"])
            mdir = fits.fit_dir(p["fit_id"]) / "model"
            if fit is None or not (mdir / "model.json").is_file():
                raise FileNotFoundError(f"fit #{p['fit_id']}'s model is not there (the fit was deleted)")
            import numpy as np
            from alseg import model as am
            from alseg import roi_io
            from alseg.fly import load_fly

            model = am.Model.load(mdir)
            model.options.gpu = on_gpu
            fly_dir, raw_dir = Path(p["fly_dir"]), Path(p["raw_dir"]) if p["raw_dir"] else None
            inputs = _fly_inputs(fly_dir, raw_dir, model, odormeta.OdorIndex(db.list_odors()))
            f = load_fly(fly_dir, model, raw_dir=raw_dir, name=p["fly"], glomeruli=load_glomeruli(),
                         tentative=model.options.tentative)
            print(f"{sum(r['used'] for r in inputs['recordings'])} recordings of its odors, planes (z, y, x) "
                  f"{f['average'].shape} of {np.round(f['spacing'], 3).tolist()} um"
                  + (f", its own ROIs from {inputs['roiset']}" if f.get("labels") is not None else ", no ROIs of its own")
                  + (", its anatomical stack" if f.get("hires") else ""))
            predictions.update(db, p["id"], progress={"stage": "predict", "device": "GPU" if on_gpu else "CPU"})
            name, scores = p["fly"], None
            if f.get("labels") is not None:
                res = model.evaluate([f], d)
                final = res["final"]
                row = lambda kind: next(r for r in res["summary"] if r["rois"] == kind)  # noqa: E731
                num = lambda v: v if isinstance(v, (int, float)) else None  # noqa: E731
                ident = next((r for r in res["identity"] if r["rois"] == final), {})
                scores = {"manual": row("manual")["n"], "manual_2d": row("manual").get("n_2d"),
                          **{k: num(row(final).get(k)) for k in SCORES}, "manual_leak": num(row("manual")["leak"]),
                          "named": ident.get("correct", 0), "named_of": ident.get("total", 0)}
                n = row(final)["n"]
                with open(d / f"{name}_identities_{final}.csv", newline="") as fh:
                    named = sorted({r["assigned"] for r in csv.DictReader(fh) if r["assigned"]})
            else:
                pr = model.predict(f)
                final = pr["final"]
                for kind, lab in pr["labels"].items():
                    am.write_volume(d / f"{name}_{kind}.tif", lab.astype(np.uint16))
                    roi_io.write_roiset(lab, {l: v[0] for l, v in pr["identities"][kind].items()}, d / f"{name}_{kind}_RoiSet.zip")
                if pr["atlas_maxprob"] is not None:
                    am.write_volume(d / f"{name}_atlas_maxprob.tif", pr["atlas_maxprob"])
                ids = pr["identities"][final]
                with open(d / f"{name}_identities.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["label", "assigned", "cost", "confidence"])
                    w.writerows([l, nm or "", f"{cost:.3f}", f"{conf:.3f}"] for l, (nm, cost, conf) in sorted(ids.items()))
                n = len(np.unique(pr["labels"][final])) - 1
                named = sorted({v[0] for v in ids.values() if v[0]})
            shutil.copyfile(d / f"{name}_{final}_RoiSet.zip", d / f"{name}_RoiSet.zip")
            am.write_volume(d / predictions.AVERAGE, f["average"].astype(np.float32))
            seconds = time.time() - t0
            result = {"final": final, "n": int(n), "named": named, "scores": scores, "device": "GPU" if on_gpu else "CPU",
                      "seconds": seconds}
            v = versions()
            prov = {
                "prediction": p["id"], "asked_for": {"by": p["name"], "from": p["ip"], "at": p["created_at"]},
                "started": p["started_at"], "finished": predictions.now(), "seconds": seconds, "device": device,
                "fly": {"name": name, "dir": p["fly_dir"], "raw_dir": p["raw_dir"], "spacing_um": list(f["spacing"]),
                        "planes": list(f["average"].shape), **inputs},
                "model": {"fit": p["fit_id"], "folder": str(mdir), "fit_by": fit["name"], "fit_asked_for": fit["created_at"],
                          "fit_finished": fit["finished_at"], "options": fit["options"], "flies": fit["flies"],
                          "odors": model.odors, "info": model.info},
                "software": {"al-diag-seg": v["alseg"], "hong2p": v["hong2p"], "olfdb": version.VERSION, **gpu()},
                "outputs": {"rois": f"{name}_RoiSet.zip", "final": final, "n_rois": int(n), "named": named, "scores": scores},
            }
            (d / predictions.PROVENANCE).write_text(json.dumps(prov, indent=1, default=str))
            (d / "README.txt").write_text(PRED_README.format(
                fly=name, fit=p["fit_id"], name=p["name"], created=p["created_at"], device=device, final=final,
                final_what=KINDS.get(final, final),
                manual_ids=", and the fly's own ROI it overlaps" if scores else "",
                scores="summary.csv and glomeruli.csv\n                    each kind's scores against the fly's own ROIs\n" if scores else ""))
            print(f"{n} ROIs ({final}), {len(named)} named; {seconds:.0f} s")
            status = "done"
        except Exception as e:  # noqa: BLE001 (any failure is the prediction's, said in its log and record)
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    predictions.update(db, p["id"], status=status, finished_at=predictions.now(), error=error, result=result,
                       progress={"stage": status, "device": "GPU" if on_gpu else "CPU"})
    log.info("prediction %s: %s%s", p["id"], status, f" ({error})" if error else "")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # (after logging's own handler, which keeps writing to the process's stderr)
    sys.stdout, sys.stderr = _Streams(sys.stdout), _Streams(sys.stderr)
    db = Database(config.DB_PATH)
    filelog.start(db)  # first, to see what it reads as it starts
    # CUDA opens the GPU's device files, and its threads' /proc entries, to write to them
    status = sandbox.start(files=[*sorted(glob.glob("/dev/nvidia*")), "/proc/self/task"])
    for i in fits.interrupt_running(db):
        log.warning("fit %s was left running by a worker before this one: marked interrupted", i)
    for i in predictions.interrupt_running(db):
        log.warning("prediction %s was left running by a worker before this one: marked interrupted", i)
    info = {"host": host_name(), "pid": os.getpid(), "started": fits.now(), **gpu(), **versions(),
            "landlock": status["kernel"]["enforced"]}
    log.info("fit worker on %s: GPU %s, al-diag-seg %s", info["host"], info["gpu"], info["alseg"])
    state = {"fit": None, "prediction": None}
    gpu_lock = threading.Lock()  # held by a fit, or a prediction on the GPU

    def beat():
        n = 0
        while True:
            try:
                fits.heartbeat(db, {**info, "fit": state["fit"], "prediction": state["prediction"]})
                n += 1
                if n * fits.HEARTBEAT_S % filelog.SAVE_S < fits.HEARTBEAT_S:
                    filelog.flush(db)
            except Exception:  # noqa: BLE001
                log.exception("could not say the worker is there, or save the files it read and wrote")
            time.sleep(fits.HEARTBEAT_S)

    threading.Thread(target=beat, daemon=True).start()
    threading.Thread(target=predict_loop, args=(db, state, gpu_lock, bool(info["gpu"])), daemon=True).start()
    fit_loop(db, state, gpu_lock)


def fit_loop(db, state: dict, gpu_lock: threading.Lock, stop: Optional[threading.Event] = None) -> None:
    """Runs the fits asked for, one at a time, each holding `gpu_lock` (once a prediction on the
    GPU, if one is, ends) and giving back the GPU's memory and the RAM it freed when done
    (`_free_gpu`, `_free_ram`); until `stop` is set, if given. ``state["fit"]``: the one running."""
    while not (stop and stop.is_set()):
        fit = fits.claim(db)
        if fit is None:
            time.sleep(POLL_S)
            continue
        state["fit"] = fit["id"]
        log.info("fit %s: started (asked for by %s)", fit["id"], fit["name"])
        try:
            with gpu_lock:
                try:
                    run(db, fit)
                finally:
                    _free_gpu()
                    _free_ram()
        finally:
            state["fit"] = None


def predict_loop(db, state: dict, gpu_lock: threading.Lock, has_gpu: bool, stop: Optional[threading.Event] = None) -> None:
    """Runs the predictions asked for, one at a time: on the GPU (holding `gpu_lock`) if there is
    one and no fit holds it, else on the CPU; until `stop` is set, if given. ``state["prediction"]``:
    the one running."""
    while not (stop and stop.is_set()):
        try:
            p = predictions.claim(db)
        except Exception:  # noqa: BLE001
            log.exception("could not look for predictions to make")
            p = None
        if p is None:
            time.sleep(POLL_S)
            continue
        state["prediction"] = p["id"]
        on_gpu = has_gpu and gpu_lock.acquire(blocking=False)
        log.info("prediction %s: started, on the %s (asked for by %s)", p["id"], "GPU" if on_gpu else "CPU", p["name"])
        try:
            run_prediction(db, p, on_gpu, None if on_gpu or not has_gpu else "a fit is using the GPU")
        except Exception:  # noqa: BLE001 (said in its record; the thread goes on)
            log.exception("prediction %s", p["id"])
        finally:
            if on_gpu:
                _free_gpu()
                gpu_lock.release()
            _free_ram()
            state["prediction"] = None

if __name__ == "__main__":
    main()
