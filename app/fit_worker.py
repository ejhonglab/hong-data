"""The fit worker: runs the fits asked for in the AL segmentation tab (`fits`), one at a time,
with al-diag-seg, on the GPU if there is one: ``python -m app.fit_worker``. In Docker it is
the ``worker`` service (the image's ``worker`` target: this app, al-diag-seg, PyTorch and
Cellpose-SAM's weights), the only one given the GPU.

Beside them, in a thread of its own, it predicts the ROIs of flies with fits' models, as asked
in the tab (`predictions`), one at a time: on the GPU when no fit is using it (a fit waits for
one to end), else on the CPU, so that a prediction does not wait hours for a fit. Refinements of
predictions' ROIs (`refinement`) and extractions of signals (`extract`) run in threads of their
own, on the CPU. Each thread's printing goes to its own log (`_Streams`).

It is held to the folders the server may use as the web server is (`sandbox.start`), and
may also write to the GPU's device files (and its own threads' /proc entries, as CUDA does); the
files it reads and writes in them are noted as
the web server's are (`filelog`), for the Admin tab. It says it is there every `fits.HEARTBEAT_S`
seconds (`fits.heartbeat`), with which GPU and al-diag-seg it has, for the tab to show; and
as it starts, marks fits a worker before it left running interrupted. If it can no longer use
the GPU it started with (`gpu_lost`), it puts what it was about to run back in the queue and
exits, for Docker to start it again, with the GPU.
"""
from __future__ import annotations

import contextlib
import csv
import ctypes
import dataclasses
import datetime as dt
import gc
import glob
import json
import logging
import os
import shutil
import signal
import socket
import stat
import sys
import threading
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Optional

from . import batches, config, extract, filelog, fits, nmf_check, odormeta, predictions, refinement, roi_training, sandbox, version
from .db import Database
from .glomeruli import load_glomeruli

log = logging.getLogger("fit_worker")
POLL_S = 3
SAVE_S = 2  # progress is saved (and a request to stop looked for) at most this often


class Cancelled(Exception):
    pass


class GpuLost(Exception):
    """The worker can no longer use the GPU it started with (`gpu_lost` says why)."""


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


def gpu_lost(devices: Optional[list] = None) -> Optional[str]:
    """Why the worker can no longer use its GPU (None if it can): one of the GPU's device files
    (`devices`, by default /dev/nvidia*) can no longer be opened. The container's device rules,
    which the NVIDIA Container Toolkit sets, are reset under it when systemd reloads (Docker's
    systemd cgroup driver): fit 5 then trained on the CPU, as Cellpose falls back to it, and
    failed an hour later. Only a new container (the worker restarted) has the GPU again."""
    for p in sorted(glob.glob("/dev/nvidia*")) if devices is None else devices:
        try:
            if stat.S_ISCHR(os.stat(p).st_mode):
                os.close(os.open(p, os.O_RDWR))
        except OSError as e:
            return f"{p}: {e.strerror}"
    return None


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
    fit (`Cancelled`) when asked to; or, as training starts, if the worker has a GPU (`gpu`) it
    can no longer use (`GpuLost`), rather than train on the CPU."""

    def __init__(self, db, fit_id: int, gpu: bool = False):
        self.db, self.id, self.gpu = db, fit_id, gpu
        self.stage, self.started, self.seconds, self.last = None, {}, {}, 0.0

    def __call__(self, stage: str, done: int, total: int) -> None:
        t = time.time()
        new = stage != self.stage
        if new and stage == "train" and self.gpu and (why := gpu_lost()):
            raise GpuLost(why)
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
# how far ROIs keep from each other and how much of a neighbour they carry (al-diag-seg's `refine.purity_report`,
# per in-plane mask: medians, and the fraction touching another), of the final ROIs and (``manual_<score>``) of the
# fly's own; and of those matching one of its own, the fraction of ROI-odor pairs responding (``breadth``; the
# manual ones' ``breadth_manual``). None for fits scored before al-diag-seg had them.
PURITY = ("gap_um", "touching", "edge_share", "whole_share")
MANUAL = ("leak", *PURITY)  # the scores the fly's own ROIs have too


def _scores(row: dict, manual: dict) -> dict:
    """A fly's scores of `SCORES`, `PURITY` and breadth, from its rows of al-diag-seg's summary."""
    num = lambda v: v if isinstance(v, (int, float)) else None  # noqa: E731
    return {**{k: num(row.get(k)) for k in (*SCORES, *PURITY, "breadth", "breadth_manual")},
            **{f"manual_{k}": num(manual.get(k)) for k in MANUAL}}


def summary(res: dict) -> dict:
    """What the tab shows of a fit's scores (al-diag-seg's ``Model.evaluate``): per held-out fly,
    its final ROIs' recall, precision and Dice, matched to its own as 3D objects and (``*_2d``,
    where al-diag-seg scores them so) plane by plane, their leak and purity (and its manual ROIs': `_scores`), and how
    many were named as the manual ROI they overlap; their means; per glomerulus; and how they were
    scored (``scoring``: "partly drawn", as glomeruli are often drawn on some of their planes only,
    or "strict"; None before al-diag-seg said: strict)."""
    final = res["final"]
    rows = {r["fly"]: r for r in res["summary"] if r["rois"] == final}
    manual = {r["fly"]: r for r in res["summary"] if r["rois"] == "manual"}
    named = {r["fly"]: r for r in res["identity"] if r["rois"] == final}
    flies = [{"fly": f, "n": r["n"], **_scores(r, manual.get(f, {})),
              "named": named.get(f, {}).get("correct", 0), "named_of": named.get(f, {}).get("total", 0)} for f, r in rows.items()]
    mean = {}
    for k in _scores({}, {}):
        vs = [f[k] for f in flies if f[k] is not None]
        mean[k] = sum(vs) / len(vs) if vs else None
    return {"final": final, "flies": flies, "mean": mean, "named": sum(f["named"] for f in flies),
            "named_of": sum(f["named_of"] for f in flies), "glomeruli": res["glomeruli"], "scoring": res.get("scoring")}


README = """al-diag-seg model: fit #{id} in the odor response database's AL segmentation tab, asked for by
{name} on {created}. Fit to {n_train} flies; scored on {n_val} held out ({held_out}).
Its odors: {odors}.
{images}A fly need not have all of them (at least 3): its ROIs are then joined and named by those it has, as by a model fit to
the same flies with those odors alone; where it lacks some of the odors the images are made from, its ROIs are found
from images of fewer odors than Cellpose was trained on, and fewer are found (al-diag-seg's missing_diagnostics.md).

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


def run(db, fit: dict, gpu: bool = False) -> None:
    """Runs one fit (claimed: `fits.claim`) to the end: done, failed or cancelled. If the worker
    has a GPU (`gpu`) it can no longer use, as training starts or once the fit failed, puts the
    fit back in the queue and raises `GpuLost`."""
    d = fits.fit_dir(fit["id"])
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    o, prog = fits.fit_options(fit), Progress(db, fit["id"], gpu)
    status, error, result, lost = "failed", None, None, None
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
            # only the flies with the odors asked for, if any were; the odors all the tab's flies share are what
            # Cellpose's images are made from, unless asked otherwise (fits.DEFAULTS says why)
            shared, listed = fits.shared_odors(m["flies"]), len(m["flies"])
            m["flies"] = fits.with_odors(m["flies"], o["require_odors"])
            if len(m["flies"]) < 2:
                raise ValueError(f"{len(m['flies'])} of the {listed} flies have {' and '.join(o['require_odors'])}: at least 2 are needed")
            (d / "manifest.json").write_text(json.dumps(m, indent=1))
            names = fits.fly_names(m["flies"])
            held = fits.choose_held_out(m["flies"], o["held_out"])
            flies = {"train": [n for n in names if n not in held], "held_out": held,
                     "stacks": sum(1 for f in m["flies"] if f["stacks"]),
                     "datasets": sorted({x["id"] for f in m["flies"] for x in f["datasets"]}),
                     "listed": listed, "odors": fits.shared_odors(m["flies"]),
                     "image_odors": shared if o["image_odors"] == "shared" else None}
            fits.update(db, fit["id"], flies=flies)
            print(f"{len(names)} flies ({len(found['skipped'])} left out, as the tab says why"
                  + (f"; {listed - len(names)} more for lacking {' or '.join(o['require_odors'])}" if listed > len(names) else "")
                  + f"); held out: {', '.join(held)}")
            print(f"their {len(flies['odors'])} odors"
                  + (f"; Cellpose's images from the {len(shared)} that all {listed} flies share" if flies["image_odors"] and listed > len(names) else ""))
            from alseg import model as am
            from alseg import olfdb as ao

            data = ao.load_export(d / "manifest.json", tentative=o["tentative"], progress=prog)
            (d / "work").mkdir()  # (Cellpose makes only its models/ in the folder it is given)
            model = am.fit(data["flies"], held, am.Options(epochs=o["epochs"], atlas=o["atlas"], tentative=o["tentative"],
                                                           refs=[str(p) for p in refs], image_odors=flies["image_odors"],
                                                           # (a fit asked for before these were options: al-diag-seg's)
                                                           **{k: o[k] for k in fits.ROI_DEFAULTS if k in o}),
                           out=d / "work", progress=prog, odors=data["odors"], aliases=data["aliases"],
                           glomeruli=data["glomeruli"])
            model.info["olfdb"] = {"fit": fit["id"], "by": fit["name"], "datasets": flies["datasets"]}
            model.save(d / "model")
            v = versions()
            (d / "model" / "README.txt").write_text(README.format(
                id=fit["id"], name=fit["name"], created=fit["created_at"], n_train=len(flies["train"]), n_val=len(held),
                held_out=", ".join(held), odors=", ".join(model.odors),
                images="" if len(model.image_odors()) == len(model.odors) else
                f"Cellpose's images of a fly's responses are made from {len(model.image_odors())} of them, those every fly the database had "
                f"then shares (all but {', '.join(x for x in model.odors if x not in model.image_odors())}).\n",
                hong2p=f"@{v['hong2p']}" if v["hong2p"] else "",
                alseg=f"@{v['alseg']['commit']}" if v["alseg"] and v["alseg"]["commit"] else ""))
            prog("evaluate", 0, len(held))
            res = model.evaluate([f for f in data["flies"] if f["name"] in held], d / "results", progress=prog)
            prog.finish()
            result, status = summary(res), "done"
        except Cancelled:
            status = "cancelled"
            print("stopped, as asked")
        except GpuLost as e:
            lost = e
        except Exception as e:  # noqa: BLE001 (any failure is the fit's, said in its log and record)
            traceback.print_exc()
            why = gpu_lost() if gpu else None
            if why:
                lost = GpuLost(why)
            else:
                error = f"{type(e).__name__}: {e}"
        if lost:
            print(f"stopped: the worker can no longer use the GPU ({lost}); the fit is back in the queue, "
                  "for the worker to run once it has started again")
    prog.finish()
    if lost:
        fits.update(db, fit["id"], status="queued", started_at=None, progress=None, flies=None)
        raise lost
    shutil.rmtree(d / "work", ignore_errors=True)  # (Cellpose's: its weights are model/'s, see fits.py)
    done = fits.get(db, fit["id"]) or fit
    sz = fits.sizes(done)
    model_bytes = sum(p.stat().st_size for p in (d / "model").rglob("*") if p.is_file()) if (d / "model").is_dir() else None
    fits.update(db, fit["id"], status=status, finished_at=fits.now(), error=error, result=result, model_bytes=model_bytes,
                stages={s: {"seconds": prog.seconds.get(s, 0.0), "size": sz[s]} for s in fits.STAGES} if status == "done" else None)
    log.info("fit %s: %s%s", fit["id"], status, f" ({error})" if error else "")

# ---------------------------------------------------------------- predictions

PRED_README = """Glomerulus ROIs of {fly}, predicted by the model of fit #{fit} in the odor response database's
AL segmentation tab, asked for by {name} on {created}, on the {device}.

{fly}_RoiSet.zip    the ROIs to use, named by glomerulus where one fits (numbered where none does), per
                    plane (hyperstack Z), for ImageJ and al_analysis: the model's own (pred), or once they
                    were refined, those refined (split, or clean: {fly}_refinement.json says how)
{fly}_<kind>.tif and {fly}_<kind>_RoiSet.zip
                    each kind of ROIs al-diag-seg makes, as label volumes (Z, Y, X) and RoiSets:
                    pred (Cellpose's, the model's own), refined_tuning (refined by odor tuning),
                    refined_prior (refined with the atlas, if the model has one), clean_<step> (each
                    cleaning step alone, as by default, on the model's own); and once refined, split (the
                    model's own with outlines over two glomeruli cut in two, joined into 3D and named
                    again) and clean (those after the other steps asked for)
{fly}_refinement.json
                    (once refined) how, by whom and when; each kind's scores; and per refined ROI, the planes
                    where it is a part of an outline cut in two (and of which), and where trim cut its edge
{fly}_cellprob.tif  Cellpose's cell probability (logits) at each voxel, which refining by core goes by
{fly}_identities*.csv
                    the name given each ROI (by its label in the .tif), with its cost and confidence{manual_ids}
{fly}_identity_terms*.json
                    what each name cost: for each ROI (by its label) and its 5 best glomeruli, the value of each
                    term of the cost (odor tuning, position, shape, atlas overlap, layout: 0 for a perfect fit),
                    their weights, and the most a name may cost (max_cost: an ROI none costs less for is numbered)
{fly}_atlas_prior.npz
                    (if the model has an atlas) the atlas on the fly's planes: each glomerulus' probability at each
                    voxel (prior: glomeruli, Z, Y, X; names: the glomeruli), which the ROIs were named by
average.tif         the fly's average image, as the model saw it
{fly}_nmf.json, {fly}_nmf.tif, {fly}_nmf_candidates.tif and {fly}_nmf_RoiSet.zip
                    (only once glomeruli its ROIs miss were looked for, in the tab's viewer) what that found in the
                    fly's movies: candidates (glomeruli with no ROI, more of an ROI on a plane it has no outline on,
                    parts of ROIs that hold two), to merge into the edited ROIs there or not. The .json says how they
                    were found, each one's outlines and name, and every ROI's name with all of them merged in; the
                    .tif files are the ROIs with all of them in (labels) and the candidate each voxel is of; the
                    RoiSet the candidates alone
{fly}_pred-edited_RoiSet.zip and {fly}_pred-edited.json
                    (only once they were edited in the tab's viewer) {fly}_RoiSet.zip's ROIs as edited there: their
                    outlines simplified, and points of them moved, added or deleted; and when, by whom, by how many
                    pixels simplified. {fly}_RoiSet.zip stays as predicted
missing_odors.csv   each odor of the model: whether the fly has it, and whether Cellpose's images are made from it
                    (a fly without some is joined and named by those it has)
{scores}provenance.json    the model (its fit, its flies, the al-diag-seg that made it), the fly's files read (with
                    their sizes and when they were modified), the software that predicted, and who asked
log.txt             what it printed

To make these again elsewhere, download fit #{fit}'s model, and run alseg-predict on the fly, as its README says.
"""


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
            predictions.update(db, p["id"], progress={"stage": "load", "device": "GPU" if on_gpu else "CPU", "stage_started": t0})
            fit = fits.get(db, p["fit_id"])
            mdir = fits.fit_dir(p["fit_id"]) / "model"
            if fit is None or not (mdir / "model.json").is_file():
                raise FileNotFoundError(f"fit #{p['fit_id']}'s model is not there (the fit was deleted)")
            import numpy as np
            from alseg import model as am
            from alseg.fly import load_fly

            model = am.Model.load(mdir)
            model.options.gpu = on_gpu
            # (the model's own ROIs: refined after, as its options ask, as a refinement asked for later is: refinement.run)
            model.options.split, model.options.clean = False, []
            if model.options.final == "clean":
                model.options.final = "pred"
            po = p.get("options") or {}
            fly_dir, raw_dir = Path(p["fly_dir"]), Path(p["raw_dir"]) if p["raw_dir"] else None
            inputs = _fly_inputs(fly_dir, raw_dir, model, odormeta.OdorIndex(db.list_odors()))
            f = load_fly(fly_dir, model, raw_dir=raw_dir, name=p["fly"], glomeruli=load_glomeruli(),
                         tentative=model.options.tentative)
            missing = list(f.get("missing") or [])
            if missing:  # (joined and named by those it has; found from images of fewer odors, if it lacks some of those)
                short = [x for x in model.image_odors() if x in missing]
                print(f"it lacks {len(missing)} of the model's {len(model.odors)} odors ({', '.join(missing)})"
                      + (f": {len(short)} of them among the {len(model.image_odors())} that Cellpose's images are made from" if short else
                         ": none that Cellpose's images are made from"))
            print(f"{sum(r['used'] for r in inputs['recordings'])} recordings of its odors, planes (z, y, x) "
                  f"{f['average'].shape} of {np.round(f['spacing'], 3).tolist()} um"
                  + (f", its own ROIs from {inputs['roiset']}" if f.get("labels") is not None else ", no ROIs of its own")
                  + (", its anatomical stack" if f.get("hires") else ""))
            # (what its time depends on, for when it should end: predictions.estimate)
            t1, stack = time.time(), bool(inputs["anatomical_stack"])
            predictions.update(db, p["id"], progress={"stage": "predict", "device": "GPU" if on_gpu else "CPU", "stage_started": t1,
                                                      "stack": stack, "scored": f.get("labels") is not None})
            name, scores = p["fly"], None
            if f.get("labels") is not None:
                res = model.evaluate([f], d, priors=True)
                final = res["final"]
                row = lambda kind: next(r for r in res["summary"] if r["rois"] == kind)  # noqa: E731
                num = lambda v: v if isinstance(v, (int, float)) else None  # noqa: E731
                ident = next((r for r in res["identity"] if r["rois"] == final), {})
                scores = {"manual": row("manual")["n"], "manual_2d": row("manual").get("n_2d"),
                          **_scores(row(final), row("manual")),
                          "named": ident.get("correct", 0), "named_of": ident.get("total", 0), "scoring": res.get("scoring")}
                n = row(final)["n"]
                with open(d / f"{name}_identities_{final}.csv", newline="") as fh:
                    named = sorted({r["assigned"] for r in csv.DictReader(fh) if r["assigned"]})
                cellprob = f.get("cellprob")
            else:
                pr = model.predict(f)
                final, cellprob = pr["final"], pr.get("cellprob")
                for kind, lab in pr["labels"].items():
                    refinement.write_kind(d, name, kind, lab, pr["identities"][kind], pr["identity_terms"][kind])
                if pr["atlas_maxprob"] is not None:
                    am.write_volume(d / f"{name}_atlas_maxprob.tif", pr["atlas_maxprob"])
                    am.write_prior(d / f"{name}_atlas_prior.npz", pr["prior"], model.atlas.names)
                ids = pr["identities"][final]
                with open(d / f"{name}_identities.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["label", "assigned", "cost", "confidence"])
                    w.writerows([l, nm or "", f"{cost:.3f}", f"{conf:.3f}"] for l, (nm, cost, conf) in sorted(ids.items()))
                am.write_identity_terms(d / f"{name}_identity_terms.json", pr["identity_terms"][final])
                n = len(np.unique(pr["labels"][final])) - 1
                named = sorted({v[0] for v in ids.values() if v[0]})
            shutil.copyfile(d / f"{name}_{final}_RoiSet.zip", d / f"{name}_RoiSet.zip")
            predictions.write_missing(d, list(model.odors), list(f["odors"]), model.image_odors())
            am.write_volume(d / predictions.AVERAGE, f["average"].astype(np.float32))
            if cellprob is not None:  # (what a refinement's core goes by)
                am.write_volume(d / f"{name}_{refinement.CELLPROB}.tif", np.asarray(cellprob, np.float32))
            result = {"final": final, "n": int(n), "named": named, "scores": scores, "device": "GPU" if on_gpu else "CPU",
                      # the model's odors it lacks, and those of them Cellpose's images are made from
                      "missing_odors": missing, "missing_image_odors": [x for x in model.image_odors() if x in missing],
                      "model_odors": len(model.odors)}
            if refinement.asked(po):  # (refined as one asked for later is, from what it has kept)
                print("its ROIs refined: " + _steps_said(po))
                refinement.start(db, p, po)
                try:
                    out = refinement.run(db, {**p, "result": result, "refine": {"name": p["name"], "ip": p["ip"], "asked_at": p["created_at"]}},
                                         model, f, po, print)
                    refinement.update(db, p["id"], "done", finished_at=predictions.now(), result={**out, "seconds": time.time() - t1})
                    rr = predictions.with_refined(result, out)  # (what its provenance and log say: the refined ROIs)
                    final, n, named, scores = rr["final"], rr["n"], rr["named"], rr["scores"]
                except Exception as e:  # noqa: BLE001 (the refinement's failure, said in its record: the model's own ROIs are the prediction's)
                    traceback.print_exc()
                    refinement.update(db, p["id"], "failed", finished_at=predictions.now(), error=f"{type(e).__name__}: {e}")
                    refinement.undo(db, {**p, "result": result, "refine": None})
            seconds = time.time() - t0
            result.update(seconds=seconds, stages={"load": t1 - t0, "predict": t0 + seconds - t1}, stack=stack)
            v = versions()
            prov = {
                "prediction": p["id"], "asked_for": {"by": p["name"], "from": p["ip"], "at": p["created_at"]},
                "started": p["started_at"], "finished": predictions.now(), "seconds": seconds, "stages": result["stages"], "device": device,
                "fly": {"name": name, "dir": p["fly_dir"], "raw_dir": p["raw_dir"], "spacing_um": list(f["spacing"]),
                        "planes": list(f["average"].shape), "odors": list(f["odors"]), "missing_odors": missing, **inputs},
                "model": {"fit": p["fit_id"], "folder": str(mdir), "fit_by": fit["name"], "fit_asked_for": fit["created_at"],
                          "fit_finished": fit["finished_at"], "options": fit["options"], "flies": fit["flies"],
                          # (al-diag-seg's, as the model ran: those it was saved without, e.g. how
                          # Cellpose's masks are made, are this al-diag-seg's defaults)
                          "alseg_options": dataclasses.asdict(model.options),
                          "odors": model.odors, "info": model.info},
                "software": {"al-diag-seg": v["alseg"], "hong2p": v["hong2p"], "olfdb": version.VERSION, **gpu()},
                "outputs": {"rois": f"{name}_RoiSet.zip", "final": final, "n_rois": int(n), "named": named, "scores": scores},
            }
            (d / predictions.PROVENANCE).write_text(json.dumps(prov, indent=1, default=str))
            (d / "README.txt").write_text(PRED_README.format(
                fly=name, fit=p["fit_id"], name=p["name"], created=p["created_at"], device=device,
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


def run_nmf(db, p: dict, on_gpu: bool, why_cpu: Optional[str] = None) -> None:
    """Runs one look for glomeruli a prediction's ROIs miss (claimed: `nmf_check.claim`) to the
    end, on the GPU or the CPU: done or failed. Its files go in the prediction's folder
    (`nmf_check` says which), and what it prints after what the prediction did, in its log."""
    d = predictions.pred_dir(p["id"])
    t0, dev = time.time(), "GPU" if on_gpu else "CPU"
    device = f"GPU ({gpu()['gpu']})" if on_gpu else f"CPU{f' ({why_cpu})' if why_cpu else ''}"
    status, error, result = "failed", None, None
    stages, at, plan = {}, [None, t0, 0.0], list(nmf_check.STAGES)

    def progress(stage: str, done=None, total=None) -> None:
        now, new = time.time(), stage != at[0]
        if new:
            if at[0]:
                stages[at[0]] = stages.get(at[0], 0.0) + now - at[1]
            at[0], at[1] = stage, now
        if new or now - at[2] >= SAVE_S:
            at[2] = now
            nmf_check.update(db, p["id"], progress={"stage": stage, "done": done, "total": total, "device": dev, "stage_started": at[1],
                                                    "stages": plan})

    with open(d / "log.txt", "a", buffering=1) as out, _output_to(out):
        try:
            asked = p.get("nmf") or {}
            print(f"\nglomeruli its ROIs miss, looked for in its movies: asked for by {asked.get('name')} ({asked.get('ip')}) at "
                  f"{asked.get('asked_at')}; on the {device}")
            mdir = fits.fit_dir(p["fit_id"]) / "model"
            if fits.get(db, p["fit_id"]) is None or not (mdir / "model.json").is_file():
                raise FileNotFoundError(f"fit #{p['fit_id']}'s model is not there (the fit was deleted)")
            from alseg import model as am
            from alseg.fly import load_fly

            model = am.Model.load(mdir)
            model.options.gpu = on_gpu
            # (the atlas on the fly's planes: kept by the prediction, unless made before it was)
            kept = model.atlas is None or nmf_check.files(p)["prior"].is_file()
            if kept:
                plan.remove("register")
            progress("load")
            fly_dir, raw_dir = Path(p["fly_dir"]), Path(p["raw_dir"]) if p["raw_dir"] else None
            _fly_inputs(fly_dir, raw_dir, model, odormeta.OdorIndex(db.list_odors()))
            f = load_fly(fly_dir, model, raw_dir=raw_dir, name=p["fly"], glomeruli=load_glomeruli(), tentative=model.options.tentative,
                         stack=not kept)
            result = nmf_check.run(db, p, model, f, on_gpu, print, progress)
            progress("done")
            seconds = time.time() - t0
            result.update(device=dev, seconds=seconds, stages={k: v for k, v in stages.items() if k in nmf_check.STAGES})
            print(f"{seconds:.0f} s")
            status = "done"
        except Exception as e:  # noqa: BLE001 (any failure is the look's, said in the prediction's log and its record)
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    nmf_check.update(db, p["id"], status, finished_at=predictions.now(), error=error, result=result, progress=None)
    log.info("prediction %s, glomeruli its ROIs miss: %s%s", p["id"], status, f" ({error})" if error else "")


# --------------------------------------------------------------- refinements

def _steps_said(opts: dict) -> str:
    """The steps of a refinement (`predictions.options`), as its log says them."""
    return ", ".join([*(["masks over two glomeruli split"] if opts.get("split") else []),
                      *(f"{st} {opts['clean_params'].get(st) or ''}".strip() for st in opts.get("clean") or [])])


def run_refinement(db, p: dict) -> None:
    """Refines one prediction's ROIs (claimed: `refinement.claim`), as asked, to the end, on the CPU (it needs no
    GPU): done or failed. Its files go in the prediction's folder (`refinement` says which), and what it prints after
    what the prediction did, in its log."""
    d = predictions.pred_dir(p["id"])
    t0 = time.time()
    status, error, result = "failed", None, None
    stages, at = {}, [None, t0]

    def progress(stage: str) -> None:
        now = time.time()
        if at[0]:
            stages[at[0]] = stages.get(at[0], 0.0) + now - at[1]
        at[0], at[1] = stage, now
        refinement.update(db, p["id"], progress={"stage": stage, "stage_started": now})

    with open(d / "log.txt", "a", buffering=1) as out, _output_to(out):
        try:
            asked = p.get("refine") or {}
            opts = asked.get("options") or predictions.options({})
            print(f"\nits ROIs refined: asked for by {asked.get('name')} ({asked.get('ip')}) at {asked.get('asked_at')}: {_steps_said(opts)}")
            mdir = fits.fit_dir(p["fit_id"]) / "model"
            if fits.get(db, p["fit_id"]) is None or not (mdir / "model.json").is_file():
                raise FileNotFoundError(f"fit #{p['fit_id']}'s model is not there (the fit was deleted)")
            from alseg import model as am
            from alseg.fly import load_fly

            model = am.Model.load(mdir)
            model.options.gpu = False
            progress("load")
            # (its anatomical stack only if the atlas is to be registered again: the prediction did not keep it)
            stack = model.atlas is not None and not refinement.files(p)["prior"].is_file()
            fly_dir, raw_dir = Path(p["fly_dir"]), Path(p["raw_dir"]) if p["raw_dir"] else None
            _fly_inputs(fly_dir, raw_dir, model, odormeta.OdorIndex(db.list_odors()))
            f = load_fly(fly_dir, model, raw_dir=raw_dir, name=p["fly"], glomeruli=load_glomeruli(), tentative=model.options.tentative,
                         stack=stack)
            result = refinement.run(db, p, model, f, opts, print, progress)
            result.pop("named", None)
            progress("done")
            result.update(seconds=time.time() - t0, stages={k: v for k, v in stages.items() if k in refinement.STAGES})
            status = "done"
        except Exception as e:  # noqa: BLE001 (any failure is the refinement's, said in the prediction's log and its record)
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            try:  # (its ROIs the model's own again)
                refinement.undo(db, {**p, "refine": None})
            except Exception:  # noqa: BLE001
                traceback.print_exc()
    refinement.update(db, p["id"], status, finished_at=predictions.now(), error=error, result=result, progress=None)
    log.info("prediction %s, refinement: %s%s", p["id"], status, f" ({error})" if error else "")


def refine_loop(db, state: dict, stop: Optional[threading.Event] = None) -> None:
    """Refines the predictions' ROIs asked for, one at a time, beside fits and predictions (on the CPU: they need
    no GPU), until `stop` is set, if given. ``state["refinement"]``: the prediction whose ROIs are being refined."""
    while not (stop and stop.is_set()):
        try:
            p = refinement.claim(db)
        except Exception:  # noqa: BLE001
            log.exception("could not look for refinements to make")
            p = None
        if p is None:
            time.sleep(POLL_S)
            continue
        state["refinement"] = p["id"]
        log.info("prediction %s, refinement: started (asked for by %s)", p["id"], (p["refine"] or {}).get("name"))
        try:
            run_refinement(db, p)
        except Exception:  # noqa: BLE001 (said in its record; the thread goes on)
            log.exception("prediction %s, refinement", p["id"])
        finally:
            _free_ram()
            state["refinement"] = None


# ---------------------------------------------------------------- extractions

def run_extraction(db, e: dict) -> None:
    """Runs one extraction of a fly's ROIs' signals (claimed: `extract.claim`) to the end: done or
    failed. Its folder is made anew (`extract.README` says what is in it)."""
    d = extract.out_dir(e["id"])
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    status, error, result, last = "failed", None, None, [0.0]

    def progress(done: float, total: float, recording: str) -> None:
        if time.time() - last[0] >= SAVE_S:
            last[0] = time.time()
            extract.update(db, e["id"], progress={"recording": recording, "done": done, "total": total})

    with open(d / "log.txt", "a", buffering=1) as out, _output_to(out):
        try:
            print(f"extraction {e['id']}: fly {e['fly']} ({e['fly_dir']}), ROIs {json.dumps(e['rois'])}, options {json.dumps(e['options'])}, "
                  f"asked for by {e['name']} ({e['ip']}) at {e['created_at']}")
            result = extract.run(db, e, print, progress)
            status = "done"
        except Exception as err:  # noqa: BLE001 (any failure is the extraction's, said in its log and record)
            error = str(err) if isinstance(err, extract.ExtractError) else f"{type(err).__name__}: {err}"
            traceback.print_exc()
    extract.update(db, e["id"], status=status, finished_at=extract.now(), error=error, result=result, progress=None)
    log.info("extraction %s: %s%s", e["id"], status, f" ({error})" if error else "")


def extract_loop(db, state: dict, stop: Optional[threading.Event] = None) -> None:
    """Runs the extractions asked for, one at a time, beside fits and predictions (they need no
    GPU, and little but reading a fly's movies), until `stop` is set, if given.
    ``state["extraction"]``: the one running. Each done becomes a temporary dataset (one fly's:
    `batches.make_extraction_datasets`; a batch's, all joined, once all are over: `batches.finish`),
    as do those done before that was so, as it starts."""
    try:
        for ds in batches.make_extraction_datasets(db):
            log.info("dataset %s made of an extraction's signals (done before)", ds)
        for ds in batches.finish(db):  # (a batch whose last extraction ended just before the worker stopped)
            log.info("dataset %s made of a batch's signals (all over before)", ds)
    except Exception:  # noqa: BLE001
        log.exception("could not make datasets of the extractions done before")
    try:
        for ds in batches.retitle(db):
            log.info("temporary dataset %s retitled (by what it was made of, not its panels)", ds)
    except Exception:  # noqa: BLE001
        log.exception("could not retitle temporary datasets")
    while not (stop and stop.is_set()):
        try:
            e = extract.claim(db)
        except Exception:  # noqa: BLE001
            log.exception("could not look for extractions to make")
            e = None
        if e is None:
            time.sleep(POLL_S)
            continue
        state["extraction"] = e["id"]
        log.info("extraction %s: started (asked for by %s)", e["id"], e["name"])
        try:
            run_extraction(db, e)
        except Exception:  # noqa: BLE001 (said in its record; the thread goes on)
            log.exception("extraction %s", e["id"])
        try:  # (one fly's signals: a dataset of their own; a batch's all extracted: joined into a dataset)
            for ds in batches.make_extraction_datasets(db, [e["id"]]):
                log.info("dataset %s made of extraction %s's signals", ds, e["id"])
            for ds in batches.finish(db):
                log.info("dataset %s made of a batch's signals", ds)
        except Exception:  # noqa: BLE001
            log.exception("could not join a batch's signals")
        finally:
            _free_ram()
            state["extraction"] = None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # (after logging's own handler, which keeps writing to the process's stderr)
    sys.stdout, sys.stderr = _Streams(sys.stdout), _Streams(sys.stderr)
    # As the container's PID 1, it would ignore SIGTERM, so `docker stop` waited out its timeout
    # and killed it, which left its network endpoint behind (2026-11-13): the next container of
    # that name then could not start. A fit or prediction it stops is marked interrupted on start.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    db = Database(config.DB_PATH)
    filelog.start(db)  # first, to see what it reads as it starts
    # CUDA opens the GPU's device files, and its threads' /proc entries, to write to them
    status = sandbox.start(files=[*sorted(glob.glob("/dev/nvidia*")), "/proc/self/task"])
    for i in fits.interrupt_running(db):
        log.warning("fit %s was left running by a worker before this one: marked interrupted", i)
    if freed := fits.drop_work():
        log.info("fits' work/ folders removed: %.1f GB freed", freed / 1e9)
    for i in predictions.interrupt_running(db):
        log.warning("prediction %s was left running by a worker before this one: marked interrupted", i)
    for i in nmf_check.interrupt_running(db):
        log.warning("prediction %s's look for missed glomeruli was left running by a worker before this one: marked interrupted", i)
    for i in refinement.interrupt_running(db):
        log.warning("prediction %s's refinement was left running by a worker before this one: marked interrupted", i)
    for i in extract.interrupt_running(db):
        log.warning("extraction %s was left running by a worker before this one: marked interrupted", i)
    info = {"host": host_name(), "pid": os.getpid(), "started": fits.now(), **gpu(), **versions(),
            "landlock": status["kernel"]["enforced"]}
    log.info("fit worker on %s: GPU %s, al-diag-seg %s", info["host"], info["gpu"], info["alseg"])
    state = {"fit": None, "prediction": None, "extraction": None, "refinement": None}
    gpu_lock = threading.Lock()  # held by a fit, or a prediction on the GPU

    def beat():
        n = 0
        while True:
            try:
                fits.heartbeat(db, {**info, "fit": state["fit"], "prediction": state["prediction"], "extraction": state["extraction"],
                                    "refinement": state["refinement"]})
                n += 1
                if n * fits.HEARTBEAT_S % filelog.SAVE_S < fits.HEARTBEAT_S:
                    filelog.flush(db)
            except Exception:  # noqa: BLE001
                log.exception("could not say the worker is there, or save the files it read and wrote")
            time.sleep(fits.HEARTBEAT_S)

    threading.Thread(target=beat, daemon=True).start()
    stop = threading.Event()  # (set once the GPU can no longer be used)
    predictor = threading.Thread(target=predict_loop, args=(db, state, gpu_lock, bool(info["gpu"]), stop), daemon=True)
    predictor.start()
    extractor = threading.Thread(target=extract_loop, args=(db, state, stop), daemon=True)
    extractor.start()
    refiner = threading.Thread(target=refine_loop, args=(db, state, stop), daemon=True)
    refiner.start()
    threading.Thread(target=call_loop, args=(db,), daemon=True).start()
    fit_loop(db, state, gpu_lock, bool(info["gpu"]), stop)
    predictor.join()  # (a prediction on the CPU ends first)
    extractor.join()  # (and an extraction under way)
    refiner.join()  # (and a refinement)
    log.error("exiting, for Docker to start the worker again, with the GPU")
    sys.exit(1)


def _gpu_lost(why: str, stop: Optional[threading.Event]) -> None:
    log.error("the GPU can no longer be used (%s): what was to run is back in the queue", why)
    if stop:
        stop.set()


def fit_loop(db, state: dict, gpu_lock: threading.Lock, has_gpu: bool = False, stop: Optional[threading.Event] = None) -> None:
    """Runs the fits asked for, one at a time, each holding `gpu_lock` (once a prediction on the
    GPU, if one is, ends) and giving back the GPU's memory and the RAM it freed when done
    (`_free_gpu`, `_free_ram`); until `stop` is set, if given. ``state["fit"]``: the one running.
    If there is a GPU (`has_gpu`) it can no longer use (`gpu_lost`), puts the fit back in the
    queue (`run` does, once it started), sets `stop` and returns."""
    while not (stop and stop.is_set()):
        fit = fits.claim(db)
        if fit is None:
            time.sleep(POLL_S)
            continue
        state["fit"] = fit["id"]
        try:
            with gpu_lock:
                if has_gpu and (why := gpu_lost()):
                    fits.update(db, fit["id"], status="queued", started_at=None)
                    _gpu_lost(why, stop)
                    return
                log.info("fit %s: started (asked for by %s)", fit["id"], fit["name"])
                try:
                    run(db, fit, has_gpu)
                except GpuLost as e:
                    _gpu_lost(str(e), stop)
                    return
                finally:
                    _free_gpu()
                    _free_ram()
        finally:
            state["fit"] = None


def call_loop(db) -> None:
    """Does the quick things the web server asks of the worker (`nmf_check.call`: naming a
    prediction's ROIs again), as they come, beside all else."""
    while True:
        try:
            if nmf_check.answer(db):
                continue
        except Exception:  # noqa: BLE001
            log.exception("could not do what the web server asked")
        time.sleep(nmf_check.CALL_POLL_S)


def predict_loop(db, state: dict, gpu_lock: threading.Lock, has_gpu: bool, stop: Optional[threading.Event] = None) -> None:
    """Runs the predictions asked for, and then the looks for glomeruli a prediction's ROIs miss
    (`run_nmf`), one at a time: on the GPU (holding `gpu_lock`) if there is one and no fit holds
    it, else on the CPU; until `stop` is set, if given. ``state["prediction"]``: the prediction
    being made, or looked at. If there is a GPU it can no longer use (`gpu_lost`), puts what it
    took back in the queue, sets `stop` and returns."""
    while not (stop and stop.is_set()):
        look = False
        try:
            p = predictions.claim(db)
            if p is None:
                p, look = nmf_check.claim(db), True
        except Exception:  # noqa: BLE001
            log.exception("could not look for predictions to make")
            p = None
        if p is None:
            time.sleep(POLL_S)
            continue
        on_gpu = has_gpu and gpu_lock.acquire(blocking=False)
        if on_gpu and (why := gpu_lost()):
            if look:
                nmf_check.update(db, p["id"], "queued", started_at=None)
            else:
                predictions.update(db, p["id"], status="queued", started_at=None)
            gpu_lock.release()
            _gpu_lost(why, stop)
            return
        state["prediction"] = p["id"]
        log.info("prediction %s%s: started, on the %s (asked for by %s)", p["id"], ", glomeruli its ROIs miss" if look else "",
                 "GPU" if on_gpu else "CPU", (p["nmf"] or {}).get("name") if look else p["name"])
        try:
            (run_nmf if look else run_prediction)(db, p, on_gpu, None if on_gpu or not has_gpu else "a fit is using the GPU")
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
