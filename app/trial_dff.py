"""Each trial's dF/F image of a movie, and each odor's mean of them, for the movie player: as
al_analysis makes them (``trial_dff.tif`` and ``trialmean_dff.tif`` in a recording's folder,
``trial_dff_concat.tif`` and ``trialmean_dff_concat.tif`` in a fly's), read where al_analysis left
them, else computed here, in the background, from the movie and its trials.

A trial's image is its dF/F at each voxel, (F - F0) / F0, averaged over its first `N_RESPONSE`
volumes from the odor's onset, F0 the mean of its volumes before the onset (al_analysis'
``delta_f_over_f``, with its defaults: no fewer baseline volumes, the mean, not the median); an
odor's, the mean of its trials' (those presented one after another: al_analysis' trials of one
odor are). On Tom's 2023-05-08/1 diagnostics1 both are al_analysis' files to the last bit (the
means within 2.4e-7).

al_analysis' files are used only where they are as new as the movie: one made from an older
motion correction is not of the movie shown (Sam's 2024-06-11/2, whose trial_dff.tif predates
its mocorr.tif, correlates 0.51 with what the movie gives). Computed ones are kept in
``data/trial_dff/`` (by the movie's path, size and time), as ImageJ hyperstacks, (trial or odor,
Z, [C,] Y, X), float32. One is computed at a time, those asked for last first (`want`).
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import config, movies

N_RESPONSE = 2  # al_analysis' n_volumes_for_response
OWN = {False: ("trial_dff.tif", "trialmean_dff.tif"), True: ("trial_dff_concat.tif", "trialmean_dff_concat.tif")}


def cache_dir() -> Path:
    return config.DATA_DIR / "trial_dff"


def _key(movie: Path) -> str:
    st = os.stat(movie)
    return hashlib.sha1(f"{os.path.realpath(movie)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20]


def files(movie: Path) -> dict:
    """Where a movie's per-trial and per-odor dF/F images are, or would be: ``trials`` and ``odors``
    (paths), ``source`` ("al_analysis" or "computed") and ``ready`` (whether both are there)."""
    movie = Path(movie)
    concat = movie.name == movies.CONCAT
    own = [movie.parent / n for n in OWN[concat]]
    try:
        made = os.stat(os.path.realpath(movie)).st_mtime
        if all(p.is_file() and os.stat(p).st_mtime >= made for p in own):
            return {"trials": str(own[0]), "odors": str(own[1]), "source": "al_analysis", "ready": True}
    except OSError:
        pass
    k = _key(movie)
    out = [cache_dir() / f"{k}_trials.tif", cache_dir() / f"{k}_odors.tif"]
    return {"trials": str(out[0]), "odors": str(out[1]), "source": "computed", "ready": all(p.is_file() for p in out)}


def compute(movie: Path, trials: Path, progress=None) -> dict:
    """Computes a movie's per-trial and per-odor dF/F images (see the module's docstring) into the
    cache (`files`), `progress` told (trials done, of how many). Returns `files`'."""
    import tifffile

    m = movies.open_movie(Path(movie))
    tr = movies.read_trials(Path(trials))
    s = m.shape
    tr = [t for t in tr if 0 <= t["start"] < t["onset"] <= t["end"] < s["t"]]
    if not tr:
        raise movies.MovieError("none of its trials has volumes before its odor and with it, in the movie")
    out = np.empty((len(tr), s["z"], s["c"], s["y"], s["x"]), np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, t in enumerate(tr):
            for c in range(s["c"]):
                base = m.volumes(t["start"], t["onset"] - t["start"], c).mean(axis=0, dtype=np.float64)
                resp = m.volumes(t["onset"], min(N_RESPONSE, t["end"] + 1 - t["onset"]), c).mean(axis=0, dtype=np.float64)
                out[k, :, c] = (resp - base) / base
            if progress:
                progress(k + 1, len(tr))
    groups = [k for k, t in enumerate(tr) if k == 0 or tr[k - 1]["odor"] != t["odor"]] + [len(tr)]
    means = np.stack([out[a:b].mean(axis=0) for a, b in zip(groups[:-1], groups[1:])])
    f = files(movie)
    cache_dir().mkdir(parents=True, exist_ok=True)
    for arr, path in ((out, f["trials"]), (means, f["odors"])):
        a = arr if s["c"] > 1 else arr[:, :, 0]
        tmp = Path(path).with_name(f".{Path(path).name}.tmp.tif")
        tifffile.imwrite(tmp, a, imagej=True, metadata={"axes": "TZCYX" if s["c"] > 1 else "TZYX"})
        os.replace(tmp, path)
    return files(movie)


def labels(trials: Path, by: str) -> list[dict]:
    """The labels of a per-trial (``by`` "trials") or per-odor ("odors") image's time points, as
    `movies.read_trials` gives a movie's trials: each its ``start``, ``onset`` and ``end`` (the time
    point it is) and ``odor``."""
    tr = movies.read_trials(Path(trials))
    if by == "odors":
        tr = [t for k, t in enumerate(tr) if k == 0 or tr[k - 1]["odor"] != t["odor"]]
    return [{"start": i, "onset": i, "end": i, "odor": t["odor"]} for i, t in enumerate(tr)]


# ------------------------------------------------------------- in the background

class _Queue:
    """Movies to compute, one at a time, in a thread of the web server: the one asked for last first."""

    def __init__(self):
        self.lock = threading.Condition()
        self.order: list[str] = []  # movie paths waiting, the next first
        self.trials: dict[str, str] = {}
        self.state: dict[str, dict] = {}  # movie path -> {status, done, total, error}
        self.thread: Optional[threading.Thread] = None

    def want(self, items: list[dict], first: Optional[str] = None) -> None:
        with self.lock:
            for it in items:
                p = it["path"]
                if p in self.order or self.state.get(p, {}).get("status") in ("running", "failed"):
                    continue
                self.order.append(p)
                self.trials[p] = it["trials"]
                self.state[p] = {"status": "queued", "done": 0, "total": None}
            if first in self.order:
                self.order.remove(first)
                self.order.insert(0, first)
            if self.order and (self.thread is None or not self.thread.is_alive()):
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            self.lock.notify_all()

    def _run(self) -> None:
        while True:
            with self.lock:
                if not self.order:
                    return
                p = self.order.pop(0)
                self.state[p] = {"status": "running", "done": 0, "total": None, "started": time.time()}

            def tell(done, total, p=p):
                self.state[p].update(done=done, total=total)

            try:
                if not files(Path(p))["ready"]:
                    compute(Path(p), Path(self.trials[p]), tell)
                self.state[p] = {"status": "done"}
            except Exception as e:  # noqa: BLE001 (said to the player)
                self.state[p] = {"status": "failed", "error": str(e) if isinstance(e, (movies.MovieError, OSError)) else f"{type(e).__name__}: {e}"}


QUEUE = _Queue()


def status(items: list[dict], first: Optional[str] = None) -> list[dict]:
    """Of each movie (``path``, ``trials``: its trials file), its images (`files`) and how they are
    coming: ``status`` "ready", "queued", "running" (with ``fraction``) or "failed" (with ``error``);
    those not ready and not under way are asked for (`first` before the others)."""
    out, missing = [], []
    for it in items:
        f = files(Path(it["path"]))
        if f["ready"]:
            out.append({**it, **f, "status": "ready"})
            continue
        st = QUEUE.state.get(it["path"])
        if st is None or st["status"] == "done":
            missing.append(it)
            st = {"status": "queued"}
        frac = (st["done"] / st["total"]) if st.get("total") else 0.0
        out.append({**it, **f, "status": st["status"], "fraction": frac, "error": st.get("error")})
    if missing or first:
        QUEUE.want(missing, first)
    return out
