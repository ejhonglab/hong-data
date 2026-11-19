"""One ImageJ ROI's mean response to each odor, for the movie player's plot of them (P, and Alt+P
to add an ROI to the plot shown): what al_analysis' ``plot_roi.py`` plots for Sam's ImageJ macros
(``imagej_macros``' P), computed as `extract` computes an extraction's responses.

The ROI is applied to every recording of its fly, as ``plot_roi.py`` applies it: the fly's
recordings registered together (a fly with a `movies.CONCAT`) are all taken (`extract.recordings`);
a movie of a fly without one is taken alone. Each trial's response is `extract.responses`' (its
baseline, statistic and response window: by default as many volumes as the recording's odor pulse
reaches, `extract.window`), each odor's the mean of its trials'.

Only what that needs is read: of each recording's movie, the ROI's plane, the rows its mask
spans and the time points of each trial up to the end of its response window, each time point's
rows in one read, several at once (`_trace`). On a fly of Sam's (6 recordings, 1.3 GB) that is
0.1 to 0.4 s where the movies are in the server's memory, against 2 to 15 s for reading them whole
as an extraction does; read from tensor the first time it is about as slow either way (tensor
reads whole records of its disks, whatever part of one is asked for). So the first ROI of a fly
plotted also has its recordings' movies read whole, in the background (`warm`), for the next ROIs:
another plane's ROI took 1.4 to 3.6 s before that, 0.3 s after.

Odors are in al_analysis' order (hong2p's ``panel2name_order``: `orders`, as ``plot_roi.py`` sorts them with
``sort_odors``): diagnostic panels first, then the others in the order of their recordings; in a
panel hong2p has an order of, in that order, and otherwise in the order of its recordings'
stimulus YAMLs (``pins2odors``), else as first presented; one odor's concentrations together, the
lowest first.
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import extract, movies, sandbox

# hong2p's odor orders of the lab's panels (its ``olf.panel2name_order``, al_analysis' before 5a84e53 moved them
# there), as of hong2p ae2abda: the hong2p pinned for the site predates them (`orders` takes hong2p's, if it has them)
PANEL2NAME_ORDER = {
    "glomeruli_diagnostics": ["3mtp", "va", "t2h", "ms", "a-terp", "geos", "e3hb", "mhex", "ma", "ea", "2-but", "2h", "ga", "HCl",
                              "carene", "farn", "4h-ol", "2but", "2,3-b", "p-cre", "aphe", "1-6ol", "paa", "fench", "CO2", "ecrot",
                              "elac", "acetoin"],
    "megamat": ["2h", "IaA", "pa", "2-but", "eb", "ep", "aa", "va", "B-cit", "Lin", "6al", "t2h", "1-8ol", "1-5ol", "1-6ol", "benz",
                "ms"],
    "validation2": ["+pul", "menth", "long", "sab", "-bCar", "-aPine", "euc", "2-mib", "geos", "guai", "mchav", "PEA", "paa", "PAA",
                    "bbenz", "B-myr", "ger", "1-3ol", "1-prop", "1p3one", "1o3one", "EtOct", "2-mba", "Z4-7al"],
    "kiwi": ["pfo", "EtOH", "IAol", "IaA", "ea", "eb", "ea+eb", "ea+eb (air mix)", "eb+ea (air mix)", "~kiwi", "kmix", "kmix0",
             "kmix-1", "kmix-2"],
    "control": ["pfo", "ms", "va", "fur", "2h", "oct", "1o3ol", "1o3ol+2h", "2h+oct (air mix)", "2h+1o3ol (air mix)",
                "1o3ol+2h (air mix)", "control mix", "cmix", "cmix0", "cmix-1", "cmix-2"],
}
WARM_CHUNK = 8 << 20  # bytes of a movie read at once, warming it
WARM_AGAIN_S = 1800  # a movie warmed is not read again for this long
READERS = 16  # reads of a movie at once (each a time point's rows of the ROI's plane): tensor answers them together


class RoiResponsesError(ValueError):
    pass


_warm_queue: "queue.Queue[Path]" = queue.Queue()
_warmed: dict[str, float] = {}
_warm_lock = threading.Lock()
_warm_thread: Optional[threading.Thread] = None


def warm(paths: list[Path]) -> None:
    """Has movies read whole, one after another, in a thread of their own, for the server (the
    computer's memory) to have them for the next ROI plotted: but not one warmed in the last
    `WARM_AGAIN_S`."""
    global _warm_thread
    now = time.monotonic()
    with _warm_lock:
        for p in paths:
            if now - _warmed.get(str(p), -math.inf) > WARM_AGAIN_S:
                _warmed[str(p)] = now
                _warm_queue.put(p)
        if _warm_thread is None:
            _warm_thread = threading.Thread(target=_warm_loop, name="roi-responses-warm", daemon=True)
            _warm_thread.start()


def _warm_loop() -> None:
    global _warm_thread
    while True:
        try:
            p = _warm_queue.get(timeout=60)
        except queue.Empty:
            with _warm_lock:  # (not if `warm` gave it more meanwhile)
                if _warm_queue.empty():
                    _warm_thread = None
                    return
            continue
        try:
            with open(p, "rb", buffering=0) as f:
                while f.read(WARM_CHUNK):
                    pass
        except OSError:
            pass


def recordings(movie: Path, trials: Optional[Path]) -> tuple[Path, list[dict]]:
    """The recordings an ROI drawn on `movie` (its path as listed, not resolved: a fly's
    `movies.CONCAT` is a link) is applied to, as `extract.recordings` gives them: all of its
    fly's, if the fly has its recordings registered together; else `movie` alone, with its
    `trials`. Returns (the fly's folder, its recordings)."""
    fly = movie.parent if movie.name == movies.CONCAT else movie.parent.parent
    if movie.name == movies.CONCAT or (fly / movies.CONCAT).exists():
        recs = extract.recordings(fly)
        if recs:
            return fly, recs
    if trials is None:
        raise RoiResponsesError("the movie has no trials (trial_frames_and_odors.json), and its fly no recordings with theirs")
    return fly, [{"name": movie.parent.name, "dir": movie.parent, "movie": movie, "trials": trials}]


def raw_dir(fly: Path) -> Optional[Path]:
    """The fly's raw data folder, where hong2p lays one out (`extract.RAW` beside `extract.MOCORR`), if there."""
    if extract.MOCORR not in fly.parts:
        return None
    guess = Path(*(extract.RAW if x == extract.MOCORR else x for x in fly.parts))
    return guess if guess.is_dir() and sandbox.can_read(guess) else None


def read_roi(path: Path, index: int, name: Optional[str]):
    """The ROI at `index` of a RoiSet.zip (or .roi), checked to be named `name` (as the player listed it)."""
    import roifile

    try:
        rois = roifile.ImagejRoi.fromfile(path)
    except Exception as e:  # noqa: BLE001 (roifile raises many kinds)
        raise RoiResponsesError(f"cannot read the ImageJ ROIs ({e})") from None
    rois = rois if isinstance(rois, list) else [rois]
    if not 0 <= index < len(rois):
        raise RoiResponsesError(f"no ROI {index + 1} in {path.name} (it has {len(rois)})")
    r = rois[index]
    if name is not None and (r.name or f"{index + 1}") != name:
        raise RoiResponsesError(f"ROI {index + 1} of {path.name} is not {name} (the file changed?): open the movie again")
    return r


def needed(trials: list[dict], window: int, t: int) -> np.ndarray:
    """The time points a response is computed from: of each trial, from its start (its baseline)
    to the end of its response window (its last time point at most), of the movie's `t`."""
    out = set()
    for tr in trials:
        out.update(range(max(0, tr["start"]), min(tr["end"], tr["onset"] + window - 1, t - 1) + 1))
    return np.array(sorted(out), dtype=np.int64)


def _trace(movie: movies.Movie, roi: dict, frames: np.ndarray) -> np.ndarray:
    """The ROI's trace (its mean over its mask, as `extract.plane_traces` computes it, to the last
    bit) at `frames`: each a read of the rows of its plane its mask spans, where the movie is in
    one piece (a memory map), else from the whole movie read."""
    d, m = movie.data, roi["mask"]
    h, w = m.shape
    if d is None or not d.flags.c_contiguous or not len(frames):
        return extract.plane_traces(movie, [roi])[frames, 0]
    strides = dict(zip(movie.roles, d.strides))
    row, nx = strides["y"], movie.shape["x"]
    start = d.offset + roi["z"] * strides.get("z", 0) + roi["y0"] * row
    fd = os.open(movie.path, os.O_RDONLY)
    try:
        def one(t: int) -> bytes:
            pos, b = start + int(t) * strides.get("t", 0), b""
            while len(b) < h * row:
                more = os.pread(fd, h * row - len(b), pos + len(b))
                if not more:
                    raise RoiResponsesError(f"{movie.path.name} is shorter than its header says")
                b += more
            return b

        with ThreadPoolExecutor(READERS) as pool:
            bands = list(pool.map(one, frames))
    finally:
        os.close(fd)
    box = np.frombuffer(b"".join(bands), dtype=d.dtype).reshape(len(frames), h, nx)[:, :, roi["x0"]:roi["x0"] + w]
    return np.mean(box[:, m], axis=1)


def _yaml_order(path: Optional[str]) -> list[str]:
    """The odors of a stimulus YAML (olf's generated ones: ``pins2odors``), in its order, named as
    `extract.odor_str` names them."""
    if not path:
        return []
    try:
        import yaml

        data = yaml.safe_load(extract._read_text(Path(path)) or "") or {}
        return [extract.odor_str(o) for o in (data.get("pins2odors") or {}).values() if isinstance(o, dict) and o.get("name")]
    except Exception:  # noqa: BLE001 (not a YAML of olf's)
        return []


def _name(odor: str) -> str:
    """An odor's names (of each of a mixture's components), without their concentrations."""
    return " + ".join(p.split(" @ ")[0] for p in odor.split(" + "))


def _conc(odor: str) -> float:
    try:
        return float(odor.split(" + ")[0].split(" @ ")[1])
    except (IndexError, ValueError):
        return math.nan


def order(keys: list[tuple[str, str]], panels: list[str], yaml_odors: list[str]) -> list[tuple[str, str]]:
    """(panel, odor) keys, as `__doc__` says: diagnostic panels first, then `panels` (in the
    order of their recordings); within a panel, by hong2p's order of its odors' names, if it
    has one, else by `yaml_odors` (the recordings' stimulus YAMLs' odors, in order), else as
    `keys` are (as first presented); an odor's concentrations together, lowest first."""
    first: dict[str, int] = {}
    for o in [*map(_name, yaml_odors), *(_name(k[1]) for k in keys)]:
        first.setdefault(o, len(first))
    panel_rank = {p: i for i, p in enumerate(sorted(dict.fromkeys(panels), key=lambda p: p != extract.DIAGNOSTIC))}

    def rank(k: tuple[str, str]):
        fixed = orders().get(k[0]) or []
        name = _name(k[1])
        r = fixed.index(name) if name in fixed else len(fixed) + first[name]
        c = _conc(k[1])
        return panel_rank.get(k[0], len(panel_rank)), r, math.isnan(c), 0 if math.isnan(c) else c

    return sorted(keys, key=rank)


def orders() -> dict[str, list[str]]:
    """Odor names in order, by panel: hong2p's, if it has them, else `PANEL2NAME_ORDER`."""
    from hong2p import olf

    return getattr(olf, "panel2name_order", None) or PANEL2NAME_ORDER


def labels(keys: list[tuple[str, str]]) -> list[str]:
    """What each column is called: its odors' names, with their concentrations where another column has the same names."""
    count: dict[str, int] = {}
    for k in keys:
        count[_name(k[1])] = count.get(_name(k[1]), 0) + 1
    return [_name(o) if count[_name(o)] == 1 else o for _, o in keys]


def compute(movie: Path, trials: Optional[Path], roiset: Path, index: int, name: Optional[str], window_option,
            also: list[Path], check: Callable[[Path], Path]) -> dict:
    """An ROI's mean response to each odor (see `__doc__`): the ROI `index` (named `name`) of
    `roiset`, drawn on `movie` (with its `trials`), with an extraction's response window
    (`window_option`: `extract.WINDOW`, or a number; `also`: `extract.stimulus_folders`). `check`
    gives the path of a recording's file to read, or raises if it may not be read.

    Returns:
        ``roi``, ``columns`` (each odor: ``panel``, ``odor`` (as al_analysis names it), ``label``,
        ``diagnostic``, ``trials``), ``values`` (its mean response; None where no trial had one),
        ``recordings`` (each one's ``name``, ``panel``, ``window`` and ``how`` it was worked out,
        ``trials`` and ``read``: the time points read, of ``of``), ``warnings``.

    Raises:
        RoiResponsesError: if the ROI or the recordings cannot be read, or none has a response.
    """
    fly, recs = recordings(movie, trials)
    raw = raw_dir(fly)
    r = read_roi(roiset, index, name)
    roi_name = r.name or f"{index + 1}"
    per: dict[tuple[str, str], list[float]] = {}
    panels, yaml_odors, out_recs, warnings, read = [], [], [], [], []
    for rec in recs:
        panel = extract.panel(rec["name"])
        try:
            m = movies.open_movie(check(rec["movie"]))
            tr = extract.read_trials(check(rec["trials"]))
        except (movies.MovieError, extract.ExtractError) as e:
            warnings.append(f"{rec['name']}: left out ({e})")
            continue
        s = m.shape
        mask = extract.roi_mask(r, (s["z"], s["y"], s["x"]))
        if isinstance(mask, str):
            warnings.append(f"{rec['name']}: left out (the ROI has {mask} in its movie of {s['z']} planes, {s['x']} × {s['y']})")
            continue
        win = extract.window(raw / rec["name"] if raw else None, window_option, also)
        frames = needed(tr, win["volumes"], s["t"])
        trace = np.full((s["t"], 1), np.nan)
        try:
            trace[frames, 0] = _trace(m, mask, frames)
        except OSError as e:
            warnings.append(f"{rec['name']}: left out (cannot read its movie: {e})")
            continue
        read.append(m)
        resp = extract.responses(trace, tr, win["volumes"])[:, 0]
        for t, v in zip(tr, resp):
            per.setdefault((panel, " + ".join(t["odors"])), []).append(float(v))
        panels.append(panel)
        yaml_odors += _yaml_order(win.get("yaml"))
        out_recs.append({"name": rec["name"], "panel": panel, "window": win["volumes"], "how": win["how"], "trials": len(tr),
                         "read": int(len(frames)), "of": s["t"]})
    warm([m.path for m in read])
    if not per:
        raise RoiResponsesError(f"no response of {roi_name} in any recording" + (f": {'; '.join(warnings)}" if warnings else ""))
    keys = order(list(per), panels, yaml_odors)
    values = []
    for k in keys:
        v = np.array(per[k])
        values.append(float(np.nanmean(v)) if np.isfinite(v).any() else None)
    return {"roi": roi_name, "fly": str(fly), "values": values, "recordings": out_recs, "warnings": warnings,
            "columns": [{"panel": p, "odor": o, "label": lab, "diagnostic": p == extract.DIAGNOSTIC, "trials": len(per[(p, o)])}
                        for (p, o), lab in zip(keys, labels(keys))]}
