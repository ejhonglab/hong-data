"""Signals of a fly's ROIs, extracted from its motion-corrected movies as al_analysis does: each
ROI's trace in every recording, and its response to each trial's odor (one number per ROI and
trial). Asked for in the AL segmentation tab (*Extract signals*), with the ROIs a model
predicted (as they are, or as edited in the viewer) or the fly's own, and run by the fit worker
(`fit_worker.extract_loop`), beside fits and predictions.

What al_analysis does (its ``ij_traces`` and ``compute_trial_stats``, with hong2p's
``ijroi_masks``, ``extract_traces_bool_masks`` and ``rois2best_planes_only``), and so what is
done here, with the same numbers out:

- A recording is a folder of the fly's with a movie (the first of `movies.PREFERRED`:
  ``mocorr.tif``) and its trials (`movies.TRIALS`: each trial's first frame, first frame with
  odor, and last frame). Its ROIs are those of the ``RoiSet.zip`` in it (al_analysis links every
  recording's to one), each a polygon on one plane; those named ``AL`` (outlines of the whole
  antennal lobe) and those whose name ends in ``+`` are left out.
- An ROI's mask is the pixels (y, x) whose point (x, y) is inside its polygon, as matplotlib's
  ``Path.contains_points`` says (`inside`): the rule hong2p uses, which is half a pixel down and
  right of ImageJ's own (a pixel's centre, at x + 0.5, y + 0.5).
- Its trace is the mean of the movie over its mask, at each time point.
- Its dF/F in a trial is (F - F0) / F0, F0 the mean of its trace over the trial's frames
  before the odor (all of them: al_analysis' ``n_volumes_for_baseline = None``); its response
  is the dF/F of largest magnitude, with its sign, among the first time points from the
  odor's onset (``sign_preserving_maxabs``): al_analysis' `N_RESPONSE`
  (``n_volumes_for_response = 2``) if the response window is set to that.
- An ROI drawn on several planes (one name, several polygons) is, in each recording, the one
  of them whose largest response over the recording's trials is greatest (`MODES`'
  ``best_plane``). The other `MODES` are other ways to make one signal of its planes.

Faster than al_analysis in that masks are only made over each polygon's bounding box, a
movie is read once, in pieces, and responses are computed for all ROIs at once. An extraction
is a row of the ``extractions`` table and a folder, `out_dir`, which `README` describes.

Where it differs from al_analysis (notes/trace_extraction_notes.md says more of each):

- labels: an odor's abbreviation is hong2p's hardcoded one, else its own in the recording's
  trials (al_analysis also has hong2p's cache of every YAML loaded on its computer); a
  recording's panel, where al_analysis knows none, is its folder's name less a number at its
  end (`panel`); ``is_pair`` is always False;
- recordings: one done again (``<name>_redo``) is in place of the first, whole, where
  al_analysis blanks only the odors given again, whatever the folders are called; every folder
  with a movie and its trials is taken (also of one plane), none of al_analysis' own choices of
  what to analyze applied;
- ROIs: a recording's ``RoiSet.zip``, else the fly's folder's (which al_analysis does not
  read); rectangles, ovals and composite ROIs are read, which its reader refuses;
- nothing after al_analysis' per-recording table (``ij_trial_df``) is done: no merging of ROIs
  across recordings, dropping of uncertain ROIs, blanking of concentrations not final, or means;
  ``responses_glomeruli.csv`` is of the ROIs an upload accepts, not al_analysis' certain ones;
- the response window is by default as many volumes as a recording's odor pulse can reach
  (`window`: from the pulse's length in its stimulus YAML and a volume's in its
  Experiment.xml; `N_RESPONSE` at least, and where they are not found), not al_analysis'
  hardcoded `N_RESPONSE`, which it can be set to (as to any number); a best plane is chosen
  by responses over the same window;
- its other settings are al_analysis' current defaults, fixed (the baseline, the statistic);
  only ImageJ's ROIs, not suite2p's;
- the other `MODES` are not al_analysis' at all.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import config, movies, predictions, sandbox, tensor, thorimage

N_RESPONSE = 2  # time points from the odor's onset al_analysis takes a response from (its n_volumes_for_response): the fewest here
WINDOW = "pulse"  # the response window by default: as many volumes as the odor pulse can reach (`window_volumes`)
MAX_WINDOW = 100  # the most volumes a response window may be set to
STIMULI = "stimulus_data_files"  # where hong2p keeps a data folder's stimulus YAMLs (beside its raw data)
XML = "Experiment.xml"  # ThorImage's, in a raw recording's folder
RAW, MOCORR = "raw_data", "analysis_intermediates"  # hong2p's folders of a data folder: a fly's are at the same path in each
OUTLINE = "AL"  # ROIs so named outline the whole antennal lobe on a plane (hong2p's is_ijroi_plane_outline)
MAXIMAL = "+"  # ROIs whose name ends so may hold several glomeruli: left out (hong2p's drop_maximal_extent_rois)
DIAGNOSTIC = "glomeruli_diagnostics"  # al_analysis' panel of recordings with "diag" in their name
# al_analysis' other panels, by what their recordings' names have in them (its get_panel), in its order
PANELS = ("kiwi", "control", "megamat", "validation2", "ban_2but", "ban_solvent", "2but_solvent", "isoamylacetate_solvent",
          "ban_purestrain_solvent", "mono_screen_solvent", "nat_screen_solvent")
# how one signal is made of an ROI drawn on several planes
MODES = {
    "best_plane": "its best plane, chosen in each recording (as al_analysis does)",
    "best_plane_fly": "its best plane, the same one in every recording of the fly",
    "all_planes": "all of its voxels, on every plane it is drawn on (their mean)",
    "weighted": "its planes averaged, the less noisy ones counting more",
}
ROI_KINDS = ("own", "predicted", "edited")
STATUSES = ("queued", "running", "done", "failed", "interrupted")
ACTIVE = ("queued", "running")
RECENT = 10  # the extractions done that one's time is estimated from
GUESS_S_PER_GB = 20.0  # seconds per GB of movies, until one is done to estimate from
CHUNK_BYTES = 256 << 20  # of a movie's plane read at once


class ExtractError(ValueError):
    pass


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------------------ masks

def inside(points: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Whether each (x, y) is inside the closed polygon of `points` ((n, 2): x, y), exactly as
    matplotlib's ``Path(points).contains_points`` says (its ``point_in_path``: a ray to +x
    crossing each edge whose ends are on either side of y, an end at y counting as above it),
    which is what hong2p makes ROIs' masks with. Points on the outline are in or out by that
    rule, so not alike on every side of a polygon."""
    pts = np.asarray(points, dtype=np.float64)
    tx, ty = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    out = np.zeros(np.broadcast(tx, ty).shape, dtype=bool)
    if len(pts) < 3:
        return out
    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for i in range(len(pts)):
        above0, above1 = y0[i] >= ty, y1[i] >= ty
        hit = ((y1[i] - ty) * (x0[i] - x1[i]) >= (x1[i] - tx) * (y0[i] - y1[i])) == above1
        out ^= (above0 != above1) & hit
    return out


def polygon_mask(points: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    """The mask of a polygon ((n, 2): x, y, in pixels, as ImageJ's) on a plane of `shape` (y, x),
    exactly as hong2p's ``ijroi2mask`` makes it: pixel (y, x) is in it if the point (x, y) is
    `inside`.

    Returns:
        (y0, x0, mask): the mask over the polygon's bounding box alone (empty if it is off the
        plane), and where that box starts.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3 or not np.isfinite(pts).all():
        return 0, 0, np.zeros((0, 0), bool)
    x0, y0 = max(0, int(np.floor(pts[:, 0].min()))), max(0, int(np.floor(pts[:, 1].min())))
    x1, y1 = min(shape[1], int(np.ceil(pts[:, 0].max())) + 1), min(shape[0], int(np.ceil(pts[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, np.zeros((0, 0), bool)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    # (rows as matplotlib's x and columns as its y, as hong2p gives it them: the same polygon, but
    # points exactly on its outline, which ImageJ's quarter-pixel points make common, fall as there)
    return y0, x0, inside(pts[:, ::-1], yy, xx)


def read_rois(path: Path, shape: tuple[int, int, int], say: Callable[[str], None] = print) -> list[dict]:
    """The ROIs of a RoiSet.zip (or one .roi) to take signals from, in its order, as al_analysis
    reads them (hong2p's ``ijroi_masks``, for a movie of `shape`: z, y, x): each a polygon on
    one plane, with its ``name``, ``z`` (its hyperstack Z, else its stack position, from 0; 0
    on a movie of one plane), where its mask is (``y0``, ``x0``, ``mask``: `polygon_mask`) and
    ``pixels`` (how many). Left out, saying so: those named to be (`MAXIMAL`, `OUTLINE`), and
    those al_analysis could not use: of no plane, of one the movie lacks, or with no pixel.

    Raises:
        ExtractError: if it cannot be read, or none of its ROIs is left.
    """
    import roifile

    try:
        rois = roifile.ImagejRoi.fromfile(path)
    except Exception as e:  # noqa: BLE001 (roifile raises many kinds)
        raise ExtractError(f"cannot read the ImageJ ROIs of {path} ({e})") from None
    rois = rois if isinstance(rois, list) else [rois]
    nz, ny, nx = shape
    out, left = [], {"maximal": [], "outline": 0, "no plane": [], "no pixels": []}
    for i, r in enumerate(rois):
        name = r.name or f"{i + 1}"
        if name.endswith(MAXIMAL):
            left["maximal"].append(name)
            continue
        if name == OUTLINE:
            left["outline"] += 1
            continue
        z = r.z_position - 1 if r.z_position > 0 else r.position - 1 if r.position > 0 else None
        if nz == 1 and z is None:
            z = 0
        if z is None or not 0 <= z < nz:
            left["no plane"].append(name)
            continue
        mask = np.zeros((ny, nx), bool)
        box = [ny, nx, 0, 0]
        for p in r.coordinates(multi=True):  # (one outline, but for composite ROIs: in one and not another of theirs)
            y0, x0, m = polygon_mask(np.asarray(p, dtype=np.float64), (ny, nx))
            if m.any():
                mask[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] ^= m
                box = [min(box[0], y0), min(box[1], x0), max(box[2], y0 + m.shape[0]), max(box[3], x0 + m.shape[1])]
        if not mask.any():
            left["no pixels"].append(name)
            continue
        m = mask[box[0]:box[2], box[1]:box[3]]
        out.append({"name": name, "z": int(z), "y0": box[0], "x0": box[1], "mask": m, "pixels": int(m.sum())})
    if left["maximal"]:
        say(f"left out: {len(left['maximal'])} ROIs whose name ends in {MAXIMAL} ({', '.join(left['maximal'])}), as al_analysis leaves them out")
    if left["outline"]:
        say(f"left out: {left['outline']} outlines of the whole antennal lobe ({OUTLINE})")
    if left["no plane"]:
        say(f"left out: {len(left['no plane'])} ROIs on no plane of the movie's {nz} ({', '.join(left['no plane'])})")
    if left["no pixels"]:
        say(f"left out: {len(left['no pixels'])} ROIs with no pixel inside them ({', '.join(left['no pixels'])})")
    if not out:
        raise ExtractError(f"no ROI of {path} to take a signal from")
    return out


# ----------------------------------------------------------------- traces

def plane_traces(movie: movies.Movie, rois: list[dict], progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Each ROI's trace ((time points, ROIs), float64): the mean of the movie over its mask at
    each time point, as hong2p's ``extract_traces_bool_masks`` computes it (the mean of its
    pixels in their order in the plane, in the file's own number type). The movie is read once,
    from its start, in pieces of at most `CHUNK_BYTES`; `progress` is told (time points read, of
    how many)."""
    s = movie.shape
    out = np.empty((s["t"], len(rois)), dtype=np.float64)
    step = max(1, CHUNK_BYTES // (s["z"] * s["y"] * s["x"] * movie.dtype.itemsize))
    for t0 in range(0, s["t"], step):
        data = movie.volumes(t0, step)
        for i, r in enumerate(rois):
            m = r["mask"]
            box = data[:, r["z"], r["y0"]:r["y0"] + m.shape[0], r["x0"]:r["x0"] + m.shape[1]]
            out[t0:t0 + len(data), i] = np.mean(box[:, m], axis=1)
        if progress:
            progress(t0 + len(data), s["t"])
    return out


def read_trials(path: Path) -> list[dict]:
    """A recording's trials, from al_analysis' `movies.TRIALS`: each its frames (``start``,
    ``onset``: the first with odor, ``end``: its last), and its odors as al_analysis' tables
    have them (``odors``: one string per component, abbreviated as it abbreviates them, sorted,
    e.g. ``['ea @ -4.2']``; ``recorded``: as the file names them).

    Raises:
        ExtractError: if it cannot be read as one.
    """
    from hong2p import olf

    try:
        frames = movies.read_trials(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (movies.MovieError, OSError, ValueError) as e:
        raise ExtractError(str(e)) from None
    out = []
    for f, t in zip(frames, raw):
        if isinstance(t.get("name"), str) and "log10_conc" in t:
            parts = [{"name": t["name"], "log10_conc": t["log10_conc"], "abbrev": t.get("abbrev")}]
        else:  # (several odors at once, or none of its own details: as its string says)
            try:
                parts = [dict(o) for o in olf.parse_odor_list(str(t.get("odors")))]
            except Exception:  # noqa: BLE001
                parts = [{"name": str(t.get("odors"))}]
        strs = []
        for o in parts:
            # (hong2p's abbreviation of it, as al_analysis' tables have; else its own, from its odor's YAML)
            own = o.get("abbrev") if isinstance(o.get("abbrev"), str) and o.get("abbrev") else o["name"]
            strs.append(olf.format_odor({**{k: v for k, v in o.items() if k != "abbrev"}, "name": olf.odor2abbrev.get(o["name"], own)}))
        out.append({**f, "odors": sorted(strs), "recorded": f["odor"]})
    return out


def responses(traces: np.ndarray, trials: list[dict], n_response: int = N_RESPONSE) -> np.ndarray:
    """Each ROI's response in each trial ((trials, ROIs)), from its trace ((time points, ROIs)),
    as al_analysis' ``compute_trial_stats``: its dF/F of largest magnitude, with its sign, among
    the first `n_response` time points from the odor's onset, dF/F being (F - F0) / F0 with F0
    the mean of its trace over the trial's time points before the onset."""
    out = np.full((len(trials), traces.shape[1]), np.nan)
    cols = np.arange(traces.shape[1])
    for i, t in enumerate(trials):
        base = _baseline(traces, t)
        window = traces[t["onset"]:min(t["end"] + 1, t["onset"] + n_response)]
        if not len(window):
            continue
        dff = (window - base) / base
        # (the first of those of the same magnitude, and NaN only where all are, as pandas' idxmax)
        mag = np.where(np.isnan(dff), -np.inf, np.abs(dff))
        out[i] = dff[mag.argmax(axis=0), cols]
    return out


def _baseline(traces: np.ndarray, trial: dict) -> np.ndarray:
    """Each ROI's F0 in a trial: the mean of its trace over the trial's time points before the odor."""
    before = traces[trial["start"]:trial["onset"]]
    return before.mean(axis=0) if len(before) else np.full(traces.shape[1], np.nan)


def dff_traces(traces: np.ndarray, trials: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Each ROI's dF/F at every time point of every trial, each trial's against its own F0 (as
    al_analysis' ``trial_response_traces`` with ``keep_pre_odor``).

    Returns:
        (rows, dff): ``rows`` (n, 3): each row's time point in the movie, its trial (from 0) and
        its time point from the odor's onset (negative before it); ``dff`` (n, ROIs).
    """
    rows, parts = [], []
    for i, t in enumerate(trials):
        frames = np.arange(t["start"], min(t["end"] + 1, len(traces)))
        base = _baseline(traces, t)
        parts.append((traces[frames] - base) / base)
        rows.append(np.stack([frames, np.full(len(frames), i), frames - t["onset"]], axis=1))
    if not parts:
        return np.zeros((0, 3), int), np.zeros((0, traces.shape[1]))
    return np.concatenate(rows), np.concatenate(parts)


def quality(resp: np.ndarray) -> np.ndarray:
    """What al_analysis chooses an ROI's best plane by: each ROI-plane's largest response over a
    recording's trials (NaN left out; NaN if all are)."""
    out = np.full(resp.shape[1], np.nan)
    ok = ~np.isnan(resp).all(axis=0) if len(resp) else np.zeros(resp.shape[1], bool)
    if ok.any():
        out[ok] = np.nanmax(resp[:, ok], axis=0)
    return out


def best_planes(names: list[str], q: np.ndarray) -> list[int]:
    """The ROI-planes kept, one per name: of those of a name, the one of greatest `q` (the
    first, of several alike; the first of them, if none has one), in their order among all of
    them, as hong2p's ``rois2best_planes_only`` keeps them."""
    best: dict[str, int] = {}
    for i, n in enumerate(names):
        j = best.get(n)
        if j is None or (not np.isnan(q[i]) and (np.isnan(q[j]) or q[i] > q[j])):
            best[n] = i
    return sorted(best.values())


def noise(traces: np.ndarray, trials: list[dict]) -> np.ndarray:
    """How noisy each ROI-plane's dF/F is where there is no odor: its variance over the time
    points before each trial's onset, about that trial's F0 (pooled over the trials)."""
    ss, dof = np.zeros(traces.shape[1]), 0
    for t in trials:
        before = traces[t["start"]:t["onset"]]
        if len(before) > 1:
            base = before.mean(axis=0)
            ss += (((before - base) / base) ** 2).sum(axis=0)
            dof += len(before) - 1
    return ss / dof if dof else np.full(traces.shape[1], np.nan)


def combine(mode: str, rois: list[dict], recordings: list[dict]) -> None:
    """Makes one signal of each ROI's planes, in each recording (of `recordings`: each with its
    ``traces`` and ``trials``, and after this ``names`` (its ROIs, one per name of `rois`, in
    their order), ``roi_traces`` (time points, ROIs), ``planes`` (per ROI-plane: its ``quality``,
    ``noise``, ``weight`` in its ROI's signal, and whether it is its ``best`` plane there) and
    ``relative`` (whether its traces are of F relative to each plane's mean, not F), by `mode`
    (of `MODES`):

    - ``best_plane``: the plane of greatest `quality` in that recording (al_analysis');
    - ``best_plane_fly``: the plane of greatest `quality` in any of the recordings, in all;
    - ``all_planes``: the mean over all its voxels (its planes' means, each by its pixels);
    - ``weighted``: its planes' traces, each divided by its mean over the recording (so that
      planes of different brightness count alike), averaged with weights 1 / `noise`: the
      average of least noise, were the planes to hold the same signal.
    """
    names = [r["name"] for r in rois]
    pixels = np.array([r["pixels"] for r in rois], dtype=float)
    for rec in recordings:
        rec["q"] = quality(responses(rec["traces"], rec["trials"], rec.get("n_response", N_RESPONSE)))
        rec["noise"] = noise(rec["traces"], rec["trials"])
    if mode == "best_plane_fly":
        stacked = np.stack([rec["q"] for rec in recordings])
        overall = np.where(np.isnan(stacked).all(axis=0), np.nan, np.nanmax(np.where(np.isnan(stacked), -np.inf, stacked), axis=0))
        fly_best = best_planes(names, overall)
    order = list(dict.fromkeys(names))  # (each name once, by its first ROI-plane)
    groups = {n: [i for i, x in enumerate(names) if x == n] for n in order}
    for rec in recordings:
        tr, weight = rec["traces"], np.zeros(len(rois))
        best = best_planes(names, rec["q"])
        if mode in ("best_plane", "best_plane_fly"):
            kept = best if mode == "best_plane" else fly_best
            weight[kept] = 1.0
            rec["names"], rec["roi_traces"] = [names[i] for i in kept], tr[:, kept]
        else:
            if mode == "weighted":
                w = 1.0 / rec["noise"]
                w[~np.isfinite(w)] = 0.0
                scale = tr.mean(axis=0)
            else:
                w, scale = pixels.copy(), np.ones(len(rois))
            cols = []
            for n in order:
                g = groups[n]
                wg = w[g] if w[g].sum() > 0 else np.ones(len(g))  # (no noise known of any: alike)
                weight[g] = wg / wg.sum()
                cols.append((tr[:, g] / scale[g]) @ weight[g])
            rec["names"], rec["roi_traces"] = order, np.stack(cols, axis=1)
        rec["relative"] = mode == "weighted"
        rec["planes"] = [{"roi": names[i], "z": rois[i]["z"], "pixels": rois[i]["pixels"], "quality": _num(rec["q"][i]),
                          "noise": _num(rec["noise"][i]), "weight": float(weight[i]), "best": i in best} for i in range(len(rois))]


def _num(v) -> Optional[float]:
    return None if v is None or not np.isfinite(v) else float(v)


# ------------------------------------------------------- the response window

def window_volumes(pulse_s: float, volume_s: float) -> int:
    """How many volumes, from a trial's first odor volume on, an odor pulse of `pulse_s` seconds
    can have a part in, the volumes `volume_s` seconds each: floor(pulse / volume + 1/2) + 1.

    The first odor volume is the first whose middle is after the valve opens (hong2p's
    ``assign_frames_to_odor_presentations``), so the pulse begins at most half a volume before
    that middle, and ends at most pulse + volume / 2 after the first odor volume began. In the
    ThorSync data of twelve recordings (0.34 to 3 s volumes, pulses of 1 to 3 s), this was the most
    volumes any trial's pulse reached, every time; the other trials' reached one fewer.
    """
    return int(math.floor(pulse_s / volume_s + 0.5)) + 1


def _read_text(p: Path) -> Optional[str]:
    try:
        if p.stat().st_size > tensor.MAX_FILE_BYTES:
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def stimulus_folders(db, tensor_root: Path) -> list[Path]:
    """Where a recording's stimulus YAML is looked for besides `STIMULI` above it: every user's
    folders of odor YAMLs (set in Settings), on the server."""
    with db.connect() as con:
        rows = [r["odor_dirs"] for r in con.execute("SELECT odor_dirs FROM users ORDER BY id")]
    out: dict[Path, None] = {}
    for dirs in rows:
        try:
            out.update((tensor_root / str(d), None) for d in json.loads(dirs or "[]"))
        except (ValueError, TypeError):
            continue
    return list(out)


def stimulus_yaml(recording: Path, names: list[str], also: list[Path] = ()) -> Optional[Path]:
    """The stimulus YAML of a raw recording (a ThorImage folder), of those its notes name
    (`names`: `thorimage.notes_yaml_names`), where hong2p keeps them: `STIMULI`, in a folder
    above the recording (else in one of `also`: `stimulus_folders`), the file in it or in
    the folder olf generated it into (``<date>_<time>_stimuli/``). None if it is not found."""
    for stim in [*(above / STIMULI for above in list(recording.parents)[:5]), *also]:
        if not stim.is_dir():
            continue
        for name in names:
            made = tensor.GENERATED.fullmatch(name)
            for p in (stim / name, *([stim / f"{made[1]}_stimuli" / name] if made else [])):
                if p.is_file():
                    return p
    return None


def recording_timing(recording: Optional[Path], also: list[Path] = ()) -> dict:
    """What a response window is worked out from, of a raw recording (its ThorImage folder):
    ``volume_s`` (a volume's seconds: its planes and flyback frames over its frame rate, by its
    Experiment.xml, with ``planes``, ``flyback`` and ``fps``) and ``pulse_s`` (its odor pulse's
    seconds: ``settings: timing: pulse_us`` of the stimulus YAML its notes name, ``yaml``, as
    olf's generators write it; else that file's own ``pulse_s``, as their configuration has
    it; ``pulse_key``: which). What is not known is None, ``missing`` saying why."""
    out: dict = {"volume_s": None, "pulse_s": None, "pulse_key": None, "xml": None, "yaml": None, "missing": None}
    if recording is None:
        return {**out, "missing": "the fly's raw data folder is not known"}
    xml = recording / XML
    text = _read_text(xml) if xml.is_file() else None
    if text is None:
        return {**out, "missing": f"no Experiment.xml in {recording}"}
    out["xml"] = str(xml)
    try:
        layout = thorimage.raw_layout(text)
    except thorimage.ExperimentXmlError as e:
        return {**out, "missing": str(e)}
    if not layout["fps"]:
        return {**out, "missing": "its Experiment.xml gives no frame rate"}
    out.update(planes=layout["z"], flyback=layout["flyback"], fps=layout["fps"],
               volume_s=(layout["z"] + layout["flyback"]) / layout["fps"])
    names = thorimage.notes_yaml_names(text)
    if not names:
        return {**out, "missing": "its Experiment.xml's notes name no stimulus YAML"}
    path = stimulus_yaml(recording, names, also)
    if path is None:
        return {**out, "missing": f"the stimulus YAML its notes name ({', '.join(names)}) is in no {STIMULI}/ above it, nor in a "
                                  "folder of odor YAMLs set in Settings"}
    out["yaml"] = str(path)
    try:
        import yaml

        data = yaml.safe_load(_read_text(path) or "") or {}
        # (as olf's generators write it out; else as their own configuration has it, which notes may name instead)
        timing = (data.get("settings") or {}).get("timing") or {}
        key, v = next((k, x[k]) for x in (timing, data) for k in ("pulse_us", "pulse_s") if k in x)
        pulse = float(v) / (1e6 if key == "pulse_us" else 1)
    except Exception:  # noqa: BLE001 (not a YAML, or with no pulse length where olf writes it)
        return {**out, "missing": f"{path.name} gives no pulse length (settings: timing: pulse_us, or pulse_s)"}
    if not pulse > 0:
        return {**out, "missing": f"{path.name} gives a pulse of {pulse} s"}
    return {**out, "pulse_s": pulse, "pulse_key": key}


def window(recording: Optional[Path], option, also: list[Path] = ()) -> dict:
    """The response window of a recording: how many volumes, from each trial's first odor
    volume on, a response is taken from (``volumes``), and how that was arrived at (``how``, in
    words; ``set``: whether it was given). By default (`option`: `WINDOW`) as many as its odor
    pulse can reach (`window_volumes`, of `recording_timing`'s pulse and volume: a raw
    recording's folder), `N_RESPONSE` at least; `N_RESPONSE` where they are not known
    (``missing``: why). Else the number given, ``reach`` still how many the pulse can (None
    if not known). `also`: as `stimulus_yaml`'s."""
    t = recording_timing(recording, also)
    vols = lambda n: f"{n} volume{'' if n == 1 else 's'}"  # noqa: E731
    if t["missing"]:
        reach, basis = None, f"How many the odor pulse can reach could not be worked out ({t['missing']})."
    else:
        reach = window_volumes(t["pulse_s"], t["volume_s"])
        planes = f"{t['planes']} plane{'' if t['planes'] == 1 else 's'}" + (f" and {t['flyback']} flyback" if t["flyback"] else "")
        basis = (f"The odor pulse is {t['pulse_s']:g} s ({t['pulse_key']} of {Path(t['yaml']).name}, the stimulus file its notes name) and a "
                 f"volume {t['volume_s']:.4g} s ({planes} at {t['fps']:.4g} frames/s, by its Experiment.xml). The first odor volume is "
                 f"the first whose middle is after the valve opens, so the pulse can begin up to half a volume before that middle, and "
                 f"reach floor({t['pulse_s']:g} / {t['volume_s']:.4g} + 1/2) + 1 = {vols(reach)} from it")
    if option != WINDOW:
        n = int(option)
        return {**t, "reach": reach, "volumes": n, "set": True,
                "how": f"{vols(n)} from the odor's onset, as set. {basis}{'' if reach is None else '.'}"}
    if reach is None:
        return {**t, "reach": None, "volumes": N_RESPONSE, "set": False, "how": f"{vols(N_RESPONSE)} from the odor's onset, as al_analysis. {basis}"}
    least = f"; never fewer than al_analysis' {N_RESPONSE}." if reach < N_RESPONSE else f" (al_analysis takes {N_RESPONSE})."
    return {**t, "reach": reach, "volumes": max(N_RESPONSE, reach), "set": False,
            "how": f"{vols(max(N_RESPONSE, reach))} from the odor's onset. {basis}{least}"}


def panel(recording: str) -> str:
    """A recording's panel, by its folder's name, as al_analysis' ``get_panel`` gives it
    (`DIAGNOSTIC` for one with "diag" in it, else the first of `PANELS` in it); for a name it
    knows no panel of, the name less a number at its end (``my_odors_2``: ``my_odors``)."""
    low = recording.lower()
    if "diag" in low:
        return DIAGNOSTIC
    known = next((p for p in PANELS if p in low), None)
    return known or re.sub(r"[_-]?\d+$", "", recording.removesuffix(movies.REDO)) or recording


# ------------------------------------------------------------- recordings

def recordings(fly_dir: Path) -> list[dict]:
    """A fly's recordings to take signals from, in the order of their names (numbers in them by
    their value): the folders of its motion-corrected data folder with a movie (``movie``: the
    first of `movies.PREFERRED` there, as al_analysis takes it) and its trials (``trials``),
    each with its ``name`` and ``dir``; but not one done again (``<name>_redo`` there too: the
    redo is in its place, as in the site's lists of movies)."""
    found = []
    for d in tensor._subdirs(fly_dir):
        if d.name.startswith(movies.NOT_RECORDINGS):
            continue
        movie = next((d / n for n in movies.PREFERRED if (d / n).is_file()), None)
        if movie is not None and (d / movies.TRIALS).is_file():
            found.append({"name": d.name, "dir": d, "movie": movie, "trials": d / movies.TRIALS})
    redone = {r["name"].removesuffix(movies.REDO) for r in found if r["name"].endswith(movies.REDO)}
    return sorted((r for r in found if r["name"] not in redone), key=lambda r: movies._natural(r["name"]))


def own_roiset(rec_dir: Path, fly_dir: Path) -> Optional[Path]:
    """The fly's own ROIs for a recording, where al_analysis reads them: the ``RoiSet.zip`` in the
    recording's folder (it links every recording's to one), else the one in the fly's."""
    return next((p for p in (rec_dir / movies.ROISET, fly_dir / movies.ROISET) if p.is_file()), None)


def _file(p: Path) -> dict:
    """A file read, for an extraction's provenance: its path, where it leads (if a link), size
    and when it was modified."""
    st = os.stat(p)
    real = os.path.realpath(p)
    return {"path": str(p), **({"links_to": real} if real != str(p) else {}), "bytes": st.st_size,
            "modified": dt.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- records

def root() -> Path:
    return config.DATA_DIR / "extractions"


def out_dir(ext_id: int) -> Path:
    return root() / str(int(ext_id))


def _row(r) -> dict:
    d = dict(r)
    for k in ("rois", "options", "progress", "result"):
        v = d.pop(f"{k}_json")
        d[k] = json.loads(v) if v else None
    return d


def get(db, ext_id: int) -> Optional[dict]:
    with db.connect() as con:
        r = con.execute("SELECT * FROM extractions WHERE id = ?", (ext_id,)).fetchone()
    return _row(r) if r else None


def all_extractions(db) -> list[dict]:
    with db.connect() as con:
        return [_row(r) for r in con.execute("SELECT * FROM extractions ORDER BY id DESC")]


def update(db, ext_id: int, **fields) -> None:
    """Sets columns of an extraction (``*_json`` ones given as their names without it, as values)."""
    cols, vals = [], []
    for k, v in fields.items():
        if k in ("rois", "options", "progress", "result"):
            k, v = f"{k}_json", None if v is None else json.dumps(v)
        cols.append(f"{k} = ?")
        vals.append(v)
    with db.connect() as con:
        con.execute(f"UPDATE extractions SET {', '.join(cols)} WHERE id = ?", (*vals, ext_id))


def options(body: dict) -> dict:
    """The options of an extraction asked for: ``mode`` (of `MODES`; al_analysis' by default) and
    ``window`` (the response window: `WINDOW`, by default, for as many volumes as the odor pulse
    can reach, or a number of volumes from 1 to `MAX_WINDOW`).

    Raises:
        ValueError: for one that is not.
    """
    mode = body.get("mode") or "best_plane"
    if mode not in MODES:
        raise ValueError(f"mode: one of {', '.join(MODES)}")
    win = body.get("window")
    if win in (None, "", WINDOW):
        win = WINDOW
    else:
        try:
            ok = not isinstance(win, bool) and float(win) == int(float(win)) and 1 <= int(float(win)) <= MAX_WINDOW
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            raise ValueError(f"window: a whole number of volumes from 1 to {MAX_WINDOW}, or none (for those the odor pulse can reach)")
        win = int(float(win))
    return {"mode": mode, "window": win}


def locate(db, body: dict, tensor_root: Path, bases: list[Path]) -> dict:
    """What an extraction asked for is of: {"rois": "own" | "predicted" | "edited", and "dir" (the
    fly's motion-corrected data folder, from the top of tensor or absolute: for its own ROIs;
    with "raw_dir", its raw data folder, if it is not where hong2p would have it)
    or "prediction" (its id: for the ROIs it predicted, as they are or as edited)}.

    Returns:
        ``fly_dir`` (on the server), ``raw_dir`` (its raw data folder: as given, a
        prediction's, else the same fly in `RAW` beside `MOCORR`; None if not known: what the
        response window is worked out from), ``fly`` (its name, as a prediction's), ``bytes``
        (of its recordings' movies: what its time is estimated by) and ``rois``: their
        ``kind``, and of a prediction's its ``prediction`` and ``fit``.

    Raises:
        ValueError: saying what is missing: the folder, a prediction done (or its edited ROIs),
            recordings with a movie and trials, or (for its own ROIs) their ``RoiSet.zip``.
    """
    kind = body.get("rois")
    if kind not in ROI_KINDS:
        raise ValueError(f"rois: one of {', '.join(ROI_KINDS)}")
    rois: dict = {"kind": kind}
    if kind == "own":
        if not str(body.get("dir") or "").strip():
            raise ValueError("the fly's motion-corrected data folder: none given")
        found = tensor.find_data(body["dir"], tensor_root, bases)
        if found["status"] != "ok":
            raise ValueError(f"the fly's motion-corrected data folder: {found['error']}")
        if not found["is_dir"]:
            raise ValueError("the fly's motion-corrected data folder: a file, not a folder")
        fly_dir = Path(os.path.realpath(found["path"]))
        name = predictions.fly_name(found["tensor_path"], fly_dir)
        raw = None
        if str(body.get("raw_dir") or "").strip():
            found = tensor.find_data(body["raw_dir"], tensor_root, bases)
            if found["status"] != "ok" or not found["is_dir"]:
                raise ValueError(f"the fly's raw data folder: {found.get('error') or 'a file, not a folder'}")
            raw = os.path.realpath(found["path"])
        elif MOCORR in fly_dir.parts:  # (as hong2p lays a data folder out: the same fly in raw_data/)
            guess = Path(*(RAW if x == MOCORR else x for x in fly_dir.parts))
            raw = str(guess) if guess.is_dir() and sandbox.can_read(guess) else None
    else:
        try:
            p = predictions.get(db, int(body.get("prediction")))
        except (TypeError, ValueError):
            p = None
        if p is None or p["status"] != "done":
            raise ValueError("prediction: the id of a prediction that is done")
        if not (predictions.edited_roiset(p) if kind == "edited" else predictions.roiset(p)).is_file():
            raise ValueError("that prediction's ROIs were not edited (or its edits were deleted)" if kind == "edited"
                             else "that prediction's ROIs are not there")
        fly_dir, name, raw = Path(p["fly_dir"]), p["fly"], p["raw_dir"]
        rois.update(prediction=p["id"], fit=p["fit_id"])
    recs = recordings(fly_dir)
    if not recs:
        raise ValueError(f"no recording in {fly_dir} with a movie ({' / '.join(movies.PREFERRED)}) and its trials "
                         f"({movies.TRIALS}), as al_analysis leaves them")
    if kind == "own" and not any(own_roiset(r["dir"], fly_dir) for r in recs):
        raise ValueError(f"no {movies.ROISET} in the fly's recordings, nor in its folder")
    return {"fly_dir": str(fly_dir), "raw_dir": raw, "fly": name, "rois": rois, "bytes": sum(r["movie"].stat().st_size for r in recs)}


def windows(fly_dir: Path, raw_dir: Optional[str], option, also: list[Path] = ()) -> list[dict]:
    """The response window of each of a fly's recordings (`recordings`), as `window` gives it
    (`option`: an extraction's ``window``; `also`: `stimulus_folders`), each with its
    ``recording``; from its raw recording of the same name in `raw_dir` (the fly's raw data
    folder, if known)."""
    return [{"recording": r["name"], **window(Path(raw_dir) / r["name"] if raw_dir else None, option, also)} for r in recordings(fly_dir)]


def preview(db, body: dict, tensor_root: Path, bases: list[Path]) -> dict:
    """What an extraction asked for as `body` (as `locate` and `options` take it) would take
    its responses from, before it is asked for: each recording's response window (``windows``:
    `windows`), the ``window`` asked for, and ``raw_dir``: the fly's raw data folder they
    were worked out from (None if not known).

    Raises:
        ValueError: as `options` and `locate`.
    """
    opts = options(body)
    of = locate(db, body, tensor_root, bases)
    return {"window": opts["window"], "raw_dir": of["raw_dir"], "al_analysis": N_RESPONSE,
            "windows": windows(Path(of["fly_dir"]), of["raw_dir"], opts["window"], stimulus_folders(db, tensor_root))}


def create(db, of: dict, opts: dict, name: str, ip: Optional[str]) -> dict:
    """Asks for the signals of a fly's ROIs (`of`: as `locate` gives it; `opts`: `options`)."""
    with db.connect() as con:
        ext_id = con.execute(
            "INSERT INTO extractions (fly_dir, raw_dir, fly, rois_json, options_json, bytes, name, ip, created_at, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (of["fly_dir"], of.get("raw_dir"), of["fly"], json.dumps(of["rois"]), json.dumps(opts), of["bytes"], name, ip, now(),
             "queued")).lastrowid
    return get(db, ext_id)


def delete(db, ext_id: int) -> Optional[bool]:
    """Deletes an extraction and its folder. None if there is none such; False if it is running."""
    e = get(db, ext_id)
    if e is None:
        return None
    if e["status"] == "running":
        return False
    with db.connect() as con:
        con.execute("DELETE FROM extractions WHERE id = ? AND status != 'running'", (ext_id,))
    shutil.rmtree(out_dir(ext_id), ignore_errors=True)
    return True


def claim(db) -> Optional[dict]:
    """The oldest queued extraction, now running (for the worker), or None."""
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT id FROM extractions WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
        if r is None:
            return None
        con.execute("UPDATE extractions SET status = 'running', started_at = ? WHERE id = ?", (now(), r["id"]))
    return get(db, r["id"])


def interrupt_running(db) -> list[int]:
    """Marks extractions left running (by a worker that stopped) interrupted. Returns their ids."""
    with db.connect() as con:
        ids = [r["id"] for r in con.execute("SELECT id FROM extractions WHERE status = 'running'")]
        con.execute("UPDATE extractions SET status = 'interrupted', finished_at = ?, error = ? WHERE status = 'running'",
                    (now(), "the fit worker stopped (restarted, or the server did) while it ran"))
    return ids


# ---------------------------------------------------------------- running

README = """Signals of the ROIs of {fly}, extracted from its motion-corrected movies in the odor response
database's AL segmentation tab (Extract signals), asked for by {name} on {created}.

ROIs: {rois}
Recordings: {recordings}
One signal of an ROI drawn on several planes: {mode}

responses.csv       each ROI's response to each trial's odor: trials (panel, is_pair, odor1, ..., repeat)
                    by ROIs ({columns}), every recording's trials one under another. A panel is named
                    as al_analysis names it, from its recordings' names; an odor as al_analysis' tables
                    do (abbreviated @ its log10 concentration); repeat counts the earlier trials of the
                    same odors in the same panel, through all of its recordings
{glomeruli}recordings/<recording>/
  traces.csv        each ROI's trace: time points of the movie (frame, from 0) by ROIs{relative}
  plane_traces.csv  the trace of each of its planes: time points by (roi, z: the plane, from 0), the mean
                    of the movie over the ROI's pixels on that plane (al_analysis' full_traces)
  dff.csv           each ROI's dF/F at each time point of each trial, against that trial's own baseline:
                    frame, trial (from 0), from_onset (time points from its first with odor), odor,
                    repeat, then the ROIs
  responses.csv     the recording's own trials by ROIs, as al_analysis' ij_trial_df has them (repeat
                    counting within the recording; thorimage_id: the recording)
  planes.csv        each ROI's planes: its pixels there, the largest response of that plane alone over
                    the recording's trials (quality: what a best plane is chosen by), the variance of
                    its dF/F before the odor (noise), how much of the ROI's signal is that plane's
                    (weight), and whether it is the plane al_analysis would use (best)
provenance.json     the files read (with their sizes and when they were modified; the ROIs' checksum),
                    how the signals were computed, the software, and who asked
log.txt             what it printed (e.g. ROIs left out, and why)

How they are computed, as al_analysis computes them (with the same numbers, for its way of choosing a
plane): an ROI's mask on a plane is the pixels (y, x) whose point (x, y) is inside its polygon (hong2p's
rule: half a pixel down and right of ImageJ's own, which takes a pixel's centre); its trace there, the mean
of the movie over them. dF/F = (F - F0) / F0, F0 the mean of the trace over the trial's time points before
the odor. A response is the dF/F of largest magnitude, with its sign, among the first time points from the
odor's onset (the response window): {window}. ROIs named {outline} (outlines of the whole antennal lobe) or
ending in {maximal} are left out.

Where these are not al_analysis' own tables: a response's window is al_analysis' {n_response} time points only where
that is what it is above; an odor is abbreviated as hong2p's hardcoded table or the
recording's trials abbreviate it (al_analysis also uses a cache of its own computer's, so a few names differ:
1-prop for its 1-3ol); a recording done again (<name>_redo) is here in place of the first, whole, where
al_analysis blanks only the odors given again; every recording with a movie and its trials is here, none
left out as al_analysis leaves some; and nothing al_analysis does after its per-recording table is done
(ROIs merged across recordings, uncertain ROIs dropped where there is a certain one, concentrations that are
not a panel's final ones blanked, means taken). Another way than its best plane in each recording to make
one signal of an ROI's planes (above) is not al_analysis' either.
"""


def _window_words(recs: list[dict], option) -> str:
    """In words, for an extraction's README: the response window of its recordings (`recs`: each
    with its ``n_response`` and the ``window`` it is of), asked for as `option`."""
    ns = {rec["n_response"] for rec in recs}
    each = f"{ns.pop()} in every recording" if len(ns) == 1 else ", ".join(f"{rec['n_response']} in {rec['name']}" for rec in recs)
    if option != WINDOW:
        return f"{each}, as set when this was asked for (al_analysis takes {N_RESPONSE})"
    unknown = [rec["name"] for rec in recs if rec["window"].get("missing")]
    if len(unknown) == len(recs):
        return (f"{each}, as al_analysis takes: as many volumes as a recording's odor pulse can reach were asked for, but the pulse's "
                "or a volume's length was found for none (log.txt and provenance.json say why)")
    return (f"{each}, which is as many volumes as the recording's odor pulse can reach, by the pulse's and a volume's lengths, and "
            f"never fewer than al_analysis' {N_RESPONSE}"
            + (f" ({N_RESPONSE} too where those lengths were not found: {', '.join(unknown)})" if unknown else "")
            + "; log.txt and provenance.json say what each was worked out from")


def _trial_index(trials: list[dict], panel_name: str, n_odors: int, seen: dict):
    """The rows of a table of trials, as al_analysis' are: (panel, is_pair, odor1, ..., repeat),
    each trial's odors padded with solvent to `n_odors`, ``repeat`` the trials of the same odors
    counted before it in `seen` (which is added to)."""
    import pandas as pd
    from hong2p import olf

    rows = []
    for t in trials:
        odors = tuple(t["odors"] + [olf.solvent_str] * (n_odors - len(t["odors"])))
        rows.append((panel_name, False, *odors, seen.get((panel_name, odors), 0)))
        seen[(panel_name, odors)] = seen.get((panel_name, odors), 0) + 1
    return pd.MultiIndex.from_tuples(rows, names=["panel", "is_pair", *(f"odor{i + 1}" for i in range(n_odors)), "repeat"])


def run(db, e: dict, say: Callable[[str], None] = print, progress: Optional[Callable[[float, float, str], None]] = None) -> dict:
    """Extracts the signals an extraction asks for (`e`: its record), and writes them into its
    folder (`out_dir`, which must be there: `README` says what). `say` is told what to put in
    its log; `progress`, how far it is: (bytes of movies read, of how many, the recording).

    Returns:
        its result: ``mode``, ``recordings`` (each its ``name``, ``panel``, ``trials``,
        ``frames``, ``planes``), ``rois``, ``roi_planes``, ``glomeruli`` (how many ROIs are
        named as one), ``bytes`` (of movies read) and ``seconds``.

    Raises:
        ExtractError: for what cannot be extracted (no recording with ROIs, ROIs that cannot be
            read, trials beyond a movie's end, a prediction since deleted).
    """
    import pandas as pd
    from hong2p import olf

    t0 = time.time()
    d, fly_dir, mode, kind = out_dir(e["id"]), Path(e["fly_dir"]), e["options"]["mode"], e["rois"]["kind"]
    recs = recordings(fly_dir)
    if not recs:
        raise ExtractError(f"no recording in {fly_dir} with a movie and its trials")
    shared, pred = None, None
    if kind != "own":
        pred = predictions.get(db, e["rois"]["prediction"])
        if pred is None or pred["status"] != "done":
            raise ExtractError("the prediction these ROIs were of is no longer there")
        shared = predictions.edited_roiset(pred) if kind == "edited" else predictions.roiset(pred)
        if not shared.is_file():
            raise ExtractError(f"{shared.name} is no longer there")
    # (one asked for before there was a choice of window was asked for al_analysis')
    wins = {w["recording"]: w for w in windows(fly_dir, e.get("raw_dir"), e["options"].get("window", N_RESPONSE),
                                               stimulus_folders(db, config.TENSOR_ROOT))}
    sizes = {r["name"]: r["movie"].stat().st_size for r in recs}
    total, read = sum(sizes.values()), 0
    groups: dict[tuple, dict] = {}  # recordings by the ROIs' file and the movie's planes: one set of masks each
    used = []
    for r in recs:
        path = shared if shared is not None else own_roiset(r["dir"], fly_dir)
        if path is None:
            say(f"{r['name']}: left out, with no {movies.ROISET} in it, nor in the fly's folder")
            read += sizes[r["name"]]
            continue
        try:
            movie = movies.open_movie(r["movie"])
        except movies.MovieError as err:
            raise ExtractError(f"{r['movie']}: {err}") from None
        s = movie.shape
        key = (os.path.realpath(path), s["z"], s["y"], s["x"])
        if key not in groups:
            say(f"ROIs: {path} ({_file(path)['modified']}), on planes of {s['z']} x {s['y']} x {s['x']} (z, y, x)")
            groups[key] = {"path": path, "file": {**_file(path), "sha256": _sha256(path)}, "recordings": [],
                           "rois": read_rois(path, (s["z"], s["y"], s["x"]), say)}
        g = groups[key]
        trials = read_trials(r["trials"])
        if any(t["end"] >= s["t"] or not 0 <= t["start"] <= t["onset"] <= t["end"] for t in trials):
            raise ExtractError(f"{r['name']}: its trials ({movies.TRIALS}) have frames out of order, or beyond its movie's {s['t']} time points")
        t1 = time.time()
        traces = plane_traces(movie, g["rois"], (lambda a, b, base=read, size=sizes[r["name"]], name=r["name"]:
                                                 progress(base + size * a / b, total, name)) if progress else None)
        read += sizes[r["name"]]
        win = wins[r["name"]]
        rec = {"name": r["name"], "panel": panel(r["name"]), "trials": trials, "traces": traces, "group": g,
               "movie": {**_file(r["movie"]), "shape": dict(s), "dtype": str(movie.volumes(0, 1).dtype), "fps": movie.fps},
               "trials_file": _file(r["trials"]), "window": win, "n_response": win["volumes"]}
        say(f"{r['name']} ({rec['panel']}): {s['t']} time points x {s['z']} planes, {len(trials)} trials, {len(g['rois'])} ROI-planes; "
            f"read in {time.time() - t1:.1f} s")
        say(f"{r['name']}: a response is taken from {win['how']}")
        short = sum(t["end"] - t["onset"] + 1 < win["volumes"] for t in trials)
        if short:
            say(f"{r['name']}: {short} of its trials end{'s' if short == 1 else ''} before that many volumes: theirs is from those they have")
        g["recordings"].append(rec)
        used.append(rec)
    if not used:
        raise ExtractError(f"no recording of the fly has ROIs ({movies.ROISET})")
    for g in groups.values():
        combine(mode, g["rois"], g["recordings"])

    fly, _ = movies._path_fly(fly_dir / "x")
    date_fly = [fly["date"], fly["fly_num"]] if fly else []
    n_odors = max(len(t["odors"]) for rec in used for t in rec["trials"])
    seen: dict = {}
    parts = []
    for rec in used:
        rd = d / "recordings" / rec["name"]
        rd.mkdir(parents=True)
        names, rois = rec["names"], rec["group"]["rois"]
        pd.DataFrame(rec["roi_traces"], columns=pd.Index(names, name="roi")).rename_axis("frame").to_csv(rd / "traces.csv")
        cols = pd.MultiIndex.from_arrays([[x["name"] for x in rois], [x["z"] for x in rois]], names=["roi", "z"])
        pd.DataFrame(rec["traces"], columns=cols).rename_axis("frame").to_csv(rd / "plane_traces.csv")
        resp = responses(rec["roi_traces"], rec["trials"], rec["n_response"])
        own = _trial_index(rec["trials"], rec["panel"], max(len(t["odors"]) for t in rec["trials"]), {})
        top = pd.MultiIndex.from_tuples([(*date_fly, rec["name"], n) for n in names], names=[*(["date", "fly_num"] if fly else []), "thorimage_id", "roi"])
        pd.DataFrame(resp, index=own, columns=top).to_csv(rd / "responses.csv")
        rows, dff = dff_traces(rec["roi_traces"], rec["trials"])
        labels = pd.DataFrame({"frame": rows[:, 0], "trial": rows[:, 1], "from_onset": rows[:, 2],
                               "odor": [olf.format_mix_from_strs(rec["trials"][i]["odors"]) for i in rows[:, 1]],
                               "repeat": [own.get_level_values("repeat")[i] for i in rows[:, 1]]})
        pd.concat([labels, pd.DataFrame(dff, columns=names)], axis=1).to_csv(rd / "dff.csv", index=False)
        pd.DataFrame(rec["planes"]).to_csv(rd / "planes.csv", index=False)
        parts.append(pd.DataFrame(resp, index=_trial_index(rec["trials"], rec["panel"], n_odors, seen), columns=names))
    table = pd.concat(parts, sort=False)
    names = list(table.columns)
    table.columns = pd.MultiIndex.from_tuples([(*date_fly, n) for n in names], names=[*(["date", "fly_num"] if fly else []), "roi"])
    table.to_csv(d / "responses.csv")
    # those an upload takes: named as glomeruli, and tentative ones (DA1_t0, which it reads as DA1) of no other ROI's glomerulus
    kinds = {n: movies.roi_kind(n) for n in names}
    bare = [movies.ROI_SUFFIX.sub("", n) for n in names]
    gloms = [n for n, b in zip(names, bare) if kinds[n] == "glomerulus" or (kinds[n] == "tentative" and bare.count(b) == 1)]
    if gloms:
        table.loc[:, [n in gloms for n in names]].to_csv(d / "responses_glomeruli.csv")

    seconds = time.time() - t0
    result = {"mode": mode, "rois": len(names), "roi_planes": max(len(g["rois"]) for g in groups.values()), "glomeruli": len(gloms),
              "recordings": [{"name": rec["name"], "panel": rec["panel"], "trials": len(rec["trials"]), "frames": rec["movie"]["shape"]["t"],
                              "planes": rec["movie"]["shape"]["z"],
                              "window": {k: rec["window"].get(k) for k in ("volumes", "set", "reach", "pulse_s", "volume_s", "missing", "how")}}
                             for rec in used],
              "bytes": total, "seconds": seconds}
    keys = ("volumes", "set", "how", "reach", "pulse_s", "volume_s", "planes", "flyback", "fps", "xml", "yaml", "missing")
    what = {"own": "the fly's own", "predicted": "as predicted", "edited": "as predicted, then edited in the viewer"}[kind]
    of = f", of fit #{pred['fit_id']}'s model (prediction {pred['id']})" if pred else ""
    prov = {
        "extraction": e["id"], "asked_for": {"by": e["name"], "from": e["ip"], "at": e["created_at"]}, "started": e["started_at"],
        "finished": now(), "seconds": seconds,
        "fly": {"name": e["fly"], "dir": str(fly_dir), **(fly or {})},
        "rois": {"kind": kind, "what": what, **({"prediction": pred["id"], "fit": pred["fit_id"],
                                                 "edited": predictions.edited_info(pred) if kind == "edited" else None} if pred else {}),
                 "files": [{**g["file"], "roi_planes": len(g["rois"]), "recordings": [rec["name"] for rec in g["recordings"]]} for g in groups.values()]},
        "recordings": [{"name": rec["name"], "panel": rec["panel"], "movie": rec["movie"], "trials": rec["trials_file"],
                        "n_trials": len(rec["trials"]), "rois": len(rec["names"]),
                        "response_window": {k: rec["window"][k] for k in keys if k in rec["window"]}} for rec in used],
        "computed": {"mode": mode, "mode_what": MODES[mode], "as": "al_analysis' ij_traces and compute_trial_stats",
                     "mask": "pixel (y, x) is in an ROI if the point (x, y) is inside its polygon (matplotlib's Path.contains_points, as hong2p's ijroi2mask)",
                     "trace": "the mean of the movie over the mask, at each time point",
                     "baseline": "F0: the mean of the trace over all of the trial's time points before the odor",
                     "response": "the dF/F of largest magnitude, with its sign, among the first time points from the odor's onset: "
                                 "as many as each recording's response_window says (its volumes), or as its trial has",
                     "response_window": e["options"].get("window", N_RESPONSE),
                     "n_volumes_for_response": {rec["name"]: rec["n_response"] for rec in used},
                     "al_analysis_n_volumes_for_response": N_RESPONSE, "n_volumes_for_baseline": None,
                     "response_stat_fn": "sign_preserving_maxabs",
                     "left_out": f"ROIs named {OUTLINE}, or ending in {MAXIMAL}"},
        "software": _software(),
        "outputs": {k: v for k, v in result.items() if k != "seconds"},
    }
    (d / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    (d / "README.txt").write_text(README.format(
        fly=e["fly"], name=e["name"], created=e["created_at"], rois=f"{what}{of}: {', '.join(str(g['path']) for g in groups.values())}",
        recordings=", ".join(rec["name"] for rec in used), mode=MODES[mode],
        columns="date, fly_num, roi" if fly else "roi",
        glomeruli=("responses_glomeruli.csv\n                    the same, of the ROIs named as glomeruli alone (those of the site's list; also with\n"
                   "                    _t0, _t1, ... after their name, where no other ROI is that glomerulus). The Upload tab\n"
                   f"                    takes this file as it is{'' if fly else ', once the fly of its ROIs (date, fly_num) is added: its folder does not say'}\n")
        if gloms else "",
        relative=" (F relative to each plane's mean\n                    over the recording, not F: see above)" if mode == "weighted" else "",
        window=_window_words(used, e["options"].get("window", N_RESPONSE)), n_response=N_RESPONSE, outline=OUTLINE, maximal=MAXIMAL))
    say(f"{len(names)} ROIs ({result['roi_planes']} ROI-planes), {len(gloms)} of them named as glomeruli, in {len(used)} recordings; {seconds:.1f} s")
    return result


def _software() -> dict:
    """The versions of what an extraction was computed with."""
    from importlib import metadata

    from . import version

    out = {"olfdb": version.VERSION, "numpy": np.__version__}
    for pkg in ("pandas", "tifffile", "roifile", "hong2p"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    try:
        direct = json.loads(metadata.distribution("hong2p").read_text("direct_url.json") or "{}")
        out["hong2p_commit"] = (direct.get("vcs_info") or {}).get("commit_id")
    except (metadata.PackageNotFoundError, ValueError, TypeError):
        pass
    return out


# -------------------------------------------------------------- estimates

def _rate(done: list[dict]) -> tuple[float, int]:
    """Seconds an extraction takes per byte of movies: the median of the last `RECENT` done (of
    `done`, newest first), and how many that is of; `GUESS_S_PER_GB`'s and 0 with none."""
    rates = [x["result"]["seconds"] / x["bytes"] for x in done if x.get("bytes") and (x.get("result") or {}).get("seconds") is not None][:RECENT]
    return (statistics.median(rates), len(rates)) if rates else (GUESS_S_PER_GB / 2**30, 0)


def estimate(exts: list[dict], t: Optional[float] = None, new_bytes: Optional[int] = None) -> Optional[dict]:
    """Gives each extraction of `exts` queued or running its ``eta``: ``seconds`` it has left
    (one running: from how far it has got, once it has got anywhere; else from the bytes of its
    fly's movies, at the rate of those done: `_rate`), ``until`` it ends (one runs at a time),
    ``ends_at``, and ``from``: how many extractions done the rate is of (0: a guess).

    Returns:
        the same for one of `new_bytes` of movies asked for now, if given.
    """
    t = time.time() if t is None else t
    rate, n = _rate(sorted((x for x in exts if x["status"] == "done"), key=lambda x: x.get("finished_at") or "", reverse=True))
    at = t

    def eta(left: float) -> dict:
        nonlocal at
        at += left
        return {"seconds": left, "until": at - t, "from": n, "ends_at": dt.datetime.fromtimestamp(at).astimezone().isoformat(timespec="seconds")}

    for x in sorted((x for x in exts if x["status"] in ACTIVE), key=lambda x: (x["status"] != "running", x["id"])):
        left = rate * (x.get("bytes") or 0)
        pr = x.get("progress") or {}
        if x["status"] == "running" and x.get("started_at"):
            elapsed = t - dt.datetime.fromisoformat(x["started_at"]).timestamp()
            left = elapsed / pr["done"] * (pr["total"] - pr["done"]) if pr.get("done") and pr.get("total") else max(0.0, left - elapsed)
        x["eta"] = eta(max(0.0, left))
    return eta(rate * new_bytes) if new_bytes is not None else None


def listing(db, tensor_root: Path) -> dict:
    """Every extraction (newest first: each with ``tensor_dir``, its fly's folder from the top of
    tensor, and those queued or running with ``eta``: `estimate`), how one signal may be made of
    an ROI's planes (``modes``: `MODES`), and ``rate``: the seconds an extraction takes per GB of
    movies, with how many done that is of (``from``)."""
    from . import fits, roi_training

    exts = all_extractions(db)
    for x in exts:
        x["tensor_dir"] = roi_training._arcname(Path(x["fly_dir"]), tensor_root, [])
    estimate(exts)
    rate, n = _rate([x for x in exts if x["status"] == "done"])
    return {"extractions": exts, "modes": MODES, "rate": {"seconds_per_gb": rate * 2**30, "from": n}, "worker": fits.worker_status(db),
            "tensor_root": str(tensor_root), "window": {"al_analysis": N_RESPONSE, "max": MAX_WINDOW}}


def stream_zip(ext_id: int, top: str):
    """An extraction's folder as a zip, under ``top``, as it is written."""
    return predictions.zip_folder(out_dir(ext_id), top)
