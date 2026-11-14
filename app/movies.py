"""Movies of a dataset's flies on tensor: finding them, and reading the planes shown.

Motion-corrected movies are TIFFs under the dataset's motion-corrected data folder, as
al_analysis writes them in each fly's folder: all of its recordings registered together
(``mocorr_concat.tif``, of the recordings whose own motion-corrected movies are next to
it, in the suite2p folder), else one movie per recording folder (``mocorr.tif``, else
``flipped.tif`` or ``raw.tif``). A fly's folder with neither, or a folder with no fly's
folder in it, has every TIFF in it listed. Each movie has the ImageJ ROIs drawn on it
(``RoiSet.zip``, next to the movie or in a folder above it) and its trials
(``trial_frames_and_odors.json``: frames and odors), if any. Raw movies are ThorImage
``.raw`` files, with their ``Experiment.xml`` next to them. Planes are read on the server,
only those shown (Z max projections computed here too), and sent to the browser as plain
little-endian numbers, so it never has to parse a TIFF or read a whole movie.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import posixpath
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import glomeruli as glom
from . import tensor, thorimage
from .ingest import ROI_SUFFIX

TIFF_SUFFIXES = (".tif", ".tiff")
PREFERRED = ("mocorr.tif", "flipped.tif", "raw.tif")  # al_analysis' names, most processed first
CONCAT = "mocorr_concat.tif"  # al_analysis' movie of a fly's recordings registered together
TRIALS = "trial_frames_and_odors.json"  # al_analysis' trials of a movie: their frames and odors
NOT_RECORDINGS = ("suite2p",)  # folders in a fly's that are not recordings (al_analysis' suite2p runs)
NOT_MOVIES = {"roimask.raw"}  # ThorImage's other .raw files, beside a recording's movie (lower case)
REDO = "_redo"  # a recording's folder <recording>_redo: it done again, in place of <recording>
ROISET = "RoiSet.zip"
DEPTH = 3  # folders scanned below a fly's folder (or the data folder, if no fly folder is found)
MAX_DIRS = 2000
MAX_PLANES_BYTES = 64 << 20  # per request for planes
CACHED = 8  # movies kept open


class MovieError(ValueError):
    pass


# ----------------------------------------------------------------------- finding


def _walk(top: Path, depth: int):
    """(folder, its entries) for `top` and its subfolders, down to `depth` levels below it."""
    todo, n = [(top, 0)], 0
    while todo and n < MAX_DIRS:
        d, level = todo.pop(0)
        n += 1
        entries = tensor._entries(d)
        yield d, entries
        if level < depth:
            todo += [(Path(e.path), level + 1) for e in entries if tensor._is_dir(e)]


def _size(e: os.DirEntry) -> Optional[int]:
    try:
        return e.stat().st_size
    except OSError:
        return None


def find(folder: Path, kind: str, flies: list[dict], skipped: Optional[set] = None) -> list[dict]:
    """Movies under one of a dataset's data folders (`kind`: 'raw' or 'mocorr').

    Looked for in its flies' folders (``<folder>/<date>/<fly>/``, as for the upload
    lookup): for motion-corrected data, as al_analysis lays them out (`_fly_movies`), else
    in the fly's folder and the folders below it. If none of its flies' folders is there
    (the folder may be a single fly's or recording's), in `folder` and the folders below
    it, each then of the fly the folders in its path say (`_fly_of`), if any, but for
    those in the folder of a fly not one of `flies` below `folder` (``<date>/<fly>``):
    their flies' (date, fly_num) are added to `skipped`.

    Returns:
        dicts with ``path`` (on the server), ``label`` (from `folder`; a fly's movie of
        its recordings registered together: the fly, ``<date>/<fly_num>``), ``format``
        ('tiff' or 'raw'), ``size``, ``fly`` (its date and number, if known), ``by`` (how:
        'fly folder', 'path', or None), ``recording`` (the folder in the fly's it is in;
        None if it is in the fly's itself, or its fly is not known), ``recordings`` (those
        in a fly's movie of them registered together, in the order of their names;
        None for any other movie), ``rois`` (the RoiSet.zip drawn on it, if any, for a
        TIFF) and ``trials`` (its `TRIALS`, if any).
    """
    tops = [(d, f) for f in flies for d in tensor.fly_dirs(folder, f["date"], f["fly_num"])] or [(folder, None)]
    out: list[dict] = []
    for top, fly in tops:
        laid_out = _fly_movies(folder, top, fly) if kind == "mocorr" and fly else None
        if laid_out is not None:
            out += laid_out
            continue
        rois: dict[Path, Path] = {}
        for d, entries in _walk(top, DEPTH):
            names = {e.name for e in entries}
            # the nearest RoiSet.zip at or above the folder (up to the fly's)
            rois[d] = d / ROISET if ROISET in names else rois.get(d.parent)
            files = [e for e in entries if not tensor._is_dir(e)]
            if kind == "mocorr":
                movies = [e for e in files if e.name.lower().endswith(TIFF_SUFFIXES)]
                movies.sort(key=lambda e: (PREFERRED.index(e.name) if e.name in PREFERRED else len(PREFERRED), e.name))
            else:
                movies = [e for e in files if e.name.lower().endswith(".raw") and e.name.lower() not in NOT_MOVIES
                          and "Experiment.xml" in names]
            for e in movies:
                size = _size(e)
                if size is None:
                    continue
                below = Path(e.path).relative_to(top).parts
                if fly:
                    of, by, recording = {"date": fly["date"], "fly_num": fly["fly_num"]}, "fly folder", below[0] if len(below) > 1 else None
                else:
                    other, _ = _path_fly(Path(e.path).relative_to(folder))
                    if other and not any(other == {"date": f["date"], "fly_num": f["fly_num"]} for f in flies):
                        if skipped is not None:
                            skipped.add((other["date"], other["fly_num"]))
                        continue
                    of, recording = _fly_of(Path(e.path), flies)
                    by = "path" if of else None
                out.append({
                    "path": e.path, "label": Path(e.path).relative_to(folder).as_posix(),
                    "format": "raw" if kind == "raw" else "tiff", "size": size, "fly": of, "by": by, "recording": recording,
                    "recordings": None, "rois": str(rois[d]) if kind == "mocorr" and rois.get(d) else None,
                    "trials": str(d / TRIALS) if TRIALS in names else None,
                })
    return out


def _fly_movies(folder: Path, top: Path, fly: dict) -> Optional[list[dict]]:
    """A fly's motion-corrected movies (as `find` lists them), if its folder (`top`) is laid
    out as al_analysis writes it: its recordings registered together (`CONCAT`, in the
    fly's folder), and one movie per recording folder not in it (the first of `PREFERRED`
    in it). None if it is not (there is neither), for `find` to list every TIFF in it.

    Fly-level TIFFs other than `CONCAT` (al_analysis' averages and concatenated dF/F) and
    the folders of `NOT_RECORDINGS` are not movies of its recordings, and are left out.
    """
    of = {"date": fly["date"], "fly_num": fly["fly_num"]}
    entries = tensor._entries(top)
    files = {e.name: e for e in entries if not tensor._is_dir(e)}
    # each recording folder's files, by name
    recs = {e.name: (Path(e.path), {x.name: x for x in tensor._entries(Path(e.path)) if not tensor._is_dir(x)})
            for e in entries if tensor._is_dir(e) and not e.name.startswith(NOT_RECORDINGS)}
    best = {r: next((n for n in PREFERRED if n in names), None) for r, (_, names) in recs.items()}
    concat = files.get(CONCAT)
    if concat is None and not any(best.values()):
        return None
    trials = {r: str(d / TRIALS) for r, (d, names) in recs.items() if TRIALS in names}
    out, covered = [], set()
    if concat is not None and (size := _size(concat)) is not None:
        in_it = _concat_recordings(top, Path(concat.path))
        covered = set(in_it)
        out.append({
            "path": concat.path, "label": f"{of['date']}/{of['fly_num']}", "format": "tiff", "size": size, "fly": of,
            "by": "fly folder", "recording": None, "recordings": in_it,
            "rois": str(top / ROISET) if ROISET in files else _shared_rois(recs, in_it),
            "trials": str(top / TRIALS) if TRIALS in files else None,
            # each recording's own trials (the frames of its raw data)
            "recording_trials": {r: trials[r] for r in in_it if r in trials},
        })
    for r in sorted(best, key=_natural):
        d, names = recs[r]
        if best[r] is None or r in covered or (size := _size(names[best[r]])) is None:
            continue
        path = d / best[r]
        out.append({
            "path": str(path), "label": path.relative_to(folder).as_posix(), "format": "tiff", "size": size, "fly": of,
            "by": "fly folder", "recording": r, "recordings": None,
            "rois": str(d / ROISET) if ROISET in names else str(top / ROISET) if ROISET in files else None,
            "trials": trials.get(r),
        })
    return out


def _concat_recordings(top: Path, concat: Path) -> list[str]:
    """The recordings in a fly's movie of them registered together (`CONCAT`, in its folder
    `top`), in the order of their names: those whose motion-corrected movies
    (``<recording>.tif``) are next to the movie it links to, where al_analysis writes them
    (none known if it is in the fly's folder itself)."""
    try:
        real = concat.resolve(strict=True)
        if real.parent == top.resolve():
            return []
    except OSError:
        return []
    names = [Path(e.name).stem for e in tensor._entries(real.parent)
             if not tensor._is_dir(e) and e.name.lower().endswith(TIFF_SUFFIXES) and not e.name.startswith("mocorr_concat")]
    return sorted(names, key=_natural)


def _shared_rois(recs: dict[str, tuple[Path, dict]], recordings: list[str]) -> Optional[str]:
    """The RoiSet.zip of `recordings` (folder, files by name: as in `_fly_movies`), if they
    have one and all theirs are the same file (al_analysis links the others to one); that of
    the recording it is in, if it is one of them."""
    paths = [recs[r][0] / ROISET for r in recordings if r in recs and ROISET in recs[r][1]]
    try:
        real = {p.resolve(strict=True) for p in paths}
    except OSError:
        return None
    if len(real) != 1:
        return None
    return str(next((p for p in paths if not p.is_symlink()), paths[0]))


def _fly_of(path: Path, flies: list[dict]) -> tuple[Optional[dict], Optional[str]]:
    """The fly (of `flies`) a movie found outside their folders is of, by the folders in its
    path (``.../<date>/<fly>/[<recording>/...]``), and the folder in the fly's it is in;
    (None, None) if no fly is in its path."""
    parts = path.parts[:-1]
    for i in range(len(parts) - 1, 0, -1):  # the folder nearest the movie
        for f in flies:
            if parts[i - 1] in tensor.date_names(f["date"]) and tensor.fly_name(f["fly_num"]).fullmatch(parts[i]):
                return {"date": f["date"], "fly_num": f["fly_num"]}, parts[i + 1] if i + 1 < len(parts) else None
    return None, None


def _path_fly(path: Path) -> tuple[Optional[dict], Optional[str]]:
    """The fly the folders in a movie's path name, whatever fly it is (``.../<date>/<fly>/
    [<recording>/...]``, the nearest), and the folder in the fly's it is in; (None, None) if none."""
    parts = path.parts[:-1]
    for i in range(len(parts) - 1, 0, -1):
        fly = re.fullmatch(r"(?:fly)?[_-]?0*(\d+)", parts[i], re.IGNORECASE)
        for fmt in tensor.DATE_FORMATS if fly else ():
            try:
                date = dt.datetime.strptime(parts[i - 1], fmt).date().isoformat()
            except ValueError:
                continue
            return {"date": date, "fly_num": int(fly[1])}, parts[i + 1] if i + 1 < len(parts) else None
    return None, None


def _recording_key(m: dict) -> tuple:
    """A movie's recording, to pair those of the raw and motion-corrected data: its fly (of
    the dataset's, else the one its path names, else None) and the folder in the fly's it
    is in; for one in no fly's folder, the folder it is in."""
    if m["fly"]:
        return True, m["fly"]["date"], m["fly"]["fly_num"], m["recording"]
    fly, recording = _path_fly(Path(m["path"]))
    if fly:
        return False, fly["date"], fly["fly_num"], recording
    return False, None, None, posixpath.dirname(m["label"]) or None


def _recording_keys(m: dict) -> list[tuple]:
    """The recordings of a movie (`_recording_key`): each of those in a fly's movie of them
    registered together, else its own."""
    key = _recording_key(m)
    return [(*key[:3], r) for r in m["recordings"]] if m.get("recordings") else [key]


def _anatomical(key: tuple) -> bool:
    return bool(key[1] and key[3] and tensor.ANATOMICAL.fullmatch(key[3]))


def drop_redone(raw: list[dict], mocorr: list[dict]) -> tuple[list[dict], list[dict]]:
    """`find`'s movies of a dataset's raw and motion-corrected data folders, but for those of
    recordings redone: a fly's ``<recording>_redo`` (`REDO`, in either folder) is in place
    of its ``<recording>``, whose movies are left out of both, as is it from a fly's movie of
    its recordings registered together (that movie too, if it was all such).

    Returns:
        (raw, mocorr): the movies kept, in the order found.
    """
    keys = {k for m in (*raw, *mocorr) for k in _recording_keys(m)}
    redone = {(*k[:3], k[3].removesuffix(REDO)) for k in keys if k[3] and k[3].endswith(REDO)}

    def kept(m: dict) -> bool:
        if not m.get("recordings"):
            return _recording_key(m) not in redone
        fly = _recording_key(m)[:3]
        m["recordings"] = [r for r in m["recordings"] if (*fly, r) not in redone]
        if m.get("recording_trials"):
            m["recording_trials"] = {r: p for r, p in m["recording_trials"].items() if r in m["recordings"]}
        return bool(m["recordings"])

    return [m for m in raw if kept(m)], [m for m in mocorr if kept(m)]


def unmatched(raw: list[dict], mocorr: list[dict]) -> dict[str, list[dict]]:
    """The recordings with movies in one of a dataset's data folders and none in the other
    (`find`'s movies of each: by fly and the name of the recording's folder, which
    al_analysis takes from the raw data's; see `_recording_keys`), but for anatomical ones
    (`tensor.ANATOMICAL`, never motion-corrected). Each movie whose recordings are all such
    is marked ``unmatched`` (and the rest not).

    Returns:
        by kind ('raw', 'mocorr'): its recordings not in the other, each with ``date``,
        ``fly_num`` (None if in no fly's folder), ``of_dataset`` (whether the fly is one of
        the dataset's), ``recording`` and ``n_movies``.
    """
    found = {"raw": raw, "mocorr": mocorr}
    by_kind: dict[str, dict[tuple, list[dict]]] = {}
    for kind, movies in found.items():
        by_kind[kind] = {}
        for m in movies:
            for key in _recording_keys(m):
                if not _anatomical(key):
                    by_kind[kind].setdefault(key, []).append(m)
    result = {}
    for kind, other in (("raw", "mocorr"), ("mocorr", "raw")):
        result[kind] = []
        for key, movies in by_kind[kind].items():
            if key not in by_kind[other]:
                of_dataset, date, fly_num, recording = key
                result[kind].append({"date": date, "fly_num": fly_num, "of_dataset": of_dataset, "recording": recording,
                                     "n_movies": len(movies)})
        for m in found[kind]:
            keys = [k for k in _recording_keys(m) if not _anatomical(k)]
            m["unmatched"] = bool(keys) and all(k not in by_kind[other] for k in keys)
    return result


def link_trials(raw: list[dict], mocorr: list[dict]) -> None:
    """Give each raw movie without trials of its own those of its recording in the
    motion-corrected data (`find`'s ``trials``, or a fly's movie's ``recording_trials``):
    a recording's frames are the same in both."""
    of: dict[tuple, str] = {}
    for m in mocorr:
        if m.get("recordings"):
            for r, path in (m.get("recording_trials") or {}).items():
                of.setdefault((*_recording_key(m)[:3], r), path)
        elif m.get("trials") and m["recording"]:
            of.setdefault(_recording_key(m), m["trials"])
    for m in raw:
        if not m.get("trials"):
            m["trials"] = of.get(_recording_key(m))


def _natural(name: Optional[str]) -> list:
    """`name` to sort by, numbers in it by their value (megamat_2 before megamat_10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name or "")]


def side_by_side(raw: list[dict], mocorr: list[dict]) -> list[dict]:
    """Every recording in a dataset's raw and motion-corrected data folders, paired as
    `unmatched` pairs them (`find`'s movies of each, a fly's movie of its recordings
    registered together in each of them; those of flies not the dataset's after its own,
    then those in no fly's folder, by folder), in the order they were recorded: by fly,
    then by when its raw recording says it was made (its Experiment.xml's date; if one of
    the fly's does not say, when they were written); one with no raw data after the one
    before it by name (megamat_3 after megamat_2), there being no time it was recorded.

    Returns:
        rows with ``date``, ``fly_num`` (both None if in no fly's folder), ``of_dataset``
        (whether the fly is one of the dataset's; those that are not after them),
        ``recording`` (None: the fly's folder itself; for one in no fly's folder, the
        folder it is in, from the data folder's),
        ``raw`` and ``mocorr`` (the names of its movies in each, None if it has none
        there), ``recorded`` (when it was recorded, as its Experiment.xml says:
        YYYY-MM-DD HH:MM:SS, the acquisition computer's time), ``time`` (when its
        Experiment.xml was written, in seconds since 1970; both None without raw data),
        ``matched`` and ``anatomical``.
    """
    rows: dict[tuple, dict] = {}
    for kind, found in (("raw", raw), ("mocorr", mocorr)):
        for m in found:
            for key in _recording_keys(m):
                row = rows.setdefault(key, {"date": key[1], "fly_num": key[2], "of_dataset": key[0], "recording": key[3],
                                            "raw": None, "mocorr": None, "recorded": None, "time": None})
                row[kind] = [*(row[kind] or []), Path(m["path"]).name]
            if kind == "raw" and row["time"] is None:
                xml = Path(m["path"]).parent / "Experiment.xml"
                try:
                    row["time"] = os.path.getmtime(xml)
                    if os.path.getsize(xml) < tensor.MAX_FILE_BYTES:
                        at = thorimage.recorded_at(xml.read_text(errors="replace"))
                        row["recorded"] = at.isoformat(sep=" ") if at else None
                except OSError:
                    pass
    out = []
    flies = {(r["of_dataset"], r["date"], r["fly_num"]) for r in rows.values()}
    for fly in sorted(flies, key=lambda f: (not f[0], f[1] is None, f[1:] if f[1] else ())):
        mine = [r for r in rows.values() if (r["of_dataset"], r["date"], r["fly_num"]) == fly]
        # by when they say they were recorded, else (none of them saying) when their Experiment.xml was written
        by = "recorded" if all(r["recorded"] for r in mine if r["time"] is not None) else "time"
        order = sorted((r for r in mine if r["time"] is not None), key=lambda r: (r[by], _natural(r["recording"])))
        for r in sorted((r for r in mine if r["time"] is None), key=lambda r: _natural(r["recording"])):
            before = [o for o in order if _natural(o["recording"]) <= _natural(r["recording"])]
            order.insert(order.index(max(before, key=lambda o: _natural(o["recording"]))) + 1 if before else 0, r)
        out += order
    for r in out:
        r.update(matched=r["raw"] is not None and r["mocorr"] is not None,
                 anatomical=bool(r["date"] and r["recording"] and tensor.ANATOMICAL.fullmatch(r["recording"])))
    return out


# ----------------------------------------------------------------------- reading


class Movie:
    """A movie's shape (``t``, ``z``, ``c``, ``y``, ``x``) and its planes."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.warnings: list[str] = []
        self.fps: Optional[float] = None
        self.tif = None
        if path.suffix.lower() == ".raw":
            self._open_raw()
        else:
            self._open_tiff()

    def _open_raw(self) -> None:
        xml = self.path.with_name("Experiment.xml")
        try:
            layout = thorimage.raw_layout(xml.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            raise MovieError(f"cannot read the Experiment.xml next to it ({e})")
        except thorimage.ExperimentXmlError as e:
            raise MovieError(str(e))
        y, x, z, flyback, c = (layout[k] for k in ("y", "x", "z", "flyback", "c"))
        volume = (z + flyback) * c * y * x * 2
        size = os.path.getsize(self.path)
        t, extra = divmod(size, volume)
        if extra:
            self.warnings.append(f"the file is {extra} bytes longer than {t} whole volumes of the Experiment.xml's "
                                 f"layout ({z} planes, {flyback} flyback frames, {c} channels of {x} × {y}); "
                                 "the rest is not shown")
        if not t:
            raise MovieError(f"the file is smaller than one volume of the Experiment.xml's layout ({volume} bytes)")
        self.data = np.memmap(self.path, dtype="<u2", mode="r", shape=(t, z + flyback, c, y, x))
        self.roles = "tzcyx"
        self.shape = {"t": t, "z": z, "c": c, "y": y, "x": x}
        self.flyback = flyback
        if layout["fps"]:
            self.fps = layout["fps"] / (z + flyback)  # volumes per second
        self.dtype = np.dtype("<u2")

    def _open_tiff(self) -> None:
        import tifffile

        try:
            self.tif = tifffile.TiffFile(self.path)
            series = self.tif.series[0]
        except Exception as e:  # tifffile raises many kinds
            raise MovieError(f"cannot read it as a TIFF ({e})")
        axes, shape = series.axes, series.shape
        if "S" in axes and shape[axes.index("S")] > 1:
            raise MovieError("colour (RGB) TIFFs are not supported")
        # roles of the axes: t (T, else a generic one: I, Q, ...), z (Z), c (C), y, x; others take their first index
        roles = []
        for a in axes:
            if a in "YX":
                roles.append(a.lower())
            elif a == "Z" or a == "C":
                roles.append(a.lower())
            elif a == "T" or (a in "IQ" and "T" not in axes and "t" not in roles):
                roles.append("t")
            else:
                roles.append("-")
        if roles[-2:] != ["y", "x"]:
            raise MovieError(f"unexpected axes {axes}: the last two should be Y and X")
        self.roles = "".join(roles)
        self.shape = {k: (shape[self.roles.index(k)] if k in self.roles else 1) for k in "tzcyx"}
        if series.dtype.kind not in "uifb":
            raise MovieError(f"unsupported pixel type {series.dtype}")
        self.dtype = _sent(series.dtype)
        offset = series.dataoffset
        # memory-mapped when the data are in one piece (as tifffile writes them), else read page by page;
        # in the file's byte order (tifffile gives the type in the machine's: hong2p writes big-endian TIFFs)
        stored = series.dtype.newbyteorder(self.tif.byteorder)
        self.data = np.memmap(self.path, dtype=stored, mode="r", offset=offset, shape=shape) if offset else None
        self.series = series
        self.axes = axes
        ij = self.tif.imagej_metadata or {}
        if ij.get("finterval"):
            self.fps = 1 / float(ij["finterval"])
        elif ij.get("fps"):
            self.fps = float(ij["fps"])

    def info(self) -> dict:
        return {"path": str(self.path), "format": "raw" if self.tif is None else "tiff", **self.shape,
                "dtype": self.dtype.name, "fps": self.fps, "warnings": self.warnings}

    def planes(self, t0: int, n: int, z: Optional[int], c: int) -> np.ndarray:
        """(n, y, x) planes from time `t0` on, of plane `z` (None: the max projection over Z) of channel `c`."""
        s = self.shape
        if not (0 <= t0 < s["t"] and n > 0 and 0 <= c < s["c"] and (z is None or 0 <= z < s["z"])):
            raise MovieError("no such plane")
        n = min(n, s["t"] - t0)
        if z is None and s["z"] == 1:
            z = 0
        if self.tif is None:  # raw: (t, z + flyback, c, y, x)
            out = self.data[t0:t0 + n, z if z is not None else slice(0, s["z"]), c]
        elif self.data is not None:
            key = tuple(_index(r, t0, n, z, c) for r in self.roles)
            kept = [r for r in self.roles if r in "tyx" or (r == "z" and z is None)]
            out = np.asarray(self.data[key]).transpose([kept.index(r) for r in "tzyx" if r in kept])
            if "t" not in kept:
                out = out[np.newaxis]
        else:
            out = self._pages(t0, n, z, c)
        if z is None:
            out = out.max(axis=1)
        return np.asarray(out).astype(self.dtype, copy=False)

    def volumes(self, t0: int, n: int, c: int = 0) -> np.ndarray:
        """(n, z, y, x) volumes from time `t0` on, of channel `c`, in the file's own number type
        (not the one sent to the browser, as `planes`'), read in one piece where the file has
        them so: for computing on a movie (`extract`), not showing it."""
        s = self.shape
        if not (0 <= t0 < s["t"] and n > 0 and 0 <= c < s["c"]):
            raise MovieError("no such volume")
        n = min(n, s["t"] - t0)
        if self.tif is None:  # raw: (t, z + flyback, c, y, x)
            return np.asarray(self.data[t0:t0 + n, :s["z"], c])
        if self.data is None:
            return self._pages(t0, n, None, c)
        key = tuple(_index(r, t0, n, None, c) for r in self.roles)
        kept = [r for r in self.roles if r in "tzyx"]
        out = np.asarray(self.data[key]).transpose([kept.index(r) for r in "tzyx" if r in kept])
        if "t" not in kept:
            out = out[np.newaxis]
        return out if "z" in kept else out[:, np.newaxis]

    def _pages(self, t0: int, n: int, z: Optional[int], c: int) -> np.ndarray:
        """Like `planes`, from the TIFF's pages one at a time (data not in one piece)."""
        dims = [self.series.shape[i] for i, r in enumerate(self.roles) if r not in "yx"]
        roles = [r for r in self.roles if r not in "yx"]
        zs = range(self.shape["z"]) if z is None else [z]
        frames = []
        with self.lock:
            for t in range(t0, t0 + n):
                vol = []
                for zz in zs:
                    idx = [t if r == "t" else zz if r == "z" else c if r == "c" else 0 for r in roles]
                    vol.append(self.series.asarray(key=int(np.ravel_multi_index(idx, dims)) if dims else 0))
                frames.append(vol)
        out = np.array(frames)
        return out if z is None else out[:, 0]

    def close(self) -> None:
        if self.tif is not None:
            self.tif.close()


def _index(role: str, t0: int, n: int, z: Optional[int], c: int):
    """What to take along an axis of a TIFF's data, by its role (see `Movie.planes`)."""
    if role == "t":
        return slice(t0, t0 + n)
    if role == "z":
        return slice(None) if z is None else z
    if role == "c":
        return c
    return slice(None) if role in "yx" else 0


def _sent(dtype: np.dtype) -> np.dtype:
    """The little-endian type planes are sent as (to a JavaScript typed array)."""
    if dtype.kind == "b":
        return np.dtype("u1")
    if dtype.itemsize == 8:
        return np.dtype("<f4")
    return dtype.newbyteorder("<")


_open: OrderedDict[tuple, Movie] = OrderedDict()
_open_lock = threading.Lock()


def open_movie(path: Path) -> Movie:
    """The movie at `path`, kept open (the last `CACHED` used, until changed on disk)."""
    st = os.stat(path)
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _open_lock:
        if key in _open:
            _open.move_to_end(key)
            return _open[key]
    movie = Movie(path)
    with _open_lock:
        _open[key] = movie
        while len(_open) > CACHED:
            _open.popitem(last=False)[1].close()
    return movie


# -------------------------------------------------------------------------- ROIs


def read_rois(path: Path) -> list[dict]:
    """The ImageJ ROIs in a RoiSet.zip (or one .roi): ``name``, ``type``, ``z``, ``position``
    (0-based: the Z plane of a hyperstack, the stack position otherwise; None if not set),
    ``color`` ('#rrggbb', None if not set), ``closed`` and ``paths`` (lists of [x, y], in
    pixels, from the corner of the image, as ImageJ draws them)."""
    import roifile

    try:
        rois = roifile.ImagejRoi.fromfile(path)
    except Exception as e:
        raise MovieError(f"cannot read the ImageJ ROIs ({e})")
    rois = rois if isinstance(rois, list) else [rois]
    closed = {roifile.ROI_TYPE.POLYGON, roifile.ROI_TYPE.RECT, roifile.ROI_TYPE.OVAL, roifile.ROI_TYPE.FREEHAND,
              roifile.ROI_TYPE.TRACED}
    out = []
    for i, r in enumerate(rois):
        color = r.stroke_color
        paths = [np.round(np.asarray(p, dtype=float), 2).tolist() for p in r.coordinates(multi=True) if len(p)]
        out.append({
            "name": r.name or f"{i + 1}", "type": r.roitype.name.lower(),
            "z": r.z_position - 1 if r.z_position > 0 else None,
            "position": r.position - 1 if r.position > 0 else None,
            "color": f"#{color[1:].hex()}" if color and len(color) == 4 and any(color) else None,
            "closed": r.roitype in closed or r.composite, "paths": paths,
        })
    return out


def roi_kind(name: str, names: frozenset[str] | set[str] = glom.GLOMERULI) -> str:
    """An ImageJ ROI's name as a glomerulus's: 'glomerulus' if it is one of `names` (by
    default the glomeruli an upload accepts) or several of them that could not be told apart
    (``VA1d/VA1v``, as an upload accepts them); 'tentative' if it is but for a ``_t<n>``
    suffix (``DA1_t0``: Sam's mark of a name not certain, which an upload removes); else
    'other' (numbered ROIs, and names ending in ``+`` or ``?``, among others)."""
    def named(n: str) -> bool:
        return n in names or all(p in names for p in n.split("/"))

    if named(name):
        return "glomerulus"
    bare = ROI_SUFFIX.sub("", name)
    return "tentative" if bare != name and named(bare) else "other"


def read_trials(path: Path) -> list[dict]:
    """The trials of a movie in an al_analysis `TRIALS` file: ``start``, ``onset`` (its first
    frame with the odor) and ``end`` (its last frame), as time points of the movie from 0,
    and ``odor`` (as al_analysis names it, e.g. ``ea @ -4.2``; None if not given)."""
    try:
        if os.path.getsize(path) > tensor.MAX_FILE_BYTES:
            raise MovieError(f"the trials file is larger than {tensor.MAX_FILE_BYTES >> 20} MB")
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise MovieError(f"cannot read the trials ({e})")
    if not isinstance(data, list):
        raise MovieError("the trials file is not a list of trials")
    out = []
    for i, t in enumerate(data):
        try:
            start, onset, end = (int(t[k]) for k in ("start_frame", "first_odor_frame", "end_frame"))
        except (TypeError, KeyError, ValueError):
            raise MovieError(f"trial {i + 1} has no whole start_frame, first_odor_frame and end_frame")
        odor = t.get("odors")
        if odor is None and t.get("name") is not None:
            odor = f"{t['name']} @ {t['log10_conc']}" if t.get("log10_conc") is not None else t["name"]
        out.append({"start": start, "onset": onset, "end": end, "odor": None if odor is None else str(odor)})
    return out


def is_movie(path: Path) -> bool:
    return path.suffix.lower() in (*TIFF_SUFFIXES, ".raw")


def chunk(movie: Movie, n: int, z: Optional[int]) -> int:
    """`n`, cut down so that a request for planes stays under `MAX_PLANES_BYTES`."""
    s = movie.shape
    per = s["y"] * s["x"] * movie.dtype.itemsize * (s["z"] if z is None else 1)
    return max(1, min(n, MAX_PLANES_BYTES // max(per, 1)))
