"""Movies of a dataset's flies on tensor: finding them, and reading the planes shown.

Motion-corrected movies are TIFFs in the folders of a fly's recordings under the dataset's
motion-corrected data folder (as al_analysis writes them: ``mocorr.tif``, else
``flipped.tif`` or ``raw.tif``, though any TIFF is listed), with the ImageJ ROIs drawn on
them (``RoiSet.zip``, next to the movie or in a folder above it). Raw movies are ThorImage
``.raw`` files, with their ``Experiment.xml`` next to them. Planes are read on the server,
only those shown (Z max projections computed here too), and sent to the browser as plain
little-endian numbers, so it never has to parse a TIFF or read a whole movie.
"""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import tensor, thorimage

TIFF_SUFFIXES = (".tif", ".tiff")
PREFERRED = ("mocorr.tif", "flipped.tif", "raw.tif")  # al_analysis' names, most processed first
ROISET = "RoiSet.zip"
DEPTH = 3  # folders scanned below a fly's folder (or the data folder, if no fly folder is found)
MAX_MOVIES = 200
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


def find(folder: Path, kind: str, flies: list[dict]) -> list[dict]:
    """Movies under one of a dataset's data folders (`kind`: 'raw' or 'mocorr').

    Looked for in its flies' folders (``<folder>/<date>/<fly>/``, as for the upload
    lookup) and the folders below them; if none is there (the folder may be a single
    fly's or recording's), in `folder` and the folders below it.

    Returns:
        dicts with ``path`` (on the server), ``label`` (from `folder`), ``format``
        ('tiff' or 'raw'), ``size``, ``fly`` (its date and number, if found by them),
        ``recording`` (the folder in the fly's it is in, None if it is in the fly's
        itself or not found by fly) and ``rois`` (the RoiSet.zip drawn on it, if any, for a
        TIFF).
    """
    tops = [(d, f) for f in flies for d in tensor.fly_dirs(folder, f["date"], f["fly_num"])] or [(folder, None)]
    out: list[dict] = []
    for top, fly in tops:
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
                movies = [e for e in files if e.name.lower().endswith(".raw") and "Experiment.xml" in names]
            for e in movies:
                try:
                    size = e.stat().st_size
                except OSError:
                    continue
                below = Path(e.path).relative_to(top).parts
                out.append({
                    "path": e.path, "label": Path(e.path).relative_to(folder).as_posix(),
                    "format": "raw" if kind == "raw" else "tiff", "size": size,
                    "fly": {"date": fly["date"], "fly_num": fly["fly_num"]} if fly else None,
                    "recording": below[0] if fly and len(below) > 1 else None,
                    "rois": str(rois[d]) if kind == "mocorr" and rois.get(d) else None,
                })
                if len(out) >= MAX_MOVIES:
                    return out
    return out


def unmatched(raw: list[dict], mocorr: list[dict]) -> dict[str, list[dict]]:
    """The recordings with movies in one of a dataset's data folders and none in the other
    (`find`'s movies of each: by fly and the name of the recording's folder, which
    al_analysis takes from the raw data's), but for anatomical ones (`tensor.ANATOMICAL`,
    never motion-corrected) and movies not found by fly. Each movie of one is marked
    ``unmatched`` (and the rest not).

    Returns:
        by kind ('raw', 'mocorr'): its recordings not in the other, each with ``date``,
        ``fly_num``, ``recording`` and ``n_movies``.
    """
    def recordings(found: list[dict]) -> dict[tuple, list[dict]]:
        out: dict[tuple, list[dict]] = {}
        for m in found:
            m["unmatched"] = False
            if m["fly"] and not (m["recording"] and tensor.ANATOMICAL.fullmatch(m["recording"])):
                out.setdefault((m["fly"]["date"], m["fly"]["fly_num"], m["recording"]), []).append(m)
        return out

    by_kind = {"raw": recordings(raw), "mocorr": recordings(mocorr)}
    result = {}
    for kind, other in (("raw", "mocorr"), ("mocorr", "raw")):
        result[kind] = []
        for (date, fly_num, recording), found in by_kind[kind].items():
            if (date, fly_num, recording) not in by_kind[other]:
                result[kind].append({"date": date, "fly_num": fly_num, "recording": recording, "n_movies": len(found)})
                for m in found:
                    m["unmatched"] = True
    return result


def _natural(name: Optional[str]) -> list:
    """`name` to sort by, numbers in it by their value (megamat_2 before megamat_10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name or "")]


def side_by_side(raw: list[dict], mocorr: list[dict]) -> list[dict]:
    """Every recording of a dataset's flies in its raw and motion-corrected data folders,
    paired as `unmatched` pairs them (`find`'s movies of each found by fly), in the order
    they were recorded: by fly, then by when its raw recording says it was made (its
    Experiment.xml's date; if one of the fly's does not say, when they were written); one
    with no raw data after the one before it by name (megamat_3 after megamat_2), there
    being no time it was recorded.

    Returns:
        rows with ``date``, ``fly_num``, ``recording`` (None: the fly's folder itself),
        ``raw`` and ``mocorr`` (the names of its movies in each, None if it has none
        there), ``recorded`` (when it was recorded, as its Experiment.xml says:
        YYYY-MM-DD HH:MM:SS, the acquisition computer's time), ``time`` (when its
        Experiment.xml was written, in seconds since 1970; both None without raw data),
        ``matched`` and ``anatomical``.
    """
    rows: dict[tuple, dict] = {}
    for kind, found in (("raw", raw), ("mocorr", mocorr)):
        for m in found:
            if not m["fly"]:
                continue
            key = (m["fly"]["date"], m["fly"]["fly_num"], m["recording"])
            row = rows.setdefault(key, {"date": key[0], "fly_num": key[1], "recording": key[2], "raw": None, "mocorr": None,
                                        "recorded": None, "time": None})
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
    for fly in sorted({(r["date"], r["fly_num"]) for r in rows.values()}):
        mine = [r for r in rows.values() if (r["date"], r["fly_num"]) == fly]
        # by when they say they were recorded, else (none of them saying) when their Experiment.xml was written
        by = "recorded" if all(r["recorded"] for r in mine if r["time"] is not None) else "time"
        order = sorted((r for r in mine if r["time"] is not None), key=lambda r: (r[by], _natural(r["recording"])))
        for r in sorted((r for r in mine if r["time"] is None), key=lambda r: _natural(r["recording"])):
            before = [o for o in order if _natural(o["recording"]) <= _natural(r["recording"])]
            order.insert(order.index(max(before, key=lambda o: _natural(o["recording"]))) + 1 if before else 0, r)
        out += order
    for r in out:
        r.update(matched=r["raw"] is not None and r["mocorr"] is not None,
                 anatomical=bool(r["recording"] and tensor.ANATOMICAL.fullmatch(r["recording"])))
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
        # memory-mapped when the data are in one piece (as tifffile writes them), else read page by page
        self.data = np.memmap(self.path, dtype=series.dtype, mode="r", offset=offset, shape=shape) if offset else None
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


def is_movie(path: Path) -> bool:
    return path.suffix.lower() in (*TIFF_SUFFIXES, ".raw")


def chunk(movie: Movie, n: int, z: Optional[int]) -> int:
    """`n`, cut down so that a request for planes stays under `MAX_PLANES_BYTES`."""
    s = movie.shape
    per = s["y"] * s["x"] * movie.dtype.itemsize * (s["z"] if z is None else 1)
    return max(1, min(n, MAX_PLANES_BYTES // max(per, 1)))
