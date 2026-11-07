"""Data to train a model to draw glomerulus ROIs (al-diag-seg,
https://github.com/tom-f-oconnell/al-diag-seg): what is on tensor of each fly of a glomeruli
dataset with a diagnostic panel, and a download of it (AL segmentation tab).

A fly is exported if its folder in the dataset's motion-corrected data (as al_analysis lays
it out) has:

- diagnostic recordings: recording folders with al_analysis' `TRIALMEAN` (odor, Z, Y, X: the
  mean dF/F of each odor over its trials, the odors in the order they were first presented)
  and `TRIALS` (each trial's odor), more than `DIAGNOSTIC_SHARE` of whose odors are of a
  diagnostic panel of the dataset (matched through the odor table);
- its ROIs: a `ROISET` (al_analysis' is in its first diagnostic recording, drawn on the
  fly's `CONCAT`, and the other recordings' are links to it);
- its average image: al_analysis' `CONCAT_AVG` (the mean of `CONCAT` over time), else that
  mean, computed here from `CONCAT` and kept (`config.ROI_TRAINING_DIR`), so each is computed once;

and its folder in the raw data has the Experiment.xml of at least one of its diagnostic
recordings (their voxel sizes). Its anatomical stacks (`tensor.ANATOMICAL` folders of its
raw data: higher resolution stacks without odors, which some flies have), which al-diag-seg
registers flies to each other through, go in too unless asked not to: as recorded, so not
flipped as al_analysis flips right ALs' functional data.

The zip has each file where it is on tensor (its path from the top of tensor), the average
image computed here where al_analysis writes it, and `MANIFEST`, saying which file is what (each
recording's odors as in the odor table, so the same odor is named alike in every fly).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from . import config, odormeta, sandbox, tensor
from .movies import _natural, roi_kind
from .glomeruli import load_glomeruli
from .ingest import ROI_SUFFIX, is_diagnostic_panel
from .metadata import is_kc
from .odors import OdorFormatError, format_conc, parse_odor

TRIALMEAN = "trialmean_dff.tif"
TRIALS = "trial_frames_and_odors.json"
ROISET = "RoiSet.zip"
CONCAT = "mocorr_concat.tif"
CONCAT_AVG = "mocorr_concat_avg.tif"
XML = "Experiment.xml"
MANIFEST = "roi_training.json"
# a recording is diagnostic if more than this share of its odors are of a diagnostic panel
DIAGNOSTIC_SHARE = 0.5
# the files of an anatomical stack exported: its metadata, and its images as ThorImage writes them
STACK_FILE = re.compile(r"Experiment\.xml|ChanA_\d+_\d+_\d+_\d+\.tif|Image_\d+_\d+\.raw")
CHUNK = 1 << 20
# files read at once, ahead of being sent: each file on tensor takes a while to start reading
# (an anatomical stack's ~1000 files: 45 s one after another, 3 s eight at a time)
PREFETCH = 8
SMALL = 4 << 20  # files read whole when read ahead; larger ones are read as they are sent


def _size(p: Path) -> int:
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def _is_file(p: Path) -> bool:
    return sandbox.can_read(p) and os.path.isfile(p)


def _tiff_shape(p: Path) -> Optional[tuple[int, ...]]:
    """The shape of a TIFF's first series (from its header only), or None if it cannot be read."""
    import tifffile

    if not sandbox.can_read(p):
        return None
    try:
        with tifffile.TiffFile(p) as t:
            return tuple(int(n) for n in t.series[0].shape)
    except Exception:  # noqa: BLE001 (any file that is not a readable TIFF)
        return None


def _key(label: Optional[str], conc: Optional[float]) -> Optional[tuple[str, str]]:
    return (label.casefold(), format_conc(conc)) if label and conc is not None else None


def recording_odors(path: Path, index: odormeta.OdorIndex) -> Optional[list[dict]]:
    """The odors of a recording, in the order al_analysis' `TRIALMEAN` has them (each odor's
    first trial), from its `TRIALS`: each with ``recorded`` (as there), ``odor`` (the odor
    table's name of it @ its concentration, else as recorded), ``odor_id`` and ``keys`` (what
    it may be matched to a dataset's odors by). None if the file cannot be read as one."""
    try:
        trials = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(trials, list) or not all(isinstance(t, dict) and isinstance(t.get("odors"), str) for t in trials):
        return None
    out: dict[str, dict] = {}
    for t in trials:
        s = t["odors"]
        if s in out:
            continue
        name, abbrev, conc, cid = t.get("name"), t.get("abbrev"), t.get("log10_conc"), t.get("cid")
        if not isinstance(conc, (int, float)) or not isinstance(name, str):
            try:
                o = parse_odor(s, strict=False)
                name, conc = o.name, o.log10_conc
            except OdorFormatError:
                name, conc = None, None
        try:
            cid = int(cid) if cid not in (None, "") else None
        except (TypeError, ValueError):
            cid = None
        labels = [x for x in (name, abbrev if isinstance(abbrev, str) else None) if x]
        odor_id = None
        if labels:
            odor_id, _ = index.match_chemical({"cid": cid, "name": name, "abbrev": abbrev if isinstance(abbrev, str) else None},
                                              labels[0])
        keys = {k for k in (_key(x, conc) for x in labels) if k}
        if odor_id is not None and conc is not None:
            keys.add((odor_id, format_conc(conc)))
        out[s] = {"recorded": s, "odor_id": odor_id, "keys": keys,
                  "odor": f"{index.by_id[odor_id]['name']} @ {format_conc(conc)}" if odor_id is not None and conc is not None else s}
    return list(out.values())


def diagnostic_keys(ds: dict, trials: list[dict], index: odormeta.OdorIndex, yaml_odors: list) -> set:
    """What the odors of a dataset's diagnostic panels may be matched by: each one's label
    and odor table odor (if known), with its concentration."""
    keys = set()
    for r in odormeta.dataset_odors(ds, trials, index, yaml_odors):
        if not is_diagnostic_panel(r["panel"]) or r["log10_conc"] is None:
            continue
        keys.add(_key(r["label"], r["log10_conc"]))
        if r["odor_id"] is not None:
            keys.add((r["odor_id"], format_conc(r["log10_conc"])))
    return keys


def candidates(db) -> list[dict]:
    """The datasets whose flies may be exported: of glomeruli, with a diagnostic panel."""
    return [d for d in sorted(db.list_datasets(), key=lambda d: d["id"])
            if " ".join((d["coding_unit"] or "").split()).casefold() == "glomeruli" and not is_kc(d["coding_unit"])
            and any(is_diagnostic_panel(p["panel"]) for p in d.get("panels") or [])]


def fly_panels(db, ds_id: int) -> dict[tuple[str, int], set[str]]:
    """The panels (as shown) each fly of a dataset, by its date and number, has a response in."""
    m = db.get_matrix_arrays(ds_id)
    if m is None:
        return {}
    cols: dict[str, list[int]] = {}
    for j, t in enumerate(m["trials"]):
        if t["panel"] is not None:
            cols.setdefault(t["panel"], []).append(j)
    rows: dict[tuple[str, int], list[int]] = {}
    for i, u in enumerate(m["units"]):
        if u["date"] is not None:
            rows.setdefault((u["date"], u["fly_num"]), []).append(i)
    finite = np.isfinite(m["values"])
    return {fly: {p for p, js in cols.items() if finite[np.ix_(idx, js)].any()} for fly, idx in rows.items()}


def roi_glomeruli(path: Path) -> tuple[Optional[dict], Optional[str]]:
    """The glomeruli of a fly's `ROISET`, as al-diag-seg trains on them: ROIs named as a
    glomerulus (or several joined by /: `movies.roi_kind`), and tentative ones (``DA1_t0``) as
    that glomerulus; not numbered ROIs, nor names ending in + or ?.

    Returns:
        ({``glomeruli``: their names, ``tentative``: those of them only named tentatively},
        None), or (None, why the file could not be read).
    """
    import roifile

    if not sandbox.can_read(path):
        return None, "outside the folders the server may read"
    try:
        rois = roifile.ImagejRoi.fromfile(path)
    except Exception as e:  # noqa: BLE001 (roifile raises many kinds)
        return None, f"its {ROISET} could not be read ({e})"
    plain, tentative = set(), set()
    for r in rois if isinstance(rois, list) else [rois]:
        kind = roi_kind(r.name or "")
        if kind == "glomerulus":
            plain.add(r.name)
        elif kind == "tentative":
            tentative.add(ROI_SUFFIX.sub("", r.name))
    return {"glomeruli": sorted(plain | tentative), "tentative": sorted(tentative - plain)}, None


def _roiset(fly_dir: Path, diagnostic: list[Path], others: list[Path]) -> Optional[Path]:
    """A fly's RoiSet.zip: in its folder, else in its diagnostic recordings' (one that is
    not a link to another first, as al_analysis draws them in one and links the rest to it),
    else in another recording's."""
    if _is_file(fly_dir / ROISET):
        return fly_dir / ROISET
    for group in (diagnostic, others):
        found = [d / ROISET for d in group if _is_file(d / ROISET)]
        if found:
            return next((p for p in found if not p.is_symlink()), found[0])
    return None


def _average(fly_dir: Path, grid: tuple[int, ...]) -> tuple[Optional[dict], Optional[str]]:
    """A fly's average image ({``path``: al_analysis' `CONCAT_AVG`, or ``source``: the `CONCAT`
    to average}), or why there is none."""
    concat, avg = fly_dir / CONCAT, fly_dir / CONCAT_AVG
    shape = _tiff_shape(concat) if _is_file(concat) else None
    concat_ok = shape is not None and len(shape) == 4 and shape[1:] == grid
    if _is_file(avg) and _tiff_shape(avg) == grid:
        try:
            stale = concat_ok and os.path.getmtime(concat) > os.path.getmtime(avg)
        except OSError:
            stale = False
        if not stale:
            return {"path": avg, "computed": False}, None
    if concat_ok:
        return {"source": concat, "computed": True}, None
    return None, f"no {CONCAT} (or {CONCAT_AVG}) of its {grid[0]} × {grid[1]} × {grid[2]} planes in its folder, for its average image"


def _stacks(raw_fly: Optional[Path]) -> list[dict]:
    """A fly's anatomical stacks: ``dir``, ``files`` (those exported) and their ``bytes``."""
    out = []
    for d in (tensor._subdirs(raw_fly) if raw_fly else []):
        if not tensor.ANATOMICAL.fullmatch(d.name) or not _is_file(d / XML):
            continue
        sizes = {}
        for e in tensor._entries(d):
            if STACK_FILE.fullmatch(e.name):
                try:
                    st = e.stat()
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    sizes[Path(e.path)] = st.st_size
        if any(p.name != XML for p in sizes):
            out.append({"dir": d, "files": list(sizes), "bytes": sum(sizes.values())})
    return out


def _fly(date: str, fly_num: int, mocorr: Path, raw: Optional[Path], keys: set,
         index: odormeta.OdorIndex) -> tuple[Optional[dict], Optional[str]]:
    """What there is to export of one fly (see the module's docstring), or why it is left out."""
    dirs = tensor.fly_dirs(mocorr, date, fly_num)
    if not dirs:
        return None, "no folder of it in the motion-corrected data"
    fly_dir = dirs[0]
    recs, others, problems = [], [], []
    for d in sorted(tensor._subdirs(fly_dir), key=lambda p: _natural(p.name)):
        if not (_is_file(d / TRIALMEAN) and _is_file(d / TRIALS)):
            others.append(d)
            continue
        odors = recording_odors(d / TRIALS, index)
        if not odors:
            problems.append(f"{d.name}: its {TRIALS} could not be read")
            others.append(d)
            continue
        if sum(1 for o in odors if o["keys"] & keys) <= DIAGNOSTIC_SHARE * len(odors):
            others.append(d)
            continue
        shape = _tiff_shape(d / TRIALMEAN)
        shape = (1, *shape) if shape is not None and len(shape) == 3 else shape
        if shape is None or len(shape) != 4:
            problems.append(f"{d.name}: its {TRIALMEAN} could not be read as (odor, Z, Y, X)")
        elif shape[0] != len(odors):
            problems.append(f"{d.name}: its {TRIALMEAN} has {shape[0]} odors, but its {TRIALS} {len(odors)}")
        else:
            recs.append({"dir": d, "odors": odors, "shape": shape})
    if not recs:
        return None, "; ".join(problems) or "no diagnostic recording (with its odors' mean dF/F, trialmean_dff.tif) in its folder"
    grids = {r["shape"][1:] for r in recs}
    if len(grids) > 1:
        return None, f"its diagnostic recordings are of different planes ({', '.join(' × '.join(map(str, g)) for g in sorted(grids))})"
    grid = grids.pop()
    roiset = _roiset(fly_dir, [r["dir"] for r in recs], others)
    if roiset is None:
        return None, "no RoiSet.zip (its ROIs) in its folder or its recordings'"
    average, why = _average(fly_dir, grid)
    if average is None:
        return None, why
    raw_dirs = tensor.fly_dirs(raw, date, fly_num) if raw else []
    raw_fly = raw_dirs[0] if raw_dirs else None
    for r in recs:
        xml = raw_fly / r["dir"].name / XML if raw_fly else None
        r["xml"] = xml if xml and _is_file(xml) else None
    if not any(r["xml"] for r in recs):
        return None, ("no folder of it in the raw data" if raw_fly is None else
                      f"no {XML} of its diagnostic recordings in its raw data folder (their voxel sizes)")
    return {"date": date, "fly_num": fly_num, "dir": fly_dir, "raw_dir": raw_fly, "recordings": recs, "grid": grid,
            "roiset": roiset, "average": average, "stacks": _stacks(raw_fly), "problems": problems}, None


def inventory(db, root: Path, bases: list[Path]) -> dict:
    """Every fly to export (one per folder, with the datasets it is of), and those left out.

    Returns:
        dict with ``flies`` (see `_fly`, each with ``datasets``: of each, ``id``, ``title``
        and ``uploader``, and ``panels``: those it has a response in, in them) and ``skipped``
        (``dataset``, ``date``, ``fly_num`` (None for all of a dataset's), ``why``).
    """
    index = odormeta.OdorIndex(db.list_odors())
    flies: dict[Path, dict] = {}
    skipped = []
    for d in candidates(db):
        ds = db.get_dataset(d["id"])
        about = {"id": ds["id"], "title": ds["title"], "uploader": ds["uploader_name"]}
        found = {k: tensor.find_data(ds["metadata"].get(f"{k}_data_path"), root, bases) for k in ("raw", "mocorr")}
        if not (found["mocorr"]["status"] == "ok" and found["mocorr"]["is_dir"]):
            skipped.append({"dataset": about, "date": None, "fly_num": None,
                            "why": f"its motion-corrected data folder: {found['mocorr']['error'] or 'not a folder'}"})
            continue
        raw = Path(found["raw"]["path"]) if found["raw"]["status"] == "ok" and found["raw"]["is_dir"] else None
        keys = diagnostic_keys(ds, db.get_trials(ds["id"]), index, odormeta.dataset_yaml_odors(ds, config.DATASETS_DIR))
        panels = fly_panels(db, ds["id"])
        for f in ds["summary"].get("flies") or []:
            if f.get("date") is None or f.get("fly_num") is None:
                continue
            fly, why = _fly(f["date"], f["fly_num"], Path(found["mocorr"]["path"]), raw, keys, index)
            if fly is None:
                skipped.append({"dataset": about, "date": f["date"], "fly_num": f["fly_num"], "why": why})
            elif fly["dir"] in flies:
                flies[fly["dir"]]["datasets"].append(about)
                flies[fly["dir"]]["panels"] |= panels.get((f["date"], f["fly_num"]), set())
            else:
                flies[fly["dir"]] = {**fly, "datasets": [about], "panels": set(panels.get((f["date"], f["fly_num"]), set()))}
    return {"flies": list(flies.values()), "skipped": skipped}


def _arcname(p: Path, root: Path, bases: list[Path]) -> str:
    """Where a file goes in the zip: its path from the top of tensor (or from a folder set in
    the Admin tab, under its name)."""
    for top, prefix in ((root, ""), *((b, f"{b.name}/") for b in bases)):
        try:
            return prefix + p.relative_to(top).as_posix()
        except ValueError:
            continue
    return p.name


def computed_average(source: Path) -> Path:
    """The mean of a `CONCAT` over time ((Z, Y, X) float32, as al_analysis writes its
    `CONCAT_AVG`), computed once and kept in `config.ROI_TRAINING_DIR` (by the file's path,
    size and time)."""
    import tifffile

    st = os.stat(source)
    key = hashlib.sha256(f"{source.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:32]
    out = config.ROI_TRAINING_DIR / f"{key}.tif"
    if out.is_file():
        return out
    config.ROI_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(source) as t:
        s = t.series[0]
        if s.dataoffset is not None:  # (one block of pixels: read a few volumes at a time)
            movie = tifffile.memmap(source, mode="r")
            total = np.zeros(movie.shape[1:], np.float64)
            for i in range(0, movie.shape[0], 50):
                total += movie[i:i + 50].sum(axis=0, dtype=np.float64)
            n = movie.shape[0]
            del movie
        else:
            movie = s.asarray()
            total, n = movie.sum(axis=0, dtype=np.float64), movie.shape[0]
    tmp = out.with_name(f"{key}.{os.getpid()}.tmp")
    tifffile.imwrite(tmp, (total / n).astype(np.float32), imagej=True, metadata={"axes": "ZYX"})
    tmp.replace(out)
    return out


def manifest(found: dict, root: Path, bases: list[Path], stacks: bool, failed: dict[str, str]) -> dict:
    """`MANIFEST`: what the zip has of each fly (paths as in the zip, from its top folder)."""
    arc = lambda p: _arcname(p, root, bases) if p else None  # noqa: E731
    flies = []
    for f in found["flies"]:
        mine = {arc(p) for p, _ in files({"flies": [f]}, stacks)}
        flies.append({
            "fly": f"{f['date']}/{f['fly_num']}", "date": f["date"], "fly_num": f["fly_num"], "datasets": f["datasets"],
            "dir": arc(f["dir"]), "raw_dir": arc(f["raw_dir"]), "shape": list(f["grid"]),
            "average": arc(f["dir"] / CONCAT_AVG),
            "average_from": f"the mean of {CONCAT} over time, computed for this export" if f["average"]["computed"] else "al_analysis",
            "roiset": arc(f["roiset"]),
            "recordings": [{"name": r["dir"].name, "responses": arc(r["dir"] / TRIALMEAN), "trials": arc(r["dir"] / TRIALS),
                            "experiment_xml": arc(r["xml"]), "odors": [o["odor"] for o in r["odors"]],
                            "as_recorded": [o["recorded"] for o in r["odors"]]} for r in f["recordings"]],
            "stacks": [{"name": s["dir"].name, "dir": arc(s["dir"]), "experiment_xml": arc(s["dir"] / XML),
                        "orientation": "as recorded (al_analysis mirrors right ALs' functional data left-right, but not these)"}
                       for s in f["stacks"]] if stacks else [],
            "failed": {n: why for n, why in failed.items() if n in mine},
        })
    return {
        "format": "olfdb ROI training export", "version": 1,
        "exported_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "about": ("Each file is where it is on tensor (its path from the top of tensor). Per fly: its diagnostic "
                  f"recordings' {TRIALMEAN} (odor, Z, Y, X: mean dF/F of each odor, in the order of `odors`, named as in "
                  f"the odor table) and {TRIALS}, its {ROISET}, its average image ({CONCAT_AVG}: the mean of {CONCAT}), the "
                  f"{XML} of its recordings in its raw data, and its anatomical stacks if asked for."),
        # the glomerulus names uploads accept (config/glomeruli.txt): ROIs named as one of these, or several joined
        # by /, are glomeruli; with a _t<n> suffix, tentative ones (movies.roi_kind); others numbered or not identified
        "glomeruli": load_glomeruli(),
        "stacks": stacks, "flies": flies,
        "skipped": [{**s, "dataset": s["dataset"]["id"]} for s in found["skipped"]],
    }


class _Sink(io.RawIOBase):
    """What a zip being written is written to: kept until taken, to be sent (not seekable,
    so zipfile writes each file's sizes after it)."""

    def __init__(self):
        self.chunks: list[bytes] = []

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        self.chunks.append(bytes(b))
        return len(b)

    def take(self) -> bytes:
        out = b"".join(self.chunks)
        self.chunks.clear()
        return out


def files(found: dict, stacks: bool) -> list[tuple[Path, Callable[[], Path]]]:
    """(where each file goes, as on tensor; a function giving the file to send), of every fly."""
    out = []
    for f in found["flies"]:
        a = f["average"]
        out.append((f["dir"] / CONCAT_AVG, (lambda p=a["path"]: p) if not a["computed"] else (lambda s=a["source"]: computed_average(s))))
        out.append((f["roiset"], lambda p=f["roiset"]: p))
        for r in f["recordings"]:
            for p in (r["dir"] / TRIALMEAN, r["dir"] / TRIALS, r["xml"]):
                if p is not None:
                    out.append((p, lambda p=p: p))
        for s in f["stacks"] if stacks else []:
            out += [(p, lambda p=p: p) for p in s["files"]]
    return out


def _fetch(get: Callable[[], Path]) -> tuple[Path, Optional[bytes]]:
    """A file to send (`files`), read whole if it is small (else None, to read as it is sent)."""
    src = get()
    if not sandbox.can_read(src):
        raise OSError("outside the folders the server may read")
    if os.path.getsize(src) > SMALL:
        return src, None
    with open(src, "rb") as f:
        return src, f.read()


def stream_zip(found: dict, root: Path, bases: list[Path], stacks: bool, top: str) -> Iterator[bytes]:
    """The zip (stored: TIFFs hardly compress), as it is written: every file of `files`
    under `top` (`PREFETCH` read ahead at once), then `MANIFEST` (saying which could not be
    read, if any)."""
    sink = _Sink()
    failed: dict[str, str] = {}
    when = dt.datetime.now().timetuple()[:6]
    entries = iter(files(found, stacks))
    pool = ThreadPoolExecutor(PREFETCH)
    ahead: deque = deque()

    def read_ahead():
        while len(ahead) < 2 * PREFETCH and (e := next(entries, None)) is not None:
            ahead.append((e[0], pool.submit(_fetch, e[1])))

    try:
        with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            read_ahead()
            while ahead:
                dest, future = ahead.popleft()
                read_ahead()
                name = _arcname(dest, root, bases)
                try:
                    src, data = future.result()
                    with zf.open(zipfile.ZipInfo(f"{top}/{name}", when), "w", force_zip64=True) as w:
                        if data is not None:
                            w.write(data)
                        else:
                            with open(src, "rb") as fin:
                                while chunk := fin.read(CHUNK):
                                    w.write(chunk)
                                    yield sink.take()
                except Exception as e:  # noqa: BLE001 (any file that cannot be read: said so in the manifest)
                    failed[name] = str(e) or type(e).__name__
                yield sink.take()
            zf.writestr(zipfile.ZipInfo(f"{top}/{MANIFEST}", when),
                        json.dumps(manifest(found, root, bases, stacks, failed), indent=1))
        yield sink.take()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def summary(found: dict, root: Path, bases: list[Path]) -> dict:
    """What the AL segmentation tab shows: each fly to export and those left out, with sizes;
    and of each fly, its glomeruli (`roi_glomeruli`: ``glomeruli`` and ``tentative``, None
    with ``roi_error`` if its ROIs could not be read) and ``panels``: those but diagnostic ones."""
    with ThreadPoolExecutor(PREFETCH) as pool:  # (each file on tensor takes a while to start reading)
        rois = list(pool.map(lambda f: roi_glomeruli(f["roiset"]), found["flies"]))
    flies = []
    for f, (glomeruli, roi_error) in zip(found["flies"], rois):
        size = sum(_size(p) for p, get in files({"flies": [f]}, False) if not (p.name == CONCAT_AVG and f["average"]["computed"]))
        if f["average"]["computed"]:
            z, y, x = f["grid"]
            size += z * y * x * 4
        flies.append({
            "date": f["date"], "fly_num": f["fly_num"], "datasets": f["datasets"],
            "dir": _arcname(f["dir"], root, bases), "raw_dir": _arcname(f["raw_dir"], root, bases) if f["raw_dir"] else None,
            "roiset": _arcname(f["roiset"], root, bases), "shape": list(f["grid"]),
            "average": "computed" if f["average"]["computed"] else "al_analysis",
            "recordings": [{"name": r["dir"].name, "n_odors": len(r["odors"]), "xml": r["xml"] is not None} for r in f["recordings"]],
            "stacks": [{"name": s["dir"].name, "bytes": s["bytes"]} for s in f["stacks"]],
            "bytes": size, "problems": f["problems"],
            "glomeruli": glomeruli and glomeruli["glomeruli"], "tentative": glomeruli and glomeruli["tentative"],
            "roi_error": roi_error, "panels": sorted(p for p in f["panels"] if not is_diagnostic_panel(p)),
        })
    return {"flies": flies, "skipped": found["skipped"], "bytes": sum(f["bytes"] for f in flies),
            "stack_bytes": sum(s["bytes"] for f in flies for s in f["stacks"])}
