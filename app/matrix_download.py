"""Downloads of a matrix other than a dataset's: a reference dataset (Reference data tab) or a
model run (Models tab), each as its original files (``files``: a zip of them as added, with a
README) or as its matrix (``wide``: one CSV of its values, each row's metadata in its first
columns and each column's in its header lines, a level each, as pandas writes a table whose
index and columns are a MultiIndex).

A matrix may be downloaded as shown: of the rows and columns given (``rows``, ``cols``: their
indices in the matrix as read, in the order shown), and ``transpose``d.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from . import bundle_import, model_data, odormeta, reference
from .odors import format_conc
from .sparse import Sparse

FORMATS = ("files", "wide")
# kept in a zip as they are (compressed already)
STORED = (".gz", ".tgz", ".zip", ".bz2", ".xz", ".parquet", ".pq", ".nc", ".nc4", ".netcdf")

Levels = list[tuple[str, list]]  # (name, a value per item), in order


class DownloadError(ValueError):
    """A download that cannot be made (why, for people)."""


def _cell(v: Any) -> Any:
    """A level's value as a CSV cell: '' for none, and a list or dict as text."""
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join(map(str, v))
    return json.dumps(v) if isinstance(v, dict) else v


def _named(levels: Levels) -> Levels:
    """Of `levels`, the first and those with a value for some item, each named apart."""
    out, seen = [], set()
    for k, (name, values) in enumerate(levels):
        cells = [_cell(v) for v in values]
        if k and all(c == "" for c in cells):
            continue
        n, i = name, 2
        while n in seen:
            n, i = f"{name}_{i}", i + 1
        seen.add(n)
        out.append((n, cells))
    return out


def _index(levels: Levels) -> pd.Index:
    if len(levels) == 1:
        return pd.Index(levels[0][1], name=levels[0][0], dtype=object)
    return pd.MultiIndex.from_arrays([v for _, v in levels], names=[n for n, _ in levels])


def _pick(n: int, given: Optional[Sequence], what: str) -> np.ndarray:
    """The indices of the rows or columns given (all, in order, if none are)."""
    if given is None:
        return np.arange(n)
    try:
        ix = np.array([int(i) for i in given], dtype=np.intp)
    except (TypeError, ValueError):
        raise DownloadError(f"its {what} must be given by number") from None
    if ix.size and (ix.min() < 0 or ix.max() >= n):
        raise DownloadError(f"its {what} are numbered from 0 to {n - 1}")
    return ix


def _cells(values: np.ndarray | Sparse, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The values of rows `r` by columns `c` (of a `Sparse` matrix, only those made whole)."""
    if not isinstance(values, Sparse):
        return np.asarray(values, dtype=np.float32)[np.ix_(r, c)]
    n_r, n_c = values.shape
    if len(set(r.tolist())) < len(r) or len(set(c.tolist())) < len(c):
        return values.toarray()[np.ix_(r, c)]
    at_r, at_c = np.full(n_r, -1, dtype=np.intp), np.full(n_c, -1, dtype=np.intp)
    at_r[r], at_c[c] = np.arange(len(r)), np.arange(len(c))
    rows, cols = at_r[values.index // n_c], at_c[values.index % n_c]
    keep = (rows >= 0) & (cols >= 0)
    out = np.zeros((len(r), len(c)), dtype=np.float32)
    out[rows[keep], cols[keep]] = values.values[keep]
    return out


def wide_frame(values: np.ndarray | Sparse, rows: Levels, cols: Levels, *, pick_rows: Optional[Sequence] = None,
               pick_cols: Optional[Sequence] = None, transpose: bool = False) -> pd.DataFrame:
    """A matrix as a table: its `rows`' levels its index, its `cols`' its columns (levels with
    no values left out, but for the first); of the rows and columns picked, in that order
    (`_pick`), and transposed if asked."""
    shape = values.shape
    r, c = _pick(shape[0], pick_rows, "rows"), _pick(shape[1], pick_cols, "columns")
    sub = lambda levels, ix: [(n, [v[i] for i in ix]) for n, v in levels]  # noqa: E731
    rows, cols, cells = _named(sub(rows, r)), _named(sub(cols, c)), _cells(values, r, c)
    if transpose:
        rows, cols, cells = cols, rows, cells.T
    return pd.DataFrame(cells, index=_index(rows), columns=_index(cols))


def _write_csv(frame: pd.DataFrame, dest: Path) -> None:
    with open(dest, "w", encoding="utf-8", newline="") as f:
        frame.to_csv(f)


def _zip_file(zf: zipfile.ZipFile, path: Path, name: str) -> None:
    stored = path.name.lower().endswith(STORED)
    zf.write(path, name, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)


def _odor_levels(odors: list[dict], conc: list) -> Levels:
    """The odor table's of chemicals matched to it (each a dict with ``name``, ``abbrev`` and
    ``cid``, or None), and their concentrations (if written as datasets write odors)."""
    return [("odor_table_name", [o and o.get("name") for o in odors]), ("odor_table_abbrev", [o and o.get("abbrev") for o in odors]),
            ("cid", [o and o.get("cid") for o in odors]), ("log10_conc", conc)]


# ------------------------------------------------------------- reference datasets


def reference_levels(o: dict, axis: str) -> Levels:
    """The levels of a reference dataset's rows or columns ('rows' | 'cols'; `o`: as
    `reference.opened` gives it): of a table of one value per line or a DataArray, each field
    chosen for them; else their labels (headed as the Admin tab says, or as the file does),
    groups, identifiers given and a bundle's own (e.g. MetaCyc frame IDs), and what is said
    of each; then, of its chemicals, the odor table's name, abbreviation and CID of each,
    and of glomeruli, their names as the app knows them (where not as labelled)."""
    record, meta, chem, gl = o["record"], o["meta"], o["chem"], o["gl"]
    items = record[axis]
    fields = [it.get("fields") for it in items]
    levels: Levels = []
    if items and all(f is not None for f in fields):
        levels += [(k, [f.get(k) for f in fields]) for k in dict.fromkeys(k for f in fields for k in f)]
    else:
        head = (meta.get("row_label" if axis == "rows" else "col_label") or (record.get("label_fields") or {}).get(axis)
                or ("odor" if chem.get("odors") and chem["axis"] == axis else None))
        levels.append((head or ("row" if axis == "rows" else "column"), [it["label"] for it in items]))
        levels.append(("group", [it.get("group") for it in items]))
        ids = (record.get("ids") or {}).get(axis)
        if ids:
            levels += [(k, [d.get(k) for d in ids]) for k in dict.fromkeys(k for d in ids for k in d)]
        for k in dict.fromkeys(k for it in items for k in it if k not in ("label", "group", "detail", "fields")):
            values = [it.get(k) for it in items]
            if any(isinstance(v, dict) for v in values):  # (e.g. a LOTUS taxon's level, name and kingdom)
                levels += [(f"{k}_{s}", [v.get(s) if isinstance(v, dict) else None for v in values])
                           for s in dict.fromkeys(s for v in values if isinstance(v, dict) for s in v)]
            else:
                levels.append((k, values))
        levels.append(("detail", [it.get("detail") for it in items]))
    if chem["axis"] == axis or (record.get("pairwise") and chem["axis"]):
        its = chem["items"]
        odors = [{"name": c["name"], "abbrev": c["abbrev"], "cid": c["cid"]} if c["odor_id"] is not None or c["cid"] else None
                 for c in its]
        add = _odor_levels(odors, [c["conc"] for c in its])
        names = {n for n, _ in levels}
        levels += [(n, v) for n, v in add if n not in names]
    if gl["axis"] == axis and any(name != it["label"] for name, it in zip(gl["names"], items)):
        levels.append(("glomerulus", gl["names"]))
    return levels


def _reference_readme(o: dict, files: list[str]) -> str:
    e, m, record = o["entry"], o["meta"], o.get("record") or {}
    origin = "added in the Admin tab" if m["origin"] == "admin" else "built in (reference_data/)"
    lines = [f"# {m['title']}", ""]
    if m.get("description"):
        lines += [m["description"], ""]
    if m.get("source"):
        lines += [f"Source: {m['source']}", ""]
    lines += [f"Reference dataset `{m['id']}` of the odor response database, {origin}: read from `{m['path']}`"
              + (f" (last on {record['imported_at'].replace('T', ' ')[:16]})" if record.get("imported_at") else "") + ".",
              f"Downloaded {dt.datetime.now():%Y-%m-%d %H:%M}.", ""]
    how = {"chemicals": e.get("chemical_axis") or "detect", "colors": e.get("kind") or "detect"}
    fields = reference._fields(e.get("fields"))
    if fields:
        how["fields"] = fields
    lines += ["How it is read (as set when it was added):", "", "```json", json.dumps(how, indent=2), "```", ""]
    if record.get("shape"):
        lines.append(f"As read: {record['shape'][0]:,} rows × {record['shape'][1]:,} columns ({record.get('kind')}).")
        lines += [f"- {n[0].upper() + n[1:]}." for n in record.get("notes") or []]
        lines.append("")
    lines += ["## Files", "", *files]
    return "\n".join(lines) + "\n"


def reference_download(db, ref_id: str, index: odormeta.OdorIndex, fmt: str, dest: Path, *, rows: Optional[Sequence] = None,
                       cols: Optional[Sequence] = None, transpose: bool = False) -> tuple[str, str]:
    """Write a reference dataset's download (`fmt`, see the module's docstring) to `dest`:
    (the name to offer it as, its media type). Raises LookupError for no such dataset, and
    DownloadError for one that cannot be read (or rows and columns not of it)."""
    o = reference.opened(db, ref_id, index)
    if o is None:
        raise LookupError("no such reference dataset")
    if o["state"] != "ok":
        raise DownloadError(f"it cannot be read: {o['error']}")
    stem = f"olfdb_reference-{ref_id}"
    if fmt == "wide":
        frame = wide_frame(o["values"], reference_levels(o, "rows"), reference_levels(o, "cols"),
                           pick_rows=rows, pick_cols=cols, transpose=transpose)
        _write_csv(frame, dest)
        return f"{stem}.csv", "text/csv"
    path = Path(o["entry"]["path"])
    raw = path.parent / "raw"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        if reference.is_bundle_path(path) and raw.is_dir() and (path.parent / bundle_import.RECORD).is_file():
            for p in sorted(raw.rglob("*")):
                if p.is_file():
                    _zip_file(zf, p, f"{stem}/raw/{p.relative_to(raw).as_posix()}")
            files = [f"- `raw/`: the files its bundle (`{path.name}`) was built from, as uploaded (those gzipped as kept, "
                     "ending in .gz), laid out as its build script reads them."]
        else:
            _zip_file(zf, path, f"{stem}/{path.name}")
            files = [f"- `{path.name}`: the file, as it is on the server."]
        zf.writestr(f"{stem}/README.md", _reference_readme(o, files))
    return f"{stem}_files.zip", "application/zip"


# ------------------------------------------------------------------- model runs


def model_download(db, run_id: str, index: odormeta.OdorIndex, fmt: str, dest: Path, root: Path, *,
                   rows: Optional[Sequence] = None, cols: Optional[Sequence] = None, transpose: bool = False) -> tuple[str, str]:
    """Write a model run's download (`fmt`, see the module's docstring; its rows its KCs, as
    the Models tab shows them) to `dest`: (the name to offer it as, its media type). Raises
    LookupError for no such run (`model_data.run_info`), DownloadError for rows and columns
    not of it."""
    info = model_data.run_info(db, run_id, root)
    stem = "olfdb_model_" + re.sub(r"[^\w.-]+", "_", info["file"].removesuffix(".parquet"))
    if fmt == "wide":
        df, _ = model_data.run_frame(info["path"])
        odors = model_data.run_odors(df.columns, index)
        ix = df.index
        kcs = [(str(n) if n is not None else f"level {i}", list(ix.get_level_values(i))) for i, n in enumerate(ix.names)]
        conc = [None if o["log10_conc"] is None else format_conc(o["log10_conc"]) for o in odors]  # (as written: -3, not -3.0)
        cols_levels = [("odor", [o["odor"] for o in odors]), *_odor_levels([o["chem"] for o in odors], conc)]
        values = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        _write_csv(wide_frame(values, kcs, cols_levels, pick_rows=rows, pick_cols=cols, transpose=transpose), dest)
        return f"{stem}.csv", "text/csv"
    exp, model, b = info["experiment"], info["model"], info.get("baseline")
    exp_dir = info["folder"] / exp["key"]
    params = "\n".join(f"- {k}: {'' if v is None else v}" for k, v in model["params"].items())
    if info.get("job"):  # (a baseline run on datasets, model_runs.py: all of it is its own download)
        where = f"run #{info['job']}"
        origin = (f"From olfdb's model run #{info['job']} (a baseline model run on datasets, in the Models tab), whose own download "
                  f"has everything it wrote and what it was run on. Downloaded {dt.datetime.now():%Y-%m-%d %H:%M}.")
    elif b:  # (a baseline's imported copy)
        where = b["experiment"]
        origin = (f"From olfdb's copy of a baseline model's files (chosen in the Admin tab), imported {b['imported_at'][:16]} "
                  f"from `{where}` on tensor (from its top). Downloaded {dt.datetime.now():%Y-%m-%d %H:%M}.")
    else:
        where = f"{info['source']['path']}/{exp['key']}".rstrip("/")
        origin = (f"From `{info['source']['path']}/{info['file']}` on tensor (from its top), of the folder of model data "
                  f"`{info['source']['path']}` (added in the Admin tab). Downloaded {dt.datetime.now():%Y-%m-%d %H:%M}.")
    lines = [f"# Model run {info['file']}", "", origin, "",
             f"Its model: row {model['row'] + 1} of `{where}/params.csv` (below, its parameters).", "", params or "(none)", "",
             "## Files", "", f"- `{info['path'].name}`: its spike counts, as {'run' if info.get('job') else 'imported' if b else 'on tensor'}."]
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        _zip_file(zf, info["path"], f"{stem}/{info['path'].name}")
        for name, what in ((model_data.PARAMS, "the parameters of its experiment's models (its own: the row above)"),
                           (model_data.PROVENANCE, "how its experiment's models were made, as it says"),
                           *([(model["fit_provenance"], "how its model was made (fitted), as it says")] if model.get("fit_provenance") else [])):
            if (exp_dir / name).is_file():
                _zip_file(zf, exp_dir / name, f"{stem}/{name}")
                lines.append(f"- `{name}`: {what}.")
        zf.writestr(f"{stem}/README.md", "\n".join(lines) + "\n")
    return f"{stem}_files.zip", "application/zip"
