"""Reference datasets (chemicals x fruits, chemicals x pathways, ...), for the Reference
data tab. Two kinds:

- built in: listed in ``reference_data/manifest.json`` (format in reference_data/README.md),
  read from there when asked for;
- added in the Admin tab: a file anywhere on the server, which is never written to. It is
  read ("imported") into ``data/references/<id>/`` (``import.json``: its labels,
  identifiers and notes; ``values.f32``: its numbers, or ``values.npz`` those of a matrix
  most of whose cells are 0, `sparse.Sparse`), and again whenever it changes.
  Deleting it deletes only that copy. A file outside the folders the server may read is
  added to them, and read once the server restarts.

Either is a CSV, TSV or parquet file: row labels in the first column, an optional
``group`` column (row grouping), and one column per matrix column (blank = missing). Or
an xarray DataArray of two dimensions saved as netCDF (`DataArray.to_netcdf`), its
coordinates labelling and identifying its items (`_parse_dataarray`). Or a bundle (an
SQLite file whose ``about`` says what it is, `bundle_kind`): of MetaCyc (metacyc.py) or
LOTUS (lotus.py), one of whose views is made with the odor table's chemicals matched by the
bundle's module, and again whenever the odor table changes.
Chemicals can be identified by columns named ``cid`` (or ``pubchem_cid``, ...),
``inchi``, ``cas``, ``metacyc``, ``lotus``, ``name`` or ``abbrev`` (then the rows are
chemicals), or by rows labelled so (then the columns are). They are matched to the odor
table when served (`chemicals`), so that they are shown by its names whatever it is
changed to: by CID, else InChI, else InChI but for stereochemistry one of them does not
specify (`odormeta.stereo_compatible`: warned of, `stereo_warning`), else the other
identifiers and labels, never otherwise against a CID or InChI. Several of them one odor
are one in the Compare tab, as their ``merge`` field says (`MERGES`, e.g. VCF's amounts
of stereoisomers summed). Glomeruli are found by their labels (`glomeruli`): such a dataset (of
glomeruli by anything else, and no odors) is compared with datasets on their glomeruli.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import re
import shutil
import sqlite3
import stat
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import config, glomeruli as glom, lotus, metacyc, odormeta, sandbox
from .odors import Odor, OdorFormatError, format_conc, parse_odor
from .sparse import Sparse

NETCDF_SUFFIXES = (".nc", ".nc4", ".netcdf")  # xarray DataArrays
BUNDLES = {"metacyc": metacyc, "lotus": lotus}  # bundles, by the kind their about says: the module reading them
BUNDLE_ERRORS = (metacyc.MetacycError, lotus.LotusError)
BUNDLE_SUFFIXES = tuple(dict.fromkeys(x for m in BUNDLES.values() for x in m.SUFFIXES))
# a table's: laid out as its matrix, or one value per line (see `_parse_list`); a bundle's view (its kind)
LAYOUTS = ("matrix", "list", *BUNDLES)
MAX_CELLS = 20_000_000  # a matrix made from a list: its rows × columns, at most
SUFFIXES = (".csv", ".tsv", ".tab", ".txt", ".parquet", ".pq", *NETCDF_SUFFIXES, *BUNDLE_SUFFIXES)
_TABLES = [x for x in SUFFIXES if x not in BUNDLE_SUFFIXES]
NOT_A_TABLE = (f"not a table it can read: give a {', '.join(_TABLES[:-1])} or {_TABLES[-1]} file, or a MetaCyc or LOTUS "
               f"bundle ({', '.join(BUNDLE_SUFFIXES)}, made by app/metacyc_build.py or app/lotus_build.py)")
NOT_A_BUNDLE = "not a bundle it can read: a MetaCyc or LOTUS bundle (made by app/metacyc_build.py or app/lotus_build.py)"
KINDS = ("detect", "sequential", "diverging", "binary")
AXES = ("detect", "rows", "cols", "none")
# identifier columns (or rows) of a chemical, by accepted header (matched as odor sheet headers are)
ID_HEADERS = {
    "cid": ("cid", "pubchem cid", "pubchem id", "pubchem"),
    "inchi": ("inchi", "std inchi", "standard inchi"),
    "inchikey": ("inchikey", "inchi key", "std inchikey", "standard inchikey"),
    "smiles": ("smiles", "canonical smiles", "isomeric smiles"),
    "cas": odormeta.HEADERS["cas"],
    "metacyc": odormeta.HEADERS["metacyc"],
    "lotus": odormeta.HEADERS["lotus"],
    "name": ("name", "chemical name", "compound name", "common name", "iupac name"),
    "abbrev": odormeta.HEADERS["abbrev"],
}
_ID_FIELD = {h: f for f, hs in ID_HEADERS.items() for h in hs}
MEMO_SIZE = 8  # files read but not imported (built in, or being checked) kept parsed
# how the Compare tab makes one of several items of its chemical axis that are one odor
# (``merge`` of its fields; the first, the default): cell by cell, of those not missing
MERGES = ("mean", "sum", "max")

_lock = threading.Lock()  # one import at a time
_memo: "OrderedDict[tuple, tuple[dict, np.ndarray | Sparse]]" = OrderedDict()
_memo_lock = threading.Lock()


class ReferenceError(ValueError):
    pass


# --------------------------------------------------------------------- reading


def _header_key(h: str) -> str:
    return re.sub(r"[\s_]+", " ", h.strip().lower()).rstrip(" .:#")


def _text(v) -> Optional[str]:
    """A cell as text (None if empty)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    s = (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)).strip()
    return s or None


def _id_value(field: str, v) -> Optional[Any]:
    """An identifier cell, checked (None if empty or malformed)."""
    s = _text(v)
    if s is None:
        return None
    if field == "cid":
        m = re.fullmatch(r"(?:cid[:\s]*)?(\d+)(?:\.0+)?", s, re.I)
        return int(m[1]) if m else None
    if field == "inchi":
        s = f"InChI={s}" if s.startswith("1S/") else s  # (as some databases give them)
        return s if s.startswith("InChI=") else None
    return s


def is_bundle_path(path: Path) -> bool:
    """Whether `path` is named as a bundle may be (not whether it is one)."""
    return Path(path).suffix.lower() in BUNDLE_SUFFIXES


_kinds: dict[tuple, Optional[str]] = {}


def bundle_kind(path: Path) -> Optional[str]:
    """The kind of bundle at `path` (a key of `BUNDLES`), as its ``about`` says, for the file
    as it is now; None if it is not named as one, cannot be read, or is of no kind known."""
    if not is_bundle_path(path):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _kinds:
        try:
            con = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro&immutable=1", uri=True)
            try:
                row = con.execute("SELECT value FROM about WHERE key = 'kind'").fetchone()
            finally:
                con.close()
            kind = json.loads(row[0]) if row else None
        except (sqlite3.DatabaseError, ValueError, OSError):
            kind = None
        if len(_kinds) > 64:
            _kinds.clear()
        _kinds[key] = kind if kind in BUNDLES else None
    return _kinds[key]


def _bundle(path: Path):
    """The module reading the bundle at `path` (`BUNDLES`).

    Raises:
        ReferenceError: if it is none it can read.
    """
    kind = bundle_kind(path)
    if kind is None:
        raise ReferenceError(NOT_A_BUNDLE)
    return BUNDLES[kind]


def read_table(path: Path) -> pd.DataFrame:
    """The file as a table of text cells (numbers as they are in parquet), labelled by its first column."""
    suffix = path.suffix.lower()
    if suffix not in SUFFIXES:
        raise ReferenceError(NOT_A_TABLE)
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        if isinstance(df.index, pd.RangeIndex) and df.shape[1]:
            df = df.set_index(df.columns[0])
        return df
    sep = {".csv": ",", ".tsv": "\t", ".tab": "\t"}.get(suffix)
    return pd.read_csv(path, index_col=0, dtype=str, sep=sep, engine="c" if sep else "python")


def parse(path: Path, chemical_axis: str = "detect", kind: str = "detect", fields: Optional[dict] = None,
          odors: Optional[list[dict]] = None) -> tuple[dict, np.ndarray | Sparse]:
    """A reference dataset file read: (record, values).

    Args:
        chemical_axis: where the chemicals are: 'rows', 'cols', 'none', or 'detect' (rows
            if there are identifier columns, else columns if there are identifier rows,
            else decided by labels when served, see `chemicals`).
        kind: its colormap kind, or 'detect' (binary if all 0/1, diverging if signed).
        fields: which of its fields make the matrix's rows, columns and values, of a table
            of one value per line (``layout`` 'list': see `_parse_list`) or a DataArray
            (`_parse_dataarray`); the view of a bundle (``layout`` its kind, see
            `metacyc.options`, `lotus.options`); else (None, or ``layout`` 'matrix') a table
            as laid out.
        odors: the odor table (as `Database.list_odors` gives it), over whose chemicals a
            bundle's view is made.

    Returns:
        record: ``rows`` [{label, group}], ``cols`` [{label}] (of a table of one value per
        line or a DataArray, each with its ``detail`` and ``fields``, see `_items`), ``ids``
        {rows, cols: per item, the identifiers given for it, or None}, of a table laid out
        as its matrix ``label_fields`` {rows: its first column's header}, ``chemical_axis``, ``kind``,
        ``shape``, ``notes`` (what was left out or not understood); a MetaCyc view's also
        ``metacyc`` (its options) and, of distances between chemicals, ``pairwise``
        ('distance': its columns are its rows); a LOTUS view's ``lotus`` (its options, and
        how it was made: `lotus.describe`); a bundle's view may have ``matched``: per
        item of its chemical axis, the odor it is ([id, how], or None: none), as its
        module matched them (see `chemicals`). values: float32 (rows x cols, NaN where
        missing), or of a bundle's view, maybe a `Sparse` matrix.

        Of a file not a bundle, also ``nonzero`` {rows, cols: per item, how many of its
        cells are not 0 or missing}.

    Raises:
        ReferenceError: if it has no numbers. Errors reading the file are raised as they are.
    """
    if is_bundle_path(path):
        mod = _bundle(path)
        try:
            record, values = mod.parse(path, fields, odors or [])
        except BUNDLE_ERRORS as e:
            raise ReferenceError(str(e)) from None
        if kind != "detect":
            record["kind"] = kind
        return record, values
    if path.suffix.lower() in NETCDF_SUFFIXES:
        record, values = _parse_dataarray(path, chemical_axis, kind, fields)
    elif fields and fields.get("layout") == "list":
        record, values = _parse_list(path, chemical_axis, kind, fields)
    else:
        record, values = _parse_matrix(path, chemical_axis, kind)
    held = np.isfinite(values) & (values != 0)
    record["nonzero"] = {"rows": held.sum(axis=1).tolist(), "cols": held.sum(axis=0).tolist()}
    return record, values


def _parse_matrix(path: Path, chemical_axis: str, kind: str) -> tuple[dict, np.ndarray]:
    """`parse` of a table laid out as its matrix."""
    df = read_table(path)
    labels = [_text(v) or "" for v in df.index]
    headers = [_text(c) or "" for c in df.columns]
    if not labels or not headers:
        raise ReferenceError("it has no rows or no columns")
    notes = []
    id_cols = {} if chemical_axis in ("cols", "none") else {j: _ID_FIELD[_header_key(h)] for j, h in enumerate(headers)
                                                            if _header_key(h) in _ID_FIELD}
    id_rows = {} if chemical_axis in ("rows", "none") or id_cols else {i: _ID_FIELD[_header_key(r)] for i, r in enumerate(labels)
                                                                      if _header_key(r) in _ID_FIELD}
    group_col = next((j for j, h in enumerate(headers) if _header_key(h) == "group" and j not in id_cols), None)
    cols = [j for j in range(len(headers)) if j not in id_cols and j != group_col]
    rows = [i for i in range(len(labels)) if i not in id_rows]
    block = df.iloc[rows, cols]
    numbers = block.apply(lambda s: pd.to_numeric(s, errors="coerce")).to_numpy(dtype=np.float64)
    given = block.notna().to_numpy()
    # rows and columns of text without a single number are left out (notes, units), as
    # are blank ones without a label; blank ones with a label are all missing values
    finite = np.isfinite(numbers)
    text = given & ~finite
    blank_c = [not h or h.startswith("Unnamed: ") for h in (headers[j] for j in cols)]
    keep_c = finite.any(axis=0) | (~text.any(axis=0) & ~np.array(blank_c, dtype=bool))
    keep_r = finite[:, keep_c].any(axis=1) | (~text[:, keep_c].any(axis=1) & np.array([bool(labels[i]) for i in rows]))
    if not finite[np.ix_(keep_r, keep_c)].any():
        raise ReferenceError("it has no numbers (besides identifiers)")
    for what, keep, names, texts in (("column", keep_c, [headers[j] for j in cols], text.any(axis=0)),
                                     ("row", keep_r, [labels[i] for i in rows], text[:, keep_c].any(axis=1))):
        dropped = [n for n, k, t in zip(names, keep, texts) if not k and t]
        if dropped:
            shown = ", ".join(f"'{n}'" for n in dropped[:5]) + (f" and {len(dropped) - 5} more" if len(dropped) > 5 else "")
            notes.append(f"left out, as {'they have' if len(dropped) > 1 else 'it has'} no numbers: {what}{'s' if len(dropped) > 1 else ''} {shown}")
    cols = [j for j, k in zip(cols, keep_c) if k]
    rows = [i for i, k in zip(rows, keep_r) if k]
    values = numbers[np.ix_(keep_r, keep_c)]
    n_text = int((given[np.ix_(keep_r, keep_c)] & ~np.isfinite(values)).sum())
    if n_text:
        notes.append(f"{n_text} cell{'s are' if n_text > 1 else ' is'} not a number: shown as missing")

    def ids(items: list[int], fields: dict[int, str], cell) -> Optional[list[dict]]:
        if not fields:
            return None
        out, bad = [], {}
        for k in items:
            got = {}
            for pos, field in fields.items():
                raw = cell(k, pos)
                v = _id_value(field, raw)
                if v is not None:
                    got[field] = v
                elif _text(raw) is not None:
                    bad[field] = bad.get(field, 0) + 1
            out.append(got)
        notes.extend(f"{n} {f.upper() if f in ('cid', 'cas') else 'InChI' if f == 'inchi' else f} value{'s' if n > 1 else ''} not understood"
                     for f, n in bad.items())
        return out

    record = {
        "rows": [{"label": labels[i], "group": _text(df.iat[i, group_col]) or "" if group_col is not None else None} for i in rows],
        "cols": [{"label": headers[j]} for j in cols],
        "label_fields": {"rows": _text(df.index.name), "cols": None},  # (what its rows' labels are headed, if anything)
        "ids": {"rows": ids(rows, id_cols, lambda i, j: df.iat[i, j]), "cols": ids(cols, id_rows, lambda j, i: df.iat[i, j])},
        "chemical_axis": chemical_axis,
        "kind": kind if kind != "detect" else _detect_kind(values),
        "shape": list(values.shape),
        "notes": notes,
    }
    return record, values.astype(np.float32)


def _items(labels: list[str], coords: dict[str, list[Optional[str]]], label_key: Optional[str],
           id_fields: dict[str, str]) -> tuple[list[dict], Optional[list[dict]]]:
    """Items of a matrix's axis, labelled by `labels`: those labelled alike told apart by
    the values of `coords` (each item's, by field) that differ among them (coarsest first:
    the fields with the fewest values; not `label_key`'s, identifiers or links), all of
    which are each item's ``detail``; and each item's identifiers (`id_fields`: field ->
    identifier, of `coords`), or None if none are. Each keeps its value of every field too
    (``fields``, in their order: `label_key`'s first), a level each in a download of it.

    Returns:
        ([{label, detail, fields}], [{identifier: value}] or None)
    """
    links = lambda vs: all(v is None or re.match(r"https?://", v) for v in vs)
    told = sorted((k for k in coords if k != label_key and k not in id_fields and not links(coords[k])), key=lambda k: len(set(coords[k])))
    alike: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        alike.setdefault(lab, []).append(i)
    out = []
    for i, lab in enumerate(labels):
        same = alike[lab]
        apart = [coords[k][i] for k in told if len(same) > 1 and len({coords[k][j] for j in same}) > 1 and coords[k][i]]
        detail = " · ".join(f"{k.replace('_', ' ')}: {coords[k][i]}" for k in told if coords[k][i])
        out.append({"label": f"{lab} — {' · '.join(apart)}" if apart else lab, "detail": detail or None,
                    "fields": {k: coords[k][i] for k in coords}})
    n, seen = Counter(o["label"] for o in out), Counter()
    for o in out:  # still alike: numbered
        if n[o["label"]] > 1:
            seen[o["label"]] += 1
            o["label"] = f"{o['label']} ({seen[o['label']]})"
    ids = [{f: v for k, f in id_fields.items() if (v := _id_value(f, coords[k][i])) is not None}
           for i in range(len(labels))] if id_fields else None
    return out, ids


def _id_fields(names: list[str]) -> dict[str, str]:
    """Of fields (columns, coordinates), those named as identifiers: field -> identifier."""
    return {k: _ID_FIELD[_header_key(k)] for k in names if _header_key(k) in _ID_FIELD}


def _chosen(spec: Optional[dict], available: list[str], what: str) -> tuple[Optional[str], list[str]]:
    """A ``{"label", "keep"}`` choice of fields (of `available`) for a matrix's rows or
    columns: (the field labelling them, or None for their numbers; the others kept)."""
    label = (spec or {}).get("label") or None
    keep = [k for k in dict.fromkeys((spec or {}).get("keep") or []) if k != label]
    unknown = [k for k in [label, *keep] if k and k not in available]
    if unknown:
        raise ReferenceError(f"its {what} are to be of {', '.join(map(repr, unknown))}, which it has no field of")
    return label, keep


def _parse_dataarray(path: Path, chemical_axis: str, kind: str, fields: Optional[dict] = None) -> tuple[dict, np.ndarray]:
    """`parse` of an xarray DataArray file (netCDF): a matrix of its two dimensions (rows the
    first), its missing values 0 (absent).

    Each item of a dimension is labelled by its coordinate chosen (`fields`' ``rows`` or
    ``cols``: ``label``), else the one named ``name`` (or ``label`` or ``title``, or
    ending in "name", e.g. ``product_name``), else the dimension's own coordinate, else
    its number; it is described by the others chosen (``keep``; by default, all of them),
    the rest ignored. Items labelled alike are told apart by those that differ among them
    (`_items`). Coordinates named as identifier columns (``cid``, ``inchi``, ``name``, ...)
    identify chemicals, as those columns do.
    """
    # HDF5 (netCDF4's) locks a file it opens, which fails on NFS and read-only mounts, as
    # tensor's may be: only ever read here
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import xarray as xr

    da = xr.load_dataarray(path)
    if da.ndim != 2:
        raise ReferenceError(f"it has {da.ndim} dimension{'' if da.ndim == 1 else 's'} ({', '.join(map(str, da.dims))}), and a matrix has 2")
    try:
        numbers = np.asarray(da.values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ReferenceError("its values are not numbers") from None
    if not np.isfinite(numbers).any():
        raise ReferenceError("it has no numbers")
    notes = []
    n_missing = int(np.isnan(numbers).sum())
    if n_missing:
        notes.append(f"{n_missing:,} missing value{'s' if n_missing > 1 else ''} taken as 0 (absent)")
    values = np.where(np.isnan(numbers), 0.0, numbers)

    def axis(dim: str, spec: Optional[dict], ids_allowed: bool, what: str) -> tuple[list[dict], Optional[list[dict]]]:
        coords = {str(k): [_text(v) for v in c.values] for k, c in da.coords.items() if c.dims == (dim,)}
        if spec is None:
            keys = list(coords)
            label = (next((k for k in keys if _header_key(k) in ("name", "label", "title")), None)
                     or next((k for k in keys if k.lower().endswith("name")), None) or (dim if dim in coords else None))
            keep = [k for k in keys if k != label]
        else:
            label, keep = _chosen(spec, list(coords), what)
        coords = {k: coords[k] for k in [label, *keep] if k}
        labels = [v or "" for v in coords[label]] if label else [str(i + 1) for i in range(da.sizes[dim])]
        return _items(labels, coords, label, _id_fields(list(coords)) if ids_allowed else {})

    rows_dim, cols_dim = map(str, da.dims)
    fields = fields or {}
    rows, row_ids = axis(rows_dim, fields.get("rows"), chemical_axis not in ("cols", "none"), "rows")
    cols, col_ids = axis(cols_dim, fields.get("cols"), chemical_axis not in ("rows", "none") and not row_ids, "columns")
    record = {
        "rows": [{"label": r["label"], "group": None, "detail": r["detail"], "fields": r["fields"]} for r in rows],
        "cols": [{"label": c["label"], "detail": c["detail"], "fields": c["fields"]} for c in cols],
        "ids": {"rows": row_ids, "cols": col_ids},
        "chemical_axis": chemical_axis,
        "kind": kind if kind != "detect" else _detect_kind(values),
        "shape": list(values.shape),
        "notes": notes,
    }
    return record, values.astype(np.float32)


def read_rows(path: Path) -> pd.DataFrame:
    """A table file as its lines, every column as named in its header (text cells; numbers as they are in parquet)."""
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        return df if isinstance(df.index, pd.RangeIndex) else df.reset_index()
    sep = {".csv": ",", ".tsv": "\t", ".tab": "\t"}.get(suffix)
    return pd.read_csv(path, dtype=str, sep=sep, engine="c" if sep else "python")


def _parse_list(path: Path, chemical_axis: str, kind: str, fields: dict) -> tuple[dict, np.ndarray]:
    """`parse` of a table of one value per line (a "list"), made into a matrix by its fields
    chosen (`fields`): ``rows`` and ``cols`` (each ``{"label": field, "keep": [fields]}``:
    every combination of these fields' values on a line is a row, or a column, labelled by
    `label`'s and told apart by the others where need be, see `_items`), and ``value``
    (a field of numbers, several lines' of one cell their mean; or None: 1 where a line
    lists the row and the column, else 0); ``empty`` ('zero' or 'missing', the default):
    a cell no line lists. Every other field is ignored, and so are lines with no label
    for their row or column. Identifier fields (``cid``, ``inchi``, ...) of the rows (or,
    if they have none, of the columns) identify chemicals, as identifier columns do.
    """
    df = read_rows(path)
    names = [str(c) for c in df.columns]
    df.columns = names
    row_label, row_keep = _chosen(fields.get("rows"), names, "rows")
    col_label, col_keep = _chosen(fields.get("cols"), names, "columns")
    value = fields.get("value") or None
    if not row_label or not col_label:
        raise ReferenceError("choose the fields labelling its rows and its columns")
    if value is not None and value not in names:
        raise ReferenceError(f"its values are to be of {value!r}, which it has no field of")
    if value is not None and _header_key(value) in _ID_FIELD:
        raise ReferenceError(f"{value!r} identifies chemicals ({_ID_FIELD[_header_key(value)]}), so it cannot be its values")
    row_keys, col_keys = [row_label, *row_keep], [col_label, *col_keep]
    both = set(row_keys) & set(col_keys) | ({value} & set(row_keys + col_keys))
    if both:
        raise ReferenceError(f"{', '.join(map(repr, sorted(both)))} cannot be of both its rows and its columns (or values)")
    text = {k: [_text(v) for v in df[k]] for k in dict.fromkeys(row_keys + col_keys)}
    lines = [i for i in range(len(df)) if text[row_label][i] and text[col_label][i]]
    notes = []
    if len(lines) < len(df):
        n = len(df) - len(lines)
        notes.append(f"{n:,} line{'s' if n > 1 else ''} with no {row_label!r} or {col_label!r} left out")
    if not lines:
        raise ReferenceError("no line has a label for both its row and its column")
    row_of: dict[tuple, int] = {}
    col_of: dict[tuple, int] = {}
    ri = np.fromiter((row_of.setdefault(tuple(text[k][i] for k in row_keys), len(row_of)) for i in lines), dtype=np.intp, count=len(lines))
    ci = np.fromiter((col_of.setdefault(tuple(text[k][i] for k in col_keys), len(col_of)) for i in lines), dtype=np.intp, count=len(lines))
    n_rows, n_cols = len(row_of), len(col_of)
    if n_rows * n_cols > MAX_CELLS:
        raise ReferenceError(f"it would be {n_rows:,} rows × {n_cols:,} columns ({n_rows * n_cols:,} cells, and at most "
                             f"{MAX_CELLS:,} are shown): choose fewer fields for its rows or columns")
    values_given = None
    if value is None:
        values = np.zeros((n_rows, n_cols), dtype=np.float32)
        values[ri, ci] = 1
        notes.append("1 where a line lists the row and column, else 0")
    else:
        numbers = pd.to_numeric(df[value].iloc[lines], errors="coerce").to_numpy(dtype=np.float64)
        given = np.array([_text(v) is not None for v in df[value].iloc[lines]])
        values = np.full((n_rows, n_cols), 0.0 if fields.get("empty") == "zero" else np.nan, dtype=np.float32)
        values[ri, ci] = np.nan  # listed, but maybe with no number
        cells = pd.DataFrame({"r": ri, "c": ci, "v": numbers}).groupby(["r", "c"])["v"]
        mean, count = cells.mean(), cells.count()
        values[mean.index.get_level_values(0), mean.index.get_level_values(1)] = mean.to_numpy(dtype=np.float32)
        n_text = int((given & ~np.isfinite(numbers)).sum())
        examples = ""
        if n_text:
            examples = ", ".join(repr(v) for v in pd.Series(df[value].iloc[lines].to_numpy()[given & ~np.isfinite(numbers)]).value_counts().index[:3])
            notes.append(f"{n_text:,} value{'s are' if n_text > 1 else ' is'} not a number (e.g. {examples}): missing")
        values_given = {"lines": int(given.sum()), "not_numbers": n_text, "examples": examples}
        n_many = int((count > 1).sum())
        if n_many:
            notes.append(f"{n_many:,} cell{'s are' if n_many > 1 else ' is'} on more than one line: the mean of their numbers")
        if not np.isfinite(values).any():
            raise ReferenceError(f"{value!r} has no numbers: choose another field for its values, or presence")

    def axis(keys: list[str], of: dict[tuple, int], ids_allowed: bool) -> tuple[list[dict], Optional[list[dict]]]:
        combos = list(of)  # in the order they first come
        coords = {k: [c[j] for c in combos] for j, k in enumerate(keys)}
        return _items([c[0] for c in combos], coords, keys[0], _id_fields(keys) if ids_allowed else {})

    rows, row_ids = axis(row_keys, row_of, chemical_axis not in ("cols", "none"))
    cols, col_ids = axis(col_keys, col_of, chemical_axis not in ("rows", "none") and not row_ids)
    record = {
        "rows": [{"label": r["label"], "group": None, "detail": r["detail"], "fields": r["fields"]} for r in rows],
        "cols": [{"label": c["label"], "detail": c["detail"], "fields": c["fields"]} for c in cols],
        "ids": {"rows": row_ids, "cols": col_ids},
        "chemical_axis": chemical_axis,
        "kind": kind if kind != "detect" else _detect_kind(np.nan_to_num(values)),
        "shape": [n_rows, n_cols],
        "notes": notes,
        "values_given": values_given,  # (of its value field: how many lines give one, how many not a number)
    }
    return record, values


def describe_fields(path: Path) -> dict:
    """A file's fields, for choosing which make a matrix's rows, columns and values (the
    Admin tab): ``kind`` ('table' | 'dataarray'), ``fields`` ({name, samples (a few
    values), distinct (how many), numeric (share of its values that are numbers), and
    for a DataArray, ``dim``; of a table, ``id``: the identifier it is, if any}), ``dims``
    (a DataArray's) and ``suggested`` (choices to start from, as `parse` takes them: for a
    table, ``layout`` 'list' if it looks like one, one value per line, else 'matrix'; of
    VCF's, as for it, then with ``recognized`` 'vcf', `_vcf_suggested`). Of a MetaCyc bundle: ``kind`` 'metacyc', its
    ``about`` (version, when built, counts) and its view's options to start from; of a
    LOTUS bundle, ``kind`` 'lotus', its ``about`` (`lotus.about_summary`), the options to
    start from and the ``choices`` of each (`lotus.choices`)."""
    if is_bundle_path(path) and bundle_kind(path) == "lotus":
        try:
            about = lotus.about_summary(path)
        except lotus.LotusError as e:
            raise ReferenceError(str(e)) from None
        return {"kind": "lotus", "dims": None, "fields": [], "suggested": {**lotus.options(None), "layout": "lotus"},
                "about": about, "choices": lotus.choices(about)}
    if is_bundle_path(path):
        _bundle(path)
        try:
            about = metacyc.read_about(path)
        except metacyc.MetacycError as e:
            raise ReferenceError(str(e)) from None
        return {"kind": "metacyc", "dims": None, "fields": [], "suggested": metacyc.options({"view": "pathways"}) | {"layout": "metacyc"},
                "about": {k: about.get(k) for k in ("version", "built_at", "counts")}}
    if path.suffix.lower() in NETCDF_SUFFIXES:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        import xarray as xr

        da = xr.load_dataarray(path)
        dims = list(map(str, da.dims))
        out = []
        for k, c in da.coords.items():
            if len(c.dims) == 1 and str(c.dims[0]) in dims:
                vals = [_text(v) for v in c.values]
                out.append({"name": str(k), "dim": str(c.dims[0]), "samples": list(dict.fromkeys(v for v in vals if v))[:3],
                            "distinct": len(set(vals))})
        suggested = {}
        for d, key in zip(dims, ("rows", "cols")):
            names = [f["name"] for f in out if f["dim"] == d]
            label = (next((k for k in names if _header_key(k) in ("name", "label", "title")), None)
                     or next((k for k in names if k.lower().endswith("name")), None) or (d if d in names else None))
            suggested[key] = {"label": label, "keep": [k for k in names if k != label]}
        return {"kind": "dataarray", "dims": dims, "fields": out, "suggested": suggested}

    df = read_rows(path)
    out = []
    for k in df.columns:
        vals = [_text(v) for v in df[k]]
        given = [v for v in vals if v is not None]
        numeric = float(pd.to_numeric(pd.Series(given, dtype=object), errors="coerce").notna().mean()) if given else 0.0
        out.append({"name": str(k), "samples": list(dict.fromkeys(given))[:3], "distinct": len(set(given)), "numeric": round(numeric, 3),
                    "id": _ID_FIELD.get(_header_key(str(k)))})
    names = [f["name"] for f in out]
    vcf_like = _vcf_suggested(names)
    if vcf_like:
        return {"kind": "table", "dims": None, "fields": out, "suggested": vcf_like, "recognized": "vcf"}
    n = len(df)
    ids = _id_fields(names)
    links = lambda f: bool(f["samples"]) and all(re.match(r"https?://", v) for v in f["samples"])
    numeric = [f["name"] for f in out if f["numeric"] > 0.5 and f["name"] not in ids]
    text = [f for f in out if f["name"] not in numeric]
    row = (next((f for f in text if _header_key(f["name"]) in ID_HEADERS["name"]), None)
           or next((f for f in text if f["name"] not in ids), None))
    row_label = row and row["name"]
    # a column of mostly numbers, by its name if one says so (of those, the most numbers); else presence
    value_like = lambda f: bool(re.search(r"value|quantit|conc|amount|abundance|response|ppm|ppb", f["name"], re.I))
    likely = [f for f in out if f["numeric"] > 0.5 and value_like(f) and f["name"] not in ids]
    value = max(likely, key=lambda f: f["numeric"])["name"] if likely else (numeric or [None])[0]
    # a column's label: text (not a value's, however few of them are numbers)
    col_label = next((f["name"] for f in text if f["name"] not in ids and f["name"] != row_label and f["name"] != value
                      and not value_like(f) and f["numeric"] <= 0.1 and 1 < f["distinct"] < n and not links(f)), None)
    # one value per line: its rows' labels repeat (each row on several lines), as a matrix's do
    # not, and a column's label can be found (however many other columns have numbers, e.g.
    # properties of each row's chemical)
    looks_long = bool(row) and col_label is not None and 0 < row["distinct"] <= LONG_LABELS * n
    suggested = {"layout": "list" if looks_long else "matrix",
                 "rows": {"label": row_label, "keep": [k for k in ids if k != row_label]},
                 "cols": {"label": col_label, "keep": []}, "value": value, "empty": "missing"}
    return {"kind": "table", "dims": None, "fields": out, "suggested": suggested}


LONG_LABELS = 0.8  # a table whose rows' labels are no more than this share of its lines is taken for one value per line


def _vcf_suggested(names: list[str]) -> Optional[dict]:
    """The fields to make a matrix of a file that is VCF's, as made by the vcf repo's scripts
    (with Henry's law constants or vapor pressures, as the Headspace calculator takes it,
    `vcf.PROPERTY_COLUMNS`; and ``name`` and ``product_name``): a chemical's lines one row
    (told apart by its InChI and CID), a product as one paper (``reference``) gives it one
    column (with its group and category), the midpoints of quantities' ranges its values if
    the file has them (``mid_ppm``), and of its chemicals one odor, their amounts summed. None
    if it is not VCF's."""
    from . import vcf  # (which imports this module)

    have = set(names)
    if not have & set(vcf.PROPERTY_COLUMNS) or not {"name", "product_name"} <= have:
        return None
    return {"layout": "list", "rows": {"label": "name", "keep": [k for k in ("inchi", "cid") if k in have]},
            "cols": {"label": "product_name", "keep": [k for k in ("reference", "group", "category") if k in have]},
            "value": next((v for v in ("mid_ppm", "quantity") if v in have), None), "empty": "missing", "merge": "sum"}


def _fields(spec: Any) -> Optional[dict]:
    """Choices of fields (as `parse` takes them), as given in a request or stored (JSON):
    None for none; else only their known keys, of the expected types."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            return None
    if not isinstance(spec, dict):
        return None
    out: dict[str, Any] = {}
    if spec.get("layout") in LAYOUTS:
        out["layout"] = spec["layout"]
    for key in ("rows", "cols"):
        part = spec.get(key)
        if isinstance(part, dict):
            label = part.get("label")
            keep = [str(k) for k in part.get("keep") or [] if isinstance(k, str)]
            out[key] = {"label": str(label) if label else None, "keep": keep}
    if "value" in spec:
        out["value"] = str(spec["value"]) if spec["value"] else None
    if spec.get("empty") in ("zero", "missing"):
        out["empty"] = spec["empty"]
    if spec.get("merge") in MERGES:
        out["merge"] = spec["merge"]
    if out.get("layout") == "metacyc":  # a view and its options (checked by metacyc.options)
        out.update({k: spec[k] for k in ("view", "via", "classes") if isinstance(spec.get(k), str)})
        if "max_pathways" in spec:
            mp = spec["max_pathways"]
            out["max_pathways"] = mp if mp is None or (isinstance(mp, int) and not isinstance(mp, bool)) else None
    if out.get("layout") == "lotus":  # its options (checked by lotus.options)
        out.update({k: spec[k] for k in ("level", "value", "stereo", "taxonomy") if isinstance(spec.get(k), str)})
    return out or None


def _detect_kind(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size and np.isin(finite, (0, 1)).all():
        return "binary"
    return "diverging" if finite.size and finite.min() < 0 < finite.max() else "sequential"


def _source(path: Path) -> dict:
    st = os.stat(path)
    if not stat.S_ISREG(st.st_mode):
        raise ReferenceError("this is a folder, not a file" if stat.S_ISDIR(st.st_mode) else "not a file")
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _read_error(path: Path, e: Exception) -> str:
    """Why `path` could not be read, for people."""
    if isinstance(e, FileNotFoundError):
        return "not found on the server (with Docker, only folders mounted into the container are there)"
    if isinstance(e, PermissionError):
        if os.access(path, os.R_OK):  # the kernel (Landlock) refused, not the file's permissions
            return "the server may not read it now (if the file was replaced since the server started, restart it)"
        return "the server's user is not allowed to read it"
    if isinstance(e, (ReferenceError, OSError)):
        return str(e) if isinstance(e, ReferenceError) else f"could not be read ({e.strerror or e})"
    return f"could not be read as a table ({type(e).__name__}: {e})"


def _parsed(path: Path, chemical_axis: str, kind: str, fields: Optional[dict] = None,
            odors: Optional[list[dict]] = None) -> tuple[dict, np.ndarray | Sparse]:
    """`parse`, kept in memory for the file as it is now (for files not imported), and of a
    bundle, the odor table as it is."""
    mod = BUNDLES.get(bundle_kind(path))
    key = (str(path), json.dumps(_source(path)), chemical_axis, kind, json.dumps(fields, sort_keys=True),
           mod.odors_fingerprint(odors or []) if mod else None)
    with _memo_lock:
        if key in _memo:
            _memo.move_to_end(key)
            return _memo[key]
        out = _memo[key] = parse(path, chemical_axis, kind, fields, odors)
        while len(_memo) > MEMO_SIZE:
            _memo.popitem(last=False)
        return out


# --------------------------------------------------------------------- imports


# of an import's record of a file (not a bundle): one of an older version is read again
# (2: its items' fields kept, a level each in a download, see `_items`; 3: ``nonzero``)
RECORD_VERSION = 3


def cache_dir(ref_id: str) -> Path:
    return config.REFERENCE_CACHE_DIR / ref_id


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _odor_settings(path: Path, odors: Optional[list[dict]]) -> dict:
    """What of the odor table (and of how it is read) the import of a file depends on: of a
    bundle, which is made over the odor table's chemicals; else nothing."""
    kind = bundle_kind(path)
    if kind is None:
        return {}
    return {"odors": BUNDLES[kind].odors_fingerprint(odors or []), f"{kind}_version": BUNDLES[kind].VERSION}


def state(ref: dict, odors: Optional[list[dict]] = None) -> dict:
    """An added reference dataset's state, importing its file if it changed since it was
    last imported (or never was), or of a MetaCyc bundle, if the odor table (`odors`) did.

    Returns:
        ``state`` ('ok' | 'pending': readable once the server restarts | 'error'),
        ``error`` (why, for people), ``record`` (the import's, as from `parse`, plus
        ``imported_at``; None unless ok).
    """
    path = Path(ref["path"])
    if not sandbox.can_read(path):
        if sandbox.readable_after_restart(path):
            return {"state": "pending", "error": "the server reads it once it restarts (Admin tab)", "record": None}
        return {"state": "error", "record": None,
                "error": "not in the folders the server may read: add it (or a folder it is in) in the Admin tab, and restart"}
    try:
        source = _source(path)
    except (OSError, ReferenceError) as e:
        return {"state": "error", "error": _read_error(path, e), "record": None}
    d = cache_dir(ref["id"])
    fields = _fields(ref.get("fields"))
    settings = {"source": source, "chemical_axis": ref["chemical_axis"], "kind_setting": ref["kind"], "fields": fields,
                **_odor_settings(path, odors), **({} if bundle_kind(path) else {"record_version": RECORD_VERSION})}
    with _lock:
        try:
            record = json.loads((d / "import.json").read_text())
        except (OSError, ValueError):
            record = None
        if record is None or any(record.get(k) != v for k, v in settings.items()):
            imported_at = dt.datetime.now().isoformat(timespec="seconds")
            try:
                record, values = parse(path, ref["chemical_axis"], ref["kind"], fields, odors)
            except Exception as e:  # remembered for the file as it is, so as not to be tried again
                record, values = {"error": _read_error(path, e)}, None
            record.update(settings, imported_at=imported_at)
            d.mkdir(parents=True, exist_ok=True)
            if values is not None:
                _write_values(d, values, record)
            _write(d / "import.json", json.dumps(record).encode())
    if record.get("error"):
        return {"state": "error", "error": record["error"], "record": None}
    return {"state": "ok", "error": None, "record": record}


def _write_values(d: Path, values: np.ndarray | Sparse, record: dict) -> None:
    """An import's values written to its folder `d`: dense (``values.f32``), or of a `Sparse`
    matrix its cells other than 0 (``values.npz``, ``storage`` 'sparse' in its record); the
    other file, if any, deleted."""
    if isinstance(values, Sparse):
        buf = io.BytesIO()
        np.savez(buf, index=values.index.astype("<i8"), values=values.values.astype("<f4"))
        _write(d / "values.npz", buf.getvalue())
        record["storage"] = "sparse"
        (d / "values.f32").unlink(missing_ok=True)
    else:
        _write(d / "values.f32", values.astype("<f4").tobytes())
        (d / "values.npz").unlink(missing_ok=True)


def _values(ref_id: str, record: dict) -> np.ndarray | Sparse:
    if record.get("storage") == "sparse":
        with np.load(cache_dir(ref_id) / "values.npz") as z:
            return Sparse(tuple(record["shape"]), z["index"], z["values"])
    return np.fromfile(cache_dir(ref_id) / "values.f32", dtype="<f4").reshape(record["shape"])


# ------------------------------------------------------------------ chemicals


def _odor_string(label: str) -> Optional[Odor]:
    """A label as a dataset writes an odor ('<abbreviation> @ <log10 conc>', `parse_odor`),
    read; None if it is not one (or is the solvent)."""
    try:
        o = parse_odor(label, strict=False)
    except OdorFormatError:
        return None
    return None if o.is_solvent else o


def odor_axis(record: dict) -> Optional[str]:
    """The axis whose labels are odors as datasets write them (`_odor_string`): at least 3 of
    them, and more than half of its labels; of both, the one with more. None if neither."""
    best, most = None, 0
    for axis in ("rows", "cols"):
        k = sum(_odor_string(it["label"]) is not None for it in record[axis])
        if k >= 3 and 2 * k > len(record[axis]) and k > most:
            best, most = axis, k
    return best


def chemicals(record: dict, index: odormeta.OdorIndex) -> dict:
    """A reference dataset's chemicals matched to the odor table (`OdorIndex.match_chemical`).

    Of a bundle's view, they are as its module matched them (its record's ``matched``),
    each identified by the bundle's ID of it. An axis whose labels are odors as datasets
    write them (`odor_axis`: '<abbreviation> @ <log10 conc>') is its chemicals (unless
    another is said), matched as a dataset's odors are: by their abbreviations
    (`OdorIndex.match`).

    Returns:
        ``axis`` ('rows' | 'cols' | None), ``items`` (per item of that axis: ``odor_id``,
        ``how``, the odor's ``name`` and ``abbrev`` if matched, its ``cid`` and
        ``inchi``, from the file or else the odor table, whether it is ``identified``,
        ``conc``: its log10 concentration as written, if an odor as datasets write
        them, ``candidates``: the odors it could be, if ambiguous, and ``maybe``: those
        it may be by its name, if not identified, `OdorIndex.maybe`), ``n``, ``matched`` (to the odor table), ``identified`` (with a
        CID or InChI, or a bundle's ID), ``unidentified`` and ``ambiguous`` (labels), and
        ``unrecognized`` (looked for but none found: no identifiers, and no labels the odor
        table knows), and ``odors`` (its labels odors as datasets write them).
    """
    out = {"axis": None, "items": [], "n": 0, "matched": 0, "identified": 0, "unidentified": [], "ambiguous": [],
           "unrecognized": False, "odors": False}
    setting, given = record["chemical_axis"], record["ids"]
    if setting == "none":
        return out
    odors_on = odor_axis(record) if record.get("matched") is None else None

    def match(axis: str) -> list[tuple[Optional[int], str, list[int]]]:
        if record.get("matched") is not None:  # (as the bundle's module matched them)
            return [(m[0], m[1], []) if m and m[0] in index.by_id else (None, "none", []) for m in record["matched"]]
        ids = given[axis] or [{}] * len(record[axis])
        found = []
        for i, it in zip(ids, record[axis]):
            o = _odor_string(it["label"]) if axis == odors_on else None
            cands: list[int] = []
            # odors as datasets write them: by their abbreviations, as a dataset's odors are matched (but by
            # identifiers given for one, if any)
            if o is not None and not i:
                found.append(index.match(o.name))
            else:
                found.append((*index.match_chemical(i, o.name if o else it["label"], cands), cands))
        return found

    if setting in ("rows", "cols"):
        axis = setting
    elif given["rows"] or given["cols"]:
        axis = "rows" if given["rows"] else "cols"
    elif odors_on:  # (its labels odors as datasets write them, e.g. its other axis glomeruli)
        axis = odors_on
    else:  # by labels alone: the axis more of whose labels the odor table knows
        found = {a: sum(m[0] is not None for m in match(a)) for a in ("rows", "cols")}
        if not any(found.values()):
            return {**out, "unrecognized": True}
        axis = "rows" if found["rows"] >= found["cols"] else "cols"
    items, bundled = [], record.get("matched") is not None
    for ids, it, (odor_id, how, cands) in zip(given[axis] or [{}] * len(record[axis]), record[axis], match(axis)):
        o = index.by_id.get(odor_id) or {}
        given_odor = _odor_string(it["label"]) if axis == odors_on else None
        # (a bundle's chemical matched as the same but for charge or stereochemistry: the odor's)
        first = (o, ids) if bundled else (ids, o)
        cid, inchi = first[0].get("cid") or first[1].get("cid"), first[0].get("inchi") or first[1].get("inchi")
        identified = bool(cid or inchi) or bundled
        items.append({"odor_id": odor_id, "how": how, "name": o.get("name"), "abbrev": o.get("abbrev"), "cid": cid, "inchi": inchi,
                      "identified": identified, "conc": format_conc(given_odor.log10_conc) if given_odor else None,
                      "candidates": cands if how == "ambiguous" else [],
                      "maybe": index.maybe(it["label"]) if not identified and odor_id is None and not given_odor else []})
        if how == "ambiguous":
            out["ambiguous"].append(it["label"])
        if not identified:
            out["unidentified"].append(it["label"])
    return {**out, "axis": axis, "items": items, "n": len(items), "matched": sum(i["odor_id"] is not None for i in items),
            "identified": sum(i["identified"] for i in items), "odors": axis == odors_on}


def glomeruli(record: dict) -> dict:
    """A reference dataset's glomeruli: on the axis more of whose labels are glomerulus names
    (as accepted for datasets' ROIs, but for case), if at least 3 of them are, and at least
    half of that axis.

    Returns:
        ``axis`` ('rows' | 'cols' | None), ``names`` (per item of that axis, the glomerulus
        it is, or None), ``n`` (its items) and ``matched`` (those that are glomeruli).
    """
    best = {"axis": None, "names": [], "n": 0, "matched": 0}
    for axis in ("rows", "cols"):
        names = [it["label"] if glom.is_glomerulus(it["label"]) else glom.same_but_case(it["label"]) for it in record[axis]]
        k = sum(n is not None for n in names)
        if k >= 3 and 2 * k >= len(names) and k > best["matched"]:
            best = {"axis": axis, "names": names, "n": len(names), "matched": k}
    return best


def warning(chem: dict, glomeruli: Optional[dict] = None) -> Optional[str]:
    """The yellow warning of a matrix whose chemicals are not all identified (None if they
    are, or if it has none, but glomeruli)."""
    if chem["unrecognized"] and glomeruli and glomeruli["axis"]:
        return None
    if chem["unrecognized"]:
        return ("no chemicals recognized: no identifier columns or rows (CID, InChI, ...), and no labels the Odors "
                "table knows (if it has none, say so when adding it)")
    if chem.get("odors") and chem["matched"] < chem["n"]:
        k = chem["n"] - chem["matched"]
        return (f"{k} of its {chem['n']} odors {'is' if k == 1 else 'are'} not one odor in the Odors table (by abbreviation, "
                f"as a dataset's are): add {'it' if k == 1 else 'them'} to the odor sheet and sync it")
    n = len(chem["unidentified"])
    if n:
        return f"{n} of {chem['n']} chemicals could not be identified: no CID or InChI, in the file or the Odors table"
    return None


def _pct(a: int, b: int) -> str:
    p = 100 * a / b if b else 0
    return f"{p:.0f}%" if p >= 10 or p == 0 else f"{p:.1f}%" if p >= 1 else f"{p:.2g}%"


def stereo_warning(chem: dict, record: dict, short: bool = False) -> Optional[str]:
    """The warning of a matrix some of whose chemicals are the Odors table's only as the same
    but for stereochemistry (`OdorIndex.match_chemical`'s 'stereo'): how many of the odors it
    has are (each once, however many of its items are one: as the Compare tab merges them),
    which (unless `short`), and how many of its values (not 0 or missing) are theirs. None if
    none are."""
    items, axis = chem["items"], chem["axis"]
    stereo = [k for k, it in enumerate(items) if it["how"] == "stereo"]
    if not stereo:
        return None
    n = len({it["odor_id"] for it in items if it["odor_id"] is not None})
    odors = {items[k]["odor_id"] for k in stereo}
    counts = (record.get("nonzero") or {}).get(axis)
    of_odors = sum(counts[k] for k, it in enumerate(items) if it["odor_id"] is not None) if counts else 0
    mine = sum(counts[k] for k in stereo) if counts else 0
    rows = "rows" if axis == "rows" else "columns"
    if short:
        return (f"{len(odors)} of its {n} odors of the Odors table matched only as the same but for stereochemistry"
                + (f" ({_pct(mine, of_odors)} of their values)" if counts else ""))
    which = {}
    for k in stereo:
        which.setdefault(items[k]["name"], []).append(record[axis][k]["label"])
    shown = "; ".join(f"{name}, as {', '.join(labels)}" for name, labels in which.items())
    out = (f"{len(odors)} of the {n} odors of the Odors table it has (each once, its {rows} of one merged) "
           f"{'is' if len(odors) == 1 else 'are'} matched only as the same but for stereochemistry that one of the two "
           f"leaves unspecified: {shown}.")
    if counts:
        total = sum(counts)
        out += (f" Its {rows} so matched have {mine:,} of the {of_odors:,} values (not 0 or missing) of those {n} "
                f"({_pct(mine, of_odors)}), and of its {total:,} ({_pct(mine, total)}).")
    return out


VALUES_NOT_NUMBERS = 0.1  # a value field more of whose lines than this are not numbers is warned of


def value_warning(record: dict, value: Optional[str]) -> Optional[str]:
    """Of a table of one value per line, a warning if many of the lines of its value field
    (`value`) are not numbers (then missing), as VCF's ``quantity`` (ranges, 'trace') whose
    ``mid_ppm`` has numbers for them all; None if few are."""
    g = record.get("values_given")
    if not g or not value or g["not_numbers"] <= VALUES_NOT_NUMBERS * g["lines"]:
        return None
    return (f"{_pct(g['not_numbers'], g['lines'])} of its lines' {value!r} are not numbers ({g['not_numbers']:,} of "
            f"{g['lines']:,}, e.g. {g['examples']}), so their cells are missing: is another field the numbers of them "
            f"(e.g. the midpoints of ranges)?")


def may_overlap(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two chemicals that are one odor may be one substance, or one may include the
    other (a total of its stereoisomers given beside one of them), rather than distinct ones
    (then summing them counts some twice): of the InChIs given, `odormeta.stereo_compatible`
    ones, or either with none."""
    return not a or not b or odormeta.stereo_compatible(a, b)


def unresolved(chem: dict, record: dict, values: Optional[np.ndarray]) -> list[dict]:
    """The odors of a reference dataset's chemicals (`chemicals`) whose values may be wrong in
    ways not resolved, each once: ``odor_id``; ``overlap`` (where two of its items that
    `may_overlap` both have a value, not 0 or missing: how many ``cells`` of its other axis,
    and which items, by their ``labels``; of `values`, the dataset's, if given); ``maybe``
    (the labels of chemicals not identified that may be it by their names, `OdorIndex.maybe`:
    not counted as it); ``ambiguous`` (those of chemicals that could be it or another:
    counted as neither). Of those with any."""
    items, axis = chem["items"], chem["axis"]
    labels = [it["label"] for it in record[axis]] if axis else []
    out: dict[int, dict] = {}

    def of(odor_id: int) -> dict:
        return out.setdefault(odor_id, {"odor_id": odor_id, "overlap": {"cells": 0, "labels": []}, "maybe": [], "ambiguous": []})

    groups: dict[int, list[int]] = {}
    for k, it in enumerate(items):
        if it["odor_id"] is not None:
            groups.setdefault(it["odor_id"], []).append(k)
        for i in it["maybe"]:
            of(i)["maybe"].append(labels[k])
        for i in it["candidates"]:
            of(i)["ambiguous"].append(labels[k])
    if isinstance(values, np.ndarray):
        for odor_id, ks in groups.items():
            if len(ks) < 2:
                continue
            held = {k: np.isfinite(v) & (v != 0) for k in ks for v in [values[k] if axis == "rows" else values[:, k]]}
            both = np.zeros(values.shape[1 if axis == "rows" else 0], dtype=bool)
            involved = set()
            for x, a in enumerate(ks):
                for b in ks[x + 1:]:
                    if may_overlap(items[a]["inchi"], items[b]["inchi"]):
                        cells = held[a] & held[b]
                        if cells.any():
                            both |= cells
                            involved |= {a, b}
            if both.any():
                of(odor_id)["overlap"] = {"cells": int(both.sum()), "labels": [labels[k] for k in ks if k in involved]}
    return [u for u in out.values() if u["overlap"]["cells"] or u["maybe"] or u["ambiguous"]]


def unresolved_warning(chem: dict, record: dict, values: Optional[np.ndarray], index: odormeta.OdorIndex,
                       merge: str = MERGES[0]) -> Optional[str]:
    """`unresolved`, said (None if none)."""
    found = unresolved(chem, record, values)
    if not found:
        return None
    other = "columns" if chem["axis"] == "rows" else "rows"
    parts = []
    for u in found:
        said = []
        if u["overlap"]["cells"]:
            o = u["overlap"]
            said.append(f"in {o['cells']:,} of its {other}, {', '.join(o['labels'])}, which may be one another's total or the "
                        f"same, not distinct isomers ({merge_said(merge)}{', so counted twice if so' if merge == 'sum' else ''})")
        if u["maybe"]:
            said.append(f"{', '.join(u['maybe'])}, of no structure given, may be it (not counted as it)")
        if u["ambiguous"]:
            said.append(f"{', '.join(u['ambiguous'])} could be it or another (counted as neither)")
        parts.append(f"{index.by_id[u['odor_id']]['name']}: {'; '.join(said)}")
    n = len(found)
    return (f"{n} odor{'s' if n > 1 else ''} of the Odors table it has, or may have, may be miscounted, unresolved: "
            + ". ".join(parts) + ".")


def merge_said(merge: str) -> str:
    return {"sum": "summed", "mean": "averaged", "max": "the highest taken"}.get(merge, merge)


def metacyc_source(db, ref_id: str) -> Optional[tuple[Path, Optional[dict]]]:
    """`bundle_source` of a MetaCyc bundle."""
    return bundle_source(db, ref_id, "metacyc")


def bundle_source(db, ref_id: str, kind: str) -> Optional[tuple[Path, Optional[dict]]]:
    """A reference dataset of a bundle of `kind`, without reading more of it than its kind:
    (its bundle, its view's fields); None if there is no such reference dataset, or it is
    not of a bundle of that kind.

    Raises:
        ReferenceError: if the server may not read its bundle (now).
    """
    e = next((e for e in manifest() if e["id"] == ref_id), None)
    if e is not None:
        path = config.REFERENCE_DIR / e["file"]
    else:
        e = next((r for r in db.list_references() if r["id"] == ref_id), None)
        if e is None:
            return None
        path = Path(e["path"])
        if not sandbox.can_read(path):
            raise ReferenceError("the server may not read its file now (see the Admin tab)")
    if bundle_kind(path) != kind:
        return None
    return path, _fields(e.get("fields"))


# ---------------------------------------------------------------------- listing


def manifest() -> list[dict]:
    """The built-in reference datasets, from reference_data/manifest.json."""
    p = config.REFERENCE_DIR / "manifest.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [dict(d) for d in data.get("datasets", []) if d.get("id") and d.get("file")]


def _builtin_state(entry: dict, odors: Optional[list[dict]] = None) -> dict:
    path = config.REFERENCE_DIR / entry["file"]
    try:
        record, _ = _parsed(path, entry.get("chemical_axis", "detect"), entry.get("kind") or "detect", _fields(entry.get("fields")), odors)
    except Exception as e:
        error = "file missing from reference_data/" if isinstance(e, FileNotFoundError) else _read_error(path, e)
        return {"state": "error", "error": error, "record": None}
    return {"state": "ok", "error": None, "record": record}


def entries(db, odors: Optional[list[dict]] = None) -> list[dict]:
    """Every reference dataset, built in then added: its settings, ``origin`` ('builtin' |
    'admin') and `state` (built-in ones are read when asked for, and never pending). The
    built-in examples (``"example": true`` in the manifest) only while there is no other.
    Bundles' views are made over `odors` (the odor table, read from `db` if not given)."""
    builtin, added = manifest(), db.list_references()
    if added or any(not e.get("example") for e in builtin):
        builtin = [e for e in builtin if not e.get("example")]
    if odors is None and any(is_bundle_path(Path(e.get("file") or e.get("path") or "")) for e in builtin + added):
        odors = db.list_odors()
    out = [{**e, "origin": "builtin", "path": str(config.REFERENCE_DIR / e["file"]), **_builtin_state(e, odors)} for e in builtin]
    out += [{**r, "origin": "admin", **state(r, odors)} for r in added]
    return out


def listing(db, index: odormeta.OdorIndex) -> list[dict]:
    """`entries` for the list in the Reference data tab and the Admin tab."""
    out = []
    for e in entries(db, list(index.by_id.values())):
        rec = e.pop("record")
        item = {k: e.get(k) for k in ("id", "title", "description", "origin", "path", "state", "error", "source",
                                      "read_grant", "created_at", "value_label", "row_label", "col_label", "chemical_axis")}
        item["kind_setting"] = e.get("kind")
        item["fields"] = _fields(e.get("fields"))  # (as chosen: to change them in the Admin tab)
        item["warning"] = item["stereo"] = None
        if rec:
            chem, gl = chemicals(rec, index), glomeruli(rec)
            item.update(kind=rec["kind"], shape=rec["shape"], notes=rec["notes"], imported_at=rec.get("imported_at"),
                        pairwise=rec.get("pairwise"), metacyc=rec.get("metacyc"), lotus=rec.get("lotus"),
                        chemicals={**{k: chem[k] for k in ("axis", "n", "matched", "identified")},
                                   "odor_ids": [i["odor_id"] for i in chem["items"] if i["odor_id"] is not None]},
                        glomeruli={"axis": gl["axis"], "n": gl["n"], "matched": gl["matched"],
                                   "names": [n for n in gl["names"] if n is not None]},
                        warning=warning(chem, gl), stereo=stereo_warning(chem, rec, short=True))
        out.append(item)
    return out


def opened(db, ref_id: str, index: odormeta.OdorIndex) -> Optional[dict[str, Any]]:
    """A reference dataset as read (None if there is none by that id): its entry (`entries`),
    ``meta``, ``state`` and ``error``, and if it can be read, its ``record``, ``values`` (an
    array, or `Sparse`), ``chem`` (`chemicals`) and ``gl`` (`glomeruli`)."""
    odors = list(index.by_id.values())
    e = next((e for e in entries(db, odors) if e["id"] == ref_id), None)
    if e is None:
        return None
    meta = {k: e.get(k) for k in ("id", "title", "description", "origin", "path", "value_label", "row_label", "col_label", "source")}
    if e["state"] != "ok":
        return {"entry": e, "meta": meta, "state": e["state"], "error": e["error"]}
    record = e["record"]
    if e["origin"] == "admin":
        values = _values(ref_id, record)
    else:
        values = _parsed(Path(e["path"]), e.get("chemical_axis", "detect"), e.get("kind") or "detect", _fields(e.get("fields")), odors)[1]
    return {"entry": e, "meta": meta, "state": "ok", "error": None, "record": record, "values": values,
            "chem": chemicals(record, index), "gl": glomeruli(record)}


def load(db, ref_id: str, index: odormeta.OdorIndex) -> Optional[dict[str, Any]]:
    """A reference dataset for the viewer (None if there is none by that id; ``error`` set if it cannot be read)."""
    o = opened(db, ref_id, index)
    if o is None or o["state"] != "ok":
        return o and {k: o[k] for k in ("meta", "state", "error")}
    meta, record, values, chem, gl = o["meta"], o["record"], o["values"], o["chem"], o["gl"]
    merge = (_fields(o["entry"].get("fields")) or {}).get("merge", MERGES[0])
    meta["kind"] = record["kind"]
    meta["metacyc"] = record.get("metacyc")
    meta["lotus"] = record.get("lotus")
    meta["scale"] = lotus.scale(lotus.options(record["lotus"])) if record.get("lotus") else None
    rows = [{k: v for k, v in r.items() if k != "fields"} for r in record["rows"]]  # (the fields: for downloads)
    cols = [{k: v for k, v in c.items() if k != "fields"} for c in record["cols"]]
    if chem["axis"]:
        for it, c in zip(rows if chem["axis"] == "rows" else cols, chem["items"]):
            it["chem"] = {k: c[k] for k in ("odor_id", "name", "abbrev", "cid", "how", "identified")}
            if c["conc"] is not None:  # (an odor as datasets write them)
                it["chem"]["conc"] = c["conc"]
        if record.get("pairwise"):  # (its columns are its rows)
            for it, r in zip(cols, rows):
                it["chem"] = r["chem"]
    if gl["axis"]:
        for it, name in zip(rows if gl["axis"] == "rows" else cols, gl["names"]):
            it["glomerulus"] = name
    if isinstance(values, Sparse):  # (its cells other than 0)
        vrange, encoded = values.range(), {"values_sparse": values.payload()}
    else:
        finite = values[np.isfinite(values)]
        vrange = (float(finite.min()), float(finite.max())) if finite.size else None
        encoded = {"values_b64": base64.b64encode(values.astype("<f4").tobytes()).decode("ascii")}
    return {
        "meta": meta, "state": "ok", "rows": rows, "cols": cols, "shape": list(values.shape), "dtype": "float32", **encoded,
        "vmin": vrange[0] if vrange else None, "vmax": vrange[1] if vrange else None,
        "chemicals": {**{k: v for k, v in chem.items() if k != "items"}, "merge": merge}, "warning": warning(chem, gl),
        "stereo_warning": stereo_warning(chem, record),
        "unresolved_warning": unresolved_warning(chem, record, values if isinstance(values, np.ndarray) else None, index, merge),
        "notes": record["notes"],
        "glomeruli": {k: gl[k] for k in ("axis", "n", "matched")},
        "imported_at": record.get("imported_at"),
        **({"square": True, "pairwise": record["pairwise"]} if record.get("pairwise") else {}),
    }


# ------------------------------------------------------------------ Admin tab


def _described(path: Path) -> dict:
    """`describe_fields`, kept in memory for the file as it is now."""
    key = ("fields", str(path), json.dumps(_source(path)))
    with _memo_lock:
        if key in _memo:
            _memo.move_to_end(key)
            return _memo[key]
        out = _memo[key] = describe_fields(path)
        while len(_memo) > MEMO_SIZE:
            _memo.popitem(last=False)
        return out


def check(path_input: str, chemical_axis: str, kind: str, index: odormeta.OdorIndex, supervisor: Optional[str],
          fields: Any = None) -> dict:
    """Check a file entered in the Admin tab as a reference dataset (`path_input`: absolute on
    the server, or else from the top of tensor), made into a matrix by `fields` (as `parse`
    takes them; else as suggested for it, see `describe_fields`).

    Returns:
        ``input``, ``path`` (normalized, or None), ``ok`` (it can be added), ``error``,
        ``readable`` (now: then ``fields``, its fields as `describe_fields` gives them,
        with ``used``: those choices; and if it can be made into a matrix by them,
        ``preview`` has its ``shape``, ``kind``, ``notes``, ``chemicals``, ``warning``,
        ``stereo_warning`` and ``value_warning``), ``grant`` (the path adding it would add to the folders the server
        may read), ``restart`` (whether it is read only after a restart, then with the
        ``steps`` it takes, see `sandbox.access_steps`).
    """
    out = {"input": path_input, "path": None, "ok": False, "error": None, "readable": False, "preview": None,
           "grant": None, "restart": False, "steps": []}
    given = path_input.strip()
    if given and not given.startswith("/"):  # (from the top of tensor, as paths on tensor are given elsewhere)
        given = f"{str(config.TENSOR_ROOT).rstrip('/')}/{given}"
    try:
        p = out["path"] = sandbox.normalize(given)
    except sandbox.SandboxError as e:
        return {**out, "error": str(e)}
    if Path(p).suffix.lower() not in SUFFIXES:
        return {**out, "error": NOT_A_TABLE}
    if not sandbox.can_read(p):
        if os.path.isdir(p):
            return {**out, "error": "this is a folder, not a file"}
        grant = None if sandbox.readable_after_restart(p) else p
        return {**out, "ok": True, "grant": grant, "restart": True, "steps": sandbox.access_steps(p, supervisor)}
    try:
        described = _described(Path(p))
    except Exception as e:
        return {**out, "error": _read_error(Path(p), e)}
    used = _fields(fields) or described["suggested"]
    if described["kind"] in BUNDLES:  # a bundle's view, its options checked
        k = described["kind"]
        used = {**BUNDLES[k].options(used if used.get("layout") == k else None), "layout": k}
    elif used.get("layout") in BUNDLES:  # (chosen for a bundle, and now another file)
        used = described["suggested"]
    out = {**out, "readable": True, "fields": {**described, "used": used}}
    try:
        record, _ = _parsed(Path(p), chemical_axis, kind, used, list(index.by_id.values()))
    except Exception as e:
        return {**out, "error": _read_error(Path(p), e)}
    chem, gl = chemicals(record, index), glomeruli(record)
    preview = {"shape": record["shape"], "kind": record["kind"], "notes": record["notes"], "warning": warning(chem, gl),
               "stereo_warning": stereo_warning(chem, record), "value_warning": value_warning(record, used.get("value")),
               "pairwise": record.get("pairwise"), "metacyc": record.get("metacyc"), "lotus": record.get("lotus"),
               "chemicals": {k: chem[k] for k in ("axis", "n", "matched", "identified", "unidentified", "ambiguous", "odors")},
               "glomeruli": {k: gl[k] for k in ("axis", "n", "matched")},
               # (its odors as datasets write them, to be checked against the odor table as a dataset's are)
               "odor_labels": [it["label"] for it in record[chem["axis"]]] if chem["odors"] else None}
    return {**out, "ok": True, "readable": True, "preview": preview}


def _slug(title: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60].strip("-") or "reference"
    slug, n = base, 2
    while slug in taken:
        slug, n = f"{base}-{n}", n + 1
    return slug


def add(db, body: dict, index: odormeta.OdorIndex, supervisor: Optional[str]) -> tuple[int, dict]:
    """Add a reference dataset entered in the Admin tab: (status, response). A file the
    server may not read yet is added to the folders it may read (as itself), to be read
    once it restarts."""
    fields = {k: (str(body.get(k) or "").strip() or None) for k in ("title", "description", "value_label", "row_label",
                                                                    "col_label", "source")}
    kind, axis = body.get("kind") or "detect", body.get("chemical_axis") or "detect"
    errors = []
    if not fields["title"]:
        errors.append("give it a title")
    if kind not in KINDS or axis not in AXES:
        errors.append("unknown colors or chemicals setting")
    c = check(str(body.get("path") or ""), axis, kind, index, supervisor, body.get("fields"))
    if not c["ok"]:
        errors.append(f"the file: {c['error']}")
    if errors:
        return 422, {"ok": False, "errors": errors, "check": c}
    used = _bundle_fields(c["path"], c["fields"]["used"] if c.get("fields") else _fields(body.get("fields")))
    taken = {e["id"] for e in manifest()} | {r["id"] for r in db.list_references()}
    ref = {**fields, "id": _slug(fields["title"], taken), "created_at": dt.datetime.now().isoformat(timespec="seconds"),
           "path": c["path"], "kind": kind, "chemical_axis": axis, "read_grant": None, "fields": json.dumps(used) if used else None}
    if not c["readable"]:
        ref["read_grant"] = sandbox.grant_read(c["path"])
    db.add_reference(ref)
    s = state(ref, list(index.by_id.values()))
    return 200, {"ok": True, "id": ref["id"], "state": s["state"], "error": s["error"], "grant": ref["read_grant"],
                 "restart": c["restart"], "steps": c["steps"]}


def _bundle_fields(path: str, used: Optional[dict]) -> Optional[dict]:
    """The fields of a bundle's view, its options checked (of a bundle the server cannot
    read yet, as its layout says; a MetaCyc one's, if it says none)."""
    if not used or not is_bundle_path(Path(path)):
        return used
    kind = bundle_kind(Path(path)) or (used.get("layout") if used.get("layout") in BUNDLES else "metacyc")
    return {**BUNDLES[kind].options(used if used.get("layout") == kind else None), "layout": kind}


EDITABLE = ("title", "description", "value_label", "row_label", "col_label", "source")


def update(db, ref_id: str, body: dict, index: odormeta.OdorIndex, supervisor: Optional[str]) -> tuple[int, dict]:
    """Change the settings of a reference dataset added in the Admin tab (any of those `add`
    takes but its file; those not in `body` are kept): (status, response). Its file is read
    again by them (now, if the server may read it)."""
    ref = next((r for r in db.list_references() if r["id"] == ref_id), None)
    if ref is None:
        if any(e["id"] == ref_id for e in manifest()):
            return 409, {"ok": False, "errors": ["it is built in: change it in reference_data/manifest.json on the server"]}
        return 404, {"ok": False, "errors": ["reference dataset not found"]}
    new = dict(ref)
    for k in EDITABLE:
        if k in body:
            new[k] = str(body.get(k) or "").strip() or None
    if "kind" in body:
        new["kind"] = body.get("kind") or "detect"
    if "chemical_axis" in body:
        new["chemical_axis"] = body.get("chemical_axis") or "detect"
    errors = []
    if not new["title"]:
        errors.append("give it a title")
    if new["kind"] not in KINDS or new["chemical_axis"] not in AXES:
        errors.append("unknown colors or chemicals setting")
    fields = body["fields"] if "fields" in body else _fields(ref.get("fields"))
    c = check(ref["path"], new["chemical_axis"], new["kind"], index, supervisor, fields)
    if not c["ok"]:
        errors.append(f"the file: {c['error']}")
    if errors:
        return 422, {"ok": False, "errors": errors, "check": c}
    used = _bundle_fields(c["path"], c["fields"]["used"] if c.get("fields") else _fields(fields))
    new["fields"] = json.dumps(used) if used else None
    db.update_reference(new)
    s = state(new, list(index.by_id.values()))
    return 200, {"ok": True, "id": ref_id, "state": s["state"], "error": s["error"], "restart": c["restart"], "steps": c["steps"]}


def delete(db, ref_id: str) -> tuple[int, dict]:
    """Delete a reference dataset added in the Admin tab: its import (never its file), and
    the path added for it to the folders the server may read, if no other one needs that:
    (status, response)."""
    refs = db.list_references()
    ref = next((r for r in refs if r["id"] == ref_id), None)
    if ref is None:
        if any(e["id"] == ref_id for e in manifest()):
            return 409, {"error": "it is built in: remove it from reference_data/manifest.json on the server"}
        return 404, {"error": "reference dataset not found"}
    with _lock:
        db.delete_reference(ref_id)
        shutil.rmtree(cache_dir(ref_id), ignore_errors=True)
    grant = ref["read_grant"]
    others = [r for r in refs if r["id"] != ref_id]
    revoked = bool(grant) and not any(grant in (r["read_grant"], r["path"]) for r in others) and sandbox.revoke_read(grant)
    return 200, {"ok": True, "revoked": grant if revoked else None}
