"""Loading + validating uploaded response matrices (CSV or parquet).

Accepted layouts
----------------
*wide* (what ``DataFrame.to_csv`` / ``to_parquet`` write for a frame with MultiIndex
rows and columns):

    columns: MultiIndex with levels  date, fly_num, [any others,] roi
    rows:    MultiIndex with levels  panel, [is_pair,] odor1[, odor2, ...], repeat
             (``odor`` alone is accepted in place of ``odor1``)

Dates may be written any way pandas reads one (e.g. 2024-07-10), or as YYYYMMDD
(20240710, as text or a number), each value on its own (`date_str`).

Column levels besides date, fly_num and roi (e.g. ``hemibrain``: L or R) are kept with
each ROI (its ``extra``): ROIs of the same fly and name but different values of them are
different ROIs (e.g. cells of the two sides of the brain of one fly). ``bad`` is not one
of them (see below).

The ROI level may be called ``cells`` instead of ``roi``: its values are then KC ids,
which must be non-negative integers, every one of each fly's from 0 or 1 to its highest
(`kc_id_problems`): all of a fly's KCs, those not to be used marked so in a level ``bad``.

Any of these may have one more column level, ``bad`` (True or False): whether each ROI
is one not to be used (hidden by default where KCs are shown, left out of comparisons).

Every ROI's fly (date and fly_num) and every trial's number (``repeat``, or ``trial``:
integers) are needed: a file without them is refused, saying what it lacks. Files of ROIs
alone are recognised only to say so: in a CSV, a single header line of the row levels'
names then the ROIs' (as ``to_csv`` writes a frame with plain columns).

*long* (one row per value):

    columns: date, fly_num, roi (or cells)[, bad], panel, odor1[, odor2, ...], repeat, value

  (any other columns being more levels of the ROIs, as in the wide layout, but for
  ``Unnamed: <n>`` ones, as ``to_csv`` writes an index, which are ignored)

(``trial`` is read as ``repeat``, in either layout.)

An odor level may hold a mixture itself (``a @ -3 + b @ -5``), split into its components
as if they were in levels of their own. Odors written ``name(conc)`` are read as
``name @ conc``, and a solvent alone (``PFO``, ``water``) as at 0 (``pfo @ 0``), saying so
(`rewritten_odor`). ROI names lose a ``_t<n>`` suffix (``DA1_t0`` -> ``DA1``), and may name
several glomeruli that could not be told apart (``VA1d/VA1v``: `glomeruli.merged`).

Row levels that say nothing are dropped before anything else (`uninformative_levels`):
``is_pair`` when it is False on every row, and the last ``odor<n>`` levels when they hold
nothing but one of missing / 'solvent' / 'pfo' / 'water' / 'pfo @ 0' / 'water @ 0'.

Validation collects *all* problems it can find and raises :class:`ValidationError`
with the full list, so a user can fix a file in one go.
"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import itertools
import math
import re
import warnings as warnings_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import glomeruli as glom
from .odors import (
    COMPONENT_DELIM,
    SOLVENT_STR,
    OdorFormatError,
    format_conc,
    format_mix_from_strs,
    is_odor_level,
    odor_level_sort_key,
    parse_odor,
    validate_odor_levels,
)

COL_LEVELS = ("date", "fly_num", "roi")
CELLS = "cells"  # the roi level, by another name: its values KC ids (integers)
BAD = "bad"  # a column level that may be added: whether each ROI is not to be used (True/False)
MAX_KC_PROBLEMS = 5  # flies whose KC ids have gaps, named in that error
TRIAL = "trial"  # the repeat level, by another name
# what a file needs of flies and trials, said when it lacks them
FLIES_NEEDED = (f"each ROI's fly is needed, by its date and its number that day: in a wide file, header rows date, fly_num, "
                f"then roi (or {CELLS}; and {BAD} if any) before the row naming the row levels, as pandas' to_csv writes a "
                "frame whose columns are a MultiIndex of those levels; in a long file, columns date and fly_num")
PANEL_NEEDED = ("each trial's panel is needed (the name of the set of odors presented in that experiment, e.g. megamat; "
                "which odors it has may vary somewhat from experiment to experiment), in a row level (in a long file, a "
                "column) named panel, on every row")
TRIALS_NEEDED = (f"each trial's number is needed, in a row level (in a long file, a column) named repeat (or {TRIAL}) of "
                 "integers (0, 1, 2... or 1, 2, 3...): one row per trial, not averaged over trials")
KNOWN_ROW_LEVELS = {"panel", "is_pair", "repeat"}
# names other column levels (a ROI's `extra`) may not have: those downloads add, and the row levels'
RESERVED_COL_LEVELS = {"dataset_id", "value", TRIAL, *KNOWN_ROW_LEVELS}
MAX_HEADER_ROWS = 12  # of a wide CSV: its column levels, then the row naming the row levels
_YYYYMMDD = re.compile(r"(?:19|20)\d{6}")
# an odor written 'name(conc)' (as '1-5ol(-3)'), read as 'name @ conc'
_PAREN_ODOR = re.compile(r"(?P<name>[^@]*?\S)\s*\((?P<conc>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)")
SOLVENT_NAMES = ("pfo", "water")  # a solvent alone (as 'PFO'), read as at 0 ('pfo @ 0')
LONG_VALUE_COLUMNS = ("value", "response", "resp", "df_over_f", "dff", "mean_dff", "delta_f", "mean_df_over_f")
MAX_EXAMPLES = 15
ROI_SUFFIX = re.compile(r"_t\d+$")  # removed from ROI names ('DA1_t0' -> 'DA1')
# odor<n> values meaning that nothing was presented on that manifold (as is a missing one);
# the '@ 0' ones however the 0 is written ('pfo @ 0.0', ...)
EMPTY_ODOR_VALUES = (SOLVENT_STR, "pfo", "water", "pfo @ 0", "water @ 0")
_SOLVENT_AT = re.compile(r"(pfo|water)\s*@\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


class ValidationError(Exception):
    """Why a data file was not accepted (`errors`), with `warnings`, and `details` of some
    errors: ``rois``, when ROI names were not accepted (see `_validate_wide`)."""

    def __init__(self, errors: list[str], warnings: Optional[list[str]] = None, details: Optional[dict] = None):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.warnings = warnings or []
        self.details = details or {}


@dataclass
class ParsedData:
    units: list[dict]
    trials: list[dict]
    values: np.ndarray  # float32, shape (n_units, n_trials)
    row_levels: list[str]
    col_levels: list[str]
    source_format: str
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ summaries
    def summary(self) -> dict:
        flies: dict[tuple, dict] = {}
        n_glom = n_int = 0
        for u in self.units:
            k = (u["date"], u["fly_num"])  # (a fly's ROIs, whatever their other levels)
            if u["date"] is not None:  # (none, without flies)
                flies.setdefault(k, {"date": u["date"], "fly_num": u["fly_num"], "n_rois": 0})
                flies[k]["n_rois"] += 1
            if u["roi_kind"] == "glomerulus":
                n_glom += 1
            else:
                n_int += 1
        fly_list = [flies[k] for k in sorted(flies)]

        panels: dict[Optional[str], dict] = {}
        cond_counts: dict[tuple, int] = {}
        for t in self.trials:
            p = panels.setdefault(t["panel"], {"panel": t["panel"], "odors": [], "n_trials": 0})
            p["n_trials"] += 1
            if t["odor"] not in p["odors"]:
                p["odors"].append(t["odor"])
            cond_counts[(t["panel"], t["odor"])] = cond_counts.get((t["panel"], t["odor"]), 0) + 1

        finite = np.isfinite(self.values)
        vmin = float(np.nanmin(self.values)) if finite.any() else None
        vmax = float(np.nanmax(self.values)) if finite.any() else None
        all_odors = []
        for t in self.trials:
            if t["odor"] not in all_odors:
                all_odors.append(t["odor"])
        if n_glom and n_int:
            roi_kind = "mixed"
        elif n_glom:
            roi_kind = "glomeruli"
        else:
            roi_kind = "integers"
        return {
            "n_units": len(self.units),
            "n_trials": len(self.trials),
            "n_flies": len(fly_list),
            "flies": fly_list,
            "first_date": fly_list[0]["date"] if fly_list else None,
            "last_date": fly_list[-1]["date"] if fly_list else None,
            "panels": list(panels.values()),
            "panel_names": [p for p in panels if p is not None],
            "n_odors": len(all_odors),
            "odors": all_odors,
            "repeats_min": min(cond_counts.values()) if cond_counts else 0,
            "repeats_max": max(cond_counts.values()) if cond_counts else 0,
            "has_repeat_level": "repeat" in self.row_levels,
            "n_glomerulus_rois": n_glom,
            "n_int_rois": n_int,
            "n_bad": sum(1 for u in self.units if u.get("bad")),
            # the ROIs' other column levels (e.g. hemibrain), each with its values
            "extra_levels": {lvl: sorted({(u.get("extra") or {}).get(lvl) for u in self.units} - {None})
                             for lvl in dict.fromkeys(k for u in self.units for k in (u.get("extra") or {}))},
            "roi_kind": roi_kind,
            "nan_fraction": float(1 - finite.mean()) if self.values.size else 0.0,
            "vmin": vmin,
            "vmax": vmax,
            "row_levels": self.row_levels,
            "col_levels": self.col_levels,
            "source_format": self.source_format,
        }

    def to_payload(self) -> dict:
        vals = np.ascontiguousarray(self.values.astype("<f4"))
        return {
            "units": self.units,
            "trials": self.trials,
            "shape": list(vals.shape),
            "dtype": "float32",
            "values_b64": base64.b64encode(vals.tobytes()).decode("ascii"),
            "summary": self.summary(),
            "warnings": self.warnings,
        }

    def component_odors(self) -> list[tuple[str, float]]:
        """Distinct (name, log10_conc) pairs for every non-solvent component odor."""
        return list(dict.fromkeys(o for odors in self.panel_odors().values() for o in odors))

    def panel_odors(self) -> dict[Optional[str], list[tuple[str, float]]]:
        """`component_odors` of each panel (None: trials without one), in trial order."""
        seen: dict[Optional[str], dict[tuple[str, float], None]] = {}
        for t in self.trials:
            for c in t["components"]:
                if c == SOLVENT_STR:
                    continue
                o = parse_odor(c, strict=False)
                seen.setdefault(t["panel"], {}).setdefault((o.name, o.log10_conc), None)
        return {panel: list(odors) for panel, odors in seen.items()}


# ---------------------------------------------------------------------- loading


def load_file(path: Path, *, original_name: Optional[str] = None) -> ParsedData:
    """Load a CSV or parquet file and validate it. Raises ValidationError."""
    name = (original_name or path.name).lower()
    if name.endswith((".parquet", ".pq", ".parq")):
        return _load_parquet(path)
    if name.endswith((".csv", ".txt", ".tsv")):
        return _load_csv(path)
    # fall back on sniffing: parquet files start with the magic bytes 'PAR1'
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"PAR1":
        return _load_parquet(path)
    return _load_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    """A parquet file as pandas wrote it, but for column levels of True and False (a `BAD`
    level), which pandas writes as text and cannot read back ('False' would be True): they
    are read as the text they are."""
    import json

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    if b"pandas" in meta:
        pandas_meta = json.loads(meta[b"pandas"])
        for level in pandas_meta.get("column_indexes") or []:
            if level.get("pandas_type") == "bool":
                level.update(pandas_type="unicode", numpy_type="object")
        table = table.replace_schema_metadata({**meta, b"pandas": json.dumps(pandas_meta).encode()})
    return table.to_pandas()


def _load_parquet(path: Path) -> ParsedData:
    try:
        df = _read_parquet(path)
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ValidationError([f"parquet support is not installed on the server ({e}); install pyarrow"])
    except Exception as e:
        raise ValidationError([f"could not read parquet file: {e}"])
    if isinstance(df.columns, pd.MultiIndex) or any(n is not None and _is_row_level(str(n)) for n in df.index.names):
        return _validate_wide(df, source_format="wide-parquet")  # (plain columns: ROIs alone, no flies)
    return _long_to_wide(df, source_format="long-parquet")


def _read_head(path: Path, n: int, delim: Optional[str] = None) -> tuple[list[list[str]], str]:
    lines: list[str] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for line in f:
            lines.append(line)
            if len(lines) >= n:
                break
    if not lines or not any(l.strip() for l in lines):
        raise ValidationError(["file is empty"])
    if delim is None:
        delim = "\t" if lines[0].count("\t") > lines[0].count(",") else ","
    rows = list(csv.reader(lines, delimiter=delim))
    return rows, delim


def _sniff_csv(path: Path) -> tuple[str, int, int, bool, str]:
    """Return (layout, n_index_cols, n_header_rows, has_names_row, delimiter).

    layout is 'wide' (header rows of the column levels, date / fly_num / roi and any
    others, then one naming the row levels, as pandas writes a frame with MultiIndex
    columns), 'wide-rois' (one header row: the row levels' names, then the ROIs', as
    pandas writes a frame with plain columns) or 'long'.
    """
    rows, delim = _read_head(path, MAX_HEADER_ROWS + 1)
    while len(rows) < MAX_HEADER_ROWS + 1:
        rows.append([])
    n_index = 1 + sum(1 for _ in itertools.takewhile(lambda c: c.strip() == "", rows[0][1:]))

    def is_names_row(r: list[str]) -> bool:
        return len(r) >= n_index and all(c.strip() == "" for c in r[n_index:]) and any(c.strip() for c in r[:n_index])

    # header rows run until the row naming the index levels (empty after the index columns)
    n_header = next((k for k in range(1, MAX_HEADER_ROWS + 1) if is_names_row(rows[k])), None)
    if n_index > 1 and len(rows[0]) > n_index:
        levels = [r[0].strip() for r in rows[:n_header or MAX_HEADER_ROWS] if len(r) > n_index and any(c.strip() for c in r[n_index:])]
        if {"date", "fly_num"} <= {_col_level(n) for n in levels} and "roi" in {_col_level(n) for n in levels}:
            if n_header is None:
                return "wide", n_index, len(levels), False, delim
            return "wide", n_index, n_header, True, delim
    # a single header line, of the row levels' names then the ROIs' (no flies)
    header = [c.strip() for c in rows[0]]
    lead = list(itertools.takewhile(_is_row_level, header))
    if lead and len(header) > len(lead) and not ({*COL_LEVELS, CELLS, *LONG_VALUE_COLUMNS} & set(header)):
        return "wide-rois", len(lead), 1, False, delim
    # maybe a wide file with the wrong header levels: report them if it looks like one
    if n_index > 1 and len(rows[0]) > n_index:
        levels = []
        for r in rows[:MAX_HEADER_ROWS]:
            if len(r) <= n_index or all(c.strip() == "" for c in r[n_index:]):
                break
            levels.append(r[0].strip())
        if 1 <= len(levels) <= MAX_HEADER_ROWS - 1 and all(not _is_number(x) for x in levels):
            raise ValidationError([_col_levels_error(levels)])
    return "long", 0, 0, False, delim


def _col_levels_error(found: list[str]) -> str:
    """Why column levels `found` (as named in a file) are not accepted: the fly's levels it
    lacks, if those are all it lacks; else what they must be."""
    names = [_col_level(n) for n in found]
    missing = [n for n in COL_LEVELS if n not in names]
    repeated = sorted({n for n in names if names.count(n) > 1})
    reserved = sorted(set(names) & RESERVED_COL_LEVELS)
    if missing and set(missing) <= {"date", "fly_num"} and not repeated and not reserved:
        return f"no {' or '.join(missing)} column level: {FLIES_NEEDED} (found {found})"
    why = (f"each named once (not {repeated})" if repeated else f"none named {reserved}" if reserved and not missing
           else f"missing {missing}")
    return (f"column header levels must include {list(COL_LEVELS)} ('{CELLS}' for 'roi' too), and may have '{BAD}' and "
            f"others (e.g. hemibrain), {why}; found {found}")


def _col_level(name: str) -> str:
    """A column level's name, 'cells' as 'roi'."""
    return "roi" if name == CELLS else name


def _is_row_level(name: str) -> bool:
    return name in KNOWN_ROW_LEVELS or name == TRIAL or is_odor_level(name)


def _is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def _load_csv(path: Path) -> ParsedData:
    layout, n_index, n_header, has_names_row, delim = _sniff_csv(path)
    if layout == "wide-rois":
        names = [c.strip() for c in _read_head(path, 1, delim)[0][0]]
        try:
            df = pd.read_csv(path, sep=delim, encoding="utf-8-sig", header=None, skiprows=1,
                             index_col=list(range(n_index)), keep_default_na=True)
        except Exception as e:
            raise ValidationError([f"could not parse wide-format CSV: {e}"])
        if df.shape[1] != len(names) - n_index:
            raise ValidationError([f"the header line lists {len(names) - n_index} ROIs but data rows have {df.shape[1]}"])
        df.columns = pd.Index(names[n_index:], name="roi")  # (as written: pandas would rename a repeated one)
        df.index = df.index.set_names(names[:n_index]) if isinstance(df.index, pd.MultiIndex) else df.index.rename(names[0])
        return _validate_wide(df, source_format="wide-csv")
    if layout == "long":
        try:
            df = pd.read_csv(path, sep=delim, encoding="utf-8-sig")
        except Exception as e:
            raise ValidationError([f"could not parse CSV: {e}"])
        return _long_to_wide(df, source_format="long-csv")

    # Wide: build the column MultiIndex from the raw header rows ourselves, because
    # pandas silently renames duplicated column labels ('DM2' -> 'DM2.1'), which would
    # hide the uniqueness check.
    head, _ = _read_head(path, n_header + 1, delim)
    header_rows = head[:n_header]
    width = max(len(r) for r in header_rows)
    header_rows = [r + [""] * (width - len(r)) for r in header_rows]
    level_names = [r[0].strip() for r in header_rows]
    arrays = [[c.strip() for c in r[n_index:]] for r in header_rows]
    if not has_names_row:
        raise ValidationError(
            ["the 4th line should name the row-index levels (e.g. 'panel,is_pair,odor1,odor2,repeat'); "
             "write the file with pandas' DataFrame.to_csv from a frame with a named (Multi)Index"]
        )
    index_names = [c.strip() for c in head[n_header][:n_index]]
    try:
        df = pd.read_csv(
            path, sep=delim, encoding="utf-8-sig", header=None, skiprows=n_header + 1,
            index_col=list(range(n_index)), keep_default_na=True,
        )
    except Exception as e:
        raise ValidationError([f"could not parse wide-format CSV: {e}"])
    if df.shape[1] != len(arrays[0]):
        raise ValidationError(
            [f"header rows list {len(arrays[0])} data columns but data rows have {df.shape[1]}"]
        )
    df.columns = pd.MultiIndex.from_arrays(arrays, names=level_names)
    df.index = df.index.set_names(index_names) if isinstance(df.index, pd.MultiIndex) else df.index.rename(index_names[0])
    return _validate_wide(df, source_format="wide-csv")


def _long_to_wide(df: pd.DataFrame, *, source_format: str) -> ParsedData:
    cols = [str(c) for c in df.columns]
    if TRIAL in cols and "repeat" not in cols:
        cols = ["repeat" if c == TRIAL else c for c in cols]
    df.columns = cols
    errors = []
    roi = CELLS if CELLS in cols and "roi" not in cols else "roi"
    levels = ("date", "fly_num", roi)
    missing = [c for c in levels if c not in cols]
    if roi in cols and missing in (["date"], ["fly_num"]):
        raise ValidationError([f"no {missing[0]} column: {FLIES_NEEDED}"])
    if missing and missing != ["date", "fly_num"]:
        errors.append(
            f"this does not look like the expected format. For a wide (matrix) file the "
            f"first three header rows must start with 'date', 'fly_num', 'roi'; for a long file the columns must include "
            f"{list(COL_LEVELS)} ('{CELLS}' for 'roi' too; missing {missing}). "
            f"Found columns: {cols[:12]}{'...' if len(cols) > 12 else ''}"
        )
        raise ValidationError(errors)
    odor_cols = [c for c in cols if is_odor_level(c)]
    if not odor_cols:
        raise ValidationError(["long-format file has no 'odor1' (or 'odor') column"])
    value_col = next((c for c in LONG_VALUE_COLUMNS if c in cols), None)
    row_cols = [c for c in cols if c in KNOWN_ROW_LEVELS or is_odor_level(c)]
    index_cols = [c for c in cols if re.fullmatch(r"Unnamed: \d+", c)]  # (as to_csv writes an index)
    leftover = [c for c in cols if c not in (*levels, BAD) and c not in row_cols and c not in index_cols]
    if value_col is None:
        if len(leftover) == 1:
            value_col = leftover[0]
        else:
            raise ValidationError(
                [f"long-format file needs a value column named one of {list(LONG_VALUE_COLUMNS)}; "
                 f"ambiguous leftover columns: {leftover}"]
            )
    # any other columns: more levels of the ROIs (as in the wide layout)
    extra_cols = [c for c in leftover if c != value_col]
    col_levels = [c for c in ("date", "fly_num", *(extra_cols if "date" in cols or "fly_num" in cols else []), roi, BAD)
                  if c in cols]  # (roi alone: no flies)
    warnings = []
    if index_cols:
        warnings.append(f"ignoring index columns in long-format file: {index_cols}")
    # order row levels: panel, is_pair, odor*, repeat (like the wide example)
    ordered_rows = [c for c in ("panel", "is_pair") if c in row_cols]
    ordered_rows += sorted(odor_cols, key=lambda c: (len(c), c))
    if "repeat" in row_cols:
        ordered_rows.append("repeat")
    dup = df.duplicated(subset=ordered_rows + col_levels, keep=False)
    if dup.any():
        raise ValidationError(
            [f"{int(dup.sum())} rows in the long-format file share the same (row index, {', '.join(col_levels)}) key"]
        )
    # build the matrix by hand so that row/column order follows first appearance and no
    # cartesian product of index levels is created
    row_keys = list(df[ordered_rows].itertuples(index=False, name=None))
    col_keys = list(df[col_levels].itertuples(index=False, name=None))
    row_index: dict[tuple, int] = {}
    col_index: dict[tuple, int] = {}
    for k in row_keys:
        row_index.setdefault(k, len(row_index))
    for k in col_keys:
        col_index.setdefault(k, len(col_index))
    values = np.full((len(row_index), len(col_index)), np.nan, dtype=object)
    ri = np.fromiter((row_index[k] for k in row_keys), dtype=np.int64, count=len(row_keys))
    ci = np.fromiter((col_index[k] for k in col_keys), dtype=np.int64, count=len(col_keys))
    values[ri, ci] = df[value_col].to_numpy(dtype=object)
    wide = pd.DataFrame(
        values,
        index=pd.MultiIndex.from_tuples(list(row_index), names=ordered_rows),
        columns=(pd.MultiIndex.from_tuples(list(col_index), names=col_levels) if len(col_levels) > 1
                 else pd.Index([k[0] for k in col_index], name=roi)),
    )
    pd_ = _validate_wide(wide, source_format=source_format)
    pd_.warnings = warnings + pd_.warnings
    return pd_


# ------------------------------------------------------ uninformative row levels


def rewritten_odor(s: str) -> Optional[str]:
    """An odor string as this app writes odors, if it is written another way it reads:
    ``name(conc)`` as ``name @ conc`` (``'1-5ol(-3)'`` -> ``'1-5ol @ -3'``), and a solvent
    alone as at 0 (``'PFO'`` -> ``'pfo @ 0'``); None if it is neither (as it is)."""
    t = s.strip()
    if "@" in t or COMPONENT_DELIM in t:
        return None
    if t.lower() in SOLVENT_NAMES:
        return f"{t.lower()} @ 0"
    m = _PAREN_ODOR.fullmatch(t)
    if m is None:
        return None
    return f"{m['name']} @ {format_conc(float(m['conc']))}"


def date_str(v) -> Optional[str]:
    """A date as stored ('YYYY-MM-DD'), from one as a file may write it: as a date (or
    datetime), as YYYYMMDD (text or a number), or any way pandas reads one (e.g.
    '2024-07-10', '2024/07/10'); None if it is none of these (a number not YYYYMMDD
    included, which pandas would read as nanoseconds since 1970)."""
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return None if pd.isna(v) else pd.Timestamp(v).strftime("%Y-%m-%d")
    i = _to_int_like(v)
    s = str(i) if i is not None else ("" if _missing(v) else str(v).strip())
    if _YYYYMMDD.fullmatch(s):
        try:
            return dt.datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return None
    if not s or re.fullmatch(r"[-+]?\d+(\.\d*)?", s):
        return None
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        ts = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _time_of_day(v) -> bool:
    """Whether a date as written (`date_str`) has a time of day besides (but midnight)."""
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return pd.Timestamp(v).time() != dt.time()
    if not isinstance(v, str) or _YYYYMMDD.fullmatch(v.strip()):
        return False
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        ts = pd.to_datetime(v.strip(), errors="coerce")
    return not pd.isna(ts) and ts.time() != dt.time()


def fly_key(date: Optional[str], fly_num: Optional[int], extra: Optional[dict]) -> tuple:
    """What tells ROIs of a recording apart from those of another: their fly, and their
    other levels (`extra`, e.g. the side of the brain), whatever the levels' order."""
    return (date, fly_num, tuple(sorted((extra or {}).items())))


def unit_key(u: dict) -> tuple:
    """What tells a ROI apart from every other (of a dataset, or among datasets): `fly_key`, and its name."""
    return (*fly_key(u["date"], u["fly_num"], u.get("extra")), u["roi"])


def extra_text(extra: Optional[dict]) -> str:
    """A ROI's other levels as said of it, e.g. 'hemibrain L'; '' if none."""
    return ", ".join(f"{k} {v}" for k, v in (extra or {}).items())


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def _odor_value(v) -> Optional[str]:
    """An odor level's value for comparing: None if missing, solvents at a concentration of
    0 written one way ('pfo @ 0.0' -> 'pfo @ 0'), anything else as it is."""
    if _missing(v):
        return None
    s = str(v).strip()
    m = _SOLVENT_AT.fullmatch(s)
    return f"{m[1]} @ 0" if m and float(m[2]) == 0 else s


def _false(v) -> bool:
    return (isinstance(v, (bool, np.bool_)) and not v) or (isinstance(v, str) and v.strip().lower() == "false")


def uninformative_levels(columns: dict[str, list]) -> list[tuple[list[str], str]]:
    """Row levels that say nothing, to be dropped.

    These are ``is_pair`` if it is False on every row, and ``odor<n>`` (n >= 2) together
    with every odor level after it, if all their values are one and the same of: missing,
    or one of `EMPTY_ODOR_VALUES` (a 0 concentration written any way: 'pfo @ 0' and
    'pfo @ 0.0' count as the same). ``odor1`` is always kept.

    Args:
        columns: row level name -> its value on each row.

    Returns:
        [(level names, what they all hold)]
    """
    out = []
    if "is_pair" in columns and all(_false(v) for v in columns["is_pair"]):
        out.append((["is_pair"], "False"))
    odor_levels = sorted((n for n in columns if is_odor_level(n)), key=odor_level_sort_key)
    for i in range(1, len(odor_levels)):
        values = {_odor_value(v) for n in odor_levels[i:] for v in columns[n]}
        if len(values) == 1 and values <= {None, *EMPTY_ODOR_VALUES}:
            value = values.pop()
            out.append((odor_levels[i:], "empty" if value is None else f"'{value}'"))
            break
    return out


def drop_uninformative(trials: list[dict]) -> list[dict]:
    """Stored trials without the levels `uninformative_levels` finds in them (for datasets
    uploaded before those were dropped on upload)."""
    if not trials:
        return trials
    n = len(trials[0]["components"])
    columns = {f"odor{k + 1}": [t["components"][k] for t in trials] for k in range(n)}
    if all("is_pair" in t["extra"] for t in trials):
        columns["is_pair"] = [t["extra"]["is_pair"] for t in trials]
    dropped = {lvl for lvls, _ in uninformative_levels(columns) for lvl in lvls}
    if not dropped:
        return trials
    keep = sum(f"odor{k + 1}" not in dropped for k in range(n))
    return [
        {**t, "components": t["components"][:keep], "odor": format_mix_from_strs(t["components"][:keep]),
         "extra": {k: v for k, v in t["extra"].items() if k not in dropped}}
        for t in trials
    ]


# -------------------------------------------------------------------- validation


def _examples(values, n=MAX_EXAMPLES) -> str:
    vals = list(values)
    shown = ", ".join(repr(v) for v in vals[:n])
    if len(vals) > n:
        shown += f", ... ({len(vals)} total)"
    return shown


def _to_int_like(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v) if float(v).is_integer() else None
    s = str(v).strip()
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f.is_integer() else None


def _bool_like(v) -> Optional[bool]:
    """True or False as a file may write it (True, 'true', 1, '1'...), else None."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0"):
        return True
    if s in ("false", "0", "0.0"):
        return False
    return None


def _id_ranges(ids: list[int]) -> str:
    """Integers (sorted) as runs: [3, 7, 8, 9] -> '3, 7–9'; the first few, then how many more."""
    runs: list[list[int]] = []
    for i in ids:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    text = [str(a) if a == b else f"{a}–{b}" for a, b in runs]
    return ", ".join(text[:6]) + (f" and {len(text) - 6} more" if len(text) > 6 else "")


def kc_id_problems(units: list[dict]) -> Optional[str]:
    """Why KCs' ids are not as they must be, or None: each fly's (of ``units``: ``date``,
    ``fly_num``, ``roi``, the id as an integer string, and ``extra``) every one from 0 or 1
    (as the lowest is) to the highest, so that every KC segmented is there, those not to be
    used marked `BAD`. Of a fly whose ids repeat among the values of its other levels
    (e.g. both sides of its brain, each numbered from 0), each such part's."""
    by_fly: dict[tuple, dict[tuple, set[int]]] = {}
    for u in units:
        part = fly_key(u["date"], u["fly_num"], u.get("extra"))
        by_fly.setdefault(part[:2], {}).setdefault(part, set()).add(int(u["roi"]))
    checked: list[tuple[tuple, set[int]]] = []
    for fly, parts in by_fly.items():
        separate = sum(map(len, parts.values())) > len(set().union(*parts.values()))
        checked += list(parts.items()) if separate else [((*fly, ()), set().union(*parts.values()))]
    problems = []
    for (d, f, extra), ids in sorted(checked, key=lambda kv: (kv[0][0] or "", kv[0][1] or 0, kv[0][2])):
        missing = sorted(set(range(0 if min(ids) == 0 else 1, max(ids) + 1)) - ids)
        if missing:
            where = f"{d} fly {f}" if d is not None else "the KCs"
            problems.append(f"{where}{f' ({extra_text(dict(extra))})' if extra else ''} has no {_id_ranges(missing)}")
    if not problems:
        return None
    return (f"KC data must have every KC of each fly, their ids running from 0 (or 1) to the highest, none missing: "
            f"{'; '.join(problems[:MAX_KC_PROBLEMS])}{f'; and {len(problems) - MAX_KC_PROBLEMS} more flies' if len(problems) > MAX_KC_PROBLEMS else ''}. "
            f"Upload all of each fly's KCs, with one more column level '{BAD}' (True or False) marking those not to be used: "
            f"a header row after the ids in a wide file (date, fly_num, {CELLS}, {BAD}), or a column in a long one")


def _validate_wide(df: pd.DataFrame, *, source_format: str) -> ParsedData:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict = {}

    # ---- column levels -----------------------------------------------------
    # with flies (date / fly_num / roi), or ROIs alone (plain columns, or a level 'roi'); the
    # roi level called 'cells' (KC ids) as it may be
    file_levels = [str(n) if n is not None else "" for n in df.columns.names]
    cells = CELLS in file_levels
    has_bad = BAD in file_levels
    core = [_col_level(n) for n in file_levels if n != BAD]
    flies = isinstance(df.columns, pd.MultiIndex) and core != ["roi"]
    extra_levels = [n for n in core if n not in COL_LEVELS]  # (kept with each ROI: its `extra`)
    if cells:
        df = df.copy()
        df.columns = df.columns.set_names([_col_level(n) for n in file_levels])
    if (file_levels.count(BAD) > 1 or "" in core or len(set(core)) != len(core) or set(extra_levels) & RESERVED_COL_LEVELS
            or (flies and not set(COL_LEVELS) <= set(core))):
        errors.append(_col_levels_error(file_levels))
        raise ValidationError(errors)
    want = [*(["date", "fly_num", *extra_levels, "roi"] if flies else ["roi"]), *([BAD] if has_bad else [])]
    if isinstance(df.columns, pd.MultiIndex) and [_col_level(n) for n in file_levels] != want:
        df = df.reorder_levels(want, axis=1)
        warnings.append(f"column levels were reordered from {file_levels} to {[CELLS if cells and l == 'roi' else l for l in want]}")
    if not flies:
        errors.append(f"no flies: the columns are ROIs alone, without date or fly_num, but {FLIES_NEEDED}")

    # ---- row levels ---------------------------------------------------------
    row_names = list(df.index.names)
    if any(n is None or str(n).startswith("Unnamed") for n in row_names):
        errors.append(
            f"every row-index level needs a name (e.g. panel, odor1, odor2, repeat); got {row_names}"
        )
        raise ValidationError(errors)
    row_names = [str(n) for n in row_names]
    trial = TRIAL in row_names and "repeat" not in row_names  # (the repeat level by its other name)
    row_names = ["repeat" if trial and n == TRIAL else n for n in row_names]
    try:
        odor_levels = validate_odor_levels(row_names)
    except ValueError as e:
        raise ValidationError([str(e)])
    idx_frame = df.index.to_frame(index=False)
    idx_frame.columns = row_names
    rewritten: dict[str, str] = {}  # odors written another way -> as this app writes them (`rewritten_odor`)
    for lvl in odor_levels:
        def rewrite(v):
            r = rewritten_odor(v) if isinstance(v, str) else None
            if r is not None:
                rewritten.setdefault(v.strip(), r)
            return v if r is None else r
        idx_frame[lvl] = idx_frame[lvl].map(rewrite)
    if rewritten:
        warnings.append(f"{len(rewritten)} odor string(s) were read as written the usual way (name @ log10 conc): "
                        + _examples([f"{a} → {b}" for a, b in rewritten.items()]))
    dropped = uninformative_levels({n: idx_frame[n].tolist() for n in row_names})
    if dropped:
        names = [n for levels, _ in dropped for n in levels]
        idx_frame = idx_frame.drop(columns=names)
        row_names = [n for n in row_names if n not in names]
        odor_levels = [n for n in odor_levels if n not in names]
        warnings.append("row levels that say nothing were dropped: "
                        + "; ".join(f"{', '.join(levels)} (all {what})" for levels, what in dropped))
    unknown_levels = [n for n in row_names if n not in KNOWN_ROW_LEVELS and not is_odor_level(n)]
    if unknown_levels:
        warnings.append(
            f"row-index levels {unknown_levels} are not understood (known: panel, is_pair, "
            f"odor1..odorN, repeat); they are kept as trial metadata but ignored for display"
        )
    if "repeat" not in row_names:
        errors.append(f"no trials: {TRIALS_NEEDED}")

    # ---- column values: date / fly_num / roi ---------------------------------
    cols_frame = df.columns.to_frame(index=False)
    if not flies:
        cols_frame.columns = want
        cols_frame["date"] = cols_frame["fly_num"] = None
    bads: list[Optional[bool]] = [None] * len(cols_frame)
    if has_bad:
        bads = [_bool_like(v) for v in cols_frame[BAD].tolist()]
        not_bool = sorted({str(v) for v, b in zip(cols_frame[BAD].tolist(), bads) if b is None})
        if not_bool:
            errors.append(f"'{BAD}' values must be True or False: {_examples(not_bool)}")
    dates_raw = cols_frame["date"].tolist()
    distinct = {(type(d), str(d)): d for d in dates_raw}  # (each value read once: there are few)
    as_dates = {k: date_str(d) if flies else None for k, d in distinct.items()}
    date_strs = [as_dates[(type(d), str(d))] for d in dates_raw]
    bad_dates = sorted({str(d) for k, d in distinct.items() if as_dates[k] is None}) if flies else []
    if bad_dates:
        errors.append(f"unparseable 'date' values in column header (dates are written e.g. 2024-07-10 or 20240710): "
                      f"{_examples(bad_dates)}")
    elif flies and any(_time_of_day(d) for d in distinct.values()):
        warnings.append("'date' values carry a time-of-day; only the date is kept")

    fly_nums = []
    bad_fly = []
    for v in cols_frame["fly_num"].tolist():
        i = _to_int_like(v)
        if not flies:
            fly_nums.append(None)
        elif i is None or i <= 0:
            bad_fly.append(v)
            fly_nums.append(None)
        else:
            fly_nums.append(i)
    if bad_fly:
        errors.append(f"'fly_num' values must be positive integers: {_examples(sorted({str(x) for x in bad_fly}))}")

    rois = []
    roi_kinds = []
    bad_rois = []
    suffixed: dict[str, str] = {}  # ROI names with a suffix removed -> without it
    merged = []
    for v in cols_frame["roi"].tolist():
        if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
            s = str(int(v))
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            s = str(int(v))
        else:
            s = str(v).strip() if v is not None else ""
        if ROI_SUFFIX.search(s):
            suffixed[s] = s = ROI_SUFFIX.sub("", s)
        rois.append(s)
        if cells:  # KC ids
            roi_kinds.append("int" if glom.is_int_roi(s) else "bad")
            if not glom.is_int_roi(s):
                bad_rois.append(s)
        elif glom.is_glomerulus(s):
            roi_kinds.append("glomerulus")
        elif glom.merged(s):
            roi_kinds.append("glomerulus")
            merged.append(s)
        elif glom.is_int_roi(s):
            roi_kinds.append("int")
        else:
            roi_kinds.append("bad")
            bad_rois.append(s)
    if bad_rois and cells:
        errors.append(f"'{CELLS}' values are KC ids, so must be non-negative integers; not: {_examples(sorted(set(bad_rois), key=glom.roi_sort_key))}")
    elif bad_rois:
        errors.append(
            "'roi' values must be a known glomerulus name or a non-negative integer id; "
            f"unrecognised: {_examples(sorted(set(bad_rois)))}. "
            "(Edit config/glomeruli.txt on the server to extend the glomerulus list.)"
        )
        # every one, each with the accepted name it is but for case (if any), and the accepted names
        not_accepted = sorted(set(bad_rois), key=glom.roi_sort_key)
        details["rois"] = {"not_accepted": [{"roi": r, "same_but_case": glom.same_but_case(r)} for r in not_accepted],
                           "accepted": list(glom.GLOMERULUS_NAMES)}
    if "" in rois:
        errors.append("some column headers have an empty 'roi'")
    if suffixed:
        warnings.append(f"suffixes ('_t0', '_t1', ...) were removed from {len(suffixed)} ROI names: "
                        + _examples([f"{a} → {b}" for a, b in suffixed.items()]))
    if merged:
        warnings.append(f"{len(set(merged))} ROI name(s) are of glomeruli that could not be told apart, each kept as one ROI: "
                        + _examples(sorted(set(merged), key=glom.roi_sort_key)))

    # other levels of each ROI (as text: integers whole), e.g. {'hemibrain': 'L'}
    extras: list[dict] = [{} for _ in rois]
    for lvl in extra_levels:
        empty = 0
        for e, v in zip(extras, cols_frame[lvl].tolist()):
            whole = not isinstance(v, str) or re.fullmatch(r"[-+]?\d+\.0+", v.strip())  # (2.0, as a CSV writes 2: '2')
            i = _to_int_like(v) if whole else None
            text = str(i) if i is not None else ("" if _missing(v) else str(v).strip())
            empty += not text
            e[lvl] = text
        if empty:
            errors.append(f"{empty} column(s) have no '{lvl}' (each ROI's, as for the other column levels)")
    if extra_levels:
        warnings.append("column levels besides date, fly_num and roi are kept with each ROI (ROIs differing in them are "
                        "different ROIs): " + "; ".join(f"{lvl} ({_examples(sorted({e[lvl] for e in extras}), 8)})"
                                                         for lvl in extra_levels))

    units_found = [{"date": d, "fly_num": f, "roi": r, "extra": e} for d, f, r, e in zip(date_strs, fly_nums, rois, extras)]
    if cells and not bad_rois and not bad_dates and not bad_fly:
        problem = kc_id_problems(units_found)
        if problem:
            errors.append(problem)
    seen = {}
    dup_units = []
    for u in units_found:
        k = unit_key(u)
        if k in seen:
            dup_units.append(u)
        seen[k] = True
    if dup_units:
        after = " (after removing their suffixes)" if any(u["roi"] in suffixed.values() for u in dup_units) else ""
        levels = ", ".join(["date", "fly_num", *extra_levels, "roi"])
        errors.append(
            (f"columns must be unique ({levels}); duplicated: " if flies else "ROIs must be unique (there are no flies to tell them apart); duplicated: ")
            + _examples(sorted({f"{u['date']} fly{u['fly_num']}{' ' + extra_text(u['extra']) if u['extra'] else ''} {u['roi']}"
                                if flies else u["roi"] for u in dup_units})) + after
        )

    # ---- row values: odors / repeat / panel ----------------------------------
    components_per_row: list[list[str]] = [[] for _ in range(len(idx_frame))]
    mixed_in_level: list[str] = []  # odor strings of a mixture in one level, split into its components
    for lvl in odor_levels:
        bad: dict[str, str] = {}
        n_missing = 0
        parts_per_row = []
        for i, v in enumerate(idx_frame[lvl].tolist()):
            if _missing(v):
                if lvl == odor_levels[0]:
                    bad.setdefault("<empty>", f"'{lvl}' is empty on some rows")
                else:
                    n_missing += 1
                parts_per_row.append([SOLVENT_STR])
                continue
            s = str(v)
            parts = [c.strip() for c in s.split(COMPONENT_DELIM.strip())] if COMPONENT_DELIM in s else [s]
            if len(parts) > 1:
                mixed_in_level.append(s)
            for part in parts:
                try:
                    parse_odor(part)
                except OdorFormatError as e:
                    bad.setdefault(part, str(e))
            parts_per_row.append(parts)
        # as many components from this level in every row (so each its own odor<n> when downloaded)
        width = max(map(len, parts_per_row), default=1)
        for i, parts in enumerate(parts_per_row):
            components_per_row[i] += parts + [SOLVENT_STR] * (width - len(parts))
        if n_missing:
            warnings.append(f"{n_missing} rows have an empty '{lvl}'; treated as '{SOLVENT_STR}'")
        if bad:
            msgs = list(bad.values())
            errors.append(
                f"invalid odor strings in '{lvl}' ({len(bad)} distinct): "
                + "; ".join(msgs[:MAX_EXAMPLES])
                + (" ..." if len(msgs) > MAX_EXAMPLES else "")
            )

    if mixed_in_level:
        warnings.append(f"{len(mixed_in_level)} rows name a mixture in one odor level, split into its components "
                        f"(as if each were in a level of its own): {_examples(list(dict.fromkeys(mixed_in_level)))}")

    if "repeat" in row_names:
        repeats = []
        bad_rep = []
        for v in idx_frame["repeat"].tolist():
            i = _to_int_like(v)
            if i is None or i < 0:
                bad_rep.append(v)
                repeats.append(0)
            else:
                repeats.append(i)
        if bad_rep:
            errors.append(f"'{TRIAL if trial else 'repeat'}' values must be non-negative integers (each trial's number): "
                          f"{_examples(sorted({str(x) for x in bad_rep}))}")
    else:
        repeats = [0] * len(idx_frame)

    if "panel" in row_names:
        panels = []
        n_missing_panel = 0
        for v in idx_frame["panel"].tolist():
            if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
                panels.append(None)
                n_missing_panel += 1
            else:
                panels.append(str(v))
        if n_missing_panel:
            errors.append(f"{n_missing_panel} of its {len(panels)} rows have no panel, but {PANEL_NEEDED}")
    else:
        errors.append(f"no panel: {PANEL_NEEDED}")
        panels = [None] * len(idx_frame)

    # duplicate rows
    dup_rows = idx_frame.duplicated(keep=False)
    if dup_rows.any():
        ex = idx_frame[dup_rows].drop_duplicates().head(MAX_EXAMPLES)
        errors.append(
            f"{int(dup_rows.sum())} rows share identical index values (every trial must be unique), e.g. "
            + "; ".join(" / ".join(str(x) for x in r) for r in ex.itertuples(index=False))
        )

    # ---- values ---------------------------------------------------------------
    numeric = df.apply(pd.to_numeric, errors="coerce")
    non_numeric_mask = numeric.isna() & df.notna()
    # strings like '' or 'nan' are fine (treated as missing)
    if non_numeric_mask.to_numpy().any():
        raw_bad = pd.Series(df.to_numpy()[non_numeric_mask.to_numpy()]).astype(str)
        raw_bad = raw_bad[~raw_bad.str.strip().str.lower().isin({"", "nan", "none", "na", "null"})]
        if len(raw_bad):
            errors.append(f"non-numeric values in the matrix: {_examples(sorted(set(raw_bad)))}")
    values = numeric.to_numpy(dtype="float64", na_value=np.nan).T  # (units, trials)
    if values.size == 0:
        errors.append("matrix has no data cells")

    if errors:
        raise ValidationError(errors, warnings, details)

    # ---- assemble ---------------------------------------------------------------
    nan_frac = float(np.isnan(values).mean()) if values.size else 0.0
    if nan_frac > 0.5:
        warnings.append(f"{nan_frac:.0%} of cells are missing (NaN)")
    all_nan_units = int(np.all(np.isnan(values), axis=1).sum())
    if all_nan_units:
        warnings.append(f"{all_nan_units} ROI column(s) contain only NaN")
    all_nan_trials = int(np.all(np.isnan(values), axis=0).sum())
    if all_nan_trials:
        warnings.append(f"{all_nan_trials} trial row(s) contain only NaN")

    units = [{**u, "roi_kind": k, "bad": b} for u, k, b in zip(units_found, roi_kinds, bads)]
    trials = []
    for i in range(len(idx_frame)):
        extra = {
            n: _jsonable(idx_frame.iloc[i][n])
            for n in row_names
            if n not in KNOWN_ROW_LEVELS and not is_odor_level(n)
        }
        if "is_pair" in row_names:
            extra["is_pair"] = _jsonable(idx_frame.iloc[i]["is_pair"])
        trials.append(
            {
                "panel": panels[i],
                "odor": format_mix_from_strs(components_per_row[i]),
                "components": components_per_row[i],
                "repeat": repeats[i],
                "extra": extra,
            }
        )
    # consistency: same (panel, odor, repeat, extra) twice is a duplicate trial
    keys = {}
    dups = []
    for t in trials:
        k = (t["panel"], t["odor"], t["repeat"], tuple(sorted(t["extra"].items())))
        if k in keys:
            dups.append(f"{t['panel'] or ''} {t['odor']} repeat {t['repeat']}")
        keys[k] = True
    if dups:
        raise ValidationError(
            [f"duplicate trials after formatting odor mixtures (same panel, odor and repeat): {_examples(sorted(set(dups)))}"],
            warnings,
        )
    return ParsedData(
        units=units,
        trials=trials,
        values=values.astype(np.float32),
        row_levels=row_names,
        col_levels=[CELLS if cells and l == "roi" else l for l in want],
        source_format=source_format,
        warnings=warnings,
    )


def _jsonable(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


# ------------------------------------------------------------------ (de)serialise

def save_parsed(parsed: ParsedData, directory: Path) -> None:
    """Persist a ParsedData into ``directory`` (values.npy + meta.json)."""
    import json

    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "values.npy", parsed.values.astype(np.float32))
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "units": parsed.units,
                "trials": parsed.trials,
                "row_levels": parsed.row_levels,
                "col_levels": parsed.col_levels,
                "source_format": parsed.source_format,
                "warnings": parsed.warnings,
            }
        )
    )


def load_parsed(directory: Path) -> ParsedData:
    import json

    meta = json.loads((directory / "meta.json").read_text())
    values = np.load(directory / "values.npy")
    return ParsedData(values=values, **meta)
