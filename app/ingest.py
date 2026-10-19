"""Loading + validating uploaded response matrices (CSV or parquet).

Accepted layouts
----------------
*wide* (what ``DataFrame.to_csv`` / ``to_parquet`` write for a frame with MultiIndex
rows and columns):

    columns: MultiIndex with levels  date, fly_num, roi
    rows:    MultiIndex with levels  [panel, is_pair,] odor1[, odor2, ...][, repeat]
             (``odor`` alone is accepted in place of ``odor1``)

*long* (one row per value):

    columns: date, fly_num, roi, [panel,] odor1[, odor2, ...][, repeat], value

Row levels that say nothing are dropped before anything else (`uninformative_levels`):
``is_pair`` when it is False on every row, and the last ``odor<n>`` levels when they hold
nothing but one of missing / 'solvent' / 'pfo' / 'water' / 'pfo @ 0' / 'water @ 0'.

Validation collects *all* problems it can find and raises :class:`ValidationError`
with the full list, so a user can fix a file in one go.
"""
from __future__ import annotations

import base64
import csv
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
    SOLVENT_STR,
    OdorFormatError,
    format_mix_from_strs,
    is_odor_level,
    odor_level_sort_key,
    parse_odor,
    validate_odor_levels,
)

COL_LEVELS = ("date", "fly_num", "roi")
KNOWN_ROW_LEVELS = {"panel", "is_pair", "repeat"}
LONG_VALUE_COLUMNS = ("value", "response", "resp", "df_over_f", "dff", "mean_dff", "delta_f", "mean_df_over_f")
MAX_EXAMPLES = 15
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
            k = (u["date"], u["fly_num"])
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
        seen: dict[tuple[str, float], None] = {}
        for t in self.trials:
            for c in t["components"]:
                if c == SOLVENT_STR:
                    continue
                o = parse_odor(c, strict=False)
                seen.setdefault((o.name, o.log10_conc), None)
        return list(seen)


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


def _load_parquet(path: Path) -> ParsedData:
    try:
        df = pd.read_parquet(path)
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ValidationError([f"parquet support is not installed on the server ({e}); install pyarrow"])
    except Exception as e:
        raise ValidationError([f"could not read parquet file: {e}"])
    if isinstance(df.columns, pd.MultiIndex):
        return _validate_wide(df, source_format="wide-parquet")
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

    layout is 'wide' (3 header rows date/fly_num/roi, as pandas writes a frame with
    MultiIndex columns) or 'long'.
    """
    rows, delim = _read_head(path, 8)
    while len(rows) < 8:
        rows.append([])
    first_cells = [r[0].strip() if r else "" for r in rows[:3]]
    if set(first_cells) != set(COL_LEVELS):
        # maybe a wide file with the wrong header levels: report them if it looks like one
        n_index = 1 + sum(1 for _ in itertools.takewhile(lambda c: c.strip() == "", rows[0][1:]))
        if n_index > 1 and len(rows[0]) > n_index:
            levels = []
            for r in rows[:8]:
                if len(r) <= n_index or all(c.strip() == "" for c in r[n_index:]):
                    break
                levels.append(r[0].strip())
            if 1 <= len(levels) <= 6 and all(not _is_number(x) for x in levels):
                raise ValidationError(
                    [f"column header levels must be exactly {list(COL_LEVELS)}; found {levels}"]
                )
        return "long", 0, 0, False, delim
    n_index = 1 + sum(1 for _ in itertools.takewhile(lambda c: c.strip() == "", rows[0][1:]))

    def is_names_row(r: list[str]) -> bool:
        return len(r) >= n_index and all(c.strip() == "" for c in r[n_index:]) and any(c.strip() for c in r[:n_index])

    # header rows run until the row naming the index levels (empty after the index columns)
    n_header, has_names_row = 3, False
    for k in range(3, 8):
        if is_names_row(rows[k]):
            n_header, has_names_row = k, True
            break
    levels = [r[0].strip() for r in rows[:n_header]]
    if levels != list(COL_LEVELS):
        if set(levels) != set(COL_LEVELS):
            raise ValidationError([f"column header levels must be exactly {list(COL_LEVELS)}; found {levels}"])
    return "wide", n_index, n_header, has_names_row, delim


def _is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def _load_csv(path: Path) -> ParsedData:
    layout, n_index, n_header, has_names_row, delim = _sniff_csv(path)
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
    df.columns = cols
    errors = []
    missing = [c for c in COL_LEVELS if c not in cols]
    if missing:
        errors.append(
            f"this does not look like the expected format. For a wide (matrix) file the "
            f"first three header rows must start with 'date', 'fly_num', 'roi'; for a long "
            f"file the columns must include {list(COL_LEVELS)} (missing {missing}). "
            f"Found columns: {cols[:12]}{'...' if len(cols) > 12 else ''}"
        )
        raise ValidationError(errors)
    odor_cols = [c for c in cols if is_odor_level(c)]
    if not odor_cols:
        raise ValidationError(["long-format file has no 'odor1' (or 'odor') column"])
    value_col = next((c for c in LONG_VALUE_COLUMNS if c in cols), None)
    row_cols = [c for c in cols if c in KNOWN_ROW_LEVELS or is_odor_level(c)]
    if value_col is None:
        leftover = [c for c in cols if c not in COL_LEVELS and c not in row_cols]
        if len(leftover) == 1:
            value_col = leftover[0]
        else:
            raise ValidationError(
                [f"long-format file needs a value column named one of {list(LONG_VALUE_COLUMNS)}; "
                 f"ambiguous leftover columns: {leftover}"]
            )
    extra = [c for c in cols if c not in COL_LEVELS and c not in row_cols and c != value_col]
    warnings = []
    if extra:
        warnings.append(f"ignoring extra columns in long-format file: {extra}")
    # order row levels: panel, is_pair, odor*, repeat (like the wide example)
    ordered_rows = [c for c in ("panel", "is_pair") if c in row_cols]
    ordered_rows += sorted(odor_cols, key=lambda c: (len(c), c))
    if "repeat" in row_cols:
        ordered_rows.append("repeat")
    dup = df.duplicated(subset=ordered_rows + list(COL_LEVELS), keep=False)
    if dup.any():
        raise ValidationError(
            [f"{int(dup.sum())} rows in the long-format file share the same (row index, date, fly_num, roi) key"]
        )
    # build the matrix by hand so that row/column order follows first appearance and no
    # cartesian product of index levels is created
    row_keys = list(df[ordered_rows].itertuples(index=False, name=None))
    col_keys = list(df[list(COL_LEVELS)].itertuples(index=False, name=None))
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
        columns=pd.MultiIndex.from_tuples(list(col_index), names=list(COL_LEVELS)),
    )
    pd_ = _validate_wide(wide, source_format=source_format)
    pd_.warnings = warnings + pd_.warnings
    return pd_


# ------------------------------------------------------ uninformative row levels


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


def _validate_wide(df: pd.DataFrame, *, source_format: str) -> ParsedData:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict = {}

    # ---- column levels -----------------------------------------------------
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValidationError(
            ["columns are not a 3-level header (date / fly_num / roi): is this file in the expected format?"]
        )
    col_names = [str(n) if n is not None else "" for n in df.columns.names]
    if set(col_names) != set(COL_LEVELS) or len(col_names) != len(COL_LEVELS):
        errors.append(
            f"column header levels must be exactly {list(COL_LEVELS)}; found {col_names}"
        )
        raise ValidationError(errors)
    if col_names != list(COL_LEVELS):
        df = df.reorder_levels(list(COL_LEVELS), axis=1)
        warnings.append(f"column levels were reordered from {col_names} to {list(COL_LEVELS)}")

    # ---- row levels ---------------------------------------------------------
    row_names = list(df.index.names)
    if any(n is None or str(n).startswith("Unnamed") for n in row_names):
        errors.append(
            f"every row-index level needs a name (e.g. panel, odor1, odor2, repeat); got {row_names}"
        )
        raise ValidationError(errors)
    row_names = [str(n) for n in row_names]
    try:
        odor_levels = validate_odor_levels(row_names)
    except ValueError as e:
        raise ValidationError([str(e)])
    idx_frame = df.index.to_frame(index=False)
    idx_frame.columns = row_names
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
        warnings.append("no 'repeat' level in the row index: every row is treated as a single trial")

    # ---- column values: date / fly_num / roi ---------------------------------
    cols_frame = df.columns.to_frame(index=False)
    dates_raw = cols_frame["date"].tolist()
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        parsed_dates = pd.to_datetime(pd.Series(dates_raw, dtype=object), errors="coerce")
    bad_dates = sorted({str(d) for d, p in zip(dates_raw, parsed_dates) if pd.isna(p)})
    if bad_dates:
        errors.append(f"unparseable 'date' values in column header: {_examples(bad_dates)}")
    else:
        if any(ts.time() != pd.Timestamp(0).time() for ts in parsed_dates):
            warnings.append("'date' values carry a time-of-day; only the date is kept")
    date_strs = [None if pd.isna(p) else p.strftime("%Y-%m-%d") for p in parsed_dates]

    fly_nums = []
    bad_fly = []
    for v in cols_frame["fly_num"].tolist():
        i = _to_int_like(v)
        if i is None or i <= 0:
            bad_fly.append(v)
            fly_nums.append(None)
        else:
            fly_nums.append(i)
    if bad_fly:
        errors.append(f"'fly_num' values must be positive integers: {_examples(sorted({str(x) for x in bad_fly}))}")

    rois = []
    roi_kinds = []
    bad_rois = []
    for v in cols_frame["roi"].tolist():
        if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
            s = str(int(v))
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            s = str(int(v))
        else:
            s = str(v).strip() if v is not None else ""
        rois.append(s)
        if glom.is_glomerulus(s):
            roi_kinds.append("glomerulus")
        elif glom.is_int_roi(s):
            roi_kinds.append("int")
        else:
            roi_kinds.append("bad")
            bad_rois.append(s)
    if bad_rois:
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

    unit_keys = list(zip(date_strs, fly_nums, rois))
    seen = {}
    dup_units = []
    for k in unit_keys:
        if k in seen:
            dup_units.append(k)
        seen[k] = True
    if dup_units:
        errors.append(
            "columns must be unique (date, fly_num, roi); duplicated: "
            + _examples(sorted({f"{d} fly{f} {r}" for d, f, r in dup_units}))
        )

    # ---- row values: odors / repeat / panel ----------------------------------
    components_per_row: list[list[str]] = [[] for _ in range(len(idx_frame))]
    for lvl in odor_levels:
        bad: dict[str, str] = {}
        n_missing = 0
        for i, v in enumerate(idx_frame[lvl].tolist()):
            if _missing(v):
                if lvl == odor_levels[0]:
                    bad.setdefault("<empty>", f"'{lvl}' is empty on some rows")
                else:
                    n_missing += 1
                components_per_row[i].append(SOLVENT_STR)
                continue
            s = str(v)
            try:
                parse_odor(s)
            except OdorFormatError as e:
                bad.setdefault(s, str(e))
            components_per_row[i].append(s)
        if n_missing:
            warnings.append(f"{n_missing} rows have an empty '{lvl}'; treated as '{SOLVENT_STR}'")
        if bad:
            msgs = list(bad.values())
            errors.append(
                f"invalid odor strings in '{lvl}' ({len(bad)} distinct): "
                + "; ".join(msgs[:MAX_EXAMPLES])
                + (" ..." if len(msgs) > MAX_EXAMPLES else "")
            )

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
            errors.append(f"'repeat' values must be non-negative integers: {_examples(sorted({str(x) for x in bad_rep}))}")
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
        if n_missing_panel and n_missing_panel < len(panels):
            warnings.append(f"{n_missing_panel} rows have no 'panel' value")
    else:
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

    units = [
        {"date": d, "fly_num": f, "roi": r, "roi_kind": k}
        for (d, f, r), k in zip(unit_keys, roi_kinds)
    ]
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
        col_levels=list(COL_LEVELS),
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
