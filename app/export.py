"""Downloads of stored datasets.

Formats
-------
files  every file of each dataset as uploaded (data file, odor YAMLs, Experiment.xml),
       with its README.md and metadata.json
wide   the response matrices concatenated into ``responses_wide.csv``: trials (rows) x
       ROIs (columns), laid out like an upload but with ``dataset_id`` as an extra,
       outermost column level; trials are matched across datasets on panel, odor
       components and repeat
long   the same values in ``responses_long.csv``, one row per non-missing value

These are zips, which also have ``datasets.csv`` (one row of metadata per dataset),
``odors.csv`` (each dataset's odors with their metadata and links from the odor table),
each dataset's README.md and metadata.json, and a README.md describing the contents.
Except that a wide or long download of a single dataset is just its matrix, as one CSV
laid out as it was uploaded (less row levels dropped on upload), with no ``dataset_id``
and a name made from the dataset's details (`dataset_stem`). A merged download (every
dataset of a coding unit, see merge.py) is always a zip, with each dataset's values
multiplied by its scale factor, and ``scaling.csv``.

One dataset's odors can also be downloaded on their own (`dataset_odors_download`), as
that ``odors.csv`` or as a LaTeX table (`odors_tex`).

Every download but of the original files can have its odors' names (abbreviations) in
lowercase (``lower``, as chosen in Settings): all of them, but for those differing from
another in the download only in case (`odors.lowercase_map`), so that no two odors become one.
"""
from __future__ import annotations

import datetime as dt
import io
import math
import re
import unicodedata
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, TextIO

import numpy as np
import pandas as pd

from . import headspace, merge, odormeta
from .db import Database
from .ingest import BAD, CELLS
from .odors import (SOLVENT_STR, format_conc, is_odor_level, lowercase_map, lowercase_odor, odor_level_sort_key,
                    odor_names)

FORMATS = ("files", "wide", "long")
UNIT_LEVELS = ("dataset_id", "date", "fly_num", "roi")
SUMMARY_COLUMNS = (
    "n_flies", "n_units", "n_trials", "n_odors", "first_fly_date", "last_fly_date", "roi_kind",
    "data_file", "data_sha256",
)
# units per chunk when writing the long format, to bound memory
LONG_CHUNK_UNITS = 2000
DIAGNOSTICS_PANEL = re.compile("diag", re.IGNORECASE)


class ExportError(Exception):
    pass


# ------------------------------------------------------------------ trial keys


def trial_keys(matrices: list[dict]) -> tuple[list[str], list[list[tuple]]]:
    """Row-index levels shared by ``matrices`` and the key of every trial in each.

    Levels are ``panel, [extra levels], odor1..odorN, repeat``, with N the most odor
    components any dataset has (shorter trials are padded with 'solvent'). Extra
    levels (``is_pair``, or levels ingest did not understand) are kept when every
    dataset has them, or when some dataset needs them to tell its trials apart.

    Args:
        matrices: dicts with ``trials`` as returned by `Database.get_matrix_arrays`.

    Returns:
        (level names, per dataset a list with one key tuple per trial)
    """
    n_odor = max((len(t["components"]) for m in matrices for t in m["trials"]), default=1)
    seen: list[str] = []
    for m in matrices:
        for t in m["trials"]:
            seen += [k for k in t["extra"] if k not in seen]
    common = [k for k in seen if all(k in t["extra"] for m in matrices for t in m["trials"])]

    def keys_with(extras: list[str]) -> list[list[tuple]]:
        def key(t: dict) -> tuple:
            comps = list(t["components"]) + [SOLVENT_STR] * (n_odor - len(t["components"]))
            return (t["panel"], *(t["extra"].get(k) for k in extras), *comps, t["repeat"])

        return [[key(t) for t in m["trials"]] for m in matrices]

    extras = sorted(common, key=lambda k: k != "is_pair")
    keys = keys_with(extras)
    if any(len(set(ks)) < len(ks) for ks in keys):
        extras = sorted(seen, key=lambda k: k != "is_pair")
        keys = keys_with(extras)
    levels = ["panel", *extras, *(f"odor{i + 1}" for i in range(n_odor)), "repeat"]
    return levels, keys


def dataset_trial_keys(trials: list[dict], row_levels: list[str]) -> tuple[list[str], list[tuple]]:
    """Row-index levels of one dataset as uploaded, and the key of each of its trials.

    Args:
        trials: as returned by `Database.get_matrix_arrays`.
        row_levels: the row levels of the uploaded file (its summary's ``row_levels``);
            those no longer in `trials` (dropped since it was stored) are left out.

    Returns:
        (level names, one key tuple per trial)
    """
    n = len(trials[0]["components"]) if trials else 0
    odor_index = {lvl: i for i, lvl in enumerate(sorted(filter(is_odor_level, row_levels), key=odor_level_sort_key))
                  if i < n}
    extras = trials[0]["extra"] if trials else {}
    levels = [lvl for lvl in row_levels if lvl in ("panel", "repeat") or lvl in odor_index or lvl in extras]

    def value(t: dict, lvl: str):
        if lvl in ("panel", "repeat"):
            return t[lvl]
        return t["components"][odor_index[lvl]] if lvl in odor_index else t["extra"][lvl]

    return levels, [tuple(value(t, lvl) for lvl in levels) for t in trials]


def _ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9._+-]", "", re.sub(r"\s+", "-", s.strip()))


def dataset_stem(d: dict, trials: list[dict]) -> str:
    """Standard file name (without extension) for one dataset's matrix:
    ``<panels>_<uploader>_<coding unit>_<date of the last fly>``.

    Panels are those in the data (or, without a panel level, those entered at upload),
    sorted, leaving out diagnostic panels unless there are only those. The uploader's
    name is lowercased, without periods. Each part is ASCII only, with spaces as '-'.

    Args:
        d: as returned by `Database.get_dataset`.
        trials: as returned by `Database.get_matrix_arrays`.
    """
    panels = list(dict.fromkeys(t["panel"] for t in trials if t["panel"])) or d["metadata"].get("panel_names") or []
    if not all(DIAGNOSTICS_PANEL.search(p) for p in panels):
        panels = [p for p in panels if not DIAGNOSTICS_PANEL.search(p)]
    parts = [*sorted(panels, key=str.casefold), d["uploader_name"].lower().replace(".", ""), d["coding_unit"],
             d["last_fly_date"] or ""]
    return "_".join(filter(None, map(_ascii, parts))) or f"olfdb_dataset-{d['id']}"


# -------------------------------------------------------------------- matrices


def _has_bad(matrices: Iterable[dict]) -> bool:
    """Whether any of the datasets says which of its ROIs are not to be used (a `BAD` level)."""
    return any(u.get("bad") is not None for m in matrices for u in m["units"])


def extra_levels(matrices: Iterable[dict]) -> list[str]:
    """The ROIs' other column levels (e.g. hemibrain) of any of the datasets, in the order first found."""
    return list(dict.fromkeys(k for m in matrices for u in m["units"] for k in (u.get("extra") or {})))


def unit_levels(matrices: Iterable[dict]) -> list[str]:
    """The column levels of a download of several datasets: ``UNIT_LEVELS``, with the ROIs'
    other levels of any of them before roi (empty for the ROIs of those without), and `BAD`
    if any says (False for the ROIs of those that do not)."""
    matrices = list(matrices)
    return [*UNIT_LEVELS[:-1], *extra_levels(matrices), UNIT_LEVELS[-1], *([BAD] if _has_bad(matrices) else [])]


def wide_frame(matrices: list[tuple[int, dict]]) -> pd.DataFrame:
    """Trials x ROIs of several datasets, outer-joined on the trial key.

    Args:
        matrices: (dataset id, `Database.get_matrix_arrays` dict) pairs, in column order.

    Returns:
        float32 DataFrame; rows are a MultiIndex of `trial_keys` levels (in order of
        first appearance), columns a MultiIndex of `unit_levels`. Cells for trials a
        dataset does not have are NaN.
    """
    levels, keys = trial_keys([m for _, m in matrices])
    row_index: dict[tuple, int] = {}
    rows = [np.array([row_index.setdefault(k, len(row_index)) for k in ks], dtype=np.intp) for ks in keys]
    values = np.full((len(row_index), sum(len(m["units"]) for _, m in matrices)), np.nan, dtype=np.float32)
    cols = []
    names = unit_levels(m for _, m in matrices)
    extras = extra_levels(m for _, m in matrices)
    for (ds_id, m), r in zip(matrices, rows):
        values[r, len(cols):len(cols) + len(m["units"])] = m["values"].T
        cols += [(ds_id, u["date"], u["fly_num"], *((u.get("extra") or {}).get(k, "") for k in extras), u["roi"],
                  *([bool(u.get("bad"))] if BAD in names else [])) for u in m["units"]]
    return pd.DataFrame(
        values,
        index=pd.MultiIndex.from_tuples(list(row_index), names=levels),
        columns=pd.MultiIndex.from_tuples(cols, names=names),
    )


def dataset_frame(m: dict, levels: list[str], keys: list[tuple], cells: bool = False) -> pd.DataFrame:
    """One dataset's trials x ROIs as uploaded: rows indexed by `levels`, columns by date /
    fly_num / its ROIs' other levels / roi (by roi alone, for a dataset without flies, so a
    CSV of it has one header line), and `BAD` if it says which ROIs are not to be used.

    Args:
        m: as returned by `Database.get_matrix_arrays`.
        levels, keys: from `dataset_trial_keys`.
        cells: whether its ROI level was ``cells`` (KC ids), as it is named then.
    """
    flies = any(u["date"] is not None for u in m["units"])
    bad = _has_bad([m])
    extras = extra_levels([m])
    names = [*(["date", "fly_num", *extras, "roi"] if flies else ["roi"]), *([BAD] if bad else [])]
    names = [CELLS if cells and n == "roi" else n for n in names]
    cols = [(*((u["date"], u["fly_num"], *((u.get("extra") or {}).get(k, "") for k in extras)) if flies else ()), u["roi"],
             *([bool(u["bad"])] if bad else [])) for u in m["units"]]
    return pd.DataFrame(
        m["values"].T,
        index=pd.MultiIndex.from_tuples(keys, names=levels),
        columns=pd.MultiIndex.from_tuples(cols, names=names) if len(names) > 1 else pd.Index([c[0] for c in cols], name=names[0]),
    )


def write_long(f: TextIO, matrices: list[tuple[int, dict]], levels: list[str], keys: list[list[tuple]], *,
               dataset_id: bool = True) -> None:
    """Write one CSV row per non-missing value: [dataset_id,] date, fly_num, [the ROIs' other
    levels,] roi, [bad,] trial levels, value (date and fly_num left out for a dataset alone
    without flies; empty for one among others, as other levels are for a dataset without
    them; bad if any dataset says which ROIs are not to be used, False for others').

    Args:
        matrices: (dataset id, `Database.get_matrix_arrays` dict) pairs.
        levels, keys: trial levels, and the keys of each dataset's trials (from `trial_keys`, or
            `dataset_trial_keys` for a single dataset).
        dataset_id: whether to write the dataset_id column.
    """
    header = True
    flies = dataset_id or any(u["date"] is not None for _, m in matrices for u in m["units"])
    bad = _has_bad(m for _, m in matrices)
    extras = extra_levels(m for _, m in matrices) if flies else []
    for (ds_id, m), ks in zip(matrices, keys):
        trial_cols = [np.array(col, dtype=object) for col in zip(*ks)]
        for start in range(0, len(m["units"]), LONG_CHUNK_UNITS):
            units = m["units"][start:start + LONG_CHUNK_UNITS]
            block = m["values"][start:start + len(units)]
            ui, ti = np.nonzero(np.isfinite(block))
            df = pd.DataFrame({
                **({"dataset_id": np.full(len(ui), ds_id)} if dataset_id else {}),
                **({"date": np.array([u["date"] for u in units], dtype=object)[ui],
                    "fly_num": np.array([u["fly_num"] for u in units], dtype=object)[ui]} if flies else {}),
                **{k: np.array([(u.get("extra") or {}).get(k, "") for u in units], dtype=object)[ui] for k in extras},
                "roi": np.array([u["roi"] for u in units], dtype=object)[ui],
                **({BAD: np.array([bool(u.get("bad")) for u in units])[ui]} if bad else {}),
                **{lvl: col[ti] for lvl, col in zip(levels, trial_cols)},
                "value": block[ui, ti],
            })
            if len(df) or header:
                df.to_csv(f, index=False, header=header)
                header = False


def datasets_frame(datasets: list[dict]) -> pd.DataFrame:
    """One row per dataset: id, title, upload metadata and summary counts."""
    rows = []
    for d in datasets:
        md = d["metadata"]
        row = {"dataset_id": d["id"], "title": d["title"], "created_at": d["created_at"]}
        row.update({k: ", ".join(v) if isinstance(v, list) else v for k, v in md.items() if k != "notes"})
        row.update({k: d.get(k) for k in SUMMARY_COLUMNS})
        if len(d.get("data_files") or []) > 1:  # (several combined)
            row.update(data_file="; ".join(f["name"] for f in d["data_files"]),
                       data_sha256="; ".join(f["sha256"] for f in d["data_files"]))
        row["notes"] = md.get("notes")
        rows.append(row)
    return pd.DataFrame(rows)


def odors_frame(db: Database, datasets: list[dict], datasets_dir: Path, names: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """Each dataset's odors (per panel, as in the data) with their odor-table metadata and
    links; the data's names as `names` has them (`_lowercase`), if given."""
    index = odormeta.OdorIndex(db.list_odors())
    rows = []
    for d in datasets:
        odors = odormeta.dataset_odors(d, db.get_trials(d["id"]), index, odormeta.dataset_yaml_odors(d, datasets_dir))
        rows += [{"dataset_id": d["id"], **r} for r in odormeta.table_rows(odors, index)]
    if names is not None:
        _rename_rows(rows, names)
    df = pd.DataFrame(rows)
    if "cid" in df:
        df["cid"] = df["cid"].astype("Int64")
    return df


def dataset_odors_download(db: Database, ds_id: int, datasets_dir: Path, fmt: str, *,
                           lower: bool = False) -> tuple[str, str, str]:
    """One dataset's odors (per panel, as in the data, as the Browse tab's table of them) to download.

    Args:
        fmt: 'csv' (the columns of `odormeta.table_rows`, as every export's ``odors.csv``) or
            'tex' (a LaTeX table: see `odors_tex`).
        lower: the data's odor names in lowercase (see the module's docstring).

    Returns:
        (its text, its file name: the dataset's (`dataset_stem`) with ``_odors``, its media type).

    Raises:
        ExportError: no such dataset, or another format.
    """
    if fmt not in ("csv", "tex"):
        raise ExportError(f"unknown format {fmt!r}")
    d = db.get_dataset(ds_id)
    if d is None:
        raise ExportError(f"dataset not found: {ds_id}")
    index = odormeta.OdorIndex(db.list_odors())
    trials = db.get_trials(ds_id)
    rows = odormeta.table_rows(odormeta.dataset_odors(d, trials, index, odormeta.dataset_yaml_odors(d, datasets_dir)), index)
    if lower:
        _rename_rows(rows, lowercase_map(n for t in trials for c in t["components"] for n in odor_names(c)))
    stem = f"{dataset_stem(d, trials)}_odors"
    if fmt == "csv":
        df = pd.DataFrame(rows, columns=list(rows[0]) if rows else None)
        if "cid" in df:
            df["cid"] = df["cid"].astype("Int64")
        return df.to_csv(index=False), f"{stem}.csv", "text/csv"
    return odors_tex(rows, f"Odors of {d['title']}", f"tab:odors-dataset-{ds_id}"), f"{stem}.tex", "text/x-tex"


def _lowercase(trial_lists: Iterable[list[dict]]) -> dict[str, str]:
    """The odor names of trials (all of one download) lowercased (`lowercase_map`, over them
    all), the trials' odor strings changed to match, in place; and that map of the names."""
    trial_lists = list(trial_lists)
    names = lowercase_map(n for ts in trial_lists for t in ts for c in t["components"] for n in odor_names(c))
    for ts in trial_lists:
        for t in ts:
            t["components"] = [lowercase_odor(c, names) for c in t["components"]]
            if "odor" in t:
                t["odor"] = lowercase_odor(t["odor"], names)
    return names


def _rename_rows(rows: list[dict], names: dict[str, str]) -> None:
    """`odormeta.table_rows` rows' odor strings and names (abbreviations) as `names` has them."""
    for r in rows:
        r["odor"] = lowercase_odor(r["odor"], names)
        r["abbrev"] = names.get(r["abbrev"], r["abbrev"])


# LaTeX's special characters, as text
_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
                "|": r"\textbar{}"}
# Greek letters in chemical names, as LaTeX's math letters (T1 text fonts have none)
_GREEK = dict(zip("αβγδεζηθικλμνξπρστυφχψω", ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
                                               "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon",
                                               "phi", "chi", "psi", "omega"]))
# odors_tex's columns besides name, abbreviation and concentration: (heading, alignment,
# the `odormeta.table_rows` value and its URL)
_TEX_COLUMNS = [
    ("CID", "r", "cid", "pubchem_url"), ("Solvent", "l", "solvent", None), ("Target glomerulus", "l", "glomerulus", None),
    ("Supplier", "l", "supplier", None), ("Catalog no.", "l", "catalog_no", "catalog_url"),
    ("MetaCyc ID", "l", "metacyc", "metacyc_url"), ("LOTUS ID", "l", "lotus", "lotus_url"), ("CAS", "l", "cas", "cas_url"),
    ("Vapor pressure (kPa)", "r", "vapor_pressure_kpa", "vapor_pressure_url"),
]


def _tex(value) -> str:
    """Text for LaTeX: its special characters escaped, Greek letters as math."""
    out = "".join(_TEX_ESCAPES.get(c, c) for c in str(value))
    return re.sub("[" + "".join(_GREEK) + "]", lambda m: f"$\\{_GREEK[m[0]]}$", out)


def _tex_pressure(kpa: float) -> str:
    """A vapor pressure (kPa) as the site shows it: to 3 significant figures, but whole from
    100, and as a power of ten under 0.001."""
    if kpa >= 100 or kpa == 0:
        return f"{kpa:.0f}"
    if kpa >= 0.001:
        return f"{kpa:#.3g}".rstrip(".")
    m, e = f"{kpa:.2e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


# how odors_tex writes values other than as text
_TEX_FORMATS = {"vapor_pressure_kpa": _tex_pressure}


def _tex_link(url: Optional[str], text: str) -> str:
    """`text` linked to `url` (hyperref's \\href), if there is one."""
    if not url:
        return text
    escaped = re.sub(r"([%#&])", r"\\\1", url)  # a comment, a parameter, a column otherwise
    return f"\\href{{{escaped}}}{{{text}}}"


def _tex_number(v: float, digits: int = 3) -> str:
    """A number as the site shows it: to `digits` significant figures (trailing zeros
    dropped), as a power of ten under 0.001 or from 100,000."""
    if v == 0:
        return "0"
    if 1e-3 <= abs(v) < 1e5:
        return np.format_float_positional(float(f"{v:.{digits}g}"), trim="-")
    m, e = f"{v:.{digits - 1}e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def _tex_longtable(rows: list[dict], columns: list[tuple], caption: str, label: str, *, group=None,
                   last: Optional[list[str]] = None, notes: Optional[str] = None) -> str:
    """Rows as a LaTeX table to ``\\input``, like the thesis's: a booktabs longtable (over as
    many pages as it needs), footnote-sized, linked as on the site (hyperref's ``\\href``).

    Args:
        columns: (heading, alignment, cell: row -> its TeX ('' for none), always: whether
            kept when no row has anything in it (else left out)).
        group: row -> its group (e.g. its panel; rows of each together), each titled, with a
            rule between them; None for no groups.
        last: rows (each its cells' TeX, one per column kept) after a rule, e.g. a total.
        notes: TeX under the table.
    """
    cells = [[c[2](r) for c in columns] for r in rows]
    kept = [i for i, c in enumerate(columns) if c[3] or any(row[i] for row in cells)]
    n = len(kept)
    head = " & ".join(columns[i][0] for i in kept) + r" \\"
    out = [
        f"% {caption}: from the odor response database, {dt.date.today():%Y-%m-%d}.",
        r"% Needs \usepackage{booktabs}, \usepackage{longtable}, \usepackage{array} and \usepackage{hyperref}; in a landscape",
        r"% environment (\usepackage{pdflscape}) if too wide, and narrower margins if still (e.g.",
        r"% \usepackage[margin=2cm]{geometry}). Columns with nothing in them are left out.",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\begin{{longtable}}{{{''.join(columns[i][1] for i in kept)}}}",
        f"\\caption{{{_tex(caption)}}}\\label{{{label}}} \\\\",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        f"\\caption[]{{{_tex(caption)} (continued)}} \\\\",
        r"\toprule", head, r"\midrule", r"\endhead",
        r"\midrule", f"\\multicolumn{{{n}}}{{r}}{{\\emph{{continued on the next page}}}} \\\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
    ]
    current = object()  # (no group yet)
    for r, row in zip(rows, cells):
        if group is not None and group(r) != current:
            if out[-1] != r"\endlastfoot":  # (a group before it)
                out.append(r"\midrule")
            current = group(r)
            if current is not None:
                out.append(f"\\multicolumn{{{n}}}{{l}}{{\\textit{{{_tex(current)}}}}} \\\\")
        out.append(" & ".join(row[i] for i in kept) + r" \\")
    if last:
        out.append(r"\midrule")
        out += [" & ".join(row[i] for i in kept) + r" \\" for row in last]
    out += [r"\end{longtable}", *([notes] if notes else []), r"\endgroup", ""]
    return "\n".join(out)


# a column of text wrapped to a width (left-aligned: \\raggedright), for long names
_TEX_WRAP = ">{{\\raggedright\\arraybackslash}}p{{{}}}"


def _tex_head(top: str, bottom: str, align: str = "r") -> str:
    """A column's heading on two lines (e.g. its name, then its units), aligned as the column."""
    return f"\\begin{{tabular}}[b]{{@{{}}{align}@{{}}}}{top}\\\\{bottom}\\end{{tabular}}"


def _tex_cell(value, url: Optional[str] = None, fmt=None) -> str:
    """A value as a table's TeX ('' if none), linked to `url` if there is one."""
    if value in (None, ""):
        return ""
    return _tex_link(url, (fmt or _tex)(value))


def odors_tex(rows: list[dict], caption: str, label: str) -> str:
    """Odors as a LaTeX table to ``\\input``, like the thesis's (`_tex_longtable`): a row
    per odor, grouped by panel: its name (from the odor table, else from its odor YAML),
    abbreviation and concentration as in the data, then CID, solvent, target glomerulus,
    supplier, catalog number, MetaCyc and LOTUS IDs, CAS number and vapor pressure (linked
    as on the site), each only if an odor has one. InChI and structures are left out (too
    long to typeset; the CSV has InChI).

    Args:
        rows: as from `odormeta.table_rows`, in order (each panel's together).
        caption: of the table; label: its ``\\label``.
    """
    columns = [
        ("Name", "l", lambda r: _tex(r["name"]) if r["name"] else r"\textit{not in the odor table}", True),
        ("Abbrev.", "l", lambda r: _tex(r["abbrev"] or ""), True),
        (r"$\log_{10}$ conc.", "l", lambda r: f"${r['log10_conc']}$" if r["log10_conc"] not in (None, "") else "", True),
        *((heading, align, lambda r, key=key, url=url: _tex_cell(r.get(key), r.get(url) if url else None, _TEX_FORMATS.get(key)),
           False) for heading, align, key, url in _TEX_COLUMNS),
    ]
    return _tex_longtable(rows, columns, caption, label, group=lambda r: r["panel"])


_SORT_PREFIX = re.compile(r"^(?:\([^)]*\)-|[\d,]+-|[a-z]-(?=[a-z]))+")


def _odor_sort_key(name: str) -> tuple[str, str]:
    """An odor's place in the Odors tab: by its name without prefixes ('(E)-', '2-', 'a-'), as the tab sorts them."""
    return _SORT_PREFIX.sub("", name.lower()), name


def _lower_abbrevs(odors: list[dict]) -> dict[str, str]:
    """The odor table's abbreviations as shown in lowercase (`lowercase_map`, over them all)."""
    return lowercase_map(o["abbrev"] for o in odors if o.get("abbrev"))


def _download(rows: list[dict], fmt: str, stem: str, tex) -> tuple[str, str, str]:
    """`rows` as a CSV (their keys its columns, CIDs as integers) or LaTeX (`tex()`): (text, file name, media type)."""
    if fmt == "csv":
        df = pd.DataFrame(rows, columns=list(rows[0]) if rows else None)
        if "cid" in df:
            df["cid"] = df["cid"].astype("Int64")
        return df.to_csv(index=False), f"{stem}.csv", "text/csv"
    return tex(), f"{stem}.tex", "text/x-tex"


def odor_table_download(db: Database, datasets_dir: Path, fmt: str, *, query: str = "",
                        lower: bool = False) -> tuple[str, str, str]:
    """The odor table (as the Odors tab shows it, and in its order) to download.

    Args:
        fmt: 'csv' (every field of each odor, its links, the datasets using it) or 'tex' (a
            LaTeX table of its names, IDs, supplier, vapor pressure and molar mass, linked
            as on the site: `_tex_longtable`).
        query: only the odors the tab's filter shows for it (in a name, abbreviation, alias,
            CID, CAS, catalog number, supplier or inventory), if any.
        lower: abbreviations in lowercase (see the module's docstring).

    Returns:
        (its text, its file name, its media type).

    Raises:
        ExportError: another format.
    """
    if fmt not in ("csv", "tex"):
        raise ExportError(f"unknown format {fmt!r}")
    odors = db.list_odors()
    used = odormeta.usage(db, datasets_dir, odors)
    names = _lower_abbrevs(odors) if lower else {}
    q = query.strip().lower()
    fields = ("name", "abbrev", "cid", "cas", "catalog_no", "supplier", "inventory")
    shown = [o for o in odors if not q or any(q in str(v).lower() for v in [*(o.get(f) for f in fields), *o["aliases"]] if v is not None)]
    shown.sort(key=lambda o: _odor_sort_key(o["name"]))
    rows = []
    for o in shown:
        lk = odormeta.links(o)
        rows.append({
            "name": o["name"], "abbrev": names.get(o["abbrev"], o["abbrev"]), "aliases": "; ".join(o["aliases"]) or None,
            "cid": o["cid"], "cas": o["cas"], "inchi": o.get("inchi"), "solvent": o["solvent"], "supplier": o["supplier"],
            "catalog_no": o["catalog_no"], "metacyc": o["metacyc"], "lotus": o["lotus"], "inventory": o["inventory"],
            "vapor_pressure_kpa": o.get("vapor_pressure_kpa"), "vapor_pressure_measured": o.get("vapor_pressure_measured"),
            "molar_mass_g_mol": o.get("molar_mass_g_mol"),
            "datasets": " ".join(map(str, sorted(used.get(o["id"], ())))) or None, "notes": o["notes"],
            "pubchem_url": lk["pubchem"], "cas_url": lk["cas"], "catalog_url": lk["catalog"], "metacyc_url": lk["metacyc"],
            "lotus_url": lk["lotus"], "vapor_pressure_url": lk["vapor_pressure"], "molar_mass_url": lk["molar_mass"],
        })
    caption = f"Odors{f' matching “{query.strip()}”' if q else ''}"
    columns = [
        ("Name", _TEX_WRAP.format("3cm"), lambda r: _tex(r["name"]), True), ("Abbrev.", "l", lambda r: _tex_cell(r["abbrev"]), True),
        ("CID", "r", lambda r: _tex_cell(r["cid"], r["pubchem_url"]), False),
        ("CAS", "l", lambda r: _tex_cell(r["cas"], r["cas_url"]), False),
        ("Solvent", "l", lambda r: _tex_cell(r["solvent"]), False),
        ("Supplier, catalog no.", _TEX_WRAP.format("3cm"), lambda r: " ".join(x for x in (_tex_cell(r["supplier"]),
                                                                       _tex_cell(r["catalog_no"], r["catalog_url"])) if x), False),
        ("MetaCyc ID", "l", lambda r: _tex_cell(r["metacyc"], r["metacyc_url"]), False),
        ("LOTUS ID", "l", lambda r: _tex_cell(r["lotus"], r["lotus_url"]), False),
        (_tex_head("Vapor pressure", "(kPa)"), "r",
         lambda r: _tex_cell(r["vapor_pressure_kpa"], r["vapor_pressure_url"], _tex_pressure), False),
        (_tex_head("Molar mass", "(g/mol)"), "r", lambda r: _tex_cell(r["molar_mass_g_mol"], r["molar_mass_url"], lambda v: f"{v:.2f}"), False),
    ]
    return _download(rows, fmt, "olfdb_odors", lambda: _tex_longtable(rows, columns, caption, "tab:odors"))


def _tex_measured(measured: Optional[bool]) -> str:
    """A mark of whether a value is from measurements (m) or an estimate (e), if known."""
    return "" if measured is None else "$^{m}$" if measured else "$^{e}$"


def headspace_download(result: dict, odors: dict[int, dict], fmt: str, units: str, *,
                       lower: bool = False) -> tuple[str, str, str]:
    """A Headspace calculator's result (`main._headspace`) to download.

    Args:
        odors: the odor table, by id.
        fmt: 'csv' (a row per odor, then one of them all together: what was used of each,
            whether measured, where from, its headspace in every one of `headspace.UNITS`)
            or 'tex' (a LaTeX table as the tab shows it, in `units`, linked as on the site,
            values marked as measured or estimates: `_tex_longtable`).
        units: of `headspace.UNITS`, for 'tex'.
        lower: abbreviations in lowercase (see the module's docstring).

    Returns:
        (its text, its file name, its media type).

    Raises:
        ExportError: another format, or unit.
    """
    if fmt not in ("csv", "tex"):
        raise ExportError(f"unknown format {fmt!r}")
    if units not in headspace.UNITS:
        raise ExportError(f"unknown units {units!r}")
    names = _lower_abbrevs(list(odors.values())) if lower else {}
    solvent, model = result["solvent"], result["model"]
    water, henry = solvent == "water", solvent == "water" or model == "henry"
    k_prop = "k_henry" if water else "k_henry_pfo"
    rows = []
    for c in result["components"]:
        o = odors.get(c["odor_id"], {})
        pr, links, sources = c.get("properties") or {}, c.get("links") or {}, c.get("sources") or {}
        if c.get("vapor_pressure_source"):  # (the one a constant in oil was estimated with)
            vp_from = c["vapor_pressure_source"]
            vp_url = (m := re.search(r"https?://[^\s()]+", vp_from)) and m[0]
        else:
            vp_prop = "vapor_pressure" if pr.get("vapor_pressure") is not None else "est_vapor_pressure"
            vp_from, vp_url = sources.get(vp_prop), links.get(vp_prop)
        notes = [*c["notes"], *(f"{p.replace('_', ' ')} could not be looked up: {e}" for p, e in (c.get("failed") or {}).items())]
        rows.append({
            "name": o.get("name"), "abbrev": names.get(o.get("abbrev"), o.get("abbrev")), "cid": o.get("cid"),
            "log10_conc": c["log10_conc"], "solvent": solvent, "model": model, "law": c["law"],
            "molar_mass_g_mol": pr.get("molar_mass"), "density_g_ml": pr.get("density"),
            "vapor_pressure_kpa": c["vapor_pressure"], "vapor_pressure_measured": c["vapor_pressure_measured"],
            "vapor_pressure_source": vp_from,
            "k_henry_mol_m3_pa": c["k_henry"], "k_henry_measured": c["k_henry_measured"],
            "k_henry_source": sources.get(k_prop) if c["k_henry"] is not None else None,
            "mole_fraction": c["mole_fraction"],
            **{f"headspace_{k}": (None if c["pa"] is None else u[3](c["pa"], pr.get("molar_mass"))) for k, u in headspace.UNITS.items()},
            "notes": "; ".join(notes) or None,
            # (for the LaTeX: where each is from)
            "_links": {**links, "pubchem": odormeta.links(o)["pubchem"] if o else None, "vp": vp_url},
        })
    computed = [r for r, c in zip(rows, result["components"]) if c["pa"] is not None]
    total = {f"headspace_{k}": (sum(r["headspace_ug_l"] for r in computed if r["headspace_ug_l"] is not None) if k == "ug_l"
                                else u[3](result["total_pa"], None)) if computed else None for k, u in headspace.UNITS.items()}
    if len(rows) > 1:
        rows.append({**dict.fromkeys(rows[0]), "name": "All together", "solvent": solvent, "model": model, **total, "_links": {}})
    how = "by Henry's law" if water else "by Henry's law, constants estimated by chemutils" if model == "henry" else "by Raoult's law only"
    stem = f"headspace_{solvent.replace(' ', '-')}{f'_{model}' if model else ''}"

    def tex() -> str:
        label, short, tex_units, _ = headspace.UNITS[units]
        key = f"headspace_{units}"
        fmt_amount = (lambda v: f"${v:.2f}$" if math.isfinite(v) else r"$-\infty$") if units == "log10" else _tex_number
        link = lambda r, p, text: _tex_link(r["_links"].get(p), text)
        odor_rows = rows[:len(result["components"])]
        columns = [
            ("Odor", "l", lambda r: _tex_link(r["_links"].get("pubchem"), _tex(r["name"] or "")), True),
            ("Abbrev.", "l", lambda r: _tex_cell(r["abbrev"]), True),
            (_tex_head(r"$\log_{10}$", "conc."), "r", lambda r: f"${format_conc(r['log10_conc'])}$", True),
            (_tex_head("Molar mass", "(g/mol)"), "r",
             lambda r: link(r, "molar_mass", f"{r['molar_mass_g_mol']:.2f}") if r["molar_mass_g_mol"] else "", False),
            (_tex_head("Density", "(g/mL)"), "r",
             lambda r: link(r, "density", _tex_number(r["density_g_ml"])) if r["density_g_ml"] else r"1 (assumed)", True),
            (_tex_head("Vapor pressure", "(kPa)"), "r", lambda r: "" if r["vapor_pressure_kpa"] is None else
             link(r, "vp", _tex_number(r["vapor_pressure_kpa"])) + _tex_measured(r["vapor_pressure_measured"]), False),
            *([(_tex_head(f"$K_H$ in {'water' if water else 'oil'}", "(mol/m$^3$/Pa)"), "r", lambda r: "" if r["k_henry_mol_m3_pa"] is None else
                (link(r, k_prop, _tex_number(r["k_henry_mol_m3_pa"])) + _tex_measured(r["k_henry_measured"])), False)] if henry else []),
            # (only where Raoult's law used it)
            (_tex_head("Mole", "fraction"), "r",
             lambda r: _tex_number(r["mole_fraction"]) if r["mole_fraction"] is not None and r["law"] == "raoult" else "", False),
            (_tex_head("Headspace", f"({tex_units})"), "r", lambda r: fmt_amount(r[key]) if r[key] is not None else "", True),
            ("Notes", r"p{4cm}", lambda r: _tex_cell(r["notes"]), False),
        ]
        last = None
        if len(rows) > 1:
            last = [["All together", *([""] * (len(columns) - 3)), fmt_amount(total[key]) if total[key] is not None else "", ""]]
        legend = (r"\noindent $^{m}$ from measurements of the compound; $^{e}$ estimated, with nothing measured of the compound "
                  r"itself" + (r" (in oil, every $K_H$ is: no measured solubility in oil or alkanes is used; its "
                               r"vapor pressure is the one it was estimated with)" if henry and not water else "")
                  + r". Headspace at equilibrium, at 25\,\textdegree{}C"
                  r" and 1 atm.\par")
        caption = f"Headspace of {len(odor_rows)} odor{'s' if len(odor_rows) != 1 else ''} in {solvent}, {how}, in {label}"
        return _tex_longtable(odor_rows, columns, caption, f"tab:{stem}", last=last, notes=legend)

    return _download([{k: v for k, v in r.items() if k != "_links"} for r in rows], fmt, stem, tex)


# ------------------------------------------------------------------------- zip


@contextmanager
def _zip_text(zf: zipfile.ZipFile, name: str) -> Iterator[TextIO]:
    with zf.open(name, "w", force_zip64=True) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as f:
        yield f


def _stem(ids: list[int], fmt: str) -> str:
    if len(ids) == 1:
        return f"olfdb_dataset-{ids[0]}_{fmt}"
    if len(ids) <= 6:
        return f"olfdb_datasets-{'-'.join(map(str, ids))}_{fmt}"
    return f"olfdb_{len(ids)}-datasets_{fmt}"


def _readme(datasets: list[dict], fmt: str, levels: list[str], shape: tuple[int, int], units: list[str]) -> str:
    lines = [
        "# Odor response database export",
        "",
        f"Exported {dt.datetime.now():%Y-%m-%d %H:%M}; {len(datasets)} dataset{'s' if len(datasets) > 1 else ''}:",
        "",
    ]
    for d in datasets:
        lines.append(
            f"- #{d['id']} {d['title']} ({d['n_flies']} flies, {d['n_units']} ROIs, "
            f"{d['n_odors']} odors; uploaded by {d['uploader_name']} on {d['created_at'][:10]})"
        )
    lines += ["", "## Files", ""]
    if fmt == "wide":
        lines += [
            f"- `responses_wide.csv`: {shape[0]} trials (rows) x {shape[1]} ROIs (columns). Header rows: "
            f"{', '.join(units)}; row index: {', '.join(levels)}. Empty cells are trials a "
            "dataset does not have, or missing values. With pandas:",
            f"  `pd.read_csv('responses_wide.csv', header={list(range(len(units)))}, "
            f"index_col={list(range(len(levels)))})`",
        ]
    elif fmt == "long":
        lines.append(
            f"- `responses_long.csv`: one row per non-missing value; columns "
            f"{', '.join([*units, *levels, 'value'])}."
        )
    if fmt != "files":
        lines.append(
            f"- `odor1`, `odor2`, ...: components presented together; `{SOLVENT_STR}` marks an unused one."
        )
    lines.append("- `datasets.csv`: one row per dataset (details entered at upload, and summary counts).")
    lines.append(
        "- `odors.csv`: each dataset's odors, per panel, with names, PubChem CID and InChI, CAS, supplier and "
        "catalog number, MetaCyc and LOTUS IDs (and links) from the odor table; empty where an odor is not in it."
    )
    if fmt == "files":
        lines.append("- `dataset-<id>/`: every file of each dataset as uploaded, with its README.md and metadata.json.")
    else:
        lines.append("- `dataset-<id>/`: README.md and metadata.json of each dataset.")
    return "\n".join(lines) + "\n"


def write_download(db: Database, ds_ids: Iterable[int], fmt: str, dest: Path, datasets_dir: Path, *,
                   merged: Optional[dict] = None, lower: bool = False) -> tuple[str, str]:
    """Write the download of ``ds_ids`` in format ``fmt`` (see module docstring) to ``dest``.

    Args:
        db: database holding the datasets.
        ds_ids: dataset ids (duplicates are dropped; datasets go in id order).
        fmt: one of ``FORMATS``.
        dest: file to write (a zip, or a CSV for the matrix of a single dataset).
        datasets_dir: directory with one sub-directory of files per dataset id.
        merged: for a merged download, its plan (`merge.plan`), whose ``scales`` multiply
            each dataset's values: then always a zip, named after the coding unit, with
            ``scaling.csv`` and a README section on the scaling.
        lower: odor names in lowercase (see the module's docstring), but for the original files.

    Returns:
        (file name to offer the download as, its media type)

    Raises:
        ExportError: unknown format, no ids, or ids that are not in the database.
    """
    if fmt not in FORMATS:
        raise ExportError(f"format must be one of {', '.join(FORMATS)}")
    ids = sorted(set(ds_ids))
    if not ids:
        raise ExportError("no datasets requested")
    datasets = [db.get_dataset(i) for i in ids]
    missing = [i for i, d in zip(ids, datasets) if d is None]
    if missing:
        raise ExportError(f"dataset{'s' if len(missing) > 1 else ''} not found: {', '.join(map(str, missing))}")
    if len(ids) == 1 and fmt != "files" and merged is None:
        d, m = datasets[0], db.get_matrix_arrays(ids[0])
        if lower:
            _lowercase([m["trials"]])
        levels, keys = dataset_trial_keys(m["trials"], d["summary"].get("row_levels") or trial_keys([m])[0])
        with open(dest, "w", encoding="utf-8", newline="") as f:
            if fmt == "wide":
                dataset_frame(m, levels, keys, CELLS in (d["summary"].get("col_levels") or [])).to_csv(f)
            else:
                write_long(f, [(ids[0], m)], levels, [keys], dataset_id=False)
        return f"{dataset_stem(d, m['trials'])}{'_long' if fmt == 'long' else ''}.csv", "text/csv"

    stem = _stem(ids, fmt)
    matrices = [(i, db.get_matrix_arrays(i)) for i in ids] if fmt != "files" else []
    if merged is not None:
        unit = re.sub(r"[^A-Za-z0-9]+", "-", _ascii(merged["coding_unit"])).strip("-").lower() or "merged"
        stem = f"olfdb_{unit}_{len(ids)}-dataset{'s' if len(ids) > 1 else ''}_{fmt}"
        for i, m in matrices:
            m["values"] = (m["values"] * np.float32(merged["scales"][i])).astype(np.float32)
    names = _lowercase(m["trials"] for _, m in matrices) if lower and matrices else None
    levels, keys = trial_keys([m for _, m in matrices]) if matrices else ([], [])
    shape = (len({k for ks in keys for k in ks}), sum(len(m["units"]) for _, m in matrices))

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        units = unit_levels(m for _, m in matrices)
        readme = _readme(datasets, fmt, levels, shape, units)
        if BAD in units:
            readme += (f"\n`{BAD}`: whether each ROI was marked not to be used, as uploaded (False for those of datasets that "
                       "did not say).\n")
        if names is not None:
            readme += ("\nOdor names (abbreviations) are in lowercase, as chosen in Settings, but for any that differ from "
                       "another here only in case, which are as written.\n")
        if merged is not None:
            readme += "\n".join(merge.readme_lines(merged)) + "\n"
            with _zip_text(zf, f"{stem}/scaling.csv") as f:
                merge.scaling_frame(merged).to_csv(f, index=False)
        zf.writestr(f"{stem}/README.md", readme)
        with _zip_text(zf, f"{stem}/datasets.csv") as f:
            datasets_frame(datasets).to_csv(f, index=False)
        with _zip_text(zf, f"{stem}/odors.csv") as f:
            odors_frame(db, datasets, datasets_dir, names).to_csv(f, index=False)
        if fmt == "wide":
            with _zip_text(zf, f"{stem}/responses_wide.csv") as f:
                wide_frame(matrices).to_csv(f)
        elif fmt == "long":
            with _zip_text(zf, f"{stem}/responses_long.csv") as f:
                write_long(f, matrices, levels, keys)
        for d in datasets:
            src = datasets_dir / str(d["id"])
            prefix = f"{stem}/dataset-{d['id']}"
            if fmt == "files":
                for p in sorted(src.rglob("*")):
                    if p.is_file():
                        zf.write(p, f"{prefix}/{p.relative_to(src).as_posix()}")
            else:
                zf.writestr(f"{prefix}/README.md", d["readme_md"])
                if (src / "metadata.json").is_file():
                    zf.write(src / "metadata.json", f"{prefix}/metadata.json")
    return f"{stem}.zip", "application/zip"
