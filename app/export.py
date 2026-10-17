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
and a name made from the dataset's details (`dataset_stem`).
"""
from __future__ import annotations

import datetime as dt
import io
import re
import unicodedata
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import numpy as np
import pandas as pd

from . import odormeta
from .db import Database
from .ingest import COL_LEVELS
from .odors import SOLVENT_STR, is_odor_level, odor_level_sort_key

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


def wide_frame(matrices: list[tuple[int, dict]]) -> pd.DataFrame:
    """Trials x ROIs of several datasets, outer-joined on the trial key.

    Args:
        matrices: (dataset id, `Database.get_matrix_arrays` dict) pairs, in column order.

    Returns:
        float32 DataFrame; rows are a MultiIndex of `trial_keys` levels (in order of
        first appearance), columns a MultiIndex of ``UNIT_LEVELS``. Cells for trials a
        dataset does not have are NaN.
    """
    levels, keys = trial_keys([m for _, m in matrices])
    row_index: dict[tuple, int] = {}
    rows = [np.array([row_index.setdefault(k, len(row_index)) for k in ks], dtype=np.intp) for ks in keys]
    values = np.full((len(row_index), sum(len(m["units"]) for _, m in matrices)), np.nan, dtype=np.float32)
    cols = []
    for (ds_id, m), r in zip(matrices, rows):
        values[r, len(cols):len(cols) + len(m["units"])] = m["values"].T
        cols += [(ds_id, u["date"], u["fly_num"], u["roi"]) for u in m["units"]]
    return pd.DataFrame(
        values,
        index=pd.MultiIndex.from_tuples(list(row_index), names=levels),
        columns=pd.MultiIndex.from_tuples(cols, names=list(UNIT_LEVELS)),
    )


def dataset_frame(m: dict, levels: list[str], keys: list[tuple]) -> pd.DataFrame:
    """One dataset's trials x ROIs as uploaded: rows indexed by `levels`, columns by date / fly_num / roi.

    Args:
        m: as returned by `Database.get_matrix_arrays`.
        levels, keys: from `dataset_trial_keys`.
    """
    return pd.DataFrame(
        m["values"].T,
        index=pd.MultiIndex.from_tuples(keys, names=levels),
        columns=pd.MultiIndex.from_tuples([(u["date"], u["fly_num"], u["roi"]) for u in m["units"]], names=list(COL_LEVELS)),
    )


def write_long(f: TextIO, matrices: list[tuple[int, dict]], levels: list[str], keys: list[list[tuple]], *,
               dataset_id: bool = True) -> None:
    """Write one CSV row per non-missing value: [dataset_id,] date, fly_num, roi, trial levels, value.

    Args:
        matrices: (dataset id, `Database.get_matrix_arrays` dict) pairs.
        levels, keys: trial levels, and the keys of each dataset's trials (from `trial_keys`, or
            `dataset_trial_keys` for a single dataset).
        dataset_id: whether to write the dataset_id column.
    """
    header = True
    for (ds_id, m), ks in zip(matrices, keys):
        trial_cols = [np.array(col, dtype=object) for col in zip(*ks)]
        for start in range(0, len(m["units"]), LONG_CHUNK_UNITS):
            units = m["units"][start:start + LONG_CHUNK_UNITS]
            block = m["values"][start:start + len(units)]
            ui, ti = np.nonzero(np.isfinite(block))
            df = pd.DataFrame({
                **({"dataset_id": np.full(len(ui), ds_id)} if dataset_id else {}),
                "date": np.array([u["date"] for u in units], dtype=object)[ui],
                "fly_num": np.array([u["fly_num"] for u in units])[ui],
                "roi": np.array([u["roi"] for u in units], dtype=object)[ui],
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
        row["notes"] = md.get("notes")
        rows.append(row)
    return pd.DataFrame(rows)


def odors_frame(db: Database, datasets: list[dict], datasets_dir: Path) -> pd.DataFrame:
    """Each dataset's odors (per panel, as in the data) with their odor-table metadata and links."""
    index = odormeta.OdorIndex(db.list_odors())
    rows = []
    for d in datasets:
        odors = odormeta.dataset_odors(d, db.get_trials(d["id"]), index, odormeta.dataset_yaml_odors(d, datasets_dir))
        rows += [{"dataset_id": d["id"], **r} for r in odormeta.table_rows(odors, index)]
    df = pd.DataFrame(rows)
    if "cid" in df:
        df["cid"] = df["cid"].astype("Int64")
    return df


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


def _readme(datasets: list[dict], fmt: str, levels: list[str], shape: tuple[int, int]) -> str:
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
            f"{', '.join(UNIT_LEVELS)}; row index: {', '.join(levels)}. Empty cells are trials a "
            "dataset does not have, or missing values. With pandas:",
            f"  `pd.read_csv('responses_wide.csv', header={list(range(len(UNIT_LEVELS)))}, "
            f"index_col={list(range(len(levels)))})`",
        ]
    elif fmt == "long":
        lines.append(
            f"- `responses_long.csv`: one row per non-missing value; columns "
            f"{', '.join([*UNIT_LEVELS, *levels, 'value'])}."
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


def write_download(db: Database, ds_ids: Iterable[int], fmt: str, dest: Path, datasets_dir: Path) -> tuple[str, str]:
    """Write the download of ``ds_ids`` in format ``fmt`` (see module docstring) to ``dest``.

    Args:
        db: database holding the datasets.
        ds_ids: dataset ids (duplicates are dropped; datasets go in id order).
        fmt: one of ``FORMATS``.
        dest: file to write (a zip, or a CSV for the matrix of a single dataset).
        datasets_dir: directory with one sub-directory of files per dataset id.

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
    if len(ids) == 1 and fmt != "files":
        d, m = datasets[0], db.get_matrix_arrays(ids[0])
        levels, keys = dataset_trial_keys(m["trials"], d["summary"].get("row_levels") or trial_keys([m])[0])
        with open(dest, "w", encoding="utf-8", newline="") as f:
            if fmt == "wide":
                dataset_frame(m, levels, keys).to_csv(f)
            else:
                write_long(f, [(ids[0], m)], levels, [keys], dataset_id=False)
        return f"{dataset_stem(d, m['trials'])}{'_long' if fmt == 'long' else ''}.csv", "text/csv"

    stem = _stem(ids, fmt)
    matrices = [(i, db.get_matrix_arrays(i)) for i in ids] if fmt != "files" else []
    levels, keys = trial_keys([m for _, m in matrices]) if matrices else ([], [])
    shape = (len({k for ks in keys for k in ks}), sum(len(m["units"]) for _, m in matrices))

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(f"{stem}/README.md", _readme(datasets, fmt, levels, shape))
        with _zip_text(zf, f"{stem}/datasets.csv") as f:
            datasets_frame(datasets).to_csv(f, index=False)
        with _zip_text(zf, f"{stem}/odors.csv") as f:
            odors_frame(db, datasets, datasets_dir).to_csv(f, index=False)
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
