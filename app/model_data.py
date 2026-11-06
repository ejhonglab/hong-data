"""Model data for the Models tab: the outputs of models of KCs (their spike counts to odors),
read where they are on tensor, from folders added in the Admin tab (`add`; the
``model_sources`` table), never copied or changed.

In such a folder, each ``params.csv`` (at any depth) is an experiment: a row per model, its
parameters (``tag`` and ``variant`` naming it, ``base`` the model it varies), and beside it a
``provenance.json`` (what made them: ``generated``, when; the script, its commits, and the
settings its models share). Each key of that is a parameter of each of its models too,
nested ones named by their path (``base->n_spikes_for_response``), a list as its JSON; a
parameter in both is the row's (its own, even if empty, over its experiment's). A model's
responses are its ``*_spike_counts.parquet`` files, KCs × odors, one per condition (``on``,
``acute_off``, ...), in its experiment's folder or a folder under it (not one with a
params.csv or provenance.json of its own, which are of other models): found by the model's
tag or variant in their names, as whole ``_``-separated parts (``<tag>_<condition>``, or
after its base, ``<base>_sp0.05_<tag>_<condition>``), the longest of them that is, and of
models named alike, the one whose base begins the name.

The table of models (`listing`) has a column per parameter: when ``generated`` first, then
those a folder is added with to put first, then the others, those in more of the files first
(ties in the order they first come), then those to put last; those it is added with to hide,
not at all. Folders are scanned again whenever their files change (`scan`, by their sizes and
times), so models added to them are shown as soon as the tab is opened again.
"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import config, tensor
from .odormeta import OdorIndex
from .odors import OdorFormatError, parse_odor

PARAMS = "params.csv"
PROVENANCE = "provenance.json"
SUFFIX = "_spike_counts.parquet"
GENERATED = "generated"  # provenance.json's key of when its models were made: the table's first column
SEP = "->"  # between the keys of a nested parameter's path
# the column lists a folder is added with, by default (comma-separated)
DEFAULTS = {"hidden": "n_kcs,n_glomeruli", "first": "base", "last": "tag,variant,rate_on,rate_off,source_table"}
LISTS = tuple(DEFAULTS)
MAX_ENTRIES = 100_000  # files and folders scanned in a folder at most (not one of a whole user's data)
N_SPIKES = "n_spikes_for_response"  # how many spikes are a response, of a model's parameters

_lock = threading.Lock()
_scans: dict[str, tuple[tuple, dict]] = {}  # folder on the server -> (its files' sizes and times, its scan)


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_list(text: Any) -> list[str]:
    """A comma-separated list of parameter names as entered, without blanks or repeats."""
    return list(dict.fromkeys(p.strip() for p in str(text or "").split(",") if p.strip()))


# ------------------------------------------------------------------ folders added
def _source(row) -> dict:
    return {"id": row["id"], "path": row["path"], "created_at": row["created_at"], **{k: row[k] for k in LISTS}}


def sources(db) -> list[dict]:
    """The folders added, oldest first: ``id``, ``path`` (on tensor, from its top), ``created_at``
    and each of `LISTS` (as entered)."""
    with db.connect() as con:
        return [_source(r) for r in con.execute("SELECT * FROM model_sources ORDER BY id")]


def _lists(body: dict, old: Optional[dict] = None) -> dict:
    """Each of `LISTS` from a request (cleaned up), else as it was, else its default."""
    return {k: ",".join(parse_list(body[k])) if isinstance(body.get(k), str) else (old or DEFAULTS)[k] for k in LISTS}


def add(db, body: dict, root: Path) -> tuple[int, dict]:
    """Add a folder on tensor ({"path", "hidden", "first", "last"}): refused unless the server
    can read it and finds a params.csv in it. Returns (HTTP status, the folder added as in
    `sources`, with its ``scan`` summary; or {"error"})."""
    c = tensor.check_dir(str(body.get("path") or ""), root)
    if not c["ok"]:
        return 400, {"error": c["error"] or "no folder given"}
    try:
        s = scan(root / c["path"])
    except ScanError as e:
        return 400, {"error": str(e)}
    if not s["experiments"]:
        return 400, {"error": f"no {PARAMS} in it, nor in the folders under it"}
    with db.connect() as con:
        if con.execute("SELECT 1 FROM model_sources WHERE path = ?", (c["path"],)).fetchone():
            return 409, {"error": f"{c['path']} was added already"}
        lists = _lists(body)
        cur = con.execute("INSERT INTO model_sources (path, created_at, hidden, first, last) VALUES (?, ?, ?, ?, ?)",
                          (c["path"], now(), lists["hidden"], lists["first"], lists["last"]))
        row = con.execute("SELECT * FROM model_sources WHERE id = ?", (cur.lastrowid,)).fetchone()
    return 200, {**_source(row), "scan": summary(s)}


def update(db, source_id: int, body: dict) -> tuple[int, dict]:
    """Change a folder's lists of parameters (any of `LISTS`)."""
    with db.connect() as con:
        row = con.execute("SELECT * FROM model_sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            return 404, {"error": "no such folder of model data"}
        lists = _lists(body, _source(row))
        con.execute("UPDATE model_sources SET hidden = ?, first = ?, last = ? WHERE id = ?",
                    (lists["hidden"], lists["first"], lists["last"], source_id))
        row = con.execute("SELECT * FROM model_sources WHERE id = ?", (source_id,)).fetchone()
    return 200, _source(row)


def delete(db, source_id: int) -> bool:
    """Forget a folder (nothing on tensor is touched). False if there was none."""
    with db.connect() as con:
        return con.execute("DELETE FROM model_sources WHERE id = ?", (source_id,)).rowcount > 0


# ------------------------------------------------------------------ scanning a folder
class ScanError(Exception):
    pass


def _value(text: str) -> Any:
    """A params.csv cell as a value: None if empty, a bool for True / False, else a number if it
    is a finite one, else the text."""
    if text == "":
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        f = float(text)
    except ValueError:
        return text
    if not math.isfinite(f):
        return text
    return int(f) if f.is_integer() and text.lstrip("+-").isdigit() else f


def _json_value(v: Any) -> Any:
    """A provenance.json value as a cell: a list (or an empty object) as its JSON, a number not
    finite as its text."""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


def flatten(d: dict, prefix: str = "") -> dict:
    """A provenance.json's keys as parameters, nested ones by their path (`SEP` between)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flatten(v, f"{prefix}{k}{SEP}"))
        else:
            out[f"{prefix}{k}"] = _json_value(v)
    return out


def natural(s: str) -> list:
    """A sort key of text with its numbers in their order (E2 before E10)."""
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", s)]


def _files(folder: Path) -> list[tuple[str, int, int]]:
    """The params.csv, provenance.json and spike count files under a folder: (path from it,
    size, time), in order. Raises ScanError if it has over `MAX_ENTRIES`."""
    out, n = [], 0
    for d, dirs, names in os.walk(folder, onerror=lambda e: None):
        dirs.sort()
        n += len(dirs) + len(names)
        if n > MAX_ENTRIES:
            raise ScanError(f"over {MAX_ENTRIES:,} files and folders in it: choose the folder of the model outputs alone")
        for name in sorted(names):
            if name in (PARAMS, PROVENANCE) or name.endswith(SUFFIX):
                p = Path(d) / name
                try:
                    st = p.stat()
                except OSError:
                    continue
                out.append((p.relative_to(folder).as_posix(), st.st_size, st.st_mtime_ns))
    return out


def scan(folder: Path) -> dict:
    """A folder's experiments and models (see the module's docstring), made again only when its
    files have changed.

    Returns:
        dict with ``experiments`` (``key``: its folder, from the one scanned ('' for it); ``params``:
        params.csv's columns; ``provenance``: provenance.json as it is, or None; ``flat``: it as
        parameters; ``error``: why one of them could not be read, or None), ``models`` (``experiment``,
        ``row`` (its index in params.csv), ``params`` (its row, a value per column), ``runs``
        (``file`` (from the folder scanned), ``label`` (its condition, after the folder under its
        experiment's it is in, if any), ``prefix`` (what its name has before the model's)),
        ``n_spikes`` ({``value``, ``from``}, or None: `n_spikes_for_response`)), ``keys``
        (``name``, ``files`` (of how many params.csv and provenance.json files)), in the order
        they first come), and ``unmatched`` (spike count files of no model).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ScanError(f"{folder} is not a folder the server can read")
    files = _files(folder)
    sig = tuple(files)
    with _lock:
        hit = _scans.get(str(folder))
    if hit and hit[0] == sig:
        return hit[1]
    out = _scan(folder, [f for f, _, _ in files])
    with _lock:
        _scans[str(folder)] = (sig, out)
    return out


def _scan(folder: Path, files: list[str]) -> dict:
    exp_dirs = sorted({"" if (d := str(Path(f).parent)) == "." else d for f in files if Path(f).name == PARAMS}, key=natural)
    experiments, models, counts, unmatched = [], [], {}, []

    def seen(keys) -> None:
        for k in keys:
            counts[k] = counts.get(k, 0) + 1

    def owner(f: str) -> Optional[str]:
        """The experiment a file is of: the nearest folder above it with a params.csv, unless one
        with a provenance.json alone (models of no params.csv) is nearer."""
        parts = Path(f).parent.parts
        for i in range(len(parts), -1, -1):
            d = "/".join(parts[:i])
            if d in own:
                return d if d in exp_set else None
        return None

    exp_set = set(exp_dirs)
    own = exp_set | {"" if (d := str(Path(f).parent)) == "." else d for f in files if Path(f).name == PROVENANCE}
    runs_of: dict[str, list[str]] = {}
    for f in files:
        if f.endswith(SUFFIX):
            e = owner(f)
            if e is None:
                unmatched.append(f)
            else:
                runs_of.setdefault(e, []).append(f)
    for key in exp_dirs:
        d = folder / key
        exp = {"key": key, "params": [], "provenance": None, "flat": {}, "error": None}
        rows = []
        try:
            with open(d / PARAMS, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                exp["params"] = [c for c in (reader.fieldnames or []) if c]
                rows = [{c: _value((r.get(c) or "").strip()) for c in exp["params"]} for r in reader]
        except (OSError, UnicodeDecodeError, csv.Error) as e:
            exp["error"] = f"{PARAMS} could not be read: {e}"
        seen(exp["params"])
        if (d / PROVENANCE).is_file():
            try:
                exp["provenance"] = json.loads((d / PROVENANCE).read_text(encoding="utf-8"))
                if not isinstance(exp["provenance"], dict):
                    raise ValueError("it is not a JSON object")
                exp["flat"] = flatten(exp["provenance"])
            except (OSError, UnicodeDecodeError, ValueError) as e:
                exp["provenance"], exp["error"] = None, f"{PROVENANCE} could not be read: {e}"
            seen(exp["flat"])
        experiments.append(exp)
        first = len(models)
        for i, r in enumerate(rows):
            models.append({"experiment": key, "row": i, "params": r, "runs": [], "n_spikes": _n_spikes(r, exp)})
        for f in runs_of.get(key, []):
            hits = _match(f, key, models[first:])
            if not hits:
                unmatched.append(f)
            for m, run in hits:
                m["runs"].append(run)
    for m in models:
        m["runs"].sort(key=lambda r: (r["label"].count("/"), r["label"] not in ("on", ""), r["label"]))
    order = {}
    for e in experiments:
        for k in [*e["params"], *e["flat"]]:
            order.setdefault(k, len(order))
    keys = [{"name": k, "files": counts[k]} for k in sorted(order, key=order.get)]
    return {"experiments": experiments, "models": models, "keys": keys, "unmatched": sorted(unmatched)}


def _n_spikes(params: dict, exp: dict) -> Optional[dict]:
    """How many spikes are a response of a model: its own, else its experiment's base's (as
    ``base->...``, ``bases-><its base>->...`` or ``variants-><its base, variant or tag>->...``)."""
    if params.get(N_SPIKES) is not None:
        return {"value": params[N_SPIKES], "from": f"{N_SPIKES} in {PARAMS}"}
    named = [params.get(k) for k in ("base", "variant", "tag") if params.get(k) is not None]
    for k in (f"base{SEP}{N_SPIKES}", f"bases{SEP}{params.get('base')}{SEP}{N_SPIKES}",
              *(f"variants{SEP}{n}{SEP}{N_SPIKES}" for n in named), N_SPIKES):
        v = exp["flat"].get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return {"value": v, "from": f"{k} in {PROVENANCE}"}
    return None


def _spans(stem: str, name: str) -> list[tuple[int, int]]:
    """Where `name` is in `stem` as whole '_'-separated parts."""
    out, i = [], stem.find(name)
    while name and i >= 0:
        j = i + len(name)
        if (i == 0 or stem[i - 1] == "_") and (j == len(stem) or stem[j] == "_"):
            out.append((i, j))
        i = stem.find(name, i + 1)
    return out


def _match(f: str, exp: str, models: list[dict]) -> list[tuple[dict, dict]]:
    """The models of an experiment a spike count file is of (see the module's docstring), each
    with the file as one of its runs; none if it names none of them."""
    rel = Path(f).relative_to(exp) if exp else Path(f)
    stem = rel.name[: -len(SUFFIX)]
    sub = rel.parent.as_posix()
    best, hits = None, []
    for m in models:
        p = m["params"]
        for name in dict.fromkeys(str(p[k]) for k in ("tag", "variant") if p.get(k) is not None):
            for i, j in _spans(stem, name):
                prefix, base = stem[:i].rstrip("_"), str(p.get("base") or "")
                score = (len(name), len(base) if base and (prefix == base or prefix.startswith(base + "_")) else 0, -i)
                cond = stem[j:].lstrip("_")
                run = {"file": f, "label": f"{sub}/{cond}" if sub != "." else cond, "prefix": prefix}
                if best is None or score > best:
                    best, hits = score, [(m, run)]
                elif score == best and all(h[0] is not m for h in hits):
                    hits.append((m, run))
    return hits


def summary(s: dict) -> dict:
    """How much a scan found: experiments, models, runs, spike count files of no model."""
    return {"experiments": len(s["experiments"]), "models": len(s["models"]),
            "runs": sum(len(m["runs"]) for m in s["models"]), "unmatched": len(s["unmatched"])}


# ------------------------------------------------------------------ the table
def columns(keys: list[dict], hidden: list[str], first: list[str], last: list[str]) -> list[str]:
    """The table's columns: `GENERATED`, `first`, the others (those in more files first, ties
    in the order they first come), `last`; but those `hidden`, and those no file has."""
    have = {k["name"] for k in keys}
    put = {GENERATED, *first, *last}
    rest = [k["name"] for k in sorted(keys, key=lambda k: -k["files"]) if k["name"] not in put]  # (sorted is stable)
    return [c for c in dict.fromkeys([GENERATED, *first, *rest, *last]) if c in have and c not in hidden]


def listing(db, root: Path) -> dict:
    """The Models tab's table: every model of every folder added.

    Returns:
        dict with ``sources`` (as `sources`, with ``ok``, ``error`` and the ``scan`` summary),
        ``columns`` (`columns`, of every folder's keys: their lists joined), ``experiments`` (key
        ``<source id>/<folder>`` -> ``source``, ``folder``, ``params``, ``flat``, ``error``),
        ``models`` (as `scan`'s, ``experiment`` that key, each run's ``id`` its file's path on
        tensor, from its top), and ``unmatched`` (paths on tensor too).
    """
    out = {"sources": [], "experiments": {}, "models": [], "unmatched": []}
    keys: dict[str, int] = {}
    lists = {k: [] for k in LISTS}
    for src in sources(db):
        c = tensor.check_dir(src["path"], root)
        entry = {**src, "ok": False, "error": c["error"], "scan": None}
        out["sources"].append(entry)
        for k in LISTS:
            lists[k] += parse_list(src[k])
        if not c["ok"]:
            continue
        try:
            s = scan(root / c["path"])
        except ScanError as e:
            entry["error"] = str(e)
            continue
        entry.update(ok=True, scan=summary(s))
        for k in s["keys"]:
            keys[k["name"]] = keys.get(k["name"], 0) + k["files"]
        for e in s["experiments"]:
            out["experiments"][f"{src['id']}/{e['key']}"] = {"source": src["id"], "folder": e["key"], "params": e["params"],
                                                            "flat": e["flat"], "error": e["error"]}
        for m in s["models"]:
            out["models"].append({**m, "experiment": f"{src['id']}/{m['experiment']}",
                                  "runs": [{**r, "id": f"{c['path']}/{r['file']}"} for r in m["runs"]]})
        out["unmatched"] += [f"{c['path']}/{f}" for f in s["unmatched"]]
    out["columns"] = columns([{"name": k, "files": n} for k, n in keys.items()],
                             set(lists["hidden"]), list(dict.fromkeys(lists["first"])), list(dict.fromkeys(lists["last"])))
    return out


# ------------------------------------------------------------------ a model's responses
def _odor_axis(ix: pd.Index) -> bool:
    """Whether a table's index (or columns) is of odors: named so, or mostly odor strings."""
    if any(n and "odor" in str(n).lower() for n in ix.names):
        return True
    labels = [str(x) for x in ix[:50]] if ix.nlevels == 1 else []
    return bool(labels) and sum(" @ " in x for x in labels) > len(labels) / 2


def run_file(db, run_id: str, root: Path) -> tuple[dict, Path, dict]:
    """The source, spike count file and experiment (as `scan`'s) of a run (its file's path on
    tensor, from its top: in the folder added nearest it). Raises LookupError for one that is
    not a run of a folder added (or no longer readable)."""
    srcs = [s for s in sources(db) if run_id.startswith(s["path"] + "/")]
    src = max(srcs, key=lambda s: len(s["path"]), default=None)
    rel = run_id[len(src["path"]) + 1:] if src else ""
    if src is None or not rel.endswith(SUFFIX):
        raise LookupError("no such model run")
    c = tensor.check_dir(src["path"], root)
    if not c["ok"]:
        raise LookupError(f"its folder, {src['path']}, cannot be read: {c['error']}")
    folder = root / c["path"]
    try:
        s = scan(folder)
    except ScanError as e:
        raise LookupError(str(e)) from None
    key = next((m["experiment"] for m in s["models"] if any(r["file"] == rel for r in m["runs"])), None)
    if key is None:
        raise LookupError(f"{rel} is not a model run of {src['path']} (any more)")
    return src, folder / rel, next(e for e in s["experiments"] if e["key"] == key)


def read_run(path: Path, index: OdorIndex) -> dict:
    """A model run's spike counts, KCs as rows (transposed if its file has them as columns).

    Returns:
        dict with ``kcs`` (``labels``: each KC's first index level, e.g. its ``kc_id``;
        ``groups``: its ``kc_type``, if it has one, else None; ``levels``: the names of the
        file's levels of them), ``odors`` (each as in the file, ``odor``, with ``odor_id``,
        ``chem`` (``name``, ``abbrev``, ``cid``) and ``log10_conc``, if it is one of the odor
        table alone, else None), ``transposed`` (whether the file had KCs as columns) and
        ``values_b64`` (float32, KCs × odors, NaN where missing).
    """
    df = pd.read_parquet(path)
    transposed = _odor_axis(df.index) and not _odor_axis(df.columns)
    if transposed:
        df = df.T
    ix = df.index
    names = [str(n) if n is not None else f"level {i}" for i, n in enumerate(ix.names)]
    level = lambda i: [str(x) for x in ix.get_level_values(i)]  # noqa: E731
    groups = level(names.index("kc_type")) if "kc_type" in names else None
    odors = []
    for col in df.columns:
        label = str(col)
        odor_id = conc = None
        try:
            o = parse_odor(label, strict=False)
            odor_id = index.match(o.name)[0]
            conc = o.log10_conc
        except OdorFormatError:
            odor_id = index.match(label)[0]
        chem = index.by_id.get(odor_id) if odor_id is not None else None
        odors.append({"odor": label, "odor_id": odor_id, "log10_conc": conc,
                      "chem": {"name": chem["name"], "abbrev": chem.get("abbrev"), "cid": chem.get("cid")} if chem else None})
    values = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return {"kcs": {"labels": level(0), "groups": groups, "levels": names}, "odors": odors, "transposed": transposed,
            "values_b64": base64.b64encode(np.ascontiguousarray(values).tobytes()).decode()}


def run(db, run_id: str, index: OdorIndex, root: Optional[Path] = None) -> dict:
    """A model run for the Models tab: `read_run`'s, with its ``id`` and ``file`` (its path on
    tensor, from its top) and its experiment's ``provenance`` (as it is, or None). Raises
    LookupError (see `run_file`)."""
    _, path, exp = run_file(db, run_id, root or config.TENSOR_ROOT)
    return {"id": run_id, "file": run_id, "provenance": exp["provenance"], **read_run(path, index)}
