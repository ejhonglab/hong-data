"""Folders on the lab NAS (tensor), as mounted on the server, and finding a dataset's files there.

Users give folders relative to the top of tensor (`config.TENSOR_ROOT` on the server).
Their data is looked for in the layout hong2p uses::

    <raw or motion-corrected folder>/<YYYY-MM-DD>/<fly_num>/<recording>/Experiment.xml

(dates may also be written YYYYMMDD or YYYY_MM_DD, fly folders ``fly1``, ``fly_01``, ...),
and odor YAMLs up to `YAML_DEPTH` folders below an odor metadata folder: those named in
the notes of a fly's ThorImage recordings, or with one of the flies' dates in their path.
Only what is in the folders the server may read (`sandbox.can_read`) is listed or read.
"""
from __future__ import annotations

import datetime as dt
import itertools
import os
import posixpath
import re
from pathlib import Path
from typing import Optional

from . import sandbox, thorimage, yamlcheck
from .odors import format_conc

USER_DIR_KINDS = ("raw_dirs", "mocorr_dirs", "odor_dirs")
DATE_FORMATS = ("%Y-%m-%d", "%Y%m%d", "%Y_%m_%d")

# bounds on how much of tensor one lookup reads
MAX_LISTING = 5000  # entries read from any one folder
YAML_DEPTH = 3
MAX_YAML_DIRS = 2000  # folders read looking for YAMLs, per odor metadata folder
MAX_YAML_FILES = 20000
MAX_XMLS_READ = 100
MAX_YAMLS_PARSED = 200
MAX_FILE_BYTES = 20 * 1024 * 1024
LISTED = 200  # entries listed of a dataset's data folder
# folders of anatomical recordings (higher resolution, no fast sampling), not those of the
# functional ones a dataset's Experiment.xml should be from
ANATOMICAL = re.compile(r"(?:anat|anatomical)(?:_.*)?")

NOT_ALLOWED = "outside the folders the server may read (set in the Admin tab)"
NAMED = "named in the notes of an Experiment.xml"
DATED = "a fly's date in its path"


class PathError(ValueError):
    pass


def _guess_relative(p: str, root: Path) -> str:
    """The path from the top of tensor that an absolute path (``/`` separated) seems to mean:
    after tensor's top on the server, a drive letter, or a folder named like tensor's top
    (``/Volumes/tensor/...``); '' if it cannot be told."""
    top = str(root).rstrip("/") + "/"
    drive = re.match(r"[A-Za-z]:/*(.*)", p)
    if p.startswith(top):
        rest = p[len(top):]
    elif drive:
        rest = drive[1]
    else:
        parts = p.split("/")
        at = [i for i, x in enumerate(parts) if x.casefold() == root.name.casefold()]
        rest = "/".join(parts[at[-1] + 1:]) if at else ""
    return rest.strip("/")


def _absolute_error(p: str, root: Path) -> str:
    msg = "absolute paths are not accepted: give the folder relative to the top of tensor"
    rest = _guess_relative(p, root)
    return f"{msg} (did you mean '{rest}'?)" if rest else msg


def normalize(path: str, root: Path) -> str:
    """A path relative to the top of tensor, cleaned up (``a/b``).

    Backslashes are taken as separators; empty and ``.`` components are dropped.

    Raises:
        PathError: for absolute paths (with the relative path meant, when it can be
            guessed), ``..`` components, or a path naming the top of tensor itself.
    """
    p = path.strip().replace("\\", "/")
    if p.startswith(("/", "~")) or re.match(r"[A-Za-z]:", p):
        raise PathError(_absolute_error(p, root))
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if ".." in parts:
        raise PathError("'..' is not allowed: give the folder from the top of tensor")
    if not parts:
        raise PathError("give a folder inside tensor, not the top of it")
    return "/".join(parts)


def _entries(d: Path) -> list[os.DirEntry]:
    if not sandbox.can_read(d):
        return []
    try:
        with os.scandir(d) as it:
            return sorted(itertools.islice(it, MAX_LISTING), key=lambda e: e.name)
    except OSError:
        return []


def _is_dir(e: os.DirEntry) -> bool:
    try:
        return e.is_dir()
    except OSError:
        return False


def _subdirs(d: Path) -> list[Path]:
    return [Path(e.path) for e in _entries(d) if _is_dir(e)]


def root_status(root: Path) -> dict:
    """Whether tensor is there on the server: ``{'path', 'ok', 'error'}``."""
    error = None
    allowed = sandbox.readable_within(root)
    if not os.path.isdir(root):
        error = f"tensor is not available on the server ({root} does not exist)"
    elif not allowed:
        error = f"no folder on tensor ({root}) is among those the server may read (set in the Admin tab)"
    elif not os.access(root, os.R_OK | os.X_OK):
        error = f"the server is not allowed to read tensor ({root})"
    elif not any(_entries(d) for d in allowed):
        error = f"tensor looks unmounted on the server ({', '.join(map(str, allowed))} {'is' if len(allowed) == 1 else 'are'} empty)"
    return {"path": str(root), "ok": error is None, "error": error}


def _not_found(root: Path, parts: list[str]) -> str:
    d = root
    for i, part in enumerate(parts):
        if not os.path.lexists(d / part):
            where = f"'{'/'.join(parts[:i])}'" if i else "the top of tensor"
            msg = f"not found on tensor: {where} has no '{part}'"
            close = [e.name for e in _entries(d) if e.name.casefold() == part.casefold()]
            if close:
                msg += f" (did you mean '{'/'.join([*parts[:i], close[0]])}'?)"
            return msg
        d = d / part
    return "not found on tensor"


def _locate(path: str, root: Path) -> tuple[str, Path]:
    """Normalized `path` and where it is on the server, checked to be inside tensor (raises PathError)."""
    rel = normalize(path, root)
    if not os.path.isdir(root):
        raise PathError(f"tensor is not available on the server ({root} does not exist)")
    target = root / rel
    if not os.path.lexists(target):
        raise PathError(_not_found(root, rel.split("/")))
    if not os.path.exists(target):
        raise PathError("a link to something that does not exist")
    real_root = root.resolve()
    real = target.resolve()
    if real != real_root and real_root not in real.parents:
        raise PathError("leads outside tensor (through a link)")
    if not sandbox.can_read(real):
        raise PathError(NOT_ALLOWED)
    return rel, target


def check_dir(path: str, root: Path) -> dict:
    """Check a folder a user entered: ``{'input', 'path' (normalized, or None), 'ok', 'error'}``."""
    out = {"input": path, "path": None, "ok": False, "error": None}
    try:
        rel, target = _locate(path, root)
        out["path"] = rel
        if not os.path.isdir(target):
            raise PathError("this is a file, not a folder")
        if not os.access(target, os.R_OK | os.X_OK):
            raise PathError("exists, but the server is not allowed to read it")
    except PathError as e:
        out["error"] = str(e)
        return out
    out["ok"] = True
    return out


def check_data_path(path: str, root: Path) -> dict:
    """Check a dataset's data folder as entered (from the top of tensor), as `check_dir`
    does, and give a ``warning`` too: while tensor itself is not there to look in
    (`root_status`), a well-formed path is accepted with one, as it cannot be told wrong."""
    out = {**check_dir(path, root), "warning": None}
    if not out["ok"]:
        status = root_status(root)
        if not status["ok"]:
            try:
                out.update(path=normalize(path, root), ok=True, error=None, warning=f"not checked: {status['error']}")
            except PathError:
                pass
    return out


def file_path(path: str, root: Path, suffixes: tuple[str, ...]) -> tuple[str, Path]:
    """Normalized path and location on the server of a file on tensor ending in one of `suffixes`.

    Raises:
        PathError: if it is not such a file, cannot be read, or is over `MAX_FILE_BYTES`.
    """
    rel, target = _locate(path, root)
    if not rel.lower().endswith(suffixes):
        raise PathError(f"'{rel}' is not a {' / '.join(suffixes)} file")
    if not os.path.isfile(target):
        raise PathError(f"'{rel}' is not a file")
    if not os.access(target, os.R_OK):
        raise PathError(f"the server is not allowed to read '{rel}'")
    if os.path.getsize(target) > MAX_FILE_BYTES:
        raise PathError(f"'{rel}' is larger than {MAX_FILE_BYTES >> 20} MB")
    return rel, target


def find_data(path: Optional[str], root: Path, bases: list[Path]) -> dict:
    """Where a dataset's data (its raw or motion-corrected data path, as entered at upload) is on the server.

    The path is looked for from the top of tensor (`root`), then from each of `bases` (the
    folders set in the Admin tab). An absolute path is also looked for as it is, and as
    the path on tensor it seems to mean (one copied on a computer that mounts tensor as
    ``Z:``, say). Only what the server may read counts as found.

    Returns:
        dict with ``input`` (as entered), ``status`` ('ok'; 'none' if nothing was entered,
        'not_found', 'not_allowed': outside the folders the server may read, or
        'unreadable'), ``error`` (the reason, if not ok), ``path`` (where it is on the
        server), ``tensor_path`` (from the top of tensor, if it is in it), ``is_dir``, and
        for a folder its first `LISTED` ``entries`` (``name``, ``dir``, ``size``) of ``n_entries``.
    """
    out = {"input": path, "status": "none", "error": "none was entered at upload", "path": None, "tensor_path": None,
           "is_dir": None, "entries": [], "n_entries": 0}
    p = (path or "").strip().replace("\\", "/")
    if not p:
        return out
    absolute = p.startswith(("/", "~")) or bool(re.match(r"[A-Za-z]:", p))
    parts = [x for x in (_guess_relative(p, root) if absolute else p).split("/") if x not in ("", ".")]
    candidates = [Path(posixpath.normpath(p))] if p.startswith("/") else []
    if parts and ".." not in parts:
        candidates += [base.joinpath(*parts) for base in [root, *bases]]
    found = None
    out.update(status="not_found", error=f"not found on the server (looked in {root}"
                                          + (" and the folders set in the Admin tab)" if bases else ")"))
    for c in dict.fromkeys(candidates):
        if not os.path.exists(c):
            continue
        if not sandbox.can_read(c):
            out.update(status="not_allowed", error=NOT_ALLOWED)
        elif not os.access(c, os.R_OK):
            out.update(status="unreadable", error="the server is not allowed to read it")
        else:
            found = c
            break
    if found is None:
        return out
    real_root = root.resolve()
    real = found.resolve()
    out.update(status="ok", error=None, path=str(found), is_dir=os.path.isdir(found),
               tensor_path=real.relative_to(real_root).as_posix() if real_root in real.parents else None)
    if out["is_dir"]:
        entries = _entries(found)

        def size(e: os.DirEntry) -> Optional[int]:
            try:
                return e.stat().st_size
            except OSError:
                return None

        out["n_entries"] = len(entries)
        out["entries"] = [{"name": e.name, "dir": _is_dir(e), "size": None if _is_dir(e) else size(e)} for e in entries[:LISTED]]
    return out


# ------------------------------------------------------------------ lookup


def date_names(date: str) -> list[str]:
    """Ways a fly's date (YYYY-MM-DD) may be written in folder and file names."""
    d = dt.date.fromisoformat(date)
    return [d.strftime(f) for f in DATE_FORMATS]


def fly_dirs(folder: Path, date: str, fly_num: int) -> list[Path]:
    """Folders of one fly (by its date and number) in `folder`: ``<folder>/<date>/<fly>``,
    or ``<folder>/<fly>`` if `folder` is named by the date, or `folder` itself if it is
    that fly's (``<date>/<fly>``). Never those of any other fly."""
    names = date_names(date)
    pattern = re.compile(rf"(?:fly)?[_-]?0*{int(fly_num)}", re.IGNORECASE)
    if folder.parent.name in names and pattern.fullmatch(folder.name):
        return [folder]
    date_dirs = [folder] if folder.name in names else [folder / n for n in names if os.path.isdir(folder / n)]
    return [d for dd in date_dirs for d in _subdirs(dd) if pattern.fullmatch(d.name)]


def recordings(fly_dir: Path) -> list[Path]:
    """Experiment.xml of each functional ThorImage recording of a fly (in its folder, or one
    level down; not in `ANATOMICAL` folders), oldest first (by when the file was last
    written, then by name)."""
    def mtime(p: Path) -> float:
        try:
            return os.path.getmtime(p)
        except OSError:
            return 0.0

    dirs = [fly_dir, *(d for d in _subdirs(fly_dir) if not ANATOMICAL.fullmatch(d.name))]
    return sorted((d / "Experiment.xml" for d in dirs if os.path.isfile(d / "Experiment.xml")), key=mtime)


def _yaml_files(folder: Path) -> list[Path]:
    out, todo, n_dirs = [], [(folder, 0)], 0
    while todo and len(out) < MAX_YAML_FILES and n_dirs < MAX_YAML_DIRS:
        d, depth = todo.pop()
        n_dirs += 1
        for e in _entries(d):
            if _is_dir(e):
                if depth < YAML_DEPTH:
                    todo.append((Path(e.path), depth + 1))
            elif e.name.lower().endswith((".yaml", ".yml")):
                out.append(Path(e.path))
    return out


def _read(path: Path) -> Optional[str]:
    if not sandbox.can_read(path):
        return None
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _common(paths: list[str]) -> Optional[str]:
    if not paths:
        return None
    return posixpath.commonpath(paths) or "; ".join(dict.fromkeys(paths))


def _odor_yamls(root: Path, folders: list[Path], xmls: list[Path], flies: list[dict],
                matrix_odors: list[tuple[str, float]]) -> dict:
    named: set[str] = set()
    for x in xmls[:MAX_XMLS_READ]:
        text = _read(x)
        if text:
            named.update(thorimage.notes_yaml_names(text))
    index = [p for f in folders for p in _yaml_files(f)]
    dates = {n for f in flies for n in date_names(f["date"])}
    reasons: dict[Path, str] = {}
    for p in index:
        if p.name in named:
            reasons.setdefault(p, NAMED)
    for p in index:
        if any(n in p.relative_to(root).as_posix() for n in dates):
            reasons.setdefault(p, DATED)

    want = {f"{name} @ {format_conc(conc)}" for name, conc in matrix_odors}
    candidates = []
    for p, reason in itertools.islice(reasons.items(), MAX_YAMLS_PARSED):
        text = _read(p)
        try:
            odors, _ = yamlcheck.parse_odor_yaml(text or "")
        except yamlcheck.YamlError:
            continue
        covers = {m["odor"] for m in yamlcheck.check_subset(matrix_odors, odors)["matched"]}
        if covers:
            candidates.append({"path": p.relative_to(root).as_posix(), "reason": reason, "n_odors": len(odors),
                               "covers": covers})

    # fewest files that together describe every odor, preferring ones named in the notes
    selected, covered, pool = [], set(), list(candidates)
    while pool and covered != want:
        best = max(pool, key=lambda c: (len(c["covers"] - covered), c["reason"] == NAMED))
        if not best["covers"] - covered:
            break
        selected.append(best)
        covered |= best["covers"]
        pool.remove(best)
    complete = bool(want) and covered == want
    ranked = selected + sorted(pool, key=lambda c: -len(c["covers"]))
    return {
        "n_matrix_odors": len(want),
        "candidates": [{**{k: v for k, v in c.items() if k != "covers"}, "n_covered": len(c["covers"])} for c in ranked],
        "selected": [c["path"] for c in selected] if complete else [],
        "missing": sorted(want - set().union(*(c["covers"] for c in candidates))),
    }


def lookup(root: Path, dirs: dict[str, list[str]], flies: list[dict], matrix_odors: list[tuple[str, float]]) -> dict:
    """Find a dataset's flies in a user's folders on tensor, and odor YAMLs describing its odors.

    Args:
        root: top of tensor on the server.
        dirs: the user's folders (`USER_DIR_KINDS` -> paths relative to `root`).
        flies: the dataset's flies (dicts with date and fly_num), as in `ParsedData.summary`.
        matrix_odors: (name, log10_conc) of every odor in the dataset.

    Returns:
        dict with ``flies`` (per fly: its raw and motion-corrected folders and the
        Experiment.xml of its recordings, only ever in the folders of a fly of the dataset,
        by its date and number: see `fly_dirs`), ``raw_data_path`` and ``mocorr_data_path``
        (the folder holding every fly found, or None), ``experiment_xmls`` (of every
        fly, by date and number, each fly's oldest first: the last is the most recent)
        and ``odor_yamls`` (``candidates``, ``selected``: the fewest of them that together
        describe every odor, if they do, and ``missing``: odors none of them describe).
        All paths are relative to `root`.
    """
    rel = lambda p: p.relative_to(root).as_posix()
    folders = {k: [root / p for p in dirs.get(k) or []] for k in USER_DIR_KINDS}
    found = []
    for f in flies:
        raw = [d for base in folders["raw_dirs"] for d in fly_dirs(base, f["date"], f["fly_num"])]
        mocorr = [d for base in folders["mocorr_dirs"] for d in fly_dirs(base, f["date"], f["fly_num"])]
        found.append({
            "date": f["date"], "fly_num": f["fly_num"], "raw": [rel(d) for d in raw], "mocorr": [rel(d) for d in mocorr],
            "recordings": [rel(x) for d in raw for x in recordings(d)],
        })
    xmls = [x for f in sorted(found, key=lambda f: (f["date"], f["fly_num"])) for x in f["recordings"]]
    return {
        "flies": found,
        "raw_data_path": _common([d for f in found for d in f["raw"]]),
        "mocorr_data_path": _common([d for f in found for d in f["mocorr"]]),
        "experiment_xmls": xmls,
        "odor_yamls": _odor_yamls(root, folders["odor_dirs"], [root / x for x in xmls], flies, matrix_odors),
    }
