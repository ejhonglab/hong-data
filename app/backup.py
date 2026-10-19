"""Backups of the server's data (Admin tab), made into a folder it may write (one of the
write_dirs set there).

Each backup is a folder ``olfdb-backup-<date>_<time>`` laid out like the data folder: a
consistent copy of the database (SQLite's backup API), and the files the site keeps
(`data_files`: data files as uploaded, odor YAMLs, Experiment.xml, READMEs and
metadata.json of every dataset, reference data as imported, datasets deleted in the Admin
tab, odor structure images, the folders set in the Admin tab). Not what only passes
through the data folder (uploads under way, downloads being sent, temporary files), nor
raw and motion-corrected data, which stay on tensor. A file unchanged since the last
backup in the same folder (by size and time) is a hard link to its copy there, so each
backup is whole on its own but takes the space of what changed only.

A backup is made in a ``.partial`` folder, renamed once it is done, with `MANIFEST`: what
it holds, by size and time, and the fingerprint of its database (`db.content_fingerprint`),
which tell whether it is current (`status`).
"""
from __future__ import annotations

import datetime as dt
import errno
import json
import logging
import os
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from . import config, sandbox
from .db import content_fingerprint

log = logging.getLogger("uvicorn.error")

PREFIX = "olfdb-backup-"
PARTIAL = ".partial"
MANIFEST = "backup.json"
DB_NAME = "olfdb.sqlite3"

# the backup being made, if any (one at a time)
JOB: dict = {"running": False, "folder": None, "started_at": None, "done": 0, "total": 0, "error": None, "name": None}
_lock = threading.Lock()


def _kept() -> list[Path]:
    """The folders and files of the data folder a backup copies, besides the database."""
    return [config.DATA_DIR / p.name for p in (config.DATASETS_DIR, config.REFERENCE_CACHE_DIR, config.DELETED_DIR,
                                               config.STRUCTURES_DIR, config.ADMIN_FILE)]


def data_files() -> dict[str, list[int]]:
    """The files a backup copies, but the database: path relative to the data folder ->
    [size, modification time in ns]."""
    paths = []
    for p in _kept():
        if p.is_file():
            paths.append(p)
        for d, _, names in os.walk(p):
            paths += [Path(d) / n for n in names]
    out = {}
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            continue
        out[p.relative_to(config.DATA_DIR).as_posix()] = [st.st_size, st.st_mtime_ns]
    return dict(sorted(out.items()))


def write_dirs() -> list[str]:
    """The folders the server may write (Admin tab), as in effect since it started."""
    return list((sandbox.STATUS["settings"] or sandbox.load())["write_dirs"])


def _manifest(path: Path) -> Optional[dict]:
    try:
        return json.loads((path / MANIFEST).read_text())
    except (OSError, ValueError):
        return None


def _summary(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "files"}


def backups(folder: Path) -> list[dict]:
    """The backups in `folder`, newest first: each with its ``name``, and ``complete``
    (False if it was interrupted, or is being made); the rest of its `MANIFEST` but its
    file list if it is complete."""
    try:
        entries = [e for e in os.scandir(folder) if e.name.startswith(PREFIX) and e.is_dir()]
    except OSError:
        return []
    out = []
    for e in sorted(entries, key=lambda e: e.name, reverse=True):
        m = None if e.name.endswith(PARTIAL) else _manifest(Path(e.path))
        out.append({"name": e.name, "complete": False} if m is None else {"name": e.name, "complete": True, **_summary(m)})
    return out


def _newest(folder: Path) -> Optional[tuple[Path, dict]]:
    """The last complete backup in `folder`, and its manifest."""
    for b in backups(folder):
        if b["complete"]:
            m = _manifest(folder / b["name"])
            if m is not None:
                return folder / b["name"], m
    return None


def make(db, folder: Path, progress=None) -> dict:
    """Back up the data into `folder` (see the module's docstring), removing backups of it
    left unfinished there before.

    Args:
        db: the `Database`.
        folder: a folder the server may write.
        progress: called with (files done, files in all) as files are copied.

    Returns:
        the new backup's `MANIFEST`, with its ``name``.
    """
    for e in os.scandir(folder):
        if e.name.startswith(PREFIX) and e.name.endswith(PARTIAL) and e.is_dir():
            shutil.rmtree(e.path, ignore_errors=True)
    created = dt.datetime.now()
    name = PREFIX + created.strftime("%Y-%m-%d_%H%M%S")
    k = 1
    while (folder / name).exists():  # two in the same second
        k += 1
        name = f"{PREFIX}{created:%Y-%m-%d_%H%M%S}_{k}"
    tmp = folder / (name + PARTIAL)
    tmp.mkdir()
    prev = _newest(folder)

    # the database, as it is at one moment
    src, dst = sqlite3.connect(db.path, timeout=60), sqlite3.connect(tmp / DB_NAME)
    try:
        src.backup(dst)
        fingerprint = content_fingerprint(dst)
        n_datasets = dst.execute("SELECT count(*) FROM datasets").fetchone()[0]
    finally:
        dst.close()
        src.close()
    db_bytes = os.path.getsize(tmp / DB_NAME)

    files = data_files()
    kept, linked, copied, copied_bytes = {}, 0, 0, 0
    can_link = prev is not None
    for i, (rel, stat) in enumerate(files.items()):
        source, dest = config.DATA_DIR / rel, tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        done = False
        if can_link and prev[1]["files"].get(rel) == stat:
            try:
                os.link(prev[0] / rel, dest)
                linked += 1
                done = True
            except FileNotFoundError:  # gone from that backup: copied
                pass
            except OSError as e:  # links not possible there (or too many to one file): copies from now on
                log.warning("backup into %s: copying instead of linking to the backup before (%s)", folder,
                            errno.errorcode.get(e.errno, e))
                can_link = False
        if not done:
            try:
                shutil.copy2(source, dest)
            except FileNotFoundError:  # deleted since it was listed
                continue
            copied += 1
            copied_bytes += stat[0]
        kept[rel] = stat
        if progress:
            progress(i + 1, len(files))

    manifest = {
        "created_at": created.isoformat(timespec="seconds"),
        "database": {"fingerprint": fingerprint, "bytes": db_bytes, "datasets": n_datasets},
        "n_files": len(kept) + 1,
        "bytes": db_bytes + sum(s[0] for s in kept.values()),
        "copied_bytes": db_bytes + copied_bytes,  # the space it takes (not linked to the one before)
        "linked": linked,
        "linked_to": prev[0].name if prev and linked else None,
        "files": kept,
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=1))
    os.rename(tmp, folder / name)
    log.info("backed up into %s/%s: %d files, %d copied, %d linked", folder, name, len(kept) + 1, copied + 1, linked)
    return {"name": name, **manifest}


def start(db, folder: str) -> Optional[str]:
    """Start making a backup into `folder` (one of `write_dirs`) in the background; why it
    cannot be, if it cannot."""
    with _lock:
        if JOB["running"]:
            return "a backup is being made already"
        if folder not in write_dirs():
            return f"{folder} is not one of the folders the server may write (set above, in effect once it restarts)"
        path = Path(folder)
        if not path.is_dir():
            return f"{folder} is not on the server (or not mounted into the container)"
        if not sandbox.can_write(path) or not os.access(path, os.W_OK):
            return f"the server may not write {folder}"
        JOB.update(running=True, folder=folder, started_at=dt.datetime.now().isoformat(timespec="seconds"), done=0, total=0,
                   error=None, name=None)

    def run() -> None:
        try:
            JOB["name"] = make(db, path, progress=lambda done, total: JOB.update(done=done, total=total))["name"]
        except Exception as e:
            log.exception("backup into %s failed", folder)
            JOB["error"] = str(e) or type(e).__name__
        finally:
            JOB["running"] = False

    threading.Thread(target=run, name="backup", daemon=True).start()
    return None


def status(db) -> dict:
    """The backups, and whether the last one is current.

    Returns:
        dict with ``folders`` (each folder the server may write: its ``path``, ``ok`` and
        ``error``, and its ``backups``), ``pending`` (folders saved in the Admin tab, in
        effect once the server restarts), ``job`` (the backup being made, or the last one
        made since the server started), ``last`` (the newest complete backup, with its
        ``folder``) and ``current`` (None without one; else ``ok``, and what changed since:
        ``database``, and the numbers of files ``added``, ``changed`` and ``removed``).
    """
    folders = []
    for d in write_dirs():
        p = Path(d)
        error = (None if p.is_dir() and sandbox.can_write(p) and os.access(p, os.W_OK)
                 else "not on the server (or not mounted into the container)" if not p.is_dir()
                 else "the server may not write it")
        folders.append({"path": d, "ok": error is None, "error": error, "backups": backups(p) if p.is_dir() else []})
    saved = sandbox.load()["write_dirs"]
    last = max(({**b, "folder": f["path"]} for f in folders for b in f["backups"] if b["complete"]),
               key=lambda b: b["created_at"], default=None)
    current = None
    if last:
        m = _manifest(Path(last["folder"]) / last["name"]) or {"files": {}, "database": {}}
        now = data_files()
        before = m["files"]
        database = db.fingerprint() != m["database"].get("fingerprint")
        added = len(now.keys() - before.keys())
        removed = len(before.keys() - now.keys())
        changed = sum(1 for k in now.keys() & before.keys() if now[k] != before[k])
        current = {"ok": not (database or added or removed or changed), "database": database, "added": added,
                   "changed": changed, "removed": removed}
    return {"folders": folders, "pending": [d for d in saved if d not in write_dirs()], "job": dict(JOB), "last": last,
            "current": current}
