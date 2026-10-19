"""Which files in its own folders the server has ever read and written, for the Admin tab.

An audit hook (`sys.addaudithook`) sees each file the server's Python code opens (Python's
``open`` and `os.open`), moves (`os.rename`, `os.replace`) or opens as a database
(`sqlite3.connect`), and notes those in the folder it runs from (`config.ROOT`) or its
data folder, but for its Python environment and bytecode. They are saved in the database
(table ``file_log``) every `SAVE_S` seconds and when it stops. A file opened by a C library
alone (pyarrow reading a parquet file) is not seen, but each of these in its folders is one
it wrote itself first (an upload). The files already in the data folder when the record
began are noted as written then (``first_at`` None).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import config

SAVE_S = 30
STARTED_SETTING = "file_log_started"

_READ, _WROTE = 1, 2
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_EVENTS = frozenset({"open", "os.rename", "sqlite3.connect"})

# folders noted (with a trailing /), and folders in them not: none until `start`
_scope: tuple[str, ...] = ()
_skip: tuple[str, ...] = ()
_known: dict[str, int] = {}  # path: _READ | _WROTE, as saved
_pending: dict[str, int] = {}  # what is not saved yet
_lock = threading.Lock()
_installed = False

MOUNTINFO = Path("/proc/self/mountinfo")


def _in_docker() -> bool:
    return os.path.exists("/.dockerenv")


def _note(path, bits: int) -> None:
    if isinstance(path, int):  # a file descriptor
        return
    p = os.path.abspath(os.fsdecode(path))
    if not p.startswith(_scope) or p.startswith(_skip) or "/__pycache__/" in p:
        return
    if bits == _READ and not os.path.isfile(p):  # a folder, or not there (so not opened)
        return
    if bits & _WROTE and os.path.isdir(p):  # a file with no name, in this folder (O_TMPFILE)
        return
    have = _known.get(p, 0) | _pending.get(p, 0)
    if have | bits != have:
        with _lock:
            _pending[p] = _pending.get(p, 0) | bits


def _hook(event: str, args: tuple) -> None:
    if event not in _EVENTS or not _scope:
        return
    try:
        if event == "open":
            path, mode, flags = args
            if mode is None:  # os.open, whose path may be relative to a folder (dir_fd) other than the working one
                if isinstance(path, int) or not os.path.isabs(os.fsdecode(path)):
                    return
                bits = _WROTE if flags & _WRITE_FLAGS else _READ
            else:
                bits = _WROTE if any(c in mode for c in "wax+") else _READ
            _note(path, bits)
        elif event == "os.rename":
            src, dst, src_dir_fd, dst_dir_fd = args
            if src_dir_fd not in (None, -1) or dst_dir_fd not in (None, -1):
                return
            src, dst = os.path.abspath(os.fsdecode(src)), os.path.abspath(os.fsdecode(dst))
            if os.path.isdir(src):  # each file in it, where it is going
                for folder, _, names in os.walk(src):
                    for name in names:
                        _note(os.path.join(dst, os.path.relpath(os.path.join(folder, name), src)), _WROTE)
            else:
                _note(dst, _WROTE)
        else:  # sqlite3.connect
            path = os.fsdecode(args[0])
            if os.path.isabs(path):  # not ":memory:"
                for suffix in ("", "-wal", "-shm"):
                    _note(path + suffix, _READ | _WROTE)
    except Exception:  # never in the way of what is being done
        pass


def _files_in(folder: Path) -> list[str]:
    return [os.path.join(d, n) for d, _, names in os.walk(folder) for n in names]


def start(db) -> None:
    """Start noting the files the server reads and writes in its own folders, as it starts
    (an audit hook cannot be taken off: it is there until the process ends). The first
    time, note the files in the data folder as written."""
    global _scope, _skip, _installed
    _known.update({r["path"]: r["read"] * _READ | r["wrote"] * _WROTE for r in db.file_log()})
    if db.get_setting(STARTED_SETTING) is None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        db.record_files({p: _WROTE for p in _files_in(config.DATA_DIR) if p not in _known}, when=None)
        db.set_setting(STARTED_SETTING, now, when=now)
        _known.update({r["path"]: r["read"] * _READ | r["wrote"] * _WROTE for r in db.file_log()})
    _scope = tuple(dict.fromkeys(f"{str(d).rstrip('/')}/" for d in (config.ROOT, config.DATA_DIR)))
    _skip = tuple(dict.fromkeys(f"{p.rstrip('/')}/" for p in (sys.prefix, sys.exec_prefix)))
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True


def flush(db) -> None:
    """Save the files noted since this was last done."""
    global _pending
    with _lock:
        pending, _pending = _pending, {}
    if not pending:
        return
    try:
        db.record_files(pending, when=dt.datetime.now().isoformat(timespec="seconds"))
    except Exception:
        with _lock:
            for p, bits in pending.items():
                _pending[p] = _pending.get(p, 0) | bits
        raise
    for p, bits in pending.items():
        _known[p] = _known.get(p, 0) | bits


# --------------------------------------------------------- paths on the server

def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _same_tail(a: str, b: str) -> bool:
    """Whether one of two absolute paths ends with the other: the same folder, as seen from
    different roots (a file system's own, as in /proc/self/mountinfo, and the server's)."""
    pa, pb = Path(a).parts[1:], Path(b).parts[1:]
    n = min(len(pa), len(pb))
    return pa[len(pa) - n:] == pb[len(pb) - n:]


def _mount_root(mount_point: str) -> Optional[str]:
    """The folder mounted at `mount_point`, as a path in its own file system; None if nothing is."""
    try:
        lines = MOUNTINFO.read_text().splitlines()
    except OSError:
        return None
    unescape = lambda s: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)
    roots = [unescape(f[3]) for f in map(str.split, lines) if len(f) > 4 and unescape(f[4]) == mount_point]
    return roots[-1] if roots else None


def server_paths() -> tuple[Callable[[str], str], Optional[str]]:
    """How to give a path as the server sees it as a path on the server itself, and why
    this cannot be done when it cannot (else None).

    Outside Docker, they are the same. In the Docker container, its code is in
    `config.ROOT` and its data folder mounted at `config.DATA_DIR`: these are the folder of
    docker-compose.yml on the server, which Docker Compose gives as the folder it was run
    from (OLFDB_HOST_DIR, set in docker-compose.yml), and the folder in it mounted there.
    That this is so is checked by the folder mounted, as in /proc/self/mountinfo (where it
    is a path in its own file system, which may not start at the server's /).
    """
    same = lambda p: p
    if not _in_docker():
        return same, None
    host = os.environ.get("OLFDB_HOST_DIR", "").strip().rstrip("/")
    if not os.path.isabs(host):
        return same, "docker-compose.yml does not give the folder it is in on the server (OLFDB_HOST_DIR)"
    data = _mount_root(str(config.DATA_DIR))
    if data is None:
        return same, f"its data folder ({config.DATA_DIR}) is not a folder of the server mounted in the container"
    if data != "/" and not _same_tail(os.path.dirname(data), host):
        return same, (f"OLFDB_HOST_DIR ({host}), the folder Docker Compose was run from, does not hold its data folder "
                      "(run docker compose from the folder of docker-compose.yml)")
    pairs = [(str(config.DATA_DIR), os.path.join(host, os.path.basename(data) or "data")), (str(config.ROOT), host)]

    def to_server(p: str) -> str:
        for inside, outside in pairs:
            if _inside(p, inside):
                return outside + p[len(inside):]
        return p

    return to_server, None


# --------------------------------------------------------------------- listing

def _repository_files() -> Optional[set[str]]:
    """The repository's files (absolute paths), as git lists them; None if it cannot (as in
    the Docker image, which has neither git nor the repository's history)."""
    try:
        env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}  # none it may not read
        out = subprocess.run(["git", "-C", str(config.ROOT), "ls-files", "-z"], capture_output=True, timeout=10,
                             check=True, env=env).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {os.path.join(config.ROOT, os.fsdecode(p)) for p in out.split(b"\0") if p}


def listing(db, exclude: Iterable[str] = ()) -> dict:
    """The files the server has read or written in its own folders (as noted since `start`).

    Not those of the repository (as git lists them; when it cannot, all in the folder it
    runs from but those in the data folder), nor those in `exclude` (folders and files it
    may read, listed in the Admin tab already), unless that holds the data folder or the
    folder it runs from.

    Returns:
        dict with ``root`` (the folder they are in, as a path on the server: see
        `server_paths`), ``files`` (each with its ``path`` relative to that, ``read``,
        ``wrote``, ``size`` and ``mtime`` (when it was last changed, in seconds since
        1970): both None if it has been deleted since, and ``first_at``: when it was
        first read or written, None if it was in the data folder when this record
        began), ``since`` (when that was) and ``note`` (why paths are as in the Docker
        container, if they are).
    """
    flush(db)
    root, data = str(config.ROOT), str(config.DATA_DIR)
    repo = _repository_files()
    exclude = [d for d in exclude if not (_inside(data, d) or _inside(root, d))]

    def shown(p: str) -> bool:
        if _inside(p, data):
            return not any(_inside(p, d) for d in exclude)
        if _inside(p, root) if repo is None else p in repo:
            return False
        return not any(_inside(p, d) for d in exclude)

    to_server, note = server_paths()
    top = os.path.commonpath([to_server(root), to_server(data)])
    files = []
    for r in db.file_log():
        p = r["path"]
        if not shown(p):
            continue
        try:
            st = os.stat(p) if os.path.isfile(p) else None
        except OSError:
            st = None
        files.append({"path": os.path.relpath(to_server(p), top), "read": bool(r["read"]), "wrote": bool(r["wrote"]),
                      "size": st and st.st_size, "mtime": st and st.st_mtime, "first_at": r["first_at"]})
    files.sort(key=lambda f: f["path"])
    started = db.get_setting(STARTED_SETTING)
    return {"root": top, "files": files, "since": started["value"] if started else None, "note": note}
