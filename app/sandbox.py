"""Which folders on the server the app may read and write, and holding it to them.

Two lists of absolute paths, set in the Admin tab and saved in `config.ADMIN_FILE`:
folders the server may read (by default, tensor) and folders it may also write (none by
default; for backups, later). The app's own data folder is always readable and writable.
Both lists are fixed when the server starts (`start`), and changes wait for a restart:

- the app checks what it reads outside its own folders (users' folders on tensor) against
  them (`can_read`; `can_write` for writes), so that a refusal can be explained;
- where the kernel supports Landlock, the process also has the kernel deny it everything
  else (`restrict`): reading outside these folders and its code, Python and system
  folders, and writing outside its data folder and the folders it may write. A process
  cannot lift this; only a new one starts without it.

What the server can reach at all is still decided outside the app, by the Docker
container's mounts or the systemd unit (DEPLOY.md), which this can only narrow.
"""
from __future__ import annotations

import ctypes
import datetime as dt
import json
import logging
import os
import posixpath
import stat
import struct
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from . import config, glomeruli

log = logging.getLogger("uvicorn.error")

KINDS = ("read_dirs", "write_dirs")

# Read-only for the server under Landlock, as well as the folders set in the Admin tab:
# Python, the app's code and configuration, and system folders (libraries, certificates,
# DNS configuration, /proc).
SYSTEM_DIRS = ("/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/etc", "/proc", "/sys", "/dev",
               "/run/systemd/resolve")

# folders in effect since the server started (realpaths), or None if it was not started
# with `start` (as in tests): nothing is refused then
ACTIVE: Optional[dict[str, list[str]]] = None
STATUS: dict = {"settings": None, "started_at": None, "kernel": None}


class SandboxError(ValueError):
    pass


# --------------------------------------------------------------------- settings


def defaults() -> dict:
    return {"read_dirs": [str(config.TENSOR_ROOT)], "write_dirs": [], "updated_at": None}


def load() -> dict:
    """The saved settings (`KINDS` and ``updated_at``), or the defaults if there are none."""
    try:
        saved = json.loads(config.ADMIN_FILE.read_text())
    except FileNotFoundError:
        return defaults()
    except (OSError, ValueError) as e:
        log.warning("could not read %s (%s): using the default folders", config.ADMIN_FILE, e)
        return defaults()
    return {**defaults(), **{k: saved[k] for k in (*KINDS, "updated_at") if k in saved}}


def save(dirs: dict[str, list[str]]) -> dict:
    """Save normalized folders (as from `check_folders`), taking effect at the next restart."""
    settings = {**{k: dirs[k] for k in KINDS}, "updated_at": dt.datetime.now().isoformat(timespec="seconds")}
    tmp = config.ADMIN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=1))
    tmp.replace(config.ADMIN_FILE)
    log.info("folders the server may use changed (from the next restart): read %s, write %s",
             settings["read_dirs"], settings["write_dirs"])
    return settings


def normalize(path: str) -> str:
    """An absolute path on the server, cleaned up (``/a/b``).

    Raises:
        SandboxError: for relative paths, ``..`` components, or the whole filesystem (``/``).
    """
    p = path.strip()
    if not p.startswith("/"):
        raise SandboxError("give the absolute path on the server, starting with /")
    if ".." in p.split("/"):
        raise SandboxError("'..' is not allowed: give the folder itself")
    p = posixpath.normpath("/" + p.lstrip("/"))  # normpath keeps a leading //
    if p == "/":
        raise SandboxError("not the whole filesystem: give the folders the server needs")
    return p


def check_folder(path: str, write: bool = False) -> dict:
    """Check a folder entered in the Admin tab: ``{'input', 'path' (normalized, or None),
    'ok', 'error', 'warning'}``. Folders that are not there, or that the server cannot read
    (or write, with `write`) yet, are accepted with a warning: tensor may just be unmounted."""
    out = {"input": path, "path": None, "ok": False, "error": None, "warning": None}
    try:
        out["path"] = p = normalize(path)
    except SandboxError as e:
        out["error"] = str(e)
        return out
    if not os.path.exists(p):
        out["warning"] = "not on the server (now, at least; with Docker, only folders mounted into the container are)"
    elif not os.path.isdir(p):
        out["error"] = "this is a file, not a folder"
        return out
    elif not os.access(p, os.R_OK | os.X_OK):
        out["warning"] = "the server's user is not allowed to read it"
    elif write and not os.access(p, os.W_OK):
        out["warning"] = "the server cannot write here (mounted read-only, or its user is not allowed to)"
    out["ok"] = True
    return out


def check_folders(body: dict) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """Checks of the folders in `body` (lists by `KINDS`; empty ones skipped), and the
    folders normalized and without duplicates."""
    checks = {k: [check_folder(str(p), write=k == "write_dirs") for p in body.get(k) or [] if str(p).strip()]
              for k in KINDS}
    dirs = {k: list(dict.fromkeys(c["path"] for c in cs if c["ok"])) for k, cs in checks.items()}
    return checks, dirs


# ------------------------------------------------------------------ the policy


def _real(path) -> str:
    return os.path.realpath(path)


def _within(path, dirs: Iterable[str]) -> bool:
    real = _real(path)
    return any(real == d or real.startswith(d.rstrip("/") + "/") for d in dirs)


def activate(settings: dict) -> None:
    """Hold the app's own checks to `settings` (read: `KINDS` and the data folder; write: write_dirs and the data folder)."""
    global ACTIVE
    write = [_real(p) for p in [*settings["write_dirs"], config.DATA_DIR]]
    ACTIVE = {"read": [_real(p) for p in settings["read_dirs"]] + write, "write": write}


def folders() -> list[str]:
    """The folders set in the Admin tab (read, then write) in effect, or as saved if the
    server was not started with `start`."""
    settings = STATUS["settings"] or load()
    return [*settings["read_dirs"], *settings["write_dirs"]]


def can_read(path) -> bool:
    """Whether the server may read `path` (following links), by the folders in effect."""
    return ACTIVE is None or _within(path, ACTIVE["read"])


def can_write(path) -> bool:
    """Whether the server may write `path` (following links), by the folders in effect."""
    return ACTIVE is None or _within(path, ACTIVE["write"])


def readable_within(root: Path) -> list[Path]:
    """The folders the server may read in `root`: `root` itself if it may read all of it."""
    if can_read(root):
        return [root]
    real_root = _real(root)
    return [Path(d) for d in ACTIVE["read"] if d.startswith(real_root.rstrip("/") + "/")]


# -------------------------------------------------------------------- Landlock
# linux/landlock.h: syscall numbers (the same on every architecture but alpha), access rights

_CREATE_RULESET, _ADD_RULE, _RESTRICT_SELF = 444, 445, 446
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_EXECUTE, _WRITE_FILE, _READ_FILE, _READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_CHANGE_DIR = sum(1 << i for i in range(4, 13))  # remove files and folders; make files, folders, links, ...
_REFER, _TRUNCATE = 1 << 13, 1 << 14  # ABI 2 (moving files between folders), ABI 3
_FILE_ONLY = _EXECUTE | _WRITE_FILE | _READ_FILE | _TRUNCATE  # all a rule on a file (not a folder) may grant
_READ = _READ_FILE | _READ_DIR
_WRITE = _READ | _WRITE_FILE | _CHANGE_DIR | _REFER | _TRUNCATE


def _struct(fmt: str, *values) -> ctypes.Array:
    """A C struct (packed, native byte order) for the kernel: struct landlock_ruleset_attr is
    ``=Q`` (handled_access_fs, as in ABI 1), struct landlock_path_beneath_attr ``=Qi``."""
    data = struct.pack(fmt, *values)
    return ctypes.create_string_buffer(data, len(data))


def _syscall(nr: int, *args) -> tuple[int, int]:
    """(result, errno) of a raw syscall; ints are passed as C longs, as syscall(2) reads them."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    res = libc.syscall(ctypes.c_long(nr), *(ctypes.c_long(a) if isinstance(a, int) else a for a in args))
    return res, ctypes.get_errno()


def landlock_abi() -> int:
    """The Landlock ABI version the kernel offers (0 if none: kernel before 5.13, turned off, or blocked by seccomp)."""
    if sys.platform != "linux":
        return 0
    res, _ = _syscall(_CREATE_RULESET, None, 0, _CREATE_RULESET_VERSION)
    return max(res, 0)


def _system_paths() -> list[str]:
    # not sys.path[0] (the script's folder, or the working directory): the code run is in ROOT or Python's prefix
    paths = [*SYSTEM_DIRS, sys.prefix, sys.base_prefix, sys.exec_prefix, *sys.path[1:], config.ROOT,
             config.REFERENCE_DIR, glomeruli.config_path()]
    return list(dict.fromkeys(str(p) for p in paths if p and os.path.isabs(p)))


def restrict(read: list[str], write: list[str]) -> dict:
    """Have the kernel deny this thread, and the threads and processes it starts from now
    on, reading anything outside `read` and `_system_paths`, and writing anything outside
    `write` (and /dev/null). Paths that do not exist are left out.

    Returns:
        dict with ``enforced``, ``abi`` and ``reason`` (why it is not enforced, or None).
    """
    abi = landlock_abi()
    if not abi:
        reason = ("not available here: needs Linux 5.13 or later with Landlock turned on (and, in Docker, "
                  "a seccomp profile that allows it)")
        return {"enforced": False, "abi": 0, "reason": reason}
    handled = _READ | _EXECUTE | _WRITE_FILE | _CHANGE_DIR | (_REFER if abi >= 2 else 0) | (_TRUNCATE if abi >= 3 else 0)
    attr = _struct("=Q", handled)
    ruleset, err = _syscall(_CREATE_RULESET, attr, len(attr), 0)
    if ruleset < 0:
        return {"enforced": False, "abi": abi, "reason": f"the kernel refused a ruleset ({os.strerror(err)})"}
    rules = ([(p, _READ | _EXECUTE) for p in _system_paths()] + [(p, _READ) for p in read]
             + [(p, _WRITE) for p in write] + [("/dev/null", _READ_FILE | _WRITE_FILE | _TRUNCATE)])
    try:
        for path, access in rules:
            try:
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue
            try:
                if not stat.S_ISDIR(os.fstat(fd).st_mode):
                    access &= _FILE_ONLY
                rule = _struct("=Qi", access & handled, fd)
                res, err = _syscall(_ADD_RULE, ruleset, _RULE_PATH_BENEATH, rule, 0)
            finally:
                os.close(fd)
            if res < 0:
                return {"enforced": False, "abi": abi, "reason": f"the kernel refused a rule for {path} ({os.strerror(err)})"}
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)):
            return {"enforced": False, "abi": abi, "reason": f"could not set no_new_privs ({os.strerror(ctypes.get_errno())})"}
        res, err = _syscall(_RESTRICT_SELF, ruleset, 0)
        if res < 0:
            return {"enforced": False, "abi": abi, "reason": f"the kernel refused to apply the rules ({os.strerror(err)})"}
    finally:
        os.close(ruleset)
    return {"enforced": True, "abi": abi, "reason": None}


def start() -> dict:
    """Fix the folders the server may use until it stops. To be called once as it starts,
    before it starts any thread (Landlock holds the calling thread and those it starts;
    numpy's and pyarrow's worker threads, started on import, run no Python and open no files).

    Returns:
        `STATUS`: the settings in effect, when, and how the kernel holds the server to them.
    """
    config.ensure_dirs()
    tempfile.tempdir = str(config.TMP_DIR)
    settings = load()
    activate(settings)
    if config.LANDLOCK:
        kernel = restrict(ACTIVE["read"], ACTIVE["write"])
    else:
        kernel = {"enforced": False, "abi": landlock_abi(), "reason": "turned off on the server (OLFDB_LANDLOCK=0)"}
    if kernel["enforced"]:
        log.info("Landlock (ABI %d): reading only %s, writing only %s", kernel["abi"], ACTIVE["read"], ACTIVE["write"])
    else:
        log.warning("Landlock not in use, %s: only the app's own checks hold it to its folders", kernel["reason"])
    STATUS.update(settings=settings, started_at=dt.datetime.now().isoformat(timespec="milliseconds"), kernel=kernel)
    return STATUS
