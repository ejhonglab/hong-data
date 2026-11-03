"""MetaCyc imported in the Admin tab: its SmartTables uploaded (the files of the folder
metacyc_build.py reads, or an archive of it), kept in the data folder as that lays them
out, built into a bundle there in the background, and the bundle's two reference datasets
added (as ``python -m app.metacyc_build <raw> <bundle> --register <server>`` adds them).

Each import is a folder ``metacyc/<id>/`` of the data folder (`config.METACYC_DIR`):
``raw/``, the files the build reads, laid out as in ``BioCyc/raw`` (the build can be run on
it again); the bundle, ``metacyc-<version>.sqlite``; and ``import.json``: when it was
uploaded, the files and the reference datasets added. Files uploaded that the build does not
read are not kept. Once the last of its reference datasets is deleted, the folder is moved
to ``deleted/metacyc/<id>`` (`release`).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import shutil
import tarfile
import threading
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import IO, Callable, Optional, Union

from . import config, metacyc_build, reference

log = logging.getLogger("uvicorn.error")

ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
RECORD = "import.json"
# where the build reads each file it reads, by the file's name (no two are named alike)
_PLACES = {PurePosixPath(rel).name: rel for rels in metacyc_build.FILES.values() for rel in rels}

# the import under way, or the last one (one at a time)
JOB: dict = {"running": False, "id": None, "stage": None, "version": None, "started_at": None, "finished_at": None,
             "log": [], "error": None, "added": []}
_lock = threading.Lock()


class UploadError(ValueError):
    """An upload that cannot be imported: its `errors`, each saying what was expected."""

    def __init__(self, errors: Union[list[str], str]):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("\n".join(self.errors))


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def expected() -> list[dict]:
    """The files an upload is to have: for each role, its ``files`` (where in the folder the
    build reads it; the first there is read), what it is (``about``), and whether it is
    ``required``."""
    return [{"role": r, "files": fs, "about": metacyc_build.ABOUT[r], "required": r in metacyc_build.REQUIRED}
            for r, fs in metacyc_build.FILES.items()]


def is_archive(name: str) -> bool:
    return (name or "").lower().endswith(ARCHIVES)


def pick(names: list[str]) -> dict[int, str]:
    """The files of an upload the build reads: their positions in `names` (their paths in the
    folder or archive uploaded, or their names alone) -> where it reads them, in ``raw/``. Of
    several named alike, the one at that path (in any folder), else the first."""
    places: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        rel = _PLACES.get(PurePosixPath(n.replace("\\", "/")).name)
        if rel:
            places.setdefault(rel, []).append(i)
    out = {}
    for rel, idx in places.items():
        at = [i for i in idx if ("/" + names[i].replace("\\", "/")).endswith("/" + rel)]
        out[(at or idx)[0]] = rel
    return out


def claim() -> Path:
    """Start an import (one at a time): its folder, made (and those of imports the server
    stopped during, removed).

    Raises:
        UploadError: if one is under way.
    """
    with _lock:
        if JOB["running"]:
            raise UploadError(f"MetaCyc is being imported already (since {JOB['started_at']}): wait for it to finish")
        config.METACYC_DIR.mkdir(parents=True, exist_ok=True)
        for d in config.METACYC_DIR.iterdir():
            if d.is_dir() and not (d / RECORD).exists():
                shutil.rmtree(d, ignore_errors=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        d, n = config.METACYC_DIR / stamp, 2
        while d.exists():
            d, n = config.METACYC_DIR / f"{stamp}-{n}", n + 1
        (d / "raw").mkdir(parents=True)
        JOB.update(running=True, id=d.name, stage="uploading", version=None, started_at=_now(), finished_at=None, log=[],
                   error=None, added=[])
        return d


def abandon(d: Path) -> None:
    """An import refused before it started (its upload not as expected): its folder removed."""
    shutil.rmtree(d, ignore_errors=True)
    JOB.update(running=False, id=None, stage=None)


def _copy(src: IO[bytes], dest: Path, max_bytes: int, what: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with open(dest, "wb") as f:
        while chunk := src.read(1 << 20):
            size += len(chunk)
            if size > max_bytes:
                raise UploadError(f"{what}: larger than the {max_bytes / 2 ** 20:g} MB limit")
            f.write(chunk)


def extract(archive: Path, name: str, raw: Path, max_bytes: int) -> list[str]:
    """Take the files the build reads (see `pick`) out of an archive uploaded (zip or tar;
    `name`: as uploaded) into `raw`, where it reads them: those places."""
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                members = [i for i in z.infolist() if not i.is_dir()]
                chosen = pick([i.filename for i in members])
                for k, rel in chosen.items():
                    with z.open(members[k]) as src:
                        _copy(src, raw / rel, max_bytes, f"{name}: {members[k].filename}")
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                members = [m for m in t.getmembers() if m.isfile()]
                chosen = pick([m.name for m in members])
                for k, rel in chosen.items():
                    with t.extractfile(members[k]) as src:
                        _copy(src, raw / rel, max_bytes, f"{name}: {members[k].name}")
        else:
            raise UploadError(f"{name} is not a zip or tar archive")
    except (zipfile.BadZipFile, zipfile.LargeZipFile, tarfile.TarError, EOFError, zlib.error, RuntimeError, OSError) as e:
        raise UploadError(f"{name} could not be read as an archive ({e})") from e
    return list(chosen.values())


def check(raw: Path, uploaded: int) -> dict[str, str]:
    """The files of an upload (put in `raw`; `uploaded`: how many files it had), checked as
    the build checks them: role -> where the file read for it is, in `raw`.

    Raises:
        UploadError: with what is missing, or not as expected.
    """
    files = metacyc_build.resolve(raw)
    found = metacyc_build.problems(files, raw)
    if found:
        lead = [] if any(files.values()) else [
            f"none of the {uploaded} file{'s' if uploaded != 1 else ''} uploaded is one of MetaCyc's SmartTables as app/metacyc_build.py "
            "reads them: upload the folder of them (as BioCyc/raw, with its subfolders), or a .zip or .tar.gz of it"]
        raise UploadError(lead + found)
    return {role: p.relative_to(raw).as_posix() for role, p in files.items() if p}


def start(db, d: Path, version: Optional[str], files: dict[str, str], index: Callable, supervisor: Callable) -> dict:
    """Build the bundle of an upload checked (in folder `d`, from `claim`) and add its
    reference datasets, in the background: the job, as started. `index` and `supervisor`:
    the odor index and what restarts the server (as `reference.add` takes them), when they
    are added."""
    JOB.update(stage="building", version=version)
    threading.Thread(target=_run, args=(db, d, version, files, index, supervisor), name="metacyc-import", daemon=True).start()
    return job()


def _run(db, d: Path, version: Optional[str], files: dict[str, str], index: Callable, supervisor: Callable) -> None:
    say = JOB["log"].append
    try:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", version).strip("._") if version else ""
        bundle = d / (f"metacyc-{name}.sqlite" if name else "metacyc.sqlite")
        about = metacyc_build.build(d / "raw", bundle, version, log=say)
        record = {"id": d.name, "version": version, "uploaded_at": JOB["started_at"], "built_at": about["built_at"],
                  "files": files, "bundle": bundle.name, "counts": about["counts"], "added": []}
        (d / RECORD).write_text(json.dumps(record, indent=1))  # (kept from here on, as its reference datasets need it)
        JOB["stage"] = "adding"
        failed = []
        for body in metacyc_build.views(bundle, version):
            say(f"adding: {body['title']}")
            status, res = reference.add(db, body, index(), supervisor())
            if status != 200:
                failed.append(f"{body['title']} not added: {'; '.join(res.get('errors') or [res.get('error') or 'unknown error'])}")
                continue
            JOB["added"].append({"id": res["id"], "title": body["title"], "state": res["state"], "error": res["error"]})
        if not JOB["added"]:
            raise UploadError(failed)
        record["added"] = [a["id"] for a in JOB["added"]]
        (d / RECORD).write_text(json.dumps(record, indent=1))
        JOB["error"] = "\n".join(failed) or None
        log.info("MetaCyc %s imported in the Admin tab into %s: %s", version, d, ", ".join(record["added"]))
    except (metacyc_build.BuildError, UploadError) as e:
        shutil.rmtree(d, ignore_errors=True)
        JOB["error"] = str(e)
    except Exception as e:
        log.exception("MetaCyc import into %s failed", d)
        shutil.rmtree(d, ignore_errors=True)
        JOB["error"] = f"{type(e).__name__}: {e}"
    finally:
        JOB.update(running=False, stage=None, finished_at=_now())


def job() -> dict:
    """The import under way, or the last one (since the server started)."""
    return {**JOB, "log": list(JOB["log"]), "added": list(JOB["added"])}


def _folder(path: str) -> Optional[Path]:
    """The folder of the import a file is in, if it is in one."""
    try:
        rel = Path(path).resolve().relative_to(config.METACYC_DIR.resolve())
    except (ValueError, OSError):
        return None
    return config.METACYC_DIR / rel.parts[0] if rel.parts else None


def imports(refs: list[dict]) -> list[dict]:
    """The imports kept, newest first: each its record (import.json), with its ``dir``, its
    ``size`` (bytes) and ``refs``: the IDs of the reference datasets of `refs` (as
    `Database.list_references` gives them) of its bundle."""
    out = []
    if not config.METACYC_DIR.is_dir():
        return out
    for d in sorted(config.METACYC_DIR.iterdir(), reverse=True):
        try:
            rec = json.loads((d / RECORD).read_text())
        except (OSError, ValueError):
            continue
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        out.append({**rec, "dir": str(d), "size": size, "refs": [r["id"] for r in refs if _folder(r["path"]) == d]})
    return out


def release(path: str, others: list[str]) -> Optional[Path]:
    """A reference dataset deleted (`path`: its file; `others`: those of the ones left): if
    it was the last of an import's, its folder is moved to ``deleted/metacyc/``. Where to,
    or None."""
    d = _folder(path)
    if d is None or not d.is_dir() or any(_folder(p) == d for p in others):
        return None
    dest = config.DELETED_DIR / "metacyc" / d.name
    if dest.exists():
        dest = dest.with_name(f"{d.name}_{dt.datetime.now():%Y%m%d-%H%M%S}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), dest)
    log.info("MetaCyc import %s: its last reference dataset deleted, its folder moved to %s", d.name, dest)
    return dest
