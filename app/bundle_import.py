"""Reference data bundles imported in the Admin tab (MetaCyc, LOTUS): the files of the folder
a bundle's build script reads (or an archive of it) uploaded, or read from a path on tensor
(`Importer.tensor_source`: copied in the background), kept in the data folder laid out as
the script reads them, built into a bundle there by the script itself (``python -m
app.<kind>_build <raw> <bundle>``, in a process of its own, in the background), and the
bundle's reference datasets added (as the script's ``--register`` adds them).

Each import is a folder ``<kind>/<id>/`` of the data folder (`Importer.dir`): ``raw/``, the
files the build reads, laid out as it reads them (the script can be run on it again); the
bundle; and ``import.json``: when it was uploaded, the files and the reference datasets
added. The files keep the times they were last changed (as the browser gives them, or the
archive has them). Files uploaded that the build does not read are not kept, nor anything of an upload
refused or of a build that failed. Once the last of its reference datasets is deleted, the
folder is moved to ``deleted/<kind>/<id>`` (`Importer.release`).

Each kind (metacyc_import.py, lotus_import.py) is an `Importer`: which files of an upload are
read and where they go (`Importer.pick`), checked as its build checks them (`check`), and
the reference datasets of its bundle (`views`).
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
import zlib
from pathlib import Path
from typing import IO, Callable, Optional, Union

from . import config, reference, tensor

log = logging.getLogger("uvicorn.error")

ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
RECORD = "import.json"
MAX_TENSOR_FILES = 100_000  # files of a folder on tensor looked through (`Importer.tensor_source`)


class UploadError(ValueError):
    """An upload that cannot be imported: its `errors`, each saying what was expected."""

    def __init__(self, errors: Union[list[str], str]):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("\n".join(self.errors))


class BuildError(ValueError):
    """The build refused the files uploaded (what it said)."""


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _zip_time(info: zipfile.ZipInfo) -> Optional[float]:
    """When a file in a zip was last changed (as a POSIX time; its local time as zipped), if it can be told."""
    try:
        return time.mktime((*info.date_time, 0, 0, -1))
    except (OverflowError, ValueError):
        return None


def is_archive(name: str) -> bool:
    return (name or "").lower().endswith(ARCHIVES)


def _idle() -> dict:
    return {"running": False, "id": None, "stage": None, "label": None, "note": None, "started_at": None, "finished_at": None,
            "log": [], "error": None, "added": []}


class Importer:
    """Bundles of one kind imported in the Admin tab, one at a time: subclassed by each kind,
    which gives what is marked so below."""

    key = ""           # the kind: its folder in the data folder (and deleted/), and /api/admin/<key>
    name = ""          # as said
    what = ""          # what an upload is to be, as said ("MetaCyc's SmartTables")
    like = ""          # a folder of them as the build reads it, as said ("BioCyc/raw")
    module = ""        # its build script: python -m <module> <raw> <bundle> [`args`]
    dir_setting = ""   # its folder: config's name for it
    how = "name"       # how a browser finds the files of a folder that are read (`expected`): by their
    #                    names alone ("name"), or by their paths, with the build's globs ("pattern")
    gzip = False       # whether its files are kept gzipped (`gzipped`)

    def __init__(self):
        self._job = _idle()  # the import under way, or the last one
        self._lock = threading.Lock()

    # ------------------------------------------------------------ given by each kind

    def expected(self) -> list[dict]:
        """The files an upload is to have: for each role, its ``files`` (where in the folder
        the build looks for it, the first first), what it is (``about``), and whether it is
        ``required``."""
        raise NotImplementedError

    def pick(self, names: list[str]) -> dict[int, str]:
        """The files of an upload the build reads: their positions in `names` (their paths in
        the folder or archive uploaded, or their names alone) -> where in ``raw/`` they go."""
        raise NotImplementedError

    def check(self, raw: Path, uploaded: int) -> dict[str, str]:
        """The files of an upload (put in `raw`; `uploaded`: how many files it had), checked as
        the build checks them: role -> where the file read for it is, in `raw`.

        Raises:
            UploadError: with what is missing, or not as expected.
        """
        raise NotImplementedError

    def bundle_name(self, raw: Path, opts: dict) -> str:
        raise NotImplementedError

    def views(self, bundle: Path, about: dict, opts: dict) -> list[dict]:
        """The bundle's reference datasets, as the Admin tab adds them (POST /api/admin/references)."""
        raise NotImplementedError

    def read_about(self, bundle: Path) -> dict:
        raise NotImplementedError

    def options(self, form) -> dict:
        """Its options, from the form uploaded (MetaCyc's version), kept in its record."""
        return {}

    def settle(self, raw: Path, opts: dict) -> dict:
        """Its options, from those given (`options`) and the files checked (in `raw`): MetaCyc's
        version, told from them if not given.

        Raises:
            UploadError: if they are not as those given say.
        """
        return opts

    def note(self, opts: dict) -> Optional[str]:
        """What is said of its label, as said (how MetaCyc's version was told)."""
        return None

    def label(self, raw: Path, opts: dict) -> Optional[str]:
        """What tells its imports apart, as said (MetaCyc's version, LOTUS's release)."""
        return None

    def args(self, opts: dict) -> list[str]:
        """The build script's options, after its folder and bundle."""
        return []

    def gzipped(self, rel: str) -> bool:
        """Whether a file of an upload (where it goes, in ``raw/``) is gzipped as it is kept
        (to ``<rel>.gz``): those of a kind that keeps them so (`gzip`) that are not already."""
        return self.gzip and not rel.lower().endswith(".gz")

    # -------------------------------------------------------------------- an import

    @property
    def dir(self) -> Path:
        return getattr(config, self.dir_setting)

    def claim(self) -> Path:
        """Start an import (one at a time): its folder, made (and those of imports the server
        stopped during, removed).

        Raises:
            UploadError: if one is under way.
        """
        with self._lock:
            if self._job["running"]:
                raise UploadError(f"{self.name} is being imported already (since {self._job['started_at']}): wait for it to finish")
            self.dir.mkdir(parents=True, exist_ok=True)
            for d in self.dir.iterdir():
                if d.is_dir() and not (d / RECORD).exists():
                    shutil.rmtree(d, ignore_errors=True)
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            d, n = self.dir / stamp, 2
            while d.exists():
                d, n = self.dir / f"{stamp}-{n}", n + 1
            (d / "raw").mkdir(parents=True)
            self._job = {**_idle(), "running": True, "id": d.name, "stage": "uploading", "started_at": _now()}
            return d

    def abandon(self, d: Path) -> None:
        """An import refused before it started (its upload not as expected): its folder removed."""
        shutil.rmtree(d, ignore_errors=True)
        self._job.update(running=False, id=None, stage=None)

    def save(self, src: IO[bytes], raw: Path, rel: str, max_bytes: int, what: str, mtime: Optional[float] = None) -> None:
        """Keep a file of an upload (`src`) in `raw`, at `rel` (gzipped, at ``<rel>.gz``, if it
        is to be: `gzipped`), last changed at `mtime` (as the file uploaded was, if given: a
        POSIX time). `what`: the file, as said.

        Raises:
            UploadError: if more than `max_bytes` would be written.
        """
        gz = self.gzipped(rel)
        dest = raw / (rel + ".gz" if gz else rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            out = gzip.GzipFile(filename="", fileobj=f, mode="wb", compresslevel=6) if gz else f
            try:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
                    if f.tell() > max_bytes:
                        raise UploadError(f"{what}: larger than the {max_bytes / 2 ** 20:g} MB limit")
            finally:
                if gz:
                    out.close()
        if mtime is not None:
            try:
                os.utime(dest, (mtime, mtime))
            except (OverflowError, ValueError, OSError):
                pass  # (a time out of range: as written)

    def extract(self, archive: Path, name: str, raw: Path, max_bytes: int) -> list[str]:
        """Take the files the build reads (see `pick`) out of an archive uploaded (zip or tar;
        `name`: as uploaded) into `raw`, where it reads them: those places."""
        try:
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as z:
                    members = [i for i in z.infolist() if not i.is_dir()]
                    chosen = self.pick([i.filename for i in members])
                    for k, rel in chosen.items():
                        with z.open(members[k]) as src:
                            self.save(src, raw, rel, max_bytes, f"{name}: {members[k].filename}", _zip_time(members[k]))
            elif tarfile.is_tarfile(archive):
                with tarfile.open(archive) as t:
                    members = [m for m in t.getmembers() if m.isfile()]
                    chosen = self.pick([m.name for m in members])
                    for k, rel in chosen.items():
                        with t.extractfile(members[k]) as src:
                            self.save(src, raw, rel, max_bytes, f"{name}: {members[k].name}", members[k].mtime or None)
            else:
                raise UploadError(f"{name} is not a zip or tar archive")
        except (zipfile.BadZipFile, zipfile.LargeZipFile, tarfile.TarError, EOFError, zlib.error, RuntimeError, OSError) as e:
            raise UploadError(f"{name} could not be read as an archive ({e})") from e
        return list(chosen.values())

    def tensor_source(self, path: str, root: Path) -> dict:
        """A path on tensor (from its top, as entered) to import from, instead of an upload:
        the folder the build reads, or an archive of it, checked as data folders are
        (`tensor.locate`).

        Returns:
            ``input``; ``path``: normalized (or None); ``ok``; ``error``; ``kind``: 'folder'
            or 'archive'; ``size``: an archive's; of a folder, ``n``: its files, and
            ``files``: those the build reads [{path (in the folder, after its name, as a
            browser names a folder's files), dest (where in ``raw/`` it goes), size, mtime}].
        """
        out = {"input": path, "path": None, "ok": False, "error": None, "kind": None, "size": None, "n": None, "files": []}
        try:
            rel, target = tensor.locate(path, root)
            out["path"] = rel
            if os.path.isdir(target):
                found = []
                for top, dirs, names in os.walk(target):
                    dirs.sort()
                    for name in sorted(names):
                        found.append(os.path.join(top, name))
                    if len(found) > MAX_TENSOR_FILES:
                        raise tensor.PathError(f"it has more than {MAX_TENSOR_FILES:,} files: give the folder of {self.what} itself")
                names = [f"{target.name}/{Path(f).relative_to(target).as_posix()}" for f in found]
                files = []
                for k, dest in self.pick(names).items():
                    st = os.stat(found[k])
                    files.append({"path": names[k], "dest": dest, "size": st.st_size, "mtime": st.st_mtime})
                out.update(kind="folder", n=len(names), files=sorted(files, key=lambda f: f["path"]))
            elif is_archive(target.name):
                out.update(kind="archive", size=os.path.getsize(target))
            else:
                raise tensor.PathError(f"a file, but not an archive ({', '.join(ARCHIVES)}): give the folder of {self.what}, or an archive of it")
            if not os.access(target, os.R_OK):
                raise tensor.PathError("exists, but the server is not allowed to read it")
        except (tensor.PathError, OSError) as e:
            out.update(error=str(e), kind=None, files=[])
            return out
        out["ok"] = True
        return out

    def take(self, src: dict, root: Path, raw: Path, max_bytes: int) -> int:
        """The files of a path on tensor (`tensor_source`) put in `raw`, as an upload's are: a
        folder's that the build reads copied (gzipped, those to be kept so), keeping when they
        were last changed; an archive's taken out of it. How many files that was.

        Raises:
            UploadError: if they could not be read, or one is too large.
        """
        target = root / src["path"]
        if src["kind"] == "archive":
            self.extract(target, src["path"], raw, max_bytes)
            return 1
        for f in src["files"]:
            where = f"{src['path']}/{f['path'].split('/', 1)[1]}"
            try:
                with open(target.parent / f["path"], "rb") as fh:
                    self.save(fh, raw, f["dest"], max_bytes, where, f["mtime"])
            except OSError as e:
                raise UploadError(f"{where} could not be read on tensor ({e})") from e
        return len(src["files"])

    def start_from_tensor(self, db, d: Path, opts: dict, src: dict, index: Callable, supervisor: Callable) -> dict:
        """As `start`, of a path on tensor (`tensor_source`): its files first put in ``raw/``
        (`take`), checked and its options settled as an upload's are, in the background too
        (a folder's files can take minutes to copy and gzip). The job, as started."""
        self._job.update(stage="copying", note=f"from {src['path']} on tensor")

        def run() -> None:
            job = self._job
            try:
                n = self.take(src, config.TENSOR_ROOT, d / "raw", int(config.MAX_UPLOAD_MB * 1024 * 1024))
                found = self.check(d / "raw", n)
                settled = {**self.settle(d / "raw", opts), "tensor_path": src["path"]}  # (kept in its record)
            except Exception as e:
                if not isinstance(e, UploadError):
                    log.exception("%s import from %s on tensor failed", self.name, src["path"])
                shutil.rmtree(d, ignore_errors=True)
                job.update(running=False, stage=None, finished_at=_now(),
                           error=str(e) if isinstance(e, UploadError) else f"{type(e).__name__}: {e}")
                return
            job.update(stage="building", label=self.label(d / "raw", settled), note=self.note(settled))
            self._run(db, d, settled, found, index, supervisor)

        threading.Thread(target=run, name=f"{self.key}-import", daemon=True).start()
        return self.job()

    def start(self, db, d: Path, opts: dict, files: dict[str, str], index: Callable, supervisor: Callable) -> dict:
        """Build the bundle of an upload checked (in folder `d`, from `claim`) and add its
        reference datasets, in the background: the job, as started. `index` and `supervisor`:
        the odor index and what restarts the server (as `reference.add` takes them), when they
        are added."""
        self._job.update(stage="building", label=self.label(d / "raw", opts), note=self.note(opts))
        threading.Thread(target=self._run, args=(db, d, opts, files, index, supervisor), name=f"{self.key}-import",
                         daemon=True).start()
        return self.job()

    def build(self, raw: Path, bundle: Path, opts: dict, say: Callable[[str], None]) -> None:
        """Run the build script on `raw` into `bundle`, in a process of its own (so that what
        it takes, 1.7 GB for LOTUS, is given back when it is done, and the server stays as
        quick meanwhile): each line it prints said (`say`).

        Raises:
            BuildError: with what it said, if it refused the files (it exits with 2).
            RuntimeError: if it failed otherwise.
        """
        cmd = [sys.executable, "-m", self.module, str(raw), str(bundle), *self.args(opts)]
        # (its temporary files, as SQLite's in sorting to index, in the data folder: Landlock, which holds it as it holds
        # the server, lets it write nowhere else)
        tmp = str(config.TMP_DIR)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TMPDIR": tmp, "SQLITE_TMPDIR": tmp}
        config.TMP_DIR.mkdir(parents=True, exist_ok=True)
        errors = bundle.with_name("build-errors.txt")
        with open(errors, "w+b") as err:
            proc = subprocess.Popen(cmd, cwd=config.ROOT, env=env, stdout=subprocess.PIPE, stderr=err, text=True,
                                    encoding="utf-8", errors="replace")
            with proc:
                for line in proc.stdout:
                    if line.strip():
                        say(line.rstrip())
            err.seek(0)
            said = err.read().decode("utf-8", errors="replace")
        errors.unlink(missing_ok=True)
        if proc.returncode == 0:
            return
        m = re.search(r"^error: ", said, re.M)
        if proc.returncode == 2 and m:
            raise BuildError(said[m.end():].strip())
        log.error("%s build failed (%s exited with %s):\n%s", self.name, self.module, proc.returncode, said)
        lines = [x for x in said.splitlines() if x.strip()]
        raise RuntimeError(lines[-1] if lines else f"{self.module} exited with {proc.returncode}")

    def _run(self, db, d: Path, opts: dict, files: dict[str, str], index: Callable, supervisor: Callable) -> None:
        job = self._job
        say = job["log"].append
        try:
            bundle = d / self.bundle_name(d / "raw", opts)
            self.build(d / "raw", bundle, opts, say)
            about = self.read_about(bundle)
            record = {"id": d.name, "kind": self.key, "label": job["label"], "note": job["note"], **opts,
                      "uploaded_at": job["started_at"],
                      "built_at": about.get("built_at"), "files": files, "bundle": bundle.name, "counts": about.get("counts"),
                      "added": []}
            if about.get("release"):
                record["release"] = about["release"]
            (d / RECORD).write_text(json.dumps(record, indent=1))  # (kept from here on, as its reference datasets need it)
            job["stage"] = "adding"
            failed = []
            for body in self.views(bundle, about, opts):
                say(f"adding: {body['title']}")
                status, res = reference.add(db, body, index(), supervisor())
                if status != 200:
                    failed.append(f"{body['title']} not added: {'; '.join(res.get('errors') or [res.get('error') or 'unknown error'])}")
                    continue
                job["added"].append({"id": res["id"], "title": body["title"], "state": res["state"], "error": res["error"]})
            if not job["added"]:
                raise UploadError(failed)
            record["added"] = [a["id"] for a in job["added"]]
            (d / RECORD).write_text(json.dumps(record, indent=1))
            job["error"] = "\n".join(failed) or None
            log.info("%s %s imported in the Admin tab into %s: %s", self.name, job["label"] or "", d, ", ".join(record["added"]))
        except (BuildError, UploadError) as e:
            shutil.rmtree(d, ignore_errors=True)
            job["error"] = str(e)
        except Exception as e:
            log.exception("%s import into %s failed", self.name, d)
            shutil.rmtree(d, ignore_errors=True)
            job["error"] = f"{type(e).__name__}: {e}"
        finally:
            job.update(running=False, stage=None, finished_at=_now())

    def job(self) -> dict:
        """The import under way, or the last one (since the server started)."""
        return {**self._job, "log": list(self._job["log"]), "added": list(self._job["added"])}

    # ------------------------------------------------------------------ those kept

    def _folder(self, path: str) -> Optional[Path]:
        """The folder of the import a file is in, if it is in one."""
        try:
            rel = Path(path).resolve().relative_to(self.dir.resolve())
        except (ValueError, OSError):
            return None
        return self.dir / rel.parts[0] if rel.parts else None

    def imports(self, refs: list[dict]) -> list[dict]:
        """The imports kept, newest first: each its record (import.json), with its ``dir``,
        its ``size`` (bytes) and ``refs``: the IDs of the reference datasets of `refs` (as
        `Database.list_references` gives them) of its bundle."""
        out = []
        if not self.dir.is_dir():
            return out
        for d in sorted(self.dir.iterdir(), reverse=True):
            try:
                rec = json.loads((d / RECORD).read_text())
            except (OSError, ValueError):
                continue
            rec.setdefault("label", rec.get("version"))  # (MetaCyc's, imported before LOTUS's were)
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            out.append({**rec, "dir": str(d), "size": size, "refs": [r["id"] for r in refs if self._folder(r["path"]) == d]})
        return out

    def release(self, path: str, others: list[str]) -> Optional[Path]:
        """A reference dataset deleted (`path`: its file; `others`: those of the ones left): if
        it was the last of an import's, its folder is moved to ``deleted/<kind>/``. Where to,
        or None."""
        d = self._folder(path)
        if d is None or not d.is_dir() or any(self._folder(p) == d for p in others):
            return None
        dest = config.DELETED_DIR / self.key / d.name
        if dest.exists():
            dest = dest.with_name(f"{d.name}_{dt.datetime.now():%Y%m%d-%H%M%S}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), dest)
        log.info("%s import %s: its last reference dataset deleted, its folder moved to %s", self.name, d.name, dest)
        return dest
