"""LOTUS imported in the Admin tab (see bundle_import.py): its files uploaded (those of the
folder lotus_build.py reads, as ``Scraped-Databases-Copy/LOTUS/raw``, or an archive of it),
kept in the data folder laid out as they were there (``lotus/<id>/raw/``: the dates of the
release are in the folders' names), gzipped, built into a bundle there, and its reference
dataset added (as ``python -m app.lotus_build <raw> <bundle> --register <server>`` adds it):
the odor table's chemicals by the families of organisms they are reported in, its settings
changed after as any LOTUS dataset's are.

The files are found by their paths, as the build finds them (its globs, `lotus_build.FILES`),
each put at the part of its path the glob matched. Of a file there both as is and gzipped
(``.tsv`` and ``.tsv.gz``), the gzipped one is taken: the same, and smaller. Notes beside them
(``notes.txt``: where they were downloaded from, and when) are kept too, as the build reads
them.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath
from typing import Optional

from . import lotus, lotus_build
from .bundle_import import Importer, UploadError

NOTES = "notes.txt"


def place(name: str, pattern: str) -> Optional[str]:
    """Where a file of an upload (`name`: its path in the folder or archive uploaded) goes, if
    a glob of the build (`pattern`, relative to the folder it reads) finds it: the last parts
    of its path that the glob's parts match (after a leading ``**/``: its name alone), or None."""
    parts = name.replace("\\", "/").split("/")
    globs = pattern.split("/")
    if globs[0] == "**":
        globs = globs[1:]
    if len(parts) < len(globs):
        return None
    tail = parts[len(parts) - len(globs):]
    if any(t in ("", ".", "..") for t in tail) or not all(fnmatch.fnmatchcase(t, g) for t, g in zip(tail, globs)):
        return None
    return "/".join(tail)


class LotusImport(Importer):
    key, name, module, dir_setting = "lotus", "LOTUS", "app.lotus_build", "LOTUS_DIR"
    what, like = "LOTUS's files", "Scraped-Databases-Copy/LOTUS/raw"
    how = "pattern"
    gzip = True

    def expected(self) -> list[dict]:
        return [{"role": r, "files": fs, "about": lotus_build.ABOUT[r], "required": r in lotus_build.REQUIRED}
                for r, fs in lotus_build.FILES.items()]

    def pick(self, names: list[str]) -> dict[int, str]:
        """For each role, the file the build would read (`lotus_build.resolve`) of them, found
        by where it goes (but of one there both as is and gzipped, gzipped: `newest`), the first
        of those alike; and notes in its folder or the one above it."""
        norm = [n.replace("\\", "/") for n in names]
        first: dict[str, int] = {}
        for i, n in enumerate(norm):
            first.setdefault(n, i)
        out: dict[int, str] = {}
        for patterns in lotus_build.FILES.values():
            found: dict[str, int] = {}
            for pattern in patterns:
                for i, n in enumerate(norm):
                    if (p := place(n, pattern) or place(n, pattern + ".gz")) and p not in found:
                        found[p] = i
                if found:
                    break
            if not found:
                continue
            p = lotus_build.newest(list(found), gzipped=True)
            i = found[p]
            out[i] = p
            parts, path = p.split("/"), norm[i].split("/")
            lead = path[:len(path) - len(parts)]
            for up in (1, 2):
                if len(parts) < up:
                    break
                j = first.get("/".join([*lead, *parts[:-up], NOTES]))
                if j is not None:
                    out.setdefault(j, "/".join([*parts[:-up], NOTES]))
        return out

    def gzipped(self, rel: str) -> bool:
        return super().gzipped(rel) and PurePosixPath(rel).name != NOTES

    def check(self, raw: Path, uploaded: int) -> dict[str, str]:
        files = lotus_build.resolve(raw)
        found = lotus_build.problems(files, raw)
        if found:
            lead = [] if any(files.values()) else [
                f"none of the {uploaded} file{'s' if uploaded != 1 else ''} uploaded is one of LOTUS's files as "
                "app/lotus_build.py reads them: upload the folder of them (as Scraped-Databases-Copy/LOTUS/raw, with its "
                "subfolders), or a .zip or .tar.gz of it"]
            raise UploadError(lead + found)
        return {role: p.relative_to(raw).as_posix() for role, p in files.items() if p}

    def label(self, raw: Path, opts: dict) -> Optional[str]:
        return lotus_build.release(lotus_build.resolve(raw))["pairs"]

    def bundle_name(self, raw: Path, opts: dict) -> str:
        when = self.label(raw, opts)
        return f"lotus-{when}.sqlite" if when else "lotus.sqlite"

    def views(self, bundle: Path, about: dict, opts: dict) -> list[dict]:
        return [lotus_build.view(bundle, about)]

    def read_about(self, bundle: Path) -> dict:
        return lotus.read_about(bundle)


importer = LotusImport()
