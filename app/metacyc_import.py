"""MetaCyc imported in the Admin tab (see bundle_import.py): its SmartTables uploaded (the
files of the folder metacyc_build.py reads, or an archive of it), kept in the data folder as
that lays them out (``metacyc/<id>/raw/``), built into a bundle there, and the bundle's two
reference datasets added (as ``python -m app.metacyc_build <raw> <bundle> --register
<server>`` adds them).

The files are found by their names (no two the build reads are named alike), as a browser
may send no folders, and put where the build reads them. Which MetaCyc they are of is told
from them (`metacyc_build.identify`), unless given (and refused if another release's).
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Optional

from . import metacyc, metacyc_build
from .bundle_import import Importer, UploadError

# where the build reads each file it reads, by the file's name (no two are named alike)
_PLACES = {PurePosixPath(rel).name: rel for rels in metacyc_build.FILES.values() for rel in rels}


class MetacycImport(Importer):
    key, name, module, dir_setting = "metacyc", "MetaCyc", "app.metacyc_build", "METACYC_DIR"
    what, like = "MetaCyc's SmartTables", "BioCyc/raw"

    def expected(self) -> list[dict]:
        return [{"role": r, "files": fs, "about": metacyc_build.ABOUT[r], "required": r in metacyc_build.REQUIRED}
                for r, fs in metacyc_build.FILES.items()]

    def pick(self, names: list[str]) -> dict[int, str]:
        """Of several named alike, the one at the path the build reads it from (in any
        folder), else the first."""
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

    def check(self, raw: Path, uploaded: int) -> dict[str, str]:
        files = metacyc_build.resolve(raw)
        found = metacyc_build.problems(files, raw)
        if found:
            lead = [] if any(files.values()) else [
                f"none of the {uploaded} file{'s' if uploaded != 1 else ''} uploaded is one of MetaCyc's SmartTables as "
                "app/metacyc_build.py reads them: upload the folder of them (as BioCyc/raw, with its subfolders), or a .zip or "
                ".tar.gz of it"]
            raise UploadError(lead + found)
        return {role: p.relative_to(raw).as_posix() for role, p in files.items() if p}

    def options(self, form) -> dict:
        return {"version": re.sub(r"\s+", " ", str(form.get("version") or "")).strip()[:40] or None}

    def settle(self, raw: Path, opts: dict) -> dict:
        """Its version: as given, else told from the files (`metacyc_build.identify`, kept as
        ``identified``).

        Raises:
            UploadError: if the version given is another release's than they are of.
        """
        told = metacyc_build.identify(metacyc_build.resolve(raw))
        if why := metacyc_build.contradicts(opts.get("version"), told):
            raise UploadError(f"{why}. Leave the version blank, and it is told from them")
        return {**opts, "given": opts.get("version"), "version": opts.get("version") or told["version"], "identified": told}

    def note(self, opts: dict) -> Optional[str]:
        told, given = opts.get("identified") or {}, opts.get("given")
        if not given:
            how = (f"Its version, {told['version']}, was told from its files" if told.get("version")
                   else "It has no version: none was given, nor could it be told from its files")
        elif told.get("version") == given:
            how = f"Its version, {given}, was given, and its files say so too"
        elif told.get("version"):
            how = f"Its version, {given}, was given; its files say {told['version']}"
        else:
            how = f"Its version, {given}, was given; its files cannot tell it"
        return f"{how}. {told['basis']}." if told.get("basis") else f"{how}."

    def label(self, raw: Path, opts: dict) -> Optional[str]:
        return opts.get("version")

    def bundle_name(self, raw: Path, opts: dict) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", opts.get("version") or "").strip("._")
        return f"metacyc-{name}.sqlite" if name else "metacyc.sqlite"

    def args(self, opts: dict) -> list[str]:
        return ["--version", opts.get("version") or ""]

    def views(self, bundle: Path, about: dict, opts: dict) -> list[dict]:
        return metacyc_build.views(bundle, about.get("version"))

    def read_about(self, bundle: Path) -> dict:
        return metacyc.read_about(bundle)


importer = MetacycImport()
