"""Builds a MetaCyc bundle: the SmartTables exported from MetaCyc (tab-separated, as in
``BioCyc/raw``: compounds, pathways, reactions, organisms) read into one SQLite file, which
the app reads as a reference dataset (see metacyc.py and reference_data/README.md). Run:

  python -m app.metacyc_build <raw folder> <bundle.sqlite> [--version 27.1] [--register http://<server>:<port>]

``--register`` adds its two reference datasets to the app (as the Admin tab does): its
compounds by the pathways they are in, and the pathway LCA distance between them.

Each table is a SmartTable exported twice, by frame IDs and by common names, in the same
row order: IDs are taken from the first, the names of what they list from the second (by
position). Which files are read can be changed with ``--file <role>=<path>`` (see `FILES`).
They are checked before anything is read (`problems`): each there, with the columns it needs.

SmartTables do not say which MetaCyc they are of: that is told from how many pathways and
compounds they have, against how many each release's notes say it has (`identify`), unless
``--version`` is given (and refused if it is another release's).

The Admin tab can do all this too, from the folder uploaded (see metacyc_import.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# role -> the file read for it (relative to the raw folder): the first of these that exists
FILES = {
    "compounds": ["compounds/all_metacyc_compounds_with_gen_pathways__frame_ids__2024-02-05.tsv",
                  "compounds/all_metacyc_compounds_with_gen_pathways__frame_ids.tsv"],
    "compound_names": ["compounds/all_metacyc_compounds_with_gen_pathways__common_name__2024-02-05.tsv",
                       "compounds/all_metacyc_compounds_with_gen_pathways__common_name.tsv"],
    "pathways": ["pathways/all_metacyc_pathways__frame_ids__2024-01-24.txt", "pathways/all_metacyc_pathways__frame_ids.txt"],
    "pathway_names": ["pathways/all_metacyc_pathways__common_name__2024-01-24.txt", "pathways/all_metacyc_pathways__common_name.txt"],
    "reactions": ["reactions/all_metacyc_reactions__frame_ids.tsv"],
    "reaction_names": ["reactions/all_metacyc_reactions__common_name.tsv"],
    "species": ["organisms/species-from-metacyc_species.txt"],
    "genomes": ["organisms/all_organism_databases.txt"],
    "summaries": ["pathways/pathway_summary.txt"],
    "pathway_extra": ["pathways/all_metacyc_pathways(2).txt"],  # taxonomic range, engineered
    "pathway_ontology": ["ontology/pathway_ontology.txt"],
}
REQUIRED = ("compounds", "compound_names", "pathways", "pathway_names", "reactions", "reaction_names")
ABOUT = {  # role -> what the file is (as said when it is missing or not as expected)
    "compounds": "MetaCyc's SmartTable of all compounds with their pathways, exported by frame IDs",
    "compound_names": "MetaCyc's SmartTable of all compounds with their pathways, exported by common names",
    "pathways": "MetaCyc's SmartTable of all pathways, exported by frame IDs",
    "pathway_names": "MetaCyc's SmartTable of all pathways, exported by common names",
    "reactions": "MetaCyc's SmartTable of all reactions, exported by frame IDs",
    "reaction_names": "MetaCyc's SmartTable of all reactions, exported by common names",
    "species": "MetaCyc's SmartTable of its species",
    "genomes": "BioCyc's SmartTable of all its organism databases (their environments, hosts, ...)",
    "summaries": "MetaCyc's SmartTable of pathways' summaries",
    "pathway_extra": "MetaCyc's SmartTable of pathways with their taxonomic ranges and whether engineered",
    "pathway_ontology": "MetaCyc's pathway class tree as text, each level indented by 4 more spaces, from 'Pathways'",
}
_COMPOUND_COLS = ("Compound", "Common-Name", "Pathways of compound", "Ontology - direct parents of entity")
_PATHWAY_COLS = ("Object ID", "Common-Name", "Reactions of pathway", "Reactants of pathway", "Products of pathway")
COLUMNS = {  # role -> the columns its file must have (others it reads are used if there)
    "compounds": _COMPOUND_COLS, "compound_names": _COMPOUND_COLS, "pathways": _PATHWAY_COLS, "pathway_names": _PATHWAY_COLS,
    "reactions": ("Object ID", "Reactants of reaction", "Products of reaction", "In-Pathway"),
    "reaction_names": ("Object ID", "Reaction"), "species": ("Object ID",), "genomes": ("Organism name",),
    "summaries": ("Object ID", "Summary"), "pathway_extra": ("Object ID", "Taxonomic-Range", "Engineered?"),
}


# MetaCyc's releases (biocyc.org/meta/release-notes.shtml): its version, when it was released, and how many pathways (not
# counting super-pathways) and compounds its notes say it has, as the SmartTables of a folder are told by (`identify`)
RELEASES = [
    ("24.1", "2020-09-08", 2847, 16631), ("24.5", "2021-01-07", 2859, 16861), ("25.0", "2021-05-19", 2937, 17314),
    ("25.1", "2021-08-05", 2969, 17572), ("25.5", "2021-12-15", 2980, 17656), ("26.0", "2022-04-13", 3006, 18124),
    ("26.1", "2022-08-24", 3063, 18622), ("26.5", "2022-12-13", 3085, 18785), ("27.0", "2023-04-12", 3105, 18973),
    ("27.1", "2023-08-28", 3128, 19173), ("28.0", "2024-04-02", 3153, 19372), ("28.1", "2024-08-12", 3190, 19569),
    ("28.5", "2024-12-11", 3215, 19834), ("29.0", "2025-04-10", 3231, 20072), ("29.1", "2025-09-22", 3264, 20490),
    ("29.5", "2025-12-15", 3270, 20553), ("29.6", "2026-02-26", 3284, 20642), ("30.0", "2026-08-06", 3305, 20791),
]
# how near a release's counts (pathways, compounds) those of its SmartTables are (27.0's, exported in July 2023: the
# same; 28.0's, in May 2024: 1 and 4 fewer)
NEAR = (2, 10)


class BuildError(ValueError):
    """The SmartTables are not as the build reads them (what it expects said)."""


class VersionError(BuildError):
    """The version given is another release's than the SmartTables are of (`contradicts`)."""


SCHEMA = """
CREATE TABLE about (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE compounds (id TEXT PRIMARY KEY, name TEXT, formula TEXT, smiles TEXT, inchi TEXT, inchikey TEXT,
                        cid INTEGER, kegg TEXT, parents TEXT, pathways TEXT, general_pathways TEXT);
CREATE TABLE names (id TEXT PRIMARY KEY, name TEXT);  -- of every frame named anywhere (compounds, classes, ...)
CREATE TABLE pathway_classes (id TEXT PRIMARY KEY, name TEXT, parent TEXT);  -- parent: its parent class's name, where known
CREATE TABLE pathway_ontology (name TEXT PRIMARY KEY, parent TEXT);  -- every class of the tree exported, by name
CREATE TABLE pathways (id TEXT PRIMARY KEY, name TEXT, types TEXT, species TEXT, reactions TEXT, reactants TEXT,
                       products TEXT, primary_reactants TEXT, primary_products TEXT, taxonomic_range TEXT,
                       engineered INTEGER, summary TEXT, super TEXT, subs TEXT);
CREATE TABLE reactions (id TEXT PRIMARY KEY, equation TEXT, ec TEXT, kegg TEXT, direction TEXT, reactants TEXT,
                        products TEXT, pathways TEXT);
CREATE TABLE species (id TEXT PRIMARY KEY, name TEXT, strain TEXT, synonyms TEXT);
CREATE TABLE genomes (id TEXT PRIMARY KEY, organism TEXT, species TEXT, environment TEXT, depth TEXT, health TEXT,
                      host TEXT, pathogenicity TEXT, oxygen TEXT, temperature TEXT, trophic TEXT, strain TEXT);
"""

_TAG = r"</?[A-Za-z][^<>]*>"  # (not an equation's arrows: <-, <-->)
_SUB = str.maketrans("0123456789+-=()n", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙ")
_SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")


def plain(s: Optional[str]) -> str:
    """MetaCyc's HTML-ish text as plain text: entities decoded, sub- and superscripts as
    Unicode where they can be (``CO<SUB>2</SUB>`` as CO₂), other tags dropped (but arrows such as ``<-->``)."""
    if not s:
        return ""

    def script(m: re.Match) -> str:
        inner = re.sub(_TAG, "", m[2])
        table = _SUB if m[1].lower() == "sub" else _SUP
        return inner.translate(table) if all(c in "0123456789+-=()n" for c in inner) else inner

    s = re.sub(r"<(sub|sup)>(.*?)</\1>", script, s, flags=re.I | re.S)
    return html.unescape(re.sub(_TAG, "", s)).strip()


def split(s: Optional[str]) -> list[str]:
    """A SmartTable cell's values (separated by `` // ``)."""
    return [x.strip() for x in (s or "").split(" // ") if x.strip()]


def _link_text(s: str) -> str:
    """A cell that may be an HTML link (``<a href=...>45479569</a>``) as its text."""
    return plain(s)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding_errors="replace")


def _lines(path: Path) -> list[str]:
    """A text file's lines, each decoded as UTF-8, else as Windows-1252 (as some exports mix them)."""
    out = []
    for raw in path.read_bytes().split(b"\n"):
        raw = raw.rstrip(b"\r")
        try:
            out.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            out.append(raw.decode("cp1252", errors="replace"))
    return out


def _header(path: Path) -> list[str]:
    """A tab-separated file's first line, as its columns."""
    with open(path, "rb") as f:
        return f.readline(1 << 16).decode("utf-8", errors="replace").rstrip("\r\n").split("\t")


def _records(path: Path, ids: set[str]) -> tuple[list[str], dict[str, list[str]]]:
    """A SmartTable whose last column runs over several lines (unquoted, as pathway summaries
    are): its columns, and each record by its ID (its "Object ID" on the line it starts on,
    one of `ids`), its fields, the last with the lines after it joined to it."""
    lines = _lines(path)
    header = lines[0].split("\t") if lines else []
    id_field = header.index("Object ID")
    out: dict[str, list[str]] = {}
    current = None
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) > id_field and fields[id_field] in ids and fields[id_field] not in out:
            current = out[fields[id_field]] = fields
        elif current is not None:
            current[-1] += "\n" + line
    return header, out


def _named(ids: list[str], names: list[str], into: dict[str, str]) -> None:
    """Names of IDs listed in a cell (by position, as in the common-name export)."""
    if len(ids) == len(names):
        for i, n in zip(ids, names):
            if n and i not in into:
                into[i] = plain(n)


def _summary(text: str, names: dict[str, str]) -> str:
    """A pathway's summary as plain text: frames as their names, citations left out."""
    text = re.sub(r'\|FRAME:\s*([^|"\s]+)\s+"([^"]*)"\s*\|', lambda m: m[2], text)
    text = re.sub(r"\|FRAME:\s*([^|\s]+)\s*\|", lambda m: names.get(m[1], m[1]), text)
    text = re.sub(r"\s*\|CITS:[^|]*\|", "", text)
    paragraphs = re.split(r"\n\s*\n", plain(text.replace("\r", "")))  # (lines within one wrapped as typed)
    return "\n\n".join(p for p in (re.sub(r"\s*\n\s*", " ", q).strip() for q in paragraphs) if p)


def _ontology_parents(path: Path) -> dict[str, str]:
    """The pathway class tree as indented in its export: class name -> its parent's name
    (classes with their pathways folded up in it have none under them)."""
    parents: dict[str, str] = {}
    stack: list[tuple[int, str]] = []
    for line in _lines(path):
        if not line.strip():
            continue
        depth = (len(line) - len(line.lstrip(" "))) // 4
        name = re.sub(r"\s*\(\d+\)\s*$", "", line.strip())
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack and stack[-1][1] != "Pathways":
            parents.setdefault(name, stack[-1][1])
        stack.append((depth, name))
    return parents


def direction(equation: str) -> str:
    """A reaction's direction from its equation's arrow: 'L2R' (reactants to products, as
    listed by frame ID: `->` and `<-` both), 'REV' (`<-->`) or '?' (`=`, or no equation)."""
    m = re.search(r"\s(<-->|->|<-|=)\s", equation or "")
    return {"->": "L2R", "<-": "L2R", "<-->": "REV"}.get(m[1], "?") if m else "?"


def resolve(raw: Path, overrides: Optional[dict[str, str]] = None) -> dict[str, Optional[Path]]:
    """The file read for each role (`FILES`): as given in `overrides`, else the first that
    exists (None if none does)."""
    out = {}
    for role, candidates in FILES.items():
        given = (overrides or {}).get(role)
        if given:
            p = Path(given)
            out[role] = p if p.is_absolute() else raw / p
            continue
        out[role] = next((raw / c for c in candidates if (raw / c).is_file()), None)
    return out


def expected(role: str) -> str:
    """What the file of a role is, and where in the raw folder it is read from."""
    return f"{ABOUT[role]}, as {' or '.join(FILES[role])}"


def problems(files: dict[str, Optional[Path]], raw: Optional[Path] = None) -> list[str]:
    """Why the files of `resolve` cannot be read as MetaCyc's SmartTables (empty if they can):
    those required missing, and any without the columns it needs (`COLUMNS`), or that are not
    as the class tree is (by their first lines). Each says what was expected."""
    out = []

    def shown(p: Path) -> str:
        try:
            return p.relative_to(raw).as_posix() if raw else str(p)
        except ValueError:
            return str(p)

    for role in FILES:
        p = files.get(role)
        if p is None or not p.is_file():
            if role in REQUIRED or p is not None:
                out.append(f"{'missing' if p is None else f'{p} is not a file'}: {expected(role)}")
            continue
        try:
            if role == "pathway_ontology":
                with open(p, "rb") as f:
                    head = f.read(1 << 16).decode("utf-8", errors="replace")
                if "Pathways" not in (x.strip() for x in head.splitlines()[:5]):
                    out.append(f"{shown(p)} is not {ABOUT[role]} (its first lines have no 'Pathways')")
                continue
            header = _header(p)
        except OSError as e:
            out.append(f"{shown(p)} cannot be read ({e.strerror or e})")
            continue
        lacking = [c for c in COLUMNS[role] if c not in header]
        if lacking:
            has = ", ".join(header[:12]) + (", …" if len(header) > 12 else "") if len(header) > 1 else "none: it is not tab-separated"
            out.append(f"{shown(p)} is not {ABOUT[role]}: it lacks the column{'s' if len(lacking) > 1 else ''} "
                       f"{', '.join(lacking)} (its columns: {has})")
    return out


_FRAME = re.compile(r"^[^\s&<>(),';]+$")  # (all frame IDs are; two thirds of names are not)


def _frames(cells: Iterable[str], what: str, file: Path) -> None:
    """Check that the values of a column of a table exported by frame IDs are frame IDs."""
    ids = [i for c in cells for i in split(c)]
    if ids and sum(bool(_FRAME.match(i)) for i in ids) < 0.9 * len(ids):
        raise BuildError(f"{file.name}: its {what} are names, not frame IDs: give the SmartTable exported by frame IDs as it, "
                         "and the one exported by common names as the other")


def _newest_on(day: str) -> Optional[str]:
    """MetaCyc's newest release on a day (YYYY-MM-DD), of `RELEASES` (None if before them)."""
    out = None
    for v, released, *_ in RELEASES:
        if released <= day:
            out = v
    return out


def _dated(files: dict[str, Optional[Path]], roles: Iterable[str]) -> Optional[tuple[str, str]]:
    """The first and last of the dates (YYYY-MM-DD) the files of `roles` were last changed."""
    days = sorted(dt.date.fromtimestamp(files[r].stat().st_mtime).isoformat() for r in roles if files.get(r) and files[r].is_file())
    return (days[0], days[-1]) if days else None


def identify(files: dict[str, Optional[Path]], p: Optional[pd.DataFrame] = None, c: Optional[pd.DataFrame] = None) -> dict:
    """Which of MetaCyc's releases (`RELEASES`) its SmartTables are of (as `resolve` gives
    them; `p`, `c`: its pathways and compounds, if read already), told by how many pathways
    (not counting super-pathways) and compounds they have, against how many its notes say each
    has: the release they have as many as (as `NEAR`), else the last they have more than,
    if they have fewer than the next (as metacyc.org had it before the next: 27.1's, in
    January 2024, 5 pathways and 17 compounds more than its notes say).

    Returns:
        ``version`` (None if it cannot be told); ``pathways``, ``super_pathways`` and
        ``compounds``, as counted; ``dated``: when the files of its tables were last
        changed (first, last), if they are; and ``basis``: how it was told, or why it could not
        be, as said.
    """
    p = read_table(files["pathways"]) if p is None else p
    c = read_table(files["compounds"]) if c is None else c
    told = {"version": None, "pathways": None, "super_pathways": None, "compounds": len(c), "dated": _dated(files, REQUIRED)}
    if "Ontology - pathway type" not in p:
        return {**told, "basis": f"{files['pathways'].name} has no column 'Ontology - pathway type', to tell its super-pathways by"}
    sup = int(p["Ontology - pathway type"].map(lambda s: "Super-Pathways" in split(s)).sum())
    n = (len(p) - sup, len(c))
    told.update(pathways=n[0], super_pathways=sup)
    # for each count, the last release it is as many as, or more than (-1: none), and whether as many
    at = [max((i for i, r in enumerate(RELEASES) if r[2 + k] <= n[k] + NEAR[k]), default=-1) for k in (0, 1)]
    same = [at[k] >= 0 and abs(RELEASES[at[k]][2 + k] - n[k]) <= NEAR[k] for k in (0, 1)]
    supers = f"not counting {sup:,} super-pathway{'s' if sup != 1 else ''}"
    has = f"Its {n[0]:,} pathways ({supers}) and {n[1]:,} compounds"
    of = lambda r: f"{r[0]} had ({r[2]:,} and {r[3]:,}; released {r[1]})"  # noqa: E731

    def stands(k: int) -> str:  # how count k stands to the releases'
        if at[k] < 0:
            return f"fewer than any release's ({RELEASES[0][0]}: {RELEASES[0][2 + k]:,})"
        r = RELEASES[at[k]]
        return f"{'as many as' if same[k] else 'more than'} {r[0]}'s ({r[2 + k]:,})"

    if at[0] != at[1]:
        basis = (f"Its {n[0]:,} pathways ({supers}) are {stands(0)}, but its {n[1]:,} compounds "
                 f"{stands(1)}, as MetaCyc's notes count them")
    elif at[0] < 0:
        basis = f"{has} are fewer than MetaCyc {of(RELEASES[0])}, the oldest release known here"
    elif all(same):
        told["version"] = RELEASES[at[0]][0]
        basis = f"{has} are as many as MetaCyc {of(RELEASES[at[0]])}, as its notes count them"
    elif at[0] == len(RELEASES) - 1:
        basis = f"{has} are more than MetaCyc {of(RELEASES[-1])}, the newest release known here"
    else:
        r, nxt = RELEASES[at[0]], RELEASES[at[0] + 1]
        told["version"] = r[0]
        basis = (f"{has} are more than MetaCyc {of(r)}, and fewer than {of(nxt)}, as their notes count "
                 f"them: so {r[0]}, as metacyc.org had it before {nxt[0]}")

    def when(dated: tuple[str, str]) -> str:
        out = list(dict.fromkeys(filter(None, (_newest_on(dated[0]), _newest_on(dated[1])))))
        return (f"{dated[0]}{f' to {dated[1]}' if dated[1] != dated[0] else ''}, when MetaCyc's newest release was "
                f"{' then '.join(out) or f'older than {RELEASES[0][0]}'}")

    if told["dated"]:
        basis += f". The files of its tables were last changed {when(told['dated'])}"
        others = _dated(files, [r for r in FILES if r not in REQUIRED])
        if others and {_newest_on(d) for d in others} != {_newest_on(d) for d in told["dated"]}:
            basis += f"; the others it reads, {when(others)}"
    return {**told, "basis": basis}


def contradicts(given: Optional[str], told: dict) -> Optional[str]:
    """Why a version given is not the one the SmartTables are of (`identify`): if it is
    another release's (of `RELEASES`) than they are told to be of, as said. None if not, or if
    which they are of cannot be told."""
    if not given or not told.get("version") or given == told["version"] or given not in {r[0] for r in RELEASES}:
        return None
    return f"these are MetaCyc {told['version']}'s SmartTables, not {given}'s. {told['basis']}"


def build(raw: Path, out: Path, version: Optional[str] = None, overrides: Optional[dict[str, str]] = None,
          log=print) -> dict:
    """Read MetaCyc's SmartTables in `raw` into the bundle `out` (replaced if it exists), of
    MetaCyc `version` (else as told from them: `identify`).

    Returns:
        its ``about`` (as stored), with ``counts`` of what it has, and ``identified``: which
        release they were told to be of (`identify`).

    Raises:
        BuildError: if they are not as it reads them (VersionError: if `version` is another
            release's than they are of).
    """
    files = resolve(raw, overrides)
    if found := problems(files, raw):
        raise BuildError(f"the SmartTables in {raw} are not as expected:\n" + "\n".join(f"- {x}" for x in found))
    names: dict[str, str] = {}

    # ------------------------------------------------------------- compounds
    c, cn = read_table(files["compounds"]), read_table(files["compound_names"])
    if len(c) != len(cn):
        raise BuildError(f"{files['compounds'].name} and {files['compound_names'].name} differ in rows: give the two exports of one table")
    _frames(c["Compound"], "compounds", files["compounds"])
    compounds = []
    for (_, r), (_, n) in zip(c.iterrows(), cn.iterrows()):
        cid = re.sub(r"\D", "", _link_text(r.get("PubChem", "")))
        names[r["Compound"]] = plain(r["Common-Name"]) or r["Compound"]
        parents = split(r.get("Ontology - direct parents of entity"))
        _named(parents, split(n.get("Ontology - direct parents of entity")), names)
        compounds.append((r["Compound"], names[r["Compound"]], r.get("Chemical Formula") or None, r.get("SMILES") or None,
                          r.get("InChI") or None, (r.get("InChI-Key") or "").removeprefix("InChIKey=") or None,
                          int(cid) if cid else None, _link_text(r.get("Kegg", "")) or None, json.dumps(parents),
                          json.dumps(split(r.get("Pathways of compound"))), json.dumps(split(r.get("General Pathways of compound")))))
    log(f"compounds: {len(compounds):,} ({files['compounds'].name})")

    # -------------------------------------------------------------- pathways
    p, pn = read_table(files["pathways"]), read_table(files["pathway_names"])
    if len(p) != len(pn) or not (p["Object ID"].to_numpy() == pn["Object ID"].to_numpy()).all():
        raise BuildError(f"{files['pathways'].name} and {files['pathway_names'].name} differ in rows: give the two exports of one table")
    _frames(p["Reactions of pathway"], "reactions", files["pathways"])
    classes: dict[str, str] = {}
    species_names: dict[str, str] = {}
    pw: dict[str, dict] = {}
    for (_, r), (_, n) in zip(p.iterrows(), pn.iterrows()):
        pid = r["Object ID"]
        names[pid] = plain(r["Common-Name"]) or pid
        for col in ("Products of pathway", "Reactants of pathway", "Primary-Products", "Primary-Reactants"):
            _named(split(r.get(col)), split(n.get(col)), names)
        _named(split(r.get("Ontology - pathway type")), split(n.get("Ontology - pathway type")), classes)
        _named(split(r.get("Species")), split(n.get("Species")), species_names)
        pw[pid] = {"name": names[pid], "types": split(r.get("Ontology - pathway type")), "species": list(dict.fromkeys(split(r.get("Species")))),
                   "reactions": list(dict.fromkeys(split(r.get("Reactions of pathway")))),
                   "reactants": split(r.get("Reactants of pathway")), "products": split(r.get("Products of pathway")),
                   "primary_reactants": split(r.get("Primary-Reactants")), "primary_products": split(r.get("Primary-Products")),
                   "taxonomic_range": [], "engineered": None, "summary": None, "super": [], "subs": []}
    # superpathways and the pathways all of whose reactions are in them
    rsets = {k: set(v["reactions"]) for k, v in pw.items()}
    for s, v in pw.items():
        if "Super-Pathways" not in v["types"] or not rsets[s]:
            continue
        for q, rs in rsets.items():
            if q != s and rs and rs <= rsets[s] and "Super-Pathways" not in pw[q]["types"]:
                v["subs"].append(q)
                pw[q]["super"].append(s)
    listed = {q for row in compounds for q in json.loads(row[9])}
    if listed and not listed & set(pw):
        raise BuildError(f"none of the pathways of {files['compounds'].name} is in {files['pathways'].name}: give the two of one "
                         "MetaCyc, exported by frame IDs")
    log(f"pathways: {len(pw):,} ({files['pathways'].name})")
    told = identify(files, p, c)
    if why := contradicts(version, told):
        raise VersionError(why)
    version = version or told["version"]
    log(f"version: {version} ({'as given' if version != told['version'] else 'told from them'}: {told['basis']})" if version
        else f"version: not given, nor told from them ({told['basis']})")

    # ------------------------------------------------------------- reactions
    x, xn = read_table(files["reactions"]), read_table(files["reaction_names"])
    if len(x) != len(xn) or not (x["Object ID"].to_numpy() == xn["Object ID"].to_numpy()).all():
        raise BuildError(f"{files['reactions'].name} and {files['reaction_names'].name} differ in rows: give the two exports of one table")
    _frames(x["Reactants of reaction"], "reactants", files["reactions"])
    reactions = []
    for (_, r), (_, n) in zip(x.iterrows(), xn.iterrows()):
        for col in ("Reactants of reaction", "Products of reaction"):
            _named(split(r.get(col)), split(n.get(col)), names)
        equation = plain(n["Reaction"])
        reactions.append((r["Object ID"], equation, json.dumps([e.removeprefix("EC-") for e in split(r.get("EC-Number"))]),
                          _link_text(r.get("KEGG reaction", "")) or None, direction(equation),
                          json.dumps(split(r.get("Reactants of reaction"))), json.dumps(split(r.get("Products of reaction"))),
                          json.dumps(list(dict.fromkeys(split(r.get("In-Pathway")))))))
    log(f"reactions: {len(reactions):,} ({files['reactions'].name})")

    # ------------------------------------------- pathways' summaries and more
    if files["summaries"]:
        header, recs = _records(files["summaries"], set(pw))
        at = header.index("Summary")
        for pid, fields in recs.items():
            if len(fields) > at:
                pw[pid]["summary"] = _summary("\t".join(fields[at:]), {**species_names, **classes, **names}) or None
        log(f"summaries: {len(recs):,} ({files['summaries'].name})")
    if files["pathway_extra"]:
        header, recs = _records(files["pathway_extra"], set(pw))
        tr, en = header.index("Taxonomic-Range"), header.index("Engineered?")
        for pid, fields in recs.items():
            if len(fields) > max(tr, en):
                pw[pid]["taxonomic_range"] = [plain(t) for t in split(fields[tr])]
                pw[pid]["engineered"] = {"T": 1, "NIL": 0}.get(fields[en].strip().upper())
        log(f"taxonomic ranges: {sum(bool(v['taxonomic_range']) for v in pw.values()):,} ({files['pathway_extra'].name})")
    class_parents = _ontology_parents(files["pathway_ontology"]) if files["pathway_ontology"] else {}

    # -------------------------------------------------------------- organisms
    species = {}
    if files["species"]:
        s = read_table(files["species"])
        for _, r in s.iterrows():
            species[r["Object ID"]] = (r["Object ID"], plain(r.get("Common-Name") or r.get("Organisms")), r.get("Strain-Name") or None,
                                       json.dumps(list(dict.fromkeys(plain(y) for y in split(r.get("Synonyms"))))))
    for tid, name in species_names.items():
        species.setdefault(tid, (tid, name, None, "[]"))
    genomes = []
    if files["genomes"]:
        g = read_table(files["genomes"])
        by_name = {v[1]: k for k, v in species.items() if v[1]}

        def species_of(organism: str) -> Optional[str]:
            words = organism.split()
            for k in range(min(len(words), 8), 0, -1):
                if (tid := by_name.get(" ".join(words[:k]))) is not None:
                    return tid
            return None

        for _, r in g.iterrows():
            organism = plain(r.get("Organism name"))
            genomes.append((r[g.columns[0]], organism, species_of(organism),
                            *(" · ".join(dict.fromkeys(plain(v) for v in split(r.get(col)))) or None
                              for col in ("Environment", "Depth-or-Altitude", "Health-or-Disease-State", "Host", "Pathogenicity",
                                          "Relationship-to-Oxygen", "Temperature-Range", "Trophic-Level", "Strain-Name"))))
    log(f"species: {len(species):,}; genome databases: {len(genomes):,} ({sum(g[2] is not None for g in genomes):,} of a species of MetaCyc's)")

    # ------------------------------------------------------------------ write
    about = {"kind": "metacyc", "version": version, "identified": told, "built_at": dt.datetime.now().isoformat(timespec="seconds"),
             "sources": {role: {"path": str(f), "mtime": dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")}
                         for role, f in files.items() if f},
             "counts": {"compounds": len(compounds), "pathways": len(pw), "reactions": len(reactions), "species": len(species),
                        "genomes": len(genomes)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO about VALUES (?, ?)", [(k, json.dumps(v)) for k, v in about.items()])
        con.executemany("INSERT INTO compounds VALUES (?,?,?,?,?,?,?,?,?,?,?)", compounds)
        con.executemany("INSERT OR IGNORE INTO names VALUES (?, ?)", names.items())
        con.executemany("INSERT INTO pathway_classes VALUES (?, ?, ?)", [(k, v, class_parents.get(v)) for k, v in classes.items()])
        con.executemany("INSERT OR IGNORE INTO pathway_ontology VALUES (?, ?)", class_parents.items())
        con.executemany("INSERT INTO pathways VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            (k, v["name"], *(json.dumps(v[f]) for f in ("types", "species", "reactions", "reactants", "products", "primary_reactants",
                                                         "primary_products", "taxonomic_range")),
             v["engineered"], v["summary"], json.dumps(v["super"]), json.dumps(v["subs"])) for k, v in pw.items()])
        con.executemany("INSERT INTO reactions VALUES (?,?,?,?,?,?,?,?)", reactions)
        con.executemany("INSERT INTO species VALUES (?,?,?,?)", species.values())
        con.executemany("INSERT OR IGNORE INTO genomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", genomes)
        con.commit()
    finally:
        con.close()
    tmp.replace(out)
    log(f"written: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return about


# --------------------------------------------------------------------- registering

VIEWS = [
    {"view": "pathways", "title": "MetaCyc{v}: compounds by the pathways they are in", "kind": "detect",
     "description": "Every compound of MetaCyc in a pathway, by the pathways it is in: 1 itself, 0.5 only through a class it "
                    "is of (e.g. an alcohol), if it is in none itself. The odor table's chemicals come first, and can be shown "
                    "alone. Click a cell for the pathway.",
     "value_label": "in the pathway", "row_label": "chemical", "col_label": "pathway"},
    {"view": "lca", "title": "MetaCyc{v}: pathway LCA distance between odors", "kind": "sequential",
     "description": "Between two chemicals of the odor table, the fewest steps from a common ancestor to both in MetaCyc's "
                    "network of compounds and reactions (as in O'Connell's thesis, ch. 2). Click a cell for the paths.",
     "value_label": "steps to their lowest common ancestor", "row_label": "chemical", "col_label": "chemical"},
]


def views(bundle: Path, version: Optional[str]) -> list[dict]:
    """The bundle's two reference datasets, as the Admin tab adds them (POST /api/admin/references)."""
    v = f" {version}" if version else ""
    return [{"path": str(bundle.resolve()), "title": spec["title"].format(v=v), "description": spec["description"],
             "source": f"MetaCyc{v} (metacyc.org), SmartTables read by app/metacyc_build.py", "kind": spec["kind"],
             "chemical_axis": "detect", "value_label": spec["value_label"], "row_label": spec["row_label"],
             "col_label": spec["col_label"], "fields": {"layout": "metacyc", "view": spec["view"]}} for spec in VIEWS]


def register(server: str, bundle: Path, version: Optional[str], log=print) -> int:
    """Add the bundle's reference datasets to the app at `server` (its Admin tab's API): 0 if all were added."""
    failed = 0
    for body in views(bundle, version):
        req = urllib.request.Request(server.rstrip("/") + "/api/admin/references", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                res = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            res = json.loads(e.read() or b"{}")
        except urllib.error.URLError as e:
            log(f"could not reach {server}: {e.reason}")
            return 1
        if not res.get("ok"):
            failed += 1
            log(f"not added: {body['title']}: {'; '.join(res.get('errors') or [res.get('error') or 'unknown error'])}")
            continue
        log(f"added: {body['title']} (#reference/{res['id']}, {res['state']}{': ' + res['error'] if res.get('error') else ''})")
        if res.get("restart"):
            log("  the server reads it once it restarts:\n" + "\n".join(f"  - {re.sub(r'<[^>]+>', '', s)}" for s in res.get("steps") or []))
    return 1 if failed else 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.metacyc_build", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path, help="the folder of MetaCyc's SmartTables (e.g. .../BioCyc/raw)")
    ap.add_argument("out", type=Path, help="the bundle to write (e.g. metacyc-27.1.sqlite)")
    ap.add_argument("--version", help="MetaCyc's version the tables are of, shown in titles (default: told from them, by how many "
                    "pathways and compounds they have: refused if another release's)")
    ap.add_argument("--file", action="append", default=[], metavar="ROLE=PATH",
                    help=f"a file to read for a role instead (relative to the raw folder, or absolute): {', '.join(FILES)}")
    ap.add_argument("--register", metavar="URL", help="then add its two reference datasets to the app at URL (e.g. http://localhost:8000)")
    args = ap.parse_args(list(argv) if argv is not None else None)
    overrides = {}
    for f in args.file:
        role, _, path = f.partition("=")
        if role not in FILES or not path:
            ap.error(f"--file {f}: give ROLE=PATH, ROLE one of {', '.join(FILES)}")
        overrides[role] = path
    try:
        about = build(args.raw, args.out, args.version or None, overrides)
    except BuildError as e:
        print(f"error: {e}", file=sys.stderr)
        if isinstance(e, VersionError):
            print("(leave out --version, and it is told from them)", file=sys.stderr)
        elif not overrides and "missing:" in str(e):
            print("(a file elsewhere, or named otherwise: give it with --file ROLE=PATH)", file=sys.stderr)
        return 2
    return register(args.register, args.out, about["version"]) if args.register else 0


if __name__ == "__main__":
    sys.exit(main())
