"""Builds a LOTUS bundle: LOTUS's structure–organism–paper records (its frozen export of
Wikidata, ``230106_frozen.csv``) with what its metadata says of each structure, organism
and paper (``structure_metadata.tsv``, ``organism_metadata.tsv``, ``reference_metadata.tsv``,
from Zenodo), and LOTUS IDs (LTS...) from its 2021 release, read into one SQLite file that
the app reads as a reference dataset (see lotus.py and reference_data/README.md). Run:

  python -m app.lotus_build <LOTUS folder> <bundle.sqlite> [--register http://<server>:<port>]

``--register`` adds its reference dataset to the app (as the Admin tab does): the odor
table's chemicals by the families of organisms they are reported in.

The files are found in the folder as laid out in ``Scraped-Databases-Copy/LOTUS/raw``
(`FILES`); which are read can be changed with ``--file <role>=<path>``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import IO, Iterable, Optional

import pandas as pd

from .lotus import (BUNDLE_KIND, DEFAULT_OPTIONS, FAMILY_NAMES, SOURCES, describe, kingdom_of, no_rs_inchi, options,
                    view_title, value_label)
from .metacyc_build import plain

# role -> where it is looked for in the folder (glob patterns: the last match, by name, of the first that matches)
FILES = {
    "pairs": ["*_frozen/*_frozen.csv", "*_frozen/*_frozen.csv.gz", "*frozen*.csv", "*frozen*.csv.gz"],
    "structures": ["metadata_*/*/structure_metadata.tsv", "metadata_*/*/structure_metadata.tsv.gz", "**/structure_metadata.tsv*"],
    "organisms": ["metadata_*/*/organism_metadata.tsv", "metadata_*/*/organism_metadata.tsv.gz", "**/organism_metadata.tsv*"],
    "references": ["metadata_*/*/reference_metadata.tsv", "metadata_*/*/reference_metadata.tsv.gz", "**/reference_metadata.tsv*"],
    "lts": ["LOTUS_DB_LATEST/LOTUS_*_simple.sdf", "**/LOTUS_*.sdf"],  # (optional) LOTUS IDs, by InChIKey
}
REQUIRED = ("pairs", "structures", "organisms", "references")
CHUNK = 500_000  # lines of the metadata read at once

SCHEMA = """
CREATE TABLE about (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE structures (id INTEGER PRIMARY KEY, inchikey TEXT NOT NULL UNIQUE, inchi TEXT, smiles TEXT,
                         key_no_rs TEXT,  -- InChIKey of its InChI without tetrahedral stereo (/t /m /s): the same but for R/S
                         skeleton TEXT,   -- its InChIKey's first block: the same but for any stereo
                         formula TEXT, exact_mass REAL, xlogp REAL, stereocenters INTEGER, stereocenters_unspecified INTEGER,
                         cid INTEGER, name TEXT, iupac_name TEXT, wikidata TEXT, lts TEXT);
CREATE TABLE organisms (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, wikidata TEXT);
-- an organism's lineage in each taxonomy source that has it (the first row of the metadata for it there)
-- (kingdoms as GBIF names them, lotus.kingdom_of; families of two names as GBIF does, lotus.FAMILY_NAMES)
CREATE TABLE taxonomy (organism INTEGER NOT NULL, source TEXT NOT NULL, source_id TEXT, rank TEXT, kingdom TEXT,
                       phylum TEXT, class TEXT, "order" TEXT, family TEXT, genus TEXT, species TEXT, variety TEXT,
                       lineage TEXT, PRIMARY KEY (organism, source));
CREATE TABLE refs (id INTEGER PRIMARY KEY, doi TEXT NOT NULL UNIQUE, wikidata TEXT, title TEXT, journal TEXT, date TEXT,
                   author TEXT, pmid TEXT, pmcid TEXT);
CREATE TABLE pairs (structure INTEGER NOT NULL, organism INTEGER NOT NULL, ref INTEGER NOT NULL, validated INTEGER NOT NULL,
                    PRIMARY KEY (structure, organism, ref)) WITHOUT ROWID;
"""
INDEXES = """
CREATE INDEX structures_no_rs ON structures (key_no_rs);
CREATE INDEX structures_skeleton ON structures (skeleton);
CREATE INDEX structures_cid ON structures (cid);
CREATE INDEX structures_lts ON structures (lts);
CREATE INDEX pairs_organism ON pairs (organism);
"""
RANK_COLUMNS = {"kingdom": "organismCleaned_dbTaxo_1kingdom", "phylum": "organismCleaned_dbTaxo_2phylum",
                "class": "organismCleaned_dbTaxo_3class", "order": "organismCleaned_dbTaxo_4order",
                "family": "organismCleaned_dbTaxo_5family", "genus": "organismCleaned_dbTaxo_6genus",
                "species": "organismCleaned_dbTaxo_7species", "variety": "organismCleaned_dbTaxo_8variety"}


def open_peeled(path: Path) -> IO[bytes]:
    """A file opened to read, gunzipped as many times as it was gzipped (LOTUS's
    ``230106_frozen.csv.gz`` is a gzip of a gzip)."""
    f: IO[bytes] = io.BufferedReader(open(path, "rb"))  # type: ignore[arg-type]
    while f.peek(2)[:2] == b"\x1f\x8b":
        f = io.BufferedReader(gzip.GzipFile(fileobj=f))  # type: ignore[arg-type]
    return f


def _chunks(path: Path, usecols: list[str], sep: str = "\t") -> Iterable[pd.DataFrame]:
    with open_peeled(path) as f:
        yield from pd.read_csv(f, sep=sep, dtype=str, usecols=usecols, keep_default_na=False, na_values=[""],
                               chunksize=CHUNK, quoting=3 if sep == "\t" else 0, encoding_errors="replace")


def resolve(folder: Path, overrides: Optional[dict[str, str]] = None) -> dict[str, Optional[Path]]:
    """The file read for each role (`FILES`): as given in `overrides`, else the first found."""
    out: dict[str, Optional[Path]] = {}
    for role, patterns in FILES.items():
        given = (overrides or {}).get(role)
        if given:
            p = Path(given)
            out[role] = p if p.is_absolute() else folder / p
            continue
        out[role] = None
        for pat in patterns:
            found = sorted(p for p in folder.glob(pat) if p.is_file())
            if found:
                out[role] = found[-1]
                break
    missing = [r for r in REQUIRED if not out[r] or not out[r].is_file()]
    if missing:
        raise SystemExit(f"not found in {folder}: {', '.join(f'{r} ({FILES[r][0]})' for r in missing)} "
                         f"(give it with --file {missing[0]}=<path>)")
    return out


def release(files: dict[str, Optional[Path]]) -> dict[str, Optional[str]]:
    """The dates of what was read, from the files' names (as LOTUS's are): of the records
    (``230106_frozen.csv``: 2023-01-06), of the metadata (``metadata_2022-09-16/``) and of the
    release LOTUS IDs are from (``LOTUS_2021_03_simple.sdf``: 2021-03)."""
    out: dict[str, Optional[str]] = {"pairs": None, "metadata": None, "lts": None}
    if files["pairs"] and (m := re.search(r"(\d{2})(\d{2})(\d{2})_frozen", files["pairs"].name)):
        out["pairs"] = f"20{m[1]}-{m[2]}-{m[3]}"
    if files["structures"] and (m := re.search(r"(\d{4}-\d{2}-\d{2})", str(files["structures"].parent.parent.name))):
        out["metadata"] = m[1]
    if files["lts"] and (m := re.search(r"(\d{4})_(\d{2})", files["lts"].name)):
        out["lts"] = f"{m[1]}-{m[2]}"
    return out


def _notes(files: dict[str, Optional[Path]]) -> dict[str, str]:
    """``notes.txt`` beside any file read (where it came from, when), by folder."""
    out = {}
    for f in files.values():
        for d in (f.parent, f.parent.parent) if f else ():
            n = d / "notes.txt"
            if n.is_file() and str(d) not in out:
                out[str(d)] = n.read_text(errors="replace").strip()
    return out


def read_lts(path: Path, log=print) -> dict[str, str]:
    """LOTUS IDs by InChIKey: from an SDF of LOTUS (``lotus_id`` and ``inchikey`` fields of
    each record), or a table with ``lotus_id`` and ``inchikey`` columns (CSV, TSV, parquet)."""
    out: dict[str, str] = {}
    if path.suffix.lower() == ".sdf":
        rec: dict[str, str] = {}
        field = None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if field is not None:
                    rec[field] = line.strip()
                    field = None
                elif line.startswith("> <"):
                    name = line[3:line.find(">", 3)]
                    field = name if name in ("lotus_id", "inchikey") else None
                elif line.startswith("$$$$"):
                    if rec.get("lotus_id") and rec.get("inchikey"):
                        out.setdefault(rec["inchikey"], rec["lotus_id"])
                    rec = {}
    else:
        df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else \
            pd.read_csv(path, sep="\t" if path.suffix.lower() in (".tsv", ".tab") else ",", dtype=str)
        for k, v in zip(df["inchikey"], df["lotus_id"]):
            if isinstance(k, str) and isinstance(v, str):
                out.setdefault(k, v)
    log(f"LOTUS IDs: {len(out):,} ({path.name})")
    return out


def plain_title(s: Optional[str]) -> Optional[str]:
    """A paper's title (or journal) as plain text: its markup (``<i>``, ``<sub>``, ..., as
    Crossref gives some) dropped, words it ran together kept apart, sub- and superscripts as
    Unicode where they can be, entities decoded."""
    if not s:
        return s
    s = re.sub(r"(?<=[\w,.;:)])<(i|b|em|strong|scp|span)\b[^>]*>", r" <\1>", s, flags=re.I)
    s = re.sub(r"</(i|b|em|strong|scp|span)>(?=[\w(])", r"</\1> ", s, flags=re.I)
    return re.sub(r"\s+", " ", plain(s)).strip() or None


def _qid(url) -> Optional[str]:
    """A Wikidata entity's ID (``Q311176``) from its URL."""
    if not isinstance(url, str):
        return None
    m = re.search(r"(Q\d+)\s*$", url)
    return m[1] if m else None


def _num(v, cast=float):
    try:
        return cast(float(v)) if isinstance(v, str) and v.strip() else None
    except ValueError:
        return None


def _inchikey(inchi: Optional[str]) -> Optional[str]:
    if not inchi:
        return None
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        return Chem.InchiToInchiKey(inchi) or None
    except Exception:
        return None


def build(folder: Path, out: Path, overrides: Optional[dict[str, str]] = None, log=print) -> dict:
    """Read LOTUS's files in `folder` into the bundle `out` (replaced if it exists).

    Returns:
        its ``about`` (as stored), with ``counts`` of what it has.
    """
    files = resolve(folder, overrides)

    # --------------------------------------------------------------- the records
    with open_peeled(files["pairs"]) as f:
        fr = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
    need = {"structure_inchikey", "organism_name", "reference_doi"}
    if not need <= set(fr.columns):
        raise SystemExit(f"{files['pairs'].name} has no {', '.join(sorted(need - set(fr.columns)))} column")
    n_lines = len(fr)
    fr = fr.dropna(subset=list(need)).drop_duplicates(subset=["structure_inchikey", "organism_name", "reference_doi"])
    log(f"records: {len(fr):,} structure–organism–paper ({n_lines:,} lines, {files['pairs'].name})")
    keys = set(fr["structure_inchikey"])
    dois = set(fr["reference_doi"])
    org_names = set(fr["organism_name"])

    # -------------------------------------------------------------- structures
    cols = {"structureCleanedInchikey": "inchikey", "structureCleanedInchi": "inchi", "structureCleanedSmiles": "smiles",
            "structureCleaned_molecularFormula": "formula", "structureCleaned_exactMass": "exact_mass",
            "structureCleaned_xlogp": "xlogp", "structureCleaned_stereocenters_total": "stereocenters",
            "structureCleaned_stereocenters_unspecified": "stereocenters_unspecified", "structureCleaned_cid": "cid",
            "structureCleaned_nameTraditional": "name", "structureCleaned_nameIupac": "iupac_name"}
    parts = []
    for ch in _chunks(files["structures"], list(cols)):
        parts.append(ch[ch["structureCleanedInchikey"].isin(keys)].rename(columns=cols))
    st = pd.concat(parts).groupby("inchikey", sort=False).first().to_dict("index")  # (the first of each field given, of its lines)
    wd = fr.dropna(subset=["structure_wikidata"]).drop_duplicates("structure_inchikey").set_index("structure_inchikey")["structure_wikidata"] \
        if "structure_wikidata" in fr else pd.Series(dtype=str)
    lts = read_lts(files["lts"], log) if files["lts"] else {}
    structures = []
    for i, k in enumerate(sorted(keys), start=1):
        r = st.get(k) or {}
        get = (lambda c: r[c] if isinstance(r.get(c), str) else None)  # noqa: E731
        inchi = get("inchi")
        structures.append((i, k, inchi, get("smiles"), _inchikey(no_rs_inchi(inchi)) or k, k[:14], get("formula"),
                           _num(get("exact_mass")), _num(get("xlogp")), _num(get("stereocenters"), int),
                           _num(get("stereocenters_unspecified"), int), _num(get("cid"), int), get("name"), get("iupac_name"),
                           _qid(wd.get(k)), lts.get(k)))
    sid = {s[1]: s[0] for s in structures}
    log(f"structures: {len(structures):,} ({sum(k in st for k in keys):,} in {files['structures'].name}, "
        f"{sum(s[11] is not None for s in structures):,} with a CID, {sum(s[15] is not None for s in structures):,} with a LOTUS ID)")

    # --------------------------------------------------------------- organisms
    owd = fr.dropna(subset=["organism_wikidata"]).drop_duplicates("organism_name").set_index("organism_name")["organism_wikidata"] \
        if "organism_wikidata" in fr else pd.Series(dtype=str)
    organisms = [(i, n, _qid(owd.get(n))) for i, n in enumerate(sorted(org_names), start=1)]
    oid = {o[1]: o[0] for o in organisms}
    ocols = ["organismCleaned", "organismCleaned_id", "organismCleaned_rank", "organismCleaned_dbTaxo",
             "organismCleaned_dbTaxoTaxonomy", *RANK_COLUMNS.values()]
    taxonomy: dict[tuple[int, str], tuple] = {}
    for ch in _chunks(files["organisms"], ocols):
        ch = ch[ch["organismCleaned"].isin(org_names) & ch["organismCleaned_dbTaxo"].notna()]
        for r in ch.itertuples(index=False):
            key = (oid[r[0]], r[3])
            if key in taxonomy:
                continue
            ranks = {k: (v if isinstance(v, str) else None) for k, v in zip(RANK_COLUMNS, r[5:])}
            lineage = r[4] if isinstance(r[4], str) else ""
            ranks["kingdom"] = kingdom_of(r[3], ranks["kingdom"], lineage)
            ranks["family"] = FAMILY_NAMES.get(ranks["family"], ranks["family"])
            taxonomy[key] = (*key, r[1] if isinstance(r[1], str) else None, r[2] if isinstance(r[2], str) else None,
                             *ranks.values(), lineage or None)
    coverage = {}
    for (_, source) in taxonomy:
        coverage[source] = coverage.get(source, 0) + 1
    coverage = dict(sorted(coverage.items(), key=lambda kv: (SOURCES.index(kv[0]) if kv[0] in SOURCES else len(SOURCES), kv[0])))
    with_any = len({o for o, _ in taxonomy})
    log(f"organisms: {len(organisms):,}, {with_any:,} with a lineage in {files['organisms'].name} "
        f"({', '.join(f'{k}: {v:,}' for k, v in list(coverage.items())[:4])}, ...)")

    # ------------------------------------------------------------------ papers
    rcols = {"referenceCleanedDoi": "doi", "referenceCleanedTitle": "title", "referenceCleaned_journal": "journal",
             "referenceCleaned_date": "date", "referenceCleaned_author": "author", "referenceCleanedPmid": "pmid",
             "referenceCleanedPmcid": "pmcid"}
    parts = []
    for ch in _chunks(files["references"], list(rcols)):
        ch = ch[ch["referenceCleanedDoi"].isin(dois)].rename(columns=rcols)
        parts.append(ch.groupby("doi", sort=False).first())
    rf = pd.concat(parts).groupby(level=0, sort=False).first().to_dict("index") if parts else {}
    rwd = fr.dropna(subset=["reference_wikidata"]).drop_duplicates("reference_doi").set_index("reference_doi")["reference_wikidata"] \
        if "reference_wikidata" in fr else pd.Series(dtype=str)
    refs = []
    for i, d in enumerate(sorted(dois), start=1):
        r = rf.get(d) or {}
        get = (lambda c: r[c] if isinstance(r.get(c), str) else None)  # noqa: E731
        refs.append((i, d, _qid(rwd.get(d)), plain_title(get("title")), plain_title(get("journal")), get("date"), get("author"),
                     get("pmid"), get("pmcid")))
    rid = {r[1]: r[0] for r in refs}
    log(f"papers: {len(refs):,} ({sum(r[3] is not None for r in refs):,} with a title, {files['references'].name})")

    validated = fr["manual_validation"].fillna("").str.upper().eq("Y") if "manual_validation" in fr else pd.Series(False, index=fr.index)
    pairs = list(zip(fr["structure_inchikey"].map(sid), fr["organism_name"].map(oid), fr["reference_doi"].map(rid),
                     validated.astype(int)))

    # ------------------------------------------------------------------- write
    rel = release(files)
    about = {
        "kind": BUNDLE_KIND, "release": rel, "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "sources": {role: {"path": str(f), "mtime": dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")}
                    for role, f in files.items() if f},
        "notes": _notes(files),
        "counts": {"records": len(pairs), "pairs": int(fr.drop_duplicates(["structure_inchikey", "organism_name"]).shape[0]),
                   "structures": len(structures), "organisms": len(organisms), "papers": len(refs),
                   "with_cid": sum(s[11] is not None for s in structures), "with_lts": sum(s[15] is not None for s in structures),
                   "with_lineage": with_any, "validated": int(validated.sum())},
        "taxonomy_sources": coverage,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO about VALUES (?, ?)", [(k, json.dumps(v)) for k, v in about.items()])
        con.executemany(f"INSERT INTO structures VALUES ({','.join('?' * 16)})", structures)
        con.executemany("INSERT INTO organisms VALUES (?, ?, ?)", organisms)
        con.executemany(f"INSERT INTO taxonomy VALUES ({','.join('?' * 13)})", taxonomy.values())
        con.executemany(f"INSERT INTO refs VALUES ({','.join('?' * 9)})", refs)
        con.executemany("INSERT INTO pairs VALUES (?, ?, ?, ?)", pairs)
        con.executescript(INDEXES)
        con.commit()
    finally:
        con.close()
    tmp.replace(out)
    log(f"written: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return about


# --------------------------------------------------------------------- registering


def register(server: str, bundle: Path, about: dict, log=print) -> int:
    """Add the bundle's reference dataset (its default view, `DEFAULT_OPTIONS`) to the app at
    `server` (its Admin tab's API): 0 if it was added."""
    import urllib.error
    import urllib.request

    o = options(DEFAULT_OPTIONS)
    rel = about.get("release") or {}
    body = {"path": str(bundle.resolve()), "title": view_title(o, rel), "description": describe(o)["summary"],
            "source": "LOTUS (lotus.naturalproducts.net; Rutz et al., eLife 2022): its frozen Wikidata export"
                      f"{' of ' + rel['pairs'] if rel.get('pairs') else ''} and metadata, read by app/lotus_build.py",
            "kind": "detect", "chemical_axis": "rows", "value_label": value_label(o), "row_label": "chemical",
            "col_label": o["level"], "fields": {"layout": "lotus", **o}}
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
        log(f"not added: {body['title']}: {'; '.join(res.get('errors') or [res.get('error') or 'unknown error'])}")
        return 1
    log(f"added: {body['title']} (#reference/{res['id']}, {res['state']}{': ' + res['error'] if res.get('error') else ''})")
    if res.get("restart"):
        log("  the server reads it once it restarts:\n" + "\n".join(f"  - {re.sub(r'<[^>]+>', '', s)}" for s in res.get("steps") or []))
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.lotus_build", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="the folder of LOTUS's files (e.g. .../Scraped-Databases-Copy/LOTUS/raw)")
    ap.add_argument("out", type=Path, help="the bundle to write (e.g. lotus-2023-01-06.sqlite)")
    ap.add_argument("--file", action="append", default=[], metavar="ROLE=PATH",
                    help=f"a file to read for a role instead (relative to the folder, or absolute): {', '.join(FILES)}")
    ap.add_argument("--register", metavar="URL", help="then add its reference dataset to the app at URL")
    args = ap.parse_args(list(argv) if argv is not None else None)
    overrides = {}
    for f in args.file:
        role, _, path = f.partition("=")
        if role not in FILES or not path:
            ap.error(f"--file {f}: give ROLE=PATH, ROLE one of {', '.join(FILES)}")
        overrides[role] = path
    about = build(args.folder, args.out, overrides)
    return register(args.register, args.out, about) if args.register else 0


if __name__ == "__main__":
    sys.exit(main())
