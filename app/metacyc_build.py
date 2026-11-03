"""Builds a MetaCyc bundle: the SmartTables exported from MetaCyc (tab-separated, as in
``BioCyc/raw``: compounds, pathways, reactions, organisms) read into one SQLite file, which
the app reads as a reference dataset (see metacyc.py and reference_data/README.md). Run:

  python -m app.metacyc_build <raw folder> <bundle.sqlite> [--version 27.0] [--register http://<server>:<port>]

``--register`` adds its two reference datasets to the app (as the Admin tab does): its
compounds by the pathways they are in, and the pathway LCA distance between them.

Each table is a SmartTable exported twice, by frame IDs and by common names, in the same
row order: IDs are taken from the first, the names of what they list from the second (by
position). Which files are read can be changed with ``--file <role>=<path>`` (see `FILES`).
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


def _records(path: Path, id_field: int, ids: set[str]) -> dict[str, list[str]]:
    """A SmartTable whose last column runs over several lines (unquoted, as pathway summaries
    are): each record by its ID (field `id_field` of the line it starts on, one of `ids`), its
    fields, the last with the lines after it joined to it."""
    out: dict[str, list[str]] = {}
    current = None
    for line in _lines(path)[1:]:
        fields = line.split("\t")
        if len(fields) > id_field and fields[id_field] in ids and fields[id_field] not in out:
            current = out[fields[id_field]] = fields
        elif current is not None:
            current[-1] += "\n" + line
    return out


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
    """The file read for each role (`FILES`): as given in `overrides`, else the first that exists."""
    out = {}
    for role, candidates in FILES.items():
        given = (overrides or {}).get(role)
        if given:
            p = Path(given)
            out[role] = p if p.is_absolute() else raw / p
            continue
        out[role] = next((raw / c for c in candidates if (raw / c).is_file()), None)
    missing = [r for r in REQUIRED if not out[r] or not out[r].is_file()]
    if missing:
        raise SystemExit(f"not found in {raw}: {', '.join(f'{r} ({FILES[r][0]})' for r in missing)} (give it with --file {missing[0]}=<path>)")
    return out


def build(raw: Path, out: Path, version: Optional[str] = None, overrides: Optional[dict[str, str]] = None,
          log=print) -> dict:
    """Read MetaCyc's SmartTables in `raw` into the bundle `out` (replaced if it exists).

    Returns:
        its ``about`` (as stored), with ``counts`` of what it has.
    """
    files = resolve(raw, overrides)
    names: dict[str, str] = {}

    # ------------------------------------------------------------- compounds
    c, cn = read_table(files["compounds"]), read_table(files["compound_names"])
    if len(c) != len(cn):
        raise SystemExit(f"{files['compounds'].name} and {files['compound_names'].name} differ in rows: give the two exports of one table")
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
        raise SystemExit(f"{files['pathways'].name} and {files['pathway_names'].name} differ in rows: give the two exports of one table")
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
    log(f"pathways: {len(pw):,} ({files['pathways'].name})")

    # ------------------------------------------------------------- reactions
    x, xn = read_table(files["reactions"]), read_table(files["reaction_names"])
    if len(x) != len(xn) or not (x["Object ID"].to_numpy() == xn["Object ID"].to_numpy()).all():
        raise SystemExit(f"{files['reactions'].name} and {files['reaction_names'].name} differ in rows: give the two exports of one table")
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
        recs = _records(files["summaries"], 1, set(pw))
        for pid, fields in recs.items():
            if len(fields) >= 4:
                pw[pid]["summary"] = _summary("\t".join(fields[3:]), {**species_names, **classes, **names}) or None
        log(f"summaries: {len(recs):,} ({files['summaries'].name})")
    if files["pathway_extra"]:
        recs = _records(files["pathway_extra"], 2, set(pw))
        for pid, fields in recs.items():
            if len(fields) >= 12:
                pw[pid]["taxonomic_range"] = [plain(t) for t in split(fields[10])]
                pw[pid]["engineered"] = {"T": 1, "NIL": 0}.get(fields[11].strip().upper())
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
    about = {"kind": "metacyc", "version": version, "built_at": dt.datetime.now().isoformat(timespec="seconds"),
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
     "description": "Each chemical of the odor table that is in MetaCyc, by the pathways it is in: 1 itself, 0.5 only through a "
                    "class it is of (e.g. an alcohol), if it is in none itself. Click a cell for the pathway.",
     "value_label": "in the pathway", "row_label": "chemical", "col_label": "pathway"},
    {"view": "lca", "title": "MetaCyc{v}: pathway LCA distance between odors", "kind": "sequential",
     "description": "Between two chemicals of the odor table, the fewest steps from a common ancestor to both in MetaCyc's "
                    "network of compounds and reactions (as in O'Connell's thesis, ch. 2). Click a cell for the paths.",
     "value_label": "steps to their lowest common ancestor", "row_label": "chemical", "col_label": "chemical"},
]


def register(server: str, bundle: Path, version: Optional[str], log=print) -> int:
    """Add the bundle's reference datasets to the app at `server` (its Admin tab's API): 0 if all were added."""
    v = f" {version}" if version else ""
    failed = 0
    for spec in VIEWS:
        body = {"path": str(bundle.resolve()), "title": spec["title"].format(v=v), "description": spec["description"],
                "source": f"MetaCyc{v} (metacyc.org), SmartTables read by app/metacyc_build.py", "kind": spec["kind"],
                "chemical_axis": "detect", "value_label": spec["value_label"], "row_label": spec["row_label"],
                "col_label": spec["col_label"], "fields": {"layout": "metacyc", "view": spec["view"]}}
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
    ap.add_argument("out", type=Path, help="the bundle to write (e.g. metacyc-27.0.sqlite)")
    ap.add_argument("--version", default="27.0", help="MetaCyc's version the tables are of, shown in titles (default: 27.0, "
                    "as the files read by default are)")
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
    build(args.raw, args.out, args.version or None, overrides)
    return register(args.register, args.out, args.version or None) if args.register else 0


if __name__ == "__main__":
    sys.exit(main())
