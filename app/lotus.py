"""LOTUS as reference data: a bundle of its structure–organism–paper records, with its
metadata of each (an SQLite file built by lotus_build.py), shown as a matrix of the odor
table's chemicals that are in it by the taxa of the organisms they are reported in, each
cell's papers looked into in the Reference data tab.

A reference dataset of a bundle has ``fields`` ``{"layout": "lotus", ...}`` with its
options (`options`, `DEFAULT_OPTIONS`), each of which can be changed in the Admin tab:

- ``level``: the columns, one of `LEVELS` (organisms as LOTUS names them, or the species,
  genus, family, ... each is of);
- ``value``: what a cell is (`VALUES`): how many of the column's organisms the chemical is
  reported in, how many papers report it in any of them, or whether any does;
- ``stereo``: which of LOTUS's structures an odor is (`STEREO`): exactly its own, the same
  but for R/S (its InChI without tetrahedral stereo: enantiomers, diastereomers and
  unspecified stereocenters merged, cis/trans isomers kept apart), or any of its skeleton
  (the first block of its InChIKey: cis/trans isomers merged too);
- ``taxonomy``: the taxonomy source whose lineage of an organism is used first (`SOURCES`:
  the others follow in that order, each used where those before it do not give the
  organism one at the level).

`describe` says these (and what the bundle was made from) for people, as the app shows
them with the matrix. Its matrix depends on the odor table, so it is made again whenever
that changes (the reference dataset's import keeps a fingerprint of it, `odors_fingerprint`).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import sqlite3
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

BUNDLE_KIND = "lotus"
VERSION = 1  # of how matrices are made from a bundle: their imports are made again when it changes
SUFFIXES = (".sqlite", ".sqlite3", ".db")
LEVELS = ("organism", "species", "genus", "family", "order", "class", "phylum", "kingdom")
RANKS = LEVELS[1:]  # of a lineage, lowest first
PLURAL = {"organism": "organisms", "species": "species", "genus": "genera", "family": "families", "order": "orders",
          "class": "classes", "phylum": "phyla", "kingdom": "kingdoms"}
VALUES = ("organisms", "papers", "presence")
STEREO = ("enantiomers", "exact", "skeleton")
# taxonomy sources, in the order used by default after the one chosen (the most organisms of LOTUS first)
SOURCES = ("GBIF Backbone Taxonomy", "Catalogue of Life", "Open Tree of Life", "NCBI", "World Flora Online", "ITIS",
           "iNaturalist", "EUNIS", "Index Fungorum", "VASCAN", "AlgaeBase", "Mammal Species of the World",
           "BirdLife International")
DEFAULT_OPTIONS = {"level": "family", "value": "organisms", "stereo": "enantiomers", "taxonomy": SOURCES[0]}
# kingdoms by other names (of NCBI and the Open Tree of Life), as GBIF and the Catalogue of Life name them
KINGDOMS = {"Viridiplantae": "Plantae", "Chloroplastida": "Plantae", "Archaeplastida": "Plantae", "Metazoa": "Animalia"}
SOURCE_KINGDOM = {"World Flora Online": "Plantae", "VASCAN": "Plantae", "Mammal Species of the World": "Animalia",
                  "BirdLife International": "Animalia"}
# families conserved under two names (ICN Art. 18.5), by the one GBIF uses (as the bundle keeps them)
FAMILY_NAMES = {"Compositae": "Asteraceae", "Cruciferae": "Brassicaceae", "Gramineae": "Poaceae", "Guttiferae": "Clusiaceae",
                "Labiatae": "Lamiaceae", "Leguminosae": "Fabaceae", "Palmae": "Arecaceae", "Umbelliferae": "Apiaceae"}
STRUCTURE_URL = "https://lotus.naturalproducts.net/compound/lotus_id/{}"
WIKIDATA_URL = "https://www.wikidata.org/wiki/{}"
DOI_URL = "https://doi.org/{}"
MAX_ORGANISMS_SHOWN = 200  # of a cell looked into


class LotusError(ValueError):
    pass


# --------------------------------------------------------------------- options


def options(fields: Optional[dict]) -> dict:
    """A view's options, as stored (`fields`), checked and filled in from `DEFAULT_OPTIONS`."""
    f = fields or {}
    out = dict(DEFAULT_OPTIONS)
    if f.get("level") in LEVELS:
        out["level"] = f["level"]
    if out["level"] == "organism":
        out["value"] = "papers"  # (by default: of one organism, how many organisms is presence)
    if f.get("value") in VALUES:
        out["value"] = f["value"]
    if out["level"] == "organism" and out["value"] == "organisms":
        out["value"] = "presence"
    if f.get("stereo") in STEREO:
        out["stereo"] = f["stereo"]
    if isinstance(f.get("taxonomy"), str) and f["taxonomy"].strip():
        out["taxonomy"] = f["taxonomy"].strip()
    return out


def priority(first: str, available: Optional[list[str]] = None) -> list[str]:
    """Taxonomy sources in the order they are used: `first`, then `SOURCES`, then any other
    of `available` (by name)."""
    rest = [s for s in SOURCES if s != first]
    extra = sorted(s for s in (available or []) if s != first and s not in SOURCES)
    return [first, *rest, *extra]


STEREO_TEXT = {
    "enantiomers": "the same structure but for R/S: LOTUS's structures with the odor's InChI but for its tetrahedral stereo "
                   "(enantiomers, diastereomers and unspecified stereocenters merged, e.g. (+)- and (−)-α-pinene; cis/trans "
                   "isomers kept apart, e.g. geraniol and nerol)",
    "exact": "exactly the odor's structure (its InChI, stereochemistry included): a record of an enantiomer, or with "
             "stereocenters unspecified, is not the odor's",
    "skeleton": "any structure of the odor's skeleton (the first block of its InChIKey): every stereoisomer merged, cis/trans "
                "isomers too (e.g. geraniol with nerol)",
}
VALUE_TEXT = {
    "organisms": "how many of the {level}'s organisms (as LOTUS names them) the chemical is reported in",
    "papers": "how many papers report the chemical in {any}",
    "presence": "1 where the chemical is reported in {any}, else 0",
}


LEVEL_TEXT = {"organism": "organisms, as LOTUS names them (varieties and subspecies apart)",
              "species": "species (varieties and subspecies counted as their species)"}
STEREO_SHORT = {"enantiomers": "the same but for R/S: stereocenters ignored, cis/trans kept",
                "exact": "exactly the odor's structure, stereochemistry included",
                "skeleton": "any stereoisomer: stereocenters and cis/trans ignored"}
VALUE_SHORT = {"organisms": "how many of the column's organisms (as LOTUS names them) it is reported in",
               "papers": "how many papers report it in any of the column's organisms",
               "presence": "1 where any paper reports it in any of the column's organisms, else 0"}


def choices(about: Optional[dict] = None) -> dict[str, list[dict]]:
    """Each option of a view, with what each of its values means, for the Admin tab:
    {option: [{value, label, detail, default}]}; of ``taxonomy``, the sources of the bundle
    (its ``about``), each with how many of its organisms it has a lineage of."""
    cov = (about or {}).get("taxonomy_sources") or {}
    d = DEFAULT_OPTIONS
    return {
        "level": [{"value": x, "label": LEVEL_TEXT.get(x, PLURAL[x]), "default": x == d["level"]} for x in LEVELS],
        "value": [{"value": x, "label": VALUE_SHORT[x], "default": x == d["value"]} for x in VALUES],
        "stereo": [{"value": x, "label": STEREO_SHORT[x], "detail": STEREO_TEXT[x], "default": x == d["stereo"]} for x in STEREO],
        "taxonomy": [{"value": x, "label": f"{x} ({cov[x]:,} organisms)", "default": x == d["taxonomy"]}
                     for x in priority(d["taxonomy"], list(cov)) if x in cov] or
                    [{"value": x, "label": x, "default": x == d["taxonomy"]} for x in SOURCES],
    }


def value_label(o: dict) -> str:
    """What a view's cells are, short (as the viewer labels values)."""
    if o["value"] == "presence":
        return "reported"
    if o["value"] == "papers":
        return "papers reporting it"
    return f"organisms of the {o['level']} it is reported in"


def view_title(o: dict, release: Optional[dict] = None) -> str:
    """A title for a view (as `lotus_build.register` and the Admin tab suggest)."""
    rel = f" {release['pairs']}" if release and release.get("pairs") else ""
    of = "the organisms" if o["level"] == "organism" else f"the {PLURAL[o['level']]} of organisms"
    return f"LOTUS{rel}: chemicals by {of} they are reported in"


def describe(o: dict, about: Optional[dict] = None) -> dict:
    """A view's options (and what its bundle was made from), for people.

    Returns:
        ``summary`` (a sentence), and ``decisions``: [{``what``, ``chosen``, ``admin``
        (whether it can be changed in the Admin tab, else how)}], in the order shown.
    """
    level, plural = o["level"], PLURAL[o["level"]]
    any_ = "the organism" if level == "organism" else f"any of the {level}'s organisms"
    cells = VALUE_TEXT[o["value"]].format(level=level, any=any_)
    if level == "organism":
        cols = "organisms as LOTUS names them (varieties and subspecies each apart), ordered by their lineage"
    elif level == "species":
        cols = ("the species of each organism (varieties and subspecies counted as their species; organisms named only to "
                "a genus or above left out)")
    else:
        cols = f"the {level} of each organism (organisms with none in any taxonomy source left out)"
    sources = priority(o["taxonomy"], list((about or {}).get("taxonomy_sources") or {}))
    taxonomy = (f"{o['taxonomy']}'s lineage of each organism, else {', '.join(sources[1:4])} and the others in turn "
                f"(the first that gives it {'one' if level == 'organism' else 'a ' + level}); taxa told apart by kingdom; "
                f"families of two names as GBIF names them (e.g. Compositae as Asteraceae)")
    rel = (about or {}).get("release") or {}
    made = ("LOTUS's frozen export of its Wikidata records" + (f" of {rel['pairs']}" if rel.get("pairs") else "")
            + (f", with its metadata of {rel['metadata']}" if rel.get("metadata") else "")
            + (f"; LOTUS IDs (LTS…) from its release of {rel['lts']}" if rel.get("lts") else ""))
    decisions = [
        {"what": "Records", "chosen": made, "admin": "another bundle, made by app/lotus_build.py from other files"},
        {"what": "Chemicals", "chosen": "the odor table's, each by the LOTUS ID it gives, else its InChI, else its CID; "
                                        "made again whenever the odor table changes", "admin": "the odor table (its LOTUS IDs)"},
        {"what": "Stereochemistry", "chosen": STEREO_TEXT[o["stereo"]], "admin": True},
        {"what": "Columns", "chosen": cols, "admin": True},
        {"what": "Taxonomy", "chosen": taxonomy, "admin": True},
        {"what": "Cells", "chosen": cells, "admin": True},
    ]
    summary = (f"The odor table's chemicals in LOTUS by the {plural if level != 'organism' else 'organisms'} they are "
               f"reported in: {cells}. A chemical is {STEREO_TEXT[o['stereo']].split(':')[0]}.")
    return {"summary": summary, "decisions": decisions}


# ------------------------------------------------------------ structures' keys


def no_rs_inchi(inchi: Optional[str]) -> Optional[str]:
    """An InChI without its tetrahedral stereo layers (``/t``, ``/m``, ``/s``, also of its
    isotopic layer): the same for every stereoisomer but by cis/trans (``/b`` is kept)."""
    if not inchi:
        return None
    parts = inchi.split("/")
    return "/".join(p for i, p in enumerate(parts) if i < 2 or p[:1] not in ("t", "m", "s"))


def kingdom_of(source: str, kingdom: Optional[str], lineage: str = "") -> Optional[str]:
    """An organism's kingdom in a taxonomy source, as GBIF names them (`KINGDOMS`), so that
    taxa are told apart alike whichever source gives them; of sources of one kingdom, that;
    else Bacteria or Archaea if its lineage has them (NCBI's and the Open Tree's give no kingdom of these)."""
    if source in SOURCE_KINGDOM:
        return SOURCE_KINGDOM[source]
    if kingdom and not kingdom.lower().startswith("wfo-"):
        return KINGDOMS.get(kingdom, kingdom)
    parts = set((lineage or "").split("|"))
    return next((k for k in ("Bacteria", "Archaea") if k in parts), None)


def _inchikey(inchi: Optional[str]) -> Optional[str]:
    if not inchi:
        return None
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem.InchiToInchiKey(inchi) or None
    except Exception:
        return None


# ------------------------------------------------------------------ the bundle


def is_bundle_path(path: Path) -> bool:
    """Whether `path` is named as a bundle may be (not whether it is one)."""
    return Path(path).suffix.lower() in SUFFIXES


def _connect(path: Path) -> sqlite3.Connection:
    # read only, and immutable: no locks (which fail on NFS and read-only mounts)
    con = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def read_about(path: Path) -> dict:
    """A bundle's ``about`` (kind, release, built_at, sources, counts, taxonomy_sources).

    Raises:
        LotusError: if it is not a LOTUS bundle.
    """
    try:
        con = _connect(path)
        try:
            about = {r["key"]: json.loads(r["value"]) for r in con.execute("SELECT key, value FROM about")}
        finally:
            con.close()
    except sqlite3.DatabaseError as e:
        raise LotusError(f"not a LOTUS bundle (made by app/lotus_build.py): {e}") from None
    if about.get("kind") != BUNDLE_KIND:
        raise LotusError("not a LOTUS bundle (made by app/lotus_build.py)")
    return about


def is_bundle(path: Path) -> bool:
    """Whether `path` is a LOTUS bundle."""
    try:
        return is_bundle_path(path) and Path(path).is_file() and read_about(path) is not None
    except (LotusError, OSError):
        return False


class Bundle:
    """A bundle, its organisms' lineages read when first needed (the rest queried as needed)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.about = read_about(path)
        self._lock = threading.Lock()
        self._taxa: dict[tuple[str, str], dict] = {}

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        con = _connect(self.path)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def query_in(self, sql: str, ids: list, extra: tuple = ()) -> list[sqlite3.Row]:
        """`sql` with ``{ids}`` filled in by placeholders for `ids` (in chunks), after `extra` parameters."""
        out: list[sqlite3.Row] = []
        ids = list(ids)
        con = _connect(self.path)
        try:
            for i in range(0, len(ids), 900):
                part = ids[i:i + 900]
                out += con.execute(sql.format(ids=",".join("?" * len(part))), (*extra, *part)).fetchall()
        finally:
            con.close()
        return out

    @functools.cached_property
    def organisms(self) -> dict[int, dict]:
        return {r["id"]: dict(r) for r in self.query("SELECT * FROM organisms")}

    @functools.cached_property
    def lineages(self) -> dict[int, dict[str, dict]]:
        """Of each organism, its lineage in each taxonomy source that has it: {organism: {source: row}}."""
        out: dict[int, dict[str, dict]] = {}
        for r in self.query("SELECT * FROM taxonomy"):
            out.setdefault(r["organism"], {})[r["source"]] = dict(r)
        return out

    def taxa(self, level: str, first: str) -> dict:
        """Each organism's taxon at `level`, from the first taxonomy source (`priority`) that
        gives it one: ``of`` {organism: taxon key}, ``taxa`` {key: {name, kingdom, lineage
        (the ranks above, as that source gives them), source, organisms}}, ``none``
        (organisms with no taxon at the level). A key is (name, kingdom); a taxon of no
        kingdom is taken as the one of its name, if only one has a kingdom."""
        cache_key = (level, first)
        with self._lock:
            if cache_key in self._taxa:
                return self._taxa[cache_key]
        order = priority(first, list(self.about.get("taxonomy_sources") or {}))
        rank = {s: i for i, s in enumerate(order)}
        of: dict[int, tuple] = {}
        taxa: dict[tuple, dict] = {}
        none: list[int] = []
        above = RANKS[RANKS.index(level) + 1:] if level != "organism" else RANKS
        for oid, org in self.organisms.items():
            rows = sorted((self.lineages.get(oid) or {}).values(), key=lambda r: rank.get(r["source"], len(rank)))
            if level == "organism":
                r = rows[0] if rows else None
                name = org["name"]
            else:
                r = next((x for x in rows if x[level]), None)
                name = r[level] if r else None
            if not name:
                none.append(oid)
                continue
            key = ("organism", oid) if level == "organism" else (name, r["kingdom"] if r else None)
            of[oid] = key
            t = taxa.setdefault(key, {"name": name, "kingdom": key[1] if level != "organism" else (r or {}).get("kingdom"),
                                      "lineage": None, "source": None, "organisms": [], "_rank": len(rank) + 1})
            t["organisms"].append(oid)
            at = rank.get(r["source"], len(rank)) if r else len(rank) + 1
            if at < t["_rank"]:  # (its lineage as the first source of any of its organisms gives it)
                t.update(lineage={k: r.get(k) for k in above}, source=r["source"], _rank=at)
            elif t["lineage"] is None:
                t["lineage"] = {k: None for k in above}
        if level != "organism":  # of no kingdom: the one of its name, if only one has a kingdom
            kingdoms: dict[str, set] = {}
            for name, k in taxa:
                if k:
                    kingdoms.setdefault(name, set()).add(k)
            for key in [k for k in taxa if k[1] is None and len(kingdoms.get(k[0], ())) == 1]:
                into = (key[0], next(iter(kingdoms[key[0]])))
                t = taxa.pop(key)
                taxa[into]["organisms"] += t["organisms"]
                if t["_rank"] < taxa[into]["_rank"]:
                    taxa[into].update(lineage=t["lineage"], source=t["source"], _rank=t["_rank"])
                for oid in t["organisms"]:
                    of[oid] = into
        for t in taxa.values():
            t.pop("_rank", None)
        out = {"of": of, "taxa": taxa, "none": none}
        with self._lock:
            self._taxa[cache_key] = out
        return out


_bundles: "OrderedDict[tuple, Bundle]" = OrderedDict()
_bundles_lock = threading.Lock()


def bundle(path: Path) -> Bundle:
    """The bundle at `path`, as it is now (kept in memory, two at most)."""
    st = os.stat(path)
    key = (str(Path(path).resolve()), st.st_size, st.st_mtime_ns)
    with _bundles_lock:
        if key in _bundles:
            _bundles.move_to_end(key)
            return _bundles[key]
    b = Bundle(path)
    with _bundles_lock:
        _bundles[key] = b
        while len(_bundles) > 2:
            _bundles.popitem(last=False)
    return b


# ---------------------------------------------------------- odors -> structures

STEREO_COLUMN = {"exact": "inchikey", "enantiomers": "key_no_rs", "skeleton": "skeleton"}
HOW = {"lotus": "by the LOTUS ID in the odor table", "inchi": "by InChI", "cid": "by CID"}


def _keys(inchi: Optional[str]) -> Optional[dict[str, str]]:
    """An InChI's keys, as structures have them: by `STEREO_COLUMN`."""
    k = _inchikey(inchi)
    if not k:
        return None
    return {"inchikey": k, "key_no_rs": _inchikey(no_rs_inchi(inchi)) or k, "skeleton": k[:14]}


def match_odors(b: Bundle, odors: list[dict], stereo: str) -> list[dict]:
    """The structures of the bundle that each odor (of the odor table) is, as `stereo` says
    (`STEREO`): those with its key (`STEREO_COLUMN`) from the structure of the LOTUS ID the
    odor table gives it, else from its InChI, else from the structure of its CID.

    Returns:
        per odor (in their order): {``odor``, ``structures`` (IDs, those with the most
        records first), ``how`` ('lotus' | 'inchi' | 'cid' | None)}
    """
    col = STEREO_COLUMN[stereo]
    con = _connect(b.path)
    try:
        def find(**kw) -> list[sqlite3.Row]:
            (k, v), = kw.items()
            return con.execute(f"SELECT inchikey, key_no_rs, skeleton FROM structures WHERE {k} = ?", (v,)).fetchall()

        def expand(keys) -> list[int]:
            rows = con.execute(f"SELECT s.id, (SELECT COUNT(*) FROM pairs p WHERE p.structure = s.id) AS n FROM structures s "
                               f"WHERE s.{col} = ? ORDER BY n DESC, s.id", (keys[col],)).fetchall()
            return [r["id"] for r in rows]

        out = []
        for o in odors:
            found, how = [], None
            lts = (o.get("lotus") or "").strip()
            tries = [("lotus", lambda: [dict(r) for r in find(lts=lts)] if lts else []),
                     ("inchi", lambda: [k] if (k := _keys(o.get("inchi"))) else []),
                     ("cid", lambda: [dict(r) for r in find(cid=int(o["cid"]))] if o.get("cid") else [])]
            for h, get in tries:
                ids = list(dict.fromkeys(i for keys in get() for i in expand(keys)))
                if ids:
                    found, how = ids, h
                    break
            out.append({"odor": o, "structures": found, "how": how})
        return out
    finally:
        con.close()


def odors_fingerprint(odors: list[dict]) -> str:
    """What of the odor table the matrices depend on, hashed."""
    keys = sorted((o["id"], o.get("name"), o.get("cid"), o.get("inchi"), o.get("lotus")) for o in odors)
    return hashlib.sha1(json.dumps(keys, default=str).encode()).hexdigest()[:16]


def _structures(b: Bundle, ids: list[int]) -> dict[int, dict]:
    return {r["id"]: dict(r) for r in b.query_in("SELECT * FROM structures WHERE id IN ({ids})", ids)}


def _unmatched_note(matches: list[dict]) -> Optional[str]:
    missing = [m["odor"]["name"] for m in matches if not m["structures"]]
    if not missing:
        return None
    shown = ", ".join(missing[:8]) + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
    return (f"{len(missing)} of the odor table's {len(matches)} chemicals are not in LOTUS (by LOTUS ID, InChI or CID, "
            f"as the stereochemistry setting says): {shown}")


# ------------------------------------------------------------------- the view


def _sort_key(t: dict, kingdoms: dict[Optional[str], int]) -> tuple:
    """A taxon's place among columns: by its lineage (kingdoms with the most organisms first, then by name)."""
    lin = t["lineage"]
    king = lin.get("kingdom", t["kingdom"])
    rest = [lin.get(k) for k in reversed(RANKS) if k in lin and k != "kingdom"]
    return (-kingdoms.get(king, 0), (king or "~").casefold(), *((v or "~").casefold() for v in [*rest, t["name"]]))


def _lineage_text(t: dict) -> str:
    return " › ".join(v for k in RANKS if (v := t["lineage"].get(k)))


def parse(path: Path, fields: Optional[dict], odors: list[dict]) -> tuple[dict, np.ndarray]:
    """A view of the bundle at `path` over `odors` (the odor table), as `reference.parse`
    gives a matrix: (record, values). Its rows carry their structures (``lotus``: IDs), its
    columns their taxon (``lotus``: {level, name, kingdom}), to look into a cell (`cell`)."""
    try:
        b = bundle(path)
    except sqlite3.DatabaseError as e:
        raise LotusError(f"not a LOTUS bundle: {e}") from None
    o = options(fields)
    matches = match_odors(b, odors, o["stereo"])
    rows = [m for m in matches if m["structures"]]
    if not rows:
        raise LotusError("none of the odor table's chemicals is in LOTUS (by LOTUS ID, InChI or CID)")
    tx = b.taxa(o["level"], o["taxonomy"])
    all_ids = sorted({s for m in rows for s in m["structures"]})
    by_structure: dict[int, list[tuple[int, int]]] = {}
    for r in b.query_in("SELECT structure, organism, ref FROM pairs WHERE structure IN ({ids})", all_ids):
        by_structure.setdefault(r["structure"], []).append((r["organism"], r["ref"]))
    cells: dict[tuple[int, tuple], tuple[set, set]] = {}
    left_out: set[int] = set()
    for i, m in enumerate(rows):
        for s in m["structures"]:
            for org, ref in by_structure.get(s, ()):
                key = tx["of"].get(org)
                if key is None:
                    left_out.add(org)
                    continue
                orgs, refs = cells.setdefault((i, key), (set(), set()))
                orgs.add(org)
                refs.add(ref)
    kingdoms = Counter()
    for t in tx["taxa"].values():
        kingdoms[t["lineage"].get("kingdom", t["kingdom"])] += len(t["organisms"])
    keys = sorted({k for _, k in cells}, key=lambda k: _sort_key(tx["taxa"][k], kingdoms))
    at = {k: j for j, k in enumerate(keys)}
    values = np.zeros((len(rows), len(keys)), dtype=np.float32)
    for (i, k), (orgs, refs) in cells.items():
        values[i, at[k]] = len(orgs) if o["value"] == "organisms" else len(refs) if o["value"] == "papers" else 1
    info = _structures(b, all_ids)
    names = Counter(tx["taxa"][k]["name"] for k in keys)
    cols = []
    for k in keys:
        t = tx["taxa"][k]
        label = t["name"] + (f" ({t['kingdom'] or 'no kingdom'})" if names[t["name"]] > 1 and o["level"] != "organism" else "")
        n = len(t["organisms"])
        detail = " · ".join(x for x in [_lineage_text(t), f"{n:,} organism{'s' if n != 1 else ''} of LOTUS" if o["level"] != "organism" else ""] if x)
        cols.append({"label": label, "detail": detail,
                     "lotus": {"level": o["level"], "name": t["name"], "kingdom": t["kingdom"],
                               **({"organism": k[1]} if o["level"] == "organism" else {})}})

    def row(m: dict) -> dict:
        s = [info[i] for i in m["structures"] if i in info]
        first = s[0] if s else {}
        what = first.get("name") or first.get("iupac_name") or first.get("inchikey") or "?"
        ident = first.get("lts") or first.get("inchikey")
        more = f", and {len(s) - 1} other structure{'s' if len(s) > 2 else ''} of it" if len(s) > 1 else ""
        return {"label": m["odor"]["name"], "group": None, "lotus": m["structures"],
                "detail": f"LOTUS: {what} ({ident}){more}, matched {HOW[m['how']]}"}

    notes = [n for n in [_unmatched_note(matches)] if n]
    merged = [(m["odor"]["name"], len(m["structures"])) for m in rows if len(m["structures"]) > 1]
    if merged:
        notes.append(f"{len(merged)} chemical{'s are' if len(merged) > 1 else ' is'} each several of LOTUS's structures "
                     f"(as the stereochemistry setting says): " + ", ".join(f"{n} ({k})" for n, k in merged[:8])
                     + ("…" if len(merged) > 8 else ""))
    empty = [m["odor"]["name"] for i, m in enumerate(rows) if not values[i].any()]
    if empty:
        notes.append(f"{len(empty)} chemical{'s are' if len(empty) > 1 else ' is'} in LOTUS but in no organism with a "
                     f"{o['level']}: {', '.join(empty[:8])}{'…' if len(empty) > 8 else ''}")
    if left_out:
        notes.append(f"{len(left_out):,} organism{'s' if len(left_out) > 1 else ''} of these chemicals left out: no taxonomy "
                     f"source gives {'them' if len(left_out) > 1 else 'it'} a {o['level']}")
    record = {
        "rows": [row(m) for m in rows],
        "cols": cols,
        "ids": {"rows": [{k: m["odor"].get(k) for k in ("name", "cid", "inchi") if m["odor"].get(k)} for m in rows], "cols": None},
        "chemical_axis": "rows",
        "kind": "binary" if o["value"] == "presence" else "sequential",
        "shape": list(values.shape),
        "notes": notes,
        "lotus": {**o, "release": b.about.get("release"), "built_at": b.about.get("built_at"),
                  "value_label": value_label(o), "title": view_title(o, b.about.get("release")), **describe(o, b.about)},
    }
    return record, values


# ------------------------------------------------------------------- a cell


def cell(b: Bundle, fields: Optional[dict], structures: list[int], taxon: dict) -> dict:
    """What a cell of a view is of: the organisms of its column's taxon (`taxon`, as its
    column's ``lotus``) that any of its row's structures (`structures`, as its row's
    ``lotus``) are reported in, each with the papers reporting them there.

    Returns:
        ``taxon`` {level, name, kingdom, lineage, source}, ``structures`` [{id, name,
        inchikey, lts, cid, wikidata, links}], ``organisms`` [{id, name, wikidata, source,
        lineage, papers: [{doi, title, journal, date, author, wikidata, validated,
        structures, url}]}] (those with the most papers first, at most `MAX_ORGANISMS_SHOWN`),
        ``n_organisms``, ``n_papers``, ``options`` (the view's, `options`).
    """
    o = options(fields)
    tx = b.taxa(o["level"], o["taxonomy"])
    if o["level"] == "organism":
        key = ("organism", taxon.get("organism"))
    else:
        key = (taxon.get("name"), taxon.get("kingdom"))
    t = tx["taxa"].get(key)
    if t is None:
        raise LotusError(f"no {o['level']} {taxon.get('name')!r} in LOTUS")
    structures = [int(s) for s in structures]
    rows = b.query_in("SELECT structure, organism, ref, validated FROM pairs WHERE structure IN ({ids})", structures)
    members = set(t["organisms"])
    by_org: dict[int, dict[int, dict]] = {}
    for r in rows:
        if r["organism"] in members:
            p = by_org.setdefault(r["organism"], {}).setdefault(r["ref"], {"structures": [], "validated": False})
            p["structures"].append(r["structure"])
            p["validated"] = p["validated"] or bool(r["validated"])
    refs = {r["id"]: dict(r) for r in b.query_in("SELECT * FROM refs WHERE id IN ({ids})", sorted({x for d in by_org.values() for x in d}))}
    order = sorted(by_org, key=lambda org: (-len(by_org[org]), b.organisms[org]["name"].casefold()))
    first = priority(o["taxonomy"], list(b.about.get("taxonomy_sources") or {}))
    organisms = []
    for org in order[:MAX_ORGANISMS_SHOWN]:
        lin = sorted((b.lineages.get(org) or {}).values(), key=lambda r: first.index(r["source"]) if r["source"] in first else len(first))
        used = next((r for r in lin if o["level"] == "organism" or r[o["level"]]), None)
        papers = []
        for ref_id, p in sorted(by_org[org].items(), key=lambda kv: (refs.get(kv[0], {}).get("date") or "", kv[0]), reverse=True):
            r = refs.get(ref_id, {})
            papers.append({"doi": r.get("doi"), "title": r.get("title"), "journal": r.get("journal"), "date": r.get("date"),
                           "author": r.get("author"), "wikidata": r.get("wikidata"), "validated": p["validated"],
                           "structures": sorted(set(p["structures"])), "url": DOI_URL.format(r["doi"]) if r.get("doi") else None})
        org_row = b.organisms[org]
        organisms.append({"id": org, "name": org_row["name"], "wikidata": org_row["wikidata"],
                          "source": used["source"] if used else None,
                          "lineage": " › ".join(v for k in reversed(RANKS) if used and (v := used.get(k))),
                          "papers": papers})
    info = _structures(b, structures)
    return {
        "taxon": {"level": o["level"], "name": t["name"], "kingdom": t["kingdom"], "lineage": _lineage_text(t), "source": t["source"]},
        "structures": [{**{k: s.get(k) for k in ("id", "name", "iupac_name", "inchikey", "lts", "cid", "wikidata", "formula")},
                        "links": {"lotus": STRUCTURE_URL.format(s["lts"]) if s.get("lts") else None,
                                  "wikidata": WIKIDATA_URL.format(s["wikidata"]) if s.get("wikidata") else None}}
                       for s in (info[i] for i in structures if i in info)],
        "organisms": organisms,
        "n_organisms": len(by_org),
        "n_papers": len({x for d in by_org.values() for x in d}),
        "options": o,
    }


def about_summary(path: Path) -> dict[str, Any]:
    """A bundle's release, when it was built, its counts and taxonomy sources (for the Admin tab)."""
    a = read_about(path)
    return {k: a.get(k) for k in ("release", "built_at", "counts", "taxonomy_sources")}
