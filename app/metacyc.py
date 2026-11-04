"""MetaCyc as reference data: a bundle of its compounds, pathways, reactions and organisms
(an SQLite file built by metacyc_build.py from MetaCyc's SmartTables), shown as matrices
with the chemicals of the odor table that are in it marked (or over them alone), and looked
into in the Reference data tab (pathways drawn, compounds, the paths between two odors).

A reference dataset of a bundle has ``fields`` ``{"layout": "metacyc", "view": ...}``, the
view one of `VIEWS`:

- ``pathways``: every compound in a pathway × the pathways (`pathway_matrix`): 1 where the
  compound itself is, 0.5 where it is only through a class it is of (MetaCyc's "general
  pathways", e.g. an alcohol's), as ``classes`` says; the odor table's chemicals first;
- ``lca``: the odor table's chemicals × themselves, their pathway LCA distance
  (`lca_matrix`, `Network`): of every compound, it would be too large to make or keep.

Its matrix depends on the odor table (which compounds are its chemicals), so it is made
again whenever that changes (the reference dataset's import keeps a fingerprint of it,
`odors_fingerprint`).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import sqlite3
import threading
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .sparse import Sparse

VERSION = 3  # of how matrices are made from a bundle: their imports are made again when it changes
SUFFIXES = (".sqlite", ".sqlite3", ".db")
VIEWS = ("pathways", "lca")
VIAS = ("reactions", "pathways")
CLASSES = ("never", "fallback", "always")
DEFAULTS = {"pathways": {"classes": "fallback"},
            "lca": {"via": "reactions", "classes": "fallback", "max_pathways": None}}
MAX_PATHWAYS_DEFAULT = 240  # of the network through pathways, as O'Connell's thesis (ch. 2, Methods) says
# left out of the network of the LCA distance: the currency metabolites left out of it for
# O'Connell's thesis (MetaCyc 27.1, February 2024, though it says 27.0), by frame ID as then
BLACKLIST = ("ADP", "ATP", "CO-A", "GDP", "GTP", "NAD", "NADH", "NADP", "NADPH", "OXYGEN-MOLECULE", "PPI", "PROTON",
             "Pi", "UDP", "WATER")
# drawn beside a reaction's arrow, not as compounds of a pathway's graph (as are compounds
# in more than SIDE_PATHWAYS pathways)
CURRENCY = frozenset(BLACKLIST + (
    "AMP", "CARBON-DIOXIDE", "HYDROGEN-PEROXIDE", "AMMONIUM", "AMMONIA", "FAD", "FADH2", "FMN", "FMNH2", "CTP", "CDP",
    "UTP", "ITP", "IDP", "DGDP", "DGTP", "DUDP", "S-ADENOSYLMETHIONINE", "ADENOSYL-HOMO-CYS", "Acceptor", "Donor-H2",
    "Reduced-ferredoxins", "Oxidized-ferredoxins", "Red-NADPH-Hemoprotein-Reductases", "Ox-NADPH-Hemoprotein-Reductases",
    "Reduced-flavodoxins", "Oxidized-flavodoxins", "Cytochromes-C-Reduced", "Cytochromes-C-Oxidized", "HCO3", "SULFATE",
    "PAPS", "3-5-ADP", "UDP-GLUCURONATE", "GLT", "2-KETOGLUTARATE", "OXYGEN-MOLECULE", "Ubiquinones", "Ubiquinols",
    "ACETYL-COA", "MALONYL-COA", "GLUTATHIONE", "OXIDIZED-GLUTATHIONE", "Pi", "PPI", "PROTON", "WATER"))
SIDE_PATHWAYS = 100
URL = "https://biocyc.org/{what}?orgid=META&id={id}"
TAXON_URL = "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={id}"
MAX_SIDE_PATHWAYS_SHOWN = 12  # other pathways a compound of a drawn pathway is linked to, at most


class MetacycError(ValueError):
    pass


def is_bundle_path(path: Path) -> bool:
    """Whether `path` is named as a bundle may be (not whether it is one)."""
    return Path(path).suffix.lower() in SUFFIXES


def _connect(path: Path) -> sqlite3.Connection:
    # read only, and immutable: no locks (which fail on NFS and read-only mounts)
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro&immutable=1", uri=True, check_same_thread=False)


def read_about(path: Path) -> dict:
    """A bundle's ``about`` (kind, version, built_at, sources, counts).

    Raises:
        MetacycError: if it is not a bundle.
    """
    try:
        con = _connect(path)
        try:
            about = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM about")}
        finally:
            con.close()
    except sqlite3.DatabaseError as e:
        raise MetacycError(f"not a MetaCyc bundle (made by app/metacyc_build.py): {e}") from None
    if about.get("kind") != "metacyc":
        raise MetacycError("not a MetaCyc bundle (made by app/metacyc_build.py)")
    return about


def options(fields: Optional[dict]) -> dict:
    """A view's options, as stored (`fields`), checked and filled in from `DEFAULTS`."""
    f = fields or {}
    view = f.get("view") if f.get("view") in VIEWS else "pathways"
    out = {"view": view, **DEFAULTS[view]}
    if f.get("classes") in CLASSES:
        out["classes"] = f["classes"]
    if view == "lca":
        if f.get("via") in VIAS:
            out["via"] = f["via"]
        # (by default, none through reactions, as for the thesis; as its Methods say through pathways)
        mp = f["max_pathways"] if "max_pathways" in f else MAX_PATHWAYS_DEFAULT if out["via"] == "pathways" else None
        try:
            mp = int(mp) if mp is not None and str(mp).strip() else None
        except (TypeError, ValueError):
            mp = None
        out["max_pathways"] = mp if mp and mp > 0 else None
    return out


def describe(o: dict) -> str:
    """A view's options, for people."""
    if o["view"] == "pathways":
        return {"never": "compounds in the pathways themselves only",
                "fallback": "1 in the pathway itself; 0.5 only through a class it is of, for compounds in none themselves",
                "always": "1 in the pathway itself; 0.5 only through a class it is of"}[o["classes"]]
    via = ("a network of compounds and reactions (a reaction's reactants its parents, its products its children)"
           if o["via"] == "reactions" else
           "a network of compounds and pathways (a pathway's reactants its parents, every compound of its reactions its children)")
    cls = {"never": "", "fallback": "; compounds not in it taken as the classes they are of that are",
           "always": "; compounds also taken as the classes they are of"}[o["classes"]]
    hubs = f", and compounds in more than {o['max_pathways']} pathways" if o["max_pathways"] else ""
    return (f"the fewest steps from a common ancestor to both, in {via}, without {len(BLACKLIST)} currency metabolites "
            f"(ATP, NADH, H₂O, ...){hubs}{cls}")


# ------------------------------------------------------------------ the bundle


def _j(s: Optional[str]) -> Any:
    return json.loads(s) if s else []


class Bundle:
    """A bundle's tables, read when first needed."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.about = read_about(path)
        self._lock = threading.Lock()

    def _rows(self, sql: str) -> list[sqlite3.Row]:
        con = _connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()

    @functools.cached_property
    def compounds(self) -> dict[str, dict]:
        out = {}
        for r in self._rows("SELECT * FROM compounds"):
            d = dict(r)
            for k in ("parents", "pathways", "general_pathways"):
                d[k] = _j(d[k])
            out[d["id"]] = d
        return out

    @functools.cached_property
    def names(self) -> dict[str, str]:
        return {r["id"]: r["name"] for r in self._rows("SELECT id, name FROM names")}

    @functools.cached_property
    def pathways(self) -> dict[str, dict]:
        out = {}
        for r in self._rows("SELECT * FROM pathways"):
            d = dict(r)
            for k in ("types", "species", "reactions", "reactants", "products", "primary_reactants", "primary_products",
                      "taxonomic_range", "super", "subs"):
                d[k] = _j(d[k])
            out[d["id"]] = d
        return out

    @functools.cached_property
    def reactions(self) -> dict[str, dict]:
        out = {}
        for r in self._rows("SELECT * FROM reactions"):
            d = dict(r)
            for k in ("ec", "reactants", "products", "pathways"):
                d[k] = _j(d[k])
            out[d["id"]] = d
        return out

    @functools.cached_property
    def pathway_classes(self) -> dict[str, dict]:
        return {r["id"]: dict(r) for r in self._rows("SELECT * FROM pathway_classes")}

    @functools.cached_property
    def ontology(self) -> dict[str, str]:
        """Pathway classes by name: their parents' names."""
        try:
            return {r["name"]: r["parent"] for r in self._rows("SELECT * FROM pathway_ontology")}
        except sqlite3.OperationalError:
            return {}

    @functools.cached_property
    def species(self) -> dict[str, dict]:
        return {r["id"]: {**dict(r), "synonyms": _j(r["synonyms"])} for r in self._rows("SELECT * FROM species")}

    @functools.cached_property
    def genomes(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in self._rows("SELECT * FROM genomes WHERE species IS NOT NULL"):
            out.setdefault(r["species"], []).append(dict(r))
        return out

    @functools.cached_property
    def n_pathways(self) -> Counter:
        """Of every frame (compounds, and classes such as "an alcohol"), the pathways whose reactions it is in."""
        n: Counter = Counter()
        for p in self.pathways.values():
            ids = {c for r in p["reactions"] if (x := self.reactions.get(r)) for c in x["reactants"] + x["products"]}
            n.update(ids)
        return n

    @functools.cached_property
    def of_compound(self) -> dict[str, dict[str, list[str]]]:
        """Of each frame, the reactions it is a reactant and a product of: {id: {consumed, produced}}."""
        out: dict[str, dict[str, list[str]]] = {}
        for rid, r in self.reactions.items():
            for c in r["reactants"]:
                out.setdefault(c, {"consumed": [], "produced": []})["consumed"].append(rid)
            for c in r["products"]:
                out.setdefault(c, {"consumed": [], "produced": []})["produced"].append(rid)
        return out

    def name(self, frame: str) -> str:
        return self.names.get(frame) or self.pathways.get(frame, {}).get("name") or frame

    @functools.lru_cache(maxsize=8)
    def network(self, via: str, max_pathways: Optional[int]) -> "Network":
        with self._lock:
            return Network(self, via, max_pathways)


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


# ---------------------------------------------------------- odors -> compounds


def _inchikey(inchi: Optional[str]) -> Optional[str]:
    if not inchi:
        return None
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem.InchiToInchiKey(inchi) or None
    except Exception:
        return None


@functools.lru_cache(maxsize=4)
def _lookups(b: Bundle) -> dict[str, dict]:
    out: dict[str, dict] = {"inchi": {}, "inchikey": {}, "cid": {}, "skeleton": {}, "name": {}}
    for cid_, c in b.compounds.items():
        key = c["inchikey"] or _inchikey(c["inchi"])  # (computed where MetaCyc gives none)
        for kind, v in (("inchi", c["inchi"]), ("inchikey", key), ("cid", c["cid"]),
                        ("skeleton", (key or "")[:14] or None), ("name", (c["name"] or "").casefold() or None)):
            if v:
                out[kind].setdefault(v, []).append(cid_)
    return out


def _best(b: Bundle, ids: list[str]) -> str:
    """Of compounds that a chemical could be: the one in the most pathways, neutral, then by ID."""
    return min(ids, key=lambda i: (-len(b.compounds[i]["pathways"]), not (b.compounds[i]["inchikey"] or "").endswith("-N"), i))


def match_odors(b: Bundle, odors: list[dict]) -> list[dict]:
    """The compound of the bundle that each odor (of the odor table) is: by the MetaCyc ID
    the odor table gives it, else its InChI, InChIKey, CID, the first block of its InChIKey
    (its skeleton: the same compound but for charge, as MetaCyc keeps acids as their
    anions, or stereochemistry), or its name or aliases.

    Returns:
        per odor (in their order): {``odor``, ``compound`` (frame ID or None), ``how``
        ('metacyc' | 'inchi' | 'inchikey' | 'cid' | 'skeleton' | 'name' | None)}
    """
    lk = _lookups(b)
    out = []
    for o in odors:
        given = (o.get("metacyc") or "").strip()
        key = _inchikey(o.get("inchi"))
        tries = [("metacyc", [given] if given in b.compounds or given in b.names else []),
                 ("inchi", lk["inchi"].get(o.get("inchi"), []) if o.get("inchi") else []),
                 ("inchikey", lk["inchikey"].get(key, []) if key else []),
                 ("cid", lk["cid"].get(o.get("cid"), []) if o.get("cid") else []),
                 ("skeleton", lk["skeleton"].get(key[:14], []) if key else []),
                 *(("name", lk["name"].get(n.casefold(), [])) for n in [o.get("name"), *(o.get("aliases") or [])] if n)]
        compound, how = None, None
        for h, ids in tries:
            ids = [i for i in ids if i in b.compounds or h == "metacyc"]
            if ids:
                compound, how = (ids[0] if h == "metacyc" else _best(b, ids)), h
                break
        out.append({"odor": o, "compound": compound, "how": how})
    return out


_marks: "OrderedDict[tuple, dict]" = OrderedDict()


def odor_marks(b: Bundle, odors: list[dict]) -> dict[str, dict]:
    """The odors of the odor table that compounds of the bundle are: {frame ID: {id, name, abbrev}}."""
    key = (id(b), odors_fingerprint(odors))
    with _bundles_lock:
        if key in _marks:
            return _marks[key]
    out = {m["compound"]: {k: m["odor"].get(k) for k in ("id", "name", "abbrev")} for m in match_odors(b, odors) if m["compound"]}
    with _bundles_lock:
        _marks[key] = out
        while len(_marks) > 4:
            _marks.popitem(last=False)
    return out


HOW = {"metacyc": "by the MetaCyc ID in the odor table", "inchi": "by InChI", "inchikey": "by InChIKey", "cid": "by CID",
       "skeleton": "by InChIKey's first block (the same but for charge or stereochemistry)", "name": "by name"}


def odors_fingerprint(odors: list[dict]) -> str:
    """What of the odor table the matrices depend on, hashed."""
    keys = sorted((o["id"], o.get("name"), o.get("cid"), o.get("inchi"), o.get("metacyc"), tuple(o.get("aliases") or []))
                  for o in odors)
    return hashlib.sha1(json.dumps(keys, default=str).encode()).hexdigest()[:16]


def _row(b: Bundle, m: dict) -> dict:
    o, c = m["odor"], m["compound"]
    return {"label": o["name"], "group": None, "metacyc": c,
            "detail": f"MetaCyc: {b.name(c)} ({c}), matched {HOW[m['how']]}"}


def _ids(b: Bundle, frame: str) -> dict:
    """A compound's identifiers, as a reference dataset's chemicals are given them."""
    cid = (b.compounds.get(frame) or {}).get("cid")
    return {"metacyc": frame, **({"cid": cid} if cid else {})}


def _matched(m: Optional[dict]) -> Optional[list]:
    """The odor a row is (of `match_odors`), as `reference.chemicals` takes it: [id, how], or None."""
    return [m["odor"]["id"], m["how"]] if m else None


def _unmatched_note(matches: list[dict]) -> Optional[str]:
    missing = [m["odor"]["name"] for m in matches if m["compound"] is None]
    if not missing:
        return None
    shown = ", ".join(missing[:8]) + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
    return (f"{len(missing)} of the odor table's {len(matches)} chemicals are not in MetaCyc (by MetaCyc ID, InChI, CID or "
            f"name; a MetaCyc ID in the odor table matches one): {shown}")


# ------------------------------------------------------------------ the views


def parse(path: Path, fields: Optional[dict], odors: list[dict]) -> tuple[dict, np.ndarray | Sparse]:
    """A view of the bundle at `path`, its chemicals matched to `odors` (the odor table), as
    `reference.parse` gives a matrix: (record, values)."""
    try:
        b = bundle(path)
    except sqlite3.DatabaseError as e:
        raise MetacycError(f"not a MetaCyc bundle: {e}") from None
    o = options(fields)
    matches = match_odors(b, odors)
    record, values = (pathway_matrix if o["view"] == "pathways" else lca_matrix)(b, matches, o)
    record["metacyc"] = {**o, "version": b.about.get("version"), "built_at": b.about.get("built_at"), "described": describe(o)}
    return record, values


def _pathway_detail_line(b: Bundle, p: dict) -> str:
    types = [b.pathway_classes.get(t, {}).get("name") or t for t in p["types"]]
    parts = [" · ".join(types)]
    if p["taxonomic_range"]:
        parts.append("in " + ", ".join(p["taxonomic_range"][:4]) + ("…" if len(p["taxonomic_range"]) > 4 else ""))
    parts.append(f"{len(p['species'])} species" if len(p["species"]) != 1 else "1 species")
    return f"{p['id']} · " + " · ".join(x for x in parts if x)


def pathway_matrix(b: Bundle, matches: list[dict], o: dict) -> tuple[dict, Sparse]:
    """Compounds × the pathways they are in: the odor table's chemicals that are in MetaCyc
    first (in its order, each as its odor, even if in no pathway), then every other compound
    of MetaCyc in a pathway (or in one through a class it is of, as `o` says), by name."""

    def member(frame: str) -> tuple[dict[str, float], bool]:
        """A compound's pathways (by its value in each), and whether any is only through a class."""
        c = b.compounds.get(frame)
        if c is None:
            return {}, False
        got = {p: 1.0 for p in c["pathways"] if p in b.pathways}
        extra = {}
        if o["classes"] == "always" or (o["classes"] == "fallback" and not c["pathways"]):
            extra = {p: 0.5 for p in c["general_pathways"] if p not in c["pathways"] and p in b.pathways}
        return got | extra, bool(extra)

    odor_rows = [m for m in matches if m["compound"] is not None]
    taken = {m["compound"] for m in odor_rows}
    others = sorted((f for f in b.compounds if f not in taken), key=lambda f: (b.name(f).casefold(), f))
    rows = [(m["compound"], m, *member(m["compound"])) for m in odor_rows]
    rows += [(f, None, g, through) for f in others for g, through in [member(f)] if g]
    if not rows:
        raise MetacycError("no compound of MetaCyc is in a pathway")
    cols = sorted({p for _, _, g, _ in rows for p in g},
                  key=lambda p: ([b.pathway_classes.get(t, {}).get("name") or t for t in b.pathways[p]["types"]], b.pathways[p]["name"].casefold()))
    at = {p: j for j, p in enumerate(cols)}
    cells = [(i, at[p], v) for i, (_, _, g, _) in enumerate(rows) for p, v in g.items()]
    values = Sparse.of((len(rows), len(cols)), *(zip(*cells) if cells else ([], [], [])))
    n_class = sum(through for *_, through in rows)
    notes = [n for n in [_unmatched_note(matches)] if n]
    empty = [m["odor"]["name"] for _, m, g, _ in rows if m and not g]
    if empty:
        notes.append(f"{len(empty)} chemical{'s are' if len(empty) > 1 else ' is'} in MetaCyc but in no pathway"
                     f"{'' if o['classes'] == 'never' else ', even through a class'}: {', '.join(empty[:8])}{'…' if len(empty) > 8 else ''}")
    n_other = len(rows) - len(odor_rows)
    notes.append(f"its rows: the {len(odor_rows)} chemicals of the odor table in MetaCyc first, then the {n_other:,} other "
                 f"compounds of MetaCyc in a pathway{'' if o['classes'] == 'never' else ' (or through a class they are of)'}, by name")
    if n_class:
        notes.append(f"{n_class:,} compound{'s' if n_class > 1 else ''} in no pathway themselves shown in those of the classes they are of (0.5)"
                     if o["classes"] == "fallback" else f"{n_class:,} compounds also in the pathways of the classes they are of (0.5)")
    labels = Counter(b.pathways[p]["name"] for p in cols)
    names = Counter(b.name(f) for f, m, _, _ in rows if m is None)
    record = {
        "rows": [_row(b, m) if m else {"label": b.name(f) + (f" ({f})" if names[b.name(f)] > 1 else ""), "group": None, "metacyc": f,
                                       "detail": f"MetaCyc: {b.name(f)} ({f})"} for f, m, _, _ in rows],
        "cols": [{"label": b.pathways[p]["name"] + (f" ({p})" if labels[b.pathways[p]["name"]] > 1 else ""), "metacyc": p,
                  "detail": _pathway_detail_line(b, b.pathways[p])} for p in cols],
        "ids": {"rows": [_ids(b, f) for f, *_ in rows], "cols": None},
        "matched": [_matched(m) for _, m, _, _ in rows],
        "chemical_axis": "rows",
        "kind": "binary" if not n_class else "sequential",
        "shape": list(values.shape),
        "notes": notes,
    }
    return record, values


def lca_matrix(b: Bundle, matches: list[dict], o: dict) -> tuple[dict, np.ndarray]:
    """Chemicals (those of the odor table in MetaCyc's network) × themselves: their pathway LCA distance."""
    net = b.network(o["via"], o["max_pathways"])
    rows, entries = [], []
    for m in matches:
        if m["compound"] is None:
            continue
        e = net.entries(m["compound"], o["classes"])
        if e:
            rows.append(m)
            entries.append(e)
    if not rows:
        raise MetacycError("none of the odor table's chemicals is in MetaCyc's network of compounds")
    values = net.distances(entries)
    notes = [n for n in [_unmatched_note(matches)] if n]
    n_all = sum(1 for c in b.compounds if net.entries(c, o["classes"]))
    notes.append(f"only between the odor table's chemicals (made again whenever it changes): between all {n_all:,} "
                 f"compounds in the network, it would be {n_all * n_all:,} cells, too many to show")
    out = [m["odor"]["name"] for m in matches if m["compound"] is not None and m not in rows]
    if out:
        notes.append(f"{len(out)} chemical{'s are' if len(out) > 1 else ' is'} in MetaCyc but not in its network"
                     f"{'' if o['classes'] == 'never' else ', nor any class of '+('them' if len(out) > 1 else 'it')}: {', '.join(out[:8])}{'…' if len(out) > 8 else ''}")
    through = [(m["odor"]["name"], [b.name(x) for x, _ in e if x != m["compound"]]) for m, e in zip(rows, entries)
               if any(x != m["compound"] for x, _ in e)]
    if through:
        notes.append("taken as classes they are of: " + "; ".join(f"{n} as {', '.join(cs[:3])}" for n, cs in through[:10])
                     + ("…" if len(through) > 10 else ""))
    nan = int((~np.isfinite(values)).sum())
    if nan:
        notes.append(f"{nan // 2:,} pair{'s have' if nan > 2 else ' has'} no common ancestor: missing")
    notes.append("0 on the diagonal (a chemical with itself); between two chemicals, the fewest steps (edges) from a "
                 f"common ancestor other than themselves to each, summed: {describe(o)}")
    rlist = [_row(b, m) for m in rows]
    record = {
        "rows": rlist,
        "cols": [dict(r) for r in rlist],
        "ids": {"rows": [_ids(b, m["compound"]) for m in rows], "cols": None},
        "matched": [_matched(m) for m in rows],
        "chemical_axis": "rows",
        "pairwise": "distance",
        "kind": "sequential",
        "shape": list(values.shape),
        "notes": notes,
    }
    return record, values


# ------------------------------------------------------------------- the network


class Network:
    """MetaCyc as a directed graph whose nodes are its compounds (and classes of them, such
    as "a long-chain ketone", as reactions name them) and its reactions (`via` 'reactions':
    a reaction's reactants its parents, its products its children; as for O'Connell's
    thesis, ch. 2, in February 2024) or its pathways ('pathways': a pathway's reactants its
    parents, its products and every compound of its reactions its children; as the thesis's
    Methods describe). The compounds of `BLACKLIST` are left out, and with `max_pathways`,
    those in more than that many pathways.

    The pathway LCA distance of two compounds a and b: the smallest sum, over the nodes x
    other than a and b from which both can be reached, of the fewest edges from x to a and
    from x to b (x being their lowest common ancestor, LCA); missing if there is none.
    """

    def __init__(self, b: Bundle, via: str, max_pathways: Optional[int]):
        self.b, self.via, self.max_pathways = b, via, max_pathways
        drop = set(BLACKLIST)
        if max_pathways:
            drop |= {c for c, n in b.n_pathways.items() if n > max_pathways}
        self.dropped = drop
        edges: set[tuple[str, str]] = set()
        if via == "reactions":
            for rid, r in b.reactions.items():
                edges.update((c, rid) for c in r["reactants"] if c not in drop)
                edges.update((rid, c) for c in r["products"] if c not in drop)
        else:
            for pid, p in b.pathways.items():
                edges.update((c, pid) for c in p["reactants"] if c not in drop)
                edges.update((pid, c) for c in p["products"] if c not in drop)
                for rid in p["reactions"]:
                    r = b.reactions.get(rid)
                    if r:
                        edges.update((pid, c) for c in r["reactants"] + r["products"] if c not in drop)
        nodes = sorted({n for e in edges for n in e})
        self.ids = nodes
        self.at = {n: i for i, n in enumerate(nodes)}
        self.preds: list[list[int]] = [[] for _ in nodes]
        for u, v in edges:
            self.preds[self.at[v]].append(self.at[u])
        for p in self.preds:
            p.sort()

    def kind(self, node: str) -> str:
        if node in self.b.reactions:
            return "reaction"
        if node in self.b.pathways:
            return "pathway"
        return "compound" if node in self.b.compounds else "class"

    def up(self, node: str) -> tuple[np.ndarray, np.ndarray]:
        """Breadth-first up from `node`: (each node's fewest edges to it, inf where it cannot be
        reached from there; the next node on such a path toward it, -1 for none)."""
        n = len(self.ids)
        dist = np.full(n, np.inf, dtype=np.float32)
        nxt = np.full(n, -1, dtype=np.int64)
        start = self.at[node]
        dist[start] = 0
        q = deque([start])
        preds, d = self.preds, [0] * n
        seen = bytearray(n)
        seen[start] = 1
        while q:
            u = q.popleft()
            for p in preds[u]:
                if not seen[p]:
                    seen[p] = 1
                    d[p] = d[u] + 1
                    dist[p] = d[p]
                    nxt[p] = u
                    q.append(p)
        return dist, nxt

    def entries(self, compound: str, classes: str) -> list[tuple[str, int]]:
        """The nodes a compound is taken as (with the edges added on the way to it): itself,
        if it is in the network with a node above it; per `classes`, the classes it is of that
        are ('fallback': if it is not; 'always': as well) and, of the network through
        pathways, the pathways it is in through them (one edge to it)."""
        out = [(compound, 0)] if compound in self.at and self.preds[self.at[compound]] else []
        c = self.b.compounds.get(compound)
        if c and (classes == "always" or (classes == "fallback" and not out)):
            out += [(k, 0) for k in c["parents"] if k in self.at]
            if self.via == "pathways":
                out += [(p, 1) for p in c["general_pathways"] if p in self.at and p not in c["pathways"]]
        return list(dict.fromkeys(out))

    def _row(self, entries: list[tuple[str, int]]) -> np.ndarray:
        """Distances from every node to a chemical taken as `entries`, but its own nodes' (inf)."""
        dist = np.full(len(self.ids), np.inf, dtype=np.float32)
        for node, off in entries:
            dist = np.minimum(dist, self.up(node)[0] + off)
        for node, off in entries:
            if off == 0:
                dist[self.at[node]] = np.inf  # (an LCA is other than the two)
        return dist

    def distances(self, entries: list[list[tuple[str, int]]]) -> np.ndarray:
        """The pathway LCA distances between chemicals (each as its `entries`): n × n, 0 on the diagonal."""
        n = len(entries)
        d = np.stack([self._row(e) for e in entries]) if n else np.zeros((0, len(self.ids)), np.float32)
        d = d[:, (np.isfinite(d).sum(axis=0) >= 2)]  # (only nodes above two of them can be LCAs)
        out = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            out[i, i:] = (d[i][None, :] + d[i:]).min(axis=1) if d.shape[1] else np.inf
            out[i:, i] = out[i, i:]
        np.fill_diagonal(out, 0)
        out[~np.isfinite(out)] = np.nan
        return out

    def explain(self, a: list[tuple[str, int]], b: list[tuple[str, int]], most: int = 4) -> dict:
        """How far apart two chemicals (as their `entries`) are: ``distance`` (None if there is
        no common ancestor) and ``lcas``: of the nodes at it (`most` at most), each its
        ``node``, its ``paths`` to the two (lists of nodes, from it down to the entry each
        reaches), and those ``ends``: {entry, edges} (the edges from the entry to the chemical)."""
        ups = []
        for entries in (a, b):
            n = len(self.ids)
            dist, nxt, end = np.full(n, np.inf, dtype=np.float32), np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
            for node, off in entries:
                d, x = self.up(node)
                better = d + off < dist
                dist, nxt, end = np.where(better, d + off, dist), np.where(better, x, nxt), np.where(better, self.at[node], end)
            for node, off in entries:
                if off == 0:
                    dist[self.at[node]] = np.inf  # (an LCA is other than the two)
            ups.append((dist, nxt, end))
        total = ups[0][0] + ups[1][0]
        if not np.isfinite(total).any():
            return {"distance": None, "lcas": []}
        dmin = float(total.min())
        offs = [dict(e) for e in (a, b)]
        lcas = []
        for x in np.flatnonzero(total == dmin)[:most]:
            paths = [self._path(int(x), up) for up in ups]
            lcas.append({"node": self.ids[x], "paths": paths,
                         "ends": [{"entry": p[-1], "edges": off.get(p[-1], 0)} for p, off in zip(paths, offs)]})
        return {"distance": dmin, "lcas": lcas}

    def _path(self, x: int, up) -> list[str]:
        """The nodes from `x` down, by `up`'s next nodes, to the entry it reaches."""
        out = [self.ids[x]]
        while up[1][x] >= 0:
            x = int(up[1][x])
            out.append(self.ids[x])
        return out


# ------------------------------------------------------------------ looking into it


def _url(what: str, frame: str) -> str:
    return URL.format(what=what, id=frame)


def _species(b: Bundle, tax: str) -> dict:
    s = b.species.get(tax) or {"id": tax, "name": tax, "strain": None, "synonyms": []}
    genomes = b.genomes.get(tax, [])
    agg = {}
    for key in ("environment", "oxygen", "temperature", "pathogenicity", "trophic", "host"):
        vals = Counter(v for g in genomes for v in (g[key] or "").split(" · ") if v and not re.match(r"(?i)not (provided|available|collected|applicable)|missing|unknown", v))
        if key == "host":
            vals = Counter({(b.species.get(k) or {}).get("name") or k: n for k, n in vals.items()})
        agg[key] = [{"value": v, "n": n} for v, n in vals.most_common(4)]
    num = re.fullmatch(r"TAX-(\d+)", tax)
    return {"id": tax, "name": s["name"], "strain": s["strain"], "synonyms": s["synonyms"][:6],
            "url": TAXON_URL.format(id=num[1]) if num else None, "genomes": len(genomes), **agg}


def _class_lineage(b: Bundle, class_id: str) -> list[str]:
    """A pathway class's name, after those of the classes above it (as far as the exported tree tells)."""
    c = b.pathway_classes.get(class_id)
    if not c:
        return [class_id]
    out, name = [c["name"]], c.get("parent")
    while name and name not in out:
        out.insert(0, name)
        name = b.ontology.get(name)
    return out


def _compound_brief(b: Bundle, frame: str, odors: dict[str, dict]) -> dict:
    c = b.compounds.get(frame)
    return {"id": frame, "name": b.name(frame), "formula": c["formula"] if c else None, "cid": c["cid"] if c else None,
            "is_class": c is None, "n_pathways": b.n_pathways.get(frame, 0), "odor": odors.get(frame),
            "url": _url("compound", frame)}


def _side(b: Bundle, frame: str) -> bool:
    return frame in CURRENCY or b.n_pathways.get(frame, 0) > SIDE_PATHWAYS


def pathway(b: Bundle, pid: str, odors: dict[str, dict]) -> Optional[dict]:
    """A pathway, to draw (its reactions, each with its main and side compounds) and describe
    (classes, superpathways, species and their genomes, summary); None if there is none.

    Args:
        odors: frame ID -> the odor of the odor table it is ({id, name, abbrev}), to mark.
    """
    p = b.pathways.get(pid)
    if p is None:
        return None
    reactions, compounds = [], set()
    for rid in p["reactions"]:
        r = b.reactions.get(rid)
        if r is None:
            reactions.append({"id": rid, "equation": b.name(rid), "ec": [], "direction": "?", "left": [], "right": [],
                              "side_left": [], "side_right": [], "url": _url("reaction", rid)})
            continue
        sides = []
        for ids in (r["reactants"], r["products"]):
            main = [c for c in ids if not _side(b, c)]
            if not main and ids:  # (every one a side compound: the least common of them)
                main = [min(ids, key=lambda c: (b.n_pathways.get(c, 0), c))]
            sides.append((main, [c for c in ids if c not in main]))
        compounds.update(r["reactants"] + r["products"])
        reactions.append({"id": rid, "equation": r["equation"], "ec": r["ec"], "kegg": r["kegg"], "direction": r["direction"],
                          "left": sides[0][0], "right": sides[1][0], "side_left": sides[0][1], "side_right": sides[1][1],
                          "url": _url("reaction", rid)})
    main = {c for r in reactions for c in r["left"] + r["right"]}
    links = {}
    for c in main:
        others = [q for q in (b.compounds.get(c) or {}).get("pathways", []) if q != pid and q in b.pathways]
        if others:
            links[c] = [{"id": q, "name": b.pathways[q]["name"]} for q in others[:MAX_SIDE_PATHWAYS_SHOWN]] + (
                [{"id": None, "name": f"and {len(others) - MAX_SIDE_PATHWAYS_SHOWN} more"}] if len(others) > MAX_SIDE_PATHWAYS_SHOWN else [])
    return {
        "id": pid, "name": p["name"], "url": _url("pathway", pid),
        "types": [{"id": t, "name": b.pathway_classes.get(t, {}).get("name") or t, "lineage": _class_lineage(b, t)} for t in p["types"]],
        "super": [{"id": q, "name": b.pathways[q]["name"]} for q in p["super"] if q in b.pathways],
        "subs": [{"id": q, "name": b.pathways[q]["name"]} for q in p["subs"] if q in b.pathways],
        "species": [_species(b, t) for t in p["species"]],
        "taxonomic_range": p["taxonomic_range"], "engineered": p["engineered"], "summary": p["summary"],
        "primary_reactants": p["primary_reactants"], "primary_products": p["primary_products"],
        "reactants": p["reactants"], "products": p["products"],
        "reactions": reactions,
        "compounds": {c: _compound_brief(b, c, odors) for c in compounds | set(p["reactants"]) | set(p["products"])},
        "links": links,
        "version": b.about.get("version"),
    }


def compound(b: Bundle, frame: str, odors: dict[str, dict]) -> Optional[dict]:
    """A compound (or a class of them), with its classes, pathways and reactions; None if MetaCyc has none by that ID."""
    c = b.compounds.get(frame)
    if c is None and frame not in b.names:
        return None
    rx = b.of_compound.get(frame, {"consumed": [], "produced": []})
    pw = [{"id": q, "name": b.pathways[q]["name"]} for q in (c["pathways"] if c else []) if q in b.pathways]
    if c is None:  # a class: the pathways whose reactions it is in
        pw = [{"id": q, "name": v["name"]} for q, v in b.pathways.items()
              if any(frame in (b.reactions.get(r) or {}).get("reactants", []) + (b.reactions.get(r) or {}).get("products", [])
                     for r in v["reactions"])]
    direct = {q["id"] for q in pw}
    return {
        **_compound_brief(b, frame, odors),
        "smiles": c["smiles"] if c else None, "inchi": c["inchi"] if c else None, "inchikey": c["inchikey"] if c else None,
        "kegg": c["kegg"] if c else None,
        "classes": [{"id": k, "name": b.name(k)} for k in (c["parents"] if c else [])],
        "pathways": pw,
        "general_pathways": [{"id": q, "name": b.pathways[q]["name"]} for q in (c["general_pathways"] if c else [])
                             if q not in direct and q in b.pathways],
        "consumed": [{"id": r, "equation": b.reactions[r]["equation"]} for r in rx["consumed"][:50]],
        "produced": [{"id": r, "equation": b.reactions[r]["equation"]} for r in rx["produced"][:50]],
        "n_consumed": len(rx["consumed"]), "n_produced": len(rx["produced"]),
        "version": b.about.get("version"),
    }


def search(b: Bundle, query: str, most: int = 15) -> dict:
    """Pathways and compounds whose name or frame ID has `query` (but for case): the
    exact ones first, then those starting with it, then the shortest."""
    q = query.strip().casefold()
    if len(q) < 2:
        return {"pathways": [], "compounds": []}

    def ranked(items) -> list[dict]:
        hits = []
        for frame, name in items:
            n, f = (name or "").casefold(), frame.casefold()
            if q in n or q in f:
                hits.append(((n != q and f != q, not (n.startswith(q) or f.startswith(q)), len(n)), frame, name))
        return [{"id": f, "name": n} for _, f, n in sorted(hits)[:most]]

    return {"pathways": ranked((k, v["name"]) for k, v in b.pathways.items()),
            "compounds": ranked((k, v["name"]) for k, v in b.compounds.items())}


def explain(b: Bundle, fields: Optional[dict], a: str, c: str) -> dict:
    """The pathway LCA distance of two compounds (frame IDs) in a view's network, and the paths to them from their LCAs."""
    o = options({**(fields or {}), "view": "lca"})
    net = b.network(o["via"], o["max_pathways"])
    ea, ec = net.entries(a, o["classes"]), net.entries(c, o["classes"])
    if not ea or not ec:
        return {"distance": None, "lcas": [], "missing": [x for x, e in ((a, ea), (c, ec)) if not e], "nodes": {}}
    res = net.explain(ea, ec)
    nodes = {n for lca in res["lcas"] for p in lca["paths"] for n in p} | {a, c}
    return {**res, "missing": [], "described": describe(o),
            "entries": [[{"id": x, "name": b.name(x), "edges": off} for x, off in e] for e in (ea, ec)],
            "nodes": {n: {"id": n, "name": b.name(n), "kind": net.kind(n),
                          "url": _url({"reaction": "reaction", "pathway": "pathway"}.get(net.kind(n), "compound"), n),
                          **({"equation": b.reactions[n]["equation"],
                              "pathways": [{"id": q, "name": b.pathways[q]["name"]} for q in b.reactions[n]["pathways"] if q in b.pathways]}
                             if n in b.reactions else {})} for n in nodes}}
