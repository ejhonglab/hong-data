"""Foods' volatiles from VCF (Volatile Compounds in Food), for the Headspace calculator: a
reference dataset of one value per line (made into its matrix by `reference._parse_list`)
whose file also gives the Henry's law constant in water and the vapor pressure of each of
its chemicals (`PROPERTY_COLUMNS`, each with ``_units`` and ``_sources`` columns), as the
VCF CSVs made with chemutils do (e.g. ``2023-06-16_vcf_16.10_quant_w_groups_mid_ppm.csv``).
Its products are the labels of its columns (e.g. ``product_name``), with the ``category``
and ``group`` columns of their lines; its chemicals are its rows (e.g. ``name``, told
apart by ``inchi`` and ``cid``), and its values (e.g. ``mid_ppm``) are taken as mg/kg of
the sample, as VCF gives them.

A search (`search`) finds the lines of the products whose name, category or group has what
is typed in it, and reduces their quantities to one per chemical: those of one reference
(a paper's sample) on one product are averaged (as in the reference dataset's matrix),
then each product's are reduced across its references, then across products (by `STATS`,
the median by default), each over those that quantify the chemical. Its chemicals that are
one odor of the odor table are one, as in the Compare tab (`match_odors`, then merged in
each sample as the reference dataset's ``merge`` field says), and what may miscount them
is said (``ambiguity``). The headspace over
each sample, taken as water, is computed from these (`add_headspace`, by `headspace.sample`):
by Henry's law, else by Raoult's law with its vapor pressure. Both are looked up again
through chemutils (by main.py, kept apart from the odor table's), and what it looks up
is used rather than the file's, which is used where it finds none, then estimates
(`USED`); where the two differ (by more than `MISMATCH`), it is said.
"""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import headspace, reference

PROPERTY_COLUMNS = ("k_henry", "vapor_pressure")  # (either makes a reference dataset of one value per line VCF's)
# what else is read of the file, where it has them: of each line, and of each chemical (from its first line with each)
LINE_COLUMNS = ("category", "group", "reference", "quantity")
CHEMICAL_COLUMNS = ("vcf_url", "name", "inchi", "cid")
SEARCHED = ("product", "category", "group")  # what a search is matched against
# how quantities are reduced, across references and products: key -> label
STATS = {"median": "median", "mean": "mean", "geomean": "geometric mean", "max": "maximum", "min": "minimum"}
# units the file's values may be in (lowercased, without spaces) -> the factor to mol/m³/Pa, and kPa
# (mol/kg/bar of water: of 0.997 kg/L); with no units given, these
K_HENRY_UNITS = {"mol/m^3/pa": 1.0, "mol/m**3/pa": 1.0, "mol/(m^3*pa)": 1.0, "mol/kg/bar": 0.997e3 / 1e5,
                 "mol/l/atm": 1e3 / headspace.P_ATM, "m/atm": 1e3 / headspace.P_ATM}
VAPOR_PRESSURE_UNITS = {"kpa": 1.0, "pa": 1e-3, "hpa": 0.1, "mbar": 0.1, "bar": 100.0, "atm": headspace.P_ATM / 1e3,
                        "mmhg": 0.133322, "torr": 0.133322}
# standard atomic weights (g/mol), for molar masses from InChIs' formulas
ATOMIC_MASSES = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Mo": 95.95, "Ag": 107.87, "Sn": 118.71, "I": 126.90, "Hg": 200.59, "Pb": 207.2,
}
MEMO_SIZE = 2  # files kept read
# what each of the Henry's law constant and the vapor pressure used is, in order: looked up through
# chemutils, else the file's, else this estimate looked up through chemutils (`chemprops.PROPERTIES`)
USED = {"k_henry": "est_k_henry", "vapor_pressure": "est_vapor_pressure"}
MISMATCH = 0.1  # the relative difference beyond which a value looked up and the file's are said to differ

_memo: "OrderedDict[tuple, dict]" = OrderedDict()
_memo_lock = threading.Lock()


class VcfError(ValueError):
    pass


def molar_mass(inchi: Optional[str]) -> Optional[float]:
    """A chemical's molar mass (g/mol) from the formula of its InChI (of each of its parts),
    None if it has none, or an element not in `ATOMIC_MASSES`."""
    m = re.match(r"(?:InChI=)?1S?/([^/]+)", inchi or "")
    if not m:
        return None
    total = 0.0
    for part in m[1].split("."):
        k = re.fullmatch(r"(\d*)((?:[A-Z][a-z]?\d*)+)", part)
        if not k:
            return None
        for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", k[2]):
            if element not in ATOMIC_MASSES:
                return None
            total += int(k[1] or 1) * int(count or 1) * ATOMIC_MASSES[element]
    return total


def _columns(path: Path) -> list[str]:
    """The names of a table file's columns (of its header alone)."""
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        return [str(c) for c in pq.read_schema(path).names]
    if suffix not in (".csv", ".tsv", ".tab", ".txt"):
        return []
    sep = {".csv": ",", ".tsv": "\t", ".tab": "\t"}.get(suffix)
    return [str(c) for c in pd.read_csv(path, nrows=0, sep=sep, engine="c" if sep else "python").columns]


def sources(entries: list[dict]) -> list[dict]:
    """Those of the reference datasets (`reference.entries`) that are VCF's (see the module's
    docstring), which can be read: each ``id``, ``title``, ``path`` and ``fields`` (as
    `reference.parse` takes them)."""
    out = []
    for e in entries:
        fields = reference._fields(e.get("fields"))
        if e["state"] != "ok" or not _searched(fields):
            continue
        if headspace_use(e["path"], fields):
            out.append({"id": e["id"], "title": e["title"], "path": e["path"], "fields": fields})
    return out


def _searched(fields: Optional[dict]) -> bool:
    return bool(fields) and fields.get("layout") == "list" and bool(fields.get("value"))


def headspace_use(path: str, fields: Optional[dict]) -> Optional[dict]:
    """Of a file that is VCF's (with `PROPERTY_COLUMNS`), how the Headspace calculator would
    take a reference dataset made of it by `fields` (as `reference.parse` takes them):
    ``used`` (searched: one value per line, of a field of values), ``value`` (that field)
    and ``merge`` (of its chemicals that are one odor); None if the file is not VCF's (or
    cannot be read)."""
    try:
        names = _columns(Path(path))
    except Exception:
        return None
    if not any(c in names for c in PROPERTY_COLUMNS):
        return None
    f = fields or {}
    return {"used": _searched(f), "value": f.get("value"), "merge": f.get("merge", reference.MERGES[0])}


def _property(values: Optional[str], units: Optional[str], factors: dict[str, float]) -> tuple[Optional[float], Optional[str]]:
    """A property of a chemical as the file gives it, in the units of `factors`' first, and
    why it is not used, if it is given but is not used."""
    try:
        v = float(values) if values is not None else None
    except ValueError:
        return None, f"not a number ({values!r})"
    if v is None or not np.isfinite(v) or v <= 0:
        return None, None if v is None else f"not a positive number ({values})"
    key = re.sub(r"\s+", "", units or "").lower()
    if key and key not in factors:
        return None, f"in units it does not know ({units})"
    return v * factors.get(key, 1.0), None


def _read(source: dict) -> dict:
    """`load`, without keeping it."""
    fields, path = source["fields"], Path(source["path"])
    df = reference.read_rows(path)
    df.columns = [str(c) for c in df.columns]
    names = list(df.columns)
    row_keys = [fields["rows"]["label"], *fields["rows"].get("keep", [])]
    product, value = fields["cols"]["label"], fields["value"]
    for k in (*row_keys, product, value):
        if k not in names:
            raise VcfError(f"its file has no field {k!r}, which its reference dataset is made with")
    text = lambda k: df[k].map(reference._text) if k in names else pd.Series([None] * len(df), index=df.index, dtype=object)
    keys = pd.DataFrame({k: text(k) for k in dict.fromkeys(row_keys)})
    lines = pd.DataFrame({"product": text(product), **{k: text(k) for k in LINE_COLUMNS}})
    lines["value"] = pd.to_numeric(df[value], errors="coerce").astype(float)
    keep = keys[row_keys[0]].notna() & lines["product"].notna()
    keys, lines = keys[keep], lines[keep].reset_index(drop=True)
    lines["chem"], _ = pd.factorize(pd.Series(["\x1f".join(map(str, t)) for t in keys.itertuples(index=False)]))
    if lines["reference"].isna().all():  # (each line of its own)
        lines["reference"] = [f"line {i + 2}" for i in np.flatnonzero(keep)]

    # each chemical's identifiers and properties: of its first line with each
    per = pd.DataFrame({k: text(k)[keep].reset_index(drop=True) for k in
                        (*CHEMICAL_COLUMNS, *(f"{p}{s}" for p in PROPERTY_COLUMNS for s in ("", "_units", "_sources")))})
    per["label"] = keys[row_keys[0]].reset_index(drop=True)
    first = per.groupby(lines["chem"].to_numpy(), sort=True).first()
    chemicals = []
    for r in first.to_dict("records"):
        r = {k: reference._text(v) for k, v in r.items()}
        inchi = reference._id_value("inchi", r["inchi"])
        c = {"name": r["name"] or r["label"], "inchi": inchi, "cid": reference._id_value("cid", r["cid"]), "url": r["vcf_url"],
             "molar_mass": molar_mass(inchi), "notes": []}
        for p, factors, label in (("k_henry", K_HENRY_UNITS, "Henry's law constant"),
                                  ("vapor_pressure", VAPOR_PRESSURE_UNITS, "vapor pressure")):
            c[p], why = _property(r[p], r[f"{p}_units"], factors)
            c[f"{p}_source"] = r[f"{p}_sources"] if c[p] is not None else None
            if why:
                c["notes"].append(f"its {label} in the file is {why}, so not used")
        chemicals.append(c)
    folded = {k: lines[k].fillna("").str.casefold() for k in SEARCHED}
    names_of = lambda k: sorted(set(lines[k].dropna()), key=str.casefold)
    return {"lines": lines, "folded": folded, "chemicals": chemicals,
            "names": {"products": names_of("product"), "categories": names_of("category"), "groups": names_of("group")}}


def load(source: dict) -> dict:
    """A VCF reference dataset's file read (of `sources`), kept in memory for the file as it
    is now.

    Returns:
        ``lines`` (a DataFrame, of each line with labels for its row and column: ``chem``
        (its index in ``chemicals``), ``product``, ``category``, ``group``, ``reference``,
        ``quantity`` (as written; None where not given) and ``value`` (NaN if not a
        number)), ``chemicals`` (each ``name``, ``inchi``, ``cid``, ``url`` (VCF's page of
        it), ``molar_mass`` (from its InChI), ``k_henry`` (mol/m³/Pa) and ``vapor_pressure``
        (kPa), each with ``_source`` (None where not given, or not used: then said why in
        ``notes``)), ``names`` (``products``, ``categories``, ``groups``: sorted).

    Raises:
        VcfError: if it is not made of fields the file has.
    """
    st = os.stat(source["path"])
    key = (source["path"], st.st_size, st.st_mtime_ns, repr(sorted(source["fields"].items())))
    with _memo_lock:
        if key in _memo:
            _memo.move_to_end(key)
            return _memo[key]
    out = _read(source)
    with _memo_lock:
        _memo[key] = out
        while len(_memo) > MEMO_SIZE:
            _memo.popitem(last=False)
    return out


def _reduce(values: pd.Series, by, stat: str) -> pd.Series:
    """`values` reduced by `stat` (of `STATS`) in each group of `by` (as `Series.groupby` takes it)."""
    if stat == "geomean":
        with np.errstate(divide="ignore"):
            return np.exp(np.log(values).groupby(by, sort=False).mean())
    return values.groupby(by, sort=False).agg(stat)


def match_odors(table: dict, index) -> list[dict]:
    """The odor of the odor table each chemical of a file `load` read is, matched as the
    chemicals of its reference dataset are (`OdorIndex.match_chemical`, by its CID, InChI
    and name): ``odor_id`` (None if none), ``how``, ``odor`` (its name), and those it could
    be or may be, ``candidates`` (if ambiguous) and ``maybe`` (if it has no CID or InChI,
    by its name, `OdorIndex.maybe`), each [{id, name}]."""
    out = []
    named = lambda ids: [{"id": i, "name": index.by_id[i]["name"]} for i in ids]  # noqa: E731
    for c in table["chemicals"]:
        cands: list[int] = []
        odor_id, how = index.match_chemical({k: c[k] for k in ("cid", "inchi", "name") if c[k]}, c["name"], cands)
        out.append({"odor_id": odor_id, "how": how, "odor": index.by_id[odor_id]["name"] if odor_id is not None else None,
                    "candidates": named(cands),
                    "maybe": named(index.maybe(c["name"])) if odor_id is None and not (c["cid"] or c["inchi"]) else []})
    return out


_MERGE = {"mean": "mean", "sum": "sum", "max": "max"}  # (reference.MERGES, as pandas names them)


def search(table: dict, query: str, across_references: str = "median", across_products: str = "median",
           matches: Optional[list[dict]] = None, merge: str = "mean") -> dict:
    """The lines of the products (of a file `load` read) whose name, category or group has
    `query` in it (but for case), and one quantity per chemical from them (see the
    module's docstring); the headspace over the samples is from `add_headspace`. Its
    chemicals that are one odor of the odor table (`matches`) are one, as in the Compare
    tab: in each sample (a reference on a product), their quantities (each its lines'
    mean) made one by `merge`, before being reduced across references and products.

    Args:
        across_references, across_products: of `STATS`.
        matches: of each chemical of `table`, as `match_odors` gives them (None: none are
            odors of the odor table).
        merge: of `reference.MERGES`: as its reference dataset's ``merge`` field says.

    Returns:
        ``query``, ``products`` (those matched, by name: ``name``, ``category``, ``group``,
        and how many ``lines``, ``references`` and ``chemicals`` of it), ``references``
        (their texts), ``lines`` (each [its chemical's index in ``chemicals``, its
        product's in ``products``, its reference's in ``references``, ``quantity``,
        ``value``, the name of its chemical in the file]), ``chemicals`` (each of the
        lines, but those of one odor one, those with the highest quantity first: as `load`
        gives them (of several of one odor, of the one that is the odor's own CID or
        InChI, else with the most lines; named by the odor), and ``odor_id``; ``parts``
        (those of the file it is: ``name``, ``url``, ``cid``, ``inchi``, ``how`` it is
        the odor, and ``lines``), ``ppm`` (the quantity of it, from those of each product,
        ``per_product``: [product index, quantity, references]), ``lines`` (how many),
        ``references`` and ``products`` (how many quantify it), ``min`` and ``max`` (of
        its lines); and ``ambiguity``: ``stereo`` (the names of its parts that are the
        odor only as the same but for stereochemistry), ``overlap`` (None, or of the
        ``samples`` (of ``of``, those that quantify it) where two of its parts that
        `reference.may_overlap` are both given, which ``parts``), ``candidates`` and
        ``maybe`` (as `match_odors` gives them, of a chemical not an odor)).

    Raises:
        VcfError: if `query` is empty, or a stat is not one of `STATS`.
    """
    q = query.strip().casefold()
    if not q:
        raise VcfError("type part of a product's name, its category or group")
    for stat in (across_references, across_products):
        if stat not in STATS:
            raise VcfError(f"quantities are reduced by one of {', '.join(STATS)}")
    if merge not in _MERGE:
        raise VcfError(f"quantities of one odor are merged by one of {', '.join(_MERGE)}")
    chemicals = table["chemicals"]
    matches = matches or [{"odor_id": None, "how": "none", "odor": None, "candidates": [], "maybe": []}] * len(chemicals)
    lines = table["lines"]
    mask = np.zeros(len(lines), dtype=bool)
    for k in SEARCHED:
        mask |= table["folded"][k].str.contains(q, regex=False).to_numpy()
    m = lines[mask].copy()
    products = sorted(set(m["product"]), key=str.casefold)
    product_of = {p: i for i, p in enumerate(products)}
    refs = list(dict.fromkeys(m["reference"]))
    ref_of = {r: i for i, r in enumerate(refs)}
    # each chemical, or each odor of the odor table of several, a unit
    key = lambda chem: ("odor", matches[chem]["odor_id"]) if matches[chem]["odor_id"] is not None else ("chem", chem)  # noqa: E731
    keys = list(dict.fromkeys(key(c) for c in m["chem"]))
    unit_of = {k: u for u, k in enumerate(keys)}
    m["unit"] = [unit_of[key(c)] for c in m["chem"]]
    parts: list[list[int]] = [[] for _ in keys]
    for c in dict.fromkeys(m["chem"]):
        parts[unit_of[key(c)]].append(c)

    q_lines = m[np.isfinite(m["value"].to_numpy())]
    per_chem = q_lines.groupby(["chem", "product", "reference"], sort=False)["value"].mean()
    pc = per_chem.reset_index()
    pc["unit"] = [unit_of[key(c)] for c in pc["chem"]]
    per_ref = pc.groupby(["unit", "product", "reference"], sort=False)["value"].agg(_MERGE[merge])
    levels = [per_ref.index.get_level_values(0), per_ref.index.get_level_values(1)]
    per_product = _reduce(per_ref, levels, across_references)
    n_refs = per_ref.groupby(levels, sort=False).size()
    consensus = _reduce(per_product, per_product.index.get_level_values(0), across_products)
    stats = q_lines.groupby("unit")["value"].agg(["min", "max"])
    n_lines = m.groupby("unit").size()
    n_chem_lines = m.groupby("chem").size()
    n_unit_refs = per_ref.groupby(level=0).size()
    overlaps = _overlaps(pc, parts, chemicals)

    of_unit: dict[int, list] = {}
    for (unit, prod), v in per_product.items():
        of_unit.setdefault(unit, []).append([product_of[prod], float(v), int(n_refs[(unit, prod)])])
    out = []
    for unit, chems in enumerate(parts):
        # (of several: the odor's own CID or InChI first, then the most lines)
        rep = min(chems, key=lambda c: (matches[c]["how"] not in ("cid", "inchi"), -int(n_chem_lines[c]), c))
        c, mt = chemicals[rep], matches[rep]
        pp = sorted(of_unit.get(unit, []))
        out.append({**c, "notes": list(c["notes"]), "unit": unit, "odor_id": mt["odor_id"],
                    **({"name": mt["odor"]} if len(chems) > 1 else {}),
                    "parts": [{"name": chemicals[x]["name"], "url": chemicals[x]["url"], "cid": chemicals[x]["cid"],
                               "inchi": chemicals[x]["inchi"], "how": matches[x]["how"], "lines": int(n_chem_lines[x])} for x in chems],
                    "ppm": float(consensus[unit]) if unit in consensus.index else None, "per_product": pp,
                    "lines": int(n_lines[unit]), "references": int(n_unit_refs.get(unit, 0)), "products": len(pp),
                    "min": float(stats.loc[unit, "min"]) if unit in stats.index else None,
                    "max": float(stats.loc[unit, "max"]) if unit in stats.index else None,
                    "ambiguity": {"stereo": [chemicals[x]["name"] for x in chems if matches[x]["how"] == "stereo"],
                                  "overlap": overlaps.get(unit) and {**overlaps[unit], "of": int(n_unit_refs.get(unit, 0))},
                                  "candidates": mt["candidates"], "maybe": mt["maybe"]}})
    out.sort(key=lambda c: (c["ppm"] is None, -(c["ppm"] or 0), c["name"].casefold()))
    index = {c.pop("unit"): i for i, c in enumerate(out)}

    by_product = m.groupby("product")
    summary = []
    for p in products:
        g = by_product.get_group(p)
        summary.append({"name": p, "category": g["category"].dropna().iloc[0] if g["category"].notna().any() else None,
                        "group": g["group"].dropna().iloc[0] if g["group"].notna().any() else None,
                        "lines": len(g), "references": g["reference"].nunique(), "chemicals": g["unit"].nunique()})
    return {"query": query.strip(), "products": summary, "references": refs, "chemicals": out, "merge": merge,
            "lines": [[index[u], product_of[p], ref_of[r], qty, None if not np.isfinite(v) else float(v), chemicals[ch]["name"]]
                      for u, ch, p, r, qty, v in zip(m["unit"], m["chem"], m["product"], m["reference"], m["quantity"], m["value"])]}


def _overlaps(per_chem: pd.DataFrame, parts: list[list[int]], chemicals: list[dict]) -> dict[int, dict]:
    """Of each unit of several chemicals (`search`): the samples (a reference on a product) in
    which two of them that `reference.may_overlap` both have a quantity (not 0), and which
    (``samples``, ``parts``: their names); of those with any."""
    out = {}
    for unit, chems in enumerate(parts):
        if len(chems) < 2:
            continue
        pairs = [(a, b) for x, a in enumerate(chems) for b in chems[x + 1:]
                 if reference.may_overlap(chemicals[a]["inchi"], chemicals[b]["inchi"])]
        if not pairs:
            continue
        g = per_chem[(per_chem["unit"] == unit) & (per_chem["value"] != 0)]
        given = g.groupby(["product", "reference"], sort=False)["chem"].agg(set)
        hit = [s for s in given if any(a in s and b in s for a, b in pairs)]
        if hit:
            involved = set().union(*({a, b} for s in hit for a, b in pairs if a in s and b in s))
            out[unit] = {"samples": len(hit), "parts": [chemicals[c]["name"] for c in chems if c in involved]}
    return out


def add_headspace(chemicals: list[dict], looked: Optional[dict] = None, pending: Optional[set] = None) -> None:
    """The headspace over the samples of `search`'s chemicals (`headspace.sample`), each
    with its Henry's law constant and vapor pressure as used (`USED`), into each.

    Args:
        looked: CID -> property -> what was looked up of it through chemutils (as
            `db.vcf_properties` gives them).
        pending: (CID, property) of those being looked up.

    Adds to each: ``k_henry`` and ``vapor_pressure`` (as used, None if none), each with
    ``_source``, ``_from`` ('chemutils', 'file' or 'estimate') and ``_measured``; ``file``
    (the file's: ``k_henry``, ``vapor_pressure`` and their ``_source``), ``mismatches``
    (each ``prop``, ``file`` and ``looked_up``: where they differ), ``pending`` (those of
    `USED` being looked up), ``pa`` (Pa, of its quantity; None if it cannot be computed),
    ``per_product_pa`` (of each product's, in the order of ``per_product``), ``law``
    ('henry', 'raoult' or None) and ``headspace_notes``.
    """
    looked, pending = looked or {}, pending or set()
    for c in chemicals:
        lk = looked.get(c["cid"], {}) if c["cid"] else {}
        value = lambda p: (lk.get(p) or {}).get("value")
        c["file"] = {k: c[k] for k in ("k_henry", "k_henry_source", "vapor_pressure", "vapor_pressure_source")}
        c["mismatches"], notes = [], []
        for p, est in USED.items():
            mine, theirs = value(p), c["file"][p]
            if mine is not None:
                c.update({p: mine, f"{p}_source": lk[p]["source"], f"{p}_from": "chemutils", f"{p}_measured": lk[p]["measured"]})
                if theirs is not None and abs(mine - theirs) > MISMATCH * max(mine, theirs):
                    c["mismatches"].append({"prop": p, "file": theirs, "looked_up": mine})
            elif theirs is not None:
                c.update({f"{p}_from": "file", f"{p}_measured": None})
            elif value(est) is not None:
                c.update({p: value(est), f"{p}_source": lk[est]["source"], f"{p}_from": "estimate", f"{p}_measured": False})
            else:
                c.update({p: None, f"{p}_source": None, f"{p}_from": None, f"{p}_measured": None})
        if c["k_henry_from"] == "estimate":
            notes.append("its Henry's law constant estimated from its solubility in water over its vapor pressure (VP/AS), "
                         "which is only reasonable for chemicals not very soluble in water")
        c["pending"] = [p for p in (*USED, *USED.values()) if (c["cid"], p) in pending]
        props = lambda ppm: {"ppm": ppm, **{k: c[k] for k in ("molar_mass", "k_henry", "vapor_pressure")}}
        if c["ppm"] is None:
            c.update(pa=None, law=None, headspace_notes=["no quantity given (only that it is there)"])
        else:
            (r,) = headspace.sample([props(c["ppm"])])
            c.update(pa=r["pa"], law=r["law"], headspace_notes=notes + r["notes"])
        c["per_product_pa"] = [r["pa"] for r in headspace.sample([props(v) for _, v, _ in c["per_product"]])]
