"""Odor (chemical) metadata: the app's own table, synced one way from a Google sheet.

The sheet has one row per chemical, with the columns of
``config/odor_sheet_template.csv`` (header names are matched loosely, so the lab's
existing inventory sheet's "Chemical", "Abbreviation", "CAS #", ... also work):

    name, abbrev, aliases (;-separated), cid (PubChem), cas, solvent, supplier,
    catalog_no, catalog_url, metacyc, lotus, inventory, notes

It is read through its CSV export (shared as "anyone with the link can view"); nothing
is written to it. Syncing is two steps: `preview` validates the sheet (or an uploaded
CSV) and diffs it against the database, and the caller then applies the new odors plus
only those changes and deletions a user confirmed. The database copy is what the rest
of the app uses, so it keeps working without the sheet.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from . import config, yamlcheck
from .db import ODOR_COLUMNS as FIELDS
from .odors import SOLVENT_STR, OdorFormatError, format_conc, parse_odor

# accepted header names (lower case, '_' and runs of spaces as one space, trailing ' .:#' dropped)
HEADERS = {
    "name": ("name", "chemical", "common name", "compound"),
    "abbrev": ("abbrev", "abbreviation", "abbr"),
    "aliases": ("aliases", "alias", "other names", "synonyms"),
    "cid": ("cid", "pubchem cid", "pubchem id"),
    "cas": ("cas", "cas number", "cas no", "cas rn"),
    "solvent": ("solvent",),
    "supplier": ("supplier", "vendor"),
    "catalog_no": ("catalog no", "catalog number", "catalog", "product number", "product no", "sku", "sigma sku"),
    "catalog_url": ("catalog url", "product url", "url"),
    "metacyc": ("metacyc", "metacyc id", "biocyc id"),
    "lotus": ("lotus", "lotus id"),
    "inventory": ("inventory", "inventory codes"),
    "notes": ("notes", "note", "comments"),
}
_HEADER_FIELD = {h: f for f, hs in HEADERS.items() for h in hs}
_CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
_LOTUS_RE = re.compile(r"^LTS\d{7}$")
SIGMA_RE = re.compile(r"sigma|aldrich|merck|millipore", re.I)


class SheetError(Exception):
    pass


# --------------------------------------------------------------------- parsing


def _header_key(h: str) -> str:
    return re.sub(r"[\s_]+", " ", h.strip().lower()).rstrip(" .:#")


def _cas_ok(cas: str) -> bool:
    digits = cas.replace("-", "")
    return int(digits[-1]) == sum(int(c) * (i + 1) for i, c in enumerate(reversed(digits[:-1]))) % 10


def parse_sheet(text: str) -> tuple[list[dict], list[str], list[str]]:
    """Odor rows of a sheet (CSV text), with every problem found.

    Returns:
        (rows, errors, warnings). Each row has all `FIELDS` (None when empty; aliases
        a list; cid an int) plus ``row``, its line in the sheet. Errors (missing name
        column or names, duplicate names, malformed CID / CAS / LOTUS ID / URL) mean
        the sheet should not be synced; warnings (ignored columns, shared
        abbreviations) do not.
    """
    lines = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    errors: list[str] = []
    warnings: list[str] = []
    h = next((i for i, r in enumerate(lines) if any(c.strip() for c in r)), None)
    if h is None:
        return [], ["the sheet is empty"], []
    cols: dict[int, str] = {}
    ignored = []
    for j, raw in enumerate(lines[h]):
        f = _HEADER_FIELD.get(_header_key(raw))
        if f and f not in cols.values():
            cols[j] = f
        elif raw.strip():
            ignored.append(raw.strip())
    if "name" not in cols.values():
        return [], [f"no 'name' column; the first row should be the column names ({', '.join(FIELDS)})"], []
    if ignored:
        warnings.append(f"ignored columns: {', '.join(ignored)}")

    rows: list[dict] = []
    for i in range(h + 1, len(lines)):
        cells = {f: lines[i][j].strip() if j < len(lines[i]) else "" for j, f in cols.items()}
        if not any(cells.values()):
            continue
        line = i + 1
        r: dict[str, Any] = {f: (cells.get(f) or None) for f in FIELDS}
        r["row"] = line
        r["name"] = re.sub(r"\s+", " ", r["name"]) if r["name"] else None
        if not r["name"]:
            errors.append(f"row {line} has no name")
            continue
        r["aliases"] = []
        for a in re.split(r"[;\n]", cells.get("aliases", "")):
            a = a.strip()
            if a and a != r["name"] and a not in r["aliases"]:
                r["aliases"].append(a)
        if r["cid"]:
            try:
                r["cid"] = int(float(r["cid"]))
                if r["cid"] <= 0:
                    raise ValueError
            except ValueError:
                errors.append(f"row {line} ({r['name']}): CID {cells['cid']!r} is not a PubChem compound number")
                r["cid"] = None
        if r["cas"] and not (_CAS_RE.match(r["cas"]) and _cas_ok(r["cas"])):
            errors.append(f"row {line} ({r['name']}): {r['cas']!r} is not a valid CAS number")
        if r["lotus"] and not _LOTUS_RE.match(r["lotus"]):
            errors.append(f"row {line} ({r['name']}): LOTUS ID {r['lotus']!r} should look like LTS0123456")
        if r["catalog_url"] and not re.match(r"^https?://", r["catalog_url"]):
            errors.append(f"row {line} ({r['name']}): catalog URL {r['catalog_url']!r} should start with http(s)://")
        rows.append(r)

    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["name"].casefold(), []).append(r)
    for same in by_name.values():
        if len(same) > 1:
            errors.append(f"name {same[0]['name']!r} is on more than one row (rows {', '.join(str(r['row']) for r in same)})")
    labels: dict[str, set[str]] = {}
    for r in rows:
        for lab in [r["abbrev"], *r["aliases"]]:
            if lab:
                labels.setdefault(lab, set()).add(r["name"])
    for lab, names in labels.items():
        if len(names) > 1:
            warnings.append(f"abbreviation/alias {lab!r} is used by {len(names)} odors ({', '.join(sorted(names))}): "
                            "datasets without an odor YAML can't tell which one they mean")
    return rows, errors, warnings


def fingerprint(rows: list[dict]) -> str:
    """Hash of the parsed content (so edits to ignored columns don't count as changes)."""
    content = [{f: r[f] for f in FIELDS} for r in rows]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------ diff


def _comparable(odor: dict, f: str):
    v = odor.get(f)
    if f == "aliases":
        return sorted(v or [])
    return v if v not in ("", None) else None


def diff(rows: list[dict], odors: list[dict]) -> dict:
    """What syncing `rows` (from the sheet) into `odors` (the database) would do.

    Sheet rows are matched to database odors by name; rows whose name is new are then
    matched to a not-yet-matched odor with the same CID, abbreviation or an alias (a
    rename), so that editing a name in the sheet does not look like a new odor.

    Returns:
        dict with ``added`` (rows), ``changed`` ([{id, odor, row, fields: [{field, old,
        new}], rename}]), ``missing`` (database odors not in the sheet) and
        ``unchanged`` (count).
    """
    by_key = {o["name"].casefold(): o for o in odors}
    matched: dict[int, dict] = {}  # odor id -> row
    pending = []
    for r in rows:
        o = by_key.get(r["name"].casefold())
        if o is not None and o["id"] not in matched:
            matched[o["id"]] = r
        else:
            pending.append(r)
    free = lambda: [o for o in odors if o["id"] not in matched]
    added = []
    for r in pending:
        candidates = []
        for test in (
            lambda o: r["cid"] is not None and o.get("cid") == r["cid"],
            lambda o: r["abbrev"] is not None and o.get("abbrev") == r["abbrev"],
            lambda o: bool({r["name"], *r["aliases"]} & {o["name"], *(o.get("aliases") or [])}),
        ):
            candidates = [o for o in free() if test(o)]
            if candidates:
                break
        if len(candidates) == 1:
            matched[candidates[0]["id"]] = r
        else:
            added.append(r)
    changed, unchanged = [], 0
    for o in odors:
        r = matched.get(o["id"])
        if r is None:
            continue
        fields = [{"field": f, "old": o.get(f), "new": r[f]} for f in FIELDS if _comparable(o, f) != _comparable(r, f)]
        if fields:
            changed.append({"id": o["id"], "odor": o, "row": r, "fields": fields, "rename": o["name"] != r["name"]})
        else:
            unchanged += 1
    missing = [o for o in odors if o["id"] not in matched]
    return {"added": added, "changed": changed, "missing": missing, "unchanged": unchanged}


def after_sync(odors: list[dict], d: dict) -> list[dict]:
    """The odor table if every addition and change in `d` (a `diff`) were applied (deleting nothing)."""
    new = {c["id"]: {"id": c["id"], **{f: c["row"][f] for f in FIELDS}} for c in d["changed"]}
    out = [new.get(o["id"], o) for o in odors]
    out += [{"id": -(k + 1), **{f: r[f] for f in FIELDS}} for k, r in enumerate(d["added"])]
    return out


# ----------------------------------------------------------------------- sheet


def export_url(sheet_url: str) -> str:
    """CSV export URL of a Google sheet, from its link as copied from the browser."""
    u = urllib.parse.urlparse(sheet_url.strip())
    m = re.match(r"^/spreadsheets/d/([A-Za-z0-9_-]+)", u.path)
    if u.netloc != "docs.google.com" or not m:
        raise SheetError(f"not a Google sheet link: {sheet_url}")
    gid = re.search(r"gid=(\d+)", f"{u.query}#{u.fragment}")
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid.group(1) if gid else 0}"


def fetch_sheet(sheet_url: str, timeout: float = 20) -> str:
    """The sheet's CSV text. Raises SheetError if it can't be read."""
    url = export_url(sheet_url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as res:
            ctype = res.headers.get("content-type", "")
            body = res.read()
    except Exception as e:
        raise SheetError(f"could not read the Google sheet ({e})") from None
    if "csv" not in ctype:
        raise SheetError("the Google sheet did not come back as CSV: is it shared as 'anyone with the link can view'?")
    return body.decode("utf-8", errors="replace")


class SheetWatch:
    """Fingerprint of the odor sheet, refreshed at most every `interval_s`."""

    UNREAD = {"checked_at": None, "fingerprint": None, "n_rows": None, "error": None}

    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self.lock = threading.Lock()
        self.url: Optional[str] = None
        self.checked = 0.0
        self.state: dict[str, Any] = dict(self.UNREAD)

    def check(self, url: str, force: bool = False) -> dict:
        """Current state of the sheet at `url` (read first if the last read is too old, or of another sheet)."""
        if not url:
            return dict(self.UNREAD)
        with self.lock:
            if force or url != self.url or time.time() - self.checked > self.interval_s:
                self.url, self.checked = url, time.time()
                state = {"checked_at": dt.datetime.now().isoformat(timespec="seconds"), "error": None}
                try:
                    rows, _, _ = parse_sheet(fetch_sheet(url))
                    state.update(fingerprint=fingerprint(rows), n_rows=len(rows))
                except SheetError as e:
                    state.update(fingerprint=None, n_rows=None, error=str(e))
                self.state = state
            return dict(self.state)

    def record(self, url: str, rows: list[dict]) -> None:
        """Remember a fresh read of the sheet at `url` (from a preview)."""
        with self.lock:
            self.url, self.checked = url, time.time()
            self.state = {"checked_at": dt.datetime.now().isoformat(timespec="seconds"),
                          "fingerprint": fingerprint(rows), "n_rows": len(rows), "error": None}


# ----------------------------------------------------------------------- links


def links(odor: dict) -> dict[str, Optional[str]]:
    """URLs for an odor's identifiers (None where there is nothing to link)."""
    cat = odor.get("catalog_no")
    catalog = odor.get("catalog_url")
    if not catalog and cat and SIGMA_RE.search(odor.get("supplier") or ""):
        q = urllib.parse.quote(cat)
        catalog = f"https://www.sigmaaldrich.com/US/en/search/{q}?focus=products&term={q}&type=product_number"
    cid, cas, metacyc, lotus = (odor.get(k) for k in ("cid", "cas", "metacyc", "lotus"))
    return {
        "pubchem": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else None,
        "cas": f"https://commonchemistry.cas.org/detail?cas_rn={cas}" if cas else None,
        "catalog": catalog,
        "metacyc": f"https://biocyc.org/compound?orgid=META&id={urllib.parse.quote(metacyc)}" if metacyc else None,
        "lotus": f"https://lotus.naturalproducts.net/compound/lotus_id/{lotus}" if lotus else None,
        "structure": f"/api/odors/structure/{cid}" if cid else None,
    }


# ------------------------------------------------------------------ structures

PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
INCHI_BATCH = 100  # CIDs per InChI request

# PubChem allows 5 requests a second: one request at a time, at most 4 a second
_pubchem_lock = threading.RLock()
_pubchem_last = [0.0]
_failed: dict[str, float] = {}  # url -> time a request for it failed (retried after 10 minutes)


def _pubchem_get(url: str, check: Callable[[bytes], bool]) -> Optional[bytes]:
    """The body of a PubChem request, or None if it failed (or failed in the last 10 minutes)."""
    with _pubchem_lock:
        if time.time() - _failed.get(url, 0) < 600:
            return None
        time.sleep(max(0.0, 0.25 - (time.time() - _pubchem_last[0])))
        try:
            with urllib.request.urlopen(url, timeout=15) as res:
                data = res.read()
            if not check(data):
                raise ValueError("unexpected response")
        except Exception:
            _failed[url] = time.time()
            return None
        finally:
            _pubchem_last[0] = time.time()
        return data


def structure_png(cid: int) -> Optional[Path]:
    """Path of the cached PubChem structure image for `cid`, fetching it if needed."""
    path = config.STRUCTURES_DIR / f"{cid}.png"
    if path.exists():
        return path
    with _pubchem_lock:  # so that an image several requests want is fetched once
        if path.exists():
            return path
        data = _pubchem_get(f"{PUBCHEM_REST}/{cid}/PNG?image_size=300x300", lambda d: d.startswith(b"\x89PNG"))
        if data is None:
            return None
        config.STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path


def fetch_inchis(cids: list[int]) -> dict[int, str]:
    """The InChI of each of `cids`, from PubChem (`INCHI_BATCH` CIDs per request).

    Returns:
        cid -> InChI, without the CIDs PubChem has none for (or that could not be fetched).
    """
    out = {}
    for i in range(0, len(cids), INCHI_BATCH):
        batch = ",".join(str(c) for c in cids[i:i + INCHI_BATCH])
        data = _pubchem_get(f"{PUBCHEM_REST}/{batch}/property/InChI/JSON", lambda d: d.lstrip().startswith(b"{"))
        if data is None:
            continue
        for p in json.loads(data).get("PropertyTable", {}).get("Properties", []):
            if isinstance(p.get("InChI"), str) and p["InChI"].startswith("InChI="):
                out[int(p["CID"])] = p["InChI"]
    return out


# -------------------------------------------------------- datasets <-> odors


class OdorIndex:
    """Looks up database odors by name or alias, abbreviation, or CID."""

    def __init__(self, odors: list[dict]):
        self.by_id = {o["id"]: o for o in odors}
        self.by_name: dict[str, list[int]] = {}
        self.by_label: dict[str, list[int]] = {}
        self.by_cid: dict[int, list[int]] = {}
        for o in odors:
            for n in [o["name"], *(o.get("aliases") or [])]:
                self._add(self.by_name, n.casefold(), o["id"])
            for lab in [o.get("abbrev"), *(o.get("aliases") or [])]:
                if lab:
                    self._add(self.by_label, lab, o["id"])
            if o.get("cid"):
                self._add(self.by_cid, o["cid"], o["id"])

    @staticmethod
    def _add(index: dict, key, odor_id: int) -> None:
        ids = index.setdefault(key, [])
        if odor_id not in ids:
            ids.append(odor_id)

    def match(self, label: str, yaml_odor: Optional[yamlcheck.YamlOdor] = None) -> tuple[Optional[int], str, list[int]]:
        """The odor a dataset means by `label` (an odor string's abbreviation).

        Uses the dataset's odor YAML entry for it when there is one (its name, or CID),
        else the label as an abbreviation / alias, then as a name.

        Returns:
            (odor id or None, how it was matched: 'yaml' | 'abbrev' | 'name' |
            'ambiguous' | 'none', candidate ids when ambiguous)
        """
        if yaml_odor is not None:
            ids = self.by_name.get(yaml_odor.name.casefold(), [])
            if not ids and yaml_odor.extra.get("cid"):
                try:
                    ids = self.by_cid.get(int(yaml_odor.extra["cid"]), [])
                except (TypeError, ValueError):
                    ids = []
            if len(ids) == 1:
                return ids[0], "yaml", []
        for how, ids in (("abbrev", self.by_label.get(label, [])), ("name", self.by_name.get(label.casefold(), []))):
            if len(ids) == 1:
                return ids[0], how, []
            if len(ids) > 1:
                return None, "ambiguous", ids
        folded = {i for lab, ids in self.by_label.items() if lab.casefold() == label.casefold() for i in ids}
        if len(folded) == 1:
            return folded.pop(), "abbrev", []
        return None, "none", sorted(folded)


def dataset_yaml_odors(ds: dict, datasets_dir: Path) -> list[yamlcheck.YamlOdor]:
    out = []
    for name in ds.get("odor_yaml_files") or []:
        try:
            odors, _ = yamlcheck.parse_odor_yaml((datasets_dir / str(ds["id"]) / "odor_yaml" / name).read_text(encoding="utf-8"))
        except (OSError, yamlcheck.YamlError):
            continue
        out.extend(odors)
    return out


def dataset_odors(ds: dict, trials: list[dict], index: OdorIndex, yaml_odors: list[yamlcheck.YamlOdor]) -> list[dict]:
    """One row per (panel, component odor) of a dataset, in trial order, matched to the odor table.

    Args:
        ds: `Database.get_dataset` dict.
        trials: the dataset's trials (``panel`` and ``components`` used).
        index: the odor table.
        yaml_odors: the dataset's odor YAML entries.

    Returns:
        dicts with ``panel``, ``odor`` (as in the data), ``label``, ``log10_conc``,
        ``odor_id``, ``match`` and ``candidates`` (see `OdorIndex.match`), and
        ``yaml`` (that entry's solvent / glomerulus / ... or None).
    """
    seen = set()
    out = []
    for t in trials:
        for comp in t["components"]:
            if comp == SOLVENT_STR or (t["panel"], comp) in seen:
                continue
            seen.add((t["panel"], comp))
            try:
                o = parse_odor(comp, strict=False)
                label, conc = o.name, o.log10_conc
            except OdorFormatError:
                label, conc = comp, None
            yo = next((y for y in yaml_odors if label in y.labels() and yamlcheck.conc_equal(conc, y.log10_conc)), None)
            odor_id, how, candidates = index.match(label, yo)
            out.append({
                "panel": t["panel"], "odor": comp, "label": label, "log10_conc": conc,
                "odor_id": odor_id, "match": how, "candidates": candidates,
                "yaml": ({"name": yo.name, **{k: v for k, v in yo.extra.items() if isinstance(v, (str, int, float))}}
                         if yo else None),
            })
    return out


def usage(db, datasets_dir: Path, odors: list[dict]) -> dict[int, set[int]]:
    """Datasets each odor (id) is used by, as `dataset_odors` matches them."""
    index = OdorIndex(odors)
    used: dict[int, set[int]] = {}
    for d in db.list_datasets():
        ds = db.get_dataset(d["id"])
        for row in dataset_odors(ds, db.get_trials(d["id"]), index, dataset_yaml_odors(ds, datasets_dir)):
            if row["odor_id"] is not None:
                used.setdefault(row["odor_id"], set()).add(d["id"])
    return used


def table_rows(rows: list[dict], index: OdorIndex) -> list[dict]:
    """`dataset_odors` rows flattened for a CSV: data odor, metadata, URLs."""
    out = []
    for r in rows:
        o = index.by_id.get(r["odor_id"]) or {}
        y = r["yaml"] or {}
        lk = links(o) if o else {}
        out.append({
            "panel": r["panel"], "odor": r["odor"], "abbrev": r["label"],
            "log10_conc": format_conc(r["log10_conc"]) if r["log10_conc"] is not None else None,
            "name": o.get("name") or y.get("name"), "solvent": y.get("solvent") or o.get("solvent"),
            "glomerulus": y.get("glomerulus"), "cid": o.get("cid"), "inchi": o.get("inchi"), "cas": o.get("cas"),
            "supplier": o.get("supplier"),
            "catalog_no": o.get("catalog_no"), "metacyc": o.get("metacyc"), "lotus": o.get("lotus"),
            "inventory": y.get("inventory") or o.get("inventory"),
            "pubchem_url": lk.get("pubchem"), "cas_url": lk.get("cas"), "catalog_url": lk.get("catalog"),
            "metacyc_url": lk.get("metacyc"), "lotus_url": lk.get("lotus"),
            "matched_by": r["match"] if r["odor_id"] is not None else None,
        })
    return out
