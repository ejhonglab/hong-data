"""Permalinks: names of datasets that stay theirs whatever their numbers (``/d/<name>``).

A permalink is lowercase letters, digits and '-' (`slug_problem`), by default named after its
dataset as a download of its matrix is (`default_slug`: `export.dataset_stem`). It keeps what
its dataset is (`dataset_keys`): its identity (`IDENTITY`: uploader, indicator, driver,
coding unit, manipulations, response calculation and the panels that name it), its flies
(date, fly_num) and its trials (panel, odor, repeat). Its dataset deleted, it is orphaned, keeping them, until an
upload like it takes it over (`compare`) or it is deleted in the Admin tab; while it points
to a dataset, an upload of the same identity is refused (one dataset is kept of each).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .db import rename_panel
from .export import dataset_stem, main_panels
from .metadata import is_kc, manipulations_text, same_manipulations, same_response_calc

SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
MAX_LEN = 80
# what makes a dataset the one it is, but for its flies and trials (and details not required);
# keys kept before the response calculation was one have none, as a dataset without one
IDENTITY = ("uploader_name", "indicator", "driver", "coding_unit", "manipulations", "response_calc", "panels")
LABELS = {"uploader_name": "uploader", "indicator": "indicator", "driver": "driver", "coding_unit": "coding unit",
          "manipulations": "manipulations", "response_calc": "response calculation", "panels": "panels"}
EXAMPLES = 4


def slugify(text: str) -> str:
    """Text as a permalink would have it: ASCII, lowercase, each run of anything but letters
    and digits one '-', at most `MAX_LEN` long (cut at a '-')."""
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > MAX_LEN:
        s = s[:MAX_LEN + 1].rsplit("-", 1)[0] if "-" in s[:MAX_LEN + 1] else s[:MAX_LEN]
    return s.strip("-")


def slug_problem(slug: str) -> Optional[str]:
    """Why a permalink's name is refused, if it is."""
    if not slug:
        return "give it a name"
    if len(slug) > MAX_LEN:
        return f"at most {MAX_LEN} characters"
    if not SLUG.fullmatch(slug):
        return "only lowercase letters, digits and '-' (not at either end, nor two together)"
    if not re.search(r"[a-z]", slug):
        return "it needs a letter (a number alone is a dataset's number)"
    return None


def default_slug(d: dict, trials: list[dict]) -> str:
    """The name a dataset's permalink has by default (`d` as `Database.get_dataset` gives it,
    `trials` as `get_matrix_arrays`): its matrix's download's (`export.dataset_stem`), so
    ``<panels>-<uploader>-<coding unit>[-<manipulations>][-<response calculation>]-<date of the last fly>``."""
    s = slugify(dataset_stem(d, trials))
    return s if slug_problem(s) is None else f"dataset-{d['id']}"


def unique(slug: str, taken: set[str]) -> str:
    """`slug`, or with -2, -3... after it, the first not `taken`."""
    out, n = slug, 2
    while out in taken:
        tail = f"-{n}"
        out, n = slug[:MAX_LEN - len(tail)].rstrip("-") + tail, n + 1
    return out


def _sort_key(x) -> tuple:
    return tuple("" if v is None else str(v) for v in x)


def _keys(identity: dict, units: list[dict], trials: list[dict], renames: dict[str, str]) -> dict:
    flies = {(u["date"], u["fly_num"]) for u in units if u["date"] is not None}
    keys = {(rename_panel(t["panel"], renames), t["odor"], t["repeat"]) for t in trials}
    return {**identity, "flies": [list(f) for f in sorted(flies, key=_sort_key)],
            "trials": [list(t) for t in sorted(keys, key=_sort_key)], "n_rois": len(units),
            "first_fly_date": min((f[0] for f in flies), default=None), "last_fly_date": max((f[0] for f in flies), default=None)}


def dataset_keys(d: dict, m: dict) -> dict:
    """What a permalink keeps of its dataset (`d` as `Database.get_dataset` gives it, `m` as
    `get_matrix_arrays`): its number, title and upload, its identity (`IDENTITY`), its flies
    [date, fly_num] and trials [panel, odor, repeat], each once, and its number of ROIs."""
    panels = main_panels(list(dict.fromkeys(t["panel"] for t in m["trials"] if t["panel"])) or d["metadata"].get("panel_names") or [])
    identity = {k: d[k] for k in IDENTITY if k != "panels"}
    return {"dataset_id": d["id"], "title": d["title"], "uploaded_at": d["created_at"],
            **_keys({**identity, "panels": panels}, m["units"], m["trials"], {})}


def upload_keys(units: list[dict], trials: list[dict], metadata: dict, renames: dict[str, str]) -> dict:
    """`dataset_keys` of an upload (its data's units and trials, as `ingest.ParsedData` has
    them, and its metadata as entered: `metadata.Metadata`, dumped), its panels as shown (`renames`)."""
    in_data = list(dict.fromkeys(rename_panel(t["panel"], renames) for t in trials if t["panel"]))
    panels = main_panels(in_data or [rename_panel(p, renames) for p in metadata.get("panel_names") or []])
    identity = {k: metadata.get(k) for k in IDENTITY if k != "panels"}
    return _keys({**identity, "manipulations": identity["manipulations"] or [], "panels": panels}, units, trials, renames)


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").split()).casefold()


def _same(key: str, a, b) -> bool:
    if key == "manipulations":
        return same_manipulations(a, b)
    if key == "response_calc":
        return same_response_calc(a, b)
    if key == "panels":
        return {_norm(p) for p in a or []} == {_norm(p) for p in b or []}
    if key == "coding_unit" and is_kc(a) and is_kc(b):
        return True
    return _norm(a) == _norm(b)


def compare(old: dict, new: dict) -> dict:
    """How an upload's keys (`new`) are like a permalink's (`old`, as `dataset_keys`).

    Returns:
        ``differs``: the fields of `IDENTITY` that differ; ``flies`` and ``trials``: those
        only in old (``removed``) and only in new (``added``); ``same``: the same identity
        (whatever their flies and trials); ``exact``: all the same; ``related``: the same
        identity but maybe for its panels, and some panels or flies in common (an upload of
        the dataset an orphaned permalink was of, changed); ``shared``: how many flies and
        trials they have in common.
    """
    differs = [k for k in IDENTITY if not _same(k, old.get(k), new.get(k))]
    fo, fn = {tuple(f) for f in old.get("flies") or []}, {tuple(f) for f in new.get("flies") or []}
    to, tn = {tuple(t) for t in old.get("trials") or []}, {tuple(t) for t in new.get("trials") or []}
    panels = {_norm(p) for p in old.get("panels") or []} & {_norm(p) for p in new.get("panels") or []}
    return {"differs": differs,
            "flies": {"removed": sorted(fo - fn, key=_sort_key), "added": sorted(fn - fo, key=_sort_key)},
            "trials": {"removed": sorted(to - tn, key=_sort_key), "added": sorted(tn - to, key=_sort_key)},
            "same": not differs, "exact": not differs and fo == fn and to == tn,
            "related": set(differs) <= {"panels"} and bool(panels or fo & fn), "shared": len(fo & fn) + len(to & tn)}


def _value(key: str, v) -> str:
    if key == "manipulations":
        return manipulations_text(v) or "none"
    if key == "panels":
        return ", ".join(v) if v else "none"
    return str(v) if v else "none"


def _some(items: list[str]) -> str:
    return ", ".join(items[:EXAMPLES]) + (", …" if len(items) > EXAMPLES else "")


def _fly(f) -> str:
    return f"{f[0]} fly {f[1]}" if f[1] is not None else str(f[0])


def differences(c: dict, old: dict, new: dict) -> list[str]:
    """What differs between an upload (`new`) and a permalink's keys (`old`), `c` as
    `compare` gave it: each field of its identity, then its flies, odors and trials, as
    sentences (of the upload: 'here')."""
    out = [f"{LABELS[k]}: {_value(k, new.get(k))} here, {_value(k, old.get(k))} before" for k in c["differs"]]
    added, removed = c["flies"]["added"], c["flies"]["removed"]
    if added or removed:
        out.append("flies: " + "; ".join(x for x in [
            f"{len(added)} not there before ({_some([_fly(f) for f in added])})" if added else "",
            f"{len(removed)} of those before not here ({_some([_fly(f) for f in removed])})" if removed else ""] if x))
    odors = lambda ts: {(t[0], t[1]) for t in ts}
    od_old, od_new = odors(old.get("trials") or []), odors(new.get("trials") or [])
    od = lambda o: f"{o[1]} ({o[0]})" if o[0] else str(o[1])
    more, fewer = sorted(od_new - od_old, key=_sort_key), sorted(od_old - od_new, key=_sort_key)
    if more or fewer:
        out.append("odors: " + "; ".join(x for x in [
            f"{len(more)} not there before ({_some([od(o) for o in more])})" if more else "",
            f"{len(fewer)} of those before not here ({_some([od(o) for o in fewer])})" if fewer else ""] if x))
    kept = od_old & od_new
    t_added = [t for t in c["trials"]["added"] if (t[0], t[1]) in kept]
    t_removed = [t for t in c["trials"]["removed"] if (t[0], t[1]) in kept]
    if t_added or t_removed:
        out.append(f"trials of the odors in both: {len(t_added)} not there before, {len(t_removed)} of those before not here")
    return out


def summary(k: dict) -> str:
    """What a permalink points (or pointed) to, in a sentence (`k` as `dataset_keys`)."""
    flies = k.get("flies") or []
    dates = k.get("first_fly_date") if k.get("first_fly_date") == k.get("last_fly_date") else \
        f"{k.get('first_fly_date')} to {k.get('last_fly_date')}"
    odors = {(t[0], t[1]) for t in k.get("trials") or []}
    return (f"#{k.get('dataset_id')}, {k.get('title')}: uploaded by {k.get('uploader_name')}, "
            + (f"{len(flies)} {'fly' if len(flies) == 1 else 'flies'} ({dates}), " if flies else "")
            + f"{k.get('n_rois', 0):,} ROIs, {len(odors)} odors, {len(k.get('trials') or [])} trials")
