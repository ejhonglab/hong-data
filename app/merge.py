"""Merged downloads: every dataset of a coding unit, concatenated, groups of them optionally
scaled to match each other first.

Datasets are grouped by any of experimenter, recording system and power at the fly (with
power: measured, and every two in a group with their ranges, power ± its tolerance or
`DEFAULT_POWER_PM`, at most `power_tolerance` apart; datasets without it are left as they
are). Each group gets one scale factor: the one that best matches
(least squares, through zero) its mean responses to the odors it shares with the target,
a group or the average of the groups, to the target's. Responses are compared per odor,
and per ROI when every dataset's ROIs are glomeruli (so that the same glomeruli are
compared), else averaged over ROIs. An odor of a dataset stands for a target odor when
their components have the same names, at the same concentrations if the dataset has
those, else at its nearest ones, less than `tolerance` log10 units away.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .db import Database
from .odors import OdorFormatError, parse_odor

MODES = ("none", "groups")
GROUP_BY = ("uploader", "system", "power")
AVERAGE = "average"
DEFAULT_TOLERANCE = 1.0
DEFAULT_POWER_PM = 0.5  # mW: the ± of a power measured without one


class MergeError(ValueError):
    pass


@dataclass
class Group:
    key: str  # also its label
    ids: list[int]
    scale: Optional[float] = None
    n_points: int = 0
    n_stimuli: int = 0
    r: Optional[float] = None
    note: Optional[str] = None
    values: dict = field(default_factory=dict, repr=False)  # (odor key, roi) -> mean response


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").split())


def coding_units(db: Database) -> list[dict]:
    """Each coding unit (spelled as first uploaded, matched in any case) and its datasets' ids."""
    out: dict[str, dict] = {}
    for d in sorted(db.list_datasets(), key=lambda d: d["id"]):
        u = _norm(d["coding_unit"])
        out.setdefault(u.casefold(), {"coding_unit": u, "ids": []})["ids"].append(d["id"])
    return sorted(out.values(), key=lambda u: (-len(u["ids"]), u["coding_unit"].casefold()))


def datasets_of(db: Database, coding_unit: str) -> list[dict]:
    want = _norm(coding_unit).casefold()
    ids = [d["id"] for d in db.list_datasets() if _norm(d["coding_unit"]).casefold() == want]
    return [db.get_dataset(i) for i in sorted(ids)]


# ----------------------------------------------------------------------- groups


def _power(d: dict) -> tuple[Optional[float], float]:
    """A dataset's power (mW) and its ± (`DEFAULT_POWER_PM` if none was entered)."""
    md = d["metadata"]
    pm = md.get("power_tolerance_mw")
    return md.get("power_mw"), DEFAULT_POWER_PM if pm is None else pm


def _power_close(a: dict, b: dict, power_tolerance: float) -> bool:
    """Whether two datasets' power ranges (power ± its tolerance) overlap, or are at most `power_tolerance` apart."""
    (pa, ta), (pb, tb) = _power(a), _power(b)
    return abs(pa - pb) <= ta + tb + power_tolerance + 1e-9


def make_groups(datasets: list[dict], by: set[str], power_tolerance: float = 0.0) -> tuple[list[Group], list[dict]]:
    """Groups of `datasets` (by any of `GROUP_BY`; with power, every two in a group close
    enough, see `_power_close`), and the datasets left out of them (``id``, ``reason``)."""
    buckets: dict[tuple, list[dict]] = {}
    labels: dict[tuple, list[str]] = {}
    unscaled = []
    for d in datasets:
        md = d["metadata"]
        if "power" in by and md.get("power_mw") is None:
            unscaled.append({"id": d["id"], "reason": "no power measured: left as it is"})
            continue
        parts = []
        if "uploader" in by:
            parts.append(_norm(md.get("uploader_name")))
        if "system" in by:
            parts.append(_norm(md.get("recording_system")) or "recording system not given")
        key = tuple(p.casefold() for p in parts)
        buckets.setdefault(key, []).append(d)
        labels.setdefault(key, parts)
    groups = []
    for key, ds in buckets.items():
        clusters: list[list[dict]] = []
        if "power" in by:
            # in order of power, each joins the first group it is close enough to all of
            for d in sorted(ds, key=lambda d: (_power(d)[0], d["id"])):
                for cl in clusters:
                    if all(_power_close(d, e, power_tolerance) for e in cl):
                        cl.append(d)
                        break
                else:
                    clusters.append([d])
        else:
            clusters = [ds]
        for cl in clusters:
            parts = list(labels[key])
            if "power" in by:
                lo, hi = min(_power(d)[0] for d in cl), max(_power(d)[0] for d in cl)
                parts.append(f"{lo:g} mW" if lo == hi else f"{lo:g}–{hi:g} mW")
            groups.append(Group(" · ".join(parts) or "all datasets", sorted(d["id"] for d in cl)))
    seen: dict[str, int] = {}
    for g in groups:  # keys must tell groups apart
        seen[g.key] = seen.get(g.key, 0) + 1
        if seen[g.key] > 1:
            g.key += f" ({seen[g.key]})"
    return groups, unscaled


# -------------------------------------------------------------------- responses


def odor_key(components: list[str]) -> Optional[tuple]:
    """(name, log10 concentration) of each component of a trial that is not solvent, by
    name; None for none (or for odor strings that cannot be read)."""
    odors = []
    for c in components:
        try:
            o = parse_odor(c, strict=False)
        except OdorFormatError:
            return None
        if not o.is_solvent:
            odors.append((o.name, o.log10_conc))
    return tuple(sorted(odors)) or None


def dataset_means(matrix: dict, by_roi: bool) -> dict[tuple, dict[Optional[str], float]]:
    """A dataset's mean response to each odor (`odor_key`), over its repeats: per ROI name
    (over flies) with `by_roi`, else over all ROIs (under the ROI None)."""
    values, units = matrix["values"], matrix["units"]
    cols: dict[tuple, list[int]] = {}
    for j, t in enumerate(matrix["trials"]):
        k = odor_key(t["components"])
        if k is not None:
            cols.setdefault(k, []).append(j)
    rois = [u["roi"] for u in units]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # means of nothing but missing values
        for k, js in cols.items():
            per_unit = np.nanmean(values[:, js], axis=1)
            if by_roi:
                means = {}
                for roi in dict.fromkeys(rois):
                    v = np.nanmean(per_unit[[i for i, r in enumerate(rois) if r == roi]])
                    if np.isfinite(v):
                        means[roi] = float(v)
            else:
                v = np.nanmean(per_unit)
                means = {None: float(v)} if np.isfinite(v) else {}
            if means:
                out[k] = means
    return out


def match(available, target: tuple, tolerance: float) -> Optional[tuple]:
    """The odor (`odor_key`) of `available` that stands for `target`: the same, else the one
    with the same components at the nearest concentrations, all less than `tolerance` away."""
    if target in available:
        return target
    names = [n for n, _ in target]
    best, best_d = None, math.inf
    for k in available:
        if [n for n, _ in k] != names:
            continue
        d = max(abs(c - tc) for (_, c), (_, tc) in zip(k, target))
        if d < tolerance - 1e-9 and (d < best_d or (d == best_d and k > best)):
            best, best_d = k, d
    return best


def _group_values(keys, means: dict[int, dict], ids: list[int], tolerance: float) -> dict[tuple, float]:
    """A group's mean response to each of `keys`, from what its datasets have that stands
    for it (`match`), per ROI: (key, roi) -> value."""
    found: dict[tuple, list[float]] = {}
    for i in ids:
        for k in keys:
            m = match(means[i], k, tolerance)
            if m is not None:
                for roi, v in means[i][m].items():
                    found.setdefault((k, roi), []).append(v)
    return {kr: float(np.mean(vs)) for kr, vs in found.items()}


def _fit(group: Group, target: dict[tuple, float]) -> None:
    pairs = [(target[kr], v) for kr, v in group.values.items() if kr in target]
    t = np.array([p[0] for p in pairs], dtype=float)
    g = np.array([p[1] for p in pairs], dtype=float)
    group.n_points = len(pairs)
    group.n_stimuli = len({kr[0] for kr in group.values if kr in target})
    if not len(pairs) or not np.any(g):
        group.note = "no odors in common with the target: left as it is"
        return
    group.scale = float(t @ g / (g @ g))
    if len(pairs) >= 3 and np.std(t) > 0 and np.std(g) > 0:
        group.r = float(np.corrcoef(t, g)[0, 1])


def plan(db: Database, coding_unit: str, mode: str = "none", by: Optional[list[str]] = None, target: str = AVERAGE,
         tolerance: float = DEFAULT_TOLERANCE, power_tolerance: float = 0.0) -> dict:
    """How the datasets of `coding_unit` are scaled in a merged download.

    Args:
        mode: 'none' (each at its own scale) or 'groups' (groups scaled to `target`).
        by: what groups are made by (`GROUP_BY`).
        target: the key of the group scaled to, or `AVERAGE` (the average of the groups);
            the average if there is no such group.
        tolerance: how far apart (log10) concentrations of an odor must be less than to stand for each other.
        power_tolerance: how far apart (mW) the power ranges (power ± its tolerance) of two
            datasets may be for them to be in a group, when grouped by power.

    Returns:
        dict with ``datasets`` (each one's details), ``groups`` (``key``, ``ids``, ``scale``,
        ``n_points`` and ``n_stimuli`` compared, Pearson's ``r`` over them, ``note``),
        ``unscaled`` (datasets left out of the groups), ``target`` (the key of the group
        scaled to, or `AVERAGE`), ``by_roi`` (whether responses were compared per ROI) and
        ``scales`` (dataset id -> its factor).

    Raises:
        MergeError: for unknown options, or a coding unit with no datasets.
    """
    by = list(dict.fromkeys(by or []))
    if mode not in MODES or any(b not in GROUP_BY for b in by):
        raise MergeError(f"mode must be one of {', '.join(MODES)}, and groups made by any of {', '.join(GROUP_BY)}")
    if not 0 <= tolerance <= 10:
        raise MergeError("the tolerance must be between 0 and 10 (log10 units)")
    if not 0 <= power_tolerance <= 1000:
        raise MergeError("the power tolerance must be between 0 and 1000 (mW)")
    datasets = datasets_of(db, coding_unit)
    if not datasets:
        raise MergeError(f"no datasets of coding unit {coding_unit!r}")
    out = {
        "coding_unit": _norm(datasets[0]["coding_unit"]), "mode": mode, "by": by, "tolerance": tolerance,
        "power_tolerance": power_tolerance,
        "datasets": [{
            "id": d["id"], "title": d["title"], "uploader": d["metadata"].get("uploader_name"),
            "system": d["metadata"].get("recording_system"), "power_mw": d["metadata"].get("power_mw"),
            "power_tolerance_mw": d["metadata"].get("power_tolerance_mw"),
        } for d in datasets],
        "groups": [], "unscaled": [], "target": None, "by_roi": False, "scales": {d["id"]: 1.0 for d in datasets},
    }
    if mode == "none":
        return out
    groups, unscaled = make_groups(datasets, set(by), power_tolerance)
    by_roi = all(d["roi_kind"] == "glomeruli" for d in datasets)
    in_groups = [i for g in groups for i in g.ids]
    means = {i: dataset_means(db.get_matrix_arrays(i), by_roi) for i in in_groups}
    keys_of = lambda ids: list(dict.fromkeys(k for i in ids for k in means[i]))  # noqa: E731
    target_group = next((g for g in groups if g.key == target), None)
    if target_group is not None:
        keys = keys_of(target_group.ids)
        for g in groups:
            g.values = _group_values(keys, means, g.ids, tolerance)
        target_values = target_group.values
    else:
        keys = keys_of(in_groups)
        for g in groups:
            g.values = _group_values(keys, means, g.ids, tolerance)
        # the average over the groups of each response more than one of them has
        pooled: dict[tuple, list[float]] = {}
        for g in groups:
            for kr, v in g.values.items():
                pooled.setdefault(kr, []).append(v)
        target_values = {kr: float(np.mean(vs)) for kr, vs in pooled.items() if len(vs) > 1}
    for g in groups:
        if g is target_group:
            g.scale, g.note = 1.0, "the target"
            g.n_points, g.n_stimuli = len(target_values), len({kr[0] for kr in target_values})
        elif len(groups) == 1:
            g.scale, g.note = 1.0, "the only group: nothing to scale it to"
        else:
            _fit(g, target_values)
        for i in g.ids:
            out["scales"][i] = g.scale if g.scale is not None else 1.0
    out.update(
        groups=[{"key": g.key, "ids": g.ids, "scale": g.scale, "n_points": g.n_points, "n_stimuli": g.n_stimuli,
                 "r": g.r, "note": g.note} for g in groups],
        unscaled=unscaled, target=target_group.key if target_group else AVERAGE, by_roi=by_roi)
    return out


def scaling_frame(p: dict) -> pd.DataFrame:
    """scaling.csv of a merged download: each dataset's group and scale factor."""
    group_of = {i: g for g in p["groups"] for i in g["ids"]}
    reason = {u["id"]: u["reason"] for u in p["unscaled"]}
    rows = []
    for d in p["datasets"]:
        g = group_of.get(d["id"])
        rows.append({
            "dataset_id": d["id"], "title": d["title"], "group": g["key"] if g else None,
            "scale": p["scales"][d["id"]], "scaled_to": p["target"] if g else None,
            "n_points": g["n_points"] if g else None, "n_stimuli": g["n_stimuli"] if g else None,
            "r": g["r"] if g else None, "note": g["note"] if g else reason.get(d["id"]),
        })
    return pd.DataFrame(rows)


def readme_lines(p: dict) -> list[str]:
    """The section of a merged download's README on how it was scaled."""
    lines = ["", "## Scaling", ""]
    if p["mode"] == "none":
        return lines + ["Every dataset is at its own scale (not scaled)."]
    names = {"uploader": "experimenter", "system": "recording system", "power": "power at the fly"}
    by = ", ".join(names[b] for b in p["by"]) or "nothing (one group)"
    target = "the average of the groups" if p["target"] == AVERAGE else f"the group {p['target']}"
    power = [] if "power" not in p["by"] else [
        f"By power: datasets with power measured, their ranges (power ± its tolerance, ± {DEFAULT_POWER_PM:g} mW where",
        f"none was given) at most {p['power_tolerance']:g} mW apart for every two in a group.",
    ]
    return lines + power + [
        f"Datasets were grouped by {by}, and each group scaled to {target} (`scaling.csv`: each dataset's group",
        "and factor). A group's factor best matches (least squares, through zero) its mean responses to the odors",
        f"it shares with the target to the target's, {'per ROI (glomerulus)' if p['by_roi'] else 'averaged over ROIs'}.",
        f"Odors stand for each other at concentrations less than {p['tolerance']:g} log10 units apart (a dataset's exact",
        "concentration is used when it has it). In the matrices, odors keep their own concentrations.",
    ]
