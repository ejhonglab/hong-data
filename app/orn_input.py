"""ORN data made the input of a model run (model_runs.py): datasets of glomeruli, as each
glomerulus's mean response to each odor, scaled from their units (dF/F) into changes in
spike rate, which the models take.

The factor is the one that best matches (least squares, through zero, as the Merge tab's)
the responses to what Hallem & Carlson (2006) measured: the built-in reference dataset of
all they tabulate (`HALLEM_FILE`, with their lower concentrations and fruits), per odor and
glomerulus. An odor stands for theirs when it is the same odor of the odor table (else of
the same name) and their dilution is `HALLEM_SHIFT` above its concentration (their 10^-2
our 10^-3, as pyolfsysm takes them), else the nearest less than `tolerance` from that.
With too few in common (`MIN_ODORS`, `MIN_POINTS`), the datasets are first scaled to the
combination of all datasets of their coding unit, made as the Merge tab merges them (each
of the other datasets a group, each scaled to the average of the responses more than one
group has), by the responses they share with the others, and that combination to Hallem's.

The factor is from the responses to every panel's odors; the model's odors are those of the
panels chosen (by default, all but panels of diagnostic odors), as written. Only odors of one
component are taken, and only glomeruli measured for every one of them: a glomerulus left
out is one with no input, for every odor alike.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from . import config, glomeruli as glom, merge
from .odormeta import OdorIndex
from .odors import OdorFormatError, format_conc, parse_odor

HALLEM_FILE = "hallem_carlson_2006.csv"  # (reference_data/, made by its make_hallem.py)
HALLEM_SHIFT = 1.0  # their log10 dilution that stands for a log10 concentration c of ours: c + this
MIN_ODORS, MIN_POINTS = 5, 50  # odors, and odor-glomerulus pairs, in common with Hallem's for a factor from them alone
MIN_GLOMERULI = 5  # glomeruli measured for every odor, at least, for a model's input
HALLEM, COMBINATION = "hallem", "combination"


class InputError(ValueError):
    """Why datasets cannot be made a model's input, for people."""


def _ident(name: str, index: OdorIndex) -> tuple:
    """What an odor is, by its name as written: the odor of the odor table, else its name."""
    odor_id = index.match(name)[0]
    return ("odor", odor_id) if odor_id is not None else ("name", name.casefold())


def hallem(index: OdorIndex) -> dict[tuple, dict[str, float]]:
    """Hallem & Carlson's responses: (what the odor is, log10 dilution) -> glomerulus -> Δ spikes/s,
    with ``label`` under the key ``None`` of each (the odor as their dataset writes it)."""
    df = pd.read_csv(config.REFERENCE_DIR / HALLEM_FILE, index_col=0)
    out = {}
    for col in df.columns:
        o = parse_odor(str(col), strict=False)
        out[(_ident(o.name, index), float(o.log10_conc))] = {None: str(col), **{str(g): float(v) for g, v in df[col].items() if np.isfinite(v)}}
    return out


def _is_diagnostic(panel: Optional[str]) -> bool:
    return "diagnostic" in (panel or "").lower()


def panels(db, ids: list[int]) -> list[dict]:
    """The panels of the datasets `ids`, in the order first presented: each ``name``, how many
    stimuli of one odor (``n_odors``, each once) it has in them, and whether it is part of a
    model's input by default (``default``: not one of diagnostic odors, unless they all are)."""
    found: dict[str, set] = {}
    for i in ids:
        m = db.get_matrix_arrays(i)
        for t in (m or {}).get("trials", []):
            key = merge.odor_key(t["components"])
            if key is not None and len(key) == 1:
                found.setdefault(t["panel"] or "", set()).add(key)
    only_diagnostic = all(_is_diagnostic(p) for p in found)
    return [{"name": p, "n_odors": len(keys), "default": only_diagnostic or not _is_diagnostic(p)} for p, keys in found.items()]


def means(db, ids: list[int], index: OdorIndex, of_panels: Optional[list[str]] = None) -> dict:
    """The mean response of each glomerulus to each odor of one component, over the datasets
    `ids` (each its flies' mean, `merge.dataset_means`; ROIs that are not glomeruli, or marked
    not to be used, left out), of the panels `of_panels` alone, if given.

    Returns:
        ``values`` ((what the odor is, log10 concentration) -> glomerulus -> mean), ``labels``
        (that key -> the odor as first written) and ``panels`` (-> the first panel it is of),
        in the order first presented, and ``mixtures`` (the stimuli of several components,
        left out).
    """
    found: dict[tuple, dict[str, list[float]]] = {}
    labels: dict[tuple, str] = {}
    panel_of: dict[tuple, str] = {}
    mixtures: list[str] = []
    for i in ids:
        m = db.get_matrix_arrays(i)
        if m is None:
            raise InputError(f"dataset #{i} has no responses")
        rows = [k for k, u in enumerate(m["units"]) if glom.is_glomerulus(u["roi"])]
        cols = [j for j, t in enumerate(m["trials"]) if of_panels is None or (t["panel"] or "") in of_panels]
        trials = [m["trials"][j] for j in cols]
        m = {**m, "units": [m["units"][k] for k in rows], "trials": trials, "values": m["values"][np.ix_(rows, cols)]}
        first_panel = {}
        for t in trials:
            first_panel.setdefault(merge.odor_key(t["components"]), t["panel"] or "")
        for key, by_roi in merge.dataset_means(m, by_roi=True).items():
            text = " + ".join(f"{n} @ {format_conc(c)}" for n, c in key)
            if len(key) != 1:
                if text not in mixtures:
                    mixtures.append(text)
                continue
            (name, conc), = key
            k = (_ident(name, index), float(conc))
            labels.setdefault(k, text)
            panel_of.setdefault(k, first_panel.get(key, ""))
            for roi, v in by_roi.items():
                found.setdefault(k, {}).setdefault(roi, []).append(v)
    return {"values": {k: {g: float(np.mean(v)) for g, v in by.items()} for k, by in found.items()}, "labels": labels,
            "panels": panel_of, "mixtures": mixtures}


def _nearest(available, what: tuple, conc: float, tolerance: float) -> Optional[tuple]:
    """The key of `available` (what, concentration) that stands for `what` at `conc`: that,
    else of the same odor at the nearest concentration less than `tolerance` away."""
    if (what, conc) in available:
        return (what, conc)
    near = [(abs(c - conc), -c, (w, c)) for w, c in available if w == what and abs(c - conc) < tolerance - 1e-9]
    return min(near)[2] if near else None


def _fit(pairs: list[tuple[float, float]]) -> tuple[Optional[float], Optional[float]]:
    """(the factor on the second of each pair that best matches the first, least squares
    through zero; Pearson's r of them): None with nothing to fit, or too few for an r."""
    t, v = np.array([p[0] for p in pairs], dtype=float), np.array([p[1] for p in pairs], dtype=float)
    if not len(pairs) or not np.any(v):
        return None, None
    r = float(np.corrcoef(t, v)[0, 1]) if len(pairs) >= 3 and np.std(t) > 0 and np.std(v) > 0 else None
    return float(t @ v / (v @ v)), r


def to_hallem(values: dict, hal: dict, labels: dict, tolerance: float) -> dict:
    """Responses (`means`' values) against Hallem's of the same odors and glomeruli.

    Returns:
        ``factor`` and ``r`` (`_fit`'s), ``n_odors``, ``n_points``, ``enough`` (odors and
        points for the factor to be used) and ``points`` (each ``odor``, ``hallem_odor``,
        ``glomerulus``, ``response`` and ``hallem``).
    """
    points = []
    for (what, conc), by in values.items():
        k = _nearest(hal, what, conc + HALLEM_SHIFT, tolerance)
        if k is None:
            continue
        for g, v in by.items():
            if g in hal[k]:
                points.append({"odor": labels.get((what, conc), ""), "hallem_odor": hal[k][None], "glomerulus": g,
                               "response": v, "hallem": hal[k][g]})
    factor, r = _fit([(p["hallem"], p["response"]) for p in points])
    n_odors = len({p["odor"] for p in points})
    return {"factor": factor, "r": r, "n_odors": n_odors, "n_points": len(points),
            "enough": factor is not None and factor > 0 and n_odors >= MIN_ODORS and len(points) >= MIN_POINTS, "points": points}


def _combination(db, ids: list[int], mine: dict, index: OdorIndex, tolerance: float) -> dict:
    """The datasets `ids` (their `means`, `mine`) against the combination of all datasets of
    their coding unit, made as the Merge tab merges them: each of the others a group, and
    `ids` one, each scaled by the factor that best matches its responses to the average of
    the responses more than one group has (a group sharing none left as it is), then every
    response averaged over the groups that have it.

    Returns:
        ``datasets`` (the others', ``id`` and ``title``), ``values`` and ``labels`` (the
        combination's, as `means`'), and the factor on `mine` (``factor``; None if it
        shares nothing with the others), with ``r``, ``n_points`` and ``n_stimuli``.
    """
    first = db.get_dataset(ids[0])
    same = {i for s in db.calculations() if set(s) & set(ids) for i in s}  # (their recordings calculated otherwise: not others)
    others, _, _ = merge.one_calculation(db, [d for d in merge.datasets_of(db, first["coding_unit"])
                                              if d["id"] not in same and d["id"] not in ids and d["roi_kind"] == "glomeruli"])
    groups = [mine] + [means(db, [d["id"]], index) for d in others]
    keys = list(dict.fromkeys(k for g in groups for k in g["values"]))
    per_group = []
    for g in groups:
        at = {}
        for what, conc in keys:
            k = _nearest(g["values"], what, conc, tolerance)
            if k is not None:
                at[(what, conc)] = g["values"][k]
        per_group.append(at)

    def pooled(scales: list[float]) -> dict[tuple, dict[str, list[float]]]:
        out: dict[tuple, dict[str, list[float]]] = {}
        for at, f in zip(per_group, scales):
            for k, by in at.items():
                for roi, v in by.items():
                    out.setdefault(k, {}).setdefault(roi, []).append(v * f)
        return out

    target = {k: {roi: float(np.mean(vs)) for roi, vs in by.items() if len(vs) > 1} for k, by in pooled([1.0] * len(groups)).items()}
    fits = []
    for at in per_group:
        pairs = [(target[k][roi], v) for k, by in at.items() for roi, v in by.items() if roi in target.get(k, {})]
        fits.append((*_fit(pairs), pairs))
    scales = [f if f is not None and f > 0 else 1.0 for f, _, _ in fits]
    combined = {k: {roi: float(np.mean(vs)) for roi, vs in by.items()} for k, by in pooled(scales).items()}
    labels = {}
    for g in groups:
        for k, text in g["labels"].items():
            labels.setdefault(k, text)
    factor, r, pairs = fits[0]
    return {"datasets": [{"id": d["id"], "title": d["title"]} for d in others], "values": combined, "labels": labels,
            "factor": factor, "r": r, "n_points": len(pairs),
            "n_stimuli": len({k for k, by in per_group[0].items() if set(by) & set(target.get(k, {}))})}


def scale(db, ids: list[int], index: OdorIndex, of_panels: Optional[list[str]] = None,
          tolerance: float = merge.DEFAULT_TOLERANCE) -> dict:
    """The datasets `ids` as a model's input: their glomeruli's mean responses to the odors of
    the panels `of_panels` (all of them, if None), and the factor that makes them changes in
    spike rate, from their responses to every panel's odors (see the module's docstring).

    Returns:
        ``method`` (`HALLEM` | `COMBINATION`), ``factor`` (on the responses), ``hallem``
        (`to_hallem`'s of the datasets themselves), ``combination`` (None, or with
        ``datasets``, its own factor to the combination ``factor`` and ``r``, ``n_points``,
        ``n_stimuli``, and ``hallem``: `to_hallem`'s of the combination), ``glomeruli``,
        ``odors`` and ``odor_panels`` (the input's, each odor's panel), ``panels`` (those
        taken), ``dropped`` (glomeruli not measured for every odor), ``mixtures`` (stimuli
        left out), ``tolerance``, and the frames ``responses`` and ``orn_deltas`` (glomeruli
        × odors: the means, and those times the factor).

    Raises:
        InputError: if they are not datasets of glomeruli, have too few glomeruli measured
            for every odor, or cannot be scaled (too little in common with Hallem's, and
            with the other datasets, or those with Hallem's).
    """
    if not ids:
        raise InputError("choose a dataset of glomeruli")
    for i in ids:
        d = db.get_dataset(i)
        if d is None:
            raise InputError(f"there is no dataset #{i}")
        if d["roi_kind"] != "glomeruli":
            raise InputError(f"#{i} ({d['title']}) is not a dataset of glomeruli: the models take ORN responses by glomerulus")
    every = means(db, ids, index)
    mine = every if of_panels is None else means(db, ids, index, of_panels)
    if not mine["values"]:
        raise InputError("it has no responses of glomeruli to odors of one component" + ("" if of_panels is None else " in the panels chosen"))
    frame = pd.DataFrame({mine["labels"][k]: by for k, by in mine["values"].items()})
    frame = frame.loc[sorted(frame.index)]
    whole = frame.notna().all(axis=1)
    dropped = [str(g) for g in frame.index[~whole]]
    frame = frame[whole]
    if len(frame) < MIN_GLOMERULI:
        raise InputError(f"only {len(frame)} of its glomeruli {'is' if len(frame) == 1 else 'are'} measured for every odor "
                         f"(at least {MIN_GLOMERULI} are needed): {', '.join(dropped[:20])}{', …' if len(dropped) > 20 else ''} "
                         f"{'is' if len(dropped) == 1 else 'are'} not")
    frame.index.name, frame.columns.name = "glomerulus", "odor"
    hal = hallem(index)
    direct = to_hallem(every["values"], hal, every["labels"], tolerance)
    out = {"method": HALLEM, "factor": direct["factor"], "hallem": direct, "combination": None, "tolerance": tolerance,
           "glomeruli": [str(g) for g in frame.index], "odors": [str(o) for o in frame.columns],
           "odor_panels": [mine["panels"][k] for k in mine["values"]], "panels": of_panels, "dropped": dropped,
           "mixtures": mine["mixtures"]}
    if not direct["enough"]:
        c = _combination(db, ids, every, index, tolerance)
        of_all = to_hallem(c["values"], hal, c["labels"], tolerance)
        short = (f"it has {direct['n_odors']} odor{'' if direct['n_odors'] == 1 else 's'} ({direct['n_points']} odor-glomerulus "
                 f"pairs) in common with Hallem & Carlson's, and at least {MIN_ODORS} ({MIN_POINTS} pairs) are needed")
        if not c["datasets"]:
            raise InputError(f"it cannot be scaled into spike rates: {short}, and there is no other dataset of its coding unit to scale it to")
        if c["factor"] is None or c["factor"] <= 0:
            raise InputError(f"it cannot be scaled into spike rates: {short}, and it shares no responses with the other "
                             f"{len(c['datasets'])} datasets of its coding unit that would scale it to them")
        if not of_all["enough"]:
            raise InputError(f"it cannot be scaled into spike rates: {short}, and the combination of all datasets of its coding unit "
                             f"has too few too ({of_all['n_odors']} odors, {of_all['n_points']} pairs)")
        out.update(method=COMBINATION, factor=c["factor"] * of_all["factor"],
                   combination={**{k: c[k] for k in ("datasets", "factor", "r", "n_points", "n_stimuli")}, "hallem": of_all})
    if out["factor"] is None or not math.isfinite(out["factor"]):
        raise InputError("it cannot be scaled into spike rates: its responses in common with Hallem & Carlson's are all 0")
    out["responses"], out["orn_deltas"] = frame, frame * out["factor"]
    return out


def model_frame(s: dict) -> pd.DataFrame:
    """`scale`'s ``orn_deltas`` as pyolfsysm takes ORN input (and reads it: `read_orn_deltas`):
    its columns each odor under its panel."""
    df = s["orn_deltas"].copy()
    df.columns = pd.MultiIndex.from_arrays([s["odor_panels"], s["odors"]], names=["panel", "odor"])
    return df


def summary(s: dict) -> dict:
    """`scale`'s, without its frames and points (for the Models tab, and a run's record)."""
    def fit(h: dict) -> dict:
        return {k: h[k] for k in ("factor", "r", "n_odors", "n_points", "enough")}

    out = {k: s[k] for k in ("method", "factor", "tolerance", "glomeruli", "odors", "panels", "dropped", "mixtures")}
    out["hallem"] = fit(s["hallem"])
    out["combination"] = s["combination"] and {**{k: v for k, v in s["combination"].items() if k != "hallem"},
                                                "hallem": fit(s["combination"]["hallem"])}
    return out


def said(s: dict) -> str:
    """How `scale` (or its `summary`) scaled the responses, in a sentence or two."""
    h = s["hallem"]
    r = lambda x: "" if x is None else f", r = {x:.2f}"  # noqa: E731
    if s["method"] == HALLEM:
        return (f"Scaled into changes in spike rate by {s['factor']:.4g} per unit of response: the factor that best matches its "
                f"responses to Hallem & Carlson's (2006), over the {h['n_odors']} odors ({h['n_points']} odor-glomerulus pairs{r(h['r'])}) "
                f"they have in common.")
    c = s["combination"]
    return (f"Scaled into changes in spike rate by {s['factor']:.4g} per unit of response. It has too little in common with Hallem & "
            f"Carlson's (2006) for a factor of its own ({h['n_odors']} odors, {h['n_points']} odor-glomerulus pairs; at least {MIN_ODORS} "
            f"and {MIN_POINTS} are needed), so it was first scaled by {c['factor']:.4g} to the combination of all datasets of its coding "
            f"unit (it and {len(c['datasets'])} other{'' if len(c['datasets']) == 1 else 's'}: over {c['n_stimuli']} odors, {c['n_points']} "
            f"pairs{r(c['r'])}), and that by {c['hallem']['factor']:.4g} to Hallem & Carlson's ({c['hallem']['n_odors']} odors, "
            f"{c['hallem']['n_points']} pairs{r(c['hallem']['r'])}).")
