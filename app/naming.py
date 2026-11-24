"""How a prediction's ROIs were named, laid out for the AL segmentation tab's naming explorer (static/naming.js):
where the model's atlas puts each glomerulus on the fly's planes, the ROIs the names were given to, and what naming
each ROI as each glomerulus would cost (al-diag-seg's `identity.assign_identities`: an ROI is named as a glomerulus only
where that costs no more than the model's ``max_cost``).

Two parts, by who can make them:

- from what a prediction kept, in the web server (`view`, `volumes`): its fly's average image, the labels of the ROIs
  that were named (`named_kind`), the atlas on the fly's planes (``<fly>_atlas_prior.npz``: each glomerulus'
  probability at each voxel) and the terms of the cost of each ROI's few best glomeruli (`predictions.identity_terms`);
- the cost of every pair of an ROI and a glomerulus, term by term, with what the layout term is made of (`costs`):
  made by the fit worker the first time it is asked for (`make_costs`, through `nmf_check.call`: the web server has no
  al-diag-seg), by naming the ROIs again as the prediction did, and kept in the prediction's folder
  (``<fly>_naming_costs.json``, `COSTS`). With it the page names the ROIs again itself under another ``max_cost`` or
  other weights of the terms, exactly as al-diag-seg would.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Optional

from . import fits, nmf_check, predictions

COSTS = "naming_costs"  # in the name of a prediction's file of every pair's cost (`costs`)
FORMAT = 1  # of that file: one of another format is made again
MAGIC = b"OLFN"  # what `volumes` starts with
CALL = "naming_costs"  # what the web server asks of the fit worker (`nmf_check.CALLS`)
TERMS = ("tuning", "position", "shape", "atlas", "layout")  # al-diag-seg's `identity.TERMS`
CAP = 4.0  # the SDs al-diag-seg's position, shape and layout terms are capped at (and divided by)


class Unavailable(Exception):
    """A prediction has not what the explorer shows: why."""


def named_kind(p: dict) -> Optional[str]:
    """The kind of a prediction's ROIs that its names were given to: its ROIs to use (``final``), but for cleaned ones,
    which keep the names of those they were cleaned from (those split, else the model's own). None if it has none."""
    d, fly = predictions.pred_dir(p["id"]), p["fly"]
    final = (p.get("result") or {}).get("final")
    if final and final.startswith("clean"):
        return next((k for k in ("split", "pred") if (d / f"{fly}_{k}.tif").is_file()), None)
    return final


def files(p: dict) -> dict:
    """The files of a prediction the explorer reads, in its folder (``labels``: None if no kind of its ROIs was named)."""
    d, fly, kind = predictions.pred_dir(p["id"]), p["fly"], named_kind(p)
    return {"labels": d / f"{fly}_{kind}.tif" if kind else None, "prior": d / f"{fly}_atlas_prior.npz",
            "average": d / predictions.AVERAGE, "costs": d / f"{fly}_{COSTS}.json", "inputs": d / nmf_check.INPUTS,
            "provenance": d / predictions.PROVENANCE}


def _done(db, pred_id: int) -> dict:
    p = predictions.get(db, pred_id)
    if p is None or p["status"] != "done":
        raise LookupError("no such prediction done")
    return p


def _labels(p: dict):
    import tifffile

    f = files(p)
    if f["labels"] is None or not f["labels"].is_file():
        raise Unavailable("the labels of its ROIs are not in its folder: predict it again")
    lab = tifffile.imread(f["labels"])
    if lab.ndim == 2:
        lab = lab[None]
    return lab


def _prior(p: dict, shape):
    """The atlas on the fly's planes ((glomeruli, Z, Y, X) and their names), or (None, []) for a prediction without it
    (of a model with no atlas, or made before it was kept, or of other planes than its labels)."""
    import numpy as np

    path = files(p)["prior"]
    if not path.is_file():
        return None, []
    z = np.load(path)
    prior, names = z["prior"], [str(n) for n in z["names"]]
    if prior.shape[1:] != tuple(shape) or len(names) != len(prior):
        return None, []
    return prior, names


def alignment(p: dict) -> Optional[dict]:
    """How the atlas a prediction kept was moved to where its fly's ROIs say its glomeruli are, before they were named
    with it there (al-diag-seg's `alseg.model.Model.align`, as its ``<fly>_atlas_prior.npz`` has it: ``kind``,
    ``matrix`` and ``offset_um`` (a place x of the atlas, um (z, y, x), went to matrix x + offset_um), ``shift_um`` (of
    its middle), ``rois`` (how many ROIs it was fit to), ``rois_named_by`` (which: "tuning", those odor tuning alone
    names, or "atlas", those named with the atlas as registered), ``weights`` (each one's in the fit) and
    ``before_um`` and ``after_um`` (their median distance from where the atlas expected their glomeruli)). None for an
    atlas left where registration put it: too few such ROIs, a model that does not ask for it, or a prediction made
    before atlases were moved."""
    import numpy as np

    try:
        z = np.load(files(p)["prior"])
        return json.loads(str(z["alignment"])) if "alignment" in z.files else None
    except (OSError, ValueError, KeyError):
        return None


def _stat(path: Path) -> list[int]:
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def _model_info(fit_id: int) -> dict:
    """What a fit's model says of its glomeruli, as far as its folder is there: ``flies`` (how many are behind each
    one's map in the atlas), ``trained`` (how many training flies have it) and ``sd`` (the SD of its position, um),
    each by name; ``has_atlas`` and ``max_cost``."""
    import numpy as np

    mdir = fits.fit_dir(fit_id) / "model"
    out = {"flies": {}, "trained": {}, "sd": {}, "has_atlas": None, "max_cost": None}
    try:
        m = json.loads((mdir / "model.json").read_text())
        out["has_atlas"], out["max_cost"] = bool(m.get("atlas")), (m.get("options") or {}).get("max_cost")
        for n, g in json.loads((mdir / "template.json").read_text()).items():
            out["trained"][n], out["sd"][n] = g.get("n"), g.get("centroid_sd")
        if out["has_atlas"]:
            a = np.load(mdir / "atlas.npz")  # (its members are read one by one: not the maps)
            out["flies"] = {str(n): float(c) for n, c in zip(a["names"], a["counts"])}
    except (OSError, ValueError, KeyError):
        pass
    return out


def view(db, pred_id: int) -> dict:
    """What the explorer shows of a prediction, but for its volumes (`volumes`) and every pair's cost (`costs`):

    - ``id``, ``fit_id``, ``fly``, ``kind`` (the ROIs the names were given to: `named_kind`), ``final`` (its ROIs to
      use), ``shape`` ([Z, Y, X]) and ``spacing`` (um, as its provenance says; None if it does not);
    - ``rois``: per ROI its ``label``, ``name`` (None: left unnamed), ``voxels``, ``planes`` (those it is on, from 0)
      and ``centroid`` ([z, y, x], in voxels);
    - ``maps``: per glomerulus of the atlas on the fly's planes (in the order of `volumes`' maps), its ``name``,
      ``peak`` (its largest probability there), ``at`` (where: [z, y, x]), ``planes`` (its largest on each plane),
      ``centre`` (its probability's centre of mass, in voxels: where the position term expects it; None if it is 0
      everywhere) and ``mass`` (its probabilities summed, in voxels); [] without an atlas on the fly's planes;
    - ``alignment``: how that atlas was moved to where the fly's ROIs say its glomeruli are, before they were named
      with it (`alignment`; None if it was not, or there is none);
    - ``glomeruli``: {name: {``flies`` (behind its map in the model's atlas), ``trained`` (training flies with it;
      None for one of the reference atlas alone), ``sd`` (of its position, um)}}, as far as the fit's model is there;
    - ``terms``: `predictions.identity_terms` (each ROI's few best glomeruli; ``rois`` None if they were not kept);
    - ``costs``: whether every pair's cost is made already (`costs_state`);
    - ``model``: ``has_atlas``, ``max_cost``.

    Raises:
        LookupError: no such prediction done.
        Unavailable: its folder has not its ROIs' labels.
    """
    import numpy as np
    from scipy import ndimage as ndi

    p = _done(db, pred_id)
    lab = _labels(p)
    terms = predictions.identity_terms(db, pred_id) or {"rois": None}
    named = {int(l): r.get("assigned") for l, r in (terms.get("rois") or {}).items()}
    ids = [int(l) for l in np.unique(lab) if l]
    cents = ndi.center_of_mass(lab > 0, lab, ids) if ids else []
    rois = [{"label": l, "name": named.get(l), "voxels": int((lab == l).sum()),
             "planes": [int(z) for z in np.flatnonzero((lab == l).any((1, 2)))], "centroid": [float(v) for v in c]}
            for l, c in zip(ids, cents)]
    prior, names = _prior(p, lab.shape)
    maps = []
    for k, n in enumerate(names):
        m = prior[k].astype(np.float32)
        mass = float(m.sum())
        maps.append({"name": n, "peak": float(m.max()), "at": [int(v) for v in np.unravel_index(int(m.argmax()), m.shape)],
                     "planes": [float(v) for v in m.max((1, 2))], "mass": mass,
                     "centre": [float(v) for v in ndi.center_of_mass(m)] if mass > 0 else None})
    try:
        spacing = [float(s) for s in json.loads(files(p)["provenance"].read_text())["fly"]["spacing_um"]]
    except (OSError, ValueError, KeyError, TypeError):
        spacing = None
    info = _model_info(p["fit_id"])
    every = sorted(set(info["trained"]) | set(info["flies"]) | set(names))
    return {"id": p["id"], "fit_id": p["fit_id"], "fly": p["fly"], "kind": named_kind(p), "final": (p.get("result") or {}).get("final"),
            "shape": list(lab.shape), "spacing": spacing, "rois": rois, "maps": maps, "alignment": alignment(p) if maps else None,
            "glomeruli": {n: {"flies": info["flies"].get(n), "trained": info["trained"].get(n), "sd": info["sd"].get(n)} for n in every},
            "terms": terms, "costs": costs_state(p), "model": {"has_atlas": info["has_atlas"], "max_cost": info["max_cost"]}}


def volumes(db, pred_id: int) -> bytes:
    """A prediction's volumes for the explorer, as one block of bytes: `MAGIC`, the length of a header (uint32, little
    endian), the header (JSON: ``shape`` [Z, Y, X], ``maps``, the names of the atlas' glomeruli there), then, each Z × Y
    × X in C order: the fly's average image (uint8: each plane from its 1st to its 99.8th percentile), its ROIs' labels
    (uint16, little endian: those the names were given to) and each glomerulus' probability (uint8: 255 for 1).

    Raises:
        LookupError, Unavailable: as `view`.
    """
    import numpy as np
    import tifffile

    p = _done(db, pred_id)
    lab = _labels(p)
    try:
        avg = tifffile.imread(files(p)["average"]).astype(np.float32)
        avg = avg[None] if avg.ndim == 2 else avg
    except (OSError, ValueError):
        avg = None
    if avg is None or avg.shape != lab.shape:
        avg = np.zeros(lab.shape, np.float32)
    img = np.zeros(lab.shape, np.uint8)
    for z, plane in enumerate(avg):
        lo, hi = np.percentile(plane, [1, 99.8])
        if hi > lo:
            img[z] = np.clip((plane - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    prior, names = _prior(p, lab.shape)
    head = json.dumps({"shape": list(lab.shape), "maps": names}).encode()
    parts = [MAGIC, struct.pack("<I", len(head)), head, img.tobytes(), lab.astype("<u2").tobytes()]
    if prior is not None:
        parts.append(np.rint(np.clip(prior.astype(np.float32), 0, 1) * 255).astype(np.uint8).tobytes())
    return b"".join(parts)


# -------------------------------------------------------- every pair's cost

def _read_costs(p: dict) -> Optional[dict]:
    """A prediction's file of every pair's cost, if it is there, of this format and of its ROIs as they are."""
    f = files(p)
    try:
        doc = json.loads(f["costs"].read_text())
    except (OSError, ValueError):
        return None
    try:
        fresh = doc.get("format") == FORMAT and f["labels"] is not None and doc["of"]["labels"] == f["labels"].name \
            and doc["of"]["stat"] == _stat(f["labels"])
    except (OSError, KeyError, TypeError):
        fresh = False
    return doc if fresh else None


def costs_state(p: dict) -> dict:
    """Whether every pair's cost is there for a prediction (``made``: when, else None)."""
    doc = _read_costs(p)
    return {"made": doc["made"] if doc else None}


def costs(db, pred_id: int, make: bool = False) -> Optional[dict]:
    """What naming each of a prediction's ROIs as each glomerulus costs (`make_costs` says what is in it), as kept in
    its folder; if it is not there (or is of other ROIs: the prediction refined since), None, or with ``make``, made
    by the fit worker now (a few seconds, more for a fly whose files on tensor are slow to read).

    Raises:
        LookupError: no such prediction done.
        nmf_check.CallError: the fit worker did not make it (it is not running, or failed: why).
    """
    p = _done(db, pred_id)
    doc = _read_costs(p)
    if doc is None and make:
        nmf_check.call(db, CALL, {"prediction": pred_id})
        doc = _read_costs(_done(db, pred_id))
        if doc is None:
            raise nmf_check.CallError("the fit worker made it, but of other ROIs than the prediction now has: open it again")
    return doc


def _fly(db, p: dict, model) -> tuple[dict, str]:
    """In the fit worker: a prediction's fly as far as naming needs it (its ``responses``, in the model's odors it
    has, ``odors`` and ``spacing``), and where from: what a look for missed glomeruli kept of it (`nmf_check.INPUTS`),
    else its folders on tensor, read as a prediction reads them."""
    import numpy as np

    kept = files(p)["inputs"]
    if kept.is_file():
        z = np.load(kept)
        return {"responses": z["responses"], "spacing": tuple(float(s) for s in z["spacing"]),
                "odors": [str(x) for x in z["odors"]] if "odors" in z else None}, "kept"
    from alseg.fly import load_fly

    from . import fit_worker, odormeta  # (fit_worker imports this module)
    from .glomeruli import load_glomeruli

    fly_dir, raw_dir = Path(p["fly_dir"]), Path(p["raw_dir"]) if p["raw_dir"] else None
    fit_worker._fly_inputs(fly_dir, raw_dir, model, odormeta.OdorIndex(db.list_odors()))  # (the fly's odors as recorded)
    return load_fly(fly_dir, model, raw_dir=raw_dir, name=p["fly"], glomeruli=load_glomeruli(), tentative=model.options.tentative,
                    stack=False), "tensor"


def make_costs(db, args: dict) -> dict:
    """In the fit worker: names a prediction's ROIs again as it named them (`alseg.model.Model.name`, on the labels the
    names were given to, with the atlas on the fly's planes as the prediction kept it), keeping every glomerulus' cost
    for every ROI, and writes that as the prediction's ``<fly>_naming_costs.json``:

    - ``labels`` (the ROIs', in rows) and ``names`` (the glomeruli that can name one, in columns; ``reference``: which
      are of the reference atlas alone, their tuning and shape terms fixed at ``unknown_cost``);
    - ``terms``: each of `TERMS`' value for every pair (rows × columns; None where it has none), before its weight;
      ``weights``, ``weight_sum`` and ``max_cost``, as al-diag-seg gives them; ``assigned`` (per ROI, the name this gave
      it: None, unnamed), ``predicted`` (the name the prediction gave it) and ``same`` (whether those agree: they may
      not if the fly's files changed since, or a cost is within rounding of another);
    - what the layout term is recomputed from as names change: ``roi_pos`` and ``exp_pos`` (um: where each ROI is, and
      where each glomerulus is expected), ``pos_sd`` (um, per glomerulus), ``neighbors`` (per ROI, the rows of the ROIs
      next to it) and ``rounds`` (how many times the names are matched again with it);
    - ``exact``: whether the terms, these and the weights give al-diag-seg's own costs and layout term again (``why``
      not, else None), so that the page's matching under other weights is al-diag-seg's;
    - ``tuning``: each ROI's odor tuning (``rois``) and each glomerulus' mean one in the training flies
      (``glomeruli``; None for one of the reference atlas alone), z-scored over the fly's ``odors``;
    - ``of`` (the labels' file, with its size and time: the costs are out of date once it changes), ``responses``
      (where the fly's responses were read from), ``made`` and ``alseg``.

    Args:
        args: ``prediction``, its id.

    Returns:
        ``made``: when.

    Raises:
        ValueError: no such prediction done, or its folder (or its fit's) has not what naming needs.
    """
    import inspect

    import numpy as np
    from alseg import identity, refine
    from alseg import model as am
    from scipy.ndimage import center_of_mass

    p = predictions.get(db, int(args["prediction"]))
    if p is None or p["status"] != "done":
        raise ValueError("no such prediction done")
    f, kind = files(p), named_kind(p)
    try:
        labels = _labels(p)
    except Unavailable as e:
        raise ValueError(str(e)) from None
    stat = _stat(f["labels"])
    mdir = fits.fit_dir(p["fit_id"]) / "model"
    if not (mdir / "model.json").is_file():
        raise ValueError(f"fit #{p['fit_id']}'s model is not there (the fit was deleted)")
    model = am.Model.load(mdir, atlas=False)
    prior = prior_names = None
    if json.loads((mdir / "model.json").read_text()).get("atlas"):
        prior, prior_names = _prior(p, labels.shape)
        if prior is None:
            raise ValueError("the atlas on its fly's planes was not kept with it (or is of other planes): predict it again")
        prior = prior.astype(np.float32)
    fly, source = _fly(db, p, model)
    if tuple(fly["responses"].shape[1:]) != labels.shape:
        raise ValueError(f"the fly's responses are of planes {fly['responses'].shape[1:]}, its ROIs of {labels.shape}: predict it again")
    ids, terms = model.name(fly, labels, prior, prior_names, n_best=10**6)

    # what the position and layout terms are made of, as `identity.assign_identities` has it
    sp = np.asarray(fly["spacing"], float)
    template = model.template_of(fly)
    feats = identity.roi_features(fly["responses"], labels, sp)
    labs = list(feats)
    names = list(template)
    if prior is not None:
        expected = identity.atlas_centroids(prior, prior_names, sp)
        roi_pos = {l: np.array(center_of_mass(labels == l)) * sp for l in labs}
        names += [n for n in prior_names if n not in template and n in expected]
    else:
        expected = {n: template[n]["centroid"] for n in names}
        roi_pos = {l: feats[l]["centroid"] for l in labs}
    default_sd = np.median([g["centroid_sd"] for g in template.values()])
    exp_pos = np.array([expected.get(n, template.get(n, {}).get("centroid")) for n in names], float)
    pos_sd = np.array([template[n]["centroid_sd"] if n in template else default_sd for n in names], float)
    nb = refine.neighbors(labels, spacing=sp)

    nL, nN = len(labs), len(names)
    col = {n: j for j, n in enumerate(names)}
    parts = {k: np.full((nL, nN), np.nan) for k in TERMS}
    cost = np.full((nL, nN), np.nan)
    why = None
    for i, l in enumerate(labs):
        for c in terms["rois"][int(l)]["candidates"]:
            j = col.get(c["name"])
            if j is None:
                why = f"al-diag-seg names ROIs as {c['name']}, which is not among the glomeruli worked out here"
                continue
            cost[i, j] = c["cost"]
            for k in TERMS:
                if c[k] is not None:
                    parts[k][i, j] = c[k]
    if why is None and np.isnan(cost).any():
        why = "al-diag-seg did not give every glomerulus' cost for every ROI"
    w = terms["weights"]
    assign = np.array([col.get(ids[l][0], -1) if ids[l][0] else -1 for l in labs])
    if why is None:  # (the terms, the weights and what is above give al-diag-seg's own numbers again)
        d = np.array([[np.linalg.norm(roi_pos[l] - exp_pos[j]) / (pos_sd[j] + 1e-8) for j in range(nN)] for l in labs]).reshape(nL, nN)
        layout = identity._layout_cost(labs, nb, roi_pos, exp_pos, pos_sd, assign) if w["layout"] else np.zeros((nL, nN))
        total = sum((w[k] or 0) * np.nan_to_num(parts[k]) for k in TERMS) / terms["weight_sum"]
        if not np.allclose(parts["position"], np.minimum(d, CAP) / CAP, atol=1e-9):
            why = "the position term is not what it is taken to be here (this al-diag-seg computes it otherwise)"
        elif w["layout"] and not np.allclose(parts["layout"], layout, atol=1e-9):
            why = "the layout term is not what it is taken to be here (this al-diag-seg computes it otherwise)"
        elif not np.allclose(cost, total, atol=1e-9):
            why = "the cost is not the weighted mean of its terms (this al-diag-seg computes it otherwise)"
    row = {l: i for i, l in enumerate(labs)}
    predicted = _predicted(p, kind)
    assigned = [ids[l][0] for l in labs]
    num = lambda a: [[None if np.isnan(v) else float(v) for v in r] for r in a]  # noqa: E731
    doc = {
        "format": FORMAT, "prediction": p["id"], "fly": p["fly"], "kind": kind, "made": predictions.now(), "alseg": am.version(),
        "of": {"labels": f["labels"].name, "stat": stat}, "responses": source,
        "spacing": [float(s) for s in sp], "odors": list(fly.get("odors") or model.odors),
        "labels": [int(l) for l in labs], "names": names, "reference": [n not in template for n in names],
        "weights": w, "weight_sum": terms["weight_sum"], "max_cost": terms["max_cost"], "unknown_cost": terms["unknown_cost"],
        "rounds": inspect.signature(identity.assign_identities).parameters["n_layout_iter"].default,
        "terms": {k: num(parts[k]) for k in TERMS}, "assigned": assigned,
        "predicted": [predicted.get(int(l)) for l in labs] if predicted is not None else None,
        "same": predicted is not None and all(predicted.get(int(l)) == a for l, a in zip(labs, assigned)),
        "roi_pos": [[float(v) for v in roi_pos[l]] for l in labs], "exp_pos": exp_pos.tolist(), "pos_sd": pos_sd.tolist(),
        "neighbors": [sorted(row[m] for m in nb.get(l, ()) if m in row) for l in labs],
        "exact": why is None, "why": why,
        "tuning": {"rois": [np.asarray(feats[l]["tuning"], float).tolist() for l in labs],
                   "glomeruli": [np.asarray(template[n]["tuning"], float).tolist() if n in template else None for n in names]},
    }
    tmp = f["costs"].with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(f["costs"])
    return {"made": doc["made"]}


def _predicted(p: dict, kind: Optional[str]) -> Optional[dict]:
    """The names a prediction gave the ROIs of `kind` ({label: name or None}), as its folder has them; None if not."""
    import csv

    d, fly = predictions.pred_dir(p["id"]), p["fly"]
    own = (p.get("result") or {}).get("final") == kind
    for path in (d / f"{fly}_identities_{kind}.csv", *([d / f"{fly}_identities.csv"] if own else [])):
        if path.is_file():
            with open(path, newline="") as fh:
                return {int(r["label"]): r["assigned"] or None for r in csv.DictReader(fh)}
    return None


nmf_check.CALLS[CALL] = make_costs
