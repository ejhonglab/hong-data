"""How a fit's model does on each glomerulus, over the flies of one experimenter that it predicted and that have ROIs
of their own, for the AL segmentation tab's box of it (static/by_glomerulus.js): of the flies a glomerulus is drawn
in, in how many an ROI was found on it and what that ROI was named; and where else the model gave its name.

Two parts, by who can make them:

- per prediction, its ROIs matched to its fly's own (`make_one`), as al-diag-seg scores them (`alseg.model.Model.evaluate`'s
  ``glomeruli`` and `alseg.model.score`'s names, by `alseg.metrics.iou_matrix_partly_drawn`), of each kind of its ROIs
  there is (`_of`): made by the fit worker (the web server has no al-diag-seg: it asks through `nmf_check.call`, `make`)
  and kept in the prediction's folder (``<fly>_by_glomerulus.json``, `FILE`), until its ROIs or its fly's own change;
- over a fit's predictions, in the web server (`summary`).

The fly's own ROIs are those al-diag-seg scores a prediction against (`alseg.fly.load_fly`): its RoiSet.zip's ROIs
named as a glomerulus, and tentative ones as that glomerulus (if the model was fit so), as they are now (not as they
were when it was predicted).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from . import fits, naming, nmf_check, predictions, roi_training
from .movies import _natural

FILE = "by_glomerulus"  # in the name of a prediction's file of its ROIs matched to its fly's own (`make_one`)
FORMAT = 1  # of that file: one of another format is made again
CALL = "by_glomerulus"  # what the web server asks of the fit worker (`nmf_check.CALLS`)
KINDS = ("pred", "split", "clean")  # the kinds of a prediction's ROIs matched (with its final ones, if they are another)
BUDGET_S = 20.0  # how long the worker goes on to another prediction of those asked for at once (`make`)
MOST = 500  # predictions asked for at once
FLIES = ("new", "all")  # the flies summed over: those the model was not trained on, or all it predicted
ROIS = ("final", "pred")  # their ROIs: the prediction's (refined, where they were), or the model's own
OTHER = "?"  # the experimenter of a fly not named as tensor's are
OVER = 0.5  # an ROI is over or under a glomerulus drawn on other planes if this much of it, seen from above, lies in it (`make_one`'s ``over``)
# what `summary` counts of a glomerulus, in flies (beside ``other`` and ``given_wrong``, which are by name)
COUNTS = ("drawn", "found", "named", "unnamed", "none", "untouched", "moved", "given", "given_right", "given_weak", "given_stacked",
          "given_elsewhere", "given_undrawn")

_FLY = re.compile(r"(.+)_\d{4}-\d{2}-\d{2}_\d+")
_cache: dict[int, tuple[list, dict]] = {}  # prediction -> (its file's size and time, what it says)


def experimenter(p: dict) -> str:
    """Whose a prediction's fly is: the top folder of its data on tensor, which its name starts with
    (`predictions.fly_name`); `OTHER` for a fly named otherwise (one not on tensor)."""
    m = _FLY.fullmatch(p.get("fly") or "")
    return m.group(1) if m else OTHER


def _stat(path) -> Optional[list[int]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def doc_path(p: dict) -> Path:
    return predictions.pred_dir(p["id"]) / f"{p['fly']}_{FILE}.json"


def _provenance(p: dict) -> dict:
    try:
        return json.loads((predictions.pred_dir(p["id"]) / predictions.PROVENANCE).read_text())
    except (OSError, ValueError):
        return {}


def _of(p: dict, prov: Optional[dict] = None) -> dict:
    """What a prediction's matches are of: its fly's own ROIs (``roiset``: the RoiSet.zip it was scored against, as
    its provenance says, with its size and time as they are now; the path alone if it cannot be looked at; None if
    the fly had none) and each kind of its ROIs with names (``labels``: {kind: its labels' size and time})."""
    d, fly = predictions.pred_dir(p["id"]), p["fly"]
    path = ((prov if prov is not None else _provenance(p)).get("fly") or {}).get("roiset")
    final = (p.get("result") or {}).get("final")
    labels = {}
    for k in dict.fromkeys((*KINDS, *([final] if final else []))):
        st = _stat(d / f"{fly}_{k}.tif")
        if st and (d / f"{fly}_identities_{k}.csv").is_file():
            labels[k] = st
    return {"roiset": [path, *(_stat(path) or [])] if path else None, "labels": labels}


def _fresh(doc: dict, of: dict) -> bool:
    """Whether a prediction's matches kept are of its ROIs and its fly's own as they are (`_of`). A RoiSet that cannot
    be looked at here (the web server may not read every folder the worker reads) is taken as unchanged."""
    was = doc.get("of") or {}
    if doc.get("format") != FORMAT or was.get("labels") != of["labels"]:
        return False
    a, b = was.get("roiset"), of["roiset"]
    return a == b or bool(a and b and a[0] == b[0] and len(b) == 1)


def read(p: dict) -> Optional[dict]:
    """A prediction's ROIs matched to its fly's own (`make_one`'s, or ``error``: why they could not be), as kept in its
    folder; None if they are not there, or are of other ROIs than it and its fly now have."""
    path = doc_path(p)
    st = _stat(path)
    if st is None:
        _cache.pop(p["id"], None)
        return None
    kept = _cache.get(p["id"])
    if kept is None or kept[0] != st:
        try:
            kept = (st, json.loads(path.read_text()))
        except (OSError, ValueError):
            return None
        _cache[p["id"]] = kept
    return kept[1] if _fresh(kept[1], _of(p)) else None


# ------------------------------------------------------ in the fit worker

def make_one(p: dict) -> dict:
    """In the fit worker: a prediction's ROIs matched to its fly's own, as al-diag-seg scores a prediction (a
    predicted ROI's voxels on planes where a glomerulus is not drawn count against it only where they lie in another
    drawn one: glomeruli are often drawn on some of their planes only).

    Returns:
        dict with ``of`` (`_of`), ``changed`` (whether the fly's RoiSet.zip is another than when it was predicted),
        ``detect_iou`` and ``name_iou`` (al-diag-seg's: a glomerulus is found by an ROI at the first or over, and an
        ROI is on a glomerulus over the second), and ``kinds``: of each kind of its ROIs, ``manual`` (per ROI of the
        fly's own: its ``name``, the largest ``iou`` of a predicted ROI with it, that one's label, ``roi``, and
        name, ``as``: None for one left unnamed, or if none touches it) and ``predicted`` (per predicted ROI: its
        ``label``, ``name``, the fly's own ROI it overlaps most, ``manual``, their ``iou`` and, if the fly has an ROI
        of its own of that name, ``over``: how much of the predicted one, seen from above (its outlines on all its
        planes as one), lies in that one's, seen so: near 1 for one over or under it, on planes where it is not drawn).

    Raises:
        ValueError: the fly had no ROIs of its own, or they (or the prediction's) cannot be read.
    """
    import csv

    import tifffile
    from alseg import metrics, roi_io
    from alseg import model as am
    from alseg import olfdb as ao

    from .glomeruli import load_glomeruli

    d, fly, prov = predictions.pred_dir(p["id"]), p["fly"], _provenance(p)
    of = _of(p, prov)
    if not of["roiset"]:
        raise ValueError("its fly had no ROIs of its own when it was predicted")
    if len(of["roiset"]) == 1:
        raise ValueError(f"its fly's own ROIs are no longer there ({of['roiset'][0]})")
    if not of["labels"]:
        raise ValueError("its folder has not its ROIs' labels and names: predict it again")
    tentative = bool(((prov.get("model") or {}).get("alseg_options") or {}).get("tentative", True))
    keep = ao.roi_keeper(set(load_glomeruli()), tentative=tentative)
    gt = lut = None
    kinds, seen_from_above = {}, {}
    for kind in of["labels"]:
        lab = tifffile.imread(d / f"{fly}_{kind}.tif")
        lab = lab[None] if lab.ndim == 2 else lab
        if gt is None:
            try:
                gt, lut, _ = roi_io.load_roiset(of["roiset"][0], lab.shape, keep=keep)
            except Exception as e:  # noqa: BLE001 (roifile raises many kinds)
                raise ValueError(f"its fly's own ROIs could not be read ({type(e).__name__}: {e})") from None
            seen_from_above = {n: (gt == g).any(0) for g, n in lut.items()}  # (each one's outlines on all its planes as one)
        if lab.shape != gt.shape:
            continue
        with open(d / f"{fly}_identities_{kind}.csv", newline="") as fh:
            names = {int(r["label"]): r["assigned"] or None for r in csv.DictReader(fh)}
        iou, g_ids, p_ids, _ = metrics.iou_matrix_partly_drawn(gt, lab)
        manual, predicted = [], []
        for gi, g in enumerate(g_ids):
            if lut.get(int(g)) is None:
                continue
            best = int(iou[gi].argmax()) if iou.shape[1] else -1
            b = float(iou[gi, best]) if best >= 0 else 0.0
            roi = int(p_ids[best]) if b > 0 else None
            manual.append({"name": lut[int(g)], "iou": b, "roi": roi, "as": names.get(roi) if roi is not None else None})
        for j, pl in enumerate(p_ids):
            best = int(iou[:, j].argmax()) if iou.shape[0] else -1
            b = float(iou[best, j]) if best >= 0 else 0.0
            name, over = names.get(int(pl)), None
            if name in seen_from_above:
                foot = (lab == pl).any(0)
                over = float((foot & seen_from_above[name]).sum() / max(int(foot.sum()), 1))
            predicted.append({"label": int(pl), "name": name, "manual": lut.get(int(g_ids[best])) if b > 0 else None, "iou": b, "over": over})
        kinds[kind] = {"manual": manual, "predicted": predicted}
    was = (prov.get("fly") or {}).get("roiset_file") or {}
    try:
        same = was.get("bytes") == of["roiset"][1] and int(dt.datetime.fromisoformat(was["modified"]).timestamp()) == of["roiset"][2] // 10**9
    except (KeyError, TypeError, ValueError):
        same = None
    return {"format": FORMAT, "prediction": p["id"], "fly": fly, "made": predictions.now(), "alseg": am.version(), "scoring": "partly drawn",
            "detect_iou": am.DETECT_IOU, "name_iou": am.NAME_IOU, "of": of, "changed": None if same is None else not same, "kinds": kinds}


def _write(p: dict, doc: dict) -> None:
    path = doc_path(p)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)


def make(db, args: dict) -> dict:
    """In the fit worker: matches predictions' ROIs to their flies' own (`make_one`) and keeps that in their folders
    (one that cannot be matched keeps why, ``error``, until its files change), one after another until `BUDGET_S` is
    over (``seconds``, if given).

    Args:
        args: ``predictions``, their ids.

    Returns:
        ``made`` and ``left`` (the ids done, and those not got to) and ``failed`` ({id: why}, of those done).
    """
    ids = [int(i) for i in args.get("predictions") or []][:MOST]
    until = time.time() + float(args.get("seconds", BUDGET_S))
    made, failed = [], {}
    while ids and (not made or time.time() < until):
        p = predictions.get(db, ids.pop(0))
        if p is None or p["status"] != "done":
            continue
        try:
            doc = make_one(p)
        except Exception as e:  # noqa: BLE001 (said in its file, and to whoever asked)
            why = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            doc = {"format": FORMAT, "prediction": p["id"], "fly": p["fly"], "made": predictions.now(), "of": _of(p), "error": why}
            failed[str(p["id"])] = why
        _write(p, doc)
        made.append(p["id"])
    return {"made": made, "failed": failed, "left": ids}


nmf_check.CALLS[CALL] = make


# ----------------------------------------------------- in the web server

def ask(db, ids) -> dict:
    """Asks the fit worker to match predictions' ROIs to their flies' own (`make`), and waits for those it gets to.

    Raises:
        ValueError: for ids that are not a list of at most `MOST` whole numbers.
        nmf_check.CallError: the fit worker is not running, or did not answer.
    """
    if not isinstance(ids, list) or not ids or len(ids) > MOST or any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        raise ValueError(f"predictions: a list of 1 to {MOST} predictions' ids")
    return nmf_check.call(db, CALL, {"predictions": ids})


def _scored(db) -> list[dict]:
    """The predictions done of flies with ROIs of their own (scored against them when they were made)."""
    return [p for p in predictions.all_predictions(db) if p["status"] == "done" and (p.get("result") or {}).get("scores") is not None]


def _sets(fit: Optional[dict]) -> tuple[set, set]:
    fl = (fit or {}).get("flies") or {}
    return set(fl.get("train") or []), set(fl.get("held_out") or [])


def choices(db) -> list[dict]:
    """The fits with predictions to sum over (newest first), each with its ``id``, who asked for it (``by``; None for
    one deleted) and ``experimenters``: of each (`experimenter`, the one with most such flies first), how many flies
    the model ``predicted``, how many of them have ROIs of their own (``scored``) and of those, how many it was not
    trained on (``new``)."""
    all_fits = {f["id"]: f for f in fits.all_fits(db)}
    out: dict[int, dict] = {}
    for p in predictions.all_predictions(db):
        if p["status"] != "done":
            continue
        e = out.setdefault(p["fit_id"], {}).setdefault(experimenter(p), {"predicted": 0, "scored": 0, "new": 0})
        e["predicted"] += 1
        if (p.get("result") or {}).get("scores") is not None:
            e["scored"] += 1
            e["new"] += p["fly"] not in _sets(all_fits.get(p["fit_id"]))[0]
    return [{"id": i, "by": (all_fits.get(i) or {}).get("name"), "deleted": i not in all_fits,
             "experimenters": sorted(({"name": n, **e} for n, e in who.items() if e["scored"]), key=lambda e: (-e["scored"], e["name"]))}
            for i, who in sorted(out.items(), reverse=True) if any(e["scored"] for e in who.values())]


def _labels_kind(p: dict, kind: str) -> Optional[str]:
    """The kind of a prediction's ROIs whose labels those of ``kind`` have: its own, but for cleaned ones, which keep
    the labels of those they were cleaned from."""
    if not kind.startswith("clean"):
        return kind
    d = predictions.pred_dir(p["id"])
    return next((k for k in ("split", "pred") if (d / f"{p['fly']}_{k}.tif").is_file()), None)


def _site_names(who: str) -> dict[str, int]:
    """The glomeruli an experimenter's flies of the site's datasets have ROIs of (`roi_training.kept_summary`, as last
    looked at: by the top folder of each fly's data on tensor), and in how many of them."""
    out: dict[str, int] = {}
    for f in (roi_training.kept_summary() or {}).get("flies") or []:
        if (f.get("dir") or "").split("/")[0] == who:
            for g in f.get("glomeruli") or []:
                out[g] = out.get(g, 0) + 1
    return out


def summary(db, fit_id: Optional[int] = None, who: Optional[str] = None, flies: str = "new", rois: str = "final") -> dict:
    """How a fit's model did on each glomerulus, over an experimenter's flies it predicted that have ROIs of their own.

    Args:
        fit_id, who: the fit and the experimenter (`experimenter`); default the newest fit with such predictions, and
            the experimenter with most of them.
        flies: of `FLIES`: only the flies the model was not trained on ("new": those held out of it, and those not
            among its flies at all), or all it predicted.
        rois: of `ROIS`: each prediction's ROIs to use ("final": refined, where they were), or the model's own.

    Returns:
        dict with ``choices`` (`choices`), ``fit``, ``experimenter``, ``flies`` and ``rois`` (as taken; the first two
        None if there is nothing to choose), ``worker`` (whether the fit worker is running) and, with something chosen:

        - ``counts``: ``scored`` (the experimenter's flies it predicted that have ROIs of their own), ``trained``
          (those of them it was trained on), ``held_out``, ``chosen`` (those summed over, by ``flies``), ``ready``
          (of them, those whose ROIs are matched to the fly's own: the rest are ``missing``, their predictions' ids,
          for the fit worker to match: `ask`), ``changed`` (those whose own ROIs are other than when they were
          predicted) and ``kinds`` ({kind of ROIs: how many flies' are of it});
        - ``failed``: those that cannot be matched (``id``, ``fly``, ``why``);
        - ``detect_iou`` and ``name_iou`` (al-diag-seg's; None until a fly is matched);
        - ``glomeruli``: per name the experimenter gave an ROI in those flies (and, with ``drawn`` 0, in the flies of
          the site's datasets, as last looked at: ``site``, in how many), by name: ``drawn`` (the flies it is drawn
          in), ``found`` (those with a predicted ROI on it at IoU ``detect_iou`` or over) and their mean ``dice``;
          of the flies it is drawn in, by the predicted ROI overlapping it most: ``named`` (those where that one
          overlaps it over ``name_iou`` and is named as it), ``other`` ({name: flies}: named as another), ``unnamed``
          (left unnamed) and ``none`` (no ROI overlaps it so: of which ``untouched``, by none at all), and of those
          not named right, ``moved``: the flies where the model gave its name to another ROI; the ROIs the
          model named as it (``given``), by the fly's own ROI each overlaps most: ``given_right`` (that one has the
          name, and they overlap over ``name_iou``), ``given_weak`` (it has the name, but they overlap less),
          ``given_wrong`` ({name: flies}: over ``name_iou`` with another), and of the rest ``given_undrawn`` in
          flies where it is not drawn (it may be right there), ``given_stacked`` over or under the fly's own of that
          name, on planes where it is not drawn (`OVER`: it may be right too) and ``given_elsewhere`` (in a fly that
          has it drawn, and not there: misplaced); ``model``: what the fit's model has of it (``trained``: training
          flies with it, None for one of the reference atlas alone; ``atlas``: whether its atlas has it; None for a
          fit deleted); and ``flies``: each of those, with ``p`` (its prediction), ``state`` ("named", "other",
          "unnamed", "none"; or, of an ROI the model named as it that is not on the fly's own of that name, "weak",
          "stacked", "wrong", "elsewhere", "undrawn"), ``roi`` (the predicted ROI's label), ``as`` (its name),
          ``iou``, ``on`` (of an ROI the model named as it, the fly's own ROI it overlaps most) and ``has`` (of a
          fly it is drawn in and not named right: the label of the ROI the model gave its name, if it gave it);
        - ``only_model``: the same of the names the model gave that the experimenter never did;
        - ``total``: the counts above summed over ``glomeruli``;
        - ``predictions``: {id: ``fly``, ``kind`` (its ROIs', as summed), ``set`` ("held_out", "trained" or "new"),
          ``changed`` and ``explorer`` (whether the naming explorer shows the ROIs those labels are of)}.

    Raises:
        ValueError: for ``flies`` or ``rois`` not one of theirs.
    """
    if flies not in FLIES or rois not in ROIS:
        raise ValueError(f"flies: one of {', '.join(FLIES)}; rois: one of {', '.join(ROIS)}")
    ch = choices(db)
    out = {"choices": ch, "fit": None, "experimenter": None, "flies": flies, "rois": rois,
           "worker": bool(fits.worker_status(db).get("alive"))}
    c = next((x for x in ch if x["id"] == fit_id), None) if fit_id is not None else (ch[0] if ch else None)
    if c is None:
        return out
    who = who if any(e["name"] == who for e in c["experimenters"]) else c["experimenters"][0]["name"]
    out.update(fit=c["id"], experimenter=who)
    train, held = _sets(fits.get(db, c["id"]))
    mine = [p for p in _scored(db) if p["fit_id"] == c["id"] and experimenter(p) == who]
    chosen = [p for p in mine if flies == "all" or p["fly"] not in train]
    counts = {"scored": len(mine), "trained": sum(p["fly"] in train for p in mine), "held_out": sum(p["fly"] in held for p in mine),
              "chosen": len(chosen), "ready": 0, "missing": [], "changed": 0, "kinds": {}}
    info = None if c["deleted"] else naming._model_info(c["id"])
    rows: dict[str, dict] = {}
    failed, about = [], {}
    detect = name_iou = None

    def row(name: str) -> dict:
        return rows.setdefault(name, {"name": name, **dict.fromkeys(COUNTS, 0), "dice": [], "other": {}, "given_wrong": {}, "site": 0, "flies": []})

    for p in sorted(chosen, key=lambda p: p["fly"]):
        doc = read(p)
        if doc is None:
            counts["missing"].append(p["id"])
            continue
        if doc.get("error"):
            failed.append({"id": p["id"], "fly": p["fly"], "why": doc["error"]})
            continue
        final = (p.get("result") or {}).get("final")
        kind = final if rois == "final" and final in doc["kinds"] else "pred"
        if kind not in doc["kinds"]:
            failed.append({"id": p["id"], "fly": p["fly"], "why": "its folder has not the model's own ROIs"})
            continue
        counts["ready"] += 1
        counts["changed"] += bool(doc.get("changed"))
        counts["kinds"][kind] = counts["kinds"].get(kind, 0) + 1
        detect, name_iou = doc["detect_iou"], doc["name_iou"]
        about[p["id"]] = {"fly": p["fly"], "kind": kind, "set": "held_out" if p["fly"] in held else "trained" if p["fly"] in train else "new",
                          "changed": bool(doc.get("changed")), "explorer": _labels_kind(p, kind) == naming.named_kind(p)}
        k = doc["kinds"][kind]
        drawn = {m["name"] for m in k["manual"]}
        has = {q["name"]: q["label"] for q in k["predicted"] if q["name"]}  # (the ROI the model gave each name)
        for m in k["manual"]:
            r, on = row(m["name"]), m["iou"] > name_iou
            r["drawn"] += 1
            if m["iou"] >= detect:
                r["found"] += 1
                r["dice"].append(2 * m["iou"] / (1 + m["iou"]))
            state = "none" if not on else "named" if m["as"] == m["name"] else "unnamed" if m["as"] is None else "other"
            if state == "other":
                r["other"][m["as"]] = r["other"].get(m["as"], 0) + 1
            else:
                r[state] += 1
            r["untouched"] += m["roi"] is None
            other = None if state == "named" else has.get(m["name"])
            r["moved"] += other is not None and other != m["roi"]
            r["flies"].append({"p": p["id"], "state": state, "iou": m["iou"], "roi": m["roi"], "as": m["as"], "has": other})
        for q in k["predicted"]:
            if not q["name"]:
                continue
            r, on = row(q["name"]), q["iou"] > name_iou
            r["given"] += 1
            if on and q["manual"] == q["name"]:
                r["given_right"] += 1
                continue
            state = ("wrong" if on else "weak" if q["manual"] == q["name"] else "undrawn" if q["name"] not in drawn
                     else "stacked" if (q.get("over") or 0) >= OVER else "elsewhere")
            if state == "wrong":
                r["given_wrong"][q["manual"]] = r["given_wrong"].get(q["manual"], 0) + 1
            else:
                r[f"given_{state}"] += 1
            r["flies"].append({"p": p["id"], "state": state, "iou": q["iou"], "roi": q["label"], "as": q["name"], "on": q["manual"]})
    for g, n in _site_names(who).items():
        row(g)["site"] = n
    for r in rows.values():
        r["dice"] = sum(r["dice"]) / len(r["dice"]) if r["dice"] else None
        r["model"] = None if info is None else {"trained": info["trained"].get(r["name"]),
                                                "atlas": r["name"] in info["flies"] if info["has_atlas"] else None}
    ordered = sorted(rows.values(), key=lambda r: _natural(r["name"]))
    theirs = [r for r in ordered if r["drawn"] or r["site"]]
    total = {k: sum(r[k] for r in theirs) for k in COUNTS}
    total.update(other=sum(sum(r["other"].values()) for r in theirs), given_wrong=sum(sum(r["given_wrong"].values()) for r in theirs))
    out.update(counts=counts, failed=failed, detect_iou=detect, name_iou=name_iou, glomeruli=theirs,
               only_model=[r for r in ordered if not (r["drawn"] or r["site"])], total=total, predictions=about,
               model=None if info is None else {"has_atlas": info["has_atlas"], "max_cost": info["max_cost"]})
    return out
