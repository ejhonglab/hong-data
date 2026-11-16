"""Fit 7's model on its held-out flies (prep.py's cache) with some or all of the diagnostic odors' responses left out of what it is
given, scored against the flies' own ROIs as a prediction is (alseg's Model.evaluate): what its three uses of them are worth.

The model uses the responses (a) for two of Cellpose's three input images (the local correlation of tuning, and the first shared
principal component; the third is the average image), (b) to join the planes' masks into 3D ROIs (overlap and tuning), and (c) for
the tuning term of an ROI's name. A variant says what each gets: all 20 odors, k of them (the same k: the correlation image and the
joining from those; the principal component with the others at the training flies' mean; the tuning term over those), the fly's
other recordings' odors (no component image, no tuning term: the model has no tuning of those), or nothing (that image blank; planes
joined as Cellpose joins them, by overlap; no tuning term).
variants.py   (on a GPU; results in /out/variants.json)"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from alseg import identity
from alseg import model as am
from alseg import responses as R
from alseg import segment

FIT, CACHE, OUT = Path("/fit"), Path("/cache"), Path("/out")
FLIES = ["Sam_2024-06-11_2", "Tom_2023-05-08_1", "Tom_2023-11-21_3", "Tom_2024-09-12_2"]
model = am.Model.load(FIT / "model")
o = model.options
GPU = am.use_gpu(None)

from cellpose import models  # noqa: E402

_made = {}
_Cellpose = models.CellposeModel
models.CellposeModel = lambda gpu, pretrained_model: _made.setdefault((gpu, pretrained_model), _Cellpose(gpu=gpu, pretrained_model=pretrained_model))


def sub(k, seed):
    return tuple(sorted(np.random.default_rng(seed).choice(len(model.odors), k, replace=False).tolist()))


VARIANTS = [("all 20 odors", "all", "all", "all")]
VARIANTS += [(f"{k} of them (draw {s + 1})", ("sub", sub(k, s)), ("sub", sub(k, s)), ("sub", sub(k, s))) for k in (10, 5, 2) for s in range(3)]
VARIANTS += [("all for the ROIs, none for their names", "all", "all", None),
             ("none for the ROIs, all for their names", None, None, "all"),
             ("none at all", None, None, None),
             ("the fly's other odors for the ROIs (image and joining), no tuning in names", "other", "other", None),
             ("the fly's other odors for joining only, no tuning in names", None, "other", None)]


def pick(fly, what):
    return fly["responses"] if what == "all" else fly["responses"][list(what[1])] if isinstance(what, tuple) else fly["other"] if what == "other" else None


def predict(fly, seg, join, name):
    sp, blank = fly["spacing"], np.zeros(fly["average"].shape, np.float32)
    r = pick(fly, seg)
    lc = R.normalize01(R.local_correlation(r)) if r is not None else blank
    pc = blank
    if seg == "all":
        pc = R.normalize01(model.pca.transform(r)[0])
    elif isinstance(seg, tuple):  # (the odors left out at the training flies' mean: nothing of them in the component)
        v = np.tile(model.pca.mean_, (r[0].size, 1))
        v[:, list(seg[1])] = R._zscore_vectors(r)
        pc = R.normalize01(((v - model.pca.mean_) @ model.pca.components_[0]).reshape(r.shape[1:]))
    pred = segment.predict(model.segmenter, np.stack([R.normalize01(fly["average"]), lc, pc]), gpu=GPU, spacing=sp, xy_um=model.xy_um,
                           cellprob_threshold=o.cellprob_threshold, flow_threshold=o.flow_threshold, dilate=o.dilate, responses=pick(fly, join),
                           join_overlap=o.join_overlap, join_corr=o.join_corr)
    tmpl, nr, w = model.template, fly["responses"], 1.0 if name else 0.0
    if isinstance(name, tuple):
        nr = fly["responses"][list(name[1])]
        tmpl = {n: {**g, "tuning": g["tuning"][list(name[1])]} for n, g in model.template.items()}
    ids, terms = identity.assign_identities(nr, pred, tmpl, sp, fly["prior"].astype(np.float32), model.atlas.names, max_cost=o.max_cost,
                                            return_terms=True, w_tuning=w)
    return {"labels": {"pred": pred}, "identities": {"pred": ids}, "identity_terms": {"pred": terms}, "atlas_maxprob": None, "prior": None,
            "final": "pred"}


rows = []
for fly_name in FLIES:
    fly = pickle.load(open(CACHE / f"{fly_name}.pkl", "rb"))
    for label, seg, join, name in VARIANTS:
        if "other" in (seg, join) and fly["other"] is None:
            continue
        t0 = time.time()
        m = am.Model(**{k: getattr(model, k) for k in model.__dataclass_fields__})
        m.predict = lambda f, a=(seg, join, name): predict(f, *a)
        res = m.evaluate([fly])
        s = next(r for r in res["summary"] if r["rois"] == "pred")
        man = next(r for r in res["summary"] if r["rois"] == "manual")
        ident = res["identity"][0]
        rows.append({"fly": fly_name, "variant": label, "n": s["n"], "n_2d": s["n_2d"], "manual_2d": man["n_2d"], "manual": man["n"],
                     "found_2d": s["recall_2d"], "right_2d": s["precision_2d"], "found": s["recall"], "right": s["precision"],
                     "dice_2d": s["dice_2d"], "named_right": ident["correct"], "named_of": ident["total"],
                     "other_odors": len(fly["other_odors"])})
        print(f"{fly_name:18s} {label:76s} ROIs {s['n']:3d} ({s['n_2d']:3d} in-plane) found {s['recall_2d']:.2f} right {s['precision_2d']:.2f} | 3D found "
              f"{s['recall']:.2f} right {s['precision']:.2f} | named {ident['correct']}/{ident['total']}  {time.time() - t0:.0f} s", flush=True)
(OUT / "variants.json").write_text(json.dumps(rows))
print("\n| what the model is given | ROIs a fly | found, per plane | right, per plane | found, 3D | right, 3D | named right |")
print("|---|---|---|---|---|---|---|")
groups = {}
for r in rows:
    groups.setdefault(r["variant"].split(" (draw")[0], []).append(r)
for label, rs in groups.items():
    mean = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
    c, t = sum(r["named_right"] for r in rs), sum(r["named_of"] for r in rs)
    print(f"| {label} | {mean('n'):.0f} | {100 * mean('found_2d'):.0f}% | {100 * mean('right_2d'):.0f}% | {100 * mean('found'):.0f}% | {100 * mean('right'):.0f}% | "
          f"{c}/{t} ({100 * c / max(t, 1):.0f}%) |")
