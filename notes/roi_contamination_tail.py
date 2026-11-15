"""The tail of how much of a neighbour reaches whole ROIs' means (refine.purity_report's whole_share, per in-plane mask),
of fit 7's held-out flies: predicted, manual, and cleaned (al-diag-seg's trim and margin, at their defaults). Read-only.

Usage (in the olfdb-worker image, al-diag-seg on PYTHONPATH; fit 7's folder at /fit and tensor at /local/tensor,
read-only): roi_contamination_tail.py"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from alseg import clean, fly as flymod, refine

FIT = Path("/fit")
mj = json.loads((FIT / "model/model.json").read_text()); man = json.loads((FIT / "manifest.json").read_text())
m = SimpleNamespace(odors=mj["odors"], aliases=mj["aliases"], atlas=None, info=mj.get("info", {}), glomeruli=man["glomeruli"])
ents = {f"{e['dir'].split('/')[0]}_{e['date']}_{e['fly_num']}": e for e in man["flies"]}
acc = {}
for name in sorted({p.name[:-9] for p in (FIT / "results").glob("*_pred.tif")}):
    e = ents[name]
    f = flymod.load_fly(Path("/local/tensor") / e["dir"], m, raw_dir=e["raw_dir"], name=name, glomeruli=man["glomeruli"])
    P = tifffile.imread(FIT / "results" / f"{name}_pred.tif").astype(np.int32)
    for k, L, ref in (("pred", P, None), ("manual", f["labels"], None), ("trim", clean.apply(P, f, ["trim"]), P),
                      ("margin", clean.apply(P, f, ["margin"]), P)):
        # each mask's whole share, and whether it holds >= 40% of two manual ROIs (a merge)
        r = refine.purity_report(f["responses"], L, f["spacing"], ref=ref)
        acc.setdefault(k, []).extend(r["whole_share"])
for k, v in acc.items():
    v = np.array(v)
    print(f"{k:7s} n {len(v):4d}  whole share: median {np.median(v):.3f}, 75% {np.quantile(v, .75):.3f}, 90% {np.quantile(v, .9):.3f}, "
          f"95% {np.quantile(v, .95):.3f}, max {v.max():.3f}; over 0.1: {100 * (v > 0.1).mean():.1f}%, over 0.2: {100 * (v > 0.2).mean():.1f}%, "
          f"over 0.3: {100 * (v > 0.3).mean():.1f}%")
