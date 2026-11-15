"""alseg.clean_trim.split on the predicted masks, by the per-trial dF/F maps (its evidence) and by the movies' frames (dF over noise):
which masks over two manual ROIs it splits (and whether each part is on its own), and how many others.

Usage (in the olfdb-worker image (docker run --gpus for the GPU), with this folder at /code, fit 7's folder at /fit and tensor at /local/tensor (all
read-only), a scratch folder at /cache (cache.py's: 0.6 to 1.4 GB a fly) and one at /out; design.md, "Glomeruli a prediction's ROIs miss"; needs the cache of cache.py, and about 20 GB of memory):
split_eval.py"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import tifffile

from alseg import clean_trim, evidence

CACHE, FIT = Path("/cache"), Path("/fit")
man_ = json.loads((FIT / "manifest.json").read_text())
entries = {f"{e['dir'].split('/')[0]}_{e['date']}_{e['fly_num']}": e for e in man_["flies"]}
FLIES = ["Sam_2024-06-11_2", "Tom_2023-05-08_1", "Tom_2023-11-21_3", "Tom_2024-09-12_2"]
kw = [dict(), dict(max_corr=0.3), dict(max_corr=0.7)]
tot = {}
for fly in FLIES:
    f = pickle.load(open(CACHE / fly / "fly.pkl", "rb"))
    pred = tifffile.imread(FIT / "results" / f"{fly}_pred.tif").astype(np.int32)
    man, sp = f["labels"], f["spacing"]
    ev = {"trials": evidence.load(Path("/local/tensor") / entries[fly]["dir"], pred.shape)}
    D = np.load(CACHE / fly / "dff.npy").astype(np.float32)
    d = np.abs(np.diff(D, axis=0))
    sd = 1.4826 * np.median(d, axis=0) / np.sqrt(2)
    del d
    sd = np.maximum(sd, np.quantile(sd, 0.2))
    ev["frames"] = D / sd
    on = np.array(json.loads((CACHE / fly / "meta.json").read_text())["frame_on"]) == 1
    ev["frames on"] = ev["frames"][on]
    for name, E in ev.items():
        for k in kw:
            out, rep = clean_trim.split(pred, {"evidence": E, "spacing": sp, "responses": f["responses"]}, return_report=True, **k)
            key = (name, json.dumps(k))
            t = tot.setdefault(key, dict(merges=0, split_merges=0, right=0, split_one=0, split_none=0, n=0))
            for z in range(pred.shape[0]):
                mm = {l: man[z] == l for l in np.unique(man[z]) if l}
                for l in np.unique(pred[z]):
                    if not l:
                        continue
                    q = pred[z] == l
                    held = [a for a, m in mm.items() if (m & q).sum() >= 0.4 * m.sum()]
                    t["n"] += 1
                    t["merges"] += len(held) >= 2
                    if (int(l), z) in rep["split"]:
                        new = out[z] == rep["split"][int(l), z]
                        rest = q & ~new
                        main = lambda p: max(mm, key=lambda a: (mm[a] & p).sum()) if mm and max((m & p).sum() for m in mm.values()) >= 0.5 * p.sum() else None  # noqa: E731
                        a, b = main(new), main(rest)
                        if len(held) >= 2:
                            t["split_merges"] += 1
                            t["right"] += a is not None and b is not None and a != b
                        elif len(held) == 1:
                            t["split_one"] += 1
                        else:
                            t["split_none"] += 1
    print(fly, "done", flush=True)
for (name, k), t in tot.items():
    print(f"{name:10s} {k:20s}: masks {t['n']}, over two {t['merges']}: split {t['split_merges']} (each part on its own {t['right']}); split of those over one {t['split_one']}, over none {t['split_none']}")
