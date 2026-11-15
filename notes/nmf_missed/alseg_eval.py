"""al-diag-seg's alseg.nmf.find on the cached flies (cache.py), each candidate against the manual ROIs as score.py counts them; of one
that is more of an ROI, whether the manual outlines the two lie on are of one glomerulus.

Usage (in the olfdb-worker image (docker run --gpus for the GPU), with this folder at /code, fit 7's folder at /fit and tensor at /local/tensor (all
read-only), a scratch folder at /cache (cache.py's: 0.6 to 1.4 GB a fly) and one at /out; design.md, "Glomeruli a prediction's ROIs miss"; with the al-diag-seg to check on PYTHONPATH, if not the image's):
alseg_eval.py [<flies, comma separated, or -> [frames: also checks evidence.frames against the cache, on the first fly] [<find's parameters, as JSON>]"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

from alseg import evidence, nmf

CACHE, FIT = Path("/cache"), Path("/fit")
FLIES = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else "Sam_2024-06-11_2,Tom_2023-05-08_1,Tom_2023-11-21_3,Tom_2024-09-12_2").split(",")
kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


tot = {"n": 0, "hit": 0, "hit5": 0, "frag": 0, "off": 0, "ext": 0, "split": 0, "split_merge": 0}
for i, fly in enumerate(FLIES):
    f = pickle.load(open(CACHE / fly / "fly.pkl", "rb"))
    meta = json.loads((CACHE / fly / "meta.json").read_text())
    pred = tifffile.imread(FIT / "results" / f"{fly}_pred.tif")
    key = np.array(meta["frame_rec"]) * 10000 + np.array(meta["frame_trial"])
    uniq, trial = np.unique(key, return_inverse=True)
    trials = [(meta["recordings"][int(k // 10000)]["name"], str(meta["recordings"][int(k // 10000)]["trials"][int(k % 10000)]["odors"])) for k in uniq]
    movies = {"dff": np.load(CACHE / fly / "dff.npy", mmap_mode="r"), "trial": trial, "on": np.array(meta["frame_on"]) == 1, "trials": trials,
              "recordings": [r["name"] for r in meta["recordings"]]}
    if i == 0 and len(sys.argv) > 2 and sys.argv[2] == "frames":
        t0 = time.time()
        mv = evidence.frames(f["dir"], pred.shape)
        print(f"evidence.frames: {mv['dff'].shape} in {time.time() - t0:.0f} s; as the cache: {mv['dff'].shape == movies['dff'].shape and bool(np.array_equal(mv['dff'], movies['dff']))}, "
              f"trial {np.array_equal(mv['trial'], trial)}, on {np.array_equal(mv['on'], movies['on'])}, trials {len(mv['trials'])} vs {len(trials)}", flush=True)
    t0 = time.time()
    res = nmf.find(pred, f, movies, **kw)
    man, lut = f["labels"], f["lut"]
    out = []
    for c in res["candidates"]:
        own = (res["candidate"] == c["id"]) & (res["labels"] == c["label"])
        for z in c["planes"]:
            m = own[z]
            mm = {l: man[z] == l for l in np.unique(man[z]) if l}
            pm = [pred[z] == l for l in np.unique(pred[z]) if l]
            ious = {l: iou(m, q) for l, q in mm.items()}
            bl = max(ious, key=ious.get) if ious else None
            m_iou = ious[bl] if bl is not None else 0.0
            m_pred = max([iou(mm[bl], q) for q in pm] or [0.0]) if bl is not None else 0.0
            cls = "hit" if m_iou >= 0.3 and m_pred < 0.5 else "frag" if (m & (man[z] > 0)).sum() >= 0.5 * m.sum() else "off"
            if c["kind"] == "extended":
                # the manual glomerulus the ROI it is more of is on, on the planes next to this one
                mans = []
                for dz in (-1, 1):
                    if 0 <= z + dz < len(pred):
                        q = pred[z + dz] == c["of"]
                        if q.any():
                            ls, ns = np.unique(man[z + dz][q], return_counts=True)
                            mans += [(int(n), lut.get(int(l))) for l, n in zip(ls, ns) if l and n >= 0.3 * q.sum()]
                of_man = max(mans)[1] if mans else None
                cls += f" [the ROI it is more of is on manual {of_man}: {'same' if bl is not None and of_man == lut.get(int(bl)) and m_iou >= 0.3 else 'not the same' if of_man and m_iou >= 0.3 else '?'}]"
                tot.setdefault("ext_same", 0); tot.setdefault("ext_other", 0)
                tot["ext_same"] += bool(bl is not None and of_man == lut.get(int(bl)) and m_iou >= 0.3)
                tot["ext_other"] += bool(of_man and m_iou >= 0.3 and of_man != lut.get(int(bl)))
            if c["kind"] == "split":
                held = sum((q & (pred[z] == c["of"])).sum() >= 0.4 * q.sum() for q in mm.values())
                tot["split"] += 1
                tot["split_merge"] += held >= 2
                cls = f"split of a mask over {held}"
            else:
                tot["n"] += 1
                tot[cls.split(" [")[0]] += 1
                tot["hit5"] += cls.startswith("hit") and m_iou >= 0.5
                tot["ext"] += c["kind"] == "extended"
            out.append(f"    {c['id']:2d} {c['kind']:8s} z{z} label {c['label']} of {c['of']} {m.sum():4d} px eta2 {c.get('eta2', float('nan')):.2f} "
                       f"novelty {c.get('novelty', float('nan')):.2f} corr {c.get('corr', float('nan')):.2f} | {cls} (manual {lut.get(int(bl)) if bl is not None else None}, IoU {m_iou:.2f})")
    print(f"{fly}: {len(res['candidates'])} candidates on planes {res['planes']} in {time.time() - t0:.0f} s", flush=True)
    print("\n".join(out), flush=True)
print(tot)
