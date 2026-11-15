"""Caches fit 7's held-out flies for the scripts here: each one's movies as dF per trial, <cache>/<fly>/dff.npy ((T, Z, Y, X) float16:
each time point of a trial less the mean of its time points before the odor), f0.npy (the mean of the trials' baselines), meta.json
(per time point its recording, trial and whether the odor is on; the trials) and fly.pkl (alseg's load_fly: average, responses,
spacing, its manual ROIs as labels and their names).

Usage (in the olfdb-worker image (docker run --gpus for the GPU), with this folder at /code, fit 7's folder at /fit and tensor at /local/tensor (all
read-only), a scratch folder at /cache (cache.py's: 0.6 to 1.4 GB a fly) and one at /out; design.md, "Glomeruli a prediction's ROIs miss"; with the database's tree on PYTHONPATH):
cache.py [<flies, comma separated>]"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from alseg import model as am
from alseg.fly import load_fly
from app import extract, movies

FIT, CACHE = Path("/fit"), Path("/cache")
man = json.loads((FIT / "manifest.json").read_text())
entries = {f"{e['dir'].split('/')[0]}_{e['date']}_{e['fly_num']}": e for e in man["flies"]}
FLIES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Sam_2024-06-11_2", "Tom_2023-05-08_1", "Tom_2023-11-21_3", "Tom_2024-09-12_2"]
model = am.Model.load(FIT / "model")
for fly in FLIES:
    t0 = time.time()
    e = entries[fly]
    d = Path("/local/tensor") / e["dir"]
    out = CACHE / fly
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "fly.pkl").is_file():
        f = load_fly(d, model, raw_dir=e["raw_dir"], name=fly, glomeruli=man.get("glomeruli"))
        f.pop("hires", None)
        f["dir"] = str(f["dir"])
        pickle.dump(f, open(out / "fly.pkl", "wb"))
    else:
        f = pickle.load(open(out / "fly.pkl", "rb"))
    shape = f["average"].shape
    if (out / "dff.npy").is_file():
        continue
    frames, meta, f0, nb = [], {"recordings": [], "frame_rec": [], "frame_trial": [], "frame_on": []}, np.zeros(shape, np.float64), 0
    for rd in sorted(q for q in d.iterdir() if q.is_dir()):
        mv = next((rd / n for n in movies.PREFERRED if (rd / n).is_file()), None)
        if mv is None or not (rd / movies.TRIALS).is_file():
            continue
        m = movies.open_movie(mv)
        s = m.shape
        if (s["z"], s["y"], s["x"]) != tuple(shape):
            print(f"  {rd.name}: on another grid {s}", flush=True)
            continue
        trials = extract.read_trials(rd / movies.TRIALS)
        ri = len(meta["recordings"])
        meta["recordings"].append({"name": rd.name, "movie": mv.name, "t": s["t"], "trials": trials,
                                   "diag": rd.name in {r["name"] for r in e["recordings"]}})
        for k, t in enumerate(trials):
            nbase = t["onset"] - t["start"]
            if nbase <= 0:
                continue
            v = m.volumes(t["start"], t["end"] + 1 - t["start"]).astype(np.float32)
            b = v[:nbase].mean(0)
            f0 += b
            nb += 1
            frames.append((v - b).astype(np.float16))
            meta["frame_rec"] += [ri] * len(v)
            meta["frame_trial"] += [k] * len(v)
            meta["frame_on"] += [0] * nbase + [1] * (len(v) - nbase)
        print(f"  {rd.name}: {len(trials)} trials, {s['t']} volumes, {time.time() - t0:.0f} s", flush=True)
    np.save(out / "dff.npy", np.concatenate(frames))
    np.save(out / "f0.npy", (f0 / nb).astype(np.float32))
    (out / "meta.json").write_text(json.dumps(meta, default=str))
    print(f"{fly}: {sum(len(x) for x in frames)} frames, {time.time() - t0:.0f} s", flush=True)
