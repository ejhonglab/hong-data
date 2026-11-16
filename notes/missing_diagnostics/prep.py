"""Fit 7's held-out flies as its model reads them (alseg's load_fly, their anatomical stacks too), the model's atlas on each one's
planes (as a prediction registers it), and each one's responses to the odors of its other recordings (their trialmean_dff.tif, one
after another, with the odors as recorded): <cache>/<fly>.pkl. Needs the database's tree on PYTHONPATH (the odors' names)."""
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

from alseg import atlas as atlas_mod
from alseg import model as am
from alseg.fly import TRIALMEAN, TRIALS, load_fly, recorded_odors
from app import odormeta, roi_training

FIT, CACHE = Path("/fit"), Path("/cache")
man = json.loads((FIT / "manifest.json").read_text())
entries = {f"{e['dir'].split('/')[0]}_{e['date']}_{e['fly_num']}": e for e in man["flies"]}
FLIES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Sam_2024-06-11_2", "Tom_2023-05-08_1", "Tom_2023-11-21_3", "Tom_2024-09-12_2"]
con = sqlite3.connect("file:/db/olfdb.sqlite3?mode=ro&immutable=1", uri=True)
con.row_factory = sqlite3.Row
index = odormeta.OdorIndex([dict(r) for r in con.execute("select * from odors")])
model = am.Model.load(FIT / "model")
for fly in FLIES:
    if (CACHE / f"{fly}.pkl").is_file():
        continue
    t0 = time.time()
    e = entries[fly]
    d = Path("/local/tensor") / e["dir"]
    for r in sorted(x for x in d.iterdir() if x.is_dir()):  # (the fly's odors as recorded, as the odor table names them: as the worker does)
        if (r / TRIALS).is_file():
            for o in roi_training.recording_odors(r / TRIALS, index) or []:
                model.aliases.setdefault(o["recorded"], o["odor"])
    f = load_fly(d, model, raw_dir=e["raw_dir"], name=fly, glomeruli=man.get("glomeruli"))
    if f.get("hires") and f["hires"].get("mirror") is None:
        atlas_mod.orient_hires(f, model.options.hires_reg)
    prior = atlas_mod.atlas_to_fly(model.atlas, f, transform=model.options.reg, hires_transform=model.options.hires_reg)
    diag = {r["name"] for r in e["recordings"]}
    other, odors = [], []
    for r in sorted(x for x in d.iterdir() if x.is_dir()):
        if r.name in diag or not ((r / TRIALMEAN).is_file() and (r / TRIALS).is_file()):
            continue
        x = tifffile.imread(r / TRIALMEAN).astype(np.float32)
        x = x[None] if x.ndim == 3 else x
        names = [o["odor"] for o in roi_training.recording_odors(r / TRIALS, index)]
        if x.shape[1:] != f["average"].shape or len(names) != len(x):
            print(f"  {fly} {r.name}: left out ({x.shape}, {len(names)} odors)", flush=True)
            continue
        x[~np.isfinite(x)] = 0
        other.append(x)
        odors += [(r.name, n) for n in names]
    had = bool(f.pop("hires", None))
    f["dir"] = str(f["dir"])
    f.update(prior=prior.astype(np.float16), other=np.concatenate(other) if other else None, other_odors=odors, stack=had)
    pickle.dump(f, open(CACHE / f"{fly}.pkl", "wb"))
    print(f"{fly}: stack {had}, {len(odors)} other odors of {sorted({r for r, _ in odors})}, {time.time() - t0:.0f} s", flush=True)
