"""Every plane of a fly with its manual ROIs (yellow), the predicted ones (magenta) and the try-out's candidates (cyan; a mask cut in
two: orange, the rest dashed), to look at what the candidates are: <out>/<tag>/png/grid_<fly>.png.

Usage (in the olfdb-worker image (docker run --gpus for the GPU), with this folder at /code, fit 7's folder at /fit and tensor at /local/tensor (all
read-only), a scratch folder at /cache (cache.py's: 0.6 to 1.4 GB a fly) and one at /out; design.md, "Glomeruli a prediction's ROIs miss"; after run.py <tag>):
grid.py <tag> <fly> "<a Python expression on a candidate c, e.g. c['kind'] == 'add' and c['eta2'] >= 0.5>" [rms: on the rms of dF, not the average image]"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

CACHE, FIT, OUT = Path("/cache"), Path("/fit"), Path("/out")
tag, fly, expr = sys.argv[1], sys.argv[2], sys.argv[3]
rms = len(sys.argv) > 4
f = pickle.load(open(CACHE / fly / "fly.pkl", "rb"))
pred = tifffile.imread(FIT / "results" / f"{fly}_pred.tif")
cands = pickle.load(open(OUT / tag / f"{fly}_cands.pkl", "rb"))
dff = np.load(CACHE / fly / "dff.npy", mmap_mode="r") if rms else None
zs = [z for z in range(pred.shape[0]) if (pred[z] > 0).any()]
yy, xx = np.nonzero((pred > 0).any(0) | (f["labels"] > 0).any(0))
y0, y1, x0, x1 = max(0, yy.min() - 12), yy.max() + 12, max(0, xx.min() - 12), xx.max() + 12
cols = (len(zs) + 1) // 2
fig, axes = plt.subplots(2, cols, figsize=(5.2 * cols, 10.4 * (y1 - y0) / (x1 - x0) + 0.5))
for ax in axes.ravel():
    ax.axis("off")
for z, ax in zip(zs, axes.ravel()):
    img = np.sqrt((np.clip(np.asarray(dff[:, z], np.float32), 0, None) ** 2).mean(0)) if rms else f["average"][z]
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
    for lab, col, lw in ((f["labels"][z], "yellow", 0.7), (pred[z], "magenta", 0.7)):
        for l in np.unique(lab):
            if l:
                ax.contour(lab == l, [0.5], colors=col, linewidths=lw)
    for c in cands:
        if c["z"] != z or not eval(expr, {}, {"c": c}):
            continue
        col = "cyan" if c["kind"] == "add" else "orange"
        ax.contour(c["eval_mask"], [0.5], colors=col, linewidths=1.4)
        if c["kind"] == "split":
            ax.contour(c["other_mask"], [0.5], colors="orange", linewidths=1.0, linestyles="dashed")
        y_, x_ = np.nonzero(c["eval_mask"])
        ax.text(x_.mean(), y_.mean(), f"{c.get('order', c['k'])}", color=col, fontsize=8, ha="center", va="center")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title(f"z{z}", fontsize=9)
fig.tight_layout(pad=0.3)
(OUT / tag / "png").mkdir(exist_ok=True)
fig.savefig(OUT / tag / "png" / f"grid_{fly}{'_rms' if rms else ''}.png", dpi=62)
