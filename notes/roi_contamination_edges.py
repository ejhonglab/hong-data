"""Where a predicted ROI's neighbour signal is: its edge against its core, for every predicted in-plane
mask of fit 7's held-out flies (not only those matching a manual ROI), and the same for the manual ROIs.
Also: masks over two manual ROIs. Read-only.

Usage (in the olfdb-worker image, which has al-diag-seg; a fit's folder at /fit and tensor at
/local/tensor, both read-only; see demixing_plan.md):
roi_contamination_edges.py [<threshold, dF/F: 0.3>] [other]
"other": on the flies' recordings that are not diagnostic ones (megamat, validation, ...), not the
diagnostic recordings' responses the model reads."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import nnls

from alseg import fly as flymod

FIT = Path("/fit")
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
CORE = 3  # px a core is from its ROI's edge

mj = json.loads((FIT / "model/model.json").read_text())
man = json.loads((FIT / "manifest.json").read_text())
model = SimpleNamespace(odors=mj["odors"], aliases=mj["aliases"], atlas=None, info=mj.get("info", {}),
                        glomeruli=mj.get("glomeruli", man["glomeruli"]))
entries = {f"{e['dir'].split('/')[0]}_{e['date']}_{e['fly_num']}": e for e in man["flies"]}
names = sorted({p.name[: -len("_pred.tif")] for p in (FIT / "results").glob("*_pred.tif")})


def other_panels(d, like):
    """The responses (odors, Z, Y, X) of a fly's recordings that are not diagnostic ones."""
    import os
    out = []
    for r in sorted(p for p in Path(d).iterdir() if p.is_dir()):
        if not ((r / "trialmean_dff.tif").is_file() and (r / "trial_frames_and_odors.json").is_file()):
            continue
        ns = [model.aliases.get(s, s) for s in flymod.recorded_odors(r / "trial_frames_and_odors.json")]
        if sum(n in set(model.odors) for n in ns) > len(ns) / 2:
            continue
        x = np.nan_to_num(tifffile.imread(r / "trialmean_dff.tif").astype(np.float32), nan=0, posinf=0, neginf=0)
        if x.shape[1:] == like.shape[1:]:
            out.append(x)
            print("  ", r.name, x.shape[0], "odors", flush=True)
    return np.concatenate(out) if out else None


def planes(L):
    for z in range(L.shape[0]):
        for l in np.unique(L[z]):
            if l:
                yield int(l), z, L[z] == l


def edge_report(L, R, reach=4):
    """Per in-plane mask with a core and a neighbour: its core's, edge's and whole responses, and its
    nearest neighbour's core's (odors,)."""
    cores = {}
    for l, z, m in planes(L):
        c = ndi.binary_erosion(m, iterations=CORE)
        if c.sum() >= 8:
            cores[l, z] = c
    out = []
    for (l, z), c in cores.items():
        m = L[z] == l
        halo = ndi.binary_dilation(m, iterations=reach) & ~m
        ls, n = np.unique(L[z][halo], return_counts=True)
        cand = [(k, int(x)) for x, k in zip(ls, n) if x and (int(x), z) in cores]
        if not cand:
            continue
        nb = max(cand)[1]
        nbm = L[z] == nb
        d = ndi.distance_transform_edt(~nbm)
        edge = m & ~ndi.binary_erosion(m, iterations=2)
        facing = edge & (d <= reach + 2)  # the part of the edge by the neighbour
        away = edge & (d > reach + 6)
        r = lambda q: R[:, z][:, q].mean(1) if q.sum() >= 4 else None  # noqa: E731
        out.append(dict(core=r(c), whole=r(m), edge=r(edge), facing=r(facing), away=r(away), nb=r(cores[nb, z])))
    return out


def summarize(tag, recs):
    cat = lambda k, ok: np.concatenate([recs[i][k] for i in ok])  # noqa: E731
    core = cat("core", range(len(recs)))
    line = f"{tag:8s} masks {len(recs):4d}  responding (> {THR}): core {100 * (core > THR).mean():5.1f}%"
    for k in ("whole", "edge", "facing", "away"):
        ok = [i for i, r in enumerate(recs) if r[k] is not None]
        v, c, n = cat(k, ok), cat("core", ok), cat("nb", ok)
        quiet = c < THR / 2
        # weight of the neighbour's core in v = a core + b neighbour (both >= 0), per mask
        b = []
        for vi, ci, ni in ((recs[i][k], recs[i]["core"], recs[i]["nb"]) for i in ok):
            (a_, b_), _ = nnls(np.stack([ci, ni], 1), vi)
            b.append(b_ / (a_ + b_ + 1e-9))
        hot = quiet & (n > THR)  # the core quiet, the neighbour responding
        line += (f"\n    {k:7s} responding {100 * (v > THR).mean():5.1f}%; of core-quiet: {100 * (v[quiet] > THR).mean():5.2f}%;"
                 f" of core-quiet with the neighbour responding ({hot.sum()}): {100 * (v[hot] > THR).mean():5.1f}%,"
                 f" mean dF/F {v[hot].mean():.3f} (core {c[hot].mean():.3f}, neighbour {n[hot].mean():.3f});"
                 f" neighbour weight median {np.median(b):.3f}, mean {np.mean(b):.3f}")
    print(line)


P_recs, M_recs = [], []
CAT = {}
n_masks = n_two = n_match = n_none = 0
for name in names:
    e = entries[name]
    f = flymod.load_fly(Path("/local/tensor") / e["dir"], model, raw_dir=e["raw_dir"], name=name,
                        glomeruli=man["glomeruli"])
    R, M = f["responses"], f["labels"]
    P = tifffile.imread(FIT / "results" / f"{name}_pred.tif").astype(np.int32)
    if len(sys.argv) > 2:
        R = other_panels(Path("/local/tensor") / e["dir"], R)
        if R is None:
            continue
    P_recs += edge_report(P, R)
    M_recs += edge_report(M, R)
    drawn = [z for z in range(M.shape[0]) if M[z].any()]
    for l, z, m in planes(P):
        if z not in drawn:
            continue
        n_masks += 1
        ms, n = np.unique(M[z][m], return_counts=True)
        # manual ROIs this mask has at least 40% of
        got = [int(g) for g, c in zip(ms, n) if g and c >= 0.4 * (M[z] == g).sum()]
        n_two += len(got) >= 2
        CAT.setdefault(min(len(got), 2), []).append(R[:, z][:, m].mean(1))
        if len(got) == 1:
            CAT.setdefault('manual of it', []).append(R[:, z][:, M[z] == got[0]].mean(1))
        n_match += len(got) == 1
        n_none += len(got) == 0
print(f"predicted in-plane masks on planes with outlines: {n_masks}; holding >= 40% of one manual ROI: {n_match},"
      f" of two or more: {n_two}, of none: {n_none}")
summarize("pred", P_recs)
summarize("manual", M_recs)
for k, v in CAT.items():
    x = np.concatenate(v)
    print(f"masks holding {k} manual ROIs: {len(v)}; responding {100 * (x > THR).mean():.1f}%, mean dF/F {x.mean():.3f}")
