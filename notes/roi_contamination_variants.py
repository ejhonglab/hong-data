"""How much fit 7's predicted ROIs (held-out flies) touch and pick up their neighbours, against
the manual ROIs, and what some candidate refinements would do to that. Read-only.

Usage (in the olfdb-worker image, which has al-diag-seg; a fit's folder at /fit and tensor at
/local/tensor, both read-only; see demixing_plan.md):
roi_contamination_variants.py [<threshold, dF/F: 0.3>] [other]
"other": on the flies' recordings that are not diagnostic ones (megamat, validation, ...), not the
diagnostic recordings' responses the model reads."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from scipy import ndimage as ndi

from alseg import fly as flymod, refine

FIT = Path("/fit")
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3

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
    """(label, z, mask) of every in-plane mask."""
    for z in range(L.shape[0]):
        for l in np.unique(L[z]):
            if l:
                yield int(l), z, L[z] == l


def gaps(L):
    """In-plane distance (px, 0 = touching) from each mask to the nearest other ROI on its plane."""
    out = []
    for l, z, m in planes(L):
        others = (L[z] > 0) & ~m
        if others.any():
            out.append(ndi.distance_transform_edt(~others)[m].min() - 1)
    return np.array(out)


def erode(L, n):
    out = np.zeros_like(L)
    for l, z, m in planes(L):
        out[z][ndi.binary_erosion(m, iterations=n)] = l
    return out


def gap(L, g):
    """Each ROI without its pixels within g px of another ROI (in-plane)."""
    out = L.copy()
    for l, z, m in planes(L):
        others = (L[z] > 0) & ~m
        if others.any():
            out[z][m & (ndi.distance_transform_edt(~others) <= g)] = 0
    return out


def largest(L):
    out = np.zeros_like(L)
    for l, z, m in planes(L):
        cc, k = ndi.label(m)
        if k:
            out[z][cc == 1 + np.argmax(ndi.sum(m, cc, range(1, k + 1)))] = l
    return out


def trim(L, R, sigma=1.0, margin=0.0, reach=4):
    """Each ROI without the pixels whose tuning (smoothed within the ROI) is no closer to its own core's
    than to a neighbouring ROI's core's (in-plane neighbours within `reach` px)."""
    out = L.copy()
    core = {}
    for l, z, m in planes(L):
        c = ndi.binary_erosion(m, iterations=2)
        c = c if c.sum() >= 5 else m
        t = np.median(R[:, z][:, c], axis=1)
        core[l, z] = (t - t.mean()) / (np.linalg.norm(t - t.mean()) + 1e-8)
    for l, z, m in planes(L):
        near = np.unique(L[z][ndi.binary_dilation(m, iterations=reach) & ~m])
        near = [int(n) for n in near if n]
        if not near:
            continue
        w = ndi.gaussian_filter(m.astype(np.float32), sigma)
        V = np.stack([ndi.gaussian_filter(np.where(m, r, 0), sigma) / (w + 1e-8) for r in R[:, z]])[:, m].T
        V = V - V.mean(1, keepdims=True)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
        own = V @ core[l, z]
        rival = np.max([V @ core[n, z] for n in near], axis=0)
        drop = np.zeros_like(m)
        drop[m] = rival + margin >= own
        out[z][drop] = 0
    return largest(out)


def demix(L, R, sigma=1.5, wz=0.0):
    """Responses of each in-plane mask with its neighbours' unmixed: every pixel in or near an ROI as a
    sum of the ROIs' footprints (masks blurred by `sigma` px; those of the planes next to it times `wz`)."""
    keys = [(l, z) for l, z, _ in planes(L)]
    F = np.zeros((len(keys),) + L.shape, np.float32)
    for k, (l, z) in enumerate(keys):
        f = ndi.gaussian_filter((L[z] == l).astype(np.float32), sigma)
        F[k, z] = f
        if wz:
            for dz in (-1, 1):
                if 0 <= z + dz < L.shape[0]:
                    F[k, z + dz] = wz * ndi.gaussian_filter(f, sigma)
    region = F.sum(0) > 0.05
    C, *_ = np.linalg.lstsq(F[:, region].T, R[:, region].T, rcond=None)
    return dict(zip(keys, C))


def mean_resp(L, R):
    return {(l, z): R[:, z][:, m].mean(1) for l, z, m in planes(L)}


rows = {}


def add(kind, **kw):
    r = rows.setdefault(kind, {})
    for k, v in kw.items():
        r.setdefault(k, []).extend(np.atleast_1d(v).tolist())


for name in names:
    e = entries[name]
    f = flymod.load_fly(Path("/local/tensor") / e["dir"], model, raw_dir=e["raw_dir"], name=name,
                        glomeruli=man["glomeruli"])
    R, M, sp = f["responses"], f["labels"], f["spacing"]
    P = tifffile.imread(FIT / "results" / f"{name}_pred.tif").astype(np.int32)
    if len(sys.argv) > 2:
        R = other_panels(Path("/local/tensor") / e["dir"], R)
        if R is None:
            continue
    assert P.shape == M.shape, (P.shape, M.shape)
    print(name, "voxel um", np.round(sp, 2), "pred ROIs", len(np.unique(P)) - 1, "manual", len(np.unique(M)) - 1, flush=True)
    # each predicted in-plane mask's manual one (IoU >= 0.5)
    pairs = []
    for l, z, m in planes(P):
        ms, n = np.unique(M[z][m], return_counts=True)
        for g, c in zip(ms, n):
            if g and c / (m | (M[z] == g)).sum() >= 0.5:
                pairs.append((l, z, int(g)))
    mresp = mean_resp(M, R)
    variants = {"pred": P, "erode 1": erode(P, 1), "erode 2": erode(P, 2), "gap 2": gap(P, 2), "gap 3": gap(P, 3),
                "trim": trim(P, R), "gap 2 + trim": trim(gap(P, 2), R),
                "refined_tuning": tifffile.imread(FIT / "results" / f"{name}_refined_tuning.tif").astype(np.int32),
                "refined_prior": tifffile.imread(FIT / "results" / f"{name}_refined_prior.tif").astype(np.int32)}
    resp = {k: mean_resp(v, R) for k, v in variants.items()}
    for k, kw in {"demix s1.5": {}, "demix s1.5 z0.2": {"wz": 0.2}, "demix s2.5": {"sigma": 2.5}}.items():
        variants[k], resp[k] = P, demix(P, R, **kw)
    add("manual", gap=gaps(M) * sp[1], v=[mresp[g, z] for _, z, g in pairs], m=[mresp[g, z] for _, z, g in pairs])
    for k, V in variants.items():
        add(k, gap=gaps(V) * sp[1], leak=list(refine.crosstalk_report(R, V, sp)["leak"].values()))
        for l, z, g in pairs:
            m = V[z] == l
            add(k, area=m.sum() / (P[z] == l).sum())
            if not m.any():
                add(k, lost=1)
                continue
            add(k, lost=0, own=(M[z][m] == g).mean(), other=((M[z][m] > 0) & (M[z][m] != g)).mean())
            rows[k].setdefault("v", []).append(resp[k][l, z].tolist())
            rows[k].setdefault("m", []).append(mresp[g, z].tolist())

allm = np.concatenate([np.ravel(x) for x in rows["manual"]["m"]])
print(f"\nmatched in-plane pairs: {len(rows['manual']['m'])}; manual responses' quantiles (5, 25, 50, 75, 95, 99%):",
      np.round(np.quantile(allm, [.05, .25, .5, .75, .95, .99]), 2), f"; threshold {THR}")
cols = ["gap um", "touch %", "<1.5um %", "leak", "area", "lost %", "in own %", "in other %", "resp %", "extra %", "missed %", "slope", "sparse"]
print(f"{'':18s}" + "".join(f"{c:>11s}" for c in cols))
for k, r in rows.items():
    vs = [np.array(x) for x in r["v"]]
    v, m = np.concatenate(vs), np.concatenate([np.array(x) for x in r["m"]])
    quiet, loud = m < THR / 2, m > THR
    g = np.array(r["gap"])
    # lifetime sparseness of each ROI's (rectified) responses: 1 = one odor, 0 = all alike
    spars = [(1 - x.mean() ** 2 / ((x ** 2).mean() + 1e-12)) / (1 - 1 / len(x)) for x in (np.clip(q, 0, None) for q in vs)]
    vals = [np.median(g), 100 * (g <= 0).mean(), 100 * (g < 1.5).mean(), np.mean(r.get("leak", [np.nan])),
            np.median(r.get("area", [1])), 100 * np.mean(r.get("lost", [0])), 100 * np.mean(r.get("own", [1])),
            100 * np.mean(r.get("other", [0])), 100 * (v > THR).mean(), 100 * (v[quiet] > THR).mean(),
            100 * (v[loud] < THR / 2).mean(), np.polyfit(m[loud], v[loud], 1)[0], np.median(spars)]
    print(f"{k:18s}" + "".join(f"{x:11.2f}" for x in vals))
