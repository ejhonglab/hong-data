"""Runs the try-out (nmflib.py) on the cached flies (cache.py), and measures every candidate against the manual ROIs: its size, the
largest circle in it, how reliably it answers odors (eta2), how alike its time course is to its neighbours' (nb_corr) and how much
of its tuning they leave unexplained (novel), where it lies in the predicted ROIs' hull, and the manual outline it is on, if any;
of a mask cut in two, whether it holds two manual outlines and each part is on its own. Writes <out>/<tag>/rows.json (for
score.py) and each fly's candidates with their masks (<fly>_cands.pkl, for grid.py).

Usage (in the olfdb-worker image (docker run --gpus for the GPU), with this folder at /code, fit 7's folder at /fit and tensor at /local/tensor (all
read-only), a scratch folder at /cache (cache.py's: 0.6 to 1.4 GB a fly) and one at /out; design.md, "Glomeruli a prediction's ROIs miss"):
run.py <tag> [key=value ...]   (the settings at the top; sweep.sh runs those design.md gives numbers for)"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy import ndimage as ndi
from scipy.optimize import nnls
from skimage.morphology import convex_hull_image

sys.path.insert(0, "/code")
import nmflib as L  # noqa: E402

CACHE, FIT, OUT = Path("/cache"), Path("/fit"), Path("/out")
P = dict(n_free=25, sigma_um=0.5, seed_um=1.5, win_um=10.0, reach_um=2.0, iters=30, nonneg=1, theta=0.35, min_um2=10.0, open_um=1.0,
         seed_gap_um=1.5, hull_um=0.0, flies="Sam_2024-06-11_2,Tom_2023-05-08_1,Tom_2023-11-21_3,Tom_2024-09-12_2", norm="noise", on=0)
tag = sys.argv[1]
for a in sys.argv[2:]:
    k, v = a.split("=")
    P[k] = type(P[k])(v) if not isinstance(P[k], str) else v
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.3)
print(tag, P, DEV, flush=True)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def trial_groups(meta):
    rec, tr, on = (np.array(meta[k]) for k in ("frame_rec", "frame_trial", "frame_on"))
    key = rec * 10000 + tr
    sels, groups = [], []
    for kk in np.unique(key):
        sel = (key == kk) & (on == 1)
        if sel.any():
            r, t = int(kk // 10000), int(kk % 10000)
            sels.append(np.flatnonzero(sel))
            groups.append((r, tuple(meta["recordings"][r]["trials"][t]["odors"])))
    ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    return sels, np.array([ids[g] for g in groups])


def eta2(trace, sels, gi):
    """Share of the variance of a trace's trial responses (its mean over each trial's frames with odor) that the odors explain."""
    resp = np.array([trace[s].mean() for s in sels])
    tot = ((resp - resp.mean()) ** 2).sum()
    within = sum(((resp[gi == i] - resp[gi == i].mean()) ** 2).sum() for i in np.unique(gi))
    return float(1 - within / tot) if tot > 0 else 0.0


def tuning(trace, sels, gi):
    resp = np.array([trace[s].mean() for s in sels])
    return np.array([resp[gi == i].mean() for i in np.unique(gi)])


rows, t_all = [], time.time()
for fly in P["flies"].split(","):
    t0 = time.time()
    f = pickle.load(open(CACHE / fly / "fly.pkl", "rb"))
    meta = json.loads((CACHE / fly / "meta.json").read_text())
    dff = np.load(CACHE / fly / "dff.npy", mmap_mode="r")
    f0 = np.load(CACHE / fly / "f0.npy")
    pred = tifffile.imread(FIT / "results" / f"{fly}_pred.tif")
    man, lut, sp = f["labels"], f["lut"], f["spacing"]
    px = float(sp[1])
    min_px = int(round(P["min_um2"] / px ** 2))
    sels, gi = trial_groups(meta)
    keep = np.ones(dff.shape[0], bool) if not P["on"] else np.array(meta["frame_on"]) == 1
    cands_all, t_nmf, t_split = [], 0.0, 0.0
    f0_ref = float(np.median(f0[pred > 0]))
    for z in range(pred.shape[0]):
        labs = [l for l in np.unique(pred[z]) if l > 0]
        if len(labs) < 3:
            continue
        D = torch.as_tensor(np.asarray(dff[:, z], dtype=np.float32), device=DEV)
        if P["norm"] == "noise":
            sd = L.noise_sd(D)
            sd = torch.clamp(sd, min=float(sd.flatten().kthvalue(int(0.2 * sd.numel())).values))
            D = D / sd
        masks = np.array([pred[z] == l for l in labs])
        near = (pred[max(0, z - 1):z + 2] > 0).any(0)
        hull = convex_hull_image(near)
        region = ndi.distance_transform_edt(~hull) <= P["hull_um"] / px if P["hull_um"] > 0 else hull
        t1 = time.time()
        A, C, centers, peaks, X = L.factorize(D[torch.as_tensor(keep, device=DEV)] if P["on"] else D, masks, n_free=P["n_free"],
                                              sigma_px=P["sigma_um"] / px, seed_sigma_px=P["seed_um"] / px,
                                              win_px=P["win_um"] / px, reach_px=P["reach_um"] / px, iters=P["iters"], nonneg=bool(P["nonneg"]),
                                              region=region, seed_gap_px=P["seed_gap_um"] / px, log=print if z in (0, 3) else None)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t_nmf += time.time() - t1
        cs = L.additions(A, C, len(masks), pred[z], peaks, theta=P["theta"], min_px=min_px, open_px=P["open_um"] / px, near_px=3.0 / px)
        t1 = time.time()
        Xall = L.blur(D, P["sigma_um"] / px).reshape(D.shape[0], -1).T if P["on"] else X
        ss = L.splits(X, pred[z], min_px=min_px, open_px=P["open_um"] / px)
        t_split += time.time() - t1
        mlabs = [l for l in np.unique(man[z]) if l > 0]
        mmasks = {l: man[z] == l for l in mlabs}
        pmasks = {l: pred[z] == l for l in labs}
        mbest = {l: max([iou(m, q) for q in pmasks.values()] or [0.0]) for l, m in mmasks.items()}
        ptrace = {l: Xall[torch.as_tensor(np.flatnonzero(q.reshape(-1)), device=DEV)].mean(0).cpu().numpy() for l, q in pmasks.items()}
        for c in cs + ss:
            m = c["eval_mask"]
            c["z"] = z
            trace = Xall[torch.as_tensor(np.flatnonzero(m.reshape(-1)), device=DEV)].mean(0).cpu().numpy()
            c["eta2"] = eta2(trace, sels, gi)
            if c["kind"] == "add":  # how much of its responses its neighbours' (the predicted masks within 4 um) do not explain
                near = ndi.distance_transform_edt(~m) <= 4.0 / px
                nbl = [l for l, q in pmasks.items() if (q & near).any()]
                y = tuning(trace, sels, gi)
                c["n_nb"] = len(nbl)
                if nbl:
                    Rn = np.stack([tuning(ptrace[l], sels, gi) for l in nbl], 1)
                    w, rn = nnls(Rn, y)
                    c["novel"] = float(rn ** 2 / (y @ y + 1e-12))
                    w2, rn2 = nnls(np.stack([ptrace[l] for l in nbl], 1), trace)
                    c["novel_t"] = float(rn2 ** 2 / (trace @ trace + 1e-12))
                else:
                    c["novel"] = c["novel_t"] = 1.0
                c["amp"] = float(np.abs(y).max())
            c["f0"] = float(f0[z][m].mean() / f0_ref)
            c["hull3"] = float((m & hull).sum() / m.sum())
            ious = {l: iou(m, mm) for l, mm in mmasks.items()}
            bl = max(ious, key=ious.get) if ious else None
            c["m_iou"], c["m_name"], c["m_pred_iou"] = (ious[bl], lut.get(int(bl)), mbest[bl]) if bl is not None else (0.0, None, 0.0)
            c["in_manual"] = float((m & (man[z] > 0)).sum() / m.sum())
            c["area_um2"] = float(m.sum() * px ** 2)
            c["inscribed_um"] = float(ndi.distance_transform_edt(m).max() * px)
            if c["kind"] == "split":
                whole = pmasks[c["of"]]
                held = [l for l, mm in mmasks.items() if (mm & whole).sum() >= 0.4 * mm.sum()]
                c["held"] = len(held)
                o = c["other_mask"]
                main = lambda q: max(mmasks, key=lambda l: (mmasks[l] & q).sum()) if mmasks and max((mm & q).sum() for mm in mmasks.values()) >= 0.5 * q.sum() else None  # noqa: E731
                a_, b_ = main(m), main(o)
                c["parts"] = [lut.get(int(a_)) if a_ is not None else None, lut.get(int(b_)) if b_ is not None else None]
                c["right"] = bool(a_ is not None and b_ is not None and a_ != b_)
                tr2 = Xall[torch.as_tensor(np.flatnonzero(o.reshape(-1)), device=DEV)].mean(0).cpu().numpy()
                c["eta2_other"] = eta2(tr2, sels, gi)
                c["trace_corr"] = L.corr(trace, tr2)
                # the same less what the plane's other masks share (their mean trace)
                others = [ptrace[l] for l in pmasks if l != c["of"]]
                if others:
                    common = np.mean(others, 0)
                    c["ccorr"] = L.corr(trace - common, tr2 - common)
                    c["tcorr"] = L.corr(tuning(trace - common, sels, gi), tuning(tr2 - common, sels, gi))
                    # each part against the other, the common trace taken out by regression
                    def resid(y):
                        b = (y @ common) / (common @ common + 1e-12)
                        return y - b * common
                    c["rcorr"] = L.corr(tuning(resid(trace), sels, gi), tuning(resid(tr2), sels, gi))
                else:
                    c["ccorr"] = c["tcorr"] = c["rcorr"] = c["trace_corr"]
                c["tuncorr"] = L.corr(tuning(trace, sels, gi), tuning(tr2, sels, gi))
            cands_all.append(c)
            rows.append({"fly": fly, **{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in c.items()
                                        if k not in ("eval_mask", "other_mask", "c", "c_other")}})
        for l, m in mmasks.items():
            rows.append({"fly": fly, "z": z, "kind": "manual", "m_name": lut.get(int(l)), "m_pred_iou": mbest[l], "area_um2": float(m.sum() * px ** 2),
                         "inscribed_um": float(ndi.distance_transform_edt(m).max() * px), "f0": float(f0[z][m].mean() / f0_ref),
                         "best_cand_iou": max([iou(m, c["eval_mask"]) for c in cs] or [0.0]),
                         "n_pred_held": int(sum((m & q).sum() >= 0.4 * m.sum() for q in pmasks.values()))})
        for l, q in pmasks.items():
            held = [k for k, mm in mmasks.items() if (mm & q).sum() >= 0.4 * mm.sum()]
            rows.append({"fly": fly, "z": z, "kind": "pred", "label": int(l), "held": len(held), "area_um2": float(q.sum() * px ** 2)})
        del D, X, Xall
    (OUT / tag).mkdir(parents=True, exist_ok=True)
    pickle.dump(cands_all, open(OUT / tag / f"{fly}_cands.pkl", "wb"))
    print(f"{fly}: {len(cands_all)} candidates, nmf {t_nmf:.1f} s, splits {t_split:.1f} s, all {time.time() - t0:.0f} s", flush=True)
(OUT / tag / "rows.json").write_text(json.dumps(rows))
(OUT / tag / "params.json").write_text(json.dumps(P))
print(f"all in {time.time() - t_all:.0f} s")
