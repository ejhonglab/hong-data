"""Seeded NMF of ROIs' footprints from every frame of the movies (not one dF map per trial, as unmix.py's nmf), on
fit 7's held-out flies, scored as notes/unmix_validation.py scores unmixing ((a)-(d), on the diagnostic recordings'
responses); and how well each kind of evidence tells a pixel's glomerulus from its neighbour's (deep core pixels'
neighbour shares, edge pixels', and masks over two manual ROIs against halves of one). Read-only.

Footprints (variants): unmix.py's NMF from one dF map per trial (its diagnostic recordings, as unmix_validation.py);
from every frame of each trial (onset to end, less its baseline mean) of the diagnostic recordings, or of all the
fly's recordings; and with a component more per ROI-plane for the region's pixels outside every mask nearest it (its
rim), reaching 2 um, 1 px or nothing into the masks. The NMF is unmix.nmf's in PyTorch (`nmf_torch`; on the GPU if
there is one), checked against it on one plane, and timed there and on a whole plane (as a factorization finding
glomeruli without an ROI would be).

Usage (in the olfdb-worker image, this tree at /repo and on PYTHONPATH, notes/unmix_validation.py's cache folder at
/cache (its diagnostic recordings' trials, which it makes), fit 7's folder at /fit and tensor at /local/tensor, all
read-only; with --gpus for the GPU; about 40 min on 16 CPUs, 24 GB):
unmix_movie_nmf.py [<flies, comma separated>]
Results (design.md, "Neighbouring ROIs unmixed"): printed, as Markdown tables."""
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.argv = [sys.argv[0], "/cache"] + sys.argv[1:]
sys.path.insert(0, "/repo/notes")
import unmix_validation as H  # noqa: E402
from app import extract, movies, unmix  # noqa: E402

FLIES = sys.argv[2].split(",") if len(sys.argv) > 2 else H.FLIES
REACH_UM = 2.0
t_all = time.time()


def trial_frames(rd, shape):
    """A recording's trials, and every frame of each from its odor's onset to its end less its baseline mean:
    (trials, frames, Z, Y, X) float32, NaN past a trial's end; None if its movie is on another grid."""
    m = movies.open_movie(next(rd / n for n in movies.PREFERRED if (rd / n).is_file()))
    s = m.shape
    if (s["z"], s["y"], s["x"]) != tuple(shape):
        return None
    trials = extract.read_trials(rd / movies.TRIALS)
    L = max(t["end"] + 1 - t["onset"] for t in trials)
    out = np.full((len(trials), L) + tuple(shape), np.nan, np.float32)
    for k, t in enumerate(trials):
        if t["onset"] - t["start"] > 0:
            b = m.volumes(t["start"], t["onset"] - t["start"]).mean(axis=0, dtype=np.float64)
            f = m.volumes(t["onset"], t["end"] + 1 - t["onset"])
            out[k, :len(f)] = f - b
    return out, trials


def region(a, p):
    """(trials, frames, Z, Y, X) or (trials, Z, Y, X) -> (a plane's region pixels, columns), NaN columns dropped."""
    y0, y1, x0, x1 = p["box"]
    a = a[..., p["z"], y0:y1, x0:x1]
    a = a.reshape(-1, (y1 - y0) * (x1 - x0))[:, p["idx"]]
    return a[~np.isnan(a).any(axis=1)].T.astype(np.float64)


def odor_means(maps, trials):
    """(px, trials) -> (px, odors): each odor's mean map."""
    by = {}
    for k, t in enumerate(trials):
        by.setdefault(tuple(t["odors"]), []).append(k)
    return np.stack([np.nanmean(maps[:, ks], axis=1) for ks in by.values()], axis=1)


def residuals(fr, trials):
    """(trials, frames, ...) less each odor's mean over its trials, frame by frame: what is not stimulus-locked."""
    out = fr.copy()
    by = {}
    for k, t in enumerate(trials):
        by.setdefault(tuple(t["odors"]), []).append(k)
    for ks in by.values():
        if len(ks) > 1:
            out[ks] -= np.nanmean(fr[ks], axis=0)
        else:
            out[ks] = np.nan
    return out


def shares(Y, c, n):
    """Each column of Y (T, P) as a c + b n (a, b >= 0): b / (a + b)."""
    X = np.stack([c, n], 1)
    G = X.T @ X
    a, b = np.linalg.solve(G, X.T @ Y)
    only_c = b < 0
    a = np.where(only_c, np.maximum(0, c @ Y / (c @ c)), a)
    b = np.where(only_c, 0, b)
    only_n = a < 0
    b = np.where(only_n, np.maximum(0, n @ Y / (n @ n)), b)
    a = np.where(only_n, 0, a)
    return b / (a + b + 1e-12)


def halo_planes(planes, rois_n, reach_px, rim_reach_px=None):
    """Each plane with a component of its own for the region's pixels outside every mask nearest each mask (its rim):
    light round the ROIs (glomeruli without one, or their spill) then need not be taken for theirs."""
    out = []
    for p in planes:
        union = p["masks"].any(axis=0)
        halo = p["region"] & ~union
        d = np.stack([ndi.distance_transform_edt(~m) for m in p["masks"]])
        near = d.argmin(axis=0)
        extra = [halo & (near == j) for j in range(len(p["masks"]))]
        extra = [x for x in extra if x.sum() >= 8]
        q = dict(p)
        q["masks"] = np.concatenate([p["masks"], np.array(extra, bool).reshape(-1, *union.shape)])
        q["n_real"] = len(p["masks"])
        q["rim_reach_px"] = reach_px if rim_reach_px is None else rim_reach_px
        out.append(q)
    return out


import torch  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.25)  # (the GPU is the fit worker's too)


def nmf_torch(p, D, reach_px, iters=unmix.NMF_ITERS, tol=unmix.NMF_TOL, dev=DEV):
    """unmix.nmf (no background) in PyTorch, in float32, on `dev`: the same alternating least squares and HALS.
    Components past the plane's ROIs (``n_real``: rims) reach ``rim_reach_px`` beyond their pixels."""
    K0 = p.get("n_real", len(p["masks"]))
    reach = [reach_px if k < K0 else p.get("rim_reach_px", reach_px) for k in range(len(p["masks"]))]
    M = unmix._flat(p, p["masks"].astype(np.float64))
    S = unmix._flat(p, np.stack([ndi.distance_transform_edt(~m) <= r for m, r in zip(p["masks"], reach)])) > 0
    X = torch.as_tensor(unmix._known(D), device=dev, dtype=torch.float32)
    Mt = torch.as_tensor(M, device=dev, dtype=torch.float32)
    St = torch.as_tensor(S, device=dev)
    A, K, n = Mt.clone(), M.shape[1], Mt.sum(0)
    it = 0
    for it in range(1, iters + 1):
        C = torch.linalg.lstsq((A.T @ A).double().cpu(), (A.T @ X).double().cpu(), driver="gelsd").solution.to(dev).float()
        G, Hm = C @ C.T, X @ C.T
        old = A.clone()
        for k in range(K):
            if G[k, k] > 0:
                A[:, k] = torch.where(St[:, k], torch.clamp((Hm[:, k] - A @ G[:, k]) / G[k, k] + A[:, k], min=0), 0.0)
        own = (A * Mt).sum(0) / n
        dead = ~(own > 0)
        A[:, dead] = Mt[:, dead]
        A[:, ~dead] /= own[~dead]
        if float(torch.linalg.norm(A - old) / torch.linalg.norm(old)) < tol:
            break
    return A.double().cpu().numpy(), it


def footprints_nmf(planes, D, reach_px):
    A = []
    for p, d in zip(planes, D):
        a, _ = nmf_torch(p, d, reach_px)
        A.append(a)
    return A


timing = {}


def weights(planes, A, rois):
    Wun, refused = [], np.zeros(len(rois), bool)
    for p, a in zip(planes, A):
        sol = unmix.solve(p, a, False)
        K = p.get("n_real", len(p["members"]))
        W = sol["weights"][:K].copy()
        M = unmix._flat(p, p["masks"][:K].astype(float))
        W[sol["refused"][:K]] = (M / M.sum(0)).T[sol["refused"][:K]]
        refused[p["members"]] = sol["refused"][:K]
        Wun.append(W)
    return Wun, refused


info_rows, merge_rows, single_rows, score_rows = [], [], [], []
for fly in FLIES:
    t0 = time.time()
    e, recs, px = H.load(fly)
    Z, Y, X = recs[0]["base"].shape[1:]
    fly_dir = H.Path("/local/tensor") / e["dir"]
    diag = {r["name"] for r in e["recordings"]}
    allr = []
    for rd in sorted(q for q in fly_dir.iterdir() if q.is_dir()):
        if not any((rd / n).is_file() for n in movies.PREFERRED) or not (rd / movies.TRIALS).is_file():
            continue
        got = trial_frames(rd, (Z, Y, X))
        if got is not None:
            allr.append({"name": rd.name, "diag": rd.name in diag, "frames": got[0], "trials": got[1]})
    print(f"{fly}: {len(allr)} recordings ({sum(r['diag'] for r in allr)} diagnostic), "
          f"{sum(r['frames'].shape[0] for r in allr)} trials, read in {time.time() - t0:.0f} s", flush=True)
    sets = H.roisets(fly, e, (Z, Y, X))
    geo = {}
    for kind, rois in sets.items():
        full, cores = H.plane_masks(rois, (Y, X))
        nb, alln = H.neighbours(rois, full, cores)
        geo[kind] = (full, cores, nb, alln)
    reach_px = REACH_UM / px
    for kind, rois in sets.items():
        full, cores, nb, alln = geo[kind]
        planes = unmix.layout(rois, (Y, X), reach_px)
        # evidence per plane: dF maps per trial (as unmix.py: the response window's mean), and every frame
        ev = {"maps diag": [], "frames diag": [], "frames all": [], "maps all": [], "odor means diag": []}
        for p in planes:
            md = [region(np.nanmean(rec["frames"], axis=1) - rec["base"], p) for rec in recs]
            ev["maps diag"].append(np.concatenate(md, axis=1))
            ev["odor means diag"].append(np.concatenate([odor_means(m_, rec["trials"]) for m_, rec in zip(md, recs)], axis=1))
            ev["frames diag"].append(np.concatenate([region(r["frames"], p) for r in allr if r["diag"]], axis=1))
            ev["frames all"].append(np.concatenate([region(r["frames"], p) for r in allr], axis=1))
            n = recs[0]["n"]
            ev["maps all"].append(np.concatenate([region(np.nanmean(r["frames"][:, :n], axis=1), p) for r in allr], axis=1))
        # how well each kind of evidence tells a pixel's glomerulus from its nearest neighbour's (predicted ROIs)
        if kind == "pred":
            for name, D in ev.items():
                deep, edge = [], []
                for p, d in zip(planes, D):
                    y0, y1, x0, x1 = p["box"]
                    for j, i in enumerate(p["members"]):
                        if nb[i] is None or cores[i] is None:
                            continue
                        crop = lambda m: m[y0:y1, x0:x1].ravel()[p["idx"]]  # noqa: E731
                        own, other = crop(cores[i]), crop(cores[nb[i]])
                        if own.sum() < 8 or other.sum() < 8:
                            continue
                        dn = ndi.distance_transform_edt(~full[nb[i]])
                        dp = crop(ndi.binary_erosion(full[i], iterations=4) & (dn > 8))
                        ep = crop(full[i] & (dn <= 2))
                        c, o = d[own].mean(0), d[other].mean(0)
                        if dp.any():
                            deep.extend(shares(d[dp].T, c, o).tolist())
                        if ep.any():
                            edge.extend(shares(d[ep].T, c, o).tolist())
                deep, edge = np.array(deep), np.array(edge)
                info_rows.append((fly, name, D[0].shape[1], len(deep), np.median(deep) if len(deep) else 0.0,
                                  100 * (deep > 0.2).mean() if len(deep) else 0.0, len(edge), np.median(edge) if len(edge) else 0.0))
            if not timing:  # the same NMF on the CPU (numpy: unmix.nmf) and on the GPU, on one plane of every frame
                p, d = planes[0], ev["frames all"][0]
                t1 = time.time(); a_cpu, _ = unmix.nmf(p, d, reach_px, False); timing["cpu"] = time.time() - t1
                t1 = time.time(); a_gpu, its = nmf_torch(p, d, reach_px); timing["gpu"] = time.time() - t1
                timing.update(px=d.shape[0], cols=d.shape[1], k=len(p["members"]), its=its,
                              diff=float(np.abs(a_cpu - a_gpu).max() / np.abs(a_cpu).max()))
                print("  timing, one plane:", timing, flush=True)
        # unmixed responses: footprints by seeded NMF from each kind of evidence
        variants = {"nmf maps diag (unmix.py)": (planes, "maps diag"), "nmf frames diag": (planes, "frames diag"),
                    "nmf maps all": (planes, "maps all"), "nmf frames all": (planes, "frames all"),
                    "nmf frames all + rims": (halo_planes(planes, len(rois), reach_px), "frames all"),
                    "nmf frames all + rims 1 px in": (halo_planes(planes, len(rois), reach_px, 1.0), "frames all"),
                    "nmf frames all + rims outside": (halo_planes(planes, len(rois), reach_px, 0.0), "frames all"),
                    "nmf maps all + rims": (halo_planes(planes, len(rois), reach_px), "maps all")}
        for vname, (pl, evk) in variants.items():
            t1 = time.time()
            A = footprints_nmf(pl, ev[evk], reach_px)
            t_fit = time.time() - t1
            Wun, refused = weights(pl, A, rois)
            R = H.responses(recs, planes, rois, cores, Wun)
            m = H.metrics(R, rois, nb, alln, refused)
            score_rows.append((fly, kind, vname, m, t_fit))
            print(f"  {kind:6s} {vname:26s} a {m.get('a_plain', np.nan):.3f} -> {m.get('a_unmixed', np.nan):.3f} (core {m.get('a_core', np.nan):.3f}) "
                  f"b {m['b_ratio_big']:.3f} c {m['c_plain']}->{m['c_unmixed']} (+{m['c_new']}) d {m['d_plain']:.1f}->{m['d_unmixed']:.1f} "
                  f"refused {m['refused']}  fit {t_fit:.1f} s", flush=True)
    # masks over two manual ROIs, against halves of masks over one: how alike the two parts' signals are
    pr, mr = sets["pred"], sets["manual"]
    pfull, mfull = geo["pred"][0], geo["manual"][0]
    res = [{"frames": residuals(r["frames"], r["trials"])} for r in allr]
    for i, r in enumerate(pr):
        held = sorted(((pfull[i] & mfull[j]).sum(), j) for j, q in enumerate(mr) if q["z"] == r["z"]
                      and (pfull[i] & mfull[j]).sum() >= 0.4 * mfull[j].sum())
        if len(held) >= 2:
            parts = [pfull[i] & mfull[held[-1][1]], pfull[i] & mfull[held[-2][1]]]
            rows = merge_rows
        elif len(held) == 1:
            m_ = pfull[i] & mfull[held[0][1]]
            yy, xx = np.nonzero(m_)
            if len(yy) < 40:
                continue
            v = np.linalg.svd(np.stack([yy - yy.mean(), xx - xx.mean()]), full_matrices=False)[0][:, 0]
            side = (np.stack([yy - yy.mean(), xx - xx.mean()]).T @ v) > 0
            parts = [np.zeros_like(m_), np.zeros_like(m_)]
            parts[0][yy[side], xx[side]] = True
            parts[1][yy[~side], xx[~side]] = True
            rows = single_rows
        else:
            continue
        z = r["z"]

        def trace(a, q):
            x = a[..., z, :, :][..., q].mean(-1).ravel()
            return x

        out = {}
        for name, srcs in (("odor means diag", None), ("maps all", "maps"), ("frames all", "frames"), ("residual frames all", "res")):
            if name == "odor means diag":
                vs = []
                for q in parts:
                    vs.append(np.concatenate([odor_means((np.nanmean(rec["frames"], axis=1) - rec["base"])[:, z][:, q].mean(-1)[None], rec["trials"])[0]
                                              for rec in recs]))
            elif srcs == "maps":
                n = recs[0]["n"]
                vs = [np.concatenate([np.nanmean(rr["frames"][:, :n], axis=1)[:, z][:, q].mean(-1) for rr in allr]) for q in parts]
            elif srcs == "frames":
                vs = [np.concatenate([trace(rr["frames"], q) for rr in allr]) for q in parts]
            else:
                vs = [np.concatenate([trace(rr["frames"], q) for rr in res]) for q in parts]
            a_, b_ = vs
            ok = ~(np.isnan(a_) | np.isnan(b_))
            out[name] = float(np.corrcoef(a_[ok], b_[ok])[0, 1])
        out["px"] = int(min(q.sum() for q in parts))
        rows.append((fly, i, out))
    print(f"{fly}: done in {time.time() - t0:.0f} s", flush=True)
    del allr, res

print("\n## evidence: neighbour's share of deep core pixels (true ~0) and of edge pixels (within 1 um of the neighbour), predicted ROIs")
print("| evidence | columns (fly 1) | deep px | deep median | deep > 0.2 | edge px | edge median |")
print("|---|---|---|---|---|---|---|")
for name in dict.fromkeys(r[1] for r in info_rows):
    rs = [r for r in info_rows if r[1] == name]
    w = np.array([r[3] for r in rs], float)
    we = np.array([r[6] for r in rs], float)
    print(f"| {name} | {rs[0][2]} | {int(w.sum())} | {np.average([r[4] for r in rs], weights=w):.3f} | "
          f"{np.average([r[5] for r in rs], weights=w):.1f}% | {int(we.sum())} | {np.average([r[7] for r in rs], weights=we):.3f} |")

print("\n## masks over two manual ROIs: correlation of the two parts' mean signals; halves of masks over one, for comparison")
names = [k for k in merge_rows[0][2] if k != "px"] if merge_rows else []
print("pixels of the smaller part: merges", sorted(r[2]["px"] for r in merge_rows), "; halves of one: median",
      np.median([r[2]["px"] for r in single_rows]), "5%", np.quantile([r[2]["px"] for r in single_rows], 0.05))
print("| evidence | merges (each) | halves of one: 5% / 50% | merges below the halves' 5% |")
print("|---|---|---|---|")
for name in names:
    mv = np.array([r[2][name] for r in merge_rows])
    sv = np.array([r[2][name] for r in single_rows])
    q5, q50 = np.quantile(sv, [0.05, 0.5])
    print(f"| {name} | {', '.join(f'{x:.2f}' for x in sorted(mv))} | {q5:.2f} / {q50:.2f} (n {len(sv)}) | {(mv < q5).sum()} of {len(mv)} |")

for r in merge_rows:
    print("  merge", r[0], r[1], {k: round(v, 2) for k, v in r[2].items()})
print("\n## unmixing, pooled over flies ((a) weighted by pairs, (b) median over flies, (c) sums, (d) % of pairs)")
print("| footprints | ROIs | (a) plain -> unmixed (core) | (b) | (c) plain -> unmixed (new) | (d) % | fit s (sum) |")
print("|---|---|---|---|---|---|---|")
for vname in dict.fromkeys(r[2] for r in score_rows):
    for kind in ("pred", "manual"):
        rs = [r for r in score_rows if r[2] == vname and r[1] == kind]
        w = np.array([r[3].get("a_n", 0) for r in rs], float)
        a = lambda k: np.average([r[3].get(k, 0) for r in rs], weights=w)  # noqa: E731
        print(f"| {vname} | {kind} | {a('a_plain'):.3f} -> {a('a_unmixed'):.3f} ({a('a_core'):.3f}) | {np.median([r[3]['b_ratio_big'] for r in rs]):.3f} | "
              f"{sum(r[3]['c_plain'] for r in rs)} -> {sum(r[3]['c_unmixed'] for r in rs)} ({sum(r[3]['c_new'] for r in rs)}) | "
              f"{np.average([r[3]['d_plain'] for r in rs], weights=[r[3]['pairs'] for r in rs]):.1f} -> "
              f"{np.average([r[3]['d_unmixed'] for r in rs], weights=[r[3]['pairs'] for r in rs]):.1f} | {sum(r[4] for r in rs):.0f} |")
print("\none plane, seeded (unmix.nmf, numpy, CPU) against PyTorch on", DEV, timing)
# a whole plane factorized over every frame, as to find glomeruli without an ROI too: HALS, 80 components, 100 iterations
X = torch.randn(192 * 192, 2500, device=DEV).abs()
for dev in ([DEV, torch.device("cpu")] if DEV.type == "cuda" else [DEV]):
    Xd = X.to(dev)
    A, C = torch.rand(Xd.shape[0], 80, device=dev), torch.rand(80, Xd.shape[1], device=dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(100):
        C = torch.clamp(torch.linalg.solve(A.T @ A + 1e-6 * torch.eye(80, device=dev), A.T @ Xd), min=0)
        G, Hm = C @ C.T, Xd @ C.T
        for k in range(80):
            A[:, k] = torch.clamp(A[:, k] + (Hm[:, k] - A @ G[:, k]) / (G[k, k] + 1e-9), min=0)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"whole plane (36864 px x 2500 frames, 80 components, 100 iterations) on {dev}: {time.time() - t1:.1f} s", flush=True)
print(f"\nall in {time.time() - t_all:.0f} s")
