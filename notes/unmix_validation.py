"""Unmixing (app/unmix.py, Extract signals' option unmix) on fit 7's held-out flies' diagnostic recordings,
with their predicted and their manual ROIs (outlines of others, as Sam's AL_plane<n>, left out): each ROI-plane's
response to each odor (the mean over its trials of the response as extract.py computes it, from each trial's
baseline mean and response window) plain, unmixed and of the ROI's core, and

(a) where an ROI's core is quiet (< THR / 2) and its nearest neighbour's core responds (> THR), as
    notes/roi_contamination_edges.py defines them: the ROI's response plain and unmixed;
(b) where every neighbour's core is quiet: unmixed against plain (slope; median ratio, of |plain| > THR);
(c) ROI-odor pairs below -THR, plain and unmixed (new: not so plain);
(d) pairs over THR, all and on matched masks (IoU >= 0.5), predicted and manual;
(e) all of these for several settings (SETTINGS).

Usage (in the olfdb-worker image, with this tree on PYTHONPATH; fit 7's folder at /fit and tensor at
/local/tensor, both read-only; see demixing_plan.md, section 4):
unmix_validation.py <cache folder> [<settings, comma separated>] [<flies, comma separated>]
Each recording's trials' baseline means and response windows are read from its movie once, into the cache
folder (float32: ~180 MB a recording), and from there after. Results: <cache folder>/results_<time>.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import nnls

from app import extract, movies, unmix

FIT, CACHE = Path("/fit"), Path(sys.argv[1] if len(sys.argv) > 1 else "cache")
THR, CORE, NEAR = 0.3, 3, 4
man = json.loads((FIT / "manifest.json").read_text())
entries = {f"{e['dir'].split('/')[0]}_{e['date']}_{e['fly_num']}": e for e in man["flies"]}
FLIES = ["Sam_2024-06-11_2", "Tom_2023-05-08_1", "Tom_2023-11-21_3", "Tom_2024-09-12_2"]

SETTINGS = {  # (unmix.py's own but where said: no background, the region and NMF reach 2 um, the blur's width fit)
    "blur": dict(variant="blur"),
    "nmf": dict(variant="nmf"),
    "nmf rims": dict(variant="nmf_rims"),
    "blur s0.25": dict(variant="blur", sigma_um=0.25),
    "blur s0.5": dict(variant="blur", sigma_um=0.5),
    "blur s1": dict(variant="blur", sigma_um=1.0),
    "blur s1.5": dict(variant="blur", sigma_um=1.5),
    "blur reach 1": dict(variant="blur", reach=1.0),
    "blur reach 2": dict(variant="blur", reach=2.0),
    "blur reach 3": dict(variant="blur", reach=3.0),
    "blur reach 2 s0.5": dict(variant="blur", reach=2.0, sigma_um=0.5),
    "blur reach 2 s1": dict(variant="blur", reach=2.0, sigma_um=1.0),
    "blur reach 2 s1.5": dict(variant="blur", reach=2.0, sigma_um=1.5),
    "blur reach 2 s2": dict(variant="blur", reach=2.0, sigma_um=2.0),
    "nmf reach 1": dict(variant="nmf", reach=1.0),
    "nmf reach 3": dict(variant="nmf", reach=3.0),
    "nmf 20 it": dict(variant="nmf", iters=20),
    "nmf 300 it": dict(variant="nmf", iters=300),
    "nmf rims 20 it": dict(variant="nmf_rims", iters=20),
    "nmf rims 300 it": dict(variant="nmf_rims", iters=300),
    # with a background uniform over the region (as demixing_plan.md had it; the blur's then over the masks and 2 um round them)
    "blur bg": dict(variant="blur", background=True, reach=2.0),
    "nmf bg": dict(variant="nmf", background=True),
    "blur bg reach 1": dict(variant="blur", background=True, reach=1.0),
    "blur bg reach 3": dict(variant="blur", background=True, reach=3.0),
    "blur bg s0.5": dict(variant="blur", background=True, reach=2.0, sigma_um=0.5),
    "blur bg s1": dict(variant="blur", background=True, reach=2.0, sigma_um=1.0),
    "blur bg s1.5": dict(variant="blur", background=True, reach=2.0, sigma_um=1.5),
    "blur bg s2": dict(variant="blur", background=True, reach=2.0, sigma_um=2.0),
    "nmf bg reach 1": dict(variant="nmf", background=True, reach=1.0),
    "nmf bg reach 3": dict(variant="nmf", background=True, reach=3.0),
    "nmf bg 20 it": dict(variant="nmf", background=True, iters=20),
}


def cache(fly, e, r):
    """A recording's trials, response window, and each trial's baseline mean volume and response window's volumes."""
    rd = Path("/local/tensor") / e["dir"] / r["name"]
    movie = movies.open_movie(next(rd / n for n in movies.PREFERRED if (rd / n).is_file()))
    trials = extract.read_trials(rd / movies.TRIALS)
    win = extract.window(Path(e["raw_dir"]) / r["name"], extract.WINDOW)
    n, s = win["volumes"], movie.shape
    base = np.zeros((len(trials), s["z"], s["y"], s["x"]), np.float32)
    frames = np.full((len(trials), n, s["z"], s["y"], s["x"]), np.nan, np.float32)
    for k, t in enumerate(trials):
        base[k] = movie.volumes(t["start"], t["onset"] - t["start"]).mean(axis=0, dtype=np.float64)
        w = movie.volumes(t["onset"], min(t["end"] + 1, t["onset"] + n) - t["onset"])
        frames[k, :len(w)] = w
    (CACHE / fly).mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / fly / f"{r['name']}.npz", base=base, frames=frames, n=n, shape=[s["t"], s["z"], s["y"], s["x"]])
    (CACHE / fly / f"{r['name']}.json").write_text(json.dumps({"trials": trials, "window": {k: win.get(k) for k in ("volumes", "how")}}, default=str))


def load(fly):
    e = entries[fly]
    recs = []
    for r in e["recordings"]:
        if not (CACHE / fly / f"{r['name']}.npz").is_file():
            cache(fly, e, r)
        z = np.load(CACHE / fly / f"{r['name']}.npz")
        info = json.loads((CACHE / fly / f"{r['name']}.json").read_text())
        recs.append({"name": r["name"], "base": z["base"], "frames": z["frames"], "trials": info["trials"], "n": int(z["n"])})
    px = unmix.pixel_um(Path(e["raw_dir"]) / e["recordings"][0]["name"]) or unmix.PIXEL_UM
    return e, recs, px


def roisets(fly, e, shape):
    fly_dir = Path("/local/tensor") / e["dir"]
    pred = FIT / "results" / f"{fly}_pred_RoiSet.zip"
    own = extract.own_roiset(fly_dir / e["recordings"][0]["name"], fly_dir)
    out = {}
    for k, p in (("pred", pred), ("manual", own)):
        rois = extract.read_rois(p, shape, lambda s: None)
        # (outlines of others, as Sam's AL_plane<n>, left out: not glomeruli, and unmixing leaves them out of its model)
        outl = {i for q in unmix.layout(rois, shape[1:], 0) for i in q["outlines"]}
        if outl:
            print(f"  {k}: left out {len(outl)} outlines: {sorted({rois[i]['name'] for i in outl})}")
        out[k] = [r for i, r in enumerate(rois) if i not in outl]
    return out


def plane_masks(rois, yx):
    """Full-plane masks of each ROI-plane, and its core (eroded CORE px, if >= 8 px left)."""
    full, cores = [], []
    for r in rois:
        m = np.zeros(yx, bool)
        m[r["y0"]:r["y0"] + r["mask"].shape[0], r["x0"]:r["x0"] + r["mask"].shape[1]] = r["mask"]
        c = ndi.binary_erosion(m, iterations=CORE)
        full.append(m)
        cores.append(c if c.sum() >= 8 else None)
    return full, cores


def neighbours(rois, full, cores):
    """Per ROI-plane: its nearest neighbour with a core (most pixels in its NEAR px halo), and all of those."""
    nb, alln = [], []
    for i, r in enumerate(rois):
        halo = ndi.binary_dilation(full[i], iterations=NEAR) & ~full[i]
        cand = [(int((halo & full[j]).sum()), j) for j, q in enumerate(rois) if j != i and q["z"] == r["z"] and cores[j] is not None]
        cand = [c for c in cand if c[0] > 0]
        nb.append(max(cand)[1] if cand else None)
        alln.append([j for _, j in cand])
    return nb, alln


def footprints(planes, maps, s, px):
    bg = s.get("background", unmix.BACKGROUND)
    reach = s.get("reach", unmix.NMF_REACH_UM) / px
    info = {}
    if s["variant"] == "blur":
        if "sigma_um" in s:
            sig = s["sigma_um"] / px
        else:
            fit = unmix.fit_blur(planes, maps, [x / px for x in unmix.SIGMAS_UM], bg)
            sig = fit["sigma_px"]
            info = {"sigma_um": sig * px, "residuals": np.round(np.array(fit["residuals"]) / fit["residuals"][0], 4).tolist()}
        return [unmix.blur(p, sig) for p in planes], info
    out, its, rims = [], [], []
    for p, D in zip(planes, maps):
        extra = unmix.rims(p) if s["variant"] == "nmf_rims" else None
        A, i = unmix.nmf(p, D, reach, bg, iters=s.get("iters", unmix.NMF_ITERS), extra=extra)
        out.append(A)
        its.append(i["iterations"])
        rims.append(0 if extra is None else len(extra))
    return out, {"iterations": its, **({"rims": rims} if s["variant"] == "nmf_rims" else {})}


def maxabs(x):
    """sign-preserving max abs over axis 1 (NaN-skipping), as extract.responses."""
    mag = np.where(np.isnan(x), -np.inf, np.abs(x))
    return np.take_along_axis(x, mag.argmax(axis=1)[:, None], axis=1)[:, 0]


def responses(recs, planes, rois, cores, Wun):
    """Per (recording, odor): each ROI-plane's trial-mean response, plain, unmixed and of its core."""
    K = len(rois)
    out = {"plain": [], "unmixed": [], "core": []}
    for rec in recs:
        n_tr = len(rec["trials"])
        per = {k: np.full((n_tr, K), np.nan) for k in out}
        for p, Wu in zip(planes, Wun):
            y0, y1, x0, x1 = p["box"]
            idx = p["idx"]
            b = rec["base"][:, p["z"], y0:y1, x0:x1].reshape(n_tr, -1)[:, idx].astype(np.float64)
            f = rec["frames"][:, :, p["z"], y0:y1, x0:x1].reshape(n_tr, rec["frames"].shape[1], -1)[:, :, idx].astype(np.float64)
            d = f - b[:, None]
            M = unmix._flat(p, p["masks"].astype(float))
            Wp = (M / M.sum(0)).T
            C = np.zeros_like(Wp)
            has = []
            for j, i in enumerate(p["members"]):
                if cores[i] is not None:
                    c = cores[i][y0:y1, x0:x1].ravel()[idx].astype(float)
                    C[j] = c / c.sum()
                    has.append(j)
            p0, c0 = b @ Wp.T, b @ C.T
            for key, W, base in (("plain", Wp, p0), ("unmixed", Wu, p0), ("core", C, c0)):
                r = np.stack([maxabs((d[:, :, :] @ W.T)[:, :, j] / base[:, None, j]) for j in range(len(p["members"]))], axis=1)
                if key == "core":
                    r[:, [j for j in range(len(p["members"])) if j not in has]] = np.nan
                per[key][:, p["members"]] = r
        odors = {}
        for k, t in enumerate(rec["trials"]):
            odors.setdefault(tuple(t["odors"]), []).append(k)
        for key in out:
            out[key].append(np.stack([np.nanmean(per[key][ks], axis=0) for ks in odors.values()]))
    return {k: np.concatenate(v) for k, v in out.items()}


def share(v, c, n):
    (a, b), _ = nnls(np.stack([c, n], 1), v)
    return b / (a + b + 1e-9)


def metrics(R, rois, nb, alln, refused):
    P, U, C = R["plain"], R["unmixed"], R["core"]
    m = {}
    a = [(o, i) for i in range(len(rois)) if nb[i] is not None for o in range(len(P)) if C[o, i] < THR / 2 and C[o, nb[i]] > THR]
    if a:
        o, i = map(np.array, zip(*a))
        m.update(a_n=len(a), a_plain=P[o, i].mean(), a_unmixed=U[o, i].mean(), a_core=C[o, i].mean(),
                 a_plain_resp=100 * (P[o, i] > THR).mean(), a_unmixed_resp=100 * (U[o, i] > THR).mean())
    q = [(o, i) for i in range(len(rois)) if alln[i] for o in range(len(P)) if all(C[o, j] < THR / 2 for j in alln[i])]
    o, i = map(np.array, zip(*q))
    m.update(b_n=len(q), b_slope=np.polyfit(P[o, i], U[o, i], 1)[0])
    big = np.abs(P[o, i]) > THR
    m.update(b_n_big=int(big.sum()), b_slope_big=np.polyfit(P[o, i][big], U[o, i][big], 1)[0] if big.sum() > 2 else np.nan,
             b_ratio_big=np.median(U[o, i][big] / P[o, i][big]) if big.any() else np.nan)
    m.update(c_plain=int((P < -THR).sum()), c_unmixed=int((U < -THR).sum()), c_new=int(((U < -THR) & ~(P < -THR)).sum()),
             d_plain=100 * (P > THR).mean(), d_unmixed=100 * (U > THR).mean(), pairs=P.size, refused=int(refused.sum()))
    sh = [(share(P[:, i], C[:, i], C[:, nb[i]]), share(U[:, i], C[:, i], C[:, nb[i]])) for i in range(len(rois)) if nb[i] is not None and C[:, i].sum() == C[:, i].sum()]
    m.update(share_plain=float(np.median([x[0] for x in sh])), share_unmixed=float(np.median([x[1] for x in sh])))
    return m


def matched(pr, mr, pfull, mfull):
    out = []
    for i, r in enumerate(pr):
        for j, q in enumerate(mr):
            if q["z"] == r["z"]:
                inter = (pfull[i] & mfull[j]).sum()
                if inter and inter / (pfull[i] | mfull[j]).sum() >= 0.5:
                    out.append((i, j))
    return out


def main():
    names = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else list(SETTINGS)
    flies = sys.argv[3].split(",") if len(sys.argv) > 3 else FLIES
    results = {}
    for fly in flies:
        e, recs, px = load(fly)
        Z, Y, X = recs[0]["base"].shape[1:]
        sets = roisets(fly, e, (Z, Y, X))
        geo = {}
        for kind, rois in sets.items():
            full, cores = plane_masks(rois, (Y, X))
            nb, alln = neighbours(rois, full, cores)
            geo[kind] = (full, cores, nb, alln)
        pairs = matched(sets["pred"], sets["manual"], geo["pred"][0], geo["manual"][0])
        print(f"{fly}: pixel {px} um; ROI-planes pred {len(sets['pred'])}, manual {len(sets['manual'])}, matched {len(pairs)}", flush=True)
        for name in names:
            s = SETTINGS[name]
            R2 = {}
            for kind, rois in sets.items():
                t0 = time.time()
                full, cores, nb, alln = geo[kind]
                planes = unmix.layout(rois, (Y, X), s.get("reach", unmix.REGION_UM[s["variant"]]) / px)
                maps = []
                for p in planes:
                    y0, y1, x0, x1 = p["box"]
                    ds = [(np.nanmean(rec["frames"][:, :, p["z"], y0:y1, x0:x1], axis=1) - rec["base"][:, p["z"], y0:y1, x0:x1])
                          .reshape(len(rec["trials"]), -1)[:, p["idx"]].T for rec in recs]
                    maps.append(np.concatenate(ds, axis=1).astype(np.float32))
                A, info = footprints(planes, maps, s, px)
                Wun, refused, conds, vifs = [], np.zeros(len(rois), bool), [], []
                for p, a in zip(planes, A):
                    sol = unmix.solve(p, a, s.get("background", unmix.BACKGROUND))
                    W = sol["weights"].copy()
                    M = unmix._flat(p, p["masks"].astype(float))
                    W[sol["refused"]] = (M / M.sum(0)).T[sol["refused"]]
                    refused[p["members"]] = sol["refused"]
                    Wun.append(W)
                    conds.append(sol["cond"])
                    vifs.extend(sol["vif"].tolist())
                R = responses(recs, planes, rois, cores, Wun)
                R2[kind] = R
                m = metrics(R, rois, nb, alln, refused)
                m.update(info=info, cond_max=float(max(conds)), vif_median=float(np.median(vifs)), vif_max=float(max(vifs)), seconds=time.time() - t0)
                results.setdefault(name, {}).setdefault(fly, {})[kind] = m
                print(f"  {name:14s} {kind:6s} " + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items() if k != "info"),
                      info, flush=True)
            if pairs:
                i, j = map(np.array, zip(*pairs))
                d = {f"{k}_{w}": 100 * (R2[k][w][:, i if k == "pred" else j] > THR).mean() for k in ("pred", "manual") for w in ("plain", "unmixed")}
                results[name][fly]["matched"] = {"n": len(pairs), **d}
                print(f"  {name:14s} matched {len(pairs)}: " + " ".join(f"{k}={v:.1f}" for k, v in d.items()), flush=True)
    out = CACHE / f"results_{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    print("written", out)
    table(results)


def table(results):
    """The results pooled over the flies, as a Markdown table: (a) means weighted by pairs, (b) medians over flies, (c) sums,
    (d) fractions of all pairs."""
    cols = ["a n", "a plain", "a unmixed", "a core", "b slope", "b ratio", "c plain", "c unmixed", "c new", "d plain", "d unmixed", "refused"]
    print("| setting | ROIs | " + " | ".join(cols) + " | matched, over THR: pred plain / unmixed, manual plain / unmixed |")
    print("|" + "---|" * (len(cols) + 3))
    for name, flies in results.items():
        rows = list(flies.values())
        for kind in ("pred", "manual"):
            rs = [r[kind] for r in rows]
            n_a, pairs = sum(r.get("a_n", 0) for r in rs), sum(r["pairs"] for r in rs)
            w = lambda k: sum(r.get(k, 0) * r.get("a_n", 0) for r in rs) / n_a if n_a else np.nan  # noqa: E731
            p = {"a n": n_a, "a plain": w("a_plain"), "a unmixed": w("a_unmixed"), "a core": w("a_core"),
                 "b slope": np.median([r["b_slope_big"] for r in rs]), "b ratio": np.median([r["b_ratio_big"] for r in rs]),
                 **{f"c {k}": sum(r[f"c_{k}"] for r in rs) for k in ("plain", "unmixed", "new")},
                 **{f"d {k}": sum(r[f"d_{k}"] * r["pairs"] for r in rs) / pairs for k in ("plain", "unmixed")},
                 "refused": sum(r["refused"] for r in rs)}
            m = ""
            ms = [r["matched"] for r in rows if "matched" in r]
            if kind == "pred" and ms:
                f = {k: sum(x[k] * x["n"] for x in ms) / sum(x["n"] for x in ms) for k in ("pred_plain", "pred_unmixed", "manual_plain", "manual_unmixed")}
                m = f"{f['pred_plain']:.1f} / {f['pred_unmixed']:.1f}, {f['manual_plain']:.1f} / {f['manual_unmixed']:.1f}"
            print(f"| {name} | {kind} | " + " | ".join(f"{p[c]:.3g}" if isinstance(p[c], float) else str(p[c]) for c in cols) + f" | {m} |")


if __name__ == "__main__":
    main()
