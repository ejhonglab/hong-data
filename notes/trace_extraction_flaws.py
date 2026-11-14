"""How much what is questionable in al_analysis' response calculation matters, on real flies:
what made the numbers of trace_extraction_notes.md (beside this; al_analysis' BUGS.md is the
same note), with app/extract.py's functions, which compute what al_analysis does.

Usage, from the top of this repository (with the flies' folders readable, as in the site's
image with tensor mounted):

    PYTHONPATH=. OLFDB_DATA_DIR=<a scratch folder> python notes/trace_extraction_flaws.py \\
        <a fly's motion-corrected data folder>...

e.g. /local/tensor/Tom/2p_data/analysis_intermediates/2023-05-08/1. Each fly takes about a
minute (its movies are read once)."""
import sys
from pathlib import Path

import numpy as np
import roifile

from app import extract, movies

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def imagej_rois(path, shape):
    """The same ROIs as extract.read_rois, with ImageJ's own masks (pixels whose centre is inside)."""
    out = []
    for i, r in enumerate(roifile.ImagejRoi.fromfile(path)):
        name = r.name or str(i + 1)
        if name.endswith("+") or name == "AL":
            continue
        z = r.z_position - 1 if r.z_position > 0 else r.position - 1
        p = np.asarray(r.coordinates(), float)
        yy, xx = np.mgrid[0:shape[1], 0:shape[2]]
        m = extract.inside(p, xx + 0.5, yy + 0.5)
        rev = extract.polygon_mask(p[::-1], shape[1:])
        out.append({"name": name, "z": z, "y0": 0, "x0": 0, "mask": m, "pixels": int(m.sum()), "points": p, "reversed": rev})
    return out


def full(r, shape):
    out = np.zeros(shape, bool)
    out[r["y0"]:r["y0"] + r["mask"].shape[0], r["x0"]:r["x0"] + r["mask"].shape[1]] = r["mask"]
    return out


for arg in sys.argv[1:]:
    fly = Path(arg)
    print(f"\n===== {fly}")
    recs = extract.recordings(fly)
    data = []
    for r in recs:
        path = extract.own_roiset(r["dir"], fly)
        if path is None:
            continue
        movie = movies.open_movie(r["movie"])
        s = movie.shape
        shape = (s["z"], s["y"], s["x"])
        rois = extract.read_rois(path, shape, lambda x: None)
        ij = imagej_rois(path, shape)
        trials = extract.read_trials(r["trials"])
        both = extract.plane_traces(movie, rois + ij)
        data.append({"name": r["name"], "rois": rois, "ij": ij, "trials": trials, "traces": both[:, :len(rois)], "ij_traces": both[:, len(rois):],
                     "fps": movie.fps, "shape": s})
    rois, ij = data[0]["rois"], data[0]["ij"]
    shape2 = (data[0]["shape"]["y"], data[0]["shape"]["x"])

    # 1. masks: hong2p's against ImageJ's own, and against the same outline drawn the other way round
    diff, rev, sizes = [], [], []
    for a, b in zip(rois, ij):
        ma, mb = full(a, shape2), b["mask"]
        diff.append((ma ^ mb).sum() / mb.sum())
        ry0, rx0, rm = b["reversed"]
        mr = np.zeros(shape2, bool)
        mr[ry0:ry0 + rm.shape[0], rx0:rx0 + rm.shape[1]] = rm
        rev.append((ma ^ mr).sum())
        sizes.append(mb.sum())
    print(f"1. masks ({len(rois)} outlines, median {np.median(sizes):.0f} px): pixels in one of hong2p's and ImageJ's masks but not the other, of ImageJ's: "
          f"median {100 * np.median(diff):.1f}%, largest {100 * np.max(diff):.1f}%; outlines whose mask would differ drawn the other way round: "
          f"{np.mean(np.array(rev) > 0) * 100:.0f}% (by {np.median([v for v in rev if v] or [0]):.0f} px, median)")
    r_all, rel = [], []
    for d in data:
        a = extract.responses(d["traces"], d["trials"])
        b = extract.responses(d["ij_traces"], d["trials"])
        r_all.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        big = np.abs(b) > 0.2
        rel.append(np.abs(a[big] - b[big]) / np.abs(b[big]))
    rel = np.concatenate(rel)
    print(f"   responses with either mask: r = {np.min(r_all):.4f} to {np.max(r_all):.4f} over recordings; of responses over 0.2, they differ by "
          f"{100 * np.median(rel):.1f}% (median), {100 * np.percentile(rel, 95):.1f}% (95th percentile), {100 * rel.max():.0f}% (most)")

    # 2. the best plane of an ROI, from one recording to another
    names = [r["name"] for r in rois]
    multi = sorted({n for n in names if names.count(n) > 1})
    for d in data:
        d["q"] = extract.quality(extract.responses(d["traces"], d["trials"]))
        d["best"] = {names[i]: i for i in extract.best_planes(names, d["q"])}
    changes = [len({d["best"][n] for d in data}) for n in multi]
    print(f"2. best planes: {len(multi)} of {len(set(names))} ROIs are drawn on several planes; of those, {np.mean(np.array(changes) > 1) * 100:.0f}% "
          f"are read on different planes in different recordings of the fly ({len(data)} recordings; up to {max(changes)} planes)")
    # how much the choice matters: the response on the plane of the recording, against that of the fly's plane
    stacked = np.stack([d["q"] for d in data])
    fly_best = {names[i]: i for i in extract.best_planes(names, np.nanmax(stacked, axis=0))}
    ratio = []
    for d in data:
        resp = extract.responses(d["traces"], d["trials"])
        for n in multi:
            if d["best"][n] != fly_best[n]:
                a, b = resp[:, d["best"][n]], resp[:, fly_best[n]]
                strong = np.abs(b) > 0.2
                if strong.any():
                    ratio.append(np.median(a[strong] / b[strong]))
    if ratio:
        print(f"   where a recording's plane is not the fly's: its responses over 0.2 are {np.median(ratio):.2f} x those on the fly's plane "
              f"(median over {len(ratio)} ROI-recordings; range {np.min(ratio):.2f} to {np.max(ratio):.2f})")

    # 3. choosing the plane by the responses it is used for: chosen on some repeats, measured on the others
    inflate = []
    for d in data:
        resp = extract.responses(d["traces"], d["trials"])
        odors = [tuple(t["odors"]) for t in d["trials"]]
        seen, rep = {}, []
        for o in odors:
            rep.append(seen.get(o, 0))
            seen[o] = rep[-1] + 1
        rep = np.array(rep)
        if rep.max() < 2:
            continue
        for held in range(3):
            fit, test = rep != held, rep == held
            for n in multi:
                g = [i for i, x in enumerate(names) if x == n]
                q_fit = np.nanmax(resp[fit][:, g], axis=0)
                chosen = g[int(np.argmax(q_fit))]
                # the chosen plane's largest response on the trials it was chosen by, and on the others
                inflate.append((np.nanmax(resp[fit][:, chosen]), np.nanmax(resp[test][:, chosen]),
                                np.nanmax(resp[fit][:, g].mean(axis=1)), np.nanmax(resp[test][:, g].mean(axis=1))))
    if inflate:
        inf = np.array(inflate)
        strong = inf[:, 3] > 0.2
        print(f"3. selection: an ROI's largest response on its chosen plane is {np.median(inf[strong, 0] / inf[strong, 2]):.2f} x the mean of its planes' on the "
              f"trials the plane was chosen by, and {np.median(inf[strong, 1] / inf[strong, 3]):.2f} x on trials it was not (ROIs responding over 0.2; "
              f"the difference is what choosing by the same trials adds)")

    # 4. the response window: where dF/F peaks after the odor comes on
    peaks, lost, fps = [], [], data[0]["fps"]
    for d in data:
        tr = d["traces"][:, [d["best"][n] for n in dict.fromkeys(names)]]
        for t in d["trials"]:
            base = tr[t["start"]:t["onset"]].mean(axis=0)
            dff = (tr[t["onset"]:t["end"] + 1] - base) / base
            two = np.abs(dff[:2]).max(axis=0)
            whole = np.abs(dff).max(axis=0)
            strong = whole > 0.3
            peaks += list(np.abs(dff).argmax(axis=0)[strong])
            lost += list(two[strong] / whole[strong])
    peaks = np.array(peaks)
    print(f"4. window: {fps and round(fps, 2)} volumes/s; of responses whose |dF/F| passes 0.3 anywhere in the trial, the peak is at time point "
          f"(from onset) 0: {np.mean(peaks == 0) * 100:.0f}%, 1: {np.mean(peaks == 1) * 100:.0f}%, 2: {np.mean(peaks == 2) * 100:.0f}%, later: "
          f"{np.mean(peaks > 2) * 100:.0f}%; the first two time points hold {100 * np.median(lost):.0f}% of the peak (median), under 80% of it for "
          f"{np.mean(np.array(lost) < 0.8) * 100:.0f}% of them")

    # 5. the baseline: all the time points of the trial before the odor, which follow the trial before
    n_base = sorted({t["onset"] - t["start"] for d in data for t in d["trials"]})
    drift, after = [], []
    for d in data:
        tr = d["traces"][:, [d["best"][n] for n in dict.fromkeys(names)]]
        resp = extract.responses(tr, d["trials"])
        for i, t in enumerate(d["trials"]):
            before = tr[t["start"]:t["onset"]]
            if len(before) < 3 or i == 0:
                continue
            base = before.mean(axis=0)
            last2 = before[-2:].mean(axis=0)
            strong = resp[i - 1] > 0.5  # (after a strong response)
            drift += list(((base - last2) / last2)[strong])
            after += list(((base - last2) / last2)[~strong])
    print(f"5. baseline: {n_base} time points before the odor; F0 over all of them is {100 * np.median(drift):+.1f}% (median; "
          f"{100 * np.percentile(drift, 90):+.1f}% at the 90th percentile) from F0 over the last two alone after a response over 0.5 in the trial before, "
          f"{100 * np.median(after):+.1f}% after others")

    # 6. the response statistic where there is no response: on time points before the odor
    fake_max, fake_mean = [], []
    for d in data:
        tr = d["traces"][:, [d["best"][n] for n in dict.fromkeys(names)]]
        for t in d["trials"]:
            before = tr[t["start"]:t["onset"]]
            if len(before) < 4:
                continue
            base = before[:-2].mean(axis=0)
            dff = (before[-2:] - base) / base
            fake_max += list(dff[np.abs(dff).argmax(axis=0), np.arange(dff.shape[1])])
            fake_mean += list(dff.mean(axis=0))
    print(f"6. statistic: of two time points with no odor (the last two before it, against those before them), the larger in magnitude has s.d. "
          f"{np.std(fake_max):.3f}, mean magnitude {np.mean(np.abs(fake_max)):.3f}; their mean, s.d. {np.std(fake_mean):.3f}, mean magnitude "
          f"{np.mean(np.abs(fake_mean)):.3f}")
    # 7. the level F0 is of: the movie's darkest plane mean, against ROIs' F0
    low = min(float(movies.open_movie(r["movie"]).volumes(0, 20).min()) for r in recs[:1])
    f0 = np.median(data[0]["traces"][:5], axis=0)
    print(f"7. offset: the movie's lowest value in its first 20 volumes is {low:.0f}; ROIs' F0 are {np.percentile(f0, 10):.0f} to {np.percentile(f0, 90):.0f} "
          f"(10th to 90th percentile)")
