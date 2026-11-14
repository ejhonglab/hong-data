"""What a trial's F0 (the mean of its time points before the odor) costs, and whether anything else
could stand in for it: on real flies, of each ROI's best plane (as al_analysis chooses it). What
made the numbers of finding 4 of trace_extraction_notes.md (beside this), run as
trace_extraction_flaws.py is:

    PYTHONPATH=. OLFDB_DATA_DIR=<a scratch folder> python notes/trace_extraction_baseline.py \\
        <a fly's motion-corrected data folder>..."""
import sys
from pathlib import Path

import numpy as np

from app import extract, movies

for arg in sys.argv[1:]:
    fly = Path(arg)
    print(f"\n===== {fly}")
    sigma, f0_ratio, f0_ratio_diff, tail, tail_strong, tail_sd, n_pre, n_post, drift = [], [], [], [], [], [], [], [], []
    resp_sd = {k: [] for k in ("own", "smooth", "scale")}
    resp_mean = []
    for r in extract.recordings(fly):
        path = extract.own_roiset(r["dir"], fly)
        if path is None:
            continue
        movie = movies.open_movie(r["movie"])
        s = movie.shape
        rois = extract.read_rois(path, (s["z"], s["y"], s["x"]), lambda x: None)
        trials = extract.read_trials(r["trials"])
        rec = {"traces": extract.plane_traces(movie, rois), "trials": trials}
        extract.combine("best_plane", rois, [rec])
        tr = rec["roi_traces"]
        n = np.array([t["onset"] - t["start"] for t in trials])
        n_pre += list(n)
        n_post += [t["end"] - t["onset"] + 1 for t in trials]
        f0 = np.stack([tr[t["start"]:t["onset"]].mean(axis=0) for t in trials])  # (trials, ROIs)
        scale = f0.mean(axis=0)
        # the noise of one time point, relative to F0: from the time points before each odor, about their mean
        ss = sum((((tr[t["start"]:t["onset"]] - f0[i]) / scale) ** 2).sum(axis=0) for i, t in enumerate(trials))
        var = ss / (n - 1).sum()
        sigma += list(np.sqrt(var))
        # how much F0 moves from trial to trial, against what its own noise would move it
        expected = var * np.mean(1 / n)
        f0_ratio += list((f0 / scale).var(axis=0, ddof=1) / expected)
        f0_ratio_diff += list((np.diff(f0 / scale, axis=0) ** 2).mean(axis=0) / 2 / expected)
        drift.append(np.median(f0[-3:].mean(axis=0) / f0[:3].mean(axis=0)))
        # the end of the block before (its last two time points) against this trial's F0
        resp = extract.responses(tr, trials)
        for i, t in enumerate(trials):
            if i == 0:
                continue
            before = tr[t["start"] - 2:t["start"]].mean(axis=0)
            d = (before - f0[i]) / f0[i]
            strong = resp[i - 1] > 0.5
            tail += list(d[~strong])
            tail_strong += list(d[strong])
        tail_sd.append(np.sqrt(np.median(var) * (1 / 2 + np.mean(1 / n))))
        # responses with other F0s, each trial's still subtracted: how much they vary between repeats of an odor
        k = 2  # (trials either side)
        smooth = np.stack([np.median(f0[max(0, i - k):i + k + 1], axis=0) for i in range(len(trials))])
        odors = [tuple(t["odors"]) for t in trials]
        variants = {"own": lambda i: (f0[i], f0[i]), "smooth": lambda i: (smooth[i], smooth[i]), "scale": lambda i: (f0[i], smooth[i])}
        for name, pick in variants.items():
            out = np.empty_like(resp)
            for i, t in enumerate(trials):
                sub, div = pick(i)
                window = (tr[t["onset"]:t["onset"] + extract.N_RESPONSE] - sub) / div
                out[i] = window[np.abs(window).argmax(axis=0), np.arange(tr.shape[1])]
            sds, means = [], []
            for o in dict.fromkeys(odors):
                rows = [i for i, x in enumerate(odors) if x == o]
                if len(rows) >= 3:
                    sds.append(out[rows].std(axis=0, ddof=1))
                    means.append(out[rows].mean(axis=0))
            if sds:
                sds, means = np.stack(sds), np.stack(means)
                if name == "own":
                    strong_mask = means > 0.3
                    resp_mean += list(means[strong_mask])
                resp_sd[name] += list(sds[strong_mask])
    sigma = np.array(sigma)
    print(f"time points before the odor: {sorted(set(n_pre))}; from its onset on: {sorted(set(n_post))}")
    print(f"noise of one time point, of F0: median {np.median(sigma):.3f} (10th to 90th percentile {np.percentile(sigma, 10):.3f} to {np.percentile(sigma, 90):.3f})")
    print(f"F0 from trial to trial, as a multiple of what the noise of its {np.median(n_pre):.0f} time points would give: variance {np.median(f0_ratio):.1f} x "
          f"(median over ROIs; 10th to 90th percentile {np.percentile(f0_ratio, 10):.1f} to {np.percentile(f0_ratio, 90):.1f}); between successive trials alone "
          f"{np.median(f0_ratio_diff):.1f} x ({np.percentile(f0_ratio_diff, 10):.1f} to {np.percentile(f0_ratio_diff, 90):.1f}); F0 of a recording's last three "
          f"trials over its first three: {np.min(drift):.2f} to {np.max(drift):.2f}")
    tail, tail_strong = np.array(tail), np.array(tail_strong)
    print(f"the last two time points of the block before, against this trial's F0: {100 * np.median(tail):+.1f}% (median; s.d. {100 * tail.std():.1f}%, "
          f"where noise alone would give {100 * np.median(tail_sd):.1f}%); after a response over 0.5: {100 * np.median(tail_strong):+.1f}% "
          f"(10th to 90th percentile {100 * np.percentile(tail_strong, 10):+.1f}% to {100 * np.percentile(tail_strong, 90):+.1f}%; n = {len(tail_strong)})")
    own = np.array(resp_sd["own"])
    print(f"s.d. of a response between an odor's repeats (mean response over 0.3; n = {len(own)}; median response {np.median(resp_mean):.2f}): "
          f"each trial's own F0 {np.median(own):.3f}; F0 the median of 5 trials' {np.median(resp_sd['smooth']):.3f} "
          f"({np.median(np.array(resp_sd['smooth']) / own):.2f} x, ROI by ROI); own F0 subtracted, divided by the median of 5 "
          f"{np.median(resp_sd['scale']):.3f} ({np.median(np.array(resp_sd['scale']) / own):.2f} x)")
