"""How well a fly's glomeruli are told apart by their tuning alone, to each panel of odors the database has of them: leaving each
fly out in turn, its glomeruli (the ROIs named in the database) are named by matching their tuning (mean response per odor, z-scored
over the odors) one to one to the other flies' mean tuning of each glomerulus (correlation; the Hungarian algorithm, over every
glomerulus the others have), and counted right where that is the name they have.
db_tuning.py <db_tuning.json: per dataset, [date, fly, roi, panel, odor, mean response, trials]>"""
import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

data = json.load(open(sys.argv[1]))
rng = np.random.default_rng(0)
DIAG = ("glomeruli_diagnostics", "diagnostics")


def z(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def lofo(flies, odors):
    """flies: {fly: {roi: {odor: value}}}. Mean share of a fly's ROIs named right, by the odors of `odors` it has."""
    accs, ns, used = [], [], []
    for f, rois in flies.items():
        have = [o for o in odors if all(o in v for v in rois.values())]
        if len(have) < 2:
            continue
        tmpl = {}
        for g, others in ((g, [fl[g] for k, fl in flies.items() if k != f and g in fl and all(o in fl[g] for o in have)])
                          for g in {g for k, fl in flies.items() if k != f for g in fl}):
            if others:
                tmpl[g] = np.mean([z(np.array([o_[o] for o in have])) for o_ in others], 0)
        mine = [g for g in rois if g in tmpl]
        if len(mine) < 2:
            continue
        names = list(tmpl)
        T = np.stack([z(tmpl[g]) for g in names])
        R = np.stack([z(np.array([rois[g][o] for o in have])) for g in mine])
        cost = 1 - (R @ T.T) / len(have)
        r, c = linear_sum_assignment(cost)
        accs.append(np.mean([names[j] == mine[i] for i, j in zip(r, c)]))
        ns.append(len(mine))
        used.append(len(have))
    return (float(np.mean(accs)), len(accs), float(np.mean(ns)), float(np.mean(used))) if accs else None


print("| dataset | odors used | flies | glomeruli a fly (mean) | odors a fly has of them (mean) | named right by tuning alone |")
print("|---|---|---|---|---|---|")
for ds, rows in data.items():
    flies, panels = {}, {}
    for date, n, roi, panel, odor, value, _ in rows:
        flies.setdefault((date, n), {}).setdefault(roi, {})[odor] = value
        panels.setdefault(panel, set()).add(odor)
    diag = sorted(o for p in panels if p in DIAG for o in panels[p])
    other = {p: sorted(panels[p]) for p in panels if p not in DIAG}
    sets = [("diagnostics, all", diag)]
    for p, od in other.items():
        sets += [(p, od), (f"diagnostics and {p}", diag + od)]
    if len(other) > 1:
        sets.append((" and ".join(other), [o for od in other.values() for o in od]))
    for name, od in sets:
        r = lofo(flies, od)
        if r:
            print(f"| {ds} | {name} ({len(od)}) | {r[1]} | {r[2]:.0f} | {r[3]:.0f} | {100 * r[0]:.0f}% |")
    for k in (10, 5, 2):
        rs = [lofo(flies, list(rng.choice(diag, k, replace=False))) for _ in range(20)]
        rs = [r for r in rs if r]
        print(f"| {ds} | {k} of the diagnostics, at random (20 draws) | {rs[0][1]} | {rs[0][2]:.0f} | {np.mean([r[3] for r in rs]):.0f} | "
              f"{100 * np.mean([r[0] for r in rs]):.0f}% ({100 * min(r[0] for r in rs):.0f} to {100 * max(r[0] for r in rs):.0f}) |")
