"""Which diagnostic odors tell glomeruli apart most (tuning alone, as db_tuning.py counts it): odors added one at a time, each the one
that names the most right, on one dataset's flies (leaving each out in turn), then those same odors on the other datasets' flies;
and how many of them a fly needs beside another panel.
db_greedy.py <db_tuning.json> [dataset to choose on (21)]"""
import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

data = json.load(open(sys.argv[1]))
PICK = sys.argv[2] if len(sys.argv) > 2 else "21"
DIAG = ("glomeruli_diagnostics", "diagnostics")


class Flies:
    def __init__(self, rows):
        self.panels = {}
        for date, n, roi, panel, odor, value, _ in rows:
            self.panels.setdefault(panel, set()).add(odor)
        self.flies = sorted({(r[0], r[1]) for r in rows})
        self.rois = sorted({r[2] for r in rows})
        self.odors = sorted({r[4] for r in rows})
        fi, gi, oi = ({k: i for i, k in enumerate(x)} for x in (self.flies, self.rois, self.odors))
        self.X = np.full((len(self.flies), len(self.rois), len(self.odors)), np.nan)
        for date, n, roi, panel, odor, value, _ in rows:
            self.X[fi[date, n], gi[roi], oi[odor]] = value
        self.oi = oi

    def diag(self):
        return sorted(o for p in self.panels if p in DIAG for o in self.panels[p])

    def lofo(self, odors):
        """Mean share of a fly's glomeruli named right by the odors of `odors` it has (None if under 2 flies can be)."""
        cols = [self.oi[o] for o in odors if o in self.oi]
        X = self.X[:, :, cols]
        accs = []
        for f in range(len(self.flies)):
            mine = np.flatnonzero(np.isfinite(X[f]).any(1))
            have = np.flatnonzero(np.isfinite(X[f][mine]).all(0)) if len(mine) else []
            if len(have) < 2:
                continue
            Xh = X[:, :, have]
            Z = (Xh - Xh.mean(2, keepdims=True)) / (Xh.std(2, keepdims=True) + 1e-8)  # (NaN where an ROI lacks one of them)
            others = np.delete(Z, f, 0)
            with np.errstate(all="ignore"):
                T = np.nanmean(others, 0)
            ok = np.isfinite(T).all(1)
            mine = [g for g in mine if ok[g]]
            if len(mine) < 2:
                continue
            names = np.flatnonzero(ok)
            T = T[names]
            T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-8)
            r, c = linear_sum_assignment(1 - (Z[f][mine] @ T.T) / len(have))
            accs.append(np.mean(names[c] == np.array(mine)[r]))
        return float(np.mean(accs)) if len(accs) >= 2 else None


sets = {ds: Flies(rows) for ds, rows in data.items()}
P = sets[PICK]
# (of the odors nearly every fly of it has: a fly is only named by odors it has)
pool = [o for o in P.diag() if np.isfinite(P.X[:, :, P.oi[o]]).any(1).mean() >= 0.9]
print(f"dataset {PICK}: {len(P.flies)} flies, {len(P.diag())} diagnostic odors, {len(pool)} of them in 9 flies of 10; those: {100 * P.lofo(pool):.0f}%")


def greedy(F, start, pool, upto):
    chosen, out = list(start), []
    while len(chosen) - len(start) < upto:
        best = max(((F.lofo(chosen + [o]) or 0, o) for o in pool if o not in chosen), default=None)
        if best is None:
            break
        chosen.append(best[1])
        out.append((best[1], best[0]))
    return out


pair = max(((P.lofo([a_, b_]) or 0, a_, b_) for i, a_ in enumerate(pool) for b_ in pool[i + 1:]))  # (two at least, for a tuning)
order = [(pair[1], float("nan")), (pair[2], pair[0])] + greedy(P, [pair[1], pair[2]], pool, len(pool) - 2)
others = [d for d in sets if d != PICK]
print(f"| odors | added | dataset {PICK} | " + " | ".join(f"dataset {d}" for d in others) + " |")
print("|---|---|---|" + "---|" * len(others))
got = []
for k, (o, acc) in enumerate(order, 1):
    got.append(o)
    cells = []
    for d in others:
        a = sets[d].lofo(got) if sum(x in sets[d].oi for x in got) >= 2 else None
        cells.append("" if a is None else f"{100 * a:.0f}% ({sum(x in sets[d].oi for x in got)})")
    print(f"| {k} | {o} | {100 * acc:.0f}% | " + " | ".join(cells) + " |")
print("\nbeside another panel: diagnostics added one at a time, each the one that names the most right there")
for d, F in sets.items():
    for p in [p for p in F.panels if p not in DIAG]:
        base = sorted(F.panels[p])
        steps = greedy(F, base, F.diag(), 6)
        print(f"dataset {d}, {p} ({len(base)} odors): {100 * (F.lofo(base) or 0):.0f}% alone; then " + ", ".join(f"+ {o}: {100 * a:.0f}%" for o, a in steps)
              + f"; with all {len(F.diag())} diagnostics {100 * (F.lofo(base + F.diag()) or 0):.0f}%")
