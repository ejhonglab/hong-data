"""What the try-out's candidates (run.py's rows.json) lie on: a manual outline the predicted ROIs did not match ("hit": IoU >= 0.3 with
it, and under 0.5 between it and every predicted mask), one they did ("dup"), inside manual outlines but matching none ("frag"), or
where none is drawn ("off"); all of them, by each test alone, and by those al-diag-seg's alseg.nmf.find applies ("FINAL"); and of
the masks cut in two, how many hold two manual outlines, by each measure.

Usage: score.py <out>/<tag>/rows.json [-s: the FINAL rows only] [-v: every candidate]"""
import json
import sys

import numpy as np

rows = json.loads(open(sys.argv[1]).read())
man = [r for r in rows if r["kind"] == "manual"]
adds = [r for r in rows if r["kind"] == "add"]
spl = [r for r in rows if r["kind"] == "split"]
preds = [r for r in rows if r["kind"] == "pred"]
missed = [r for r in man if r["m_pred_iou"] < 0.2]
partial = [r for r in man if 0.2 <= r["m_pred_iou"] < 0.5]
found = [r for r in man if r["m_pred_iou"] >= 0.5]
print(f"manual ROI-planes: {len(man)} (found {len(found)}, partial {len(partial)}, missed {len(missed)}); manual inscribed radius um 5/50%: "
      f"{np.percentile([r['inscribed_um'] for r in man], [5, 50]).round(1)}, area um2 5/50/95%: {np.percentile([r['area_um2'] for r in man], [5, 50, 95]).round(0)}")


def cls(r):
    if r["m_iou"] >= 0.3 and r["m_pred_iou"] < 0.5:
        return "hit"       # on a manual ROI the prediction did not find
    if r["m_iou"] >= 0.3:
        return "dup"       # on a manual ROI already found
    if r["in_manual"] >= 0.5:
        return "frag"      # inside manual ROIs, matching none
    return "off"           # where nothing is drawn


for r in adds:
    r["cls"] = cls(r)


def table(sel, name):
    n = {k: sum(r["cls"] == k for r in sel) for k in ("hit", "dup", "frag", "off")}
    names = {(r["fly"], r["z"], r["m_name"]) for r in sel if r["cls"] == "hit"}
    print(f"  {name:34s}: {len(sel):3d}  hit {n['hit']:3d} ({len(names)} manual; IoU>=0.5: {sum(r['cls'] == 'hit' and r['m_iou'] >= 0.5 for r in sel):3d})  "
          f"dup {n['dup']:3d}  frag {n['frag']:3d}  off {n['off']:3d}")


print(f"additions: {len(adds)}")
for nv in (0.3, 0.5):
    table([r for r in adds if r.get("hull3", 1) >= 0.8 and r["eta2"] >= 0.5 and r["inscribed_um"] >= 2.5 and r["novel"] >= nv and r["area_um2"] >= 20], f"FINAL novel>={nv}")
if "-s" in sys.argv:
    sys.exit()
table(adds, "all")
for thr in (0.3, 0.4, 0.5, 0.6):
    table([r for r in adds if r["eta2"] >= thr], f"eta2 >= {thr}")
for thr in (2.0, 2.5, 3.0):
    table([r for r in adds if r["inscribed_um"] >= thr], f"inscribed >= {thr} um")
for thr in (0.9, 0.8, 0.7):
    table([r for r in adds if r["nb_corr"] < thr], f"nb_corr < {thr}")
for thr in (0.3, 0.5):
    table([r for r in adds if r["f0"] >= thr], f"f0 >= {thr}")
for thr in (0.1, 0.2, 0.3, 0.5):
    table([r for r in adds if r["novel"] >= thr], f"novel >= {thr}")
for thr in (0.2, 0.4, 0.6):
    table([r for r in adds if r["novel_t"] >= thr], f"novel_t >= {thr}")
for e, i, nv, a in ((0.4, 2.0, 0.2, 15), (0.4, 2.0, 0.3, 15), (0.5, 2.0, 0.3, 15), (0.5, 2.5, 0.3, 20), (0.5, 2.5, 0.5, 20), (0.6, 2.5, 0.3, 20)):
    table([r for r in adds if r["eta2"] >= e and r["inscribed_um"] >= i and r["novel"] >= nv and r["area_um2"] >= a], f"eta2>={e} insc>={i} novel>={nv} a>={a}")
for thr in (5, 10, 15):
    table([r for r in adds if r["order"] < thr], f"order < {thr}")
for e, i, c, a in ((0.4, 2.0, 0.9, 15), (0.4, 2.5, 0.9, 20), (0.5, 2.5, 0.9, 20), (0.5, 2.5, 0.8, 20), (0.5, 3.0, 0.8, 25), (0.6, 2.5, 0.8, 20)):
    table([r for r in adds if r["eta2"] >= e and r["inscribed_um"] >= i and r["nb_corr"] < c and r["area_um2"] >= a], f"eta2>={e} insc>={i} nbc<{c} area>={a}")
merges = [r for r in preds if r["held"] >= 2]
print(f"predicted masks: {len(preds)}, over two manual ROIs: {len(merges)}; splits proposed: {len(spl)}")


def stab(sel, name):
    print(f"  {name:40s}: {len(sel):3d}  of merges {sum(r['held'] >= 2 for r in sel):2d} (parts on two manual ROIs: {sum(r['held'] >= 2 and r['right'] for r in sel):2d})  "
          f"of masks over one {sum(r['held'] == 1 for r in sel):3d}  over none {sum(r['held'] == 0 for r in sel):3d}")


stab(spl, "all")
for key in ("corr2", "trace_corr", "ccorr", "tcorr", "rcorr", "tuncorr"):
    for thr in (0.6, 0.4, 0.2, 0.0):
        stab([r for r in spl if r[key] < thr], f"{key} < {thr}")
for thr in (0.05, 0.1, 0.2):
    stab([r for r in spl if r["gain"] >= thr], f"gain >= {thr}")
base = [r for r in spl if min(r["eta2"], r["eta2_other"]) >= 0.4 and r["connected"] >= 0.9]
stab(base, "eta both >= 0.4, connected >= 0.9")
for key in ("corr2", "ccorr", "tcorr", "rcorr", "tuncorr"):
    for thr in (0.4, 0.2, 0.0):
        stab([r for r in base if r[key] < thr], f"  + {key} < {thr}")
print("  merges:", sorted((round(r["tuncorr"], 2), round(r["tcorr"], 2), round(r["rcorr"], 2), round(r["gain"], 2), r["right"]) for r in spl if r["held"] >= 2))
if "-v" in sys.argv:
    for r in sorted(adds, key=lambda r: (r["fly"], r["z"], r["order"])):
        print(f"  {r['fly'][:14]:14s} z{r['z']} #{r['order']:2d} k{r['k']:3d} {r['area_um2']:6.1f} um2 insc {r['inscribed_um']:.1f} peak {r['peak']:.2f} eta2 {r['eta2']:.2f} nbc {r['nb_corr']:.2f} "
              f"f0 {r['f0']:.2f} nov {r['novel']:.2f}/{r['novel_t']:.2f} amp {r['amp']:.2f} | {r['cls']:4s} manual {str(r['m_name']):8s} IoU {r['m_iou']:.2f} (pred {r['m_pred_iou']:.2f}) in_manual {r['in_manual']:.2f}")
    for r in sorted(spl, key=lambda r: (r["fly"], r["z"])):
        print(f"  {r['fly'][:14]:14s} z{r['z']} split {r['of']:3d} frac {r['frac']:.2f} corr2 {r['corr2']:.2f} tr {r['trace_corr']:.2f} cc {r['ccorr']:.2f} tc {r['tcorr']:.2f} gain {r['gain']:.3f} conn {r['connected']:.2f} eta {r['eta2']:.2f}/{r['eta2_other']:.2f} "
              f"| held {r['held']} parts {r['parts']} right {r['right']}")
