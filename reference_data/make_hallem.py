#!/usr/bin/env python3
"""Makes hallem_carlson_2006.csv, the reference dataset of Hallem & Carlson (2006): every
ORN response it tabulates, as changes in spike rate (spikes/s, from the spontaneous rate),
glomeruli x odors (written as datasets write them, ``<odor> @ <log10 dilution>``):

- its 110 odors at 10^-2 (Table S1), named as olfsysm's
  ``pyolfsysm/data/preprocessed_hallem/hallem_orn_deltas.csv`` names them (the lab's
  abbreviations where it has one; but `RENAMED`), whose glomeruli are used too (DM3 its
  Or47a, not Or33b);
- its supplemental data, from olfsysm's ``hc_data.csv``: 10 of those odors at 10^-4, 10^-6
  and 10^-8 (their 10^-2 there is Table S1's), and 9 fruits, pure (``@ 0``) and at 10^-2,
  10^-4 and 10^-6.

Usage: make_hallem.py <an olfsysm checkout>
"""
import csv
import sys
from pathlib import Path

OUT = Path(__file__).with_name("hallem_carlson_2006.csv")
N_ODORS = 110  # of Table S1, the first rows of hc_data.csv after its two of headers
# names of that table that are not the lab's abbreviations (the odor table's): 1-octen-3-ol
RENAMED = {"oct": "1o3ol"}
DROPPED = "33b"  # the second receptor of DM3 (Or47a's is kept, as in the preprocessed table)


def main(olfsysm: Path) -> None:
    pre = list(csv.reader(open(olfsysm / "pyolfsysm/data/preprocessed_hallem/hallem_orn_deltas.csv")))
    odors = [f"{RENAMED.get(name, name)} @ {conc}" for name, conc in (o.rsplit(" @ ", 1) for o in pre[0][1:])]
    table = {r[0]: [float(x) for x in r[1:]] for r in pre[1:]}
    hc = list(csv.reader(open(olfsysm / "hc_data.csv")))
    receptors, main_rows, rest = hc[1][2:], hc[2:2 + N_ODORS], hc[2 + N_ODORS:]
    assert len(odors) == N_ODORS == len(main_rows)
    # each receptor's glomerulus: that of the preprocessed table with its responses
    glom_of = {}
    for j, rec in enumerate(receptors):
        col = [float(r[2 + j]) for r in main_rows]
        found = [g for g, v in table.items() if v == col]
        assert len(found) == (0 if rec == DROPPED else 1), (rec, found)
        if found:
            glom_of[j] = found[0]
    name_of = {r[1]: o.rsplit(" @ ", 1)[0] for r, o in zip(main_rows, odors)}
    columns = {o: {g: v[i] for g, v in table.items()} for i, o in enumerate(odors)}
    for r in rest:
        kind, label = r[0], r[1]
        if kind == "0":  # (the spontaneous firing rates: the responses are changes from them)
            continue
        name, conc = label.rsplit(" ", 1)
        name, conc = (name_of[name], conc) if kind == "11" else (name, "0" if conc == "pure" else conc)
        key = f"{name} @ {conc}"
        values = {glom_of[j]: float(r[2 + j]) for j in glom_of}
        if key in columns:
            assert columns[key] == values, key  # (the lower concentrations' table has 10^-2 again)
            continue
        columns[key] = values
    glomeruli = sorted(table)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["glomerulus", *columns])
        for g in glomeruli:
            w.writerow([g, *(f"{columns[o][g]:g}" for o in columns)])
    print(f"{OUT.name}: {len(glomeruli)} glomeruli x {len(columns)} odors")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
