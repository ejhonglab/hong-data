"""Makes the AL atlas tab's meshes of the Grabe et al. 2016 in vivo atlas (grabe2016.json and grabe2016.bin, beside this
script) from its labels, as al_atlas exports them (https://github.com/ejhonglab/al_atlas, grabe2016/labels.tif and
structures.csv, at the commit the Dockerfile pins; their checksums are checked).

Usage (with numpy, scipy, scikit-image and tifffile, e.g. al-diag-seg's environment):

    python make_grabe2016.py <al_atlas checkout>/grabe2016

Each glomerulus (both antennal lobes: 108) is a smoothed surface of its label (marching cubes), in um, centred on the middle of all of them, made at about 1 um in each direction (BIN). The .bin has each one's vertices (int16, 1/100 um)
and triangles (uint16); the .json says where, with each one's name, side, receptor, volume, centroid, neighbours
(those within NEAR_UM of it), and colour: glomeruli are coloured by name (left and right alike), so that touching ones
differ where the palette allows (DSatur graph colouring), and those of a colour lie as far apart as they can.

The anatomical axes are found from the glomeruli's names (D/V; A, P, M, L as the second letter of D and V glomeruli;
each a mean of the centroids of one kind less the other's), per side, made orthogonal (dorsal kept, anterior made
orthogonal to it, medial from their cross product, its sign as the names say), and kept with how far each was from
orthogonal; the fly's left from where the left lobe is.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

SHA256 = {"labels.tif": "5e0fa4bceb09e58832c84b9fe3b64ac742cf5beda429f8435733315b0d0cd815",
          "structures.csv": "4717da31369525b539d77bbfbb7f1834ac1fb921fc1464771e5528e7e82d253e"}
AL_ATLAS = "88ebada36bed414e7799d887b6fceca034bbc0cb"
SMOOTH_UM = 0.7  # the surfaces' smoothing (s.d. of a Gaussian over each label's mask)
BIN = 3  # pixels binned in-plane (k by k) before the surfaces are made: 1.02 um, about the planes' 0.96
NEAR_UM = 1.5  # glomeruli this close are neighbours (for their colours)
QUANT = 100  # vertices in 1/QUANT um
# Tableau 10: hues told apart on light and dark backgrounds
PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def spacing(path: Path) -> tuple[float, float, float]:
    """(z, y, x) um of an ImageJ TIFF."""
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        xy = [den / num for num, den in (page.tags[k].value for k in ("YResolution", "XResolution"))]
        return float(t.imagej_metadata.get("spacing", 1.0)), *xy


def surface(mask: np.ndarray, sp, origin) -> tuple[np.ndarray, np.ndarray]:
    """A label's smoothed surface: (vertices (n, 3) as x, y, z um, triangles (m, 3))."""
    k = BIN
    m = np.pad(mask, ((2, 2), (2 * k, 2 * k), (2 * k, 2 * k))).astype(np.float32)
    # (pixels binned in-plane, to about the planes' spacing: k^2 fewer triangles, still finer than the smoothing)
    m = m[:, : m.shape[1] // k * k, : m.shape[2] // k * k].reshape(m.shape[0], m.shape[1] // k, k, m.shape[2] // k, k).mean((2, 4))
    sp2 = (sp[0], sp[1] * k, sp[2] * k)
    m = ndi.gaussian_filter(m, [SMOOTH_UM / s for s in sp2])
    verts, faces, _, _ = marching_cubes(m, 0.5, spacing=sp2)
    # (a bin's middle is (k - 1) / 2 pixels into it; the padding, 2 pixels or bins, taken off)
    zyx = verts + np.asarray(origin) - np.array([2 * sp[0], 2 * k * sp[1], 2 * k * sp[2]]) + np.array([0, (k - 1) / 2 * sp[1], (k - 1) / 2 * sp[2]])
    return zyx[:, ::-1], faces[:, ::-1]  # (x, y, z; triangles' winding kept outward as the axes flip)


def neighbours(labels: np.ndarray, sp) -> dict[int, set[int]]:
    """Each label's neighbours: those within NEAR_UM of it."""
    r = [int(np.ceil(NEAR_UM / s)) for s in sp]
    zz, yy, xx = np.ogrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    ball = (zz * sp[0]) ** 2 + (yy * sp[1]) ** 2 + (xx * sp[2]) ** 2 <= NEAR_UM ** 2
    out = {}
    for lab, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        box = tuple(slice(max(0, s.start - p), s.stop + p) for s, p in zip(sl, r))
        near = ndi.binary_dilation(labels[box] == lab, ball)
        out[lab] = {int(v) for v in np.unique(labels[box][near]) if v and v != lab}
    return out


def colours(names: list[str], adj: dict[str, set[str]], centroid: dict[str, np.ndarray]) -> dict[str, str]:
    """A colour per glomerulus name: DSatur (most differently coloured neighbours first), each taking, of the colours
    none of its neighbours has (else those fewest have), the one whose nearest glomerulus of that colour is farthest."""
    got: dict[str, int] = {}
    while len(got) < len(names):
        left = [n for n in names if n not in got]
        n = max(left, key=lambda n: (len({got[m] for m in adj[n] if m in got}), len(adj[n]), n))
        used = [got[m] for m in adj[n] if m in got]
        fewest = min(used.count(k) for k in range(len(PALETTE)))
        ok = [k for k in range(len(PALETTE)) if used.count(k) == fewest]

        def far(k):
            same = [np.linalg.norm(centroid[n] - centroid[m]) for m in got if got[m] == k]
            return min(same) if same else np.inf

        got[n] = max(ok, key=lambda k: (far(k), -k))
    return {n: PALETTE[k] for n, k in got.items()}


def axes(rows: list[dict], centroid) -> dict:
    """The anatomical axes (unit vectors in the meshes' x, y, z) of each side's lobe, from its glomeruli's names."""
    out = {}
    for side in ("left", "right"):
        rs = [r for r in rows if r["side"] == side]

        def towards(a, b):
            ca = [centroid[r["label"]] for r in rs if a(r["glomerulus"])]
            cb = [centroid[r["label"]] for r in rs if b(r["glomerulus"])]
            v = np.mean(ca, 0) - np.mean(cb, 0)
            return v / np.linalg.norm(v)

        second = lambda c: lambda n: re.match(rf"^[DV]{c}", n) is not None  # noqa: E731
        d = towards(lambda n: n.startswith("D"), lambda n: n.startswith("V"))
        a_raw, m_raw = towards(second("A"), second("P")), towards(second("M"), second("L"))
        a = a_raw - (a_raw @ d) * d
        a /= np.linalg.norm(a)
        m = np.cross(d, a)
        m *= np.sign(m @ m_raw)
        deg = lambda u, v: float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))  # noqa: E731
        out[side] = {"dorsal": d.round(4).tolist(), "anterior": a.round(4).tolist(), "medial": m.round(4).tolist(),
                     "from_names_deg": {"anterior": round(deg(a, a_raw), 1), "medial": round(deg(m, m_raw), 1)}}
    left = np.mean([centroid[r["label"]] for r in rows if r["side"] == "left"], 0)
    right = np.mean([centroid[r["label"]] for r in rows if r["side"] == "right"], 0)
    d = np.mean([out[s]["dorsal"] for s in out], 0)
    d /= np.linalg.norm(d)
    a = np.mean([out[s]["anterior"] for s in out], 0)
    a -= (a @ d) * d
    a /= np.linalg.norm(a)
    lr = np.cross(d, a)
    lr *= np.sign(lr @ (left - right))
    out["both"] = {"dorsal": d.round(4).tolist(), "anterior": a.round(4).tolist(), "left": lr.round(4).tolist()}
    return out


def main(folder: str) -> None:
    src, here = Path(folder), Path(__file__).parent
    for name, want in SHA256.items():
        if hashlib.sha256((src / name).read_bytes()).hexdigest() != want:
            sys.exit(f"{src / name}: not the al_atlas {AL_ATLAS[:7]} file (its checksum differs)")
    labels, sp = tifffile.imread(src / "labels.tif"), spacing(src / "labels.tif")
    with open(src / "structures.csv", newline="") as fh:
        rows = [{**r, "label": int(r["label"])} for r in csv.DictReader(fh) if r["glomerulus"]]
    surfaces, centroid, boxes = {}, {}, ndi.find_objects(labels)
    for r in rows:
        sl = boxes[r["label"] - 1]
        mask = labels[sl] == r["label"]
        surfaces[r["label"]] = surface(mask, sp, [s.start * p for s, p in zip(sl, sp)])
        centroid[r["label"]] = (np.argwhere(mask).mean(0) + [s.start for s in sl])[::-1] * np.asarray(sp[::-1])
    middle = np.mean(list(centroid.values()), 0)
    adj_lab = neighbours(labels, sp)
    by_label = {r["label"]: r for r in rows}
    names = sorted({r["glomerulus"] for r in rows})
    adj = {n: set() for n in names}
    for lab, near in adj_lab.items():
        if lab in by_label:
            adj[by_label[lab]["glomerulus"]] |= {by_label[m]["glomerulus"] for m in near if m in by_label}
    name_centroid = {n: np.mean([centroid[r["label"]] for r in rows if r["glomerulus"] == n and r["side"] == "left"], 0) for n in names}
    colour = colours(names, adj, name_centroid)
    shared = sum(colour[n] == colour[m] for n in names for m in adj[n]) // 2
    blob, items = bytearray(), []
    for r in sorted(rows, key=lambda r: (r["side"], r["glomerulus"])):
        v, f = surfaces[r["label"]]
        q = np.round((v - middle) * QUANT)
        assert np.abs(q).max() < 32768 and len(v) < 65536, r["name"]
        pos = len(blob)
        blob += q.astype("<i2").tobytes()
        idx = len(blob)
        blob += f.astype("<u2").tobytes()
        blob += b"\0" * (-len(blob) % 4)
        items.append({"label": r["label"], "name": r["glomerulus"], "side": r["side"], "receptor": r["receptor"] or None,
                      "volume_um3": float(r["volume_um3"]), "centroid": ((centroid[r["label"]] - middle).round(2)).tolist(),
                      "colour": colour[r["glomerulus"]], "neighbours": sorted(adj[r["glomerulus"]]),
                      "vertices": [pos, len(v)], "triangles": [idx, len(f)]})
    (here / "grabe2016.bin").write_bytes(bytes(blob))
    doc = {"title": "Grabe et al. 2016 in vivo antennal lobe atlas",
           "citation": "Grabe V, Baschwitz A, Dweck HKM, Lavista-Llanos S, Hansson BS, Sachse S. Elucidating the Neuronal "
                       "Architecture of Olfactory Glomeruli in the Drosophila Antennal Lobe. Cell Rep. 2016;16(12):3401-3413. "
                       "doi:10.1016/j.celrep.2016.08.063",
           "source": f"https://github.com/ejhonglab/al_atlas/tree/{AL_ATLAS}/grabe2016 (labels.tif, structures.csv)",
           "made_by": "make_grabe2016.py", "spacing_um": [round(s, 4) for s in sp], "smooth_um": SMOOTH_UM, "near_um": NEAR_UM,
           "units": f"vertices: int16, 1/{QUANT} um, x y z (the atlas' columns, rows, planes), less the middle of all glomeruli",
           "quant": QUANT, "palette": PALETTE, "neighbours_sharing_a_colour": shared, "axes": axes(rows, {k: c - middle for k, c in centroid.items()}),
           "glomeruli": items}
    (here / "grabe2016.json").write_text(json.dumps(doc, indent=1))
    print(f"{len(items)} glomeruli, {sum(i['triangles'][1] for i in items)} triangles, {len(blob) / 1e6:.2f} MB; "
          f"{shared} pairs of neighbours share a colour; axes: {json.dumps(doc['axes'])}")


if __name__ == "__main__":
    main(sys.argv[1])
