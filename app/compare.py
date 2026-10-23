"""What the Compare tab shows: datasets as matrices of their odors by their ROIs, and the
odor table's chemicals as ECFP fingerprints.

A dataset there is the mean of every trial of each of its odors (as presented, by panel:
mixtures and concentrations as they are), for each ROI; for glomeruli, of every fly too,
one column per glomerulus. Each odor (row) says the chemical of the odor table it is, if
it is one alone (not a mixture) matched there, so that it can be compared with other
datasets and other spaces of chemicals. How alike odors are, and how alike those
likenesses are between spaces, is computed in the browser, as the rest of what is shown.
"""
from __future__ import annotations

import base64
import threading
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from . import glomeruli, odormeta
from .odors import SOLVENT_STR, OdorFormatError, parse_odor

FP_BITS = 2048  # ECFP fingerprints' length (bits)

_fp_lock = threading.Lock()
_fp_cache: dict[tuple, dict] = {}


def dataset(db, ds_id: int, datasets_dir: Path) -> Optional[dict]:
    """A dataset as the Compare tab shows it (see the module's docstring); None if there is none.

    Returns:
        dict with ``id``, ``title``, ``coding_unit``, ``driver``, ``kind`` ('glomeruli': a column per
        glomerulus, the mean of the flies that have it; 'rois': a column per ROI of each
        fly), ``rows`` (each odor, in the order shown in Browse: ``panel``, ``odor``,
        ``odor_id`` and ``chem`` (``name``, ``abbrev``, ``cid``) if it is one chemical of
        the odor table, ``log10_conc``), ``cols`` (``label``, and for glomeruli
        ``n_flies``) and ``values_b64`` (float32, rows × cols, NaN where nothing is).
    """
    ds = db.get_dataset(ds_id)
    arrays = db.get_matrix_arrays(ds_id)
    if ds is None or arrays is None:
        return None
    units, trials, values = arrays["units"], arrays["trials"], arrays["values"]

    # odors as Browse orders them: by panel (in the order they first come), then as in the file
    conds: dict[tuple, list[int]] = {}
    for j, t in enumerate(trials):
        conds.setdefault((t["panel"], t["odor"]), []).append(j)
    panels = list(dict.fromkeys(p for p, _ in conds))
    keys = sorted(conds, key=lambda k: panels.index(k[0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the mean of nothing but NaN: NaN
        means = np.stack([np.nanmean(values[:, conds[k]], axis=1) for k in keys], axis=1)  # units × odors

    index = odormeta.OdorIndex(db.list_odors())
    matched = {(r["panel"], r["odor"]): r["odor_id"]
               for r in odormeta.dataset_odors(ds, trials, index, odormeta.dataset_yaml_odors(ds, datasets_dir))}
    rows = []
    for panel, odor in keys:
        components = next(t["components"] for t in trials if (t["panel"], t["odor"]) == (panel, odor))
        alone = [c for c in components if c != SOLVENT_STR]
        odor_id = matched.get((panel, alone[0])) if len(alone) == 1 else None
        try:
            conc = parse_odor(alone[0], strict=False).log10_conc if len(alone) == 1 else None
        except OdorFormatError:
            conc = None
        o = index.by_id.get(odor_id) if odor_id is not None else None
        rows.append({"panel": panel, "odor": odor, "odor_id": odor_id, "log10_conc": conc,
                     "chem": {"name": o["name"], "abbrev": o.get("abbrev"), "cid": o.get("cid")} if o else None})

    if units and all(u["roi_kind"] == "glomerulus" for u in units):
        names = list(dict.fromkeys(u["roi"] for u in units))
        order = [g for g in glomeruli.GLOMERULUS_NAMES if g in names] + [g for g in names if g not in glomeruli.GLOMERULI]
        cols, columns = [], []
        for g in order:
            mine = [i for i, u in enumerate(units) if u["roi"] == g]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                columns.append(np.nanmean(means[mine], axis=0))
            cols.append({"label": g, "n_flies": len({(units[i]["date"], units[i]["fly_num"]) for i in mine if units[i]["date"] is not None})})
        matrix, kind = np.stack(columns, axis=1), "glomeruli"  # odors × glomeruli
    else:
        cols = [{"label": f"{u['date']} fly {u['fly_num']}: {u['roi']}" if u["date"] is not None else u["roi"]} for u in units]
        matrix, kind = means.T, "rois"  # odors × ROIs
    return {
        "id": ds_id, "title": ds["title"], "coding_unit": ds["metadata"].get("coding_unit"), "driver": ds["metadata"].get("driver"),
        "kind": kind,
        "rows": rows, "cols": cols, "shape": [len(rows), len(cols)],
        "values_b64": base64.b64encode(np.ascontiguousarray(matrix, dtype="<f4").tobytes()).decode("ascii"),
    }


def fingerprints(odors: list[dict], diameter: int) -> dict:
    """ECFP fingerprints of the odor table's chemicals (RDKit's Morgan fingerprints with
    ECFP's atom invariants, of radius `diameter` / 2, `FP_BITS` long), from their InChI
    (as PubChem gives it for their CID). Kept for each `diameter` while the odors' stay the same.

    Returns:
        dict with ``diameter``, ``bits`` (`FP_BITS`), ``fingerprints`` ({odor id: its bits
        set, ascending}) and ``missing`` (ids of odors with no structure to compute one from).
    """
    key = (diameter, tuple((o["id"], o.get("inchi")) for o in odors))
    with _fp_lock:
        if key in _fp_cache:
            return _fp_cache[key]
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")  # InChI warnings, for the log of each odor read
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=diameter // 2, fpSize=FP_BITS)
    fps, missing = {}, []
    for o in odors:
        mol = Chem.MolFromInchi(o["inchi"]) if o.get("inchi") else None
        if mol is None:
            missing.append(o["id"])
            continue
        fps[o["id"]] = sorted(gen.GetFingerprint(mol).GetOnBits())
    result = {"diameter": diameter, "bits": FP_BITS, "fingerprints": fps, "missing": missing}
    with _fp_lock:
        if len(_fp_cache) > 8:
            _fp_cache.clear()
        _fp_cache[key] = result
    return result
