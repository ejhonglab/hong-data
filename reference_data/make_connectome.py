#!/usr/bin/env python3
"""Makes the connectome reference datasets (``connectome/*.parquet``, and their entries in
``manifest.json``): of the hemibrain's right calyx and of the male CNS's left and right ones,
each from Pratyush's tables (his pipeline's claws of KCs, boutons of PNs and units of APL),
for each kind of connection the MB model has:

- ``pn_kc``: PN boutons -> KC claws
- ``kc_apl``: KC claws -> APL
- ``apl_kc``: APL -> KC claws
- ``pn_apl``: PN boutons -> APL
- ``apl_pn``: APL -> PN boutons

each twice:

- ``raw``: every synapse of his table counted, per (pre unit, post unit), with no rule of
  the model's: whatever its ROI, the PN's type (multiglomerular too) or the KC's (untyped
  and fragments too), and units labelled -1 (unassigned) kept as units of their own;
  synapses with no unit at all (NaN) left out. Columns: APL units.
- ``model``: what the model is given (olfsysm's ``fit_mb_model`` with ``prat_claws=True``,
  ``prat_boutons=True``, ``use_connectome_APL_weights=True``, ``multibouton_claws='argmax'``): the claws
  it keeps, each a 1 for the one bouton it takes input from (of its PN's boutons it has synapses
  from, the one with the most, ties to the lowest numbered, as E97's main bouton); and the synapses between
  each of those claws (or boutons) and APL (summed over APL's units: one column), which
  olfsysm normalizes (only their ratios matter). Every claw (or bouton) of the model is a
  row, 0 where APL has no synapse with it.

The model's steps (`model_claws`, `model_apl`) are those of olfsysm's loader
(``pyolfsysm.mb_model.connectome_wPNKC`` and ``connectome_APL_weights``), which reads the
hemibrain alone: they are checked to give exactly its wiring and APL weights for the
hemibrain (`check_hemibrain`), and applied as they are to the male CNS, with the choices of
the KC-remapping queue's E97 (``counted_claws``) where the loader has none for it: no ROI
rule (the official calyx ROI leaves out part of the left calyx; the hemibrain's keeps both
ends in CA(R)), and its PN types' newer glomerulus names as the hemibrain's (VC3 and VC5 ->
VC3, as the model merges VC3l and VC3m; VM6 -> VC5). KCs are on the rows of every table of
claws, with their ``kc_id`` (body ID), so that they can be compared with models' KCs in the
Compare tab.

Usage (with a Python that has pyolfsysm, e.g. the model worker's /opt/model-venv):
make_connectome.py [<male CNS folder>]   (default: Tom's copy of Pratyush's tables on tensor)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "connectome"
MANIFEST = HERE / "manifest.json"
MALECNS_DIR = Path("/local/tensor/Tom/claude_kc_remapping/data/Male-CNS_Tom")
MALECNS_STAMP = "2026-10-25-17-31-06"
# the hemibrain tables the model's loader reads (in olfsysm's pyolfsysm/data/from_pratyush/2025-12-05)
HEMIBRAIN_FILES = {
    "pn_kc": "PNbouton-to-KCclaw_Connectivity_2025-12-09-19-47-36.parquet",
    "apl_kc": "APLunit-to-KCclaw_Connectivity_2025-12-14-21-12-48.parquet",
    "kc_apl": "KCclaw-to-APLunit_Connectivity_2025-12-14-21-12-48.parquet",
    "apl_pn": "APLunit-to-PNbouton_Connectivity_2025-12-05-13-39-22.parquet",
    "pn_apl": "PNbouton-to-APLunit_Connectivity_2025-12-05-13-39-22.parquet",
    "boutons": "PN_Bouton_Counts_Hemibrain_2025-12-05-13-39-22.csv",
    # each synapse of a KC, PN or APL with the unit he assigned it to (where the units are, `unit_places`)
    "claw_synapses": "KC_Synapse_Assignments_Hemibrain_2025-12-09-19-47-36.parquet",
    "bouton_synapses": "PN_Synapse_Assignments_Hemibrain_2025-12-05-13-39-22.parquet",
    "apl_synapses": "APL_Assignments_2025-12-05-13-39-22.parquet",
}
MALECNS_FILES = {
    "pn_kc": "PNbouton-to-KCclaw_Connectivity_Male-CNS_{}.parquet",
    "apl_kc": "APLunit-to-KCclaw_Connectivity_Male-CNS_{}.parquet",
    "kc_apl": "KCclaw-to-APLunit_Connectivity_Male-CNS_{}.parquet",
    "apl_pn": "APLunit-to-PNbouton_Connectivity_Male-CNS_{}.parquet",
    "pn_apl": "PNbouton-to-APLunit_Connectivity_Male-CNS_{}.parquet",
    "boutons": "PN_Bouton_Counts_Male-CNS_{}.csv",
    "claw_synapses": "KC_Synapse_Assignments_Male-CNS_{}.parquet",
    "bouton_synapses": "PN_Synapse_Assignments_Male-CNS_{}.parquet",
    "apl_synapses": "APL_Synapse_Assignments_Male-CNS_{}.parquet",
}
# the calyces: (brain, the KC's or bouton's side, the ROI both ends of a synapse are in for the model, or None: any)
CALYCES = {
    "hemibrain": ("hemibrain", "R", "CA(R)"),
    "malecns-left": ("malecns", "L", None),
    "malecns-right": ("malecns", "R", None),
}
CALYX_TITLES = {"hemibrain": "hemibrain (right calyx)", "malecns-left": "male CNS, left calyx",
                "malecns-right": "male CNS, right calyx"}
# the male CNS's PN types' glomeruli, as the model (the hemibrain's, merging VC3l and VC3m) names them (E97's BRAIN_RENAMES)
RENAMES = {"hemibrain": {"VC3l": "VC3", "VC3m": "VC3"}, "malecns": {"VC5": "VC3", "VM6": "VC5"}}
KC_CLASSES = ("g", "ab", "a'b'")
CLAW_SYNAPSES = 5  # the fewest synapses from one PN a claw of the model has (the loader's)
CONNECTIONS = {  # name -> (title, pre, post)
    "pn_kc": ("PN boutons → KC claws", "bouton", "claw"),
    "kc_apl": ("KC claws → APL", "claw", "apl"),
    "apl_kc": ("APL → KC claws", "apl", "claw"),
    "pn_apl": ("PN boutons → APL", "bouton", "apl"),
    "apl_pn": ("APL → PN boutons", "apl", "bouton"),
}
CLAW_FIELDS = ["claw", "kc_id", "claw_id", "kc_type", "kc_subtype"]
BOUTON_FIELDS = ["bouton", "pn_id", "bouton_id", "glomerulus", "model_glomerulus", "pn_type", "pn_bundle"]
APL_FIELDS = ["apl", "apl_id", "apl_unit"]
FIELDS = {"claw": CLAW_FIELDS, "bouton": BOUTON_FIELDS, "apl": APL_FIELDS}
# where each unit is (`place`): its x, y, z (µm, each centered as olfsysm's `center_each_claw_coord` centers claws) and
# compartment (olfsysm's `add_compartment_index`, shape 0: 0 within half the largest distance from the units' centroid,
# else 1), named by its kind (claw_x, ..., claw_compartment)
PLACE = ("x", "y", "z", "compartment")
PIXEL_TO_UM = 8 / 1000  # Pratyush's coordinates: 8 nm voxels (olfsysm's PIXEL_TO_UM)
UNIT_COLUMNS = {"claw": "anatomical_claw_corrected", "bouton": "anatomical_bouton", "apl": "anatomical_unit"}


# ------------------------------------------------------------------- reading


def hemibrain_dir() -> Path:
    from pyolfsysm import mb_model
    return Path(mb_model.prat_hemibrain_seg_dir)


def _side(s: pd.Series) -> pd.Series:
    return s.map({"Left": "L", "Right": "R", "L": "L", "R": "R"})


def read_brain(brain: str, malecns_dir: Path = MALECNS_DIR) -> dict[str, pd.DataFrame]:
    """Pratyush's tables of `brain`, one row per synapse, in one form: of each connection,
    ``pre`` and ``post`` (body IDs), ``pre_unit`` and ``post_unit`` (claw, bouton or APL unit:
    float, NaN where none, -1 unassigned), ``pre_type``, ``post_type`` (neuprint's types),
    ``pre_instance``, ``post_instance``, ``roi_pre``, ``roi_post``, ``side`` (of its KC, or
    else its PN: 'L' / 'R'), and of PN -> KC ``dist_to_root``; and ``boutons``, the PNs'
    table (bodyId, PN_Bundle)."""
    if brain == "hemibrain":
        d, files = hemibrain_dir(), HEMIBRAIN_FILES
    else:
        d, files = Path(malecns_dir), {k: v.format(MALECNS_STAMP) for k, v in MALECNS_FILES.items()}
    out = {}
    for name, (_, pre, post) in CONNECTIONS.items():
        df = pd.read_parquet(d / files[name])
        unit = {"claw": "anatomical_claw_corrected", "bouton": "anatomical_bouton", "apl": "anatomical_unit"}

        def col(kind, end):  # the hemibrain's tables name a unit's column alone, the male CNS's by its end
            for c in (f"{unit[kind]}_{end}", unit[kind]):
                if c in df.columns:
                    return df[c].astype(float)
            raise KeyError(f"{brain} {name}: no {unit[kind]} column")

        kc_or_pn_end = "post" if post == "claw" else "pre" if pre == "claw" else "pre" if pre == "bouton" else "post"
        if brain == "hemibrain":
            side = pd.Series("R", index=df.index)
        else:
            side = _side(df[f"side_{kc_or_pn_end}"])
            # (a synapse whose side is missing at that end: of the other end, which is always the same)
            other = f"side_{'pre' if kc_or_pn_end == 'post' else 'post'}"
            if other in df.columns:
                side = side.fillna(_side(df[other]))
        out[name] = pd.DataFrame({
            "pre": df["bodyId_pre"].astype(np.int64), "post": df["bodyId_post"].astype(np.int64),
            "pre_unit": col(pre, "pre"), "post_unit": col(post, "post"),
            "pre_type": df["type_pre"], "post_type": df["type_post"],
            "pre_instance": df["instance_pre"], "post_instance": df["instance_post"],
            "roi_pre": df["roi_pre"], "roi_post": df["roi_post"], "side": side,
            **({"dist_to_root": df["dist_to_root" if "dist_to_root" in df.columns else "dist_to_root_post"],
                "x_post": df["x_post"], "y_post": df["y_post"], "z_post": df["z_post"]}
               if name == "pn_kc" else {}),
        })
    out["boutons"] = pd.read_csv(d / files["boutons"])
    out["places"] = unit_places(d, files)
    return out


def unit_places(d: Path, files: dict) -> dict[str, pd.DataFrame]:
    """Where each claw, bouton and APL unit is: the mean position (µm) of the synapses Pratyush assigned to it, of
    its neuron's every synapse (his synapse assignment tables); none for units labelled -1 (unassigned). {kind: x, y,
    z by (body ID, unit)}."""
    out = {}
    for kind, col in UNIT_COLUMNS.items():
        a = pd.read_parquet(d / files[f"{kind}_synapses"], columns=["neuron", col, "x", "y", "z"]).dropna(subset=[col])
        a = a.assign(neuron=a.neuron.astype(np.int64), unit=a[col].astype(np.int64))
        a = a[a.unit != -1]  # (a neuron's synapses he assigned to no unit, wherever they are: no place)
        out[kind] = a.groupby(["neuron", "unit"])[["x", "y", "z"]].mean() * PIXEL_TO_UM
    return out


def place(xyz: pd.DataFrame, labels: pd.Series, kind: str) -> pd.DataFrame:
    """The places of the units of a table's lines (`xyz`: x, y, z in µm per line; `labels`: each line's unit), as
    olfsysm places the model's claws: each coordinate centered over the table's units (by the middle of their
    range), and each unit's compartment (shape 0: 0 within half the largest distance from their centroid, else 1).
    Columns <kind>_x, <kind>_y, <kind>_z, <kind>_compartment (missing where a unit has no place)."""
    xyz = xyz.reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    first = ~labels.duplicated()
    units = xyz[first & xyz.notna().all(axis=1)]
    out = pd.DataFrame(index=xyz.index)
    if len(units):
        mid = (units.min() + units.max()) / 2
        centered = xyz - mid
        c = units.to_numpy(dtype=float)
        d_units = np.linalg.norm(c - c.mean(axis=0), axis=1)
        d = np.linalg.norm(xyz.to_numpy(dtype=float) - c.mean(axis=0), axis=1)
        comp = np.where(d <= 0.5 * d_units.max(), 0.0, 1.0)
        comp[~np.isfinite(d)] = np.nan
    else:
        centered, comp = xyz * np.nan, np.full(len(xyz), np.nan)
    for a in ("x", "y", "z"):
        out[f"{kind}_{a}"] = centered[a].to_numpy(dtype=float)
    out[f"{kind}_compartment"] = comp
    return out


def unit_xyz(places: dict, kind: str, body, unit) -> pd.DataFrame:
    """`unit_places`' x, y, z of each (body, unit) given (NaN where it has none)."""
    key = pd.MultiIndex.from_arrays([pd.Series(body).astype(np.int64).to_numpy(), pd.Series(unit).astype(np.int64).to_numpy()])
    return places[kind].reindex(key).reset_index(drop=True)


def kc_class(subtype) -> str | None:
    """The model's KC type of a neuprint KC type (olfsysm's `add_kc_type_col`: 'KCab-m' -> 'ab'), if one of its three."""
    if not isinstance(subtype, str):
        return None
    t = subtype.strip("KC ").split("-")[0]
    return t if t in KC_CLASSES else None


def model_glomeruli(types: pd.Series, brain: str) -> pd.Series:
    """The model's glomerulus of each PN type (olfsysm's `add_glomerulus_col_from_hemibrain_type`, after the brain's
    renames): None for the types it leaves out (with '+', WEDPN4 and WEDPN12, names not among its glomeruli)."""
    from pyolfsysm.mb_model import task_gloms
    out = {}
    for t in pd.unique(types.dropna()):
        t = str(t)
        if t in ("WEDPN12", "WEDPN4") or "+" in t or t.count("_") != 1:
            out[t] = None
            continue
        g = t.split("_")[0]
        g = RENAMES[brain].get(g, g)
        out[t] = g if g in task_gloms else None
    return types.map(out)


def glomerulus_of(t) -> str | None:
    """The glomerulus a PN type names, as the connectome names it (the part before '_')."""
    return str(t).split("_")[0] if isinstance(t, str) else None


# --------------------------------------------------------------- the model's


def _roi(df: pd.DataFrame, roi: str | None) -> pd.DataFrame:
    return df if roi is None else df[(df.roi_pre == roi) & (df.roi_post == roi)]


def model_claws(t: dict[str, pd.DataFrame], brain: str, side: str, roi: str | None) -> pd.DataFrame:
    """The model's claws of a calyx (the steps of olfsysm's `connectome_wPNKC`, ``prat_claws=True``,
    ``prat_boutons=True``): of the PN -> KC synapses (in `roi`), those with a claw label (-1 kept), of a PN type
    of the model's glomeruli, with a finite distance to the root, of a KC with a type and with a bouton label
    (not -1), summed per (KC, claw, PN); those of 5 or more kept, and of them claws of one PN alone; of the three
    KC classes. One row per claw: kc_id, claw_id, kc_type, kc_subtype, pn_id, pn_type, glomerulus, bouton_id (the
    claw's bouton with the most of its synapses, ties to the lowest numbered: ``multibouton_claws='argmax'``),
    n_synapses, x, y, z (µm, the mean of its synapses' KC ends), sorted by KC and claw."""
    s = t["pn_kc"]
    s = _roi(s[s.side == side], roi)
    s = s[s.post_unit.notna()]
    s = s.assign(glomerulus=model_glomeruli(s.pre_type, brain))
    s = s[s.glomerulus.notna()]
    s = s[np.isfinite(s.dist_to_root.astype(float)) & s.post_type.notna()]
    if s.pre_unit.isna().any():
        raise RuntimeError(f"{brain} {side}: PN -> KC synapses without a bouton label (the loader asserts none)")
    s = s[s.pre_unit != -1]
    keys = ["post", "post_unit", "pre"]
    for k in ("glomerulus", "post_type", "pre_type"):
        if (s.groupby(keys)[k].nunique() > 1).any():
            raise RuntimeError(f"{brain} {side}: more than one {k} in a (KC, claw, PN)")
    nb = s.groupby(keys + ["pre_unit"]).size().rename("n").reset_index()
    main = (nb.sort_values(keys + ["n", "pre_unit"], ascending=[True] * len(keys) + [False, True])
            .drop_duplicates(keys).set_index(keys)["pre_unit"].rename("bouton_id"))
    c = s.groupby(keys, sort=True).agg(n_synapses=("pre_unit", "size"),
                                       glomerulus=("glomerulus", "first"), kc_subtype=("post_type", "first"),
                                       pn_type=("pre_type", "first"), x=("x_post", "mean"), y=("y_post", "mean"),
                                       z=("z_post", "mean")).join(main).reset_index()
    c[["x", "y", "z"]] = c[["x", "y", "z"]] * PIXEL_TO_UM  # (the loader's claw coordinates: its synapses' KC ends)
    c = c[c.n_synapses >= CLAW_SYNAPSES]
    c = c[c.groupby(["post", "post_unit"])["pre"].transform("size") == 1]
    c = c.assign(kc_type=c.kc_subtype.map(kc_class))
    c = c[c.kc_type.notna()]
    c = c.rename(columns={"post": "kc_id", "post_unit": "claw_id", "pre": "pn_id"})
    c["claw_id"] = c.claw_id.astype(np.int64)
    c["bouton_id"] = c.bouton_id.astype(np.int64)
    return c[["kc_id", "claw_id", "kc_type", "kc_subtype", "pn_id", "pn_type", "glomerulus", "bouton_id",
              "n_synapses", "x", "y", "z"]].sort_values(["kc_id", "claw_id"]).reset_index(drop=True)


def model_boutons(claws: pd.DataFrame) -> pd.DataFrame:
    """The boutons of the model's wiring (its columns): those of its claws, by glomerulus, PN and bouton."""
    return (claws[["glomerulus", "pn_id", "bouton_id", "pn_type"]].drop_duplicates(["glomerulus", "pn_id", "bouton_id"])
            .sort_values(["glomerulus", "pn_id", "bouton_id"]).reset_index(drop=True))


def model_apl(t: dict[str, pd.DataFrame], brain: str, side: str, roi: str | None, claws: pd.DataFrame,
              boutons: pd.DataFrame) -> dict[str, pd.Series]:
    """The model's APL weights of a calyx (olfsysm's `connectome_APL_weights`, ``prat_claws=True``,
    ``prat_boutons=True``): the synapses (in `roi`) between APL and each of the model's claws (claw labels -1 kept,
    as claws of their own) and boutons (by the model's glomerulus of their PN), summed over APL's units, 0 where
    there are none. {'kc_apl', 'apl_kc': by (kc_id, claw_id); 'pn_apl', 'apl_pn': by (glomerulus, pn_id,
    bouton_id)}, in the order of `claws` and `boutons`."""
    out = {}
    claw_index = pd.MultiIndex.from_frame(claws[["kc_id", "claw_id"]])
    bouton_index = pd.MultiIndex.from_frame(boutons[["glomerulus", "pn_id", "bouton_id"]])
    for name in ("kc_apl", "apl_kc"):
        s = t[name]
        s = _roi(s[s.side == side], roi)
        kc, claw = ("pre", "pre_unit") if name == "kc_apl" else ("post", "post_unit")
        s = s[s[claw].notna()]
        n = s.groupby([s[kc].rename("kc_id"), s[claw].astype(np.int64).rename("claw_id")]).size()
        out[name] = n.reindex(claw_index, fill_value=0).astype(float)
    for name in ("pn_apl", "apl_pn"):
        s = t[name]
        s = _roi(s[s.side == side], roi)
        pn, bouton, ptype = ("pre", "pre_unit", "pre_type") if name == "pn_apl" else ("post", "post_unit", "post_type")
        s = s.assign(glomerulus=model_glomeruli(s[ptype], brain))
        s = s[s.glomerulus.notna() & s[bouton].notna()]
        n = s.groupby([s.glomerulus, s[pn].rename("pn_id"), s[bouton].astype(np.int64).rename("bouton_id")]).size()
        out[name] = n.reindex(bouton_index, fill_value=0).astype(float)
    return out


def check_hemibrain(claws: pd.DataFrame, boutons: pd.DataFrame, apl: dict[str, pd.Series]) -> None:
    """That `model_claws`, `model_boutons` and `model_apl` give the hemibrain's wiring (``multibouton_claws='argmax'``),
    claw places and APL weights exactly as olfsysm's loader does (AssertionError if not)."""
    from pyolfsysm import mb_model as m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = m.connectome_wPNKC("hemibrain", prat_claws=True, prat_boutons=True, multibouton_claws="argmax")
        wAPLKC, wKCAPL, wAPLPN, wPNAPL = m.connectome_APL_weights("hemibrain", prat_claws=True, prat_boutons=True,
                                                                   wPNKC=w)[:4]
    idx = w.index.to_frame(index=False)
    assert (idx.kc_id.to_numpy() == claws.kc_id.to_numpy()).all() and (idx.claw_id.to_numpy() == claws.claw_id.to_numpy()).all(), \
        "the model's claws are not the loader's"
    assert (idx.kc_type.to_numpy() == claws.kc_type.to_numpy()).all(), "the claws' KC types are not the loader's"
    cols = w.columns.to_frame(index=False)
    mine = boutons[["glomerulus", "pn_id", "bouton_id"]]
    assert len(cols) == len(mine) and set(map(tuple, cols.to_numpy())) == set(map(tuple, mine.to_numpy())), \
        "the model's boutons are not the loader's"
    W = w.to_numpy()
    assert set(np.unique(W)) <= {0, 1} and (W.sum(axis=1) == 1).all(), "the loader's wiring is not one bouton per claw"
    got = [tuple(x) for x in cols.to_numpy()[W.argmax(axis=1)]]
    assert got == [tuple(x) for x in claws[["glomerulus", "pn_id", "bouton_id"]].to_numpy()], \
        "the claws' boutons are not the loader's"
    mine = place(claws[["x", "y", "z"]], claws.kc_id.astype(str) + "/" + claws.claw_id.astype(str), "claw")
    for a in ("x", "y", "z"):
        assert np.allclose(idx[f"claw_{a}"].to_numpy(), mine[f"claw_{a}"].to_numpy(), rtol=0, atol=1e-9), \
            f"the claws' {a} are not the loader's"
    assert (idx.compartment.to_numpy() == mine.claw_compartment.to_numpy()).all(), "the claws' compartments are not the loader's"
    for name, ref in (("apl_kc", wAPLKC), ("kc_apl", wKCAPL)):
        assert np.array_equal(ref.to_numpy(dtype=float), apl[name].to_numpy()), f"{name}: not the loader's weights"
    for name, ref in (("apl_pn", wAPLPN), ("pn_apl", wPNAPL)):
        r = ref.copy()
        r.index = pd.MultiIndex.from_tuples([tuple(x) for x in r.index], names=["glomerulus", "pn_id", "bouton_id"])
        assert np.array_equal(r.reindex(apl[name].index).to_numpy(dtype=float), apl[name].to_numpy()), \
            f"{name}: not the loader's weights"


# --------------------------------------------------------------------- tables


def _bundles(t: dict) -> dict[int, str]:
    b = t["boutons"]
    return dict(zip(b["bodyId"].astype(np.int64), b["PN_Bundle"])) if "PN_Bundle" in b.columns else {}


def claw_items(kc_id, claw_id, subtype) -> pd.DataFrame:
    kc_id, claw_id = pd.Series(kc_id).astype(np.int64).to_numpy(), pd.Series(claw_id).astype(np.int64).to_numpy()
    sub = pd.Series(subtype).reset_index(drop=True)
    return pd.DataFrame({"claw": [f"KC {k} claw {c}" for k, c in zip(kc_id, claw_id)], "kc_id": kc_id,
                         "claw_id": claw_id, "kc_type": sub.map(kc_class), "kc_subtype": sub})


def bouton_items(pn_id, bouton_id, pn_type, brain: str, bundles: dict) -> pd.DataFrame:
    pn_id, bouton_id = pd.Series(pn_id).astype(np.int64).to_numpy(), pd.Series(bouton_id).astype(np.int64).to_numpy()
    ptype = pd.Series(pn_type).reset_index(drop=True)
    glom = ptype.map(glomerulus_of)
    return pd.DataFrame({"bouton": [f"{g or 'PN'} {p} bouton {b}" for g, p, b in zip(glom, pn_id, bouton_id)],
                         "pn_id": pn_id, "bouton_id": bouton_id, "glomerulus": glom,
                         "model_glomerulus": model_glomeruli(ptype, brain), "pn_type": ptype,
                         "pn_bundle": [bundles.get(p) for p in pn_id]})


def apl_items(apl_id, unit, side: str) -> pd.DataFrame:
    apl_id = pd.Series(apl_id).astype(np.int64).to_numpy()
    if unit is None:
        return pd.DataFrame({"apl": "APL", "apl_id": apl_id, "apl_unit": pd.array([None] * len(apl_id), dtype="Int64")})
    unit = pd.Series(unit).astype(np.int64).to_numpy()
    return pd.DataFrame({"apl": [f"APL unit {u}" for u in unit], "apl_id": apl_id, "apl_unit": unit})


def raw_table(t: dict, name: str, brain: str, side: str) -> pd.DataFrame:
    """Every synapse of a connection of a calyx (its KC's or PN's side) counted per (pre unit, post unit), units
    labelled -1 kept, those with none left out: one line per pair, ``synapses`` its count."""
    _, pre, post = CONNECTIONS[name]
    s = t[name]
    s = s[(s.side == side) & s.pre_unit.notna() & s.post_unit.notna()]
    g = (s.groupby(["pre", "pre_unit", "post", "post_unit"], sort=True)
         .agg(synapses=("side", "size"), pre_type=("pre_type", "first"), post_type=("post_type", "first"))
         .reset_index())
    # (in the order of their rows' cells (KCs' claws, else PNs' boutons), then of their columns')
    by = ["post", "post_unit", "pre", "pre_unit"] if post == "claw" or (post == "bouton" and pre == "apl") else \
        ["pre", "pre_unit", "post", "post_unit"]
    g = g.sort_values(by).reset_index(drop=True)
    return _lines(g, pre, post, brain, side, _bundles(t), ("pre", "pre_unit", "pre_type"), ("post", "post_unit", "post_type"),
                  {"synapses": g.synapses.astype(np.int64).to_numpy()}, t["places"])


def _lines(g: pd.DataFrame, pre: str, post: str, brain: str, side: str, bundles: dict, pre_cols, post_cols,
           values: dict, places: dict) -> pd.DataFrame:
    def items(kind, cols):
        body, unit, typ = (g[c] if c in g.columns else None for c in cols)
        if kind == "claw":
            it = claw_items(body, unit, typ)
        elif kind == "bouton":
            it = bouton_items(body, unit, typ, brain, bundles)
        else:
            it = apl_items(body, unit, side)
        return pd.concat([it, place(unit_xyz(places, kind, body, unit), it[kind], kind)], axis=1)
    a, b = items(pre, pre_cols), items(post, post_cols)
    # (the KCs on the rows: claws the rows, else boutons, so a table's rows are its KCs or PNs, its columns the other end)
    rows, cols = (b, a) if post == "claw" or (post == "bouton" and pre == "apl") else (a, b)
    out = pd.concat([rows.reset_index(drop=True), cols.reset_index(drop=True), pd.DataFrame(values)], axis=1)
    return out


def model_tables(t: dict, brain: str, side: str, roi: str | None, check: bool) -> dict[str, pd.DataFrame]:
    """The model's tables of a calyx (see the module's docstring): one line per claw of the model and its bouton
    (``weight`` 1, ``synapses`` from its PN), and one per claw (or bouton) and APL (``weight``: their synapses)."""
    claws = model_claws(t, brain, side, roi)
    boutons = model_boutons(claws)
    apl = model_apl(t, brain, side, roi, claws, boutons)
    if check:
        check_hemibrain(claws, boutons, apl)
    bundles = _bundles(t)
    claw_place = place(claws[["x", "y", "z"]], claws.kc_id.astype(str) + "/" + claws.claw_id.astype(str), "claw")
    bouton_label = boutons.pn_id.astype(str) + "/" + boutons.bouton_id.astype(str)
    bouton_place = place(unit_xyz(t["places"], "bouton", boutons.pn_id, boutons.bouton_id), bouton_label, "bouton")
    claws_bouton_place = place(unit_xyz(t["places"], "bouton", claws.pn_id, claws.bouton_id),
                               claws.pn_id.astype(str) + "/" + claws.bouton_id.astype(str), "bouton")
    apl_ids = {name: t[name]["pre" if name.startswith("apl") else "post"][t[name].side == side].unique()
               for name in ("kc_apl", "apl_kc", "pn_apl", "apl_pn")}
    for name, ids in apl_ids.items():
        if len(ids) != 1:
            raise RuntimeError(f"{brain} {side} {name}: {len(ids)} APLs")
    out = {"pn_kc": pd.concat([claw_items(claws.kc_id, claws.claw_id, claws.kc_subtype), claw_place,
                               bouton_items(claws.pn_id, claws.bouton_id, claws.pn_type, brain, bundles), claws_bouton_place,
                               pd.DataFrame({"weight": np.ones(len(claws)), "synapses": claws.n_synapses.to_numpy()})], axis=1)}
    for name in ("kc_apl", "apl_kc"):
        out[name] = pd.concat([claw_items(claws.kc_id, claws.claw_id, claws.kc_subtype), claw_place,
                               apl_items(np.repeat(apl_ids[name][0], len(claws)), None, side),
                               pd.DataFrame({"weight": apl[name].to_numpy()})], axis=1)
    for name in ("pn_apl", "apl_pn"):
        out[name] = pd.concat([bouton_items(boutons.pn_id, boutons.bouton_id, boutons.pn_type, brain, bundles), bouton_place,
                               apl_items(np.repeat(apl_ids[name][0], len(boutons)), None, side),
                               pd.DataFrame({"weight": apl[name].to_numpy()})], axis=1)
    return out


# ------------------------------------------------------------------- manifest


def entry(calyx: str, name: str, version: str, df: pd.DataFrame) -> dict:
    title, pre, post = CONNECTIONS[name]
    rows_kind = "claw" if "claw" in (pre, post) else "bouton"
    cols_kind = next(k for k in (pre, post) if k != rows_kind)
    keep = lambda kind: FIELDS[kind][1:] + [c for c in (f"{kind}_{a}" for a in PLACE) if c in df.columns and df[c].notna().any()]
    n_rows = df[rows_kind].nunique()
    n_cols = df[cols_kind].nunique()
    unit = {"claw": "claw", "bouton": "bouton", "apl": "APL unit" if version == "raw" else "APL"}
    if version == "raw":
        what = (f"Every synapse of Pratyush's table counted per {unit[pre]} and {unit[post]} (his units: claws of KCs, boutons "
                "of PNs, units of APL), whatever its ROI and the cells' types; units labelled -1 (unassigned) kept as units "
                "of their own, synapses with none left out.")
        value_label = "synapses"
    elif name == "pn_kc":
        what = ("The model's wiring (olfsysm, prat_claws and prat_boutons, multibouton_claws='argmax'): each claw it keeps "
                "(5 or more synapses from one PN of its glomeruli, of a KC of its three types), 1 for the bouton it takes input "
                "from (of its PN's boutons, the one with the most of its synapses, ties to the lowest numbered; the claw's "
                "synapses from its PN in the synapses field).")
        value_label = "claw (1)"
    else:
        what = ("The model's APL weights (olfsysm, use_connectome_APL_weights with prat_claws and prat_boutons): the synapses "
                f"between APL and each of the model's {unit[rows_kind]}s, summed over APL's units (0 where none). "
                "The model normalizes them: only their ratios matter.")
        value_label = "synapses"
    if version == "raw":
        rule = "The KCs (or PNs) of the right side." if calyx == "hemibrain" else f"The KCs (or PNs) of the {calyx.split('-')[1]} side."
    elif calyx == "hemibrain":
        rule = "Synapses with both ends in CA(R) alone (olfsysm's loader)."
    else:
        rule = (f"The KCs (or PNs) of the {calyx.split('-')[1]} side, by the loader's steps with no ROI rule (as the "
                "KC-remapping queue's E97), the PN types' glomeruli renamed as the hemibrain's (VC3 and VC5 -> VC3, VM6 -> VC5).")
    where = (" Each claw, bouton and APL unit has its place: x, y, z (µm, each centered on the middle of its range over the "
             "table's units of its kind) and compartment (0 within half the largest distance from their centroid, else 1), "
             "as olfsysm places the model's claws; " + ("a model claw at the mean of its synapses from its PN (the loader's "
             "claw_x/y/z and compartment), other units" if version == "model" else "each unit") +
             " at the mean of the synapses Pratyush assigned to it (units labelled -1 have none).")
    kc_note = " Its rows are KCs' claws (kc_id: body ID), summed per KC in the Compare tab, on KCs." if rows_kind == "claw" else ""
    source = ("Pratyush's hemibrain tables (2025-12-05 run), as olfsysm's pyolfsysm/data/from_pratyush/2025-12-05 has them"
              if calyx == "hemibrain" else f"Pratyush's male CNS tables (run {MALECNS_STAMP}), Tom/claude_kc_remapping/data/Male-CNS_Tom on tensor")
    return {
        "id": f"connectome-{calyx}-{name.replace('_', '-')}-{version}",
        "title": f"{CALYX_TITLES[calyx][0].upper()}{CALYX_TITLES[calyx][1:]}: {title}, {'raw synapse counts' if version == 'raw' else 'as the model has it'}",
        "group": f"Connectome: {CALYX_TITLES[calyx]}",
        "description": f"{what} {rule}{where}{kc_note} {n_rows:,} {rows_kind}s × {n_cols:,} {unit[cols_kind]}{'s' if n_cols != 1 else ''}.",
        "file": f"connectome/{calyx}_{name}_{version}.parquet",
        "kind": "sequential",
        "value_label": value_label,
        "row_label": rows_kind if rows_kind != "claw" else "KC claw",
        "col_label": unit[cols_kind],
        "chemical_axis": "none",
        "fields": {"layout": "list", "rows": {"label": rows_kind, "keep": keep(rows_kind)},
                   "cols": {"label": cols_kind, "keep": keep(cols_kind)},
                   "value": "weight" if version == "model" else "synapses", "empty": "zero"},
        "source": f"{source}; made by reference_data/make_connectome.py",
    }


def main(argv: list[str]) -> None:
    malecns_dir = Path(argv[1]) if len(argv) > 1 else MALECNS_DIR
    OUT.mkdir(exist_ok=True)
    tables = {"hemibrain": read_brain("hemibrain"), "malecns": read_brain("malecns", malecns_dir)}
    entries = []
    for calyx, (brain, side, roi) in CALYCES.items():
        t = tables[brain]
        made = {"model": model_tables(t, brain, side, roi, check=brain == "hemibrain"),
                "raw": {name: raw_table(t, name, brain, side) for name in CONNECTIONS}}
        for name in CONNECTIONS:
            for version in ("model", "raw"):
                df = made[version][name]
                path = OUT / f"{calyx}_{name}_{version}.parquet"
                df.to_parquet(path, index=False, compression="zstd")
                entries.append(entry(calyx, name, version, df))
                print(f"{path.name}: {len(df):,} lines", file=sys.stderr)
    manifest = json.loads(MANIFEST.read_text())
    keep = [d for d in manifest["datasets"] if not d["id"].startswith("connectome-")]
    at = next((i for i, d in enumerate(keep) if d.get("example")), len(keep))  # (before the examples)
    manifest["datasets"] = keep[:at] + entries + keep[at:]
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv)
