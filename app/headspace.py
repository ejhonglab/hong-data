"""Headspace concentrations of odors diluted together in a solvent, as the Headspace
calculator tab computes them: at equilibrium, at 25 °C, as partial pressures (the browser
converts them to other units), from the properties of each looked up through chemutils
(chemprops.py).

- In paraffin oil, by Henry's law (`MODELS` 'henry', the default): the partial pressure of
  each odor is its concentration in the oil over its Henry's law solubility constant
  there, as chemutils estimates it (`inchi2k_henry_pfo`: its activity coefficient in
  n-hexadecane by modified UNIFAC, and its vapor pressure; no measured solubilities in
  oil). That is Raoult's law with the activity coefficient of the odor when dilute:
  polar odors (alcohols, acids) are less soluble in oil than an ideal solution, so more
  of them is in the headspace. With no constant estimated, or by choice ('raoult'), by
  Raoult's law for an ideal solution: the partial pressure of each odor is its mole
  fraction in the liquid times its vapor pressure. The moles of each are from its volume
  (v/v), density and molar mass, and those of the oil as if it were n-hexadecane (a light
  paraffin oil's molecules are about as large).
- In water, by Henry's law: the partial pressure of each odor is its concentration in the
  water over its Henry's law solubility constant, else (with none found) by Raoult's law
  as above, which is too low for odors not very soluble in water.
- Henry's law holds for dilute solutions only: more than the solvent dissolves of an odor
  has its vapor pressure.
- An odor's vapor pressure is the one PubChem gives, else one estimated from its structure
  (SIMPOL.1); in oil by Henry's law, the one chemutils estimated the constant with. Its
  density, if none is found, is taken as `DEFAULT_DENSITY`.
"""
from __future__ import annotations

import math
from typing import Optional

T_K = 298.15
P_ATM = 101325.0  # Pa: the air's total pressure, for parts of it by volume
R = 8.314462618  # J/(mol K)
K_B = 1.380649e-23  # J/K
# solvent -> (molar mass in g/mol, density in g/mL at 25 °C), with n-hexadecane for paraffin oil
SOLVENTS = {"paraffin oil": (226.45, 0.770), "water": (18.015, 0.997)}
# how the headspace over paraffin oil is computed (see the module's docstring), the default first
MODELS = ("henry", "raoult")
# the property with each solvent's Henry's law constants (`chemprops.PROPERTIES`), with Henry's law
HENRY_PROPERTIES = {"paraffin oil": "k_henry_pfo", "water": "k_henry"}
DEFAULT_DENSITY = 1.0  # g/mL, for an odor with none found
# what a headspace can be given in, as the browser shows it: key -> (label, short label, the
# short label as LaTeX, of(partial pressure in Pa, molar mass in g/mol))
UNITS = {
    "ppm": ("ppm (by volume)", "ppm", "ppm", lambda pa, mw: pa / P_ATM * 1e6),
    "ppb": ("ppb (by volume)", "ppb", "ppb", lambda pa, mw: pa / P_ATM * 1e9),
    "log10": ("log10 v/v of the air", "log10 v/v", r"$\log_{10}$ v/v", lambda pa, mw: math.log10(pa / P_ATM) if pa > 0 else -math.inf),
    "pa": ("Pa (partial pressure)", "Pa", "Pa", lambda pa, mw: pa),
    "mol_l": ("mol/L", "mol/L", "mol/L", lambda pa, mw: pa / (R * T_K) / 1000),
    "molecules": ("molecules/cm³", "molecules/cm³", r"molecules/cm$^3$", lambda pa, mw: pa / (K_B * T_K) * 1e-6),
    "ug_l": ("µg/L (= mg/m³)", "µg/L", r"$\mu$g/L", lambda pa, mw: pa * mw / (R * T_K) * 1e3 if mw else None),
}
# names (or abbreviations) of solvents in the odor table, left out of the mixtures of panels
SOLVENT_NAMES = {"pfo", "paraffin oil", "mineral oil", "water", "h2o", "mineral oil (light)"}


def _vapor_pressure(c: dict, notes: list[str]) -> tuple[Optional[float], Optional[bool]]:
    """An odor's vapor pressure (kPa), and whether it is from measurements: PubChem's, else
    estimated from its structure (said so in `notes`)."""
    if c.get("vapor_pressure") is not None:
        return c["vapor_pressure"], c.get("vapor_pressure_measured")
    if c.get("est_vapor_pressure") is not None:
        notes.append("no vapor pressure found, so estimated from its structure (SIMPOL.1)")
        return c["est_vapor_pressure"], False
    return None, None


def calculate(components: list[dict], solvent: str, model: str = MODELS[0]) -> dict:
    """The headspace of odors diluted together in a solvent (see the module's docstring).

    Args:
        components: each with ``log10_conc`` (log10 of its volume, pure, over that of the
            solution: v/v) and its properties as `chemprops.PROPERTIES` give them (None
            where none was found): ``vapor_pressure`` and ``est_vapor_pressure`` (kPa),
            ``molar_mass`` (g/mol), ``density`` (g/mL), ``k_henry`` and ``k_henry_pfo``
            (mol/m³/Pa); with ``<property>_measured`` and ``k_henry_pfo_details`` as
            `chemprops.look_up` gives them.
        solvent: of `SOLVENTS`.
        model: of `MODELS`, for paraffin oil.

    Returns:
        dict with ``components``, one for each given, in order: ``pa`` (its partial
        pressure in the headspace, Pa; None if it cannot be computed), ``law`` ('raoult'
        or 'henry'), ``mole_fraction`` (in the liquid), ``vapor_pressure`` (kPa, as used),
        ``vapor_pressure_measured`` (whether from measurements; None if not known),
        ``vapor_pressure_source`` (if not of the odor table's: that of a Henry's law
        constant estimated with it), ``k_henry`` (mol/m³/Pa, as used, by Henry's law),
        ``k_henry_measured`` and ``notes`` (what was assumed, or why it cannot be
        computed); and ``total_pa`` (of those that can be computed).

    Raises:
        ValueError: if the odors are more than the whole solution.
    """
    fractions = [10.0 ** c["log10_conc"] for c in components]
    if sum(fractions) > 1 + 1e-9:
        raise ValueError(f"the odors are more than the whole solution (their v/v sum to {sum(fractions):.3g})")
    m_solvent, rho_solvent = SOLVENTS[solvent]
    henry = HENRY_PROPERTIES[solvent] if solvent == "water" or model == "henry" else None
    out, moles = [], []
    for c, f in zip(components, fractions):
        notes = []
        rho = c.get("density")
        if rho is None:
            rho = DEFAULT_DENSITY
            notes.append(f"no density found, so taken as {DEFAULT_DENSITY:g} g/mL")
        mass = c.get("molar_mass")
        moles.append(f * rho / mass if mass else None)  # (per mL of solution)
        if not mass:
            notes.append("no molar mass, so its headspace cannot be computed")
        out.append({"pa": None, "law": None, "mole_fraction": None, "vapor_pressure": None, "vapor_pressure_measured": None,
                    "vapor_pressure_source": None, "k_henry": None, "k_henry_measured": None, "notes": notes})

    n_total = (1 - sum(fractions)) * rho_solvent / m_solvent + sum(n for n in moles if n is not None)
    if any(n is None for n in moles):
        for r, n in zip(out, moles):
            if n is not None:
                r["notes"].append("mole fraction without the odors with no molar mass")
    for c, r, n in zip(components, out, moles):
        if n is None:
            continue
        r["mole_fraction"] = n / n_total
        details = (c.get(f"{henry}_details") or {}) if henry else {}
        if henry and c.get(henry):
            r.update(law="henry", k_henry=c[henry], k_henry_measured=c.get(f"{henry}_measured"))
            if details.get("vapor_pressure_kpa") is not None:  # (the one the constant was estimated with)
                r.update(vapor_pressure=details["vapor_pressure_kpa"], vapor_pressure_measured=details.get("vapor_pressure_measured"),
                         vapor_pressure_source=details.get("vapor_pressure_source"))
            else:
                r["vapor_pressure"], r["vapor_pressure_measured"] = _vapor_pressure(c, r["notes"])
            if details.get("notes"):
                r["notes"].append(details["notes"])
            r["pa"] = n * 1e6 / c[henry]  # mol/mL as mol/m³, over mol/m³/Pa
            if r["vapor_pressure"] is not None and r["pa"] > r["vapor_pressure"] * 1e3:
                r["pa"] = r["vapor_pressure"] * 1e3
                r["notes"].append(f"more than {solvent} dissolves by Henry's law, so at its vapor pressure")
            continue
        r["law"] = "raoult"
        if solvent == "water":
            r["notes"].append("no Henry's law constant found, so as an ideal solution (Raoult's law): "
                              "likely too low, unless it is very soluble in water")
        elif henry:
            why = f" ({details['notes']})" if details.get("notes") else ""
            r["notes"].append(f"no Henry's law constant could be estimated in oil{why}, so as an ideal solution (Raoult's law)")
        r["vapor_pressure"], r["vapor_pressure_measured"] = _vapor_pressure(c, r["notes"])
        if r["vapor_pressure"] is None:
            r["notes"].append("no vapor pressure, so its headspace cannot be computed")
            continue
        r["pa"] = r["mole_fraction"] * r["vapor_pressure"] * 1e3
    return {"components": out, "total_pa": sum(r["pa"] for r in out if r["pa"] is not None)}
