"""Headspace concentrations of odors diluted together in a solvent, as the Headspace
calculator tab computes them: at equilibrium, at 25 °C, as partial pressures (the browser
converts them to other units), from the properties of each looked up through chemutils
(chemprops.py).

- In paraffin oil (or any solvent but water), by Raoult's law for an ideal solution: the
  partial pressure of each odor is its mole fraction in the liquid times its vapor
  pressure. The moles of each are from its volume (v/v), density and molar mass, and those
  of the oil as if it were n-hexadecane (a light paraffin oil's molecules are about as
  large). Polar odors (alcohols, acids) are less soluble in oil than an ideal solution
  would be, so their headspace is likely higher than this when dilute.
- In water, by Henry's law: the partial pressure of each odor is its concentration in the
  water over its Henry's law solubility constant, else (with none found) by Raoult's law
  as above, which is too low for odors not very soluble in water. Henry's law holds for
  dilute solutions only: more than water dissolves of an odor has its vapor pressure.
- An odor's vapor pressure is the one PubChem gives, else one estimated from its structure
  (SIMPOL.1); its density, if none is found, is taken as `DEFAULT_DENSITY`.
"""
from __future__ import annotations


T_K = 298.15
# solvent -> (molar mass in g/mol, density in g/mL at 25 °C), with n-hexadecane for paraffin oil
SOLVENTS = {"paraffin oil": (226.45, 0.770), "water": (18.015, 0.997)}
DEFAULT_DENSITY = 1.0  # g/mL, for an odor with none found


def calculate(components: list[dict], solvent: str) -> dict:
    """The headspace of odors diluted together in a solvent (see the module's docstring).

    Args:
        components: each with ``log10_conc`` (log10 of its volume, pure, over that of the
            solution: v/v) and its properties as `chemprops.PROPERTIES` give them (None
            where none was found): ``vapor_pressure`` and ``est_vapor_pressure`` (kPa),
            ``molar_mass`` (g/mol), ``density`` (g/mL) and ``k_henry`` (mol/m³/Pa).
        solvent: of `SOLVENTS`.

    Returns:
        dict with ``components``, one for each given, in order: ``pa`` (its partial
        pressure in the headspace, Pa; None if it cannot be computed), ``law`` ('raoult'
        or 'henry'), ``mole_fraction`` (in the liquid, by Raoult's law), ``vapor_pressure``
        (kPa, as used) and ``notes`` (what was assumed, or why it cannot be computed); and
        ``total_pa`` (of those that can be computed).

    Raises:
        ValueError: if the odors are more than the whole solution.
    """
    fractions = [10.0 ** c["log10_conc"] for c in components]
    if sum(fractions) > 1 + 1e-9:
        raise ValueError(f"the odors are more than the whole solution (their v/v sum to {sum(fractions):.3g})")
    m_solvent, rho_solvent = SOLVENTS[solvent]
    out, moles = [], []
    for c, f in zip(components, fractions):
        notes = []
        p_sat = c.get("vapor_pressure")
        if p_sat is None and c.get("est_vapor_pressure") is not None:
            p_sat = c["est_vapor_pressure"]
            notes.append("no vapor pressure found, so estimated from its structure (SIMPOL.1)")
        rho = c.get("density")
        if rho is None:
            rho = DEFAULT_DENSITY
            notes.append(f"no density found, so taken as {DEFAULT_DENSITY:g} g/mL")
        mass = c.get("molar_mass")
        moles.append(f * rho / mass if mass else None)  # (per mL of solution)
        if not mass:
            notes.append("no molar mass, so its headspace cannot be computed")
        out.append({"pa": None, "law": None, "mole_fraction": None, "vapor_pressure": p_sat, "notes": notes})

    n_total = (1 - sum(fractions)) * rho_solvent / m_solvent + sum(n for n in moles if n is not None)
    if any(n is None for n in moles):
        for r, n in zip(out, moles):
            if n is not None:
                r["notes"].append("mole fraction without the odors with no molar mass")
    for c, r, n in zip(components, out, moles):
        if n is None:
            continue
        r["mole_fraction"] = n / n_total
        if solvent == "water" and c.get("k_henry"):
            r["law"] = "henry"
            r["pa"] = n * 1e6 / c["k_henry"]  # mol/mL as mol/m³, over mol/m³/Pa
            if r["vapor_pressure"] is not None and r["pa"] > r["vapor_pressure"] * 1e3:
                r["pa"] = r["vapor_pressure"] * 1e3
                r["notes"].append("more than water dissolves by Henry's law, so at its vapor pressure")
            continue
        r["law"] = "raoult"
        if solvent == "water":
            r["notes"].append("no Henry's law constant found, so as an ideal solution (Raoult's law): "
                              "likely too low, unless it is very soluble in water")
        if r["vapor_pressure"] is None:
            r["notes"].append("no vapor pressure, so its headspace cannot be computed")
            continue
        r["pa"] = r["mole_fraction"] * r["vapor_pressure"] * 1e3
    return {"components": out, "total_pa": sum(r["pa"] for r in out if r["pa"] is not None)}
