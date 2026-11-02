"""Headspace concentrations of odors diluted together in a solvent, as the Headspace
calculator tab computes them: at equilibrium, at 25 °C, as partial pressures (the browser
converts them to other units), from the properties of each looked up through chemutils
(chemprops.py).

- In paraffin oil, by Henry's law (`MODELS` 'henry', the default): the partial pressure of
  each odor is its concentration in the oil over its Henry's law solubility constant
  there, K_H = c_oil / (γ∞ p_sat), as chemutils estimates it for n-hexadecane
  (`inchi2k_henry_pfo`; no measured solubilities in oil), but of the oil: c_oil is the
  oil's moles in a volume, from its density and molar mass, and γ∞ the odor's activity
  coefficient when dilute by the same modified UNIFAC, in the n-alkane of about the oil's
  molar mass (`_gamma_inf`), as a larger solvent lowers γ∞ as well as c_oil. That is
  Raoult's law with that activity coefficient: polar odors (alcohols, acids) are less
  soluble in oil than an ideal solution, so more of them is in the headspace. With no
  constant estimated, or by choice ('raoult'), by Raoult's law for an ideal solution: the
  partial pressure of each odor is its mole fraction in the liquid times its vapor
  pressure. The moles of each are from its volume (v/v), density and molar mass, and those
  of the oil from its density and molar mass (`SOLVENTS`, unless others are given).
- In water, by Henry's law: the partial pressure of each odor is its concentration in the
  water over its Henry's law solubility constant, else (with none found) by Raoult's law
  as above, which is too low for odors not very soluble in water.
- Henry's law holds for dilute solutions only: more than the solvent dissolves of an odor
  has its vapor pressure.
- An odor's vapor pressure is the one PubChem gives, else one estimated from its structure
  (SIMPOL.1); in oil by Henry's law, the one chemutils estimated the constant with. Its
  density, if none is found, is taken as `DEFAULT_DENSITY`.

Also (`sample`) the headspace over samples taken as water, of volatiles quantified by mass
(mg/kg, as VCF gives them for foods: `vcf.py`), and (`invert`) the concentrations of odors
in a solvent that give a headspace, e.g. a food's.
"""
from __future__ import annotations

import collections
import functools
import itertools
import math
from typing import Optional

T_K = 298.15
P_ATM = 101325.0  # Pa: the air's total pressure, for parts of it by volume
R = 8.314462618  # J/(mol K)
K_B = 1.380649e-23  # J/K
# solvent -> (molar mass in g/mol, density in g/mL at 25 °C). Paraffin oil's are by default those
# of a viscous one (of 110-230 mPa·s at 20 °C, as Ph. Eur. has liquid paraffin): its molecules
# are a mixture, averaging from about 400 to 500 g/mol, and light ones less (see design.md).
SOLVENTS = {"paraffin oil": (450.0, 0.85), "water": (18.015, 0.997)}
# what can be given of paraffin oil (least, most), so as not to compute from a typing error
OIL_MOLAR_MASS_RANGE = (100.0, 2000.0)  # g/mol
OIL_DENSITY_RANGE = (0.5, 1.2)  # g/mL
# chemutils' `unifac_parameters` -> thermo.unifac's (as `inchi2k_henry_pfo` takes them)
UNIFAC_PARAMETERS = {"Dortmund 2016": "DOUFIP2016", "modified UNIFAC 2.0": "DOUF2IP"}
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


def _solvent(solvent: str, oil_molar_mass: Optional[float], oil_density: Optional[float]) -> tuple[float, float]:
    """A solvent's molar mass (g/mol) and density (g/mL): `SOLVENTS`', but paraffin oil's as given."""
    mass, density = SOLVENTS[solvent]
    if solvent == "paraffin oil":
        return oil_molar_mass or mass, oil_density or density
    return mass, density


def alkane_carbons(molar_mass: float) -> int:
    """The carbons of the n-alkane (C_n H_2n+2) of about a molar mass (g/mol)."""
    return max(2, round((molar_mass - 2.016) / 14.027))


@functools.lru_cache(maxsize=4096)
def _gamma_inf(groups: str, parameters: str, in_hexadecane: float, carbons: int) -> float:
    """An odor's activity coefficient when dilute in an n-alkane, by modified UNIFAC, from
    what chemutils gives of it in n-hexadecane (`inchi2k_henry_pfo`).

    Args:
        groups: its ``unifac_groups`` (e.g. ``CH3:1 CH2:1 CH3COO:1``).
        parameters: its ``unifac_parameters`` (of `UNIFAC_PARAMETERS`).
        in_hexadecane: its ``gamma_inf``.
        carbons: of the n-alkane.

    Raises:
        KeyError: for groups or parameters thermo does not have.
        ValueError: for groups that cannot be read.
    """
    from thermo import unifac  # (here, as only Henry's law in oil needs it)

    interactions = getattr(unifac, UNIFAC_PARAMETERS[parameters])
    named = collections.defaultdict(list)
    for key, group in unifac.DOUFSG.items():
        named[group.group].append(key)

    def gamma(solute: dict[int, int], n: int) -> float:
        model = unifac.UNIFAC.from_subgroups(T=T_K, xs=[0.5, 0.5], chemgroups=[solute, {1: 2, 2: n - 2}], version=1,
                                             interaction_data=interactions, subgroups=unifac.DOUFSG)
        return float(model.gammas_infinite_dilution()[0])

    counts = [g.rsplit(":", 1) for g in groups.split()]
    # (names are not unique, 'CHO' being an aldehyde's and an ether's: those that give chemutils' value)
    options = [{k: int(n) for k, (_, n) in zip(keys, counts)} for keys in itertools.product(*(named[name] for name, _ in counts))]
    if not options or not all(options):
        raise KeyError(groups)
    solute = min(options, key=lambda o: abs(gamma(o, 16) - in_hexadecane))
    return in_hexadecane * gamma(solute, carbons) / gamma(solute, 16)


def _law(c: dict, solvent: str, henry: Optional[str], r: dict, molar_mass: float, density: float) -> None:
    """How an odor's headspace over a solvent is computed, into `r` (as `calculate` gives
    it): ``law`` ('henry' if `henry`, the property with its Henry's law constant there, is
    known of it, else 'raoult'), ``k_henry`` and ``k_henry_measured`` (by Henry's law; in
    oil, of the oil's `molar_mass` and `density`, with ``gamma_inf``, ``gamma_inf_hexadecane``
    and ``unifac_groups``), the vapor pressure used and whether it was measured (by Henry's
    law, at most it), and what was assumed (``notes``)."""
    details = (c.get(f"{henry}_details") or {}) if henry else {}
    k = c.get(henry) if henry else None
    if k and solvent == "paraffin oil":
        try:
            gamma = _gamma_inf(details["unifac_groups"], details["unifac_parameters"], details["gamma_inf"], alkane_carbons(molar_mass))
            k = density / molar_mass * 1e6 / (gamma * details["vapor_pressure_kpa"] * 1e3)  # mol/mL as mol/m³, kPa as Pa
        except (KeyError, TypeError, ValueError, AttributeError):  # (nothing kept to compute it from)
            details = {"notes": "its UNIFAC groups could not be read"}
            k = None
        else:
            r.update(gamma_inf=gamma, gamma_inf_hexadecane=details["gamma_inf"], unifac_groups=details["unifac_groups"])
    if k:
        r.update(law="henry", k_henry=k, k_henry_measured=c.get(f"{henry}_measured"))
        if details.get("vapor_pressure_kpa") is not None:  # (the one the constant was estimated with)
            r.update(vapor_pressure=details["vapor_pressure_kpa"], vapor_pressure_measured=details.get("vapor_pressure_measured"),
                     vapor_pressure_source=details.get("vapor_pressure_source"))
        else:
            r["vapor_pressure"], r["vapor_pressure_measured"] = _vapor_pressure(c, r["notes"])
        if details.get("notes"):
            r["notes"].append(details["notes"])
        return
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


def calculate(components: list[dict], solvent: str, model: str = MODELS[0], oil_molar_mass: Optional[float] = None,
              oil_density: Optional[float] = None) -> dict:
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
        oil_molar_mass: of paraffin oil (g/mol; `SOLVENTS`' if None), for it.
        oil_density: of paraffin oil (g/mL; `SOLVENTS`' if None), for it.

    Returns:
        dict with ``components``, one for each given, in order: ``pa`` (its partial
        pressure in the headspace, Pa; None if it cannot be computed), ``law`` ('raoult'
        or 'henry'), ``mole_fraction`` (in the liquid), ``vapor_pressure`` (kPa, as used),
        ``vapor_pressure_measured`` (whether from measurements; None if not known),
        ``vapor_pressure_source`` (if not of the odor table's: that of a Henry's law
        constant estimated with it), ``k_henry`` (mol/m³/Pa, as used, by Henry's law),
        ``k_henry_measured`` and ``notes`` (what was assumed, or why it cannot be
        computed), with the steps to its ``pa``: ``fraction`` (v/v), ``density`` (g/mL,
        as used), ``moles`` (mol in a mL of the solution; None with no molar mass),
        ``gamma_inf`` (by Henry's law in oil: its activity coefficient when dilute in it),
        ``gamma_inf_hexadecane`` (chemutils', in n-hexadecane), ``unifac_groups`` and
        ``henry_pa`` (by Henry's law, Pa, before it is at most its vapor pressure); and
        ``total_pa`` (of those that can be computed), ``solvent_molar_mass`` (g/mol) and
        ``solvent_density`` (g/mL) as taken, ``alkane_carbons`` (in oil, of the n-alkane
        ``gamma_inf`` is in; else None), ``solvent_moles`` (mol in a mL of the solution) and
        ``total_moles`` (of the solvent and the odors with a molar mass).

    Raises:
        ValueError: if the odors are more than the whole solution.
    """
    fractions = [10.0 ** c["log10_conc"] for c in components]
    if sum(fractions) > 1 + 1e-9:
        raise ValueError(f"the odors are more than the whole solution (their v/v sum to {sum(fractions):.3g})")
    m_solvent, rho_solvent = _solvent(solvent, oil_molar_mass, oil_density)
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
                    "vapor_pressure_source": None, "k_henry": None, "k_henry_measured": None, "notes": notes,
                    "fraction": f, "density": rho, "moles": moles[-1], "gamma_inf": None, "gamma_inf_hexadecane": None,
                    "unifac_groups": None, "henry_pa": None})

    n_solvent = (1 - sum(fractions)) * rho_solvent / m_solvent
    n_total = n_solvent + sum(n for n in moles if n is not None)
    if any(n is None for n in moles):
        for r, n in zip(out, moles):
            if n is not None:
                r["notes"].append("mole fraction without the odors with no molar mass")
    for c, r, n in zip(components, out, moles):
        if n is None:
            continue
        r["mole_fraction"] = n / n_total
        _law(c, solvent, henry, r, m_solvent, rho_solvent)
        if r["law"] == "henry":
            r["pa"] = r["henry_pa"] = n * 1e6 / r["k_henry"]  # mol/mL as mol/m³, over mol/m³/Pa
            if r["vapor_pressure"] is not None and r["pa"] > r["vapor_pressure"] * 1e3:
                r["pa"] = r["vapor_pressure"] * 1e3
                r["notes"].append(f"more than {solvent} dissolves by Henry's law, so at its vapor pressure")
        elif r["vapor_pressure"] is not None:
            r["pa"] = r["mole_fraction"] * r["vapor_pressure"] * 1e3
    return {"components": out, "total_pa": sum(r["pa"] for r in out if r["pa"] is not None), "solvent_molar_mass": m_solvent,
            "solvent_density": rho_solvent, "alkane_carbons": alkane_carbons(m_solvent) if solvent == "paraffin oil" else None,
            "solvent_moles": n_solvent, "total_moles": n_total}


def sample(components: list[dict]) -> list[dict]:
    """The headspace over samples taken as water (e.g. foods, as VCF quantifies their volatiles:
    `vcf.py`), each volatile on its own (they are dilute): by Henry's law, else (with no
    constant) by Raoult's law for an ideal solution, which is too low for volatiles not
    very soluble in water.

    Args:
        components: each with ``ppm`` (mg/kg of the sample), ``molar_mass`` (g/mol),
            ``k_henry`` (in water, mol/m³/Pa) and ``vapor_pressure`` (kPa); None where not known.

    Returns:
        For each, in order: ``pa`` (its partial pressure in the headspace, Pa; None if it
        cannot be computed), ``law`` ('henry', 'raoult' or None) and ``notes``.
    """
    m_water, rho_water = SOLVENTS["water"]
    out = []
    for c in components:
        r = {"pa": None, "law": None, "notes": []}
        out.append(r)
        mass, k, vp = c.get("molar_mass"), c.get("k_henry"), c.get("vapor_pressure")
        if not mass:
            r["notes"].append("no molar mass, so its headspace cannot be computed")
            continue
        if c["ppm"] >= 1e6:
            r["notes"].append("more than the whole sample (more than 10⁶ mg/kg: likely in other units), so not computed")
            continue
        n = c["ppm"] * 1e-3 / mass  # mol/kg of the sample
        if k:
            r["law"] = "henry"
            r["pa"] = n * rho_water * 1e3 / k  # mol/m³ over mol/m³/Pa
            if vp is not None and r["pa"] > vp * 1e3:
                r["pa"] = vp * 1e3
                r["notes"].append("more than water dissolves by Henry's law, so at its vapor pressure")
        elif vp is not None:
            r["law"] = "raoult"
            r["notes"].append("no Henry's law constant found, so as an ideal solution (Raoult's law): "
                              "likely too low, unless it is very soluble in water")
            r["pa"] = n / (n + (1 - c["ppm"] * 1e-6) * 1e3 / m_water) * vp * 1e3
        else:
            r["notes"].append("no Henry's law constant or vapor pressure found, so its headspace cannot be computed")
    return out


def invert(components: list[dict], solvent: str = "paraffin oil", model: str = MODELS[0], oil_molar_mass: Optional[float] = None,
           oil_density: Optional[float] = None) -> dict:
    """The concentrations (v/v) of odors diluted together in a solvent that give each a
    headspace: `calculate` backwards (of the mixture of those that can be computed).

    By Henry's law, an odor's concentration is from its own headspace alone; by Raoult's
    law, from its mole fraction, x = p / p_sat, so that with every odor's moles n (with
    those of the solvent) N = (n_solvent (1 - F_H) + n_H) / (1 - Σ x (1 - n_solvent / n_pure)),
    over those by Raoult's law (F_H and n_H the volume and moles of those by Henry's, and
    each n the moles of a mL, pure).

    Args:
        components: each with ``pa`` (its headspace, Pa; None to leave it out) and its
            properties, as `calculate` takes them.
        solvent: of `SOLVENTS`.
        model: of `MODELS`, for paraffin oil.
        oil_molar_mass: of paraffin oil (g/mol; `SOLVENTS`' if None), for it.
        oil_density: of paraffin oil (g/mL; `SOLVENTS`' if None), for it.

    Returns:
        dict with ``components``, one for each given, in order: ``log10_conc`` (its
        concentration, log10 v/v; None if it cannot be computed or reached), as
        `calculate` gives them but ``pa`` (as given) and without the steps to it (but
        ``gamma_inf``, ``gamma_inf_hexadecane`` and ``unifac_groups``); ``total`` (the v/v
        of those computed, together; more than 1 if they cannot be); and
        ``solvent_molar_mass`` and ``solvent_density``, as taken.
    """
    m_solvent, rho_solvent = _solvent(solvent, oil_molar_mass, oil_density)
    a_solvent = rho_solvent / m_solvent  # (mol/mL, pure)
    henry = HENRY_PROPERTIES[solvent] if solvent == "water" or model == "henry" else None
    out, pure = [], []
    for c in components:
        r = {"log10_conc": None, "pa": c.get("pa"), "law": None, "mole_fraction": None, "vapor_pressure": None,
             "vapor_pressure_measured": None, "vapor_pressure_source": None, "k_henry": None, "k_henry_measured": None,
             "gamma_inf": None, "gamma_inf_hexadecane": None, "unifac_groups": None, "notes": []}
        out.append(r)
        pure.append(None)
        if r["pa"] is None:
            continue
        if r["pa"] <= 0:
            r["notes"].append("no headspace, so none of it")
            continue
        rho = c.get("density")
        if rho is None:
            rho = DEFAULT_DENSITY
            r["notes"].append(f"no density found, so taken as {DEFAULT_DENSITY:g} g/mL")
        if not c.get("molar_mass"):
            r["notes"].append("no molar mass, so its concentration cannot be computed")
            continue
        _law(c, solvent, henry, r, m_solvent, rho_solvent)
        if r["law"] == "raoult" and r["vapor_pressure"] is None:
            continue
        if r["vapor_pressure"] is not None and r["pa"] >= r["vapor_pressure"] * 1e3:
            r["notes"].append("its headspace is at least its vapor pressure, so it cannot be reached in a solution")
            continue
        pure[-1] = rho / c["molar_mass"]

    fraction = [None] * len(out)
    for i, (r, a) in enumerate(zip(out, pure)):
        if a is not None and r["law"] == "henry":
            fraction[i] = r["pa"] * r["k_henry"] / 1e6 / a
    by_raoult = [i for i, (r, a) in enumerate(zip(out, pure)) if a is not None and r["law"] == "raoult"]
    f_henry = sum(f for f in fraction if f is not None)
    n_henry = sum(f * pure[i] for i, f in enumerate(fraction) if f is not None)
    x = {i: out[i]["pa"] / (out[i]["vapor_pressure"] * 1e3) for i in by_raoult}
    rest = 1 - sum(x[i] * (1 - a_solvent / pure[i]) for i in by_raoult)
    n_total = (a_solvent * (1 - f_henry) + n_henry) / rest if rest > 0 else None
    for i in by_raoult:
        if n_total is None:
            out[i]["notes"].append("together, more than the whole solution would be needed by Raoult's law, so not computed")
        else:
            fraction[i] = x[i] * n_total / pure[i]
    if n_total is None:  # (the mole fractions of those by Henry's law, without those by Raoult's)
        n_total = a_solvent * (1 - f_henry) + n_henry
    for r, f, a in zip(out, fraction, pure):
        if f is not None:
            r["log10_conc"] = math.log10(f)
            r["mole_fraction"] = f * a / n_total
    return {"components": out, "total": sum(f for f in fraction if f is not None), "solvent_molar_mass": m_solvent,
            "solvent_density": rho_solvent}
