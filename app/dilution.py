"""Serial dilutions of odors from their neat chemicals, as the Headspace calculator tab plans
them: each odor to its concentration in the mixture (log10 v/v), in vials of its own.

Into each vial, the same amount of solvent is put (``solvent_ml`` of it: paraffin oil
weighed by its density, as it is viscous; water measured by volume), then a volume of the
odor's neat chemical (into the first vial) or of the vial before is pipetted into it. Volumes are taken as additive: v µL into V µL of solvent
dilutes by v / (v + V). The neat chemical's volume changes from odor to odor, within its
limits; that of the dilutions is, where it can be, the same for every dilution of every
odor, so that their pipette can be left on one setting. It is chosen, of round volumes
(`NICE`) and of those that just make an odor's concentration with the most or least of
its neat chemical:

1. so that as many odors as can be are made with their neat chemicals within the limits;
2. within the dilutions' limits, unless no volume within them does as well (1.);
3. the same for as many odors as can be (each volume chosen so in turn, of those left);
4. with the fewest dilutions;
5. with the least volume pipetted (of the neat chemicals and the dilutions) as large as
   can be, as pipettes are least precise at their least, then the next least, and so on;
6. round.
"""
from __future__ import annotations

import math
from typing import Optional

SOLVENT_ML = 2.0  # into each vial, by default (mL)
NEAT_UL = (2.0, 200.0)  # the least and most of a neat chemical to pipette, by default (µL)
DILUTION_UL = 100.0  # the least of a vial to pipette into the next, by default (µL)
NICE = (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8)  # mantissas of the round volumes of dilutions tried (µL)
DIGITS = 3  # significant digits of the volumes to pipette
EXACT = 3  # dilutions more than the fewest, of an odor, whose volumes that just make it are tried


def _sig(v: float, how=round) -> float:
    """`v` (> 0) to `DIGITS` significant digits, by `how` (round, math.floor or math.ceil)."""
    e = math.floor(math.log10(v)) - DIGITS + 1
    m = v / 10.0 ** e
    m = math.floor(m + 1e-9) if how is math.floor else math.ceil(m - 1e-9) if how is math.ceil else round(m)
    return float(f"{m * 10.0 ** e:.{DIGITS}g}")


def _log10_fraction(v: float, solvent_ul: float) -> float:
    """log10 of the v/v of v µL pipetted into `solvent_ul` of solvent."""
    return math.log10(v / (v + solvent_ul))


def _neat(log10_conc: float, solvent_ul: float) -> float:
    """The µL of a neat chemical into `solvent_ul` of solvent for a concentration (log10 v/v,
    < 0), to `DIGITS`."""
    c = 10.0 ** log10_conc
    return _sig(solvent_ul * c / (1 - c))


def _chain(target: float, v: float, solvent_ul: float, neat_ul: tuple[float, float]) -> dict:
    """The fewest dilutions of v µL each that make `target` (log10 v/v, < 0) from a volume of
    its neat chemical within `neat_ul` (µL: the least, the most), else those that come
    nearest to it.

    Returns:
        dict with ``dilutions`` (how many), ``neat_ul`` (to `DIGITS`) and ``off`` (log10 of
        how many times the neat chemical's volume is outside its limits; 0 if within them).
    """
    lo, hi = neat_ul
    step = math.log10((v + solvent_ul) / v)  # (the decades each dilution takes)
    fewest = max(0, math.ceil((_log10_fraction(lo, solvent_ul) - target) / step - 1e-9))
    best = None
    for n in sorted({max(fewest - 1, 0), fewest}):  # (the one fewer, in case rounding makes it)
        c = target + n * step
        if c >= 0:
            continue
        neat = _neat(c, solvent_ul)
        off = max(0.0, math.log10(lo / neat), math.log10(neat / hi))
        if best is None or off < best["off"] - 1e-12:
            best = {"dilutions": n, "neat_ul": neat, "off": off if off > 1e-9 else 0.0}
    return best


def _odor(neat: Optional[float], dilutions: int, v: Optional[float], solvent_ul: float,
          neat_ul: tuple[float, float]) -> dict:
    """An odor's dilutions, as `plan` gives them, from its neat chemical's volume (µL; None for
    the neat chemical itself)."""
    if neat is None:
        return {"neat_ul": None, "dilution_ul": None, "vials": [], "ok": True,
                "notes": ["the neat chemical itself, undiluted"]}
    vials = [_log10_fraction(neat, solvent_ul)]
    for _ in range(dilutions):
        vials.append(vials[-1] + _log10_fraction(v, solvent_ul))
    lo, hi = neat_ul
    notes = []
    if neat < lo * (1 - 1e-9):
        notes.append(f"needs {neat:g} µL of the neat chemical, less than the least ({lo:g} µL)")
    elif neat > hi * (1 + 1e-9):
        notes.append(f"needs {neat:g} µL of the neat chemical, more than the most ({hi:g} µL)")
    return {"neat_ul": neat, "dilution_ul": v if dilutions else None, "vials": vials, "ok": not notes, "notes": notes}


def plan(targets: list[float], solvent_ml: float = SOLVENT_ML, neat_ul: tuple[float, float] = NEAT_UL,
         dilution_ul: tuple[Optional[float], Optional[float]] = (None, None)) -> dict:
    """How to make each of odors at its concentration from its neat chemical, by serial
    dilutions (see the module's docstring).

    Args:
        targets: each odor's concentration (log10 v/v, at most 0).
        solvent_ml: of solvent into each vial (mL).
        neat_ul: the least and the most of a neat chemical to pipette (µL).
        dilution_ul: the least and the most of a vial to pipette into the next (µL; None
            for the default: at least `DILUTION_UL`, or the most if less, and at most
            `solvent_ml`, as it is anyway), kept to only where the odors can be made as
            well within them (1., in the module's docstring).

    Returns:
        dict with ``odors``, for each target in order: ``neat_ul`` (µL of its neat chemical,
        into its first vial; None for a target of 0, the neat chemical itself),
        ``dilution_ul`` (µL of each of its vials into the next; None with no dilutions),
        ``vials`` (log10 v/v of each, from the volumes as rounded), ``ok`` (whether its neat
        chemical's volume is within its limits) and ``notes``; ``volumes``, those of the
        dilutions, each {"ul", "odors" (the indices of those diluted by it), "within"
        (whether within the dilutions' limits)}, the one for the most odors first; and
        ``dilution_limits`` (as applied: the least, the most).
    """
    solvent_ul = solvent_ml * 1e3
    lo_neat = neat_ul[0]
    hi = min(dilution_ul[1], solvent_ul) if dilution_ul[1] is not None else solvent_ul
    lo = dilution_ul[0] if dilution_ul[0] is not None else min(DILUTION_UL, hi)
    within = lambda v: lo <= v <= hi
    lo_any = min(lo, lo_neat)  # (what is tried, beyond the limits: at most a vial's solvent)

    odors: list[Optional[dict]] = [None] * len(targets)
    need = []  # (those with less than the least of a neat chemical)
    for i, t in enumerate(targets):
        if t >= 0:
            odors[i] = _odor(None, 0, None, solvent_ul, neat_ul)
        elif (neat := _neat(t, solvent_ul)) >= lo_neat:
            odors[i] = _odor(neat, 0, None, solvent_ul, neat_ul)
        else:
            need.append(i)

    nice = {float(f"{m * 10.0 ** k:.{DIGITS}g}") for m in NICE
            for k in range(math.floor(math.log10(lo_any)) - 1, math.ceil(math.log10(solvent_ul)) + 1)}
    volumes = {v for v in nice | {lo, hi} if lo_any <= v <= solvent_ul}
    a, b = (_log10_fraction(x, solvent_ul) for x in neat_ul)
    for i in need:  # (the volumes that just make it, of its fewest dilutions, overall and within the limits)
        firsts = {math.ceil((a - targets[i]) / math.log10((x + solvent_ul) / x) - 1e-9) for x in (lo_any, lo)}
        for n in {n for f in firsts for n in range(max(f, 1), f + EXACT + 1)}:
            for c, how in ((b, math.ceil), (a, math.floor)):
                if lo_any <= (v := _sig(solvent_ul / (10.0 ** ((c - targets[i]) / n) - 1), how)) <= solvent_ul:
                    volumes.add(v)

    chains = {v: {i: _chain(targets[i], v, solvent_ul, neat_ul) for i in need} for v in volumes}
    tiers = (lambda p, v: p["off"] == 0 and within(v), lambda p, v: p["off"] == 0, lambda p, v: True)
    left, used = set(need), []
    for ok in tiers:
        while left:
            best = None
            for v in sorted(volumes):
                group = sorted(i for i in left if ok(chains[v][i], v))
                if not group:
                    continue
                ps = [chains[v][i] for i in group]
                key = (-len(group), sum(p["off"] for p in ps), sum(p["dilutions"] for p in ps),
                       tuple(-x for x in sorted([v] + [p["neat_ul"] for p in ps])), v not in nice, -v)
                if best is None or key < best[0]:
                    best = (key, v, group)
            if best is None:
                break
            _, v, group = best
            for i in group:
                odors[i] = _odor(chains[v][i]["neat_ul"], chains[v][i]["dilutions"], v, solvent_ul, neat_ul)
            if diluted := [i for i in group if chains[v][i]["dilutions"]]:
                used.append({"ul": v, "odors": diluted, "within": within(v)})
            left -= set(group)
    for i in left:  # (with no volume to try: more of a neat chemical than a vial's solvent, at least)
        odors[i] = _odor(_neat(targets[i], solvent_ul), 0, None, solvent_ul, neat_ul)
    return {"odors": odors, "volumes": used, "dilution_limits": [lo, hi]}
