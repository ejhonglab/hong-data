"""Odor-string parsing and formatting.

A deliberately small port of the conventions in ``hong2p.olf`` (olf.py):

* a single odor is written ``'<name> @ <log10_conc>'`` (``'va @ -3'``), with a single
  space on either side of ``'@'``; ``log10_conc`` is 0 or negative
* ``'solvent'`` stands for "no odor on this manifold"
* several ``odor<n>`` index levels on one trial are components of an air mixture;
  they are displayed joined with ``' + '`` after dropping solvent components
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

SOLVENT_STR = "solvent"
CONC_DELIM = "@"
COMPONENT_DELIM = " + "
COMPONENT_LEVEL_PREFIX = "odor"

# '<name> @ <conc>' with exactly one space either side of '@'. Name may contain spaces
# (e.g. 'ethyl acetate @ -2') but not '@'.
_ODOR_RE = re.compile(r"^(?P<name>[^@\s][^@]*?) @ (?P<conc>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)$")

# Abbreviations hardcoded in olf.py, used as a fallback when an odor YAML entry has no
# explicit 'abbrev'.
HARDCODED_ODOR2ABBREV = {
    "hexyl hexanoate": "hh",
    "furfural": "fur",
    "1-octen-3-ol": "1o3ol",
    "acetone": "ace",
    "butanal": "but",
    "ethyl acetate": "ea",
    "ethyl hexanoate": "eh",
    "hexyl acetate": "ha",
    "ethanol": "EtOH",
    "isoamyl alcohol": "IAol",
    "kiwi approx.": "kmix",
    "control mix": "cmix",
    "ethyl lactate": "elac",
    "methyl acetate": "ma",
    "2,3-butanedione": "2,3-b",
    "ethyl 3-hydroxybutyrate": "e3hb",
    "trans-2-hexenal": "t2h",
    "ethyl crotonate": "ecrot",
    "methyl octanoate": "moct",
    "linalool": "Lin",
    "B-citronellol": "B-cit",
    "hexanal": "6al",
    "benzaldehyde": "benz",
    "1-pentanol": "1-5ol",
    "1-octanol": "1-8ol",
    "pentyl acetate": "pa",
    "ethyl propionate": "ep",
    "acetic acid": "aa",
    "methyl salicylate": "ms",
    "2-heptanone": "2h",
    "ethyl butyrate": "eb",
    "1-hexanol": "1-6ol",
    "isoamyl acetate": "IaA",
    "2-butanone": "2-but",
    "valeric acid": "va",
    "pentanoic acid": "va",
}


class OdorFormatError(ValueError):
    """Raised for an odor string that does not follow '<name> @ <log10_conc>'."""


@dataclass(frozen=True)
class Odor:
    name: str
    log10_conc: Optional[float]  # None for solvent

    @property
    def is_solvent(self) -> bool:
        return self.log10_conc is None

    def format(self) -> str:
        if self.is_solvent:
            return SOLVENT_STR
        return f"{self.name} {CONC_DELIM} {format_conc(self.log10_conc)}"


def format_conc(log10_conc: float) -> str:
    """-3.0 -> '-3', -1.5 -> '-1.5', 0 -> '0' (matches olf's int-casting behaviour)."""
    if log10_conc is None:
        return ""
    if math.isclose(log10_conc, round(log10_conc), abs_tol=1e-9):
        return str(int(round(log10_conc)))
    return repr(float(log10_conc))


def parse_odor(odor_str: str, *, strict: bool = True) -> Odor:
    """Parse '<name> @ <log10_conc>' or 'solvent'.

    Raises OdorFormatError describing the problem. With ``strict`` (the default) the
    concentration must be 0 or negative.
    """
    if not isinstance(odor_str, str):
        raise OdorFormatError(f"odor value is not a string: {odor_str!r}")
    s = odor_str
    if s == SOLVENT_STR:
        return Odor(SOLVENT_STR, None)
    if s != s.strip():
        raise OdorFormatError(f"{odor_str!r} has leading/trailing whitespace")
    m = _ODOR_RE.match(s)
    if m is None:
        if CONC_DELIM not in s:
            raise OdorFormatError(
                f"{odor_str!r} has no '@': expected '<odor-abbreviation> @ <log10_conc>'"
            )
        if s.count(CONC_DELIM) > 1:
            raise OdorFormatError(f"{odor_str!r} contains more than one '@'")
        name_part, conc_part = (x.strip() for x in s.split(CONC_DELIM))
        if not name_part:
            raise OdorFormatError(f"{odor_str!r} has an empty odor name before '@'")
        try:
            float(conc_part)
        except ValueError:
            raise OdorFormatError(
                f"{odor_str!r}: concentration {conc_part!r} after '@' is not a number"
            ) from None
        raise OdorFormatError(
            f"{odor_str!r} must have exactly one space on each side of '@' "
            f"(expected {name_part + ' @ ' + conc_part!r})"
        )
    name = m.group("name")
    conc = float(m.group("conc"))
    if strict and conc > 0:
        raise OdorFormatError(
            f"{odor_str!r}: log10 concentration must be 0 or negative, got {conc:g}"
        )
    if math.isnan(conc) or math.isinf(conc):
        raise OdorFormatError(f"{odor_str!r}: concentration must be finite")
    # normalise -0 to 0
    if conc == 0:
        conc = 0.0
    return Odor(name, conc)


def is_odor_level(level_name) -> bool:
    """True for 'odor', 'odor1', 'odor2', ... (what olf.is_odor_var accepts)."""
    return isinstance(level_name, str) and level_name.startswith(COMPONENT_LEVEL_PREFIX)


def odor_level_sort_key(level_name: str) -> int:
    suffix = level_name[len(COMPONENT_LEVEL_PREFIX):]
    return int(suffix) if suffix.isdigit() else 0


def validate_odor_levels(level_names: Sequence[str]) -> list[str]:
    """Return odor component level names in component order, or raise ValueError.

    Rules (from olf.first_odor_level): at least one odor level; either exactly 'odor' or
    'odor<n>' levels with integer suffixes; cannot mix the two styles.
    """
    odor_levels = [n for n in level_names if is_odor_level(n)]
    if not odor_levels:
        raise ValueError(
            "row index has no odor level: expected a level named 'odor1' (or 'odor'); "
            f"found levels {list(level_names)}"
        )
    if len(set(odor_levels)) != len(odor_levels):
        raise ValueError(f"duplicate odor levels in row index: {odor_levels}")
    if COMPONENT_LEVEL_PREFIX in odor_levels:
        if odor_levels != [COMPONENT_LEVEL_PREFIX]:
            raise ValueError(
                "row index mixes an 'odor' level with numbered 'odor<n>' levels: "
                f"{odor_levels}. Use either 'odor' or 'odor1', 'odor2', ..."
            )
        return odor_levels
    bad = [n for n in odor_levels if not n[len(COMPONENT_LEVEL_PREFIX):].isdigit()]
    if bad:
        raise ValueError(
            f"odor levels must be 'odor' or 'odor<n>' with an integer suffix; got {bad}"
        )
    ordered = sorted(odor_levels, key=odor_level_sort_key)
    nums = [odor_level_sort_key(n) for n in ordered]
    if nums[0] != 1:
        raise ValueError(f"odor component numbering should start at 'odor1'; got {ordered}")
    return ordered


def format_mix_from_strs(odor_strs: Iterable[str], *, delim: str = COMPONENT_DELIM) -> str:
    """Join component odor strings, dropping solvent (unless everything is solvent).

    >>> format_mix_from_strs(['A @ -3', 'solvent'])
    'A @ -3'
    >>> format_mix_from_strs(['solvent', 'solvent'])
    'solvent'
    """
    parts = [s for s in odor_strs if s != SOLVENT_STR]
    return delim.join(parts) if parts else SOLVENT_STR


def odor_names(odor_str: str) -> list[str]:
    """The odors' names (abbreviations, as written) in an odor string, of each in a mixture:
    'A @ -3 + b @ -2' -> ['A', 'b'] (solvent left out)."""
    return [p.partition(f" {CONC_DELIM} ")[0] for p in odor_str.split(COMPONENT_DELIM) if p != SOLVENT_STR]


def lowercase_map(names: Iterable[str]) -> dict[str, str]:
    """Odor names (abbreviations) as shown in lowercase: each lowercased, but for those that
    would then be the same as another (differing from it only in case), which are left as
    they are, so that lowercasing never makes two odors one.

    >>> lowercase_map(['IaA', 'va', 'AA', 'aa'])
    {'IaA': 'iaa', 'va': 'va', 'AA': 'AA', 'aa': 'aa'}
    """
    groups: dict[str, list[str]] = {}
    for n in dict.fromkeys(names):
        groups.setdefault(n.lower(), []).append(n)
    return {n: (low if len(same) == 1 else n) for low, same in groups.items() for n in same}


def lowercase_odor(odor_str: str, names: dict[str, str]) -> str:
    """An odor string (one odor, a mixture, or solvent) with each odor's name as `names`
    has it (see `lowercase_map`), else as it is."""
    parts = []
    for p in odor_str.split(COMPONENT_DELIM):
        name, at, conc = p.partition(f" {CONC_DELIM} ")
        parts.append(names.get(name, name) + at + conc)
    return COMPONENT_DELIM.join(parts)


def strip_conc(odor_str: str) -> str:
    """'A @ -3 + B @ -2' -> 'A + B'."""
    comps = [c.strip() for c in odor_str.split(COMPONENT_DELIM)]
    out = []
    for c in comps:
        if c == SOLVENT_STR or CONC_DELIM not in c:
            out.append(c)
        else:
            out.append(c.split(CONC_DELIM)[0].strip())
    return COMPONENT_DELIM.join(out)


def odor_sort_key(odor_str: str):
    """Sort key matching olf's default: solvent first, then name (A-Z), then conc."""
    if odor_str == SOLVENT_STR:
        return (0, "", float("-inf"))
    comps = [c.strip() for c in odor_str.split(COMPONENT_DELIM)]
    keys = []
    for c in comps:
        try:
            o = parse_odor(c, strict=False)
        except OdorFormatError:
            keys.append((1, c.lower(), 0.0))
            continue
        keys.append((1, o.name.lower(), o.log10_conc if o.log10_conc is not None else float("-inf")))
    return tuple(keys)
