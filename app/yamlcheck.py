"""Odor-metadata YAML handling.

Accepts the olfactometer *input* YAML (top-level ``odors:`` list with ``name``,
``log10_conc`` and usually ``abbrev``) as well as the generated *stimulus* YAML
(``pins2odors:`` mapping pin -> odor dict). Checks that every odor in an uploaded
matrix (``'<abbrev> @ <conc>'``) is described in the YAML with a matching
abbreviation/name *and* concentration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from .odors import HARDCODED_ODOR2ABBREV, format_conc


@dataclass
class YamlOdor:
    name: str
    abbrev: Optional[str]
    log10_conc: Optional[float]
    extra: dict = field(default_factory=dict)

    def labels(self) -> set[str]:
        out = {self.name}
        if self.abbrev:
            out.add(self.abbrev)
        hard = HARDCODED_ODOR2ABBREV.get(self.name)
        if hard:
            out.add(hard)
        return out

    def display(self) -> str:
        lab = self.abbrev or HARDCODED_ODOR2ABBREV.get(self.name) or self.name
        if self.log10_conc is None:
            return lab
        return f"{lab} @ {format_conc(self.log10_conc)}"


class YamlError(ValueError):
    pass


def parse_odor_yaml(text: str) -> tuple[list[YamlOdor], dict]:
    """Return (odors, top-level metadata without the odor lists)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise YamlError(f"not valid YAML: {e}")
    if not isinstance(data, dict):
        raise YamlError("YAML must be a mapping at the top level")
    raw: list[dict] = []
    if isinstance(data.get("odors"), list):
        raw.extend(x for x in data["odors"] if isinstance(x, dict))
    if isinstance(data.get("pins2odors"), dict):
        raw.extend(x for x in data["pins2odors"].values() if isinstance(x, dict))
    if not raw:
        raise YamlError("YAML has no 'odors:' list (or 'pins2odors:' mapping) with odor entries")
    odors: list[YamlOdor] = []
    seen = set()
    for i, od in enumerate(raw):
        if "air_mix" in od and "name" not in od:  # a mixture of odors listed on their own (olfactometer configs)
            continue
        name = od.get("name")
        if not isinstance(name, str) or not name.strip():
            raise YamlError(f"odor entry #{i + 1} has no 'name'")
        conc = od.get("log10_conc", None)
        if conc is not None:
            try:
                conc = float(conc)
            except (TypeError, ValueError):
                raise YamlError(f"odor '{name}': log10_conc {conc!r} is not a number")
        abbrev = od.get("abbrev")
        abbrev = str(abbrev).strip() if abbrev is not None else None
        key = (name, abbrev, conc)
        if key in seen:
            continue
        seen.add(key)
        extra = {k: v for k, v in od.items() if k not in ("name", "abbrev", "log10_conc")}
        odors.append(YamlOdor(name=name.strip(), abbrev=abbrev or None, log10_conc=conc, extra=extra))
    meta = {k: v for k, v in data.items() if k not in ("odors", "pins2odors", "pin_sequence")}
    return odors, meta


def lint(odors: list[YamlOdor]) -> list[str]:
    """What looks wrong in an odor YAML's odors themselves: odors without a concentration,
    an abbreviation used for different odors, an odor abbreviated two ways."""
    problems = []
    no_conc = sorted({o.name for o in odors if o.log10_conc is None})
    if no_conc:
        problems.append(f"no log10_conc for {', '.join(no_conc)}")
    names_of: dict[str, set[str]] = {}
    abbrevs_of: dict[str, set[str]] = {}
    for o in odors:
        if o.abbrev:
            names_of.setdefault(o.abbrev, set()).add(o.name)
            abbrevs_of.setdefault(o.name, set()).add(o.abbrev)
    problems += [f"abbreviation {a} is used for {' and '.join(sorted(ns))}" for a, ns in names_of.items() if len(ns) > 1]
    problems += [f"{n} is abbreviated both {' and '.join(sorted(abs_))}" for n, abs_ in abbrevs_of.items() if len(abs_) > 1]
    return problems


def conc_equal(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, abs_tol=1e-6)


def check_subset(
    matrix_odors: list[tuple[str, float]], yaml_odors: list[YamlOdor]
) -> dict[str, Any]:
    """Compare the matrix's (label, conc) odors with the YAML.

    Returns a dict with ``ok`` plus ``matched``, ``missing`` (with hints) and
    ``unused_yaml`` lists.
    """
    matched = []
    missing = []
    used = set()
    for label, conc in matrix_odors:
        hit = None
        for j, yo in enumerate(yaml_odors):
            if label in yo.labels() and conc_equal(conc, yo.log10_conc):
                hit = j
                break
        disp = f"{label} @ {format_conc(conc)}"
        if hit is not None:
            used.add(hit)
            matched.append({"odor": disp, "yaml": yaml_odors[hit].display(), "name": yaml_odors[hit].name})
        else:
            same_label = [yo.display() for yo in yaml_odors if label in yo.labels()]
            if same_label:
                hint = f"YAML has {label} only at: {', '.join(same_label)}"
            else:
                hint = "no odor with this abbreviation/name in the YAML"
            missing.append({"odor": disp, "hint": hint})
    unused = [yaml_odors[j].display() for j in range(len(yaml_odors)) if j not in used]
    return {
        "ok": not missing,
        "n_matrix_odors": len(matrix_odors),
        "n_yaml_odors": len(yaml_odors),
        "matched": matched,
        "missing": missing,
        "unused_yaml": unused,
    }
