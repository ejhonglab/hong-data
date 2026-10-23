"""Properties of odors' chemicals, looked up through chemutils, which keeps what it looked
up in its cache (`config.CHEMUTILS_CACHE`): vapor pressures (shown with each odor), molar
masses (likewise), and densities and Henry's law constants (for the Headspace calculator,
headspace.py). See `PROPERTIES`.

chemutils runs in a process of its own (`look_up` starts ``python -m app.chemprops``), as
importing it changes the logging of the process that imports it (a handler on the root
logger printing stack traces with warnings, warnings sent to logging, and a log file
beside its code). Its cache is kept in the data folder, the only one the server may
write, as is its log of warnings (`config.CHEMUTILS_LOG`, read when it is imported; they
are not printed), and it is given a ChemSpider API key (empty but for the server's own: then
ChemSpider, where it looks for densities PubChem has none of, is not asked), else it
reads one from the home folder, which the server may not read (sandbox.py). Lookups a
server was too busy for (PubChem allows 5 requests a second) are tried again after a
while (`RETRY_WAITS_S`).

What is looked up is kept, with the chemutils that looked it up (`installed`), so a newer
chemutils does not change it: where it changed how it looks something up (`CHANGES`),
the Admin tab counts those looked up with an older one (`outdated`), and looks them up
again (`look_up` with ``refresh``).
"""
from __future__ import annotations

import functools
import importlib.metadata
import json
import logging
import math
import os
import re
import socket
import subprocess
import sys
import time
from typing import Callable, Iterable, Optional, Sequence

from . import config

TIMEOUT_S = 60  # for each request chemutils makes
RETRY_WAITS_S = (2, 5, 15)  # waits before each retry of a lookup a server was too busy for (HTTP 429, 503)

# property -> the units it is kept in, from the chemutils function looking it up:
# - vapor_pressure: `inchi2vapor_pressure`, the mean of the experimental values on PubChem's
#   page of the compound at or near 25 °C (else of those with no temperature given)
# - est_vapor_pressure: `inchi2est_vapor_pressure`, estimated from the structure at 25 °C
#   (SIMPOL.1; computed, not looked up), for those PubChem has none for
# - molar_mass: `cid2molar_mass`, PubChem's molecular weight
# - density: `inchi2density`, of the pure compound, from PubChem (or ChemSpider, with a key)
# - k_henry: `inchi2k_henry`, its Henry's law solubility constant in water at 25 °C
#   (concentration in water over partial pressure above it) from henrys-law.org, else NIST
# - k_henry_pfo: `inchi2k_henry_pfo`, the same in paraffin oil, estimated (modified UNIFAC
#   activity coefficient in n-hexadecane, and a vapor pressure; computed, not looked up)
PROPERTIES = {"vapor_pressure": "kPa", "est_vapor_pressure": "kPa", "molar_mass": "g/mol", "density": "g/mL",
              "k_henry": "mol/m^3/Pa", "k_henry_pfo": "mol/m^3/Pa"}
LABELS = {"vapor_pressure": "Vapor pressure", "est_vapor_pressure": "Vapor pressure estimated from the structure",
          "molar_mass": "Molar mass", "density": "Density", "k_henry": "Henry's law constant",
          "k_henry_pfo": "Henry's law constant in paraffin oil"}
COMPUTED = ("est_vapor_pressure", "k_henry_pfo")  # (not kept in chemutils' cache)
# what of inchi2k_henry_pfo's output is kept with its value (`look_up`'s ``details``)
K_HENRY_PFO_DETAILS = ("k_henry_pfo_method", "log10_k_oil_air", "solvent_surrogate", "gamma_inf", "unifac_groups",
                       "unifac_groups_source", "unifac_parameters", "vapor_pressure_kpa", "vapor_pressure_source",
                       "vapor_pressure_measured", "notes")

REPO = "https://github.com/ejhonglab/chemutils"
# chemutils' commits from the first this app was pinned to, oldest first (abbreviated), the
# last the one pinned in requirements.txt (DEPLOY.md, "Updating chemutils")
COMMITS = ("53e658bd70ad", "209e47f1eee9", "5bfa72506efe", "95a66c6566cd", "a6bcb4a3aa6f", "713f950668e8", "9196a0c9df65",
           "caf3c6bde452", "fda8e3b346b8", "7ec249c64cc9")
# those of them that changed what a lookup gives, oldest first: (commit, the properties, how)
CHANGES = [
    ("95a66c6566cd", ("vapor_pressure",), "PubChem's vapor pressures only from its values at or near 25 °C (they "
                                         "were the mean of its values at any temperature)"),
    ("9196a0c9df65", ("density",), "PubChem's densities relative to water read (most of its values, which were "
                                  "left out), and pages that could not be read parsed"),
    ("fda8e3b346b8", ("vapor_pressure", "k_henry", "k_henry_pfo"),
     "PubChem's vapor pressures that are estimates used only without measured ones, and said to be measured or not "
     "(as are those constants in oil were estimated with); Henry's law constants of rows given as inequalities (or "
     "wrong) left out rather than leaving none"),
]


@functools.lru_cache(maxsize=1)
def installed() -> Optional[str]:
    """The chemutils installed: its commit (as pip records it for an install from git),
    else its version; None if it is not installed. (Read without importing it, see above.)"""
    try:
        dist = importlib.metadata.distribution("chemutils")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        commit = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id")
    except ValueError:
        commit = None
    return commit or dist.version


def outdated(prop: str, chemutils: Optional[str]) -> Optional[tuple]:
    """The latest of `CHANGES` to how a property is looked up that was made after the
    chemutils it was looked up with, if any.

    Args:
        chemutils: as from `installed` when it was looked up; None if looked up before that
            was kept (with a chemutils older than any of `CHANGES`). One not in `COMMITS`
            (newer, or not installed from git) is taken to have them all.
    """
    if chemutils is None:
        index = -1
    else:
        index = next((i for i, c in enumerate(COMMITS) if chemutils.startswith(c)), None)
        if index is None:
            return None
    later = [c for c in CHANGES if prop in c[1] and COMMITS.index(c[0]) > index]
    return later[-1] if later else None


def look_up(odors: Iterable[tuple[int, str, Sequence[str]]], save: Callable[[int, str, Optional[dict]], None],
            refresh: bool = False) -> dict[tuple[int, str], str]:
    """Look up properties of odors, saving each as it comes.

    Args:
        odors: (CID, InChI, the `PROPERTIES` to look up) of each.
        save: called with the CID, the property and what was found, as it is: ``{"value"
            (in the units of `PROPERTIES`; None if there is none, but with details why),
            "source"`` (where it is from: a URL, or how it was estimated), ``"measured"``
            (whether from measurements, rather than an estimate; None if not known),
            ``"details"}`` (of k_henry_pfo, `K_HENRY_PFO_DETAILS`, else None), or None if
            there is none to be found.
        refresh: look them up again rather than take them from chemutils' cache (which
            keeps what this looks up instead).

    Returns:
        (CID, property) -> why it could not be looked up, for those that could not (each
        one, if chemutils could not be run).
    """
    odors = [[int(cid), inchi, list(props)] for cid, inchi, props in odors]
    odors = [o for o in odors if o[2]]
    if not odors:
        return {}
    env = {**os.environ, "CHEMUTILS_CACHE": str(config.CHEMUTILS_CACHE), "CHEMUTILS_LOG": str(config.CHEMUTILS_LOG),
           "CHEMSPIDER_API_KEY": os.environ.get("CHEMSPIDER_API_KEY", "")}
    failed, done, error = {}, set(), None
    every = [(cid, p) for cid, _, props in odors for p in props]
    try:
        proc = subprocess.Popen([sys.executable, "-m", "app.chemprops"], cwd=config.ROOT, env=env, text=True,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except OSError as e:
        return {k: f"chemutils could not be run: {e}" for k in every}
    with proc:
        proc.stdin.write(json.dumps({"odors": odors, "refresh": refresh, "retry_waits_s": RETRY_WAITS_S}))  # (all read first)
        proc.stdin.close()
        for line in proc.stdout:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "cid" not in r:
                error = r["error"]
                continue
            key = (r["cid"], r["prop"])
            if "error" in r:
                failed[key] = r["error"]
            else:
                found = {k: r.get(k) for k in ("value", "source", "measured", "details")}
                save(*key, found if r["value"] is not None or r.get("details") else None)
            done.add(key)
    error = error or f"chemutils' process ended early (exit status {proc.returncode})"
    return {**failed, **{k: error for k in every if k not in done}}


def _plain(v):
    """A value of chemutils' output as JSON takes it (numpy's as Python's, NaN as None)."""
    if v is None or isinstance(v, str):
        return v
    if hasattr(v, "item"):
        v = v.item()
    return None if isinstance(v, float) and math.isnan(v) else v


def _property(chemutils, cid: int, inchi: str, prop: str, refresh: bool = False) -> dict:
    """A property of a compound as chemutils gives it (as `look_up` saves it: ``value``, in
    the units of `PROPERTIES`, ``source``, ``measured`` and ``details``); with `refresh`,
    not from chemutils' cache."""
    kwargs = {"ignore_cache": True} if refresh and prop not in COMPUTED else {}
    if prop == "molar_mass":
        r = chemutils.cid2molar_mass(cid, **kwargs)  # (from the CID itself: `inchi2molar_mass` would look it up again)
    else:
        r = getattr(chemutils, f"inchi2{prop}")(inchi, **kwargs)
    key = prop.removeprefix("est_")  # (estimates have the keys of what they estimate)
    q, units, source = _plain(r[key]), r[f"{key}_units"], r[f"{key}_sources"]
    details = {k: _plain(r.get(k)) for k in K_HENRY_PFO_DETAILS} if prop == "k_henry_pfo" else None
    measured = {"vapor_pressure": lambda: _plain(r.get("vapor_pressure_measured")), "est_vapor_pressure": lambda: False,
                "k_henry_pfo": lambda: bool(_plain(r.get("k_henry_pfo_measured_partitioning")))}.get(prop, lambda: None)()
    if q is None:
        return {"value": None, "source": None, "measured": None, "details": details}
    if not units:
        raise ValueError(f"{q} without units")
    if isinstance(source, (list, tuple)):
        source = "; ".join(map(str, source))
    return {"value": float(chemutils.ureg.Quantity(float(q), units).to(PROPERTIES[prop]).magnitude),
            "source": source and str(source), "measured": measured, "details": details}


def _busy(e: Exception) -> bool:
    """Whether a lookup failed as a server was too busy (PubChem asks for at most 5 requests a second)."""
    return bool(re.search(r"\b(429|503)\b|Too Many Requests|ServerBusy|Service Unavailable", f"{type(e).__name__}: {e}"))


def _main() -> None:
    """The process `look_up` starts: on stdin, JSON with ``odors`` ([CID, InChI, properties]
    lists), ``refresh`` and ``retry_waits_s`` (`RETRY_WAITS_S`); on stdout, as each property is looked
    up (tried again after each wait while a server is too busy for it), a line of JSON: its
    ``cid`` and ``prop`` with `_property`'s (``value`` None if there is none), or with the
    ``error`` its lookup raised. Only an ``error`` if chemutils cannot be imported."""
    out, sys.stdout = sys.stdout, sys.stderr  # (chemutils prints)
    # a handler on the root logger first, so that an older chemutils' logging.basicConfig writes no log file
    logging.basicConfig(level=logging.ERROR, format="chemutils: %(levelname)s %(message)s")
    socket.setdefaulttimeout(TIMEOUT_S)
    given = json.load(sys.stdin)
    try:
        import chemutils
    except Exception as e:
        print(json.dumps({"error": f"chemutils could not be imported: {type(e).__name__}: {e}"}), file=out, flush=True)
        return
    for logger in (logging.getLogger(), getattr(chemutils, "logger", None)):  # (on either, by version)
        if logger is not None and getattr(chemutils, "sh", None) in logger.handlers:
            logger.removeHandler(chemutils.sh)  # (printing stack traces with warnings)
    if getattr(chemutils, "logger", None) is not None:  # (its warnings to its log file alone)
        chemutils.logger.setLevel(logging.WARNING)
        chemutils.logger.propagate = False
    if not getattr(chemutils, "chemspipy_api_key", None):
        chemutils.chemspipy_api_key = None  # (none to give: not asked, rather than refused)
    for cid, inchi, props in given["odors"]:
        for prop in props:
            for wait in (*given["retry_waits_s"], None):
                try:
                    line = {"cid": cid, "prop": prop, **_property(chemutils, cid, inchi, prop, given.get("refresh", False))}
                except Exception as e:
                    line = {"cid": cid, "prop": prop, "error": f"{type(e).__name__}: {e}"}
                    if wait is not None and _busy(e):
                        time.sleep(wait)
                        continue
                break
            print(json.dumps(line), file=out, flush=True)


if __name__ == "__main__":
    _main()
