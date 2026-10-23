"""Properties of odors' chemicals, looked up through chemutils, which keeps what it looked
up in its cache (`config.CHEMUTILS_CACHE`): vapor pressures (shown with each odor), molar
masses (likewise), and densities and Henry's law constants (for the Headspace calculator,
headspace.py). See `PROPERTIES`.

chemutils runs in a process of its own (`look_up` starts ``python -m app.chemprops``), as
importing it changes the logging of the process that imports it (a handler on the root
logger printing stack traces with warnings, warnings sent to logging, and a log file
beside its code). Its cache is kept in the data folder, the only one the server may
write, and it is given a ChemSpider API key (empty but for the server's own: then
ChemSpider, where it looks for densities PubChem has none of, is not asked), else it
reads one from the home folder, which the server may not read (sandbox.py). Lookups a
server was too busy for (PubChem allows 5 requests a second) are tried again after a
while (`RETRY_WAITS_S`).
"""
from __future__ import annotations

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
#   page of the compound (at the temperatures given there)
# - est_vapor_pressure: `inchi2est_vapor_pressure`, estimated from the structure at 25 °C
#   (SIMPOL.1; computed, not looked up), for those PubChem has none for
# - molar_mass: `cid2molar_mass`, PubChem's molecular weight
# - density: `inchi2density`, of the pure compound, from PubChem (or ChemSpider, with a key)
# - k_henry: `inchi2k_henry`, its Henry's law solubility constant in water at 25 °C
#   (concentration in water over partial pressure above it) from henrys-law.org, else NIST
PROPERTIES = {"vapor_pressure": "kPa", "est_vapor_pressure": "kPa", "molar_mass": "g/mol", "density": "g/mL",
              "k_henry": "mol/m^3/Pa"}


def look_up(odors: Iterable[tuple[int, str, Sequence[str]]],
            save: Callable[[int, str, Optional[dict]], None]) -> dict[tuple[int, str], str]:
    """Look up properties of odors, saving each as it comes.

    Args:
        odors: (CID, InChI, the `PROPERTIES` to look up) of each.
        save: called with the CID, the property and what was found, as it is: ``{"value"
            (in the units of `PROPERTIES`), "source"}`` (where it is from: a URL, or how it
            was estimated), or None if there is none to be found.

    Returns:
        (CID, property) -> why it could not be looked up, for those that could not (each
        one, if chemutils could not be run).
    """
    odors = [[int(cid), inchi, list(props)] for cid, inchi, props in odors]
    odors = [o for o in odors if o[2]]
    if not odors:
        return {}
    env = {**os.environ, "CHEMUTILS_CACHE": str(config.CHEMUTILS_CACHE),
           "CHEMSPIDER_API_KEY": os.environ.get("CHEMSPIDER_API_KEY", "")}
    failed, done, error = {}, set(), None
    every = [(cid, p) for cid, _, props in odors for p in props]
    try:
        proc = subprocess.Popen([sys.executable, "-m", "app.chemprops"], cwd=config.ROOT, env=env, text=True,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except OSError as e:
        return {k: f"chemutils could not be run: {e}" for k in every}
    with proc:
        proc.stdin.write(json.dumps({"odors": odors, "retry_waits_s": RETRY_WAITS_S}))  # (all read before anything is written back)
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
                save(*key, {"value": r["value"], "source": r["source"]} if r["value"] is not None else None)
            done.add(key)
    error = error or f"chemutils' process ended early (exit status {proc.returncode})"
    return {**failed, **{k: error for k in every if k not in done}}


def _property(chemutils, cid: int, inchi: str, prop: str) -> tuple[Optional[float], Optional[str]]:
    """A property of a compound as chemutils gives it, in the units of `PROPERTIES`, and where it is from."""
    if prop == "molar_mass":
        r = chemutils.cid2molar_mass(cid)  # (from the CID itself: `inchi2molar_mass` would look it up again)
    else:
        r = getattr(chemutils, f"inchi2{prop}")(inchi)
    key = prop.removeprefix("est_")  # (estimates have the keys of what they estimate)
    q, units, source = r[key], r[f"{key}_units"], r[f"{key}_sources"]
    if q is None or math.isnan(q := float(q)):
        return None, None
    if not units:
        raise ValueError(f"{q} without units")
    if isinstance(source, (list, tuple)):
        source = "; ".join(map(str, source))
    return float(chemutils.ureg.Quantity(q, units).to(PROPERTIES[prop]).magnitude), source and str(source)


def _busy(e: Exception) -> bool:
    """Whether a lookup failed as a server was too busy (PubChem asks for at most 5 requests a second)."""
    return bool(re.search(r"\b(429|503)\b|Too Many Requests|ServerBusy|Service Unavailable", f"{type(e).__name__}: {e}"))


def _main() -> None:
    """The process `look_up` starts: on stdin, JSON with ``odors`` ([CID, InChI, properties]
    lists) and ``retry_waits_s`` (`RETRY_WAITS_S`); on stdout, as each property is looked
    up (tried again after each wait while a server is too busy for it), a line of JSON: its
    ``cid`` and ``prop`` with ``value`` and ``source`` (both None if there is none), or with
    the ``error`` its lookup raised. Only an ``error`` if chemutils cannot be imported."""
    out, sys.stdout = sys.stdout, sys.stderr  # (chemutils prints)
    # a handler on the root logger first, so that chemutils' logging.basicConfig writes no log file
    logging.basicConfig(level=logging.ERROR, format="chemutils: %(levelname)s %(message)s")
    socket.setdefaulttimeout(TIMEOUT_S)
    given = json.load(sys.stdin)
    try:
        import chemutils
    except Exception as e:
        print(json.dumps({"error": f"chemutils could not be imported: {type(e).__name__}: {e}"}), file=out, flush=True)
        return
    root = logging.getLogger()
    if getattr(chemutils, "sh", None) in root.handlers:
        root.removeHandler(chemutils.sh)
    if not getattr(chemutils, "chemspipy_api_key", None):
        chemutils.chemspipy_api_key = None  # (none to give: not asked, rather than refused)
    for cid, inchi, props in given["odors"]:
        for prop in props:
            for wait in (*given["retry_waits_s"], None):
                try:
                    value, source = _property(chemutils, cid, inchi, prop)
                    line = {"cid": cid, "prop": prop, "value": value, "source": source}
                except Exception as e:
                    line = {"cid": cid, "prop": prop, "error": f"{type(e).__name__}: {e}"}
                    if wait is not None and _busy(e):
                        time.sleep(wait)
                        continue
                break
            print(json.dumps(line), file=out, flush=True)


if __name__ == "__main__":
    _main()
