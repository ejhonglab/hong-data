"""Vapor pressures of odors, looked up through chemutils (`chemutils.inchi2vapor_pressure`:
the mean of the experimental values on PubChem's page of the compound, in kPa, at the
temperatures given there).

chemutils runs in a process of its own (`look_up` starts ``python -m app.vapor``), as
importing it changes the logging of the process that imports it (a handler on the root
logger printing stack traces with warnings, warnings sent to logging, and a log file
beside its code). Its cache (`config.CHEMUTILS_CACHE`) is kept in the data folder, the
only one the server may write, and it is given a ChemSpider API key (empty but for the
server's own), else it reads one from the home folder, which the server may not read
(sandbox.py).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from typing import Callable, Iterable, Optional

from . import config

TIMEOUT_S = 60  # for each request chemutils makes


def look_up(odors: Iterable[tuple[int, str]], save: Callable[[int, Optional[dict]], None]) -> dict[int, str]:
    """Look up the vapor pressure of each odor, saving each as it comes.

    Args:
        odors: (CID, InChI) of each.
        save: called with the CID and what was found, as it is: ``{"kpa", "source"}`` (the
            URL of the page it is from), or None if PubChem has no vapor pressure for it.

    Returns:
        CID -> why it could not be looked up, for those that could not (each one, if chemutils
        could not be run).
    """
    odors = [[int(cid), inchi] for cid, inchi in odors]
    if not odors:
        return {}
    env = {**os.environ, "CHEMUTILS_CACHE": str(config.CHEMUTILS_CACHE),
           "CHEMSPIDER_API_KEY": os.environ.get("CHEMSPIDER_API_KEY", "")}
    failed, done, error = {}, set(), None
    try:
        proc = subprocess.Popen([sys.executable, "-m", "app.vapor"], cwd=config.ROOT, env=env, text=True,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except OSError as e:
        return {cid: f"chemutils could not be run: {e}" for cid, _ in odors}
    with proc:
        proc.stdin.write(json.dumps(odors))  # (all read before anything is written back)
        proc.stdin.close()
        for line in proc.stdout:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "cid" not in r:
                error = r["error"]
            elif "error" in r:
                failed[r["cid"]] = r["error"]
            else:
                save(r["cid"], {"kpa": r["kpa"], "source": r["source"]} if r["kpa"] is not None else None)
            done.add(r.get("cid"))
    error = error or f"chemutils' process ended early (exit status {proc.returncode})"
    return {**failed, **{cid: error for cid, _ in odors if cid not in done}}


def _main() -> None:
    """The process `look_up` starts: (CID, InChI) pairs as JSON on stdin; on stdout, as each
    is looked up, a line of JSON: its ``cid`` with ``kpa`` and ``source`` (both None if
    PubChem has no vapor pressure for it), or with the ``error`` its lookup raised. Only an
    ``error`` if chemutils cannot be imported."""
    out, sys.stdout = sys.stdout, sys.stderr  # (chemutils prints)
    # a handler on the root logger first, so that chemutils' logging.basicConfig writes no log file
    logging.basicConfig(level=logging.ERROR, format="chemutils: %(levelname)s %(message)s")
    socket.setdefaulttimeout(TIMEOUT_S)
    odors = json.load(sys.stdin)
    try:
        import chemutils
    except Exception as e:
        print(json.dumps({"error": f"chemutils could not be imported: {type(e).__name__}: {e}"}), file=out, flush=True)
        return
    root = logging.getLogger()
    if getattr(chemutils, "sh", None) in root.handlers:
        root.removeHandler(chemutils.sh)
    for cid, inchi in odors:
        try:
            r = chemutils.inchi2vapor_pressure(inchi)
            kpa, source, units = r["vapor_pressure"], r["vapor_pressure_sources"], r["vapor_pressure_units"]
            if kpa is not None and units != "kPa":
                raise ValueError(f"in {units}, not kPa")
            line = {"cid": cid, "kpa": None if kpa is None else float(kpa), "source": source if kpa is not None else None}
        except Exception as e:
            line = {"cid": cid, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(line), file=out, flush=True)


if __name__ == "__main__":
    _main()
