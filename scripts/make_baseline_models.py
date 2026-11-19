#!/usr/bin/env python3
"""Makes the reference models olfdb shows as baselines (the Models tab): the KC-remapping experiment
queue's bouton base and its controls, each run on the ORN input of the 16 APL16 odors, tuned with APL
on and then run with APL output off, written as one experiment folder laid out as the queue's archive
of model outputs (`model_data.py` reads it as any other), and, with ``--register``, chosen as
baselines on an olfdb site.

The models are pyolfsysm's (`pyolfsysm.baselines`: `BASELINE_VARIANTS`, `run_baseline`), the same
definitions the queue's code uses, so this script only lays them out:

- ``bouton_base``: the queue's bouton base (E13_bouton_coincidence's ``bouton_base``), made again, to
  check against the archived one;
- ``uniform_glomeruli_claws_uniform_apl``: 7 claws per KC, each from a random bouton of a glomerulus
  drawn with equal chance from all, and every KC<>APL and PN<>APL weight equal;
- ``shuffled_connectome_pairs_kept_seed0``, ``shuffled_connectome_pairs_broken_seed0``: the
  connectome's claws, boutons and APL weights shuffled with seed 0 (which bouton each claw has, how
  much each claw and bouton gets from and gives to APL), each claw's and bouton's two APL weights kept
  together or shuffled apart.

The folder (default `DEFAULT_OUT`, on tensor) gets, as the archive has it:

- ``params.csv``: a row per model. ``base_kwargs_json`` has every `fit_mb_model` setting of it but the
  response rate it is tuned to (``target_rate``), and each setting has a column of its own too (so
  olfdb's model worker runs it as made: `model_runs.plan`); its tuned threshold and APL weight scales
  (absolute values), its response rates with APL on and off, its number of KCs and of PN inputs;
- ``<tag>_on_spike_counts.parquet``, ``<tag>_acute_off_spike_counts.parquet`` and the same of
  ``_responses``: KCs x odors;
- ``provenance.json``: the script, when, where, the commits of hong-data, pyolfsysm and hong2p and the
  olfsysm build, the settings of every model, the input's SHA-256; ``model_provenance.json``:
  pyolfsysm's whole record of what ran (`mb_model.model_provenance`); ``orn_deltas.csv``: the input.

Usage (in an environment with olfsysm, e.g. the KC-remapping queue's venv; a model takes a few minutes
on 16 threads):

    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 PYTHONUNBUFFERED=1 nice -n 10 \\
        python scripts/make_baseline_models.py [<folder>] [--models <name> ...] [--force]
    python scripts/make_baseline_models.py [<folder>] --register http://hal:8080 [--no-run]

A model whose files are in the folder already is not run again (unless ``--force``): its row is kept.
``--register`` POSTs /api/admin/models/baselines for each model named by ``--register-models`` (by
default all but the bouton base, which is a baseline already, from the archive), finding the folder
among the folders of model data the site has (/api/models); the admin path's password, if the site's
proxy asks for one, is taken as ``user:password`` from $OLFDB_ADMIN_AUTH.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

TENSOR = Path("/local/tensor")  # where tensor is mounted on the lab's hosts
DEFAULT_OUT = TENSOR / "Tom/claude_kc_remapping/data/model_outputs/olfdb_baselines"
PARAMS, PROVENANCE, MODEL_PROVENANCE, ORN_FILE = "params.csv", "provenance.json", "model_provenance.json", "orn_deltas.csv"
BASE = "bouton"  # every model's ``base`` (as the archive names the bouton base)
REGISTERED_BY_DEFAULT = ("uniform_glomeruli_claws_uniform_apl", "shuffled_connectome_pairs_kept_seed0",
                         "shuffled_connectome_pairs_broken_seed0")
CONDITIONS = {"on": "on", "acute_off": "off"}  # file condition -> run_baseline's key suffix
TARGET = "target_sparsity"  # the fit_mb_model setting of the response rate tuned to (the ``target_rate`` column)
# the threshold and APL weight scales tuned, as fit_mb_model's params name them (absolute values)
TUNED = ("wAPLKC_scale", "wKCAPL_scale", "wAPLPN_scale", "wPNAPL_scale")
# the KC-remapping queue's note on the three odors whose ORN input is not at Yang's concentration
APPROX_CONC_NOTE = ("Model ORN input for paa (-3 in data vs -4 in ORN input), -aPine (-2 in data vs -1.5 in ORN input), "
                    "euc (-2 in data vs -1.75 in ORN input) is within 1 log10 unit of, but not at, the concentration "
                    "Yang used (allowed by Tom's rule); results involving these odors are approximate.")
# the 16 APL16 odors (Yang's KC recordings), in her order: the 12 of megamat, then 4 of validation2
MEGAMAT12 = ("2h", "IaA", "pa", "2-but", "eb", "ep", "6al", "t2h", "1-8ol", "1-5ol", "1-6ol", "ms")
VALIDATION2_4 = ("paa", "-aPine", "euc", "B-myr")


# ------------------------------------------------------------------ the input
def apl16_orn_deltas():
    """The ORN input of the 16 APL16 odors (glomeruli x (panel, odor)), as the KC-remapping queue's
    ``kcq.harness.apl16_orn_deltas`` makes it: pyolfsysm's megamat ORN responses (the 12 megamat
    odors) and validation2's (paa, -aPine, euc, B-myr), on the 36 glomeruli both have. pyolfsysm's
    own (`pyolfsysm.baselines.apl16_orn_deltas`) if it has one."""
    from pyolfsysm import baselines

    if hasattr(baselines, "apl16_orn_deltas"):
        return baselines.apl16_orn_deltas()
    import pandas as pd
    from pyolfsysm.mb_model import megamat_orn_deltas, validation2_orn_deltas

    m, v = megamat_orn_deltas(), validation2_orn_deltas()
    gl = sorted(set(m.index) & set(v.index))
    abbrev = lambda c: c[1].split(" @ ")[0]  # noqa: E731

    def pick(df, panel, odors):
        cols = [c for c in df.columns if c[0] == panel and abbrev(c) in odors]
        return df.loc[gl, sorted(cols, key=lambda c: odors.index(abbrev(c)))]

    orn = pd.concat([pick(m, "megamat", MEGAMAT12), pick(v, "validation2", VALIDATION2_4)], axis="columns")
    assert [abbrev(c) for c in orn.columns] == [*MEGAMAT12, *VALIDATION2_4] and orn.shape[0] == 36 and not orn.isna().any().any()
    return orn


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ the layout
def file_name(tag: str, condition: str, what: str = "spike_counts") -> str:
    """A model's file, as the archive names them: ``<tag>_<condition>_<what>.parquet``."""
    return f"{tag}_{condition}_{what}.parquet"


def model_files(tag: str) -> list[str]:
    return [file_name(tag, c, w) for c in CONDITIONS for w in ("spike_counts", "responses")]


def settings(kws: dict) -> dict:
    """A model's fit_mb_model settings as its ``base_kwargs_json`` has them: all but the response rate
    it is tuned to (a column of its own, ``target_rate``), and but those that change no output."""
    return {k: v for k, v in sorted(kws.items()) if k not in (TARGET, "make_plots", "silent")}


def _scalar(v: Any) -> Any:
    """A tuned parameter as one number (its mean, if it is one per KC), else None."""
    if v is None:
        return None
    try:
        import numpy as np

        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    return float(a.mean()) if a.size else None


def params_row(tag: str, kws: dict, params: dict, *, n_kcs: int, n_inputs: int, rate_on: float, rate_off: float) -> dict:
    """A model's params.csv row: what it is (``tag``, ``variant``, ``base``), its seed (of its random
    wiring, or of its shuffle), the response rate it is tuned to, its tuned threshold and APL weight
    scales (absolute), how it came out, ``base_kwargs_json`` and a column per setting of it.

    ``seed`` is given only where it is a setting of the model (``seed`` of a random wiring; a shuffle's
    is ``shuffle_connectome_seed``): olfdb's model worker passes a column that is a fit_mb_model
    argument, and not off, on to the model (`model_runs.plan`)."""
    import numpy as np

    s = settings(kws)
    thr = params.get("fixed_thr")
    thr_mean = _scalar(thr)
    row = {"source_table": "make_baseline_models.py", "tag": tag, "variant": tag, "base": BASE,
           "seed": s.get("seed"),
           "target_rate": kws.get(TARGET),
           # (one threshold for every KC: fixed_thr; else their mean alone, which is no setting to run it with)
           "fixed_thr": thr_mean if thr_mean is not None and np.ndim(thr) == 0 else None,
           "fixed_thr_mean": thr_mean,
           "wAPLKC": _scalar(params.get("wAPLKC_scale", params.get("wAPLKC"))),
           **{k: _scalar(params.get(k)) for k in TUNED},
           "n_kcs": n_kcs, "n_glomeruli": n_inputs, "rate_on": rate_on, "rate_off": rate_off,
           "base_kwargs_json": json.dumps(s, sort_keys=True)}
    for k, v in s.items():
        row.setdefault(k, v)
    return row


def write_params(out: Path, rows: list[dict]) -> None:
    """params.csv: the rows in order, their columns in the order they first come."""
    import pandas as pd

    cols = list(dict.fromkeys(c for r in rows for c in r))
    pd.DataFrame(rows, columns=cols).to_csv(out / PARAMS, index=False)


def read_params(out: Path) -> dict[str, dict]:
    """The rows of a params.csv written before, by tag (none if there is none)."""
    import pandas as pd

    p = out / PARAMS
    if not p.is_file():
        return {}
    df = pd.read_csv(p, keep_default_na=False, na_values=[""])
    return {str(r["tag"]): {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in df.to_dict("records")}


def _git(path: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(variants: dict, orn_file: Path, ran: dict) -> dict:
    """provenance.json: what made the folder (the script, when, where, the commits of hong-data,
    pyolfsysm and hong2p, the olfsysm build), the bouton base's settings (``base``), each model's
    (``variants``, by tag: shown with that model alone), the input and how long each model took (``runs``)."""
    import hong2p
    import pyolfsysm
    from hong2p.util import git_commits
    from pyolfsysm import baselines
    from pyolfsysm.mb_model import olfsysm_build_info

    here = Path(__file__).resolve().parent
    build = olfsysm_build_info()
    return {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "hong-data/scripts/make_baseline_models.py", "host": platform.node(),
        **git_commits({"hong_data": here, "pyolfsysm": Path(pyolfsysm.__file__).parent, "hong2p": Path(hong2p.__file__).parent}),
        "olfsysm_build": {k: build.get(k) for k in ("git_hash", "git_dirty", "cpp_dirty", "source_sha256", "build_time")},
        "base": settings(baselines.BOUTON_BASE_KWS), "target_rate": baselines.BOUTON_BASE_KWS.get(TARGET),
        "variants": {tag: settings(kws) for tag, kws in variants.items()},
        "conditions": {"on": "tuned with APL on to target_rate", "acute_off": "the tuned threshold kept, APL->KC and "
                       "APL->bouton weights 0 (KC->APL with them), bouton->APL at its tuned scale"},
        "inputs": {ORN_FILE: sha256(orn_file)},
        "runs": ran,
        "approx_conc_note": APPROX_CONC_NOTE,
    }


# ------------------------------------------------------------------ running
def run(out: Path, names: list[str], force: bool = False) -> list[dict]:
    """Runs the models named (pyolfsysm.baselines' names) into the folder `out`, those whose files are
    there already only if `force`; writes params.csv (a row per model, those run before kept), the
    provenance and the input. Returns the rows."""
    from pyolfsysm import baselines
    from pyolfsysm.mb_model import model_provenance

    out.mkdir(parents=True, exist_ok=True)
    orn = apl16_orn_deltas()
    orn_file = out / ORN_FILE
    orn.to_csv(orn_file)
    rows = read_params(out)
    old_prov = json.loads((out / PROVENANCE).read_text()) if (out / PROVENANCE).is_file() else {}
    ran = dict(old_prov.get("runs") or {})
    for name in names:
        if not force and name in rows and all((out / f).is_file() for f in model_files(name)):
            print(f"{name}: done before, kept (--force to run it again)", flush=True)
            continue
        t0 = dt.datetime.now(dt.timezone.utc)
        print(f"{name}: running ({json.dumps(settings(baselines.baseline_kws(name)), sort_keys=True)})", flush=True)
        r = baselines.run_baseline(name, orn)
        baselines.save_baseline_outputs(r, out, name)
        rate = lambda df: float(df.astype(float).to_numpy().mean())  # noqa: E731 (of KC-odor pairs responding)
        rows[name] = params_row(name, r["kws"], r["params"], n_kcs=len(r["spike_counts_on"]), n_inputs=int(r["wPNKC"].shape[1]),
                                rate_on=rate(r["responses_on"]), rate_off=rate(r["responses_off"]))
        secs = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
        ran[name] = {"started": t0.isoformat(timespec="seconds"), "seconds": round(secs, 1)}
        row = rows[name]
        print(f"{name}: rates on {row['rate_on']:.4f}, off {row['rate_off']:.4f}; threshold {row['fixed_thr_mean']:.4g}, "
              f"wAPLKC_scale {row['wAPLKC_scale']:.4g}, wAPLPN_scale {row['wAPLPN_scale']:.4g}; {secs:.0f} s", flush=True)
        # (written after every model, so one stopped keeps those before)
        write_params(out, list(rows.values()))
    variants = {n: baselines.baseline_kws(n) for n in dict.fromkeys([*names, *rows]) if n in baselines.BASELINE_VARIANTS}
    order = list(dict.fromkeys([n for n in baselines.BASELINE_VARIANTS if n in rows] + list(rows)))
    write_params(out, [rows[n] for n in order])
    (out / PROVENANCE).write_text(json.dumps(provenance(variants, orn_file, ran), indent=1, default=str) + "\n")
    (out / MODEL_PROVENANCE).write_text(json.dumps(model_provenance(orn_deltas=orn), indent=1, default=str) + "\n")
    return [rows[n] for n in order]


# ------------------------------------------------------------------ registering
def _request(url: str, body: Optional[dict] = None) -> Any:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Accept-Encoding": "gzip"},
                                 method="GET" if body is None else "POST")
    auth = os.environ.get("OLFDB_ADMIN_AUTH")
    if auth and "/api/admin/" in url:
        req.add_header("Authorization", "Basic " + base64.b64encode(auth.encode()).decode())
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    return json.loads(data)


def experiment_key(listing: dict, out: Path, tensor: Path = TENSOR) -> str:
    """The folder's key in /api/models' listing (``<folder id>/<its path in that folder>``), as
    /api/admin/models/baselines takes it: of the folder of model data added on the site that has it."""
    rel = Path(os.path.relpath(out.resolve(), tensor.resolve())).as_posix()
    for src in sorted(listing.get("sources", []), key=lambda s: -len(s["path"])):
        top = src["path"].strip("/")
        if rel.startswith(top + "/"):
            return f"{src['id']}/{rel[len(top) + 1:]}"
    raise SystemExit(f"{rel} is in no folder of model data the site has (add it, or the folder above it, in its Admin tab)")


def register(url: str, out: Path, names: list[str], tensor: Path = TENSOR) -> None:
    """Chooses the models named, of the folder `out`, as baselines on the olfdb site at `url`
    (POST /api/admin/models/baselines {"experiment", "row"}, each its row in params.csv)."""
    url = url.rstrip("/")
    listing = _request(f"{url}/api/models")
    key = experiment_key(listing, out, tensor)
    models = [m for m in listing.get("models", []) if m.get("experiment") == key]
    if not models:
        raise SystemExit(f"the site lists no models of {key} (is the folder there, with its {PARAMS}?)")
    for name in names:
        m = next((m for m in models if str(m["params"].get("tag")) == name), None)
        if m is None:
            print(f"{name}: not in the site's listing of {key}; not registered", flush=True)
            continue
        try:
            b = _request(f"{url}/api/admin/models/baselines", {"experiment": key, "row": m["row"]})
            print(f"{name}: baseline {b['id']} ({b['experiment']}, row {b['row']})", flush=True)
        except urllib.error.HTTPError as e:
            print(f"{name}: {e.code} {e.read().decode(errors='replace')[:300]}", flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUT, help=f"the folder written (default {DEFAULT_OUT})")
    ap.add_argument("--models", nargs="+", help="names of pyolfsysm.baselines.BASELINE_VARIANTS run (default: all)")
    ap.add_argument("--force", action="store_true", help="run a model again even if its files are there")
    ap.add_argument("--no-run", action="store_true", help="run nothing (only --register)")
    ap.add_argument("--register", metavar="URL", help="choose the models as baselines on the olfdb site at URL (e.g. http://hal:8080)")
    ap.add_argument("--register-models", nargs="+", default=list(REGISTERED_BY_DEFAULT), help="those registered (default: all but bouton_base)")
    ap.add_argument("--tensor", type=Path, default=TENSOR, help=f"where tensor is mounted, for --register (default {TENSOR})")
    args = ap.parse_args(argv)
    if not args.no_run:
        from pyolfsysm import baselines

        names = args.models or list(baselines.BASELINE_VARIANTS)
        unknown = [n for n in names if n not in baselines.BASELINE_VARIANTS]
        if unknown:
            ap.error(f"not pyolfsysm baselines: {', '.join(unknown)} (they are {', '.join(baselines.BASELINE_VARIANTS)})")
        run(args.out, names, args.force)
        print("written:", args.out, flush=True)
    if args.register:
        register(args.register, args.out, args.register_models, args.tensor)


if __name__ == "__main__":
    sys.exit(main())
