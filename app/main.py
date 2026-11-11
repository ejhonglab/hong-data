"""HTTP API + static frontend (Starlette)."""
from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import datetime as dt
import gzip
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import socket
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import quote

import markdown
import numpy as np
from pydantic import ValidationError as PydanticValidationError
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import (backup, bundle_import, chemprops, compare, config, dilution, export, filelog, fits, headspace, lotus,
               lotus_import, matrix_download, merge, metacyc, metacyc_import, model_data, movies, odormeta, permalinks,
               predictions, reference, roi_training, sandbox, tensor, thorimage, vcf, version, yamlcheck)
from .db import PANEL_RENAMES, PROPERTY_COLUMNS, Database, rename_panel
from .glomeruli import GLOMERULI, load_glomeruli
from .ingest import (ParsedData, ValidationError, combine, extra_text, files_text, fly_key, kc_id_problems, load_file,
                     load_parsed, save_parsed, unit_key)
from .metadata import (
    DEFAULT_CODING_UNITS,
    is_kc,
    DEFAULT_INDICATORS,
    DEFAULT_RECORDING_SYSTEMS,
    DEFAULT_WAVELENGTH_NM,
    REQUIRED_FIELDS,
    Metadata,
    dataset_title,
    first_name_problem,
    generate_readme,
    other_calculation,
    same_manipulations,
    same_name,
    same_unit,
)
from .odors import odor_names

config.ensure_dirs()
db = Database(config.DB_PATH)
log = logging.getLogger("uvicorn.error")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: Optional[str], fallback: str) -> str:
    name = Path(name or "").name
    name = _SAFE_NAME.sub("_", name).strip("._")
    return name or fallback


def json_error(status: int, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


# --------------------------------------------------------------------- staging


def _stage_dir(stage_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", stage_id or ""):
        raise HTTPException(404, "unknown stage")
    d = config.STAGING_DIR / stage_id
    if not d.is_dir():
        raise HTTPException(404, "stage not found (it may have expired); upload the file again")
    try:
        os.utime(d)  # kept for STAGE_TTL_HOURS after it was last used (see cleanup_stages), not after it was made
    except OSError:
        pass
    return d


def _stage_meta(d: Path) -> dict:
    return json.loads((d / "stage.json").read_text())


def _write_stage_meta(d: Path, meta: dict) -> None:
    (d / "stage.json").write_text(json.dumps(meta, indent=1))


def cleanup_stages() -> None:
    cutoff = time.time() - config.STAGE_TTL_HOURS * 3600
    if not config.STAGING_DIR.exists():
        return
    for d in config.STAGING_DIR.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


async def _save_upload(upload: UploadFile, dest: Path, *, max_bytes: int) -> int:
    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await upload.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than the {config.MAX_UPLOAD_MB:g} MB limit")
            f.write(chunk)
    return size


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------- routes


class RevalidatedStaticFiles(StaticFiles):
    """Static files browsers check for changes on every load (a 304 when unchanged), so updates show at once."""

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-cache"
        return response


async def index(request: Request) -> Response:
    return FileResponse(config.STATIC_DIR / "index.html", headers={"cache-control": "no-cache"})


async def health(request: Request) -> Response:
    return JSONResponse({"ok": True, "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                         "started": _aware(sandbox.STATUS["started_at"]), "version": version.VERSION})


async def options(request: Request) -> Response:
    existing = await run_in_threadpool(db.options)

    def merge(defaults: list[str], found: list[str]) -> list[str]:
        out = list(defaults)
        for x in found:
            if x not in out:
                out.append(x)
        return out

    return JSONResponse(
        {
            "indicators": merge(DEFAULT_INDICATORS, existing["indicators"]),
            "coding_units": merge(DEFAULT_CODING_UNITS, existing["coding_units"]),
            "recording_systems": merge(DEFAULT_RECORDING_SYSTEMS, existing["recording_systems"]),
            "drivers": existing["drivers"],
            "panels": existing["panels"],
            "uploaders": [u for u in existing["uploaders"] if not first_name_problem(u)],
            "genotypes": existing["genotypes"],
            "response_calcs": existing["response_calcs"],
            "manipulation_keys": existing["manipulation_keys"],
            "manipulation_values": existing["manipulation_values"],
            "default_wavelength_nm": DEFAULT_WAVELENGTH_NM,
            "glomeruli": load_glomeruli(),
            "max_upload_mb": config.MAX_UPLOAD_MB,
            "stage_ttl_hours": config.STAGE_TTL_HOURS,
            "tensor_root": str(config.TENSOR_ROOT),
        }
    )


async def stage_create(request: Request) -> Response:
    """Upload one or more CSV/parquet files (fields 'file') to a staging area, validate each
    and return their matrix: of several, their data as one dataset's (`ingest.combine`)."""
    form = await request.form()
    uploads = [up for up in form.getlist("file") if isinstance(up, UploadFile)]
    if not uploads:
        return json_error(400, "no file in the request (field name 'file')")
    names = [safe_filename(up.filename, "data.csv") for up in uploads]
    same = sorted({n for n in names if names.count(n) > 1})
    if same:
        return JSONResponse({"ok": False, "errors": [f"the files must have different names, but more than one is named "
                                                     f"{', '.join(same)}"], "warnings": []}, status_code=422)
    uploads, names = zip(*sorted(zip(uploads, names), key=lambda x: x[1]))  # (their ROIs in that order)
    cleanup_stages()
    stage_id = secrets.token_urlsafe(12)
    d = config.STAGING_DIR / stage_id
    d.mkdir(parents=True)
    files, parts, failed = [], [], []
    for upload, fname in zip(uploads, names):
        dest = d / fname
        try:
            size = await _save_upload(upload, dest, max_bytes=int(config.MAX_UPLOAD_MB * 1024 * 1024))
        except HTTPException as e:
            shutil.rmtree(d, ignore_errors=True)
            return json_error(e.status_code, e.detail if len(uploads) == 1 else f"{fname}: {e.detail}")
        try:
            parts.append((fname, await run_in_threadpool(load_file, dest, original_name=upload.filename)))
        except ValidationError as e:
            failed.append({"filename": fname, "errors": e.errors, "warnings": e.warnings, **e.details})
            continue
        except Exception as e:  # unexpected
            failed.append({"filename": fname, "errors": [f"unexpected error while reading the file: {e}"], "warnings": []})
            continue
        files.append({"name": fname, "original_filename": upload.filename, "size_bytes": size,
                      "sha256": await run_in_threadpool(_sha256, dest)})
    if failed:
        shutil.rmtree(d, ignore_errors=True)
        if len(uploads) == 1:
            return JSONResponse({"ok": False, **failed[0]}, status_code=422)
        # (each file's problems, and all of them as one list)
        return JSONResponse({"ok": False, "errors": [f"{f['filename']}: {e}" for f in failed for e in f["errors"]], "warnings": [],
                             "files": failed, "n_files": len(uploads)}, status_code=422)

    by_hash: dict[str, list[str]] = {}
    for f in files:
        by_hash.setdefault(f["sha256"], []).append(f["name"])
    errors = [f"{', '.join(ns)} are the same file (byte for byte)" for ns in by_hash.values() if len(ns) > 1]
    try:
        parsed: ParsedData = await run_in_threadpool(combine, parts)
    except ValidationError as e:
        errors += e.errors
        parsed = None
    if not errors:
        errors, _ = await run_in_threadpool(_duplicates, parsed, files)  # (its flies, once whose they are is entered)
    if errors:
        shutil.rmtree(d, ignore_errors=True)
        return JSONResponse({"ok": False, "errors": errors, "warnings": parsed.warnings if parsed else [],
                             **({"filename": names[0]} if len(names) == 1 else {"n_files": len(names)})}, status_code=422)
    await run_in_threadpool(save_parsed, parsed, d / "parsed")
    payload = await run_in_threadpool(parsed.to_payload)
    meta = {
        "stage_id": stage_id,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        # (of the first file; `data_files`: of each)
        "data_file": files[0]["name"],
        "original_filename": files[0]["original_filename"],
        "size_bytes": sum(f["size_bytes"] for f in files),
        "sha256": files[0]["sha256"],
        "data_files": files,
        "source_format": parsed.source_format,
        "odor_yamls": [],
        "experiment_xml": None,
        "experiment_xml_info": None,
    }
    _write_stage_meta(d, meta)
    return JSONResponse({"ok": True, "stage": _public_stage(meta), "data": payload})


def _stage_files(meta: dict) -> list[dict]:
    """A staged upload's data files: [{name, original_filename, size_bytes, sha256}] (of
    one staged before several could be)."""
    return meta.get("data_files") or [{"name": meta["data_file"], "original_filename": meta.get("original_filename"),
                                       "size_bytes": meta["size_bytes"], "sha256": meta["sha256"]}]


MAX_FLIES_NAMED = 6  # flies listed in a message of datasets overlapping


def _flies_text(flies: set[tuple]) -> str:
    """Flies (`ingest.fly_key`s) as said: e.g. '2024-07-10 fly 2 (hemibrain L)'."""
    named = [f"{d} fly {f}{f' ({extra_text(dict(e))})' if e else ''}" for d, f, e in sorted(flies)]
    return ", ".join(named[:MAX_FLIES_NAMED]) + (f" and {len(named) - MAX_FLIES_NAMED} more" if len(named) > MAX_FLIES_NAMED else "")


def _same_data(parsed: ParsedData, m: dict, renames: Optional[dict[str, str]] = None) -> bool:
    """Whether an upload's data is a stored dataset's (`Database.get_matrix_arrays`, its panels
    as shown): the same ROIs and trials (the upload's panels as shown too, by `renames`),
    whatever their order, with the same values (as float32 keeps them)."""
    def index(units, trials):
        return ({unit_key(u): i for i, u in enumerate(units)},
                {(rename_panel(t["panel"], renames or {}), t["odor"], t["repeat"]): j for j, t in enumerate(trials)})
    (ua, ta), (ub, tb) = index(parsed.units, parsed.trials), index(m["units"], m["trials"])
    if set(ua) != set(ub) or set(ta) != set(tb):
        return False
    us, ts = sorted(ua, key=str), sorted(ta, key=str)
    a = parsed.values[np.ix_([ua[k] for k in us], [ta[k] for k in ts])]
    b = m["values"][np.ix_([ub[k] for k in us], [tb[k] for k in ts])]
    return bool(np.allclose(a, b, rtol=1e-5, atol=1e-7, equal_nan=True))


_same_name, _same_unit = same_name, same_unit


def _same_flies_error(flies: set[tuple], d: dict, response_calc: Optional[str]) -> str:
    """Why an upload or edit of a dataset with the same flies (`flies`) as `d` (of the same
    uploader, coding unit and manipulations) and the response calculation `response_calc`
    is refused: one person's datasets of a coding unit cannot have the same flies, but for
    other manipulations, or as the same recordings calculated otherwise, each saying how
    (`metadata.other_calculation`). Says which of the two needs a calculation, if one does."""
    which = f"dataset #{d['id']} ({d['title']}), uploaded by {d['uploader_name']} on {d['created_at'][:10]}"
    same = f"its flies ({_flies_text(flies)}) are exactly those of {which}, also of {d['coding_unit']}"
    here, there = (response_calc or "").strip(), (d.get("response_calc") or "").strip()
    edit = f"to give #{d['id']} one, edit its details in Browse"
    if here and not there:
        return (f"{same}: the same recordings calculated otherwise are kept only when each says how, and #{d['id']} does not "
                f"({edit})")
    if there and not here:
        return (f"{same}: the same recordings calculated otherwise are kept only when each says how: enter how these "
                f"responses were calculated (Response calculation), other than #{d['id']}'s '{there}'")
    how = f"this one's response calculation is '{here}', as #{d['id']}'s" if here else f"under Response calculation; {edit}"
    return (f"{same}: one person's datasets of a coding unit cannot have the same flies, but for other manipulations "
            f"(e.g. after a drug: enter them under Manipulations), or as the same recordings calculated otherwise, each "
            f"saying how ({how})")


def _duplicates(parsed: ParsedData, files: list[dict], uploader: Optional[str] = None, coding_unit: Optional[str] = None,
                manipulations: Optional[list[dict]] = None, response_calc: Optional[str] = None) -> tuple[list[str], list[str]]:
    """An upload (its data, and its files: [{name, sha256}]) against the datasets stored:
    errors (refusing it) if any of its files is one of theirs (byte for byte), or its data
    is (the same flies, ROIs, trials and values), whoever uploaded it. With its uploader,
    coding unit and manipulations (once entered), errors too if its flies (date, fly_num,
    and their ROIs' other levels: `ingest.fly_key`) are all and only those of a dataset of
    theirs of that coding unit and those manipulations, unless they are the same recordings
    calculated otherwise, each saying how (`metadata.other_calculation`; else
    `_same_flies_error` says what is missing), and warnings if it shares only some with one
    (once for the same recordings calculated several ways: `Database.calculations`), saying
    which dataset may need to be deleted. Datasets of other uploaders, coding units or
    manipulations may have the same flies: one person's fly 1 of a day is not another's, a
    fly may be recorded in two places (as may each side of its brain, apart), and a dataset
    of the same flies otherwise treated (e.g. after a drug: `metadata.Metadata.manipulations`)
    is another."""
    flies = {fly_key(u["date"], u["fly_num"], u.get("extra")) for u in parsed.units if u["date"] is not None}
    stored = db.dataset_flies()
    datasets = sorted(db.list_datasets(), key=lambda d: d["id"])
    others = {i: s[1:] for s in db.calculations(datasets) for i in s[:1]}  # first of a set -> its other calculations
    later = {i for s in others.values() for i in s}
    errors, warnings = [], []
    for d in datasets:
        which = f"dataset #{d['id']} ({d['title']}), uploaded by {d['uploader_name']} on {d['created_at'][:10]}"
        delete = f"If this replaces it, delete #{d['id']} first (Admin tab)."
        theirs = stored.get(d["id"], set())
        same_flies = bool(flies) and theirs == flies
        mine = (uploader is not None and _same_name(d["uploader_name"], uploader) and _same_unit(d["coding_unit"], coding_unit)
                and same_manipulations(d["manipulations"], manipulations))
        again = [f["name"] for f in files if f["sha256"] in {g["sha256"] for g in d["data_files"]}]
        if again:
            what = "this file" if len(files) == 1 else files_text(again)
            where = f"it is {which}" if len(again) == 1 and len(d["data_files"]) == 1 else f"in {which}"
            errors.append(f"{what} {'was' if len(again) == 1 else 'were'} uploaded already: {where}. {delete}")
        elif same_flies and (m := db.get_matrix_arrays(d["id"])) is not None and _same_data(parsed, m, db.panel_renames()):
            errors.append(f"this is the data of {which}, already (the same ROIs, trials and values, in another file). {delete}")
        elif mine and same_flies:
            if not other_calculation(response_calc, d.get("response_calc")):
                errors.append(f"{_same_flies_error(flies, d, response_calc)}. {delete}")
        elif mine and flies & theirs and d["id"] not in later:
            if others.get(d["id"]):
                which += f" (and its other calculations, {', '.join(f'#{i}' for i in others[d['id']])})"
            shared = flies & theirs
            if theirs < flies:
                tail = (f"all of whose flies are here, with {len(flies - theirs)} more: if this replaces it, #{d['id']} "
                        "may need to be deleted (Admin tab)")
            elif flies < theirs:
                tail = (f"which has {len(theirs - flies)} more: if these are the same recordings, this repeats part of "
                        f"#{d['id']}, or #{d['id']} may need to be deleted (Admin tab) if this replaces it")
            else:
                n = len(theirs - shared)
                tail = (f"which has {n} other{'s' if n != 1 else ''}: if these are the same recordings, #{d['id']} may need "
                        "to be deleted (Admin tab), or they left out of this file")
            warnings.append(f"possible duplicate: {len(shared)} of its {len(flies)} flies ({_flies_text(shared)}) "
                            f"{'is' if len(shared) == 1 else 'are'} also in {which}, {tail}")
    return errors, warnings


def _permalink_check(parsed: ParsedData, m: Metadata) -> tuple[list[str], Optional[dict]]:
    """An upload (its data, and its metadata as entered) against the permalinks (permalinks.py):
    errors (refusing it) for each that points to a dataset of its identity
    (`permalinks.IDENTITY`), whatever their flies and trials, as one dataset is kept of each;
    and of the orphaned ones it is like (`permalinks.compare`: related), the likest (the same
    first, then the one with the most flies and trials in common), which it takes over once
    saved: {slug, exact, was (what it pointed to), orphaned_at, differences}, or None."""
    new = permalinks.upload_keys(parsed.units, parsed.trials, m.model_dump(), db.panel_renames())
    errors, best, rank = [], None, None
    for p in db.list_permalinks():
        k, c = p["keys"], permalinks.compare(p["keys"], new)
        if p["dataset_id"] is not None:
            if c["same"]:
                errors.append(
                    f"the permalink '{p['slug']}' is of dataset #{p['dataset_id']} ({k.get('title')}), of the same uploader, "
                    f"indicator, driver, coding unit, manipulations, response calculation and panels "
                    f"({', '.join(k.get('panels') or []) or 'none'}) as this, "
                    "and one dataset is kept of each, whatever their flies, odors and trials. If this replaces it, delete "
                    f"#{p['dataset_id']} first (Admin tab): its permalink then points to this, once saved")
        elif c["related"] and (rank is None or (c["exact"], c["shared"]) > rank):
            rank = (c["exact"], c["shared"])
            best = {"slug": p["slug"], "exact": c["exact"], "was": permalinks.summary(k), "orphaned_at": _aware(p["orphaned_at"]),
                    "differences": [] if c["exact"] else permalinks.differences(c, k, new)}
    return errors, best


def _public_permalink(p: Optional[dict]) -> Optional[dict]:
    """A permalink as the API gives it: what it points (or pointed) to said, not all it keeps."""
    if p is None:
        return None
    return {"slug": p["slug"], "dataset_id": p["dataset_id"], "custom": p["custom"],
            **{k: _aware(p[k]) for k in ("created_at", "updated_at", "orphaned_at")},
            "summary": permalinks.summary(p["keys"]), "panels": p["keys"].get("panels") or [],
            "uploaded_at": _aware(p["keys"].get("uploaded_at"))}


def _public_stage(meta: dict) -> dict:
    return {
        "stage_id": meta["stage_id"],
        "data_file": meta["data_file"],
        "data_files": [{k: f[k] for k in ("name", "size_bytes", "sha256")} for f in _stage_files(meta)],
        "size_bytes": meta["size_bytes"],
        "sha256": meta["sha256"],
        "source_format": meta["source_format"],
        "odor_yamls": [{"name": y["name"], "n_odors": y["n_odors"], "tensor_path": y.get("tensor_path")}
                       for y in meta["odor_yamls"]],
        "experiment_xml": meta["experiment_xml"],
        "experiment_xml_info": meta["experiment_xml_info"],
        "experiment_xml_tensor_path": meta.get("experiment_xml_tensor_path"),
    }


def _tensor_sources(meta: dict) -> dict[str, str]:
    """Attached files copied from tensor: file name -> path there."""
    out = {y["name"]: y["tensor_path"] for y in meta["odor_yamls"] if y.get("tensor_path")}
    if meta.get("experiment_xml") and meta.get("experiment_xml_tensor_path"):
        out[meta["experiment_xml"]] = meta["experiment_xml_tensor_path"]
    return out


async def _incoming(request: Request, d: Path, *, json_key: str, fallback: str, suffixes: tuple[str, ...],
                    max_bytes: int) -> tuple[list[tuple[str, Optional[str]]], list[str]]:
    """Save the files of a request in `d`: uploaded ones (multipart field 'file'), or ones
    on tensor (JSON: `json_key`, a path relative to it or a list of them).

    Returns:
        ([(saved file name, its path on tensor or None)], errors)

    Raises:
        HTTPException: 400 if there are no files (or paths) in the request at all.
    """
    saved, errors = [], []
    if request.headers.get("content-type", "").startswith("application/json"):
        paths = (await request.json()).get(json_key)
        paths = [paths] if isinstance(paths, str) else paths or []
        if not paths:
            raise HTTPException(400, f"no path on tensor in the request ('{json_key}')")
        for p in paths:
            try:
                rel, src = tensor.file_path(str(p), config.TENSOR_ROOT, suffixes)
            except tensor.PathError as e:
                errors.append(f"{p}: {e}")
                continue
            name = safe_filename(src.name, fallback)
            await run_in_threadpool(shutil.copyfile, src, d / name)
            saved.append((name, rel))
        return saved, errors
    form = await request.form()
    uploads = [up for up in form.getlist("file") if isinstance(up, UploadFile)]
    if not uploads:
        raise HTTPException(400, "no file in the request (field name 'file')")
    for up in uploads:
        name = safe_filename(up.filename, fallback)
        await _save_upload(up, d / name, max_bytes=max_bytes)
        saved.append((name, None))
    return saved, errors


async def stage_get(request: Request) -> Response:
    """A staged upload, to continue it: as when it was made, with its odor YAMLs' checks (as
    `stage_add_yaml` gives them)."""
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    parsed = await run_in_threadpool(load_parsed, d / "parsed")
    payload = await run_in_threadpool(parsed.to_payload)
    return JSONResponse({"ok": True, **await run_in_threadpool(_yaml_response, d, meta), "data": payload})


def _run_yaml_check(d: Path, meta: dict) -> dict:
    parsed = load_parsed(d / "parsed")
    yaml_odors: list[yamlcheck.YamlOdor] = []
    files = []
    for y in meta["odor_yamls"]:
        text = (d / "odor_yaml" / y["name"]).read_text(encoding="utf-8")
        odors, _ = yamlcheck.parse_odor_yaml(text)
        yaml_odors.extend(odors)
        files.append(y["name"])
    result = yamlcheck.check_subset(parsed.component_odors(), yaml_odors)
    result["files"] = files
    return result


def _yaml_files(d: Path, meta: dict) -> list[dict]:
    """Each attached odor YAML on its own: what looks wrong in it (`yamlcheck.lint`), and its
    odors against the odor table (`odormeta.check_yaml_odors`)."""
    index = _odor_index()
    out = []
    for y in meta["odor_yamls"]:
        odors, _ = yamlcheck.parse_odor_yaml((d / "odor_yaml" / y["name"]).read_text(encoding="utf-8"))
        out.append({"name": y["name"], "problems": yamlcheck.lint(odors), **odormeta.check_yaml_odors(odors, index)})
    return out


def _yaml_response(d: Path, meta: dict, **extra) -> dict:
    return {**extra, "stage": _public_stage(meta), "files": _yaml_files(d, meta),
            "check": _run_yaml_check(d, meta) if meta["odor_yamls"] else None}


def _stage_odors(d: Path, meta: dict) -> dict:
    """A staged upload's odors matched to the odor table, as a dataset's are (`_odor_matches`),
    with ``case``: their names' capitals (`_case_check`)."""
    yaml_odors = []
    for y in meta["odor_yamls"]:
        yaml_odors.extend(yamlcheck.parse_odor_yaml((d / "odor_yaml" / y["name"]).read_text(encoding="utf-8"))[0])
    trials = load_parsed(d / "parsed").trials
    return {**_odor_matches(trials, yaml_odors), "case": _case_check(trials)}


def _case_check(trials: list[dict]) -> dict:
    """The capitals in the odor names (abbreviations) of an upload's `trials`: ``lowercase_only``
    (whether new ones may only be lowercase: `_lowercase_only`), ``uppercase`` (then, its
    names with capitals not already in the database, for which it is refused) and
    ``clashes`` (its names differing from another of its only in case, each group: never
    lowercased where odors are shown in lowercase, so as not to look the same)."""
    names = sorted({n for t in trials for c in t["components"] for n in odor_names(c)})
    only = _lowercase_only()
    known = _known_odor_names() if only else set()
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(n.lower(), []).append(n)
    return {"lowercase_only": only, "uppercase": [n for n in names if only and n != n.lower() and n not in known],
            "clashes": [g for g in groups.values() if len(g) > 1]}


async def stage_odors(request: Request) -> Response:
    """A staged upload's odors matched to the odor table, through its odor YAMLs (see
    `_odor_matches`): it can be saved once each is one odor there."""
    d = _stage_dir(request.path_params["stage_id"])
    return JSONResponse(await run_in_threadpool(_stage_odors, d, _stage_meta(d)))


async def stage_add_yaml(request: Request) -> Response:
    """Add odor YAMLs: uploaded (multipart 'file', repeatable) or on tensor (JSON 'tensor_paths').
    One with the name of one attached already replaces it, if it can be read (else that one
    stays). The response says which were added (``added``: ``replaced`` when one was), and
    has each attached file's own check (``files``) and the matrix's against them all
    (``check``)."""
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    ydir = d / "odor_yaml"
    incoming = ydir / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    try:
        saved, errors = await _incoming(request, incoming, json_key="tensor_paths", fallback="odors.yaml",
                                        suffixes=(".yaml", ".yml"), max_bytes=20 * 1024 * 1024)
        added = []
        for name, rel in saved:
            try:
                odors, _ = yamlcheck.parse_odor_yaml((incoming / name).read_text(encoding="utf-8"))
            except (yamlcheck.YamlError, UnicodeDecodeError) as e:
                kept = any(y["name"] == name for y in meta["odor_yamls"])
                errors.append(f"{name}: {e}" + (f" (the {name} attached before is still attached)" if kept else ""))
                continue
            os.replace(incoming / name, ydir / name)
            replaced = any(y["name"] == name for y in meta["odor_yamls"])
            meta["odor_yamls"] = [y for y in meta["odor_yamls"] if y["name"] != name]
            meta["odor_yamls"].append({"name": name, "n_odors": len(odors), "tensor_path": rel})
            added.append({"name": name, "n_odors": len(odors), "tensor_path": rel, "replaced": replaced})
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    _write_stage_meta(d, meta)
    body = await run_in_threadpool(lambda: _yaml_response(d, meta, ok=not errors, errors=errors, added=added))
    return JSONResponse(body)


async def stage_remove_yaml(request: Request) -> Response:
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    name = safe_filename(request.path_params["name"], "")
    (d / "odor_yaml" / name).unlink(missing_ok=True)
    meta["odor_yamls"] = [y for y in meta["odor_yamls"] if y["name"] != name]
    _write_stage_meta(d, meta)
    return JSONResponse(await run_in_threadpool(lambda: _yaml_response(d, meta, ok=True, removed=name)))


async def stage_add_xml(request: Request) -> Response:
    """Add the Experiment.xml: uploaded (multipart 'file') or on tensor (JSON 'tensor_path')."""
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    saved, errors = await _incoming(request, d, json_key="tensor_path", fallback="Experiment.xml",
                                    suffixes=(".xml",), max_bytes=50 * 1024 * 1024)
    if not saved:
        return JSONResponse({"ok": False, "errors": errors}, status_code=422)
    name, rel = saved[0]
    dest = d / name
    try:
        info = thorimage.parse_experiment_xml(dest.read_text(encoding="utf-8", errors="replace"))
    except thorimage.ExperimentXmlError as e:
        dest.unlink(missing_ok=True)
        return JSONResponse({"ok": False, "errors": [str(e)]}, status_code=422)
    if meta.get("experiment_xml") and meta["experiment_xml"] != name:
        (d / meta["experiment_xml"]).unlink(missing_ok=True)
    meta["experiment_xml"] = name
    meta["experiment_xml_info"] = info["info"]
    meta["experiment_xml_tensor_path"] = rel
    _write_stage_meta(d, meta)
    return JSONResponse({"ok": True, "stage": _public_stage(meta), "info": info["info"]})


async def stage_remove_xml(request: Request) -> Response:
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    if meta.get("experiment_xml"):
        (d / meta["experiment_xml"]).unlink(missing_ok=True)
    meta["experiment_xml"] = None
    meta["experiment_xml_info"] = None
    meta["experiment_xml_tensor_path"] = None
    _write_stage_meta(d, meta)
    return JSONResponse({"ok": True, "stage": _public_stage(meta)})


async def stage_lookup(request: Request) -> Response:
    """Look for the staged dataset's flies, recordings and odor YAMLs on tensor
    ({"name", "raw_data_path"}): its flies in the raw data folder given, else in the raw
    data folders of the user named; the rest in that user's folders. ``user`` is null when
    nobody of that name has folders saved, and there is nothing more then unless a raw data
    folder was given (``raw_data_path_given``, normalized; ``raw_error`` instead when it
    cannot be looked in)."""
    d = _stage_dir(request.path_params["stage_id"])
    body = await request.json()
    name = _user_name(body.get("name"))
    user = await run_in_threadpool(db.get_user, name) if name else None
    raw = str(body.get("raw_data_path") or "").strip()
    dirs = dict(user or {})
    if raw:
        c = await run_in_threadpool(tensor.check_data_path, raw, config.TENSOR_ROOT)
        if not c["ok"] or c["warning"]:
            return JSONResponse({"ok": True, "user": user, "raw_error": c["error"] or c["warning"]})
        dirs["raw_dirs"] = [c["path"]]
    elif user is None:
        return JSONResponse({"ok": True, "user": None})

    def run() -> dict:
        parsed = load_parsed(d / "parsed")
        return tensor.lookup(config.TENSOR_ROOT, dirs, parsed.summary()["flies"], parsed.panel_odors())

    result = await run_in_threadpool(run)
    return JSONResponse({"ok": True, "user": user, "raw_data_path_given": c["path"] if raw else None, **result})


_FIELD_LABELS = {
    "uploader_name": "your first name", "indicator": "calcium indicator", "driver": "driver",
    "coding_unit": "coding unit", "age_min_days": "fly age (min)", "age_max_days": "fly age (max)",
    "power_mw": "power (mW)", "power_tolerance_mw": "power tolerance", "wavelength_nm": "wavelength",
    "panel_names": "panel name(s)", "recording_system": "recording system", "source_code_url": "analysis code link",
    "raw_data_path": "raw data path", "mocorr_data_path": "motion-corrected data path", "genotype": "genotype",
    "notes": "notes", "power_measured_same_day": "power measured on the day", "manipulations": "manipulations",
    "response_calc": "response calculation",
}


def _metadata_errors(e: PydanticValidationError) -> list[dict]:
    """``{'field', 'message'}`` for each problem ('' as the field for ones about several fields)."""
    out = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        label = _FIELD_LABELS.get(loc, loc or "metadata")
        if err["type"] in ("string_too_short", "string_type", "missing") and loc in REQUIRED_FIELDS:
            msg = f"{label} is required"
        else:
            msg = f"{label}: {err['msg'].replace('Value error, ', '')}"
        out.append({"field": loc, "message": msg})
    return out


def _parse_metadata(body: Any, kept_name: Optional[str] = None) -> tuple[Optional[Metadata], list[str]]:
    """Metadata as entered (of an upload, or an edit), validated: (it, or None; problems as
    messages). Its uploader's name must be a first name alone (`first_name_problem`),
    unless it is `kept_name` (an edited dataset's, as stored)."""
    try:
        m = Metadata.model_validate(body or {})
    except PydanticValidationError as e:
        return None, [x["message"] for x in _metadata_errors(e)]
    problem = None if _same_name(m.uploader_name, kept_name) else first_name_problem(m.uploader_name)
    return (None, [problem]) if problem else (m, [])


async def metadata_check(request: Request) -> Response:
    """Check defaults for the upload form ({"metadata": {...}}): any of its fields, validated
    as for an upload, except that the required ones may be left out."""
    body = await request.json()
    values = body.get("metadata") if isinstance(body, dict) else None
    if not isinstance(values, dict):
        return json_error(400, "expected {\"metadata\": {...}}")
    values = {k: v for k, v in values.items() if k in Metadata.model_fields}
    for k in REQUIRED_FIELDS:
        if values.get(k) is None or (isinstance(values[k], str) and not values[k].strip()):
            values[k] = "-"
    try:
        Metadata.model_validate(values)
    except PydanticValidationError as e:
        return JSONResponse({"ok": False, "errors": _metadata_errors(e)})
    return JSONResponse({"ok": True, "errors": []})


DATA_PATH_FIELDS = {"raw_data_path": "raw data path", "mocorr_data_path": "motion-corrected data path"}


def _check_data_paths(values: dict) -> tuple[dict[str, Optional[str]], dict[str, dict], list[str]]:
    """Check a dataset's data paths (`DATA_PATH_FIELDS` -> as entered, empty for none) as the
    Settings tab checks folders on tensor (`tensor.check_data_path`).

    Returns:
        (the paths, normalized, None for none; each one's check; problems as messages)
    """
    paths, checks, errors = {}, {}, []
    for key, value in values.items():
        value = (value or "").strip()
        if not value:
            paths[key] = None
            continue
        c = checks[key] = tensor.check_data_path(value, config.TENSOR_ROOT)
        paths[key] = c["path"]
        if not c["ok"]:
            errors.append(f"{DATA_PATH_FIELDS[key]}: {c['error']}")
    return paths, checks, errors


def _set_data_paths(m: Metadata) -> list[str]:
    """Check an upload's data paths, normalizing them in `m`; problems as messages."""
    paths, _, errors = _check_data_paths({k: getattr(m, k) for k in DATA_PATH_FIELDS})
    if not errors:
        for k, v in paths.items():
            setattr(m, k, v)
    return errors


async def tensor_check(request: Request) -> Response:
    """Check a dataset's data folder on tensor as entered ({"path"}): see `tensor.check_data_path`."""
    body = await request.json()
    path = body.get("path") if isinstance(body, dict) else None
    if not isinstance(path, str):
        return json_error(400, 'expected {"path": ...}')
    return JSONResponse(await run_in_threadpool(tensor.check_data_path, path, config.TENSOR_ROOT))


async def tensor_list(request: Request) -> Response:
    """What is in a folder on tensor (?path=, from its top; '' for the top), to complete a path
    being typed: its folders, and with ?archives=1 its archives too, or with ?files=reference
    the files a reference dataset can be read from (see `tensor.list_dir`)."""
    q = request.query_params
    suffixes = bundle_import.ARCHIVES if q.get("archives") in ("1", "true") else ()
    if q.get("files") == "reference":
        suffixes = reference.SUFFIXES
    return JSONResponse(await run_in_threadpool(tensor.list_dir, q.get("path", ""), config.TENSOR_ROOT, suffixes))


async def stage_readme(request: Request) -> Response:
    """Generate a README.md from the metadata (not yet saved)."""
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    body = await request.json()
    m, errors = _parse_metadata(body.get("metadata"))
    if not errors:
        errors = await run_in_threadpool(_set_data_paths, m)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=422)
    parsed = await run_in_threadpool(load_parsed, d / "parsed")
    kc = _kc_problem(parsed.units, m.coding_unit)
    if kc:
        return JSONResponse({"ok": False, "errors": [kc]}, status_code=422)
    errors, warnings = await run_in_threadpool(_duplicates, parsed, _stage_files(meta), m.uploader_name, m.coding_unit,
                                               m.manipulations, m.response_calc)
    linked, orphan = await run_in_threadpool(_permalink_check, parsed, m)
    errors += linked
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=422)
    summary = parsed.summary()
    readme = generate_readme(
        m,
        summary,
        {
            "data_files": _stage_files(meta),
            "odor_yamls": [y["name"] for y in meta["odor_yamls"]],
            "experiment_xml": meta.get("experiment_xml"),
            "sources": _tensor_sources(meta),
        },
    )
    return JSONResponse({"ok": True, "readme": readme, "title": dataset_title(m, summary), "warnings": warnings, "permalink": orphan})


def _kc_problem(units: list[dict], coding_unit: str) -> Optional[str]:
    """Why an upload of KCs (its coding unit) with ROIs numbered is refused, if it is: each
    fly's ids (of its `units`) not every one from 0 or 1 to the highest (`ingest.kc_id_problems`;
    as for a ``cells`` level whatever the coding unit, when the file was read)."""
    if not is_kc(coding_unit) or not units or any(u["roi_kind"] != "int" for u in units):
        return None
    return kc_id_problems(units)


def _commit(d: Path, meta: dict, m: Metadata, readme: str, readme_edited: bool, proceed: Optional[str] = None) -> dict:
    """Save an upload, checked again; an orphaned permalink it is like (`_permalink_check`)
    pointed to it, if the same, or if `proceed` names it (as the upload form's Proceed does)."""
    parsed = load_parsed(d / "parsed")
    summary = parsed.summary()
    # consistency checks that must pass before anything is written
    problems = [p for p in [_kc_problem(parsed.units, m.coding_unit)] if p]
    files = _stage_files(meta)
    problems += _duplicates(parsed, files, m.uploader_name, m.coding_unit, m.manipulations, m.response_calc)[0]
    linked, orphan = _permalink_check(parsed, m)
    problems += linked
    if orphan and not orphan["exact"] and proceed != orphan["slug"]:
        problems.append(f"it is like the dataset the orphaned permalink '{orphan['slug']}' was of, but not the same: Proceed "
                        "(above the Save button) for that permalink to point to this, once saved")
    if meta["odor_yamls"]:
        check = _run_yaml_check(d, meta)
        if not check["ok"]:
            problems.append(
                "odor YAML check failed: " + "; ".join(f"{x['odor']} ({x['hint']})" for x in check["missing"][:10])
            )
    staged = _stage_odors(d, meta)
    upper = staged["case"]["uppercase"]
    if upper:
        problems.append(f"{len(upper)} of its odor names have capital letters ({', '.join(upper[:10])}{', …' if len(upper) > 10 else ''}), "
                        "and new odor names must be lowercase (set in the Admin tab): rename them in the data file and add it again")
    unmatched = _unmatched(staged["rows"])
    if unmatched:
        problems.append(f"{len(unmatched)} of its odors are not one odor in the odor table ({', '.join(unmatched[:10])}"
                        f"{', …' if len(unmatched) > 10 else ''}): add them to the odor sheet and sync it (the upload form "
                        "shows how), or attach an odor YAML naming them")
    if problems:
        return {"ok": False, "errors": problems, "permalink": orphan}

    created_at = dt.datetime.now()
    title = dataset_title(m, summary)
    ds_id = db.insert_dataset(
        created_at=created_at.isoformat(timespec="seconds"),
        title=title,
        metadata=m.model_dump(),
        readme_md=readme,
        readme_edited=readme_edited,
        data_file=meta["data_file"],
        data_format=parsed.source_format,
        data_sha256=meta["sha256"],
        data_files=[{"name": f["name"], "sha256": f["sha256"]} for f in files],
        odor_yaml_files=[y["name"] for y in meta["odor_yamls"]],
        experiment_xml_file=meta.get("experiment_xml"),
        experiment_xml_info=meta.get("experiment_xml_info"),
        summary=summary,
        units=parsed.units,
        trials=parsed.trials,
        values=parsed.values,
    )
    # move files into the dataset directory
    out = config.DATASETS_DIR / str(ds_id)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(d / f["name"]), out / f["name"])
    if meta["odor_yamls"]:
        (out / "odor_yaml").mkdir(exist_ok=True)
        for y in meta["odor_yamls"]:
            shutil.move(str(d / "odor_yaml" / y["name"]), out / "odor_yaml" / y["name"])
    if meta.get("experiment_xml"):
        shutil.move(str(d / meta["experiment_xml"]), out / meta["experiment_xml"])
    (out / "README.md").write_text(readme, encoding="utf-8")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": ds_id,
                "created_at": created_at.isoformat(timespec="seconds"),
                "title": title,
                "metadata": m.model_dump(),
                "readme_edited": readme_edited,
                "data_file": meta["data_file"],
                "original_filename": meta.get("original_filename"),
                "data_sha256": meta["sha256"],
                "data_files": files,
                "odor_yaml_files": [y["name"] for y in meta["odor_yamls"]],
                "experiment_xml_file": meta.get("experiment_xml"),
                "tensor_sources": _tensor_sources(meta),
                "summary": summary,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    taken = None
    if orphan:  # (the dataset is saved whatever happens to it)
        try:
            new, m_new = db.get_dataset(ds_id), db.get_matrix_arrays(ds_id)
            db.set_permalink(orphan["slug"], ds_id, permalinks.dataset_keys(new, m_new),
                             orphan["slug"] != permalinks.default_slug(new, m_new["trials"]),
                             created_at.isoformat(timespec="seconds"), old=orphan["slug"])
            taken = orphan["slug"]
            log.info("dataset %d took over the orphaned permalink %s", ds_id, taken)
        except sqlite3.IntegrityError as e:
            log.warning("dataset %d could not take over the permalink %s: %s", ds_id, orphan["slug"], e)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "dataset_id": ds_id, "title": title, "permalink": taken}


async def stage_commit(request: Request) -> Response:
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    body = await request.json()
    m, errors = _parse_metadata(body.get("metadata"))
    if not errors:
        errors = await run_in_threadpool(_set_data_paths, m)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=422)
    readme = body.get("readme")
    if not isinstance(readme, str) or not readme.strip():
        return JSONResponse({"ok": False, "errors": ["README is empty"]}, status_code=422)
    proceed = body.get("permalink_proceed") if isinstance(body.get("permalink_proceed"), str) else None
    result = await run_in_threadpool(_commit, d, meta, m, readme, bool(body.get("readme_edited")), proceed)
    return JSONResponse(result, status_code=200 if result["ok"] else 422)


_EMPTY_THEAD = re.compile(r"<thead>\s*<tr>(?:\s*<th>\s*</th>)+\s*</tr>\s*</thead>")


def _readme_html(md: str) -> str:
    html = markdown.markdown(md, extensions=["tables"])
    return _EMPTY_THEAD.sub("", html)


async def datasets_list(request: Request) -> Response:
    rows = await run_in_threadpool(db.list_datasets)
    return JSONResponse({"datasets": [{**d, "created_at": _aware(d["created_at"])} for d in rows]})


def _readme_path(ds_id: int) -> Path:
    return config.DATASETS_DIR / str(ds_id) / "README.md"


async def dataset_get(request: Request) -> Response:
    ds_id = int(request.path_params["ds_id"])
    d = await run_in_threadpool(db.get_dataset, ds_id)
    if d is None:
        return json_error(404, "dataset not found")
    d["created_at"] = _aware(d["created_at"])
    d["readme_html"] = _readme_html(d["readme_md"])
    d["readme_path"] = str(_readme_path(ds_id))
    files = [*(f["name"] for f in d["data_files"]), "README.md", "metadata.json"]
    files += [f"odor_yaml/{y}" for y in d["odor_yaml_files"]]
    if d.get("experiment_xml_file"):
        files.append(d["experiment_xml_file"])
    d["files"] = files
    d["permalink"] = _public_permalink(d["permalink"])
    d["default_permalink"] = await run_in_threadpool(_default_permalink, d)
    d["calculations"] = await run_in_threadpool(_calculations_of, ds_id)
    return JSONResponse(d)


def _calculations_of(ds_id: int) -> list[dict]:
    """The other datasets of a dataset's recordings calculated otherwise (`Database.calculations`):
    [{id, title, response_calc}], in order of upload."""
    datasets = db.list_datasets()
    ids = next((s for s in db.calculations(datasets) if ds_id in s), [])
    by_id = {d["id"]: d for d in datasets}
    return [{"id": i, "title": by_id[i]["title"], "response_calc": by_id[i]["response_calc"]} for i in ids if i != ds_id]


def _default_permalink(d: dict) -> str:
    """The name a dataset's permalink would have by default (`permalinks.default_slug`), but
    for those of others (with -2, -3... after it)."""
    taken = {p["slug"] for p in db.list_permalinks() if p["dataset_id"] != d["id"]}
    return permalinks.unique(permalinks.default_slug(d, db.get_trials(d["id"])), taken)


_readme_lock = threading.Lock()


def _save_readme(ds_id: int, readme: str, base: str) -> tuple[int, dict]:
    """Overwrite a dataset's README (file and database) if it is still `base`: (status, response)."""
    with _readme_lock:  # one at a time, so that what is saved is what `base` was checked against
        d = db.get_dataset(ds_id)
        if d is None:
            return 404, {"error": "dataset not found"}
        if d["readme_md"] != base:
            return 409, {"error": "the README was changed on the server since you started editing it; "
                                  "copy your edits somewhere, and reload the page to see the new one"}
        path = _readme_path(ds_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("README.md.tmp")
        tmp.write_text(readme, encoding="utf-8")
        tmp.replace(path)
        db.update_readme(ds_id, readme)
        meta = path.with_name("metadata.json")
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            m["readme_edited"] = True
            meta.write_text(json.dumps(m, indent=1), encoding="utf-8")
        except (OSError, ValueError):
            pass
    log.info("README of dataset %d overwritten: %s", ds_id, path)
    return 200, {"ok": True, "readme_md": readme, "readme_html": _readme_html(readme), "readme_path": str(path)}


async def dataset_readme_save(request: Request) -> Response:
    """Overwrite a dataset's README ({"readme", "base": the README the edit started from});
    409 if it was changed on the server since then."""
    ds_id = int(request.path_params["ds_id"])
    body = await request.json()
    readme, base = (body.get("readme"), body.get("base")) if isinstance(body, dict) else (None, None)
    if not isinstance(readme, str) or not isinstance(base, str):
        return json_error(400, 'expected {"readme": ..., "base": ...}')
    if not readme.strip():
        return json_error(422, "the README is empty")
    status, result = await run_in_threadpool(_save_readme, ds_id, readme, base)
    return JSONResponse(result, status_code=status)


async def dataset_matrix(request: Request) -> Response:
    ds_id = int(request.path_params["ds_id"])
    m = await run_in_threadpool(db.get_matrix, ds_id)
    if m is None:
        return json_error(404, "dataset not found")
    return JSONResponse(m)


def _data_and_movies(d: dict) -> tuple[dict, dict, Optional[list[dict]]]:
    """A dataset's raw and motion-corrected data folders (by kind: see `tensor.find_data`),
    their movies (by kind: ``movies`` as from `movies.find`, but for those of recordings
    redone, `movies.drop_redone`; raw ones given the trials of their recordings in the
    motion-corrected data by `movies.link_trials`; ``error``: why
    its folder was not looked in, ``found``: how many were found in its flies' folders, by
    their paths and of no fly, ``skipped``: the flies, not the dataset's, whose folders in
    it were left out (see `movies.find`), and when both folders were found, ``compared``
    and ``unmatched``: the recordings with movies in that one only, see
    `movies.unmatched`), and then its recordings in both side by side
    (`movies.side_by_side`; else None)."""
    flies = d["summary"].get("flies") or []
    bases = [Path(p) for p in sandbox.folders()]
    found, out, listed, skipped = {}, {}, {}, {}
    for kind in ("raw", "mocorr"):
        f = found[kind] = tensor.find_data(d["metadata"].get(f"{kind}_data_path"), config.TENSOR_ROOT, bases)
        folder = Path(f["path"]) if f["status"] == "ok" and f["is_dir"] else None
        skipped[kind] = set()
        listed[kind] = movies.find(folder, kind, flies, skipped[kind]) if folder else []
    listed["raw"], listed["mocorr"] = movies.drop_redone(listed["raw"], listed["mocorr"])
    for kind, found_movies in listed.items():
        out[kind] = {"movies": found_movies, "error": found[kind]["error"], "compared": False, "unmatched": [],
                     # the flies (not the dataset's) whose folders in it were left out
                     "skipped": [{"date": date, "fly_num": n} for date, n in sorted(skipped[kind])],
                     # how its movies were found: in its flies' folders, by their paths, of no fly
                     "found": {b or "none": sum(1 for m in found_movies if m["by"] == b) for b in ("fly folder", "path", None)}}
    movies.link_trials(out["raw"]["movies"], out["mocorr"]["movies"])
    rows = None
    if all(f["status"] == "ok" and f["is_dir"] for f in found.values()):
        for kind, recordings in movies.unmatched(out["raw"]["movies"], out["mocorr"]["movies"]).items():
            out[kind].update(compared=True, unmatched=recordings)
        rows = movies.side_by_side(out["raw"]["movies"], out["mocorr"]["movies"])
    return found, out, rows


async def dataset_data_paths(request: Request) -> Response:
    """Where a dataset's raw and motion-corrected data are on the server, and what is in
    them (``raw``, ``mocorr``: see `tensor.find_data`), with ``compared`` and ``unmatched``:
    the recordings with movies in that folder and not the other, and ``recordings``: those
    of both side by side, or None (see `_data_and_movies`)."""
    ds_id = int(request.path_params["ds_id"])

    def run() -> Optional[dict]:
        d = db.get_dataset(ds_id)
        if d is None:
            return None
        found, out, rows = _data_and_movies(d)
        return {**{k: {**found[k], **{x: out[k][x] for x in ("compared", "unmatched", "found", "skipped")}} for k in found},
                "recordings": rows}

    result = await run_in_threadpool(run)
    if result is None:
        return json_error(404, "dataset not found")
    return JSONResponse(result)


async def dataset_data_paths_save(request: Request) -> Response:
    """Set a dataset's data paths ({"raw_data_path", "mocorr_data_path"}: either or both,
    empty for none), checked as at upload; nothing is saved if one is wrong."""
    ds_id = int(request.path_params["ds_id"])
    body = await request.json()
    values = {k: body[k] for k in DATA_PATH_FIELDS if k in body} if isinstance(body, dict) else {}
    if not values or any(v is not None and not isinstance(v, str) for v in values.values()):
        return json_error(400, f"expected {' and/or '.join(DATA_PATH_FIELDS)}")

    def run() -> tuple[int, dict]:
        if db.get_dataset(ds_id) is None:
            return 404, {"error": "dataset not found"}
        paths, checks, errors = _check_data_paths(values)
        if errors:
            return 422, {"ok": False, "errors": errors, "checks": checks}
        metadata = db.update_data_paths(ds_id, paths)
        meta = config.DATASETS_DIR / str(ds_id) / "metadata.json"
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            m.setdefault("metadata", {}).update(paths)
            meta.write_text(json.dumps(m, indent=1), encoding="utf-8")
        except (OSError, ValueError):
            pass
        log.info("data paths of dataset %d set: %s", ds_id, paths)
        return 200, {"ok": True, "metadata": metadata, "checks": checks}

    status, result = await run_in_threadpool(run)
    return JSONResponse(result, status_code=status)


_metadata_lock = threading.Lock()


def _same_flies_problems(ds_id: int, m: Metadata) -> list[str]:
    """Why a dataset's metadata as edited is refused, as an upload with it would be
    (`_duplicates`): its flies all and only those of another dataset of its uploader, coding
    unit and manipulations, unless they are the same recordings calculated otherwise, each
    saying how (`metadata.other_calculation`)."""
    stored = db.dataset_flies()
    flies = stored.get(ds_id, set())
    out = []
    for d in sorted(db.list_datasets(), key=lambda d: d["id"]):
        if (d["id"] != ds_id and flies and stored.get(d["id"], set()) == flies and _same_name(d["uploader_name"], m.uploader_name)
                and _same_unit(d["coding_unit"], m.coding_unit) and same_manipulations(d["manipulations"], m.manipulations)
                and not other_calculation(m.response_calc, d.get("response_calc"))):
            out.append(_same_flies_error(flies, d, m.response_calc))
    return out


def _as_entered(metadata: dict) -> dict:
    """Stored metadata as `Metadata` dumps it now (fields added since it was stored, at their
    defaults), to compare with an edit of it; as it is if it is no longer valid."""
    try:
        return Metadata.model_validate(metadata).model_dump()
    except PydanticValidationError:
        return metadata


def _save_metadata(ds_id: int, values: Any, base: dict, proceed: Optional[str] = None) -> tuple[int, dict]:
    """Replace a dataset's metadata as entered at upload with `values` (checked as an upload's
    are) if it is still `base` (as shown): (status, response). If its permalink is named as by
    default and its default changes, it is renamed so, once `proceed` names the new one (409
    with ``permalink_change`` {from, to} until it does)."""
    with _metadata_lock:  # one at a time, so that what is saved is what `base` was checked against
        shown, d = db.get_dataset(ds_id), db.get_dataset(ds_id, as_shown=False)
        if shown is None:
            return 404, {"error": "dataset not found"}
        if base not in (shown["metadata"], d["metadata"]):
            return 409, {"error": "its details were changed on the server since you started editing them; copy your "
                                  "edits somewhere, and reload the page to see the new ones"}
        m, errors = _parse_metadata(values, kept_name=d["metadata"].get("uploader_name"))
        if errors:
            return 422, {"ok": False, "errors": errors}
        old = _as_entered(d["metadata"])
        if m.panel_names == shown["metadata"].get("panel_names"):  # (as shown: those renamed kept as stored)
            m.panel_names = old.get("panel_names") or []
        # data paths checked on tensor as at upload, but only those changed (tensor may be away)
        paths, checks, errors = _check_data_paths({k: getattr(m, k) for k in DATA_PATH_FIELDS if getattr(m, k) != old.get(k)})
        if errors:
            return 422, {"ok": False, "errors": errors, "checks": checks}
        for k, v in paths.items():
            setattr(m, k, v)
        problems = [p for p in [_kc_problem(db.get_matrix_arrays(ds_id)["units"], m.coding_unit)] if p]
        problems += _same_flies_problems(ds_id, m)
        if problems:
            return 422, {"ok": False, "errors": problems}
        new = m.model_dump()
        changed = {k: [old.get(k), v] for k, v in new.items() if old.get(k) != v}
        if not changed:
            return 200, {"ok": True, "changed": [], "readme": "kept"}
        link, rename = shown["permalink"], None
        if link and not link["custom"]:
            trials, renames = db.get_trials(ds_id), db.panel_renames()
            edited = {**shown, "uploader_name": m.uploader_name, "coding_unit": m.coding_unit, "manipulations": m.manipulations or [],
                      "response_calc": m.response_calc, "metadata": {**shown["metadata"], "panel_names": [rename_panel(p, renames) for p in m.panel_names or []]}}
            if permalinks.default_slug(edited, trials) != permalinks.default_slug(shown, trials):
                to = permalinks.unique(permalinks.default_slug(edited, trials),
                                       {p["slug"] for p in db.list_permalinks()} - {link["slug"]})
                if to != link["slug"]:
                    if proceed != to:
                        return 409, {"ok": False, "errors": [], "permalink_change": {"from": link["slug"], "to": to}}
                    rename = to
        title = dataset_title(m, d["summary"])
        folder = config.DATASETS_DIR / str(ds_id)
        try:
            saved = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = None
        readme = None
        if not d["readme_edited"]:  # (one generated at upload follows them; one edited by hand is left alone)
            files = {"data_files": d["data_files"], "odor_yamls": d["odor_yaml_files"], "experiment_xml": d.get("experiment_xml_file"),
                     "sources": (saved or {}).get("tensor_sources") or {}}
            readme = generate_readme(m, d["summary"], files, created_at=dt.datetime.fromisoformat(d["created_at"]))
        db.update_metadata(ds_id, new, title, readme)
        if link:  # (what it keeps of the dataset, as it is now)
            when = dt.datetime.now().isoformat(timespec="seconds")
            keys = permalinks.dataset_keys(db.get_dataset(ds_id), db.get_matrix_arrays(ds_id))
            if rename:
                db.set_permalink(rename, ds_id, keys, False, when, old=link["slug"])
                log.info("permalink %s of dataset %d renamed %s, as its default", link["slug"], ds_id, rename)
            else:
                db.update_permalink_keys(ds_id, keys, when)
        if readme is not None:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "README.md").write_text(readme, encoding="utf-8")
        if saved is not None:
            saved.update(title=title, metadata=new)
            saved.setdefault("edits", []).append({"at": dt.datetime.now().isoformat(timespec="seconds"), "changed": changed})
            (folder / "metadata.json").write_text(json.dumps(saved, indent=1), encoding="utf-8")
    log.info("metadata of dataset %d edited: %s", ds_id, ", ".join(changed))
    return 200, {"ok": True, "changed": list(changed), "readme": "kept" if readme is None else "generated",
                 "permalink": rename or (link and link["slug"])}


async def dataset_metadata_save(request: Request) -> Response:
    """Replace a dataset's metadata as entered at upload ({"metadata": every field, as an
    upload sends them; "base": its metadata as the edit started from}), checked as an
    upload's (its data paths, those changed); its title and panels follow, and its README if
    it was not edited by hand. 409 if it was changed on the server since `base`, or (with
    ``permalink_change``) if its permalink would be renamed and "permalink_proceed" does not
    name the new one (see `_save_metadata`)."""
    ds_id = int(request.path_params["ds_id"])
    body = await request.json()
    values, base = (body.get("metadata"), body.get("base")) if isinstance(body, dict) else (None, None)
    if not isinstance(values, dict) or not isinstance(base, dict):
        return json_error(400, 'expected {"metadata": {...}, "base": {...}}')
    proceed = body.get("permalink_proceed") if isinstance(body.get("permalink_proceed"), str) else None
    status, result = await run_in_threadpool(_save_metadata, ds_id, values, base, proceed)
    return JSONResponse(result, status_code=status)


def _save_permalink(ds_id: int, slug: Any) -> tuple[int, dict]:
    """Name a dataset's permalink `slug` (a new one, or its own renamed): (status, response)."""
    if not isinstance(slug, str):
        return 400, {"error": 'expected {"slug": ...}'}
    problem = permalinks.slug_problem(slug)
    if problem:
        return 422, {"ok": False, "errors": [f"not a permalink's name: {problem}"]}
    with _metadata_lock:
        d = db.get_dataset(ds_id)
        if d is None:
            return 404, {"error": "dataset not found"}
        other, own = db.get_permalink(slug), d["permalink"]
        if other and other["dataset_id"] != ds_id:
            whose = (f"dataset #{other['dataset_id']}'s" if other["dataset_id"] is not None
                     else "an orphaned permalink's (its dataset was deleted; it can be deleted in the Admin tab)")
            return 409, {"ok": False, "errors": [f"'{slug}' is {whose} already"]}
        m = db.get_matrix_arrays(ds_id)
        db.set_permalink(slug, ds_id, permalinks.dataset_keys(d, m), slug != permalinks.default_slug(d, m["trials"]),
                         dt.datetime.now().isoformat(timespec="seconds"), old=own and own["slug"])
    log.info("permalink of dataset %d: %s%s", ds_id, slug, f" (was {own['slug']})" if own and own["slug"] != slug else "")
    return 200, {"ok": True, "permalink": _public_permalink(db.permalink_of(ds_id))}


async def dataset_permalink_save(request: Request) -> Response:
    """Name a dataset's permalink ({"slug"}): lowercase letters, digits and '-' (see permalinks.py)."""
    body = await request.json()
    status, result = await run_in_threadpool(_save_permalink, int(request.path_params["ds_id"]),
                                             body.get("slug") if isinstance(body, dict) else None)
    return JSONResponse(result, status_code=status)


async def permalink_get(request: Request) -> Response:
    """A permalink: the dataset it points to (``dataset_id``, null if orphaned) and what it is (or was)."""
    p = await run_in_threadpool(db.get_permalink, request.path_params["slug"])
    if p is None:
        return json_error(404, "no permalink of that name")
    return JSONResponse(_public_permalink(p))


async def permalink_follow(request: Request) -> Response:
    """A permalink followed (/d/<name>): its dataset in Browse (which says if it points to none)."""
    return RedirectResponse(f"/#browse/{quote(request.path_params['slug'], safe='')}", status_code=302)


# ---------------------------------------------------------------------- movies


def _server_file(path: Any, suffixes: tuple[str, ...]) -> Path:
    """A file on the server named by the browser (a movie, its ROIs or trials): it must be on tensor,
    in a folder set in the Admin tab, or of a prediction of glomerulus ROIs (its fly's average image
    and ROIs), and one the server may read (raises HTTPException)."""
    p = str(path or "")
    if not os.path.isabs(p) or not p.lower().endswith(suffixes):
        raise HTTPException(400, f"expected the path on the server of a {' / '.join(suffixes)} file")
    real = os.path.realpath(p)
    tops = [os.path.realpath(t).rstrip("/") for t in [config.TENSOR_ROOT, *sandbox.folders(), predictions.root()]]
    if not any(real == t or real.startswith(t + "/") for t in tops) or not sandbox.can_read(real):
        raise HTTPException(403, tensor.NOT_ALLOWED)
    if not os.path.isfile(real):
        raise HTTPException(404, "no such file on the server")
    return Path(real)


async def dataset_movies(request: Request) -> Response:
    """A dataset's movies in its raw and motion-corrected data folders (``raw``, ``mocorr``:
    see `_data_and_movies`; each movie ``unmatched`` if its recording has none in the other)."""
    ds_id = int(request.path_params["ds_id"])

    def run() -> Optional[dict]:
        d = db.get_dataset(ds_id)
        return None if d is None else _data_and_movies(d)[1]

    result = await run_in_threadpool(run)
    if result is None:
        return json_error(404, "dataset not found")
    return JSONResponse(result)


async def movie_info(request: Request) -> Response:
    """A movie's shape (t, z, c, y, x), pixel type and frame rate (``?path=``: on the server)."""
    path = _server_file(request.query_params.get("path"), (*movies.TIFF_SUFFIXES, ".raw"))
    try:
        movie = await run_in_threadpool(movies.open_movie, path)
    except movies.MovieError as e:
        return json_error(422, str(e))
    return JSONResponse(movie.info())


async def movie_planes(request: Request) -> Response:
    """Planes of a movie (``?path=&t=&n=&z=&c=``: `n` planes from time `t` on, of plane `z`, or
    its max projection with ``z=max``, of channel `c`), as little-endian numbers: their type
    in the X-Dtype header, their shape (n, y, x) in X-Shape (n may be fewer than asked)."""
    q = request.query_params
    path = _server_file(q.get("path"), (*movies.TIFF_SUFFIXES, ".raw"))
    try:
        t, n, c = int(q.get("t", 0)), int(q.get("n", 1)), int(q.get("c", 0))
        z = None if q.get("z") == "max" else int(q.get("z", 0))
    except ValueError:
        return json_error(400, "t, n, c and z must be whole numbers (z may also be 'max')")

    def run():
        movie = movies.open_movie(path)
        return movie.planes(t, movies.chunk(movie, n, z), z, c)

    try:
        planes = await run_in_threadpool(run)
    except movies.MovieError as e:
        return json_error(422, str(e))
    return Response(planes.tobytes(), media_type="application/octet-stream",
                    headers={"x-dtype": planes.dtype.name, "x-shape": ",".join(map(str, planes.shape))})


async def movie_rois(request: Request) -> Response:
    """The ImageJ ROIs of a RoiSet.zip (or .roi) on the server (``?path=``): see
    `movies.read_rois`, each with its ``kind`` as a glomerulus (`movies.roi_kind`: of the
    glomeruli an upload accepts, and those of dataset ``?dataset=``'s data, if given)."""
    path = _server_file(request.query_params.get("path"), (".zip", ".roi"))
    ds = request.query_params.get("dataset")
    if ds is not None and not ds.isdigit():
        return json_error(400, "dataset must be a dataset's id")

    def run() -> list[dict]:
        rois = movies.read_rois(path)
        names = GLOMERULI | (db.roi_names(int(ds)) if ds else set())
        return [{**r, "kind": movies.roi_kind(r["name"], names)} for r in rois]

    try:
        rois = await run_in_threadpool(run)
    except movies.MovieError as e:
        return json_error(422, str(e))
    return JSONResponse({"rois": rois})


async def movie_trials(request: Request) -> Response:
    """The trials of a movie, from an al_analysis trial_frames_and_odors.json on the server
    (``?path=``): see `movies.read_trials`."""
    path = _server_file(request.query_params.get("path"), (".json",))
    try:
        trials = await run_in_threadpool(movies.read_trials, path)
    except movies.MovieError as e:
        return json_error(422, str(e))
    return JSONResponse({"trials": trials})


async def dataset_file(request: Request) -> Response:
    ds_id = int(request.path_params["ds_id"])
    rel = request.path_params["path"]
    base = (config.DATASETS_DIR / str(ds_id)).resolve()
    target = (base / rel).resolve()
    if base not in target.parents or not target.is_file():
        return json_error(404, "file not found")
    return FileResponse(target, filename=target.name)


def cleanup_exports(max_age_s: float = 3600) -> None:
    """Remove downloads left behind by interrupted transfers."""
    cutoff = time.time() - max_age_s
    for p in config.EXPORTS_DIR.iterdir():
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


async def download(request: Request) -> Response:
    """Download of one or more datasets: ``?ids=1,2,3&format=files|wide|long`` (see export.py);
    a zip, or a CSV for the matrix of a single dataset. With ``lower=1``, odor names in
    lowercase (but in the original files)."""
    try:
        ids = [int(x) for x in request.query_params.get("ids", "").split(",") if x.strip()]
    except ValueError:
        return json_error(400, "ids must be comma-separated dataset numbers")
    fmt = request.query_params.get("format", "files")
    if not ids or fmt not in export.FORMATS:
        return json_error(400, f"expected ?ids=1,2,...&format={'|'.join(export.FORMATS)}")
    cleanup_exports()
    dest = config.EXPORTS_DIR / secrets.token_urlsafe(9)
    try:
        name, media_type = await run_in_threadpool(lambda: export.write_download(db, ids, fmt, dest, config.DATASETS_DIR,
                                                                                 lower=_lower(request)))
    except export.ExportError as e:
        dest.unlink(missing_ok=True)
        return json_error(404, str(e))
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return FileResponse(dest, filename=name, media_type=media_type, background=BackgroundTask(dest.unlink, missing_ok=True))


# ----------------------------------------------------------------------- merge


def _lower(request: Request) -> bool:
    """Whether a download's odor names are to be lowercase (``lower=1``, from Settings)."""
    return request.query_params.get("lower") in ("1", "true")


def _merge_options(q: Any) -> dict:
    """The options of a merged download (see `merge.plan`), from a query or a JSON body."""
    by = q.get("by") or []
    by = [b for b in by.split(",") if b] if isinstance(by, str) else by
    try:
        tolerance = float(q.get("tolerance", merge.DEFAULT_TOLERANCE))
        power_tolerance = float(q.get("power_tolerance", 0))
    except (TypeError, ValueError):
        raise merge.MergeError("the tolerances must be numbers") from None
    if not isinstance(by, list):
        raise merge.MergeError("by must be a list")
    return {"coding_unit": str(q.get("coding_unit") or ""), "mode": str(q.get("mode") or "none"),
            "by": [str(b) for b in by], "target": str(q.get("target") or merge.AVERAGE), "tolerance": tolerance,
            "power_tolerance": power_tolerance, "calc": str(q.get("calc") or "") or None}


async def merge_units(request: Request) -> Response:
    """The coding units recorded from, with their datasets' ids."""
    return JSONResponse({"coding_units": await run_in_threadpool(merge.coding_units, db)})


async def merge_preview(request: Request) -> Response:
    """How a merged download would be scaled ({"coding_unit", "mode", "by", "target", "tolerance"}): see `merge.plan`."""
    body = await request.json()
    try:
        p = await run_in_threadpool(lambda: merge.plan(db, **_merge_options(body if isinstance(body, dict) else {})))
    except merge.MergeError as e:
        return json_error(422, str(e))
    return JSONResponse(p)


# ---------------------------------------------------------------- ROI training

def _roi_training_found(request: Request) -> tuple[dict, list[Path]]:
    """The flies to export for ROI training (`roi_training.inventory`), of the datasets
    ``?datasets=`` lists (ids, comma-separated), if it does; and the folders set in the Admin tab."""
    bases = [Path(p) for p in sandbox.folders()]
    found = roi_training.inventory(db, config.TENSOR_ROOT, bases)
    only = {int(x) for x in re.findall(r"\d+", request.query_params.get("datasets") or "")}
    if only:
        found = {"flies": [f for f in found["flies"] if any(d["id"] in only for d in f["datasets"])],
                 "skipped": [s for s in found["skipped"] if s["dataset"]["id"] in only]}
    return found, bases


async def roi_training_flies(request: Request) -> Response:
    """The flies of glomeruli datasets with a diagnostic panel to export for ROI training, and
    those left out, with why (`roi_training.summary`)."""
    def run() -> dict:
        found, bases = _roi_training_found(request)
        return roi_training.summary(found, config.TENSOR_ROOT, bases)

    return JSONResponse(await run_in_threadpool(run))


async def roi_training_download(request: Request) -> Response:
    """A zip of what the AL segmentation tab lists of each fly, with its anatomical stack if it
    has one (but with ``?stacks=0``), sent as it is written (`roi_training.stream_zip`): it may
    be several GB."""
    found, bases = await run_in_threadpool(_roi_training_found, request)
    stacks = request.query_params.get("stacks") not in ("0", "false")
    top = f"olfdb_roi_training_{dt.date.today().isoformat()}"
    return StreamingResponse(roi_training.stream_zip(found, config.TENSOR_ROOT, bases, stacks, top), media_type="application/zip",
                             headers={"content-disposition": f'attachment; filename="{top}.zip"'})


# ------------------------------------------------------------------------- fits

def _client_ip(request: Request) -> Optional[str]:
    """Where a request came from: its client's address (Docker's port mapping keeps it), and
    the first X-Forwarded-For address before it, if a proxy put one there."""
    ip = request.client.host if request.client else None
    fwd = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    return f"{fwd} (through {ip})" if fwd else ip


async def fits_list(request: Request) -> Response:
    """Every fit (`fits.listing`): those running and queued with when they should end; and the fit worker."""
    return JSONResponse(await run_in_threadpool(fits.listing, db))


async def fits_create(request: Request) -> Response:
    """Asks for a fit: {"name": the first name set in Settings, "options": `fits.options`}. Who,
    and from where (`_client_ip`), are kept with it. 422 without a name, or for one not a first
    name alone; 400 for options out of range."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, "expected a JSON object")
    name = _user_name(body.get("name"))
    if not name:
        return json_error(422, "say who you are: enter your first name in Settings")
    if first_name_problem(name):
        return json_error(422, "the name must be a first name alone, with no spaces (e.g. Ann)")
    try:
        opts = fits.options(body.get("options") or {})
    except ValueError as e:
        return json_error(400, str(e))
    return JSONResponse(await run_in_threadpool(fits.create, db, name, _client_ip(request), opts), status_code=201)


async def fits_cancel(request: Request) -> Response:
    """Stops a fit: one queued now, one running at its next step (`fits.request_cancel`)."""
    f = await run_in_threadpool(fits.request_cancel, db, request.path_params["id"])
    return JSONResponse(f) if f else json_error(404, "no such fit")


async def fits_delete(request: Request) -> Response:
    """Deletes a fit and its files. 409 while it runs (stop it first)."""
    done = await run_in_threadpool(fits.delete, db, request.path_params["id"])
    if done is None:
        return json_error(404, "no such fit")
    if done is False:
        return json_error(409, "it is running: stop it first")
    return JSONResponse({"deleted": True})


async def fits_log(request: Request) -> Response:
    """A fit's log (its last 200 kB)."""
    p = fits.fit_dir(request.path_params["id"]) / "log.txt"
    if not p.is_file():
        return json_error(404, "no log of that fit (it has not started, or is deleted)")
    with open(p, "rb") as f:
        f.seek(max(0, p.stat().st_size - 200_000))
        return PlainTextResponse(f.read().decode("utf-8", errors="replace"))


async def fits_download(request: Request) -> Response:
    """A fit's model and results, as a zip sent as it is written (`fits.stream_zip`)."""
    fit_id = request.path_params["id"]
    f = await run_in_threadpool(fits.get, db, fit_id)
    if f is None:
        return json_error(404, "no such fit")
    if f["status"] != "done" or not (fits.fit_dir(fit_id) / "model").is_dir():
        return json_error(409, "only a fit that is done has a model to download")
    top = f"olfdb_fit_{fit_id}"
    return StreamingResponse(fits.stream_zip(fit_id, top), media_type="application/zip",
                             headers={"content-disposition": f'attachment; filename="{top}.zip"'})


# ----------------------------------------------------------------- predictions

async def predictions_list(request: Request) -> Response:
    """Every prediction of glomerulus ROIs, the models to predict with, and the fit worker (`predictions.listing`)."""
    return JSONResponse(await run_in_threadpool(predictions.listing, db, config.TENSOR_ROOT))


async def predictions_create(request: Request) -> Response:
    """Asks for a fly's glomerulus ROIs, predicted by a fit's model: {"name" (as for a fit), "fit" (its
    id), "dir", "raw_dir" (the fly's motion-corrected and raw data folders, from the top of tensor or
    absolute), "overwrite"}. 201 if queued now; 200 for one already queued or running; 409 (with
    ``prediction``) if that model has predicted that fly already, unless ``overwrite`` (then it is
    run again, its files replaced). 422 without a name; 400 for a fit with no model, or a fly not found."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, "expected a JSON object")
    name = _user_name(body.get("name"))
    if not name:
        return json_error(422, "say who you are: enter your first name in Settings")
    if first_name_problem(name):
        return json_error(422, "the name must be a first name alone, with no spaces (e.g. Ann)")
    try:
        fit_id = int(body.get("fit"))
    except (TypeError, ValueError):
        return json_error(400, "fit: the id of a fit")

    def run():
        f = fits.get(db, fit_id)
        if f is None or f["status"] != "done" or not (fits.fit_dir(fit_id) / "model" / "model.json").is_file():
            raise ValueError(f"fit #{fit_id} has no model to predict with (only fits done have one)")
        fly = predictions.locate(body.get("dir"), body.get("raw_dir"), config.TENSOR_ROOT, [Path(p) for p in sandbox.folders()])
        return predictions.create(db, fit_id, fly, name, _client_ip(request), overwrite=bool(body.get("overwrite")))

    try:
        p, queued = await run_in_threadpool(run)
    except predictions.Exists as e:
        return json_error(409, str(e), prediction=e.prediction)
    except ValueError as e:
        return json_error(400, str(e))
    return JSONResponse(p, status_code=201 if queued else 200)


async def predictions_delete(request: Request) -> Response:
    """Deletes a prediction and its files. 409 while it runs."""
    done = await run_in_threadpool(predictions.delete, db, request.path_params["id"])
    if done is None:
        return json_error(404, "no such prediction")
    if done is False:
        return json_error(409, "it is running: wait for it to end")
    return JSONResponse({"deleted": True})


async def predictions_log(request: Request) -> Response:
    """A prediction's log (its last 200 kB)."""
    p = predictions.pred_dir(request.path_params["id"]) / "log.txt"
    if not p.is_file():
        return json_error(404, "no log of that prediction (it has not started, or is deleted)")
    with open(p, "rb") as f:
        f.seek(max(0, p.stat().st_size - 200_000))
        return PlainTextResponse(f.read().decode("utf-8", errors="replace"))


async def predictions_view(request: Request) -> Response:
    """What the tab's viewer shows of a prediction done (`predictions.view`): its fly's movies, each
    with the predicted ROIs and the fly's own."""
    v = await run_in_threadpool(predictions.view, db, request.path_params["id"])
    return JSONResponse(v) if v else json_error(404, "no such prediction done")


async def predictions_download(request: Request) -> Response:
    """A prediction's folder, as a zip sent as it is written (`predictions.stream_zip`)."""
    pred_id = request.path_params["id"]
    p = await run_in_threadpool(predictions.get, db, pred_id)
    if p is None or p["status"] != "done":
        return json_error(404, "no such prediction done")
    top = f"olfdb_prediction_{pred_id}_{p['fly']}_fit_{p['fit_id']}"
    return StreamingResponse(predictions.stream_zip(pred_id, top), media_type="application/zip",
                             headers={"content-disposition": f'attachment; filename="{top}.zip"'})


def _json_gzipped(request: Request, data) -> Response:
    """`data` as JSON, gzipped if large and the browser takes it (in a thread: call it in one)."""
    body = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > 200_000 and "gzip" in request.headers.get("accept-encoding", ""):
        return Response(gzip.compress(body, 6), media_type="application/json", headers={"content-encoding": "gzip", "vary": "accept-encoding"})
    return Response(body, media_type="application/json")


async def models_list(request: Request) -> Response:
    """The Models tab's table: every model of the folders of model data added in the Admin tab
    (`model_data.listing`), gzipped if the browser takes it (~10x smaller)."""
    def get():
        out = model_data.listing(db, config.TENSOR_ROOT)
        for s in out["sources"]:
            s["created_at"] = _aware(s["created_at"])
        return _json_gzipped(request, out)
    return await run_in_threadpool(get)


async def models_baselines(request: Request) -> Response:
    """The Models tab's first table: the models chosen as baselines in the Admin tab
    (`model_data.baseline_listing`)."""
    def get():
        out = model_data.baseline_listing(db, config.TENSOR_ROOT)
        for b in out["baselines"]:
            b["created_at"], b["imported_at"] = _aware(b["created_at"]), _aware(b["imported_at"])
        return _json_gzipped(request, out)
    return await run_in_threadpool(get)


async def models_run(request: Request) -> Response:
    """A model run's spike counts (?id=<its file's path on tensor>, or of a baseline's imported
    copy), KCs × odors, its odors matched to the odor table (`model_data.run`). 404 for one that
    is not a run of a folder added (nor of a baseline's copy)."""
    run_id = request.query_params.get("id", "")

    def get():
        out = model_data.run(db, run_id, _odor_index())
        if out["imported"]:
            out["imported"]["at"] = _aware(out["imported"]["at"])
        return out
    try:
        return JSONResponse(await run_in_threadpool(get))
    except LookupError as e:
        return json_error(404, str(e))
    except (OSError, ValueError) as e:  # (pyarrow's errors are ValueErrors and OSErrors)
        return json_error(422, f"it could not be read: {e}")


async def models_run_download(request: Request) -> Response:
    """A model run to save (POST {"id", "format": "files" | "wide", and of "wide", as shown: "rows"
    (its KCs), "cols" (its odors), "transpose"}), see `matrix_download.model_download`."""
    return await _matrix_download(request, lambda body, fmt, dest, picked: matrix_download.model_download(
        db, str(body.get("id") or ""), _odor_index(), fmt, dest, config.TENSOR_ROOT, **picked))


async def _matrix_download(request: Request, write) -> Response:
    """A matrix's download (`matrix_download`), of a POST of {"format", "rows", "cols",
    "transpose", ...}: `write(body, format, dest, {rows, cols, transpose})` writes it to dest."""
    body = await request.json()
    if not isinstance(body, dict) or body.get("format") not in matrix_download.FORMATS:
        return json_error(400, f'expected {{"format": {" | ".join(map(repr, matrix_download.FORMATS))}, ...}}')
    picked = {"rows": body.get("rows"), "cols": body.get("cols"), "transpose": bool(body.get("transpose"))}
    for k in ("rows", "cols"):
        if picked[k] is not None and not isinstance(picked[k], list):
            return json_error(400, f'"{k}" must be a list of numbers')
    cleanup_exports()
    dest = config.EXPORTS_DIR / secrets.token_urlsafe(9)
    try:
        name, media_type = await run_in_threadpool(lambda: write(body, body["format"], dest, picked))
    except LookupError as e:
        dest.unlink(missing_ok=True)
        return json_error(404, str(e))
    except matrix_download.DownloadError as e:
        dest.unlink(missing_ok=True)
        return json_error(422, str(e))
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return FileResponse(dest, filename=name, media_type=media_type, background=BackgroundTask(dest.unlink, missing_ok=True))


async def merge_download(request: Request) -> Response:
    """Every dataset of a coding unit, in one download (``?format=wide|long``, and the options of
    `merge_preview`; ``lower=1`` for odor names in lowercase): a zip, with the datasets scaled as
    `merge.plan` says."""
    fmt = request.query_params.get("format", "wide")
    if fmt not in ("wide", "long"):
        return json_error(400, "format must be wide or long")
    try:
        p = await run_in_threadpool(lambda: merge.plan(db, **_merge_options(request.query_params)))
    except merge.MergeError as e:
        return json_error(422, str(e))
    cleanup_exports()
    dest = config.EXPORTS_DIR / secrets.token_urlsafe(9)
    try:
        name, media_type = await run_in_threadpool(lambda: export.write_download(
            db, [d["id"] for d in p["datasets"]], fmt, dest, config.DATASETS_DIR, merged=p, lower=_lower(request)))
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return FileResponse(dest, filename=name, media_type=media_type, background=BackgroundTask(dest.unlink, missing_ok=True))


async def merge_scales(request: Request) -> Response:
    """The scale factor of each dataset of a merged download (the options of `merge_preview`,
    as a query), as a CSV indexed by dataset_id like its matrices (`merge.scaling_frame`)."""
    try:
        p = await run_in_threadpool(lambda: merge.plan(db, **_merge_options(request.query_params)))
    except merge.MergeError as e:
        return json_error(422, str(e))
    unit = re.sub(r"[^A-Za-z0-9]+", "-", p["coding_unit"]).strip("-").lower() or "merged"
    n = len(p["datasets"])
    name = f"olfdb_{unit}_{n}-dataset{'s' if n > 1 else ''}_scales.csv"
    return Response(merge.scaling_frame(p).to_csv(index=False), media_type="text/csv",
                    headers={"content-disposition": f'attachment; filename="{name}"'})


# ----------------------------------------------------------------------- users


def _user_name(v: Any) -> str:
    return " ".join(str(v or "").split())[:200]


async def user_get(request: Request) -> Response:
    """A user's data folders on tensor (``?name=``); ``user`` is null for a name with none saved."""
    name = _user_name(request.query_params.get("name"))
    return JSONResponse({"user": await run_in_threadpool(db.get_user, name) if name else None})


async def user_save(request: Request) -> Response:
    """Check a user's data folders on tensor and, if every one is there, save them under the
    name given (when there is one): {"name", "raw_dirs", "mocorr_dirs", "odor_dirs"}, each
    a list of paths relative to tensor. 422 for a name that is not a first name alone."""
    body = await request.json()
    if not isinstance(body, dict) or any(not isinstance(body.get(k) or [], list) for k in tensor.USER_DIR_KINDS):
        return json_error(400, f"expected lists of paths in {', '.join(tensor.USER_DIR_KINDS)}")
    name = _user_name(body.get("name"))
    if first_name_problem(name):
        return json_error(422, first_name_problem(name))

    def run() -> dict:
        root = tensor.root_status(config.TENSOR_ROOT)
        checks = {
            k: [tensor.check_dir(str(p), config.TENSOR_ROOT) for p in body.get(k) or [] if str(p).strip()]
            for k in tensor.USER_DIR_KINDS
        }
        ok = root["ok"] and all(c["ok"] for cs in checks.values() for c in cs)
        user = None
        if ok and name:
            dirs = {k: list(dict.fromkeys(c["path"] for c in cs)) for k, cs in checks.items()}
            user = db.save_user(name, dirs, when=dt.datetime.now().isoformat(timespec="seconds"))
        return {"ok": ok, "root": root, "dirs": checks, "saved": user is not None, "user": user}

    return JSONResponse(await run_in_threadpool(run))


# ----------------------------------------------------------------------- odors

SHEET_SOURCE = "Google sheet"
SHEET_SETTING = "odor_sheet_url"
LOWERCASE_SETTING = "odor_names_lowercase_only"  # '1' when set
sheet_watch = odormeta.SheetWatch(config.ODOR_SHEET_CHECK_MINUTES * 60)


def _aware(when: Optional[str]) -> Optional[str]:
    """A time the server wrote as its local time (ISO, with no zone), with its UTC offset
    then (as the server's time zone has it, summer time and all), for browsers to show in
    theirs: in Docker, whose time zone is UTC, it is otherwise shown as UTC."""
    if not when:
        return when
    t = dt.datetime.fromisoformat(when)
    t = t.astimezone() if t.tzinfo is None else t
    return t.isoformat(timespec="milliseconds" if t.microsecond else "seconds")


def _syncs(syncs: list[dict]) -> list[dict]:
    """Odor syncs (`Database.odor_syncs`) with their times `_aware`."""
    return [{**s, "applied_at": _aware(s["applied_at"])} for s in syncs]


def _odor_sheet() -> dict:
    """The odor sheet's link (``url``, or None): as set in the Admin tab (``saved``, with
    ``updated_at``), else from OLFDB_ODOR_SHEET_URL (``env``)."""
    saved = db.get_setting(SHEET_SETTING)
    return {"url": saved["value"] if saved else config.ODOR_SHEET_URL or None, "saved": saved is not None,
            "updated_at": _aware(saved["updated_at"]) if saved else None, "env": bool(config.ODOR_SHEET_URL)}


def _lowercase_only() -> bool:
    """Whether new odor names (in uploaded data, and in the odor table) may only be lowercase
    (set in the Admin tab); those already in the database are allowed either way."""
    saved = db.get_setting(LOWERCASE_SETTING)
    return bool(saved and saved["value"])


def _known_odor_names() -> set[str]:
    """Every odor name already in the database: the odor table's names, abbreviations and
    aliases, and the names in every dataset's odor strings."""
    known = {n for o in db.list_odors() for n in [o["name"], o.get("abbrev"), *(o.get("aliases") or [])] if n}
    return known | {n for s in db.odor_strings() for n in odor_names(s)}


def _case_errors(rows: list[dict]) -> list[str]:
    """Odor sheet rows with names in capitals not in the database, while new ones may only be lowercase."""
    return odormeta.uppercase_errors(rows, _known_odor_names()) if _lowercase_only() else []


def _sheet_status() -> dict:
    url = _odor_sheet()["url"]
    state = sheet_watch.check(url)
    last = next(iter(_syncs(db.odor_syncs(limit=1, source=SHEET_SOURCE))), None)
    return {
        "configured": bool(url),
        "url": url,
        **state,
        "checked_at": _aware(state["checked_at"]),
        "last_sync": last,
        "changed": bool(state["fingerprint"]) and (last is None or last["fingerprint"] != state["fingerprint"]),
    }


async def odors_status(request: Request) -> Response:
    """The odor sheet, when it was last read, and whether it changed since it was last synced."""
    return JSONResponse(await run_in_threadpool(_sheet_status))


async def odors_list(request: Request) -> Response:
    def build() -> dict:
        odors = db.list_odors()
        used = odormeta.usage(db, config.DATASETS_DIR, odors)
        rows = [{**o, "links": odormeta.links(o), "datasets": sorted(used.get(o["id"], ()))} for o in odors]
        return {"odors": rows, "syncs": _syncs(db.odor_syncs(limit=20))}

    return JSONResponse(await run_in_threadpool(build))


def _cleanup_previews(max_age_s: float = 86400) -> None:
    cutoff = time.time() - max_age_s
    for p in config.ODOR_PREVIEWS_DIR.glob("*.json"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def _preview(text: str, source: str, sheet_url: Optional[str] = None) -> dict:
    rows, errors, warnings = odormeta.parse_sheet(text)
    errors += _case_errors(rows)
    fp = odormeta.fingerprint(rows)
    if sheet_url:
        sheet_watch.record(sheet_url, text, rows)
    _cleanup_previews()
    (config.ODOR_PREVIEWS_DIR / f"{fp}.json").write_text(json.dumps({"rows": rows, "source": source, "errors": errors}))
    odors = db.list_odors()
    d = odormeta.diff(rows, odors)
    used = odormeta.usage(db, config.DATASETS_DIR, odors)
    # datasets that would no longer be linked to an odor if every addition and change were applied
    used_after = odormeta.usage(db, config.DATASETS_DIR, odormeta.after_sync(odors, d))
    lost = lambda i: len(used.get(i, set()) - used_after.get(i, set()))
    return {
        "ok": not errors, "source": source, "fingerprint": fp, "errors": errors, "warnings": warnings,
        "n_rows": len(rows), "unchanged": d["unchanged"],
        "added": d["added"],
        "changed": [
            {"id": c["id"], "name": c["row"]["name"], "old_name": c["odor"]["name"], "rename": c["rename"],
             "fields": c["fields"], "datasets": len(used.get(c["id"], ())), "unlinked": lost(c["id"])}
            for c in d["changed"]
        ],
        "missing": [{"id": o["id"], "name": o["name"], "abbrev": o["abbrev"], "datasets": len(used.get(o["id"], ()))}
                    for o in d["missing"]],
    }


async def odors_preview(request: Request) -> Response:
    """Validate the odor sheet (or an uploaded CSV, field 'file') and diff it against the odor table."""
    if request.headers.get("content-type", "").startswith("multipart/"):
        form = await request.form()
        up = form.get("file")
        if not isinstance(up, UploadFile):
            return json_error(400, "no file in the request (field name 'file')")
        raw = await up.read(20 * 1024 * 1024 + 1)
        if len(raw) > 20 * 1024 * 1024:
            return json_error(413, "odor CSV files are limited to 20 MB")
        text, source = raw.decode("utf-8-sig", errors="replace"), f"file {safe_filename(up.filename, 'odors.csv')}"
    else:
        url = (await run_in_threadpool(_odor_sheet))["url"]
        if not url:
            return json_error(400, "no odor sheet is set up (its link goes in the Admin tab)")
        try:
            text = await run_in_threadpool(odormeta.fetch_sheet, url)
        except odormeta.SheetError as e:
            return json_error(502, str(e))
        return JSONResponse(await run_in_threadpool(_preview, text, SHEET_SOURCE, url))
    return JSONResponse(await run_in_threadpool(_preview, text, source))


_inchi_lock = threading.Lock()


_inchi_tried: dict[int, float] = {}  # CID -> when PubChem last gave it no InChI (asked again after an hour)


def _fetch_inchis(wait: bool = False) -> None:
    """Fetch from PubChem the InChI of every odor CID that has none in the database yet
    (in one request per 100 CIDs, but those it had none for in the last hour): skipped
    while another call is at it, or with `wait`, once that one is done."""
    if not _inchi_lock.acquire(blocking=wait):
        return
    try:
        now = time.time()
        missing = sorted({o["cid"] for o in db.list_odors() if o["cid"] and not o["inchi"] and now - _inchi_tried.get(o["cid"], 0) > 3600})
        if missing:
            found = odormeta.fetch_inchis(missing)
            db.save_inchis(found, when=dt.datetime.now().isoformat(timespec="seconds"))
            _inchi_tried.update({c: now for c in missing if c not in found})
    finally:
        _inchi_lock.release()


_props_lock = threading.Lock()
# (CID, property) -> when it last could not be looked up, and why (tried again after an hour)
_props_failed: dict[tuple[int, str], tuple[float, str]] = {}
SHOWN_PROPERTIES = ("vapor_pressure", "molar_mass")  # looked up for every odor, to show with it


def _look_up_properties(todo: list[tuple[int, str, list[str]]], refresh: bool = False,
                        saved: Optional[Callable[[], None]] = None, vcf: bool = False) -> dict[tuple[int, str], str]:
    """Look up properties of odors through chemutils (`chemprops.look_up`, with `refresh`),
    each saved as it comes (then `saved` called), with the chemutils that looked it up
    (with `vcf`, as VCF's chemicals', apart from the odor table's: `db.save_vcf_property`);
    those that could not be looked up kept in `_props_failed`, and returned."""
    inchis = {cid: inchi for cid, inchi, _ in todo}

    def save(cid: int, prop: str, found: Optional[dict]) -> None:
        when, by = dt.datetime.now().isoformat(timespec="seconds"), chemprops.installed()
        if vcf:
            db.save_vcf_property(cid, inchis[cid], prop, found, when=when, chemutils=by)
        else:
            db.save_property(cid, prop, found, when=when, chemutils=by)
        if saved:
            saved()

    now = time.time()
    failed = chemprops.look_up(todo, save, refresh=refresh)
    _props_failed.update({k: (now, e) for k, e in failed.items()})
    if failed:
        log.warning("could not look up %d properties of %s (%s), e.g.: %s", len(failed), "VCF's chemicals" if vcf else "odors",
                    ", ".join(f"CID {cid}: {prop}" for cid, prop in sorted(failed)), next(iter(failed.values())))
    return failed


def _to_look_up(odor: dict, props: Iterable[str], now: float, retry_s: float = 3600) -> list[str]:
    """Those of `props` not looked up yet for an odor with an InChI (but those that could
    not be in the last `retry_s`)."""
    return [p for p in props if odor["inchi"] and not odor[f"{p}_at"]
            and now - _props_failed.get((odor["cid"], p), (0, ""))[0] > retry_s]


def _fetch_properties() -> None:
    """Look up through chemutils the properties shown with every odor (`SHOWN_PROPERTIES`)
    of those with an InChI that has not had them looked up yet (but those that could not be
    in the last hour), once the InChIs missing are fetched (`_fetch_inchis`), and then of
    those added since: skipped while another call is at it."""
    if not _props_lock.acquire(blocking=False):
        return
    seen: set[int] = set()
    try:
        while True:
            _fetch_inchis(wait=True)
            now = time.time()
            todo = {o["cid"]: (o["cid"], o["inchi"], props) for o in db.list_odors()
                    if o["cid"] not in seen and (props := _to_look_up(o, SHOWN_PROPERTIES, now))}
            if not todo:
                break
            seen.update(todo)
            _look_up_properties(sorted(todo.values()))
    except Exception:
        log.exception("could not look up properties of odors")
    finally:
        _props_lock.release()


def _look_up_odors() -> None:
    """`_fetch_properties` (and so `_fetch_inchis`) in a thread of its own, as each lookup
    takes about a second."""
    threading.Thread(target=_fetch_properties, name="odor-properties", daemon=True).start()


# the last re-lookup started from the Admin tab (`_relookup`): ``running``, ``props``,
# ``total`` and ``done`` (properties of odors), ``failed`` (how many could not be looked up,
# ``error`` an example of why), ``started_at``, ``finished_at``
_relookup_job: dict = {"running": False}
_relookup_lock = threading.Lock()


def _relookup(props: list[str]) -> None:
    """Look up again every value of `props` looked up before, through chemutils (not from
    its cache), each saved as it comes (for the Admin tab, after chemutils changed how it
    looks them up); then what was not looked up yet (`_fetch_properties`), as after a
    sync. Waits for lookups under way."""
    job = _relookup_job
    try:
        with _props_lock:
            todo = {o["cid"]: (o["cid"], o["inchi"], ps) for o in db.list_odors()
                    if o["inchi"] and (ps := [p for p in props if o[f"{p}_at"]])}
            job["total"] = sum(len(ps) for _, _, ps in todo.values())
            failed = _look_up_properties(sorted(todo.values()), refresh=True, saved=lambda: job.update(done=job["done"] + 1))
            job.update(done=job["done"] + len(failed), failed=len(failed), error=next(iter(failed.values()), None))
        log.info("looked up again from the Admin tab: %s (%d, %d could not be)", ", ".join(props), job["total"], len(failed))
    except Exception as e:
        log.exception("could not look up %s again", ", ".join(props))
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job.update(running=False, finished_at=_aware(dt.datetime.now().isoformat(timespec="seconds")))
    _fetch_properties()


def _chemutils_state() -> dict:
    """The Admin tab's: the chemutils installed (``installed``: ``commit`` or version, and
    its ``url``), each property looked up through it (``properties``: ``prop``, ``label``,
    ``looked_up`` and ``found`` of the odor table's CIDs, ``outdated``: how many were
    looked up with a chemutils older than the last ``change`` to how it is looked up,
    ``commit``, ``how`` and ``url``) and the last re-lookup (``job``, see `_relookup_job`)."""
    installed = chemprops.installed()
    url = lambda c: f"{chemprops.REPO}/commit/{c}"
    props = []
    for prop, rows in db.property_lookups().items():
        change = next((c for c in reversed(chemprops.CHANGES) if prop in c[1]), None)
        props.append({
            "prop": prop, "label": chemprops.LABELS[prop], "looked_up": sum(r["n"] for r in rows), "found": sum(r["found"] for r in rows),
            "outdated": sum(r["n"] for r in rows if chemprops.outdated(prop, r["chemutils"])),
            "change": change and {"commit": change[0], "how": change[2], "url": url(change[0])},
        })
    is_commit = bool(installed and re.fullmatch(r"[0-9a-f]{40}", installed))
    return {"installed": installed and {"commit" if is_commit else "version": installed, "url": url(installed) if is_commit else None},
            "properties": props, "job": dict(_relookup_job)}




HEADSPACE_MAX_ODORS = 50
HEADSPACE_RETRY_S = 60  # what could not be looked up is tried again by a calculation after this (s)


def _headspace_properties(odor: dict, solvent: str, model: Optional[str]) -> list[str]:
    """The properties `headspace.calculate` needs of an odor in a solvent (with `model`, for paraffin oil)."""
    props = ["molar_mass", "density", "vapor_pressure"]
    if not odor["vapor_pressure_at"] or odor["vapor_pressure_kpa"] is None:
        props.append("est_vapor_pressure")  # (if PubChem has no vapor pressure)
    henry = solvent == "water" or model == "henry"
    return props + [headspace.HENRY_PROPERTIES[solvent]] * henry


def _headspace(solvent: str, comps: list[dict], model: Optional[str] = None, oil: Optional[dict] = None) -> dict:
    """The headspace of odors diluted together (`headspace.calculate`), with the properties
    it needs of each looked up through chemutils first, where they have not been (but those
    that could not be in the last `HEADSPACE_RETRY_S`).

    Args:
        comps: ``odor_id`` (in the odor table) and ``log10_conc`` of each.
        model: of `headspace.MODELS`, for paraffin oil (the default if None).
        oil: paraffin oil's ``oil_molar_mass`` and ``oil_density``, if given (`_checked_oil`).

    Returns:
        `headspace.calculate`'s, with each component's ``odor_id``, ``log10_conc``,
        ``properties`` (property -> its value, None if there is none; of those it needs),
        ``sources`` (property -> where it is from, as chemutils said), ``links`` (property ->
        where it is from, see `odormeta.property_link`) and ``failed`` (property -> why it
        could not be looked up, on one line, for those that could not); and ``solvent``,
        ``model`` (for paraffin oil, else None) and ``temperature_c``.
    """
    model = (model or headspace.MODELS[0]) if solvent != "water" else None
    def odors() -> dict[int, dict]:
        return {o["id"]: o for o in db.list_odors()}

    table = odors()
    if any(table[c["odor_id"]]["cid"] and not table[c["odor_id"]]["inchi"] for c in comps):
        _fetch_inchis(wait=True)
        table = odors()
    now = time.time()
    todo = {}
    for c in comps:
        o = table[c["odor_id"]]
        if o["cid"] and (props := _to_look_up(o, _headspace_properties(o, solvent, model), now, HEADSPACE_RETRY_S)):
            todo[o["cid"]] = (o["cid"], o["inchi"], props)
    if todo:
        _look_up_properties(sorted(todo.values()))
        table = odors()

    inputs, extra = [], []
    for c in comps:
        o = table[c["odor_id"]]
        props = _headspace_properties(o, solvent, model)
        values = {p: o[PROPERTY_COLUMNS[p]] for p in props}
        known = {f"{p}_{k}": o[f"{p}_{k}"] for p in props for k in ("measured", "details")}
        failed = {p: " ".join(_props_failed[(o["cid"], p)][1].split())[:300] for p in props
                  if values[p] is None and (o["cid"], p) in _props_failed}
        notes = [] if o["cid"] else ["no CID in the odor table, so nothing about it could be looked up"]
        if o["cid"] and not o["inchi"]:
            notes.append("PubChem gave no InChI for its CID (or could not be reached), so nothing about it could be looked up")
        inputs.append({"log10_conc": c["log10_conc"], **values, **known})
        extra.append({"odor_id": o["id"], "log10_conc": c["log10_conc"], "properties": values,
                      "sources": {p: o[f"{p}_source"] for p in props}, "links": {p: odormeta.property_link(o, p) for p in props},
                      "failed": failed, "notes": notes})
    result = headspace.calculate(inputs, solvent, model or headspace.MODELS[0], **(oil or {}))
    for r, e in zip(result["components"], extra):
        r.update({**e, "notes": e["notes"] + r["notes"]})
    return {**result, "solvent": solvent, "model": model, "temperature_c": headspace.T_K - 273.15}


def _headspace_panels() -> list[dict]:
    """The panels of the datasets, as mixtures for the Headspace calculator: of each, the
    odors presented in it (over every dataset with it, matched to the odor table as in
    Browse) that are one of the odor table's with a CID, but solvents
    (`headspace.SOLVENT_NAMES`), each at the highest concentration presented.

    Returns:
        By panel name: ``panel``, ``datasets`` (how many have it), ``components``
        (``odor_id``, ``log10_conc``, ``concs``: every concentration presented, highest
        first) and ``skipped`` (``odor``, as in the data, and ``why``).
    """
    index = odormeta.OdorIndex(db.list_odors())
    panels: dict[str, dict] = {}
    for ds_meta in db.list_datasets():
        ds = db.get_dataset(ds_meta["id"])
        rows = odormeta.dataset_odors(ds, db.get_trials(ds["id"]), index, odormeta.dataset_yaml_odors(ds, config.DATASETS_DIR))
        for r in rows:
            if r["panel"] is None:
                continue
            p = panels.setdefault(r["panel"], {"panel": r["panel"], "datasets": set(), "concs": {}, "skipped": {}})
            p["datasets"].add(ds["id"])
            o = index.by_id.get(r["odor_id"]) if r["odor_id"] is not None else None
            names = {v.casefold() for v in ([o["name"], o.get("abbrev"), *o["aliases"]] if o else []) if v}
            why = ("not in the odor table" if o is None else "a solvent" if names & headspace.SOLVENT_NAMES
                   else "no CID in the odor table" if not o.get("cid") else "no concentration" if r["log10_conc"] is None else None)
            if why:
                p["skipped"].setdefault(r["label"], why)
            else:
                p["concs"].setdefault(o["id"], set()).add(r["log10_conc"])
    out = []
    for name in sorted(panels, key=str.casefold):
        p = panels[name]
        comps = [{"odor_id": i, "log10_conc": max(cs), "concs": sorted(cs, reverse=True)} for i, cs in p["concs"].items()]
        out.append({"panel": name, "datasets": len(p["datasets"]), "components": comps,
                    "skipped": [{"odor": k, "why": v} for k, v in p["skipped"].items()]})
    return out


async def headspace_panels(request: Request) -> Response:
    """The panels of the datasets as mixtures (`_headspace_panels`)."""
    return JSONResponse({"panels": await run_in_threadpool(_headspace_panels)})


def _checked_oil(body: dict) -> dict | Response:
    """Paraffin oil's ``oil_molar_mass`` and ``oil_density`` asked for, as `headspace.calculate`
    takes them (None if not given), else the error to answer with."""
    oil = {}
    for key, what, (least, most) in (("oil_molar_mass", "molar mass (g/mol)", headspace.OIL_MOLAR_MASS_RANGE),
                                     ("oil_density", "density (g/mL)", headspace.OIL_DENSITY_RANGE)):
        v = body.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not least <= v <= most):
            return json_error(400, f"paraffin oil's {what} is a number from {least:g} to {most:g}")
        oil[key] = v
    return oil


async def _checked_headspace(body) -> tuple[str, Optional[str], list[dict], dict] | Response:
    """A Headspace calculation asked for (as `headspace_calc` takes it): its solvent, model,
    components and paraffin oil's properties, else the error to answer with."""
    if not isinstance(body, dict) or body.get("solvent") not in headspace.SOLVENTS or not isinstance(body.get("components"), list):
        return json_error(400, f'expected {{"solvent": one of {", ".join(headspace.SOLVENTS)}, "components": [{{"odor_id", "log10_conc"}}, ...]}}')
    if body.get("model") is not None and body["model"] not in headspace.MODELS:
        return json_error(400, f"the model is one of {', '.join(headspace.MODELS)}")
    oil = _checked_oil(body)
    if isinstance(oil, Response):
        return oil
    comps = body["components"]
    if not 1 <= len(comps) <= HEADSPACE_MAX_ODORS:
        return json_error(400, f"a mixture needs from 1 to {HEADSPACE_MAX_ODORS} odors")
    ids = {o["id"] for o in await run_in_threadpool(db.list_odors)}
    for c in comps:
        conc = c.get("log10_conc") if isinstance(c, dict) else None
        if not isinstance(c, dict) or type(c.get("odor_id")) is not int or c["odor_id"] not in ids:
            return json_error(400, f"not an odor of the odor table: {c.get('odor_id') if isinstance(c, dict) else c!r}")
        if isinstance(conc, bool) or not isinstance(conc, (int, float)) or not -50 <= conc <= 0:
            return json_error(400, "each odor's log10 concentration (v/v) is a number from -50 to 0")
    if len({c["odor_id"] for c in comps}) < len(comps):
        return json_error(400, "an odor is in the mixture more than once")
    return body["solvent"], body.get("model"), comps, oil


async def headspace_calc(request: Request) -> Response:
    """The Headspace calculator's: {"solvent" (of `headspace.SOLVENTS`), "model" (of
    `headspace.MODELS`, for paraffin oil; optional), "oil_molar_mass" and "oil_density" (for
    paraffin oil; optional), "components": [{"odor_id", "log10_conc"}, ...]} -> `_headspace`."""
    checked = await _checked_headspace(await request.json())
    if isinstance(checked, Response):
        return checked
    solvent, model, comps, oil = checked
    try:
        return JSONResponse(await run_in_threadpool(_headspace, solvent, comps, model, oil))
    except ValueError as e:
        return json_error(400, str(e))


async def headspace_file(request: Request) -> Response:
    """A Headspace calculation's table as a file to save, a CSV or a LaTeX table
    (`export.headspace_download`): ``solvent``, ``model``, ``oil_molar_mass`` and
    ``oil_density`` (optional), ``odors`` (``<odor id>:<log10 conc>,...``), ``units`` (of
    `headspace.UNITS`, for LaTeX) and ``lower=1`` for abbreviations in lowercase."""
    q = request.query_params
    try:
        comps = [{"odor_id": int(i), "log10_conc": float(c)} for i, c in (x.split(":") for x in q.get("odors", "").split(",") if x)]
    except ValueError:
        return json_error(400, "expected odors=<odor id>:<log10 concentration>,...")
    try:
        given = {k: float(q[k]) for k in ("oil_molar_mass", "oil_density") if q.get(k)}
    except ValueError:
        return json_error(400, "expected oil_molar_mass and oil_density as numbers")
    checked = await _checked_headspace({"solvent": q.get("solvent"), "model": q.get("model") or None, "components": comps, **given})
    if isinstance(checked, Response):
        return checked
    solvent, model, comps, oil = checked

    def run() -> tuple[str, str, str]:
        result = _headspace(solvent, comps, model, oil)
        return export.headspace_download(result, {o["id"]: o for o in db.list_odors()}, request.path_params["fmt"],
                                         q.get("units") or "ppm", lower=_lower(request))

    try:
        text, name, media_type = await run_in_threadpool(run)
    except (export.ExportError, ValueError) as e:
        return json_error(400, str(e))
    return Response(text, media_type=media_type, headers={"content-disposition": f'attachment; filename="{name}"'})


DILUTION_MAX_ML = 1000.0  # of solvent in each vial, at most (mL)


async def headspace_dilutions(request: Request) -> Response:
    """The Headspace calculator's serial dilutions: {"targets" (log10 v/v, from -50 to 0),
    "solvent_ml", "neat_ul" ([the least, the most]), "dilution_ul" ([the least, the most],
    each optional)} -> `dilution.plan`."""
    body = await request.json()
    num = lambda v: not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
    pair = lambda v: isinstance(v, list) and len(v) == 2
    if not isinstance(body, dict) or not isinstance(body.get("targets"), list):
        return json_error(400, 'expected {"targets": [log10 v/v, ...], "solvent_ml", "neat_ul": [least, most], "dilution_ul": [least, most]}')
    targets = body["targets"]
    if not 1 <= len(targets) <= HEADSPACE_MAX_ODORS:
        return json_error(400, f"from 1 to {HEADSPACE_MAX_ODORS} odors")
    if not all(num(t) and -50 <= t <= 0 for t in targets):
        return json_error(400, "each odor's log10 concentration (v/v) is a number from -50 to 0")
    ml = body.get("solvent_ml", dilution.SOLVENT_ML)
    if not num(ml) or not 0 < ml <= DILUTION_MAX_ML:
        return json_error(400, f"the solvent in each vial is more than 0 mL, and at most {DILUTION_MAX_ML:,g}")
    neat = body.get("neat_ul", list(dilution.NEAT_UL))
    if not pair(neat) or not all(num(v) and v > 0 for v in neat) or neat[0] > neat[1]:
        return json_error(400, "the neat chemical's least and most (µL) are numbers more than 0, the least at most the most")
    dil = body.get("dilution_ul", [None, None])
    if not pair(dil) or not all(v is None or (num(v) and v > 0) for v in dil) or (None not in dil and dil[0] > dil[1]):
        return json_error(400, "the dilutions' least and most (µL) are numbers more than 0 (or null), the least at most the most")
    return JSONResponse(await run_in_threadpool(dilution.plan, targets, ml, tuple(neat), tuple(dil)))


VCF_MAX_OIL = 500  # chemicals with a headspace over their samples, at most, to calculate in paraffin oil
VCF_LOOKUP_BATCH = 10  # CIDs looked up through chemutils at once for VCF's chemicals (`_vcf_look_up`)
# what is looked up of VCF's chemicals first (`_vcf_needs`): the Henry's law constants and vapor
# pressures VCF's file lacks (of those with neither first), then what paraffin oil needs, then
# those the file has, to check them
VCF_PRIORITY = {"neither": 0, "no k_henry": 1, "oil": 2, "check": 3}
# (CID, property) -> (priority, order asked, InChI) of what is to be looked up of VCF's
# chemicals in the background, and those being looked up (`_vcf_look_up`)
_vcf_queue: dict[tuple[int, str], tuple[int, int, str]] = {}
_vcf_busy: set[tuple[int, str]] = set()
_vcf_lock = threading.Lock()
_vcf_worker = {"running": False, "asked": 0}


def _vcf_look_up() -> None:
    """Look up through chemutils what was asked for of VCF's chemicals (`_vcf_queue`), what
    comes first first, a few CIDs at a time (with all asked for of each), each property
    saved as it comes, until nothing is left."""
    while True:
        with _vcf_lock:
            _vcf_busy.clear()
            if not _vcf_queue:
                _vcf_worker["running"] = False
                return
            cids: list[int] = []
            for (cid, _), _ in sorted(_vcf_queue.items(), key=lambda kv: kv[1][:2]):
                if cid not in cids:
                    cids.append(cid)
                    if len(cids) == VCF_LOOKUP_BATCH:
                        break
            batch: dict[int, tuple[str, list[str]]] = {}
            for key in [k for k in _vcf_queue if k[0] in cids]:
                inchi = _vcf_queue.pop(key)[2]
                batch.setdefault(key[0], (inchi, []))[1].append(key[1])
                _vcf_busy.add(key)
        try:
            _look_up_properties([(cid, inchi, props) for cid, (inchi, props) in batch.items()], vcf=True)
        except Exception:
            log.exception("could not look up properties of VCF's chemicals")


def _vcf_ask(needs: dict[tuple[int, str], tuple[int, str]]) -> set[tuple[int, str]]:
    """Have what VCF's chemicals need ((CID, property) -> (priority, InChI)) looked up in the
    background (`_vcf_look_up`), sooner if asked for sooner than it was; what of it, and of
    what was asked before, is being looked up."""
    with _vcf_lock:
        for key, (priority, inchi) in needs.items():
            if key in _vcf_busy:
                continue
            if key not in _vcf_queue or priority < _vcf_queue[key][0]:
                _vcf_worker["asked"] += 1
                _vcf_queue[key] = (priority, _vcf_worker["asked"], inchi)
        start = bool(_vcf_queue) and not _vcf_worker["running"]
        _vcf_worker["running"] |= start
        pending = set(_vcf_queue) | _vcf_busy
    if start:
        threading.Thread(target=_vcf_look_up, name="vcf-properties", daemon=True).start()
    return pending


def _vcf_row(cid: int, inchi: str, looked: dict[str, dict]) -> dict:
    """What was looked up of a chemical of VCF's (as `db.vcf_properties` gives them), as
    `db.list_odors` gives an odor's (for `_headspace_properties`, `_to_look_up` and
    `odormeta.property_link`)."""
    row = {"cid": cid, "inchi": inchi}
    for p, col in PROPERTY_COLUMNS.items():
        x = looked.get(p) or {}
        row.update({col: x.get("value"), f"{p}_source": x.get("source"), f"{p}_at": x.get("at"),
                    f"{p}_measured": x.get("measured"), f"{p}_details": x.get("details")})
    return row


def _vcf_needs(c: dict, looked: dict[str, dict], now: float) -> dict[str, int]:
    """What is to be looked up of a chemical of VCF's (`vcf.search`'s, with a CID and InChI)
    for its headspace over the samples: property -> its priority (`VCF_PRIORITY`). Its
    Henry's law constant and vapor pressure, and estimates of them (`vcf.USED`) once
    neither the file nor chemutils has one; but those that could not be looked up in the
    last hour."""
    file = c.get("file") or c
    why = "check" if file["k_henry"] is not None else "no k_henry" if file["vapor_pressure"] is not None else "neither"
    out = {}
    for p, est in vcf.USED.items():
        if p not in looked:
            out[p] = VCF_PRIORITY[why]
        elif looked[p]["value"] is None and file[p] is None and est not in looked:
            out[est] = VCF_PRIORITY[why]
    return {p: pr for p, pr in out.items() if now - _props_failed.get((c["cid"], p), (0, ""))[0] > 3600}


def _vcf_oil(chemicals: list[dict], model: str, looked: dict[int, dict], oil: Optional[dict] = None) -> dict:
    """The paraffin oil that would give the headspace over VCF's samples (`vcf.search`'s
    ``chemicals``, each with its ``pa``: `vcf.add_headspace`): the v/v of each in it
    (`headspace.invert`, of them all together, with its molar mass from its InChI), from
    what is known of each (`looked`, as `db.vcf_properties` gives it), the rest asked to be
    looked up through chemutils in the background (`_vcf_ask`); `oil` as `_headspace` takes it.

    Returns:
        ``model``, ``chemicals`` (of each given, in order: `headspace.invert`'s, with
        ``properties``, ``sources``, ``links`` and ``failed`` as `_headspace` gives them, and
        ``pending``: whether its properties are being looked up; None for those with no
        headspace), ``total`` (v/v, of those computed), ``solvent_molar_mass`` and
        ``solvent_density`` (of the oil, as taken), ``pending`` (how many are being looked
        up) and ``error`` (if there are too many to compute).
    """
    targets = [c for c in chemicals if c["pa"] is not None]
    if len(targets) > VCF_MAX_OIL:
        return {"model": model, "chemicals": [None] * len(chemicals), "total": None, "solvent_molar_mass": None, "solvent_density": None, "pending": 0,
                "error": f"{len(targets):,} chemicals with a headspace, and at most {VCF_MAX_OIL} are calculated in oil: narrow the search"}
    rows = {c["cid"]: _vcf_row(c["cid"], c["inchi"], looked.get(c["cid"], {})) for c in targets if c["cid"] and c["inchi"]}
    now = time.time()
    needs = lambda row: [p for p in _headspace_properties(row, "paraffin oil", model) if p != "molar_mass"]  # (from the InChI)
    pending_keys = _vcf_ask({(cid, p): (VCF_PRIORITY["oil"], row["inchi"]) for cid, row in rows.items()
                             for p in _to_look_up(row, needs(row), now)})
    pending = {cid for cid, row in rows.items() if any((cid, p) in pending_keys for p in needs(row))}

    inputs, extra = [], []
    for c in chemicals:
        row = rows.get(c["cid"]) if c["pa"] is not None else None
        if c["pa"] is None:
            inputs.append({"pa": None})
            extra.append(None)
            continue
        if row is None:
            inputs.append({"pa": None})
            extra.append({"properties": {}, "sources": {}, "links": {}, "failed": {}, "pending": False,
                          "notes": ["no CID or InChI in the file, so nothing could be looked up of it in oil"]})
            continue
        props = needs(row)
        values = {p: row[PROPERTY_COLUMNS[p]] for p in props}
        known = {f"{p}_{k}": row[f"{p}_{k}"] for p in props for k in ("measured", "details")}
        waiting = c["cid"] in pending
        inputs.append({"pa": None if waiting else c["pa"], "molar_mass": c["molar_mass"], **values, **known})
        failed = {p: " ".join(_props_failed[(c["cid"], p)][1].split())[:300] for p in props
                  if values[p] is None and (c["cid"], p) in _props_failed}
        extra.append({"properties": values, "sources": {p: row[f"{p}_source"] for p in props},
                      "links": {p: odormeta.property_link(row, p) for p in props}, "failed": failed, "pending": waiting,
                      "notes": ["being looked up through chemutils"] if waiting else []})
    result = headspace.invert(inputs, "paraffin oil", model, **(oil or {}))
    out = []
    for r, e in zip(result["components"], extra):
        out.append(None if e is None else {**r, **e, "notes": e["notes"] + r["notes"]})
    return {"model": model, "chemicals": out, "total": result["total"], "solvent_molar_mass": result["solvent_molar_mass"],
            "solvent_density": result["solvent_density"], "pending": len(pending), "error": None}


def _vcf_sources() -> list[dict]:
    return vcf.sources(reference.entries(db))


def _vcf_listing() -> list[dict]:
    """The VCF reference datasets (`vcf.sources`), for the Headspace calculator: each ``id``,
    ``title``, and how many ``lines`` and ``chemicals`` it has, with the names of its
    ``products``, ``categories`` and ``groups`` (to search for); or ``error``, why it
    cannot be read."""
    out = []
    for s in _vcf_sources():
        try:
            t = vcf.load(s)
        except Exception as e:
            out.append({"id": s["id"], "title": s["title"], "error": reference._read_error(Path(s["path"]), e)})
            continue
        out.append({"id": s["id"], "title": s["title"], "lines": len(t["lines"]), "chemicals": len(t["chemicals"]), **t["names"]})
    return out


def _vcf_search(body: dict) -> dict:
    """`vcf.search` of a VCF reference dataset (``source``, its id; else the first of
    `vcf.sources`) for ``query``, its quantities reduced ``across_references`` and
    ``across_products``, each chemical with the odor of the odor table it is (``odor_id``,
    matched as reference datasets' are), and its headspace over the samples
    (`vcf.add_headspace`), from what was looked up of it through chemutils, the rest asked
    to be looked up in the background (`_vcf_needs`): ``lookups`` says how many chemicals
    have something being looked up (``pending``) and differ from the file (``mismatches``,
    by more than ``tolerance``, `vcf.MISMATCH`).
    With ``model`` (of `headspace.MODELS`), the paraffin oil to match its headspace
    (``oil``: `_vcf_oil`, of ``oil_molar_mass`` and ``oil_density``, if given). With its
    ``source`` (``id``, ``title``).

    Raises:
        vcf.VcfError: if there is no such VCF reference dataset, or for `vcf.search`'s.
    """
    sources = _vcf_sources()
    source = next((s for s in sources if s["id"] == body.get("source")), None) if body.get("source") else next(iter(sources), None)
    if source is None:
        raise vcf.VcfError("no such VCF reference dataset" if body.get("source") else
                           "no VCF reference dataset: add one in the Admin tab (Reference data)")
    res = vcf.search(vcf.load(source), str(body.get("query") or ""), body.get("across_references") or "median",
                     body.get("across_products") or "median")
    chems = res["chemicals"]
    index = odormeta.OdorIndex(db.list_odors())
    for c in chems:
        c["odor_id"] = index.match_chemical({k: c[k] for k in ("cid", "inchi", "name") if c[k]}, c["name"])[0]
    known = [c for c in chems if c["cid"] and c["inchi"]]
    looked = db.vcf_properties(c["cid"] for c in known)
    now = time.time()
    pending = _vcf_ask({(c["cid"], p): (pr, c["inchi"]) for c in known for p, pr in _vcf_needs(c, looked.get(c["cid"], {}), now).items()})
    vcf.add_headspace(chems, looked, pending)
    res["lookups"] = {"pending": sum(bool(c["pending"]) for c in chems), "mismatches": sum(bool(c["mismatches"]) for c in chems),
                      "tolerance": vcf.MISMATCH}
    if body.get("model"):
        res["oil"] = _vcf_oil(chems, body["model"], looked, {k: body.get(k) for k in ("oil_molar_mass", "oil_density")})
    return {**res, "source": {"id": source["id"], "title": source["title"]}, "temperature_c": headspace.T_K - 273.15}


async def headspace_vcf(request: Request) -> Response:
    """GET: the VCF reference datasets to search (`_vcf_listing`). POST: a search of one
    ({"source", "query", "across_references", "across_products" (of `vcf.STATS`), "model"
    (of `headspace.MODELS`, or null), "oil_molar_mass" and "oil_density" (optional)}) ->
    `_vcf_search`."""
    if request.method == "GET":
        return JSONResponse({"sources": await run_in_threadpool(_vcf_listing), "stats": vcf.STATS, "max_oil": VCF_MAX_OIL})
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"source", "query", "across_references", "across_products", "model"}')
    if body.get("model") is not None and body["model"] not in headspace.MODELS:
        return json_error(400, f"the model is one of {', '.join(headspace.MODELS)}")
    if isinstance(oil := _checked_oil(body), Response):
        return oil
    try:
        return JSONResponse(await run_in_threadpool(_vcf_search, body))
    except vcf.VcfError as e:
        return json_error(400, str(e))

async def odors_file(request: Request) -> Response:
    """The odor table as a file to save: a CSV or a LaTeX table (`export.odor_table_download`;
    ``q`` for the odors the Odors tab's filter shows for it, ``lower=1`` for abbreviations in
    lowercase)."""
    try:
        text, name, media_type = await run_in_threadpool(lambda: export.odor_table_download(
            db, config.DATASETS_DIR, request.path_params["fmt"], query=request.query_params.get("q", ""), lower=_lower(request)))
    except export.ExportError as e:
        return json_error(404, str(e))
    return Response(text, media_type=media_type, headers={"content-disposition": f'attachment; filename="{name}"'})


def _apply(fp: str, change_ids: set[int], delete_ids: set[int]) -> dict:
    path = config.ODOR_PREVIEWS_DIR / f"{fp}.json"
    if not path.exists():
        return {"ok": False, "errors": ["this preview has expired: check the sheet (or import the file) again"]}
    saved = json.loads(path.read_text())
    if saved["errors"]:
        return {"ok": False, "errors": ["the sheet has errors: fix them and check it again"]}
    case = _case_errors(saved["rows"])  # (if new odor names may only be lowercase since it was checked)
    if case:
        return {"ok": False, "errors": case}
    d = odormeta.diff(saved["rows"], db.list_odors())
    changes = [c for c in d["changed"] if c["id"] in change_ids]
    deletions = [o for o in d["missing"] if o["id"] in delete_ids]
    details = {
        "added": [r["name"] for r in d["added"]],
        "changed": [{"name": c["row"]["name"], "fields": c["fields"]} for c in changes],
        "deleted": [o["name"] for o in deletions],
        "not_applied": [c["row"]["name"] for c in d["changed"] if c["id"] not in change_ids],
    }
    try:
        db.apply_odor_sync(
            added=d["added"], changed=[(c["id"], c["row"]) for c in changes], deleted=[o["id"] for o in deletions],
            source=saved["source"], fingerprint=fp, details=details,
            when=dt.datetime.now().isoformat(timespec="seconds"),
        )
    except sqlite3.IntegrityError as e:
        return {"ok": False, "errors": [f"could not apply the sync: {e}"]}
    return {"ok": True, "added": len(d["added"]), "changed": len(changes), "deleted": len(deletions)}


async def odors_apply(request: Request) -> Response:
    """Apply a preview: all new odors, plus the changes and deletions listed (by odor id).
    It is not tied to anyone (any ``applied_by`` given is ignored)."""
    body = await request.json()
    fp = str(body.get("fingerprint", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", fp):
        return json_error(400, "missing or malformed fingerprint")
    try:
        change_ids = {int(x) for x in body.get("changes") or []}
        delete_ids = {int(x) for x in body.get("deletions") or []}
    except (TypeError, ValueError):
        return json_error(400, "changes and deletions must be lists of odor ids")
    result = await run_in_threadpool(_apply, fp, change_ids, delete_ids)
    return JSONResponse(result, status_code=200 if result["ok"] else 422,
                        background=BackgroundTask(_look_up_odors) if result["ok"] else None)


async def odor_structure(request: Request) -> Response:
    """The structure of an odor in the table, drawn by RDKit from the InChI of its CID, as
    SVG (`odormeta.structure_svg`). An InChI not in the database yet is fetched from
    PubChem first, with every other one missing."""
    cid = int(request.path_params["cid"])

    def find() -> Optional[dict]:
        return next((o for o in db.list_odors() if o["cid"] == cid), None)

    def draw() -> Optional[str]:
        odor = find()
        if odor and not odor["inchi"]:
            _fetch_inchis(wait=True)
            odor = find()
        return odormeta.structure_svg(odor["inchi"]) if odor and odor["inchi"] else None

    if await run_in_threadpool(find) is None:
        return json_error(404, "no odor with this CID")
    svg = await run_in_threadpool(draw)
    if svg is None:
        return json_error(404, "no structure: PubChem has no InChI for this CID (or could not be reached), or RDKit cannot read it")
    return Response(svg, media_type="image/svg+xml", headers={"cache-control": "public, max-age=604800"})


async def odor_template(request: Request) -> Response:
    return FileResponse(config.ODOR_TEMPLATE, filename="odor_sheet_template.csv", media_type="text/csv")


def _odor_matches(trials: list[dict], yaml_odors: list[yamlcheck.YamlOdor]) -> dict:
    """The odors of a dataset's `trials` (one per panel and component odor) matched to the
    odor table through its `yaml_odors` (``rows``: see `odormeta.dataset_odors`; ``odors``:
    those of the table they are, or could be); those not in it as rows for the odor sheet
    (``missing``, see `odormeta.missing_rows`), and these as cells to paste into it
    (``sheet``: ``paste``, in the order of the sheet's columns as last read, else the
    template's: ``columns_from`` 'sheet' or 'template', with ``checked_at`` and ``error``
    from reading it, and its ``url``, None if none is set)."""
    index = odormeta.OdorIndex(db.list_odors())
    rows = odormeta.dataset_odors({}, trials, index, yaml_odors)
    ids = {r["odor_id"] for r in rows} | {i for r in rows for i in r["candidates"]}
    odors = {i: {**index.by_id[i], "links": odormeta.links(index.by_id[i])} for i in ids if i is not None}
    missing = odormeta.missing_rows(rows)
    sheet = None
    if missing:  # the sheet read only then (at most every few minutes: see SheetWatch)
        url = _odor_sheet()["url"]
        state = sheet_watch.check(url) if url else odormeta.SheetWatch.UNREAD
        sheet = {"paste": odormeta.sheet_paste(missing, state["columns"]),
                 "columns_from": "sheet" if state["columns"] else "template",
                 "url": url, "checked_at": _aware(state["checked_at"]), "error": state["error"]}
    return {"rows": rows, "odors": odors, "missing": missing, "sheet": sheet}


def _label_odor_matches(labels: list[str]) -> dict:
    """`_odor_matches` of odors as a reference dataset's labels name them (as datasets write
    them, '<abbreviation> @ <log10 conc>'), each as a trial of its own, of no panel."""
    return _odor_matches([{"panel": None, "components": [lab]} for lab in labels], [])


def _unmatched(rows: list[dict]) -> list[str]:
    """The odors (as in the data) of `_odor_matches` rows not one odor in the odor table."""
    return list(dict.fromkeys(r["odor"] for r in rows if r["odor_id"] is None))


async def compare_dataset(request: Request) -> Response:
    """A dataset as the Compare tab shows it: its odors by its ROIs (or glomeruli), each
    the mean of its trials (and flies), see `compare.dataset`."""
    result = await run_in_threadpool(compare.dataset, db, int(request.path_params["ds_id"]), config.DATASETS_DIR)
    if result is None:
        return json_error(404, "dataset not found")
    return JSONResponse(result)


async def compare_ecfp(request: Request) -> Response:
    """ECFP fingerprints of the odor table's chemicals (``?diameter=``: even, from 0 to 12;
    6, ECFP6, if not given), see `compare.fingerprints`."""
    try:
        diameter = int(request.query_params.get("diameter", 6))
    except ValueError:
        diameter = -1
    if not 0 <= diameter <= 12 or diameter % 2:
        return json_error(400, "the diameter must be an even number from 0 to 12")
    return JSONResponse(await run_in_threadpool(lambda: compare.fingerprints(db.list_odors(), diameter)))


async def dataset_odor_table(request: Request) -> Response:
    """A dataset's odors (per panel, as in the data) matched to the odor table (see `_odor_matches`)."""
    ds_id = int(request.path_params["ds_id"])

    def build() -> Optional[dict]:
        ds = db.get_dataset(ds_id)
        if ds is None:
            return None
        return _odor_matches(db.get_trials(ds_id), odormeta.dataset_yaml_odors(ds, config.DATASETS_DIR))

    result = await run_in_threadpool(build)
    if result is None:
        return json_error(404, "dataset not found")
    return JSONResponse(result)


async def dataset_odors_file(request: Request) -> Response:
    """A dataset's odors as a file to save: a CSV or a LaTeX table (`export.dataset_odors_download`;
    ``lower=1`` for its odor names in lowercase)."""
    ds_id, fmt = int(request.path_params["ds_id"]), request.path_params["fmt"]
    try:
        text, name, media_type = await run_in_threadpool(
            lambda: export.dataset_odors_download(db, ds_id, config.DATASETS_DIR, fmt, lower=_lower(request)))
    except export.ExportError as e:
        return json_error(404, str(e))
    return Response(text, media_type=media_type, headers={"content-disposition": f'attachment; filename="{name}"'})


MAX_CLUSTER_ITEMS = 30000  # as cluster.js clusters at most (in the browser)


def _clustering_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def dataset_clusterings(request: Request) -> Response:
    """Clusterings kept for a dataset ({"keys": [what each is of, as the viewer describes it]}):
    {"found": {key: {"order": base64 int32, "merges": base64 float64}}} for those there are."""
    ds_id = int(request.path_params["ds_id"])
    body = await request.json()
    keys = body.get("keys") if isinstance(body, dict) else None
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys) or len(keys) > 100:
        return json_error(400, 'expected {"keys": [...]}')
    when = dt.datetime.now().isoformat(timespec="seconds")
    found = await run_in_threadpool(db.get_clusterings, ds_id, [_clustering_key(k) for k in keys], when)
    by_hash = {_clustering_key(k): k for k in keys}
    return JSONResponse({"found": {by_hash[h]: {"order": base64.b64encode(o).decode("ascii"), "merges": base64.b64encode(m).decode("ascii")}
                                   for h, (o, m) in found.items()}})


async def dataset_clusterings_save(request: Request) -> Response:
    """Keep a clustering of a dataset, as a browser computed it, for any to use again:
    {"key" (what it is of, as the viewer describes it: JSON with its number of items, n, among
    the rest), "order" (base64 int32: the leaf order, a permutation of 0..n-1), "merges" (base64
    float64: n-1 merges (a, b, height), scipy's convention)}. Refused unless it is one of n items."""
    ds_id = int(request.path_params["ds_id"])
    body = await request.json()
    try:
        key = body["key"]
        n = json.loads(key)["n"]
        order = np.frombuffer(base64.b64decode(body["order"], validate=True), dtype="<i4")
        merges = np.frombuffer(base64.b64decode(body["merges"], validate=True), dtype="<f8").reshape(-1, 3)
    except (KeyError, TypeError, ValueError, binascii.Error):
        return json_error(400, 'expected {"key", "order", "merges"}')
    ab = merges[:, :2]
    if not (isinstance(n, int) and 2 <= n <= MAX_CLUSTER_ITEMS and len(key) <= 2_000_000 and len(order) == n and len(merges) == n - 1
            and np.array_equal(np.sort(order), np.arange(n)) and np.all(np.isfinite(merges))
            and np.all(ab == np.round(ab)) and np.all(ab >= 0) and np.all(ab < 2 * n - 1)):
        return json_error(422, "not a clustering of the items its key says")
    when = dt.datetime.now().isoformat(timespec="seconds")
    ok = await run_in_threadpool(db.save_clustering, ds_id, _clustering_key(key), order.tobytes(), merges.tobytes(), when)
    return JSONResponse({"ok": True}) if ok else json_error(404, "dataset not found")


def _odor_index() -> odormeta.OdorIndex:
    return odormeta.OdorIndex(db.list_odors())


async def reference_list(request: Request) -> Response:
    """Every reference dataset, with whether it can be shown (and why not) and whether its chemicals are identified."""
    refs = await run_in_threadpool(lambda: reference.listing(db, _odor_index()))
    return JSONResponse({"datasets": [{**r, "created_at": _aware(r.get("created_at")), "imported_at": _aware(r.get("imported_at"))}
                                      for r in refs]})


def _metacyc(request: Request) -> tuple[metacyc.Bundle, Optional[dict], dict[str, dict]]:
    """The MetaCyc bundle of the reference dataset in the path, its view's fields, and the
    odors of the odor table its compounds are ({frame ID: {id, name, abbrev}}).

    Raises:
        HTTPException: 404 if there is no such reference dataset of a bundle, 409 if its
            bundle cannot be read now.
    """
    try:
        src = reference.metacyc_source(db, request.path_params["ref_id"])
    except reference.ReferenceError as e:
        raise HTTPException(409, str(e)) from None
    if src is None:
        raise HTTPException(404, "no such reference dataset of MetaCyc")
    try:
        b = metacyc.bundle(src[0])
    except (OSError, metacyc.MetacycError) as e:
        raise HTTPException(409, f"its MetaCyc bundle cannot be read: {e}") from None
    return b, src[1], metacyc.odor_marks(b, db.list_odors())


async def metacyc_pathway(request: Request) -> Response:
    """A pathway of a MetaCyc reference dataset's bundle, to draw and describe (`metacyc.pathway`)."""
    def get():
        b, _, odors = _metacyc(request)
        return metacyc.pathway(b, request.path_params["frame"], odors)
    out = await run_in_threadpool(get)
    return JSONResponse(out) if out is not None else json_error(404, "MetaCyc has no such pathway")


async def metacyc_compound(request: Request) -> Response:
    """A compound (or class) of a MetaCyc reference dataset's bundle (`metacyc.compound`)."""
    def get():
        b, _, odors = _metacyc(request)
        return metacyc.compound(b, request.path_params["frame"], odors)
    out = await run_in_threadpool(get)
    return JSONResponse(out) if out is not None else json_error(404, "MetaCyc has no such compound")


async def metacyc_lca(request: Request) -> Response:
    """The pathway LCA distance of two compounds (``a``, ``b``: frame IDs) in the network of a
    MetaCyc reference dataset's view (or the default one's), with the paths from their LCAs (`metacyc.explain`)."""
    a, c = request.query_params.get("a", ""), request.query_params.get("b", "")
    if not a or not c:
        return json_error(400, "give two compounds' frame IDs: ?a=...&b=...")

    def get():
        b, fields, _ = _metacyc(request)
        return metacyc.explain(b, fields, a, c)
    return JSONResponse(await run_in_threadpool(get))


async def metacyc_search(request: Request) -> Response:
    """Pathways and compounds of a MetaCyc reference dataset's bundle by name or frame ID (``q``)."""
    def get():
        b, _, _ = _metacyc(request)
        return metacyc.search(b, request.query_params.get("q", ""))
    return JSONResponse(await run_in_threadpool(get))


async def metacyc_structure(request: Request) -> Response:
    """A compound's structure drawn from its InChI in a MetaCyc bundle, as SVG."""
    def get():
        b, _, _ = _metacyc(request)
        c = b.compounds.get(request.path_params["frame"])
        return odormeta.structure_svg(c["inchi"]) if c and c["inchi"] else None
    svg = await run_in_threadpool(get)
    if svg is None:
        return json_error(404, "no structure: MetaCyc has no InChI for it, or RDKit cannot read it")
    return Response(svg, media_type="image/svg+xml", headers={"cache-control": "max-age=86400"})


async def lotus_cell(request: Request) -> Response:
    """What a cell of a LOTUS reference dataset is of (`lotus.cell`): its row's structures
    (``structures``: IDs, as the row's ``lotus``) and its column's taxon (``name`` and
    ``kingdom``, or ``organism``: an ID, as the column's ``lotus``), as its view's options say."""
    q = request.query_params
    try:
        structures = [int(x) for x in q.get("structures", "").split(",") if x.strip()]
        organism = int(q["organism"]) if q.get("organism") else None
    except ValueError:
        return json_error(400, "structures: IDs separated by commas; organism: an ID")
    if not structures:
        return json_error(400, "give the row's structures: ?structures=1,2&name=...&kingdom=...")

    def get():
        try:
            src = reference.bundle_source(db, request.path_params["ref_id"], "lotus")
        except reference.ReferenceError as e:
            raise HTTPException(409, str(e)) from None
        if src is None:
            raise HTTPException(404, "no such reference dataset of LOTUS")
        try:
            b = lotus.bundle(src[0])
            return lotus.cell(b, src[1], structures, {"name": q.get("name"), "kingdom": q.get("kingdom") or None,
                                                      "organism": organism})
        except OSError as e:
            raise HTTPException(409, f"its LOTUS bundle cannot be read: {e}") from None
        except lotus.LotusError as e:
            raise HTTPException(404, str(e)) from None
    return JSONResponse(await run_in_threadpool(get))


async def reference_get(request: Request) -> Response:
    """A reference dataset for the viewer (`reference.load`): gzipped, if large and the browser takes it."""
    data = await run_in_threadpool(lambda: reference.load(db, request.path_params["ref_id"], _odor_index()))
    if data is None:
        return json_error(404, "reference dataset not found")
    data["imported_at"] = _aware(data.get("imported_at"))
    c = data.get("chemicals") or {}
    if c.get("odors"):  # (its odors checked against the odor table as an upload's are: what the odor sheet needs)
        data["odor_check"] = await run_in_threadpool(_label_odor_matches, [it["label"] for it in data[c["axis"]]])
    body = await run_in_threadpool(lambda: json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
    if len(body) > 1_000_000 and "gzip" in request.headers.get("accept-encoding", ""):
        body = await run_in_threadpool(gzip.compress, body, 6)
        return Response(body, media_type="application/json", headers={"content-encoding": "gzip", "vary": "accept-encoding"})
    return Response(body, media_type="application/json")


async def reference_download(request: Request) -> Response:
    """A reference dataset to save (POST {"format": "files" | "wide", and of "wide", as shown:
    "rows", "cols" (indices in it as read, in order), "transpose"}), see
    `matrix_download.reference_download`."""
    ref_id = request.path_params["ref_id"]
    return await _matrix_download(request, lambda body, fmt, dest, picked: matrix_download.reference_download(
        db, ref_id, _odor_index(), fmt, dest, **picked))


# ----------------------------------------------------------------------- admin

RESTART_DEADLINE_S = 30
# A full restart (restart-olfdb, on the server) is asked of systemd there by putting this file in
# the data folder, which deploy/olfdb-restart.path watches for; what restart-olfdb prints then goes
# to RESTART_LOG, its last line starting with RESTART_FAILED if it did not restart.
RESTART_REQUEST = "restart-requested"
RESTART_LOG = "restart.log"
RESTART_FAILED = "restart-olfdb: not restarted"
RESTART_TAKEN_S = 5  # how long systemd is given to take a request, before it is taken back


def _supervisor() -> Optional[str]:
    """What starts the server again when it exits: Docker (given a restart policy, as in
    docker-compose.yml) or systemd (running it directly, as deploy/olfdb-venv.service
    does); None if neither."""
    if os.path.exists("/.dockerenv"):
        return "Docker"
    if os.environ.get("SYSTEMD_EXEC_PID") == str(os.getpid()):
        return "systemd"
    return None


def _admin_state() -> dict:
    saved = sandbox.load()
    active = sandbox.STATUS["settings"]
    by = _supervisor()
    to_server, note = filelog.server_paths()
    return {
        "saved": {**saved, "updated_at": _aware(saved.get("updated_at"))},
        "active": {k: active[k] for k in sandbox.KINDS} if active else None,
        "pending": active is not None and any(saved[k] != active[k] for k in sandbox.KINDS),
        "checks": sandbox.check_folders(saved, _supervisor())[0],
        "odor_sheet": _odor_sheet(),
        "lowercase_only": _lowercase_only(),
        "data_dir": to_server(str(config.DATA_DIR)),
        "deleted_dir": to_server(str(config.DELETED_DIR)),
        "in_container": note is not None,  # these paths, which are on the server otherwise
        "tensor_root": str(config.TENSOR_ROOT),
        "started_at": _aware(sandbox.STATUS["started_at"]),
        "kernel": sandbox.STATUS["kernel"],
        "restart": {"possible": by is not None, "by": by},
    }


def _folder_lists(body: Any) -> bool:
    return isinstance(body, dict) and all(isinstance(body.get(k) or [], list) for k in sandbox.KINDS)


async def admin_get(request: Request) -> Response:
    """The folders the server may read and write: as saved, as in effect since it started, and checked."""
    return JSONResponse(await run_in_threadpool(_admin_state))


async def admin_files(request: Request) -> Response:
    """The files the server has read or written in its own folders, but for those in the
    folders it may read and write, and reference data (see `filelog.listing`)."""
    def run() -> dict:
        settings = [sandbox.load(), sandbox.STATUS["settings"] or {}]
        listed = [d for s in settings for k in sandbox.KINDS for d in s.get(k, [])]
        listed += [str(config.REFERENCE_DIR), *(r["path"] for r in db.list_references())]
        out = filelog.listing(db, exclude=listed)
        return {**out, "since": _aware(out.get("since")), "files": [{**f, "first_at": _aware(f.get("first_at"))} for f in out["files"]]}

    return JSONResponse(await run_in_threadpool(run))


def _backup_status() -> dict:
    """`backup.status`, its times `_aware`."""
    s = backup.status(db)
    folders = [{**f, "backups": [{**b, "created_at": _aware(b.get("created_at"))} for b in f["backups"]]} for f in s["folders"]]
    last = s["last"] and {**s["last"], "created_at": _aware(s["last"].get("created_at"))}
    return {**s, "folders": folders, "last": last, "job": {**s["job"], "started_at": _aware(s["job"].get("started_at"))}}


async def admin_backups(request: Request) -> Response:
    """The backups, in the folders the server may write, and whether the last one is current (see `backup.status`)."""
    return JSONResponse(await run_in_threadpool(_backup_status))


async def admin_backup_start(request: Request) -> Response:
    """Start a backup into one of the folders the server may write ({"folder"}), made in
    the background (see `backup.make`); the backups then, as from GET."""
    body = await request.json()
    error = backup.start(db, str(body.get("folder") or "") if isinstance(body, dict) else "")
    if error:
        return json_error(409, error)
    log.info("backup into %s started from the Admin tab", body["folder"])
    return JSONResponse(await run_in_threadpool(_backup_status))


async def admin_check(request: Request) -> Response:
    """Check folders for the Admin tab ({"read_dirs", "write_dirs"}), without saving them."""
    body = await request.json()
    if not _folder_lists(body):
        return json_error(400, f"expected lists of paths in {', '.join(sandbox.KINDS)}")
    checks, _ = await run_in_threadpool(sandbox.check_folders, body, _supervisor())
    return JSONResponse({"ok": all(c["ok"] for cs in checks.values() for c in cs), "checks": checks})


async def admin_save(request: Request) -> Response:
    """Save the folders the server may read and write ({"read_dirs", "write_dirs"}), in
    effect from the next restart; nothing is saved if any of them is malformed."""
    body = await request.json()
    if not _folder_lists(body):
        return json_error(400, f"expected lists of paths in {', '.join(sandbox.KINDS)}")
    checks, dirs = await run_in_threadpool(sandbox.check_folders, body, _supervisor())
    errors = [f"{c['input']}: {c['error']}" for cs in checks.values() for c in cs if not c["ok"]]
    if errors:
        return JSONResponse({"ok": False, "errors": errors, "checks": checks}, status_code=422)
    await run_in_threadpool(sandbox.save, dirs)
    return JSONResponse({"ok": True, **await run_in_threadpool(_admin_state)})


async def admin_odor_sheet(request: Request) -> Response:
    """Set the odor sheet's link ({"url"}; empty to clear it), in use at once. The sheet is
    read right away (``read``: as in /api/odors/status), but saved even if it cannot be."""
    body = await request.json()
    url = str(body.get("url") or "").strip() if isinstance(body, dict) else ""
    if url:
        try:
            odormeta.export_url(url)
        except odormeta.SheetError as e:
            return JSONResponse({"ok": False, "errors": [str(e)]}, status_code=422)

    def run() -> dict:
        db.set_setting(SHEET_SETTING, url or None, when=dt.datetime.now().isoformat(timespec="seconds"))
        sheet = _odor_sheet()
        log.info("odor sheet set in the Admin tab: %s", url or "none")
        return {"ok": True, "odor_sheet": sheet, "read": sheet_watch.check(sheet["url"], force=True) if sheet["url"] else None}

    return JSONResponse(await run_in_threadpool(run))


async def admin_odor_case(request: Request) -> Response:
    """Whether new odor names may only be lowercase ({"lowercase_only": bool}), at once."""
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("lowercase_only"), bool):
        return json_error(400, 'expected {"lowercase_only": true or false}')

    def run() -> dict:
        db.set_setting(LOWERCASE_SETTING, "1" if body["lowercase_only"] else None, when=dt.datetime.now().isoformat(timespec="seconds"))
        log.info("new odor names lowercase only (Admin tab): %s", body["lowercase_only"])
        return {"ok": True, "lowercase_only": _lowercase_only()}

    return JSONResponse(await run_in_threadpool(run))


def _host() -> dict:
    """The server's name and IP addresses (``name``, ``source``, ``addresses``).

    Outside Docker: its own name (``source`` 'server'), and its address on the network
    along with those DNS gives that name. In a Docker container, which has a random name
    and addresses of its own: the server's name from its /etc/hostname, as
    docker-compose.yml mounts it in ('mounted'), and the addresses DNS gives that name;
    the container's name ('container') and no addresses if it is not mounted.
    """
    docker = os.path.exists("/.dockerenv")
    name, source = socket.gethostname(), "server"
    if docker:
        try:
            name, source = config.HOST_HOSTNAME_FILE.read_text().strip(), "mounted"
        except OSError:
            source = "container"
    addresses = []
    if source != "container":
        try:
            addresses += [a[4][0] for a in socket.getaddrinfo(name, None, type=socket.SOCK_STREAM)]
        except OSError:
            pass
    if not docker:
        try:  # the interface with a route to the network (nothing is sent)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))
                addresses.insert(0, s.getsockname()[0])
        except OSError:
            pass

    def on_network(a: str) -> bool:
        try:
            ip = ipaddress.ip_address(a.split("%")[0])
        except ValueError:
            return False
        return not (ip.is_loopback or ip.is_link_local or ip.is_unspecified)

    return {"name": name, "source": source, "addresses": [a for a in dict.fromkeys(addresses) if on_network(a)]}


def _panels_state() -> dict:
    """The Admin tab's panel names: ``renames`` (as saved: [{"from", "to"}]) and ``panels``
    (each panel of the datasets as named in their data: ``name``, ``datasets`` with it, and
    ``shown``: its name as shown)."""
    renames = db.panel_renames()
    used = db.data_panels()
    return {"renames": [{"from": k, "to": v} for k, v in renames.items()],
            "panels": [{"name": n, "datasets": len(used[n]), "shown": rename_panel(n, renames)}
                       for n in sorted(used, key=str.casefold)]}


def _panel_rename_errors(renames: dict[str, str]) -> list[str]:
    """Why panel renames can't be saved: a panel renamed to one that is itself renamed, or two
    panels of one dataset renamed to one name (they would become one panel)."""
    errors = [f"'{k}' is renamed to '{v}', itself renamed (to '{renames[v]}'): rename both to the same name directly"
              for k, v in renames.items() if v in renames]
    by_dataset: dict[int, list[str]] = {}
    for name, ids in db.data_panels().items():
        for i in ids:
            by_dataset.setdefault(i, []).append(name)
    titles = {d["id"]: d["title"] for d in db.list_datasets()}
    for i, names in sorted(by_dataset.items()):
        shown: dict[str, list[str]] = {}
        for n in sorted(names):
            shown.setdefault(rename_panel(n, renames), []).append(n)
        for to, froms in shown.items():
            if len(froms) > 1:
                errors.append(f"dataset #{i} ({titles.get(i)}) has panels {', '.join(repr(f) for f in froms)}, which would all be "
                              f"'{to}': one panel of it, two trials of an odor and repeat then the same")
    return errors


async def admin_panels(request: Request) -> Response:
    """The panels of the datasets, and those shown under other names (`_panels_state`)."""
    return JSONResponse(await run_in_threadpool(_panels_state))


async def admin_panels_save(request: Request) -> Response:
    """Save the panels to show under other names ({"renames": [{"from", "to"}, ...]}), at once,
    for every dataset, in every display and download; refused (with ``errors``) if any is not
    a name, is given twice, or `_panel_rename_errors`."""
    body = await request.json()
    rows = body.get("renames") if isinstance(body, dict) else None
    if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("from"), str) and isinstance(r.get("to"), str)
                                             for r in rows):
        return json_error(400, 'expected {"renames": [{"from": a panel\'s name, "to": the name to show}, ...]}')
    renames: dict[str, str] = {}
    errors = []
    for r in rows:
        a, b = " ".join(r["from"].split()), " ".join(r["to"].split())
        if not a and not b:
            continue
        if not a or not b:
            errors.append(f"a rename needs both names (not '{a or b}' alone)")
        elif a == b:
            errors.append(f"'{a}' is renamed to itself")
        elif a in renames:
            errors.append(f"'{a}' is renamed twice")
        else:
            renames[a] = b

    def run() -> dict:
        problems = errors or _panel_rename_errors(renames)
        if problems:
            return {"ok": False, "errors": problems, **_panels_state()}
        db.set_setting(PANEL_RENAMES, json.dumps(renames) if renames else None, when=dt.datetime.now().isoformat(timespec="seconds"))
        log.info("panels renamed (Admin tab): %s", ", ".join(f"{k} -> {v}" for k, v in renames.items()) or "none")
        return {"ok": True, **_panels_state()}

    return JSONResponse(await run_in_threadpool(run))


async def admin_chemutils(request: Request) -> Response:
    """The chemutils installed, what was looked up through it, and the last re-lookup (`_chemutils_state`)."""
    return JSONResponse(await run_in_threadpool(_chemutils_state))


async def admin_chemutils_relookup(request: Request) -> Response:
    """Look up again, in the background, every value of some properties looked up through
    chemutils ({"props": [...]}, of those kept with the odor table, `PROPERTY_COLUMNS`: see
    `_relookup`); then, as from GET."""
    body = await request.json()
    props = body.get("props") if isinstance(body, dict) else None
    if not isinstance(props, list) or not props or any(p not in PROPERTY_COLUMNS for p in props):
        return json_error(400, f"expected {{\"props\": [...]}}, of {', '.join(PROPERTY_COLUMNS)}")
    with _relookup_lock:
        if _relookup_job.get("running"):
            return json_error(409, "values are being looked up again already")
        _relookup_job.clear()
        _relookup_job.update(running=True, props=list(dict.fromkeys(props)), total=None, done=0, failed=0, error=None,
                             started_at=_aware(dt.datetime.now().isoformat(timespec="seconds")), finished_at=None)
    threading.Thread(target=_relookup, args=(_relookup_job["props"],), name="relookup", daemon=True).start()
    log.info("looking up %s again, from the Admin tab", ", ".join(props))
    return JSONResponse(await run_in_threadpool(_chemutils_state))


async def admin_host(request: Request) -> Response:
    """The server's name and IP addresses (see `_host`)."""
    return JSONResponse(await run_in_threadpool(_host))


def _delete_dataset(ds_id: int) -> tuple[int, dict]:
    """Delete a dataset from the database, its folder moved to data/deleted/<id>/ (from
    where it could be uploaded again) as part of it: (status, response)."""
    src = config.DATASETS_DIR / str(ds_id)
    dest = config.DELETED_DIR / str(ds_id)
    if dest.exists():  # put there by hand
        dest = dest.with_name(f"{ds_id}_{dt.datetime.now():%Y%m%d-%H%M%S}")
    moved = []

    def move() -> None:
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dest)
            moved.append(dest)

    link = db.permalink_of(ds_id)
    keys = None
    if link:  # (what its permalink keeps of it, as it is deleted)
        d, m = db.get_dataset(ds_id), db.get_matrix_arrays(ds_id)
        keys = permalinks.dataset_keys(d, m) if d and m else None
    try:
        with _readme_lock:  # not while its README is being overwritten
            if not db.delete_dataset(ds_id, before_commit=move, permalink_keys=keys,
                                     when=dt.datetime.now().isoformat(timespec="seconds")):
                return 404, {"error": "dataset not found"}
    except OSError as e:
        return 500, {"error": f"its folder could not be moved to {dest.parent}, so nothing was deleted ({e})"}
    log.info("dataset %d deleted in the Admin tab; its files: %s%s", ds_id, moved[0] if moved else "none",
             f"; its permalink {link['slug']} orphaned" if link else "")
    return 200, {"ok": True, "files": str(moved[0]) if moved else None, "orphaned": link and link["slug"]}


async def admin_permalinks(request: Request) -> Response:
    """Every permalink (see permalinks.py), the orphaned last."""
    rows = await run_in_threadpool(db.list_permalinks)
    return JSONResponse({"permalinks": [_public_permalink(p) for p in rows]})


async def admin_permalink_delete(request: Request) -> Response:
    """Delete an orphaned permalink (409 if it points to a dataset)."""
    slug = request.path_params["slug"]
    done = await run_in_threadpool(db.delete_permalink, slug)
    if done is None:
        return json_error(404, "no permalink of that name")
    if not done:
        return json_error(409, "it points to a dataset: rename it in Browse, or delete the dataset first")
    log.info("orphaned permalink %s deleted in the Admin tab", slug)
    return JSONResponse({"ok": True})


async def admin_reference_check(request: Request) -> Response:
    """Check a file to add as a reference dataset ({"path", "chemical_axis", "kind", "fields"}), see `reference.check`."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"path": ...}')
    axis = body.get("chemical_axis") if body.get("chemical_axis") in reference.AXES else "detect"
    kind = body.get("kind") if body.get("kind") in reference.KINDS else "detect"
    c = await run_in_threadpool(
        lambda: reference.check(str(body.get("path") or ""), axis, kind, _odor_index(), _supervisor(), body.get("fields")))
    labels = (c.get("preview") or {}).pop("odor_labels", None)
    if labels:  # (its odors checked against the odor table as an upload's are: what the odor sheet needs)
        c["odor_check"] = await run_in_threadpool(_label_odor_matches, labels)
    return JSONResponse(c)


async def admin_reference_add(request: Request) -> Response:
    """Add a reference dataset ({"path", "title", "description", "kind", "chemical_axis",
    "value_label", "row_label", "col_label", "source", "fields"}), see `reference.add`."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"path": ..., "title": ...}')
    status, result = await run_in_threadpool(lambda: reference.add(db, body, _odor_index(), _supervisor()))
    if status == 200:
        log.info("reference dataset %s added in the Admin tab: %s", result["id"], body.get("path"))
    return JSONResponse(result, status_code=status)


async def admin_reference_update(request: Request) -> Response:
    """Change a reference dataset's settings (any of those adding it takes but "path"), see `reference.update`."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"title": ..., "fields": ...}')
    ref_id = request.path_params["ref_id"]
    status, result = await run_in_threadpool(lambda: reference.update(db, ref_id, body, _odor_index(), _supervisor()))
    if status == 200:
        log.info("reference dataset %s changed in the Admin tab: %s", ref_id, json.dumps(body.get("fields")))
    return JSONResponse(result, status_code=status)


async def admin_reference_delete(request: Request) -> Response:
    """Delete a reference dataset added in the Admin tab (its import, not its file), see
    `reference.delete`; the last of a bundle imported there (MetaCyc, LOTUS), its folder moved
    to deleted/ (see `bundle_import.Importer.release`: then ``moved``, where to)."""
    ref_id = request.path_params["ref_id"]

    def run() -> tuple[int, dict]:
        refs = db.list_references()
        status, result = reference.delete(db, ref_id)
        ref = next((r for r in refs if r["id"] == ref_id), None)
        if status == 200 and ref:
            others = [r["path"] for r in refs if r["id"] != ref_id]
            moved = next((m for imp in IMPORTERS.values() if (m := imp.release(ref["path"], others))), None)
            result["moved"] = filelog.server_paths()[0](str(moved)) if moved else None
        return status, result

    status, result = await run_in_threadpool(run)
    if status == 200:
        log.info("reference dataset %s deleted in the Admin tab", ref_id)
    return JSONResponse(result, status_code=status)


async def admin_models_add(request: Request) -> Response:
    """Add a folder of model data on tensor ({"path", "hidden", "first", "last"}: the last three
    comma-separated parameters), see `model_data.add`."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"path": ...}')
    status, result = await run_in_threadpool(model_data.add, db, body, config.TENSOR_ROOT)
    if status == 200:
        log.info("model data %s added in the Admin tab: %s", result["id"], result["path"])
        result["created_at"] = _aware(result["created_at"])
    return JSONResponse(result, status_code=status)


async def admin_models_update(request: Request) -> Response:
    """Change a folder of model data's lists of parameters ({"hidden", "first", "last"})."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"hidden": ..., "first": ..., "last": ...}')
    status, result = await run_in_threadpool(model_data.update, db, request.path_params["id"], body)
    if status == 200:
        result["created_at"] = _aware(result["created_at"])
    return JSONResponse(result, status_code=status)


async def admin_models_delete(request: Request) -> Response:
    """Forget a folder of model data (nothing on tensor is touched)."""
    if not await run_in_threadpool(model_data.delete, db, request.path_params["id"]):
        return json_error(404, "no such folder of model data")
    log.info("model data %s deleted in the Admin tab", request.path_params["id"])
    return JSONResponse({"deleted": True})


async def admin_models_baseline_add(request: Request) -> Response:
    """Choose a model as a baseline ({"experiment": "<folder id>/<its folder>", "row": <its row>},
    as /api/models lists it), see `model_data.add_baseline`."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"experiment": ..., "row": ...}')
    status, result = await run_in_threadpool(model_data.add_baseline, db, body, config.TENSOR_ROOT)
    if status == 200:
        log.info("model %s of %s made a baseline in the Admin tab", result["name"], result["experiment"])
        result["created_at"], result["imported_at"] = _aware(result["created_at"]), _aware(result["imported_at"])
    return JSONResponse(result, status_code=status)


async def admin_models_baseline_import(request: Request) -> Response:
    """A baseline's files imported again from tensor, in place of its copy (`model_data.import_baseline`)."""
    status, result = await run_in_threadpool(model_data.import_baseline, db, request.path_params["id"], config.TENSOR_ROOT)
    if status == 200:
        log.info("baseline %s of %s imported again in the Admin tab", result["name"], result["experiment"])
        result["created_at"], result["imported_at"] = _aware(result["created_at"]), _aware(result["imported_at"])
    return JSONResponse(result, status_code=status)


async def admin_models_baseline_delete(request: Request) -> Response:
    """A model no longer a baseline (its imported files moved to the folder of deleted things)."""
    if not await run_in_threadpool(model_data.delete_baseline, db, request.path_params["id"]):
        return json_error(404, "no such baseline")
    return JSONResponse({"deleted": True})


# bundles of reference data imported in the Admin tab, by kind (see bundle_import.py)
IMPORTERS = {imp.key: imp for imp in (metacyc_import.importer, lotus_import.importer)}


def _bundle_imports(imp: bundle_import.Importer) -> dict:
    """A kind of bundle imported in the Admin tab: what an upload is to have (and how the
    browser finds its files), the import under way (or the last one), and those kept."""
    to_server, _ = filelog.server_paths()
    kept = imp.imports(db.list_references())
    return {"kind": imp.key, "name": imp.name, "what": imp.what, "like": imp.like, "script": f"{imp.module.replace('.', '/')}.py",
            "how": imp.how, "gzip": imp.gzip, "expected": imp.expected(), "archives": bundle_import.ARCHIVES,
            "max_upload_mb": config.MAX_UPLOAD_MB, "job": _import_job(imp), "dir": to_server(str(imp.dir)),
            "deleted_dir": to_server(str(config.DELETED_DIR / imp.key)),
            "imports": [{**i, "dir": to_server(i["dir"]), "uploaded_at": _aware(i.get("uploaded_at")),
                         "built_at": _aware(i.get("built_at"))} for i in kept]}


def _import_job(imp: bundle_import.Importer) -> dict:
    """`Importer.job`, its times `_aware`."""
    j = imp.job()
    return {**j, "started_at": _aware(j["started_at"]), "finished_at": _aware(j["finished_at"])}


async def _admin_import(request: Request, imp: bundle_import.Importer) -> Response:
    """Import a bundle: its files uploaded (multipart 'file', repeatable: the files of the
    folder its build script reads, or an archive of it; 'mtimes', if given: a JSON object of
    when each was last changed, by its name as uploaded, in ms since 1970, kept as theirs; and
    its options, as MetaCyc's 'version'), kept with the bundle built of them, whose reference
    datasets are added (in the background: the job, as GET /api/admin/<kind> gives it). 422
    with what is missing or not as expected, 409 while another of its kind is under way (see
    bundle_import.py)."""
    try:
        d = imp.claim()
    except bundle_import.UploadError as e:
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=409)
    max_bytes = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    try:
        form = await request.form()
        uploads = [up for up in form.getlist("file") if isinstance(up, UploadFile)]
        if not uploads:
            raise bundle_import.UploadError(f"no file in the request (field 'file'): upload the folder of {imp.what}, "
                                            "or an archive of it")
        opts = imp.options(form)
        try:
            mtimes = json.loads(str(form.get("mtimes") or "{}"))
        except ValueError:
            mtimes = {}
        mtimes = ({k: v / 1000 for k, v in mtimes.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0}
                  if isinstance(mtimes, dict) else {})
        archives = [up for up in uploads if bundle_import.is_archive(up.filename)]
        files = [up for up in uploads if not bundle_import.is_archive(up.filename)]
        for k, rel in imp.pick([up.filename or "" for up in files]).items():
            await run_in_threadpool(imp.save, files[k].file, d / "raw", rel, max_bytes, files[k].filename, mtimes.get(files[k].filename))
        for i, up in enumerate(archives):
            dest = d / f"upload-{i}{''.join(Path(up.filename).suffixes[-2:])}"
            await _save_upload(up, dest, max_bytes=max_bytes)
            try:
                await run_in_threadpool(imp.extract, dest, up.filename, d / "raw", max_bytes)
            finally:
                dest.unlink(missing_ok=True)
        found = await run_in_threadpool(imp.check, d / "raw", len(uploads))
        opts = await run_in_threadpool(imp.settle, d / "raw", opts)
    except bundle_import.UploadError as e:
        imp.abandon(d)
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=422)
    except HTTPException as e:
        imp.abandon(d)
        return JSONResponse({"ok": False, "errors": [e.detail]}, status_code=e.status_code)
    except BaseException:
        imp.abandon(d)
        raise
    imp.start(db, d, opts, found, _odor_index, _supervisor)
    job = _import_job(imp)
    log.info("%s %s uploaded in the Admin tab (%d files), being imported into %s", imp.name, job["label"] or "", len(uploads), d)
    return JSONResponse({"ok": True, "job": job, "files": found})


async def _admin_tensor_check(request: Request, imp: bundle_import.Importer) -> Response:
    """Check a path on tensor ({"path"}, from its top) to import a bundle from: the folder its
    build reads (with the files of it read) or an archive of it, see `Importer.tensor_source`."""
    body = await request.json()
    path = body.get("path") if isinstance(body, dict) else None
    if not isinstance(path, str):
        return json_error(400, 'expected {"path": ...}')
    return JSONResponse(await run_in_threadpool(imp.tensor_source, path, config.TENSOR_ROOT))


async def _admin_import_tensor(request: Request, imp: bundle_import.Importer) -> Response:
    """Import a bundle from a path on tensor ({"path", and its options, as MetaCyc's "version"}):
    the folder its build reads, or an archive of it, checked (422 with why not), then its files
    copied (or taken out of the archive) and checked as an upload's, and the bundle built, in
    the background (the job, as GET /api/admin/<kind> gives it); 409 while another is under way."""
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("path"), str):
        return json_error(400, 'expected {"path": ...}')
    src = await run_in_threadpool(imp.tensor_source, body["path"], config.TENSOR_ROOT)
    if not src["ok"]:
        return JSONResponse({"ok": False, "errors": [f"{src['path'] or src['input']}: {src['error']}"]}, status_code=422)
    try:
        d = imp.claim()
    except bundle_import.UploadError as e:
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=409)
    imp.start_from_tensor(db, d, imp.options(body), src, _odor_index, _supervisor)
    job = _import_job(imp)
    log.info("%s from %s on tensor (%s) being imported in the Admin tab into %s", imp.name, src["path"], src["kind"], d)
    return JSONResponse({"ok": True, "job": job, "source": src})


async def admin_metacyc(request: Request) -> Response:
    """MetaCyc imported in the Admin tab (see `_bundle_imports`)."""
    return JSONResponse(await run_in_threadpool(_bundle_imports, metacyc_import.importer))


async def admin_metacyc_import(request: Request) -> Response:
    """Import MetaCyc: its SmartTables uploaded (the files of the folder app/metacyc_build.py
    reads, or an archive of it; 'version', MetaCyc's, else told from them: refused if another
    release's), see `_admin_import`."""
    return await _admin_import(request, metacyc_import.importer)


async def admin_metacyc_tensor_check(request: Request) -> Response:
    return await _admin_tensor_check(request, metacyc_import.importer)


async def admin_metacyc_tensor(request: Request) -> Response:
    """Import MetaCyc from a path on tensor, see `_admin_import_tensor`."""
    return await _admin_import_tensor(request, metacyc_import.importer)


async def admin_lotus(request: Request) -> Response:
    """LOTUS imported in the Admin tab (see `_bundle_imports`)."""
    return JSONResponse(await run_in_threadpool(_bundle_imports, lotus_import.importer))


async def admin_lotus_import(request: Request) -> Response:
    """Import LOTUS: its files uploaded (those of the folder app/lotus_build.py reads, as
    Scraped-Databases-Copy/LOTUS/raw, or an archive of it), see `_admin_import`."""
    return await _admin_import(request, lotus_import.importer)


async def admin_lotus_tensor_check(request: Request) -> Response:
    return await _admin_tensor_check(request, lotus_import.importer)


async def admin_lotus_tensor(request: Request) -> Response:
    """Import LOTUS from a path on tensor, see `_admin_import_tensor`."""
    return await _admin_import_tensor(request, lotus_import.importer)


async def admin_dataset_delete(request: Request) -> Response:
    """Delete a dataset (see `_delete_dataset`)."""
    status, result = await run_in_threadpool(_delete_dataset, int(request.path_params["ds_id"]))
    return JSONResponse(result, status_code=status)


def _restart() -> None:
    """Shut down as for `docker stop` (uvicorn finishes the requests under way; at once if
    that takes over `RESTART_DEADLINE_S`), for Docker or systemd to start the server again."""
    deadline = threading.Timer(RESTART_DEADLINE_S, os._exit, args=(1,))
    deadline.daemon = True
    deadline.start()
    os.kill(os.getpid(), signal.SIGTERM)


def _ends_at(started_at: Optional[str], done, total) -> Optional[str]:
    """When a job started at `started_at` should end, having done `done` of `total`, at the
    rate it has gone so far; None if not known."""
    if not (started_at and done and total):
        return None
    t0 = dt.datetime.fromisoformat(_aware(started_at)).timestamp()
    end = t0 + (time.time() - t0) * total / done
    return dt.datetime.fromtimestamp(end).astimezone().isoformat(timespec="seconds")


def _running_jobs(full: bool) -> dict:
    """What a restart would stop: the web server's jobs under way (a backup, an import of
    reference data, a re-lookup of odor properties) and, if `full` (the fit worker made again
    too), the fits and predictions running.

    Returns:
        dict with ``jobs`` (each its ``what``, ``started_at``, ``ends_at``: when it should end,
        None if not known, and ``then``: what becomes of it) and ``queued``: how many fits and
        predictions are waiting (a restart leaves them to start once it is over).
    """
    jobs = []
    b = dict(backup.JOB)
    if b["running"]:
        jobs.append({"what": f"A backup to {b['folder']}", "started_at": _aware(b["started_at"]),
                     "ends_at": _ends_at(b["started_at"], b["done"], b["total"]),
                     "then": "It would be left incomplete, to be made again."})
    for imp in IMPORTERS.values():
        j = imp.job()
        if j["running"]:
            jobs.append({"what": f"An import of {imp.name}", "started_at": _aware(j["started_at"]), "ends_at": None,
                         "then": "It would be left unfinished, to be imported again."})
    r = dict(_relookup_job)
    if r.get("running"):
        jobs.append({"what": f"Looking up {', '.join(r['props'])} again", "started_at": r["started_at"],
                     "ends_at": _ends_at(r["started_at"], r["done"], r["total"]),
                     "then": "It would stop part way (what it has looked up is kept), to be run again."})
    if not full:
        return {"jobs": jobs, "queued": 0}
    worker = predictions.worker_jobs(db)
    return {"jobs": jobs + worker["jobs"], "queued": worker["queued"]}


async def _ask_full_restart() -> Optional[str]:
    """Asks systemd on the server for a full restart (restart-olfdb, run by
    deploy/olfdb-restart.service), by putting `RESTART_REQUEST` in the data folder, the one
    folder there the server may write; one not taken within `RESTART_TAKEN_S` is taken back.
    Returns why it was not taken, or None."""
    req = config.DATA_DIR / RESTART_REQUEST
    with contextlib.suppress(OSError):
        (config.DATA_DIR / RESTART_LOG).unlink()  # the last one's
    req.write_text(f"asked from the Admin tab at {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n")
    for _ in range(max(1, round(RESTART_TAKEN_S * 10))):
        await asyncio.sleep(0.1)
        if not req.exists():
            return None
    try:
        req.unlink()
    except FileNotFoundError:  # taken just now
        return None
    return ("nothing on the server took the request: olfdb-restart.path, which watches for it, is not installed or not "
            "running. Running restart-olfdb on the server installs it (DEPLOY.md, section 4).")


async def admin_restart(request: Request) -> Response:
    """Restart the server: the web server alone (``{}``), if Docker or systemd starts it
    again after it exits; or everything, as restart-olfdb does on the server (``{"full":
    true}``, asked of systemd there: `_ask_full_restart`). While jobs it would stop are under
    way (`_running_jobs`), refused (409, with their ``jobs`` and ``queued``) unless
    ``"stop_jobs": true``."""
    try:
        body = await request.json()
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    full = bool(body.get("full"))
    supervisor = _supervisor()
    if supervisor is None and not full:
        return json_error(409, "the server cannot restart itself here: it is not run by Docker or systemd, which would "
                               "start it again (DEPLOY.md). Restart it by hand.")
    if not body.get("stop_jobs"):
        running = await run_in_threadpool(_running_jobs, full)
        if running["jobs"]:
            return JSONResponse({"error": "a restart would stop jobs under way", **running}, status_code=409)
    started_at = _aware(sandbox.STATUS["started_at"])
    if full:
        error = await _ask_full_restart()
        if error:
            return json_error(409, error)
        log.info("full restart (restart-olfdb) requested from the Admin tab")
        return JSONResponse({"ok": True, "full": True, "by": "restart-olfdb", "started_at": started_at})
    log.info("restart requested from the Admin tab")
    return JSONResponse({"ok": True, "by": supervisor, "started_at": started_at}, background=BackgroundTask(_restart))


async def admin_restart_log(request: Request) -> Response:
    """What restart-olfdb has printed of the full restart asked for last (`RESTART_LOG`): its
    last ``lines`` (30 at most), and ``failed``: whether it ended without restarting, the site
    left as it was."""
    try:
        text = (config.DATA_DIR / RESTART_LOG).read_text(errors="replace")
    except FileNotFoundError:
        text = ""
    lines = text.rstrip().splitlines()[-30:]
    return JSONResponse({"lines": lines, "failed": bool(lines) and lines[-1].startswith(RESTART_FAILED)})


async def admin_jobs(request: Request) -> Response:
    """What a full restart would stop (`_running_jobs`), for restart-olfdb to ask about."""
    return JSONResponse(await run_in_threadpool(_running_jobs, True))


async def http_exception(request: Request, exc: HTTPException) -> Response:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


routes = [
    Route("/", index),
    Route("/api/health", health),
    Route("/api/options", options),
    Route("/api/metadata/check", metadata_check, methods=["POST"]),
    Route("/api/user", user_get),
    Route("/api/user", user_save, methods=["POST"]),
    Route("/api/stage", stage_create, methods=["POST"]),
    Route("/api/stage/{stage_id}", stage_get),
    Route("/api/stage/{stage_id}/odors", stage_odors),
    Route("/api/stage/{stage_id}/odor_yaml", stage_add_yaml, methods=["POST"]),
    Route("/api/stage/{stage_id}/odor_yaml/{name}", stage_remove_yaml, methods=["DELETE"]),
    Route("/api/stage/{stage_id}/experiment_xml", stage_add_xml, methods=["POST"]),
    Route("/api/stage/{stage_id}/experiment_xml", stage_remove_xml, methods=["DELETE"]),
    Route("/api/stage/{stage_id}/lookup", stage_lookup, methods=["POST"]),
    Route("/api/stage/{stage_id}/readme", stage_readme, methods=["POST"]),
    Route("/api/stage/{stage_id}/commit", stage_commit, methods=["POST"]),
    Route("/api/datasets", datasets_list),
    Route("/api/datasets/{ds_id:int}", dataset_get),
    Route("/api/datasets/{ds_id:int}/matrix", dataset_matrix),
    Route("/api/datasets/{ds_id:int}/readme", dataset_readme_save, methods=["PUT"]),
    Route("/api/datasets/{ds_id:int}/metadata", dataset_metadata_save, methods=["PUT"]),
    Route("/api/datasets/{ds_id:int}/permalink", dataset_permalink_save, methods=["PUT"]),
    Route("/api/permalinks/{slug}", permalink_get),
    Route("/d/{slug}", permalink_follow),
    Route("/api/datasets/{ds_id:int}/files/{path:path}", dataset_file),
    Route("/api/datasets/{ds_id:int}/odors", dataset_odor_table),
    Route("/api/datasets/{ds_id:int}/odors.{fmt}", dataset_odors_file),
    Route("/api/datasets/{ds_id:int}/clusterings/lookup", dataset_clusterings, methods=["POST"]),
    Route("/api/datasets/{ds_id:int}/clusterings", dataset_clusterings_save, methods=["POST"]),
    Route("/api/compare/datasets/{ds_id:int}", compare_dataset),
    Route("/api/compare/ecfp", compare_ecfp),
    Route("/api/datasets/{ds_id:int}/data_paths", dataset_data_paths),
    Route("/api/datasets/{ds_id:int}/data_paths", dataset_data_paths_save, methods=["PUT"]),
    Route("/api/tensor/check", tensor_check, methods=["POST"]),
    Route("/api/tensor/list", tensor_list),
    Route("/api/datasets/{ds_id:int}/movies", dataset_movies),
    Route("/api/movie", movie_info),
    Route("/api/movie/planes", movie_planes),
    Route("/api/movie/rois", movie_rois),
    Route("/api/movie/trials", movie_trials),
    Route("/api/download", download),
    Route("/api/roi_training", roi_training_flies),
    Route("/api/roi_training/download", roi_training_download),
    Route("/api/fits", fits_list),
    Route("/api/fits", fits_create, methods=["POST"]),
    Route("/api/fits/{id:int}", fits_delete, methods=["DELETE"]),
    Route("/api/fits/{id:int}/cancel", fits_cancel, methods=["POST"]),
    Route("/api/fits/{id:int}/log", fits_log),
    Route("/api/fits/{id:int}/download", fits_download),
    Route("/api/predictions", predictions_list),
    Route("/api/predictions", predictions_create, methods=["POST"]),
    Route("/api/predictions/{id:int}", predictions_delete, methods=["DELETE"]),
    Route("/api/predictions/{id:int}/log", predictions_log),
    Route("/api/predictions/{id:int}/view", predictions_view),
    Route("/api/predictions/{id:int}/download", predictions_download),
    Route("/api/models", models_list),
    Route("/api/models/baselines", models_baselines),
    Route("/api/models/run", models_run),
    Route("/api/models/run/download", models_run_download, methods=["POST"]),
    Route("/api/merge", merge_units),
    Route("/api/merge/preview", merge_preview, methods=["POST"]),
    Route("/api/merge/download", merge_download),
    Route("/api/merge/scales", merge_scales),
    Route("/api/odors", odors_list),
    Route("/api/odors.{fmt}", odors_file),
    Route("/api/odors/status", odors_status),
    Route("/api/odors/preview", odors_preview, methods=["POST"]),
    Route("/api/odors/apply", odors_apply, methods=["POST"]),
    Route("/api/odors/structure/{cid:int}", odor_structure),
    Route("/api/odors/template.csv", odor_template),
    Route("/api/headspace", headspace_calc, methods=["POST"]),
    Route("/api/headspace.{fmt}", headspace_file),
    Route("/api/headspace/panels", headspace_panels),
    Route("/api/headspace/dilutions", headspace_dilutions, methods=["POST"]),
    Route("/api/headspace/vcf", headspace_vcf, methods=["GET", "POST"]),
    Route("/api/reference", reference_list),
    Route("/api/reference/{ref_id}", reference_get),
    Route("/api/reference/{ref_id}/download", reference_download, methods=["POST"]),
    Route("/api/reference/{ref_id}/metacyc/pathway/{frame:path}", metacyc_pathway),
    Route("/api/reference/{ref_id}/metacyc/compound/{frame:path}", metacyc_compound),
    Route("/api/reference/{ref_id}/metacyc/structure/{frame:path}", metacyc_structure),
    Route("/api/reference/{ref_id}/metacyc/lca", metacyc_lca),
    Route("/api/reference/{ref_id}/metacyc/search", metacyc_search),
    Route("/api/reference/{ref_id}/lotus/cell", lotus_cell),
    Route("/api/admin", admin_get),
    Route("/api/admin", admin_save, methods=["POST"]),
    Route("/api/admin/check", admin_check, methods=["POST"]),
    Route("/api/admin/files", admin_files),
    Route("/api/admin/backups", admin_backups),
    Route("/api/admin/backups", admin_backup_start, methods=["POST"]),
    Route("/api/admin/odor_sheet", admin_odor_sheet, methods=["POST"]),
    Route("/api/admin/odor_case", admin_odor_case, methods=["POST"]),
    Route("/api/admin/host", admin_host),
    Route("/api/admin/chemutils", admin_chemutils),
    Route("/api/admin/panels", admin_panels),
    Route("/api/admin/panels", admin_panels_save, methods=["POST"]),
    Route("/api/admin/chemutils/relookup", admin_chemutils_relookup, methods=["POST"]),
    Route("/api/admin/restart", admin_restart, methods=["POST"]),
    Route("/api/admin/restart", admin_restart_log),
    Route("/api/admin/jobs", admin_jobs),
    Route("/api/admin/datasets/{ds_id:int}", admin_dataset_delete, methods=["DELETE"]),
    Route("/api/admin/permalinks", admin_permalinks),
    Route("/api/admin/permalinks/{slug}", admin_permalink_delete, methods=["DELETE"]),
    Route("/api/admin/references/check", admin_reference_check, methods=["POST"]),
    Route("/api/admin/references", admin_reference_add, methods=["POST"]),
    Route("/api/admin/references/{ref_id}", admin_reference_delete, methods=["DELETE"]),
    Route("/api/admin/references/{ref_id}", admin_reference_update, methods=["PATCH"]),
    Route("/api/admin/models", admin_models_add, methods=["POST"]),
    Route("/api/admin/models/baselines", admin_models_baseline_add, methods=["POST"]),
    Route("/api/admin/models/baselines/{id:int}", admin_models_baseline_delete, methods=["DELETE"]),
    Route("/api/admin/models/baselines/{id:int}/import", admin_models_baseline_import, methods=["POST"]),
    Route("/api/admin/models/{id:int}", admin_models_update, methods=["PATCH"]),
    Route("/api/admin/models/{id:int}", admin_models_delete, methods=["DELETE"]),
    Route("/api/admin/metacyc", admin_metacyc),
    Route("/api/admin/metacyc", admin_metacyc_import, methods=["POST"]),
    Route("/api/admin/lotus", admin_lotus),
    Route("/api/admin/lotus", admin_lotus_import, methods=["POST"]),
    Route("/api/admin/metacyc/tensor/check", admin_metacyc_tensor_check, methods=["POST"]),
    Route("/api/admin/metacyc/tensor", admin_metacyc_tensor, methods=["POST"]),
    Route("/api/admin/lotus/tensor/check", admin_lotus_tensor_check, methods=["POST"]),
    Route("/api/admin/lotus/tensor", admin_lotus_tensor, methods=["POST"]),
    Mount("/static", RevalidatedStaticFiles(directory=str(config.STATIC_DIR)), name="static"),
]



async def _save_file_log() -> None:
    while True:
        await asyncio.sleep(filelog.SAVE_S)
        try:
            await run_in_threadpool(filelog.flush, db)
        except Exception:
            log.exception("could not save the files read and written (Admin tab)")


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    filelog.start(db)  # first, to see what it reads as it starts
    # before the first request, so before any worker thread starts (see sandbox.start)
    sandbox.start()
    _look_up_odors()  # (those added, or that could not be looked up, while it was not running)
    saving = asyncio.create_task(_save_file_log())
    yield
    saving.cancel()
    filelog.flush(db)


app = Starlette(routes=routes, exception_handlers={HTTPException: http_exception}, lifespan=lifespan)
