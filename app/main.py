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
import os
import re
import secrets
import shutil
import signal
import socket
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import markdown
import numpy as np
from pydantic import ValidationError as PydanticValidationError
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import (backup, chemprops, compare, config, export, filelog, headspace, merge, movies, odormeta, reference, sandbox, tensor,
               thorimage, vcf, yamlcheck)
from .db import PANEL_RENAMES, PROPERTY_COLUMNS, Database, rename_panel
from .glomeruli import load_glomeruli
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
    generate_readme,
    same_manipulations,
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
                         "started": _aware(sandbox.STATUS["started_at"])})


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
            "uploaders": existing["uploaders"],
            "genotypes": existing["genotypes"],
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


def _same_name(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two names (of uploaders) are one, whatever their case and spacing."""
    return " ".join((a or "").split()).casefold() == " ".join((b or "").split()).casefold()


def _same_unit(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two coding units are one: KCs however written, else whatever their case and spacing."""
    return (is_kc(a) and is_kc(b)) or _same_name(a, b)


def _duplicates(parsed: ParsedData, files: list[dict], uploader: Optional[str] = None,
                coding_unit: Optional[str] = None, manipulations: Optional[list[dict]] = None) -> tuple[list[str], list[str]]:
    """An upload (its data, and its files: [{name, sha256}]) against the datasets stored:
    errors (refusing it) if any of its files is one of theirs (byte for byte), or its data
    is (the same flies, ROIs, trials and values), whoever uploaded it. With its uploader,
    coding unit and manipulations (once entered), errors too if its flies (date, fly_num,
    and their ROIs' other levels: `ingest.fly_key`) are all and only those of a dataset of
    theirs of that coding unit and those manipulations, and warnings if it shares only
    some with one, saying which dataset may need to be deleted. Datasets of other
    uploaders, coding units or manipulations may have the same flies: one person's fly 1
    of a day is not another's, a fly may be recorded in two places (as may each side of
    its brain, apart), and a dataset of the same flies otherwise treated (e.g. after a
    drug: `metadata.Metadata.manipulations`) is another."""
    flies = {fly_key(u["date"], u["fly_num"], u.get("extra")) for u in parsed.units if u["date"] is not None}
    stored = db.dataset_flies()
    errors, warnings = [], []
    for d in sorted(db.list_datasets(), key=lambda d: d["id"]):
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
            errors.append(f"its flies ({_flies_text(flies)}) are exactly those of {which}, also of {d['coding_unit']}: "
                          f"one person's datasets of a coding unit cannot have the same flies, but for other manipulations "
                          f"(e.g. after a drug: enter them under Manipulations). {delete}")
        elif mine and flies & theirs:
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
    "uploader_name": "your name", "indicator": "calcium indicator", "driver": "driver",
    "coding_unit": "coding unit", "age_min_days": "fly age (min)", "age_max_days": "fly age (max)",
    "power_mw": "power (mW)", "power_tolerance_mw": "power tolerance", "wavelength_nm": "wavelength",
    "panel_names": "panel name(s)", "recording_system": "recording system", "source_code_url": "analysis code link",
    "raw_data_path": "raw data path", "mocorr_data_path": "motion-corrected data path", "genotype": "genotype",
    "notes": "notes", "power_measured_same_day": "power measured on the day", "manipulations": "manipulations",
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


def _parse_metadata(body: Any) -> tuple[Optional[Metadata], list[str]]:
    try:
        return Metadata.model_validate(body or {}), []
    except PydanticValidationError as e:
        return None, [x["message"] for x in _metadata_errors(e)]


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
    kc = _kc_problem(parsed, m.coding_unit)
    if kc:
        return JSONResponse({"ok": False, "errors": [kc]}, status_code=422)
    errors, warnings = await run_in_threadpool(_duplicates, parsed, _stage_files(meta), m.uploader_name, m.coding_unit,
                                               m.manipulations)
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
    return JSONResponse({"ok": True, "readme": readme, "title": dataset_title(m, summary), "warnings": warnings})


def _kc_problem(parsed: ParsedData, coding_unit: str) -> Optional[str]:
    """Why an upload of KCs (its coding unit) with ROIs numbered is refused, if it is: each
    fly's ids not every one from 0 or 1 to the highest (`ingest.kc_id_problems`; as for a
    ``cells`` level whatever the coding unit, when the file was read)."""
    if not is_kc(coding_unit) or not parsed.units or any(u["roi_kind"] != "int" for u in parsed.units):
        return None
    return kc_id_problems(parsed.units)


def _commit(d: Path, meta: dict, m: Metadata, readme: str, readme_edited: bool) -> dict:
    parsed = load_parsed(d / "parsed")
    summary = parsed.summary()
    # consistency checks that must pass before anything is written
    problems = [p for p in [_kc_problem(parsed, m.coding_unit)] if p]
    files = _stage_files(meta)
    problems += _duplicates(parsed, files, m.uploader_name, m.coding_unit, m.manipulations)[0]
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
        return {"ok": False, "errors": problems}

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
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "dataset_id": ds_id, "title": title}


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
    result = await run_in_threadpool(_commit, d, meta, m, readme, bool(body.get("readme_edited")))
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
    return JSONResponse(d)


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
    their movies (by kind: ``movies`` as from `movies.find`, ``error``: why its folder was
    not looked in, ``found``: how many were found in its flies' folders, by their paths
    and of no fly, ``skipped``: the flies, not the dataset's, whose folders in it were
    left out (see `movies.find`), and when both folders were found, ``compared`` and
    ``unmatched``: the recordings with movies in that one only, see `movies.unmatched`),
    and then its recordings in both side by side (`movies.side_by_side`; else None)."""
    flies = d["summary"].get("flies") or []
    bases = [Path(p) for p in sandbox.folders()]
    found, out = {}, {}
    for kind in ("raw", "mocorr"):
        f = found[kind] = tensor.find_data(d["metadata"].get(f"{kind}_data_path"), config.TENSOR_ROOT, bases)
        folder = Path(f["path"]) if f["status"] == "ok" and f["is_dir"] else None
        skipped: set = set()
        found_movies = movies.find(folder, kind, flies, skipped) if folder else []
        out[kind] = {"movies": found_movies, "error": f["error"], "compared": False, "unmatched": [],
                     # the flies (not the dataset's) whose folders in it were left out
                     "skipped": [{"date": date, "fly_num": n} for date, n in sorted(skipped)],
                     # how its movies were found: in its flies' folders, by their paths, of no fly
                     "found": {b or "none": sum(1 for m in found_movies if m["by"] == b) for b in ("fly folder", "path", None)}}
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


# ---------------------------------------------------------------------- movies


def _server_file(path: Any, suffixes: tuple[str, ...]) -> Path:
    """A file on the server named by the browser (a movie, or its ROIs): it must be on tensor
    or in a folder set in the Admin tab, and one the server may read (raises HTTPException)."""
    p = str(path or "")
    if not os.path.isabs(p) or not p.lower().endswith(suffixes):
        raise HTTPException(400, f"expected the path on the server of a {' / '.join(suffixes)} file")
    real = os.path.realpath(p)
    tops = [os.path.realpath(t).rstrip("/") for t in [config.TENSOR_ROOT, *sandbox.folders()]]
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
    """The ImageJ ROIs of a RoiSet.zip (or .roi) on the server (``?path=``): see `movies.read_rois`."""
    path = _server_file(request.query_params.get("path"), (".zip", ".roi"))
    try:
        rois = await run_in_threadpool(movies.read_rois, path)
    except movies.MovieError as e:
        return json_error(422, str(e))
    return JSONResponse({"rois": rois})


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
            "power_tolerance": power_tolerance}


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
    a list of paths relative to tensor."""
    body = await request.json()
    if not isinstance(body, dict) or any(not isinstance(body.get(k) or [], list) for k in tensor.USER_DIR_KINDS):
        return json_error(400, f"expected lists of paths in {', '.join(tensor.USER_DIR_KINDS)}")
    name = _user_name(body.get("name"))

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
                        saved: Optional[Callable[[], None]] = None) -> dict[tuple[int, str], str]:
    """Look up properties of odors through chemutils (`chemprops.look_up`, with `refresh`),
    each saved as it comes (then `saved` called), with the chemutils that looked it up;
    those that could not be looked up kept in `_props_failed`, and returned."""
    def save(cid: int, prop: str, found: Optional[dict]) -> None:
        db.save_property(cid, prop, found, when=dt.datetime.now().isoformat(timespec="seconds"), chemutils=chemprops.installed())
        if saved:
            saved()

    now = time.time()
    failed = chemprops.look_up(todo, save, refresh=refresh)
    _props_failed.update({k: (now, e) for k, e in failed.items()})
    if failed:
        log.warning("could not look up %d properties of odors (%s), e.g.: %s", len(failed),
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


def _headspace(solvent: str, comps: list[dict], model: Optional[str] = None) -> dict:
    """The headspace of odors diluted together (`headspace.calculate`), with the properties
    it needs of each looked up through chemutils first, where they have not been (but those
    that could not be in the last `HEADSPACE_RETRY_S`).

    Args:
        comps: ``odor_id`` (in the odor table) and ``log10_conc`` of each.
        model: of `headspace.MODELS`, for paraffin oil (the default if None).

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
    result = headspace.calculate(inputs, solvent, model or headspace.MODELS[0])
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


async def _checked_headspace(body) -> tuple[str, Optional[str], list[dict]] | Response:
    """A Headspace calculation asked for (as `headspace_calc` takes it): its solvent, model and
    components, else the error to answer with."""
    if not isinstance(body, dict) or body.get("solvent") not in headspace.SOLVENTS or not isinstance(body.get("components"), list):
        return json_error(400, f'expected {{"solvent": one of {", ".join(headspace.SOLVENTS)}, "components": [{{"odor_id", "log10_conc"}}, ...]}}')
    if body.get("model") is not None and body["model"] not in headspace.MODELS:
        return json_error(400, f"the model is one of {', '.join(headspace.MODELS)}")
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
    return body["solvent"], body.get("model"), comps


async def headspace_calc(request: Request) -> Response:
    """The Headspace calculator's: {"solvent" (of `headspace.SOLVENTS`), "model" (of
    `headspace.MODELS`, for paraffin oil; optional), "components": [{"odor_id", "log10_conc"},
    ...]} -> `_headspace`."""
    checked = await _checked_headspace(await request.json())
    if isinstance(checked, Response):
        return checked
    solvent, model, comps = checked
    try:
        return JSONResponse(await run_in_threadpool(_headspace, solvent, comps, model))
    except ValueError as e:
        return json_error(400, str(e))


async def headspace_file(request: Request) -> Response:
    """A Headspace calculation's table as a file to save, a CSV or a LaTeX table
    (`export.headspace_download`): ``solvent``, ``model`` (optional), ``odors``
    (``<odor id>:<log10 conc>,...``), ``units`` (of `headspace.UNITS`, for LaTeX) and
    ``lower=1`` for abbreviations in lowercase."""
    q = request.query_params
    try:
        comps = [{"odor_id": int(i), "log10_conc": float(c)} for i, c in (x.split(":") for x in q.get("odors", "").split(",") if x)]
    except ValueError:
        return json_error(400, "expected odors=<odor id>:<log10 concentration>,...")
    checked = await _checked_headspace({"solvent": q.get("solvent"), "model": q.get("model") or None, "components": comps})
    if isinstance(checked, Response):
        return checked
    solvent, model, comps = checked

    def run() -> tuple[str, str, str]:
        result = _headspace(solvent, comps, model)
        return export.headspace_download(result, {o["id"]: o for o in db.list_odors()}, request.path_params["fmt"],
                                         q.get("units") or "ppm", lower=_lower(request))

    try:
        text, name, media_type = await run_in_threadpool(run)
    except (export.ExportError, ValueError) as e:
        return json_error(400, str(e))
    return Response(text, media_type=media_type, headers={"content-disposition": f'attachment; filename="{name}"'})



VCF_MAX_OIL = 500  # chemicals with a headspace over their samples, at most, to calculate in paraffin oil
VCF_LOOKUP_BATCH = 10  # CIDs looked up through chemutils at once for VCF's chemicals (`_vcf_look_up`)
# CID -> (CID, InChI, properties) of VCF's chemicals to look up in the background, for the
# paraffin oil to match their headspace, and those being looked up (`_vcf_look_up`)
_vcf_queue: "OrderedDict[int, tuple[int, str, list[str]]]" = OrderedDict()
_vcf_busy: set[int] = set()
_vcf_lock = threading.Lock()
_vcf_worker = {"running": False}


def _vcf_look_up() -> None:
    """Look up through chemutils what was asked for of VCF's chemicals (`_vcf_queue`), a few
    CIDs at a time (each property saved as it comes), until nothing is left."""
    while True:
        with _vcf_lock:
            _vcf_busy.clear()
            if not _vcf_queue:
                _vcf_worker["running"] = False
                return
            batch = [_vcf_queue.popitem(last=False)[1] for _ in range(min(VCF_LOOKUP_BATCH, len(_vcf_queue)))]
            _vcf_busy.update(cid for cid, _, _ in batch)
        try:
            _look_up_properties(batch)
        except Exception:
            log.exception("could not look up properties of VCF's chemicals")


def _vcf_oil(chemicals: list[dict], model: str) -> dict:
    """The paraffin oil that would give the headspace over VCF's samples (`vcf.search`'s
    ``chemicals``, each with its ``pa``): the v/v of each in it (`headspace.invert`, of them
    all together, with its molar mass from its InChI), from what is known of each, the rest
    queued to be looked up through chemutils in the background (`_vcf_look_up`).

    Returns:
        ``model``, ``chemicals`` (of each given, in order: `headspace.invert`'s, with
        ``properties``, ``sources``, ``links`` and ``failed`` as `_headspace` gives them, and
        ``pending``: whether its properties are being looked up; None for those with no
        headspace), ``total`` (v/v, of those computed), ``pending`` (how many are being
        looked up) and ``error`` (if there are too many to compute).
    """
    targets = [c for c in chemicals if c["pa"] is not None]
    if len(targets) > VCF_MAX_OIL:
        return {"model": model, "chemicals": [None] * len(chemicals), "total": None, "pending": 0,
                "error": f"{len(targets):,} chemicals with a headspace, and at most {VCF_MAX_OIL} are calculated in oil: narrow the search"}
    ids = {c["cid"]: c["inchi"] for c in targets if c["cid"] and c["inchi"]}
    rows = db.chemicals(ids)
    if missing := {cid: inchi for cid, inchi in ids.items() if cid not in rows}:
        db.add_inchis(missing, when=dt.datetime.now().isoformat(timespec="seconds"))
        rows.update(db.chemicals(missing))
    now = time.time()
    needs = lambda row: [p for p in _headspace_properties(row, "paraffin oil", model) if p != "molar_mass"]  # (from the InChI)
    with _vcf_lock:
        for cid, row in rows.items():
            if cid not in _vcf_queue and cid not in _vcf_busy and (todo := _to_look_up(row, needs(row), now)):
                _vcf_queue[cid] = (cid, row["inchi"], todo)
        pending = {cid for cid in rows if cid in _vcf_queue or cid in _vcf_busy}
        start = bool(_vcf_queue) and not _vcf_worker["running"]
        _vcf_worker["running"] |= start
    if start:
        threading.Thread(target=_vcf_look_up, name="vcf-properties", daemon=True).start()

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
    result = headspace.invert(inputs, "paraffin oil", model)
    out = []
    for r, e in zip(result["components"], extra):
        out.append(None if e is None else {**r, **e, "notes": e["notes"] + r["notes"]})
    return {"model": model, "chemicals": out, "total": result["total"], "pending": len(pending), "error": None}


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
    matched as reference datasets' are), and with ``model`` (of `headspace.MODELS`), the
    paraffin oil to match its headspace (``oil``: `_vcf_oil`). With its ``source``
    (``id``, ``title``).

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
    index = odormeta.OdorIndex(db.list_odors())
    for c in res["chemicals"]:
        c["odor_id"] = index.match_chemical({k: c[k] for k in ("cid", "inchi", "name") if c[k]}, c["name"])[0]
    if body.get("model"):
        res["oil"] = _vcf_oil(res["chemicals"], body["model"])
    return {**res, "source": {"id": source["id"], "title": source["title"]}, "temperature_c": headspace.T_K - 273.15}


async def headspace_vcf(request: Request) -> Response:
    """GET: the VCF reference datasets to search (`_vcf_listing`). POST: a search of one
    ({"source", "query", "across_references", "across_products" (of `vcf.STATS`), "model"
    (of `headspace.MODELS`, or null)}) -> `_vcf_search`."""
    if request.method == "GET":
        return JSONResponse({"sources": await run_in_threadpool(_vcf_listing), "stats": vcf.STATS, "max_oil": VCF_MAX_OIL})
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"source", "query", "across_references", "across_products", "model"}')
    if body.get("model") is not None and body["model"] not in headspace.MODELS:
        return json_error(400, f"the model is one of {', '.join(headspace.MODELS)}")
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


async def reference_get(request: Request) -> Response:
    """A reference dataset for the viewer (`reference.load`): gzipped, if large and the browser takes it."""
    data = await run_in_threadpool(lambda: reference.load(db, request.path_params["ref_id"], _odor_index()))
    if data is None:
        return json_error(404, "reference dataset not found")
    data["imported_at"] = _aware(data.get("imported_at"))
    body = await run_in_threadpool(lambda: json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
    if len(body) > 1_000_000 and "gzip" in request.headers.get("accept-encoding", ""):
        body = await run_in_threadpool(gzip.compress, body, 6)
        return Response(body, media_type="application/json", headers={"content-encoding": "gzip", "vary": "accept-encoding"})
    return Response(body, media_type="application/json")


# ----------------------------------------------------------------------- admin

RESTART_DEADLINE_S = 30


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
    chemutils ({"props": [...]}, of `chemprops.PROPERTIES`: see `_relookup`); then, as from GET."""
    body = await request.json()
    props = body.get("props") if isinstance(body, dict) else None
    if not isinstance(props, list) or not props or any(p not in chemprops.PROPERTIES for p in props):
        return json_error(400, f"expected {{\"props\": [...]}}, of {', '.join(chemprops.PROPERTIES)}")
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

    try:
        with _readme_lock:  # not while its README is being overwritten
            if not db.delete_dataset(ds_id, before_commit=move):
                return 404, {"error": "dataset not found"}
    except OSError as e:
        return 500, {"error": f"its folder could not be moved to {dest.parent}, so nothing was deleted ({e})"}
    log.info("dataset %d deleted in the Admin tab; its files: %s", ds_id, moved[0] if moved else "none")
    return 200, {"ok": True, "files": str(moved[0]) if moved else None}


async def admin_reference_check(request: Request) -> Response:
    """Check a file to add as a reference dataset ({"path", "chemical_axis", "kind", "fields"}), see `reference.check`."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"path": ...}')
    axis = body.get("chemical_axis") if body.get("chemical_axis") in reference.AXES else "detect"
    kind = body.get("kind") if body.get("kind") in reference.KINDS else "detect"
    return JSONResponse(await run_in_threadpool(
        lambda: reference.check(str(body.get("path") or ""), axis, kind, _odor_index(), _supervisor(), body.get("fields"))))


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


async def admin_reference_delete(request: Request) -> Response:
    """Delete a reference dataset added in the Admin tab (its import, not its file), see `reference.delete`."""
    ref_id = request.path_params["ref_id"]
    status, result = await run_in_threadpool(reference.delete, db, ref_id)
    if status == 200:
        log.info("reference dataset %s deleted in the Admin tab", ref_id)
    return JSONResponse(result, status_code=status)


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


async def admin_restart(request: Request) -> Response:
    """Restart the server, if Docker or systemd starts it again after it exits."""
    supervisor = _supervisor()
    if supervisor is None:
        return json_error(409, "the server cannot restart itself here: it is not run by Docker or systemd, which would "
                               "start it again (DEPLOY.md). Restart it by hand.")
    log.info("restart requested from the Admin tab")
    return JSONResponse({"ok": True, "by": supervisor, "started_at": _aware(sandbox.STATUS["started_at"])},
                        background=BackgroundTask(_restart))


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
    Route("/api/datasets/{ds_id:int}/movies", dataset_movies),
    Route("/api/movie", movie_info),
    Route("/api/movie/planes", movie_planes),
    Route("/api/movie/rois", movie_rois),
    Route("/api/download", download),
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
    Route("/api/headspace/vcf", headspace_vcf, methods=["GET", "POST"]),
    Route("/api/reference", reference_list),
    Route("/api/reference/{ref_id}", reference_get),
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
    Route("/api/admin/datasets/{ds_id:int}", admin_dataset_delete, methods=["DELETE"]),
    Route("/api/admin/references/check", admin_reference_check, methods=["POST"]),
    Route("/api/admin/references", admin_reference_add, methods=["POST"]),
    Route("/api/admin/references/{ref_id}", admin_reference_delete, methods=["DELETE"]),
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
