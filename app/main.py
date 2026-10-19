"""HTTP API + static frontend (Starlette)."""
from __future__ import annotations

import contextlib
import datetime as dt
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
from pathlib import Path
from typing import Any, Optional

import markdown
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

from . import config, export, merge, movies, odormeta, reference, sandbox, tensor, thorimage, yamlcheck
from .db import Database
from .glomeruli import load_glomeruli
from .ingest import ParsedData, ValidationError, load_file, load_parsed, save_parsed
from .metadata import (
    DEFAULT_CODING_UNITS,
    DEFAULT_INDICATORS,
    DEFAULT_RECORDING_SYSTEMS,
    DEFAULT_WAVELENGTH_NM,
    REQUIRED_FIELDS,
    Metadata,
    dataset_title,
    generate_readme,
)

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
    return JSONResponse({"ok": True, "time": dt.datetime.now().isoformat(timespec="seconds"),
                         "started": sandbox.STATUS["started_at"]})


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
            "default_wavelength_nm": DEFAULT_WAVELENGTH_NM,
            "glomeruli": load_glomeruli(),
            "max_upload_mb": config.MAX_UPLOAD_MB,
            "tensor_root": str(config.TENSOR_ROOT),
        }
    )


async def stage_create(request: Request) -> Response:
    """Upload a CSV/parquet to a staging area, validate it and return the matrix."""
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        return json_error(400, "no file in the request (field name 'file')")
    cleanup_stages()
    stage_id = secrets.token_urlsafe(12)
    d = config.STAGING_DIR / stage_id
    d.mkdir(parents=True)
    fname = safe_filename(upload.filename, "data.csv")
    dest = d / fname
    try:
        size = await _save_upload(upload, dest, max_bytes=int(config.MAX_UPLOAD_MB * 1024 * 1024))
    except HTTPException as e:
        shutil.rmtree(d, ignore_errors=True)
        return json_error(e.status_code, e.detail)

    try:
        parsed: ParsedData = await run_in_threadpool(load_file, dest, original_name=upload.filename)
    except ValidationError as e:
        shutil.rmtree(d, ignore_errors=True)
        return JSONResponse({"ok": False, "errors": e.errors, "warnings": e.warnings, "filename": fname}, status_code=422)
    except Exception as e:  # unexpected
        shutil.rmtree(d, ignore_errors=True)
        return JSONResponse({"ok": False, "errors": [f"unexpected error while reading the file: {e}"], "warnings": []}, status_code=422)

    await run_in_threadpool(save_parsed, parsed, d / "parsed")
    payload = await run_in_threadpool(parsed.to_payload)
    meta = {
        "stage_id": stage_id,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data_file": fname,
        "original_filename": upload.filename,
        "size_bytes": size,
        "sha256": _sha256(dest),
        "source_format": parsed.source_format,
        "odor_yamls": [],
        "experiment_xml": None,
        "experiment_xml_info": None,
    }
    _write_stage_meta(d, meta)
    return JSONResponse({"ok": True, "stage": _public_stage(meta), "data": payload})


def _public_stage(meta: dict) -> dict:
    return {
        "stage_id": meta["stage_id"],
        "data_file": meta["data_file"],
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
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    parsed = await run_in_threadpool(load_parsed, d / "parsed")
    payload = await run_in_threadpool(parsed.to_payload)
    return JSONResponse({"ok": True, "stage": _public_stage(meta), "data": payload})


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


async def stage_add_yaml(request: Request) -> Response:
    """Add odor YAMLs: uploaded (multipart 'file', repeatable) or on tensor (JSON 'tensor_paths')."""
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    ydir = d / "odor_yaml"
    ydir.mkdir(exist_ok=True)
    saved, errors = await _incoming(request, ydir, json_key="tensor_paths", fallback="odors.yaml",
                                    suffixes=(".yaml", ".yml"), max_bytes=20 * 1024 * 1024)
    for name, rel in saved:
        dest = ydir / name
        try:
            odors, _ = yamlcheck.parse_odor_yaml(dest.read_text(encoding="utf-8"))
        except (yamlcheck.YamlError, UnicodeDecodeError) as e:
            dest.unlink(missing_ok=True)
            errors.append(f"{name}: {e}")
            continue
        meta["odor_yamls"] = [y for y in meta["odor_yamls"] if y["name"] != name]
        meta["odor_yamls"].append({"name": name, "n_odors": len(odors), "tensor_path": rel})
    _write_stage_meta(d, meta)
    check = await run_in_threadpool(_run_yaml_check, d, meta) if meta["odor_yamls"] else None
    return JSONResponse({"ok": not errors, "errors": errors, "stage": _public_stage(meta), "check": check})


async def stage_remove_yaml(request: Request) -> Response:
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    name = safe_filename(request.path_params["name"], "")
    (d / "odor_yaml" / name).unlink(missing_ok=True)
    meta["odor_yamls"] = [y for y in meta["odor_yamls"] if y["name"] != name]
    _write_stage_meta(d, meta)
    check = await run_in_threadpool(_run_yaml_check, d, meta) if meta["odor_yamls"] else None
    return JSONResponse({"ok": True, "stage": _public_stage(meta), "check": check})


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
    """Look for the staged dataset's flies, recordings and odor YAMLs in a user's folders on
    tensor ({"name": ...}); ``user`` is null when nobody of that name has folders saved."""
    d = _stage_dir(request.path_params["stage_id"])
    body = await request.json()
    user = await run_in_threadpool(db.get_user, _user_name(body.get("name")))
    if user is None:
        return JSONResponse({"ok": True, "user": None})

    def run() -> dict:
        parsed = load_parsed(d / "parsed")
        return tensor.lookup(config.TENSOR_ROOT, user, parsed.summary()["flies"], parsed.component_odors())

    result = await run_in_threadpool(run)
    return JSONResponse({"ok": True, "user": user, **result})


_FIELD_LABELS = {
    "uploader_name": "your name", "indicator": "calcium indicator", "driver": "Gal4 driver",
    "coding_unit": "coding unit", "age_min_days": "fly age (min)", "age_max_days": "fly age (max)",
    "power_mw": "power (mW)", "power_tolerance_mw": "power tolerance", "wavelength_nm": "wavelength",
    "panel_names": "panel name(s)", "recording_system": "recording system", "source_code_url": "analysis code link",
    "raw_data_path": "raw data path", "mocorr_data_path": "motion-corrected data path", "genotype": "genotype",
    "notes": "notes", "power_measured_same_day": "power measured on the day",
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
    summary = parsed.summary()
    readme = generate_readme(
        m,
        summary,
        {
            "data": meta["data_file"],
            "sha256": meta["sha256"],
            "odor_yamls": [y["name"] for y in meta["odor_yamls"]],
            "experiment_xml": meta.get("experiment_xml"),
            "sources": _tensor_sources(meta),
        },
    )
    return JSONResponse({"ok": True, "readme": readme, "title": dataset_title(m, summary)})


def _commit(d: Path, meta: dict, m: Metadata, readme: str, readme_edited: bool) -> dict:
    parsed = load_parsed(d / "parsed")
    summary = parsed.summary()
    # consistency checks that must pass before anything is written
    problems = []
    if meta["odor_yamls"]:
        check = _run_yaml_check(d, meta)
        if not check["ok"]:
            problems.append(
                "odor YAML check failed: " + "; ".join(f"{x['odor']} ({x['hint']})" for x in check["missing"][:10])
            )
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
    shutil.move(str(d / meta["data_file"]), out / meta["data_file"])
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
    return JSONResponse({"datasets": rows})


def _readme_path(ds_id: int) -> Path:
    return config.DATASETS_DIR / str(ds_id) / "README.md"


async def dataset_get(request: Request) -> Response:
    ds_id = int(request.path_params["ds_id"])
    d = await run_in_threadpool(db.get_dataset, ds_id)
    if d is None:
        return json_error(404, "dataset not found")
    d["readme_html"] = _readme_html(d["readme_md"])
    d["readme_path"] = str(_readme_path(ds_id))
    files = [d["data_file"], "README.md", "metadata.json"]
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


async def dataset_data_paths(request: Request) -> Response:
    """Where a dataset's raw and motion-corrected data are on the server, and what is in
    them (``raw``, ``mocorr``: see `tensor.find_data`)."""
    ds_id = int(request.path_params["ds_id"])

    def run() -> Optional[dict]:
        d = db.get_dataset(ds_id)
        if d is None:
            return None
        bases = [Path(p) for p in sandbox.folders()]
        return {k: tensor.find_data(d["metadata"].get(f"{k}_data_path"), config.TENSOR_ROOT, bases) for k in ("raw", "mocorr")}

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
    ``movies`` as from `movies.find`, and ``error``: why its folder was not looked in)."""
    ds_id = int(request.path_params["ds_id"])

    def run() -> Optional[dict]:
        d = db.get_dataset(ds_id)
        if d is None:
            return None
        flies = d["summary"].get("flies") or []
        bases = [Path(p) for p in sandbox.folders()]
        out = {}
        for kind in ("raw", "mocorr"):
            found = tensor.find_data(d["metadata"].get(f"{kind}_data_path"), config.TENSOR_ROOT, bases)
            folder = Path(found["path"]) if found["status"] == "ok" and found["is_dir"] else None
            out[kind] = {"movies": movies.find(folder, kind, flies) if folder else [], "error": found["error"]}
        return out

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
    a zip, or a CSV for the matrix of a single dataset."""
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
        name, media_type = await run_in_threadpool(export.write_download, db, ids, fmt, dest, config.DATASETS_DIR)
    except export.ExportError as e:
        dest.unlink(missing_ok=True)
        return json_error(404, str(e))
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return FileResponse(dest, filename=name, media_type=media_type, background=BackgroundTask(dest.unlink, missing_ok=True))


# ----------------------------------------------------------------------- merge


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
    `merge_preview`): a zip, with the datasets scaled as `merge.plan` says."""
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
        name, media_type = await run_in_threadpool(export.write_download, db, [d["id"] for d in p["datasets"]], fmt, dest,
                                                   config.DATASETS_DIR, merged=p)
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
sheet_watch = odormeta.SheetWatch(config.ODOR_SHEET_CHECK_MINUTES * 60)


def _odor_sheet() -> dict:
    """The odor sheet's link (``url``, or None): as set in the Admin tab (``saved``, with
    ``updated_at``), else from OLFDB_ODOR_SHEET_URL (``env``)."""
    saved = db.get_setting(SHEET_SETTING)
    return {"url": saved["value"] if saved else config.ODOR_SHEET_URL or None, "saved": saved is not None,
            "updated_at": saved["updated_at"] if saved else None, "env": bool(config.ODOR_SHEET_URL)}


def _sheet_status() -> dict:
    url = _odor_sheet()["url"]
    state = sheet_watch.check(url)
    last = next(iter(db.odor_syncs(limit=1, source=SHEET_SOURCE)), None)
    return {
        "configured": bool(url),
        "url": url,
        **state,
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
        return {"odors": rows, "syncs": db.odor_syncs(limit=20)}

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
    fp = odormeta.fingerprint(rows)
    if sheet_url:
        sheet_watch.record(sheet_url, rows)
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


def _fetch_inchis() -> None:
    """Fetch from PubChem the InChI of every odor CID that has none in the database yet
    (in one request per 100 CIDs; skipped while another call is at it)."""
    if not _inchi_lock.acquire(blocking=False):
        return
    try:
        missing = sorted({o["cid"] for o in db.list_odors() if o["cid"] and not o["inchi"]})
        if missing:
            db.save_inchis(odormeta.fetch_inchis(missing), when=dt.datetime.now().isoformat(timespec="seconds"))
    finally:
        _inchi_lock.release()


def _prefetch_pubchem() -> None:
    """Structure images and InChIs of every odor's CID."""
    for o in db.list_odors():
        if o["cid"]:
            odormeta.structure_png(o["cid"])
    _fetch_inchis()


def _apply(fp: str, change_ids: set[int], delete_ids: set[int], applied_by: str) -> dict:
    path = config.ODOR_PREVIEWS_DIR / f"{fp}.json"
    if not path.exists():
        return {"ok": False, "errors": ["this preview has expired: check the sheet (or import the file) again"]}
    saved = json.loads(path.read_text())
    if saved["errors"]:
        return {"ok": False, "errors": ["the sheet has errors: fix them and check it again"]}
    sheet = saved["source"] == SHEET_SOURCE
    if sheet and not applied_by:
        return {"ok": False, "errors": ["your name is required"]}
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
            applied_by=applied_by if sheet else "", source=saved["source"], fingerprint=fp, details=details,
            when=dt.datetime.now().isoformat(timespec="seconds"),
        )
    except sqlite3.IntegrityError as e:
        return {"ok": False, "errors": [f"could not apply the sync: {e}"]}
    return {"ok": True, "added": len(d["added"]), "changed": len(changes), "deleted": len(deletions)}


async def odors_apply(request: Request) -> Response:
    """Apply a preview: all new odors, plus the changes and deletions listed (by odor id).
    ``applied_by``, the name of whoever applies it, is required for the sheet; a CSV import
    is not tied to anyone (any name given is ignored)."""
    body = await request.json()
    fp = str(body.get("fingerprint", ""))
    applied_by = str(body.get("applied_by") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", fp):
        return json_error(400, "missing or malformed fingerprint")
    try:
        change_ids = {int(x) for x in body.get("changes") or []}
        delete_ids = {int(x) for x in body.get("deletions") or []}
    except (TypeError, ValueError):
        return json_error(400, "changes and deletions must be lists of odor ids")
    result = await run_in_threadpool(_apply, fp, change_ids, delete_ids, applied_by)
    return JSONResponse(result, status_code=200 if result["ok"] else 422,
                        background=BackgroundTask(_prefetch_pubchem) if result["ok"] else None)


async def odor_structure(request: Request) -> Response:
    """PubChem structure image of an odor in the table (cached on the server). The InChIs
    of odors without one are fetched afterwards, if this one has none."""
    cid = int(request.path_params["cid"])
    odor = next((o for o in await run_in_threadpool(db.list_odors) if o["cid"] == cid), None)
    if odor is None:
        return json_error(404, "no odor with this CID")
    background = BackgroundTask(_fetch_inchis) if not odor["inchi"] else None
    path = await run_in_threadpool(odormeta.structure_png, cid)
    if path is None:
        return JSONResponse({"error": "structure image not available"}, status_code=404, background=background)
    return FileResponse(path, media_type="image/png", headers={"cache-control": "public, max-age=604800"},
                        background=background)


async def odor_template(request: Request) -> Response:
    return FileResponse(config.ODOR_TEMPLATE, filename="odor_sheet_template.csv", media_type="text/csv")


async def dataset_odor_table(request: Request) -> Response:
    """A dataset's odors (per panel, as in the data) matched to the odor table."""
    ds_id = int(request.path_params["ds_id"])

    def build() -> Optional[dict]:
        ds = db.get_dataset(ds_id)
        if ds is None:
            return None
        index = odormeta.OdorIndex(db.list_odors())
        rows = odormeta.dataset_odors(ds, db.get_trials(ds_id), index, odormeta.dataset_yaml_odors(ds, config.DATASETS_DIR))
        ids = {r["odor_id"] for r in rows} | {i for r in rows for i in r["candidates"]}
        odors = {i: {**index.by_id[i], "links": odormeta.links(index.by_id[i])} for i in ids if i is not None}
        return {"rows": rows, "odors": odors}

    result = await run_in_threadpool(build)
    if result is None:
        return json_error(404, "dataset not found")
    return JSONResponse(result)


def _odor_index() -> odormeta.OdorIndex:
    return odormeta.OdorIndex(db.list_odors())


async def reference_list(request: Request) -> Response:
    """Every reference matrix, with whether it can be shown (and why not) and whether its chemicals are identified."""
    return JSONResponse({"datasets": await run_in_threadpool(lambda: reference.listing(db, _odor_index()))})


async def reference_get(request: Request) -> Response:
    data = await run_in_threadpool(lambda: reference.load(db, request.path_params["ref_id"], _odor_index()))
    if data is None:
        return json_error(404, "reference matrix not found")
    return JSONResponse(data)


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
    return {
        "saved": saved,
        "active": {k: active[k] for k in sandbox.KINDS} if active else None,
        "pending": active is not None and any(saved[k] != active[k] for k in sandbox.KINDS),
        "checks": sandbox.check_folders(saved)[0],
        "odor_sheet": _odor_sheet(),
        "data_dir": str(config.DATA_DIR),
        "deleted_dir": str(config.DELETED_DIR),
        "tensor_root": str(config.TENSOR_ROOT),
        "started_at": sandbox.STATUS["started_at"],
        "kernel": sandbox.STATUS["kernel"],
        "restart": {"possible": by is not None, "by": by},
    }


def _folder_lists(body: Any) -> bool:
    return isinstance(body, dict) and all(isinstance(body.get(k) or [], list) for k in sandbox.KINDS)


async def admin_get(request: Request) -> Response:
    """The folders the server may read and write: as saved, as in effect since it started, and checked."""
    return JSONResponse(await run_in_threadpool(_admin_state))


async def admin_check(request: Request) -> Response:
    """Check folders for the Admin tab ({"read_dirs", "write_dirs"}), without saving them."""
    body = await request.json()
    if not _folder_lists(body):
        return json_error(400, f"expected lists of paths in {', '.join(sandbox.KINDS)}")
    checks, _ = await run_in_threadpool(sandbox.check_folders, body)
    return JSONResponse({"ok": all(c["ok"] for cs in checks.values() for c in cs), "checks": checks})


async def admin_save(request: Request) -> Response:
    """Save the folders the server may read and write ({"read_dirs", "write_dirs"}), in
    effect from the next restart; nothing is saved if any of them is malformed."""
    body = await request.json()
    if not _folder_lists(body):
        return json_error(400, f"expected lists of paths in {', '.join(sandbox.KINDS)}")
    checks, dirs = await run_in_threadpool(sandbox.check_folders, body)
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
    """Check a file to add as a reference matrix ({"path", "chemical_axis", "kind"}), see `reference.check`."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"path": ...}')
    axis = body.get("chemical_axis") if body.get("chemical_axis") in reference.AXES else "detect"
    kind = body.get("kind") if body.get("kind") in reference.KINDS else "detect"
    return JSONResponse(await run_in_threadpool(
        lambda: reference.check(str(body.get("path") or ""), axis, kind, _odor_index(), _supervisor())))


async def admin_reference_add(request: Request) -> Response:
    """Add a reference matrix ({"path", "title", "description", "kind", "chemical_axis",
    "value_label", "row_label", "col_label", "source"}), see `reference.add`."""
    body = await request.json()
    if not isinstance(body, dict):
        return json_error(400, 'expected {"path": ..., "title": ...}')
    status, result = await run_in_threadpool(lambda: reference.add(db, body, _odor_index(), _supervisor()))
    if status == 200:
        log.info("reference matrix %s added in the Admin tab: %s", result["id"], body.get("path"))
    return JSONResponse(result, status_code=status)


async def admin_reference_delete(request: Request) -> Response:
    """Delete a reference matrix added in the Admin tab (its import, not its file), see `reference.delete`."""
    ref_id = request.path_params["ref_id"]
    status, result = await run_in_threadpool(reference.delete, db, ref_id)
    if status == 200:
        log.info("reference matrix %s deleted in the Admin tab", ref_id)
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
    return JSONResponse({"ok": True, "by": supervisor, "started_at": sandbox.STATUS["started_at"]},
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
    Route("/api/odors/status", odors_status),
    Route("/api/odors/preview", odors_preview, methods=["POST"]),
    Route("/api/odors/apply", odors_apply, methods=["POST"]),
    Route("/api/odors/structure/{cid:int}", odor_structure),
    Route("/api/odors/template.csv", odor_template),
    Route("/api/reference", reference_list),
    Route("/api/reference/{ref_id}", reference_get),
    Route("/api/admin", admin_get),
    Route("/api/admin", admin_save, methods=["POST"]),
    Route("/api/admin/check", admin_check, methods=["POST"]),
    Route("/api/admin/odor_sheet", admin_odor_sheet, methods=["POST"]),
    Route("/api/admin/host", admin_host),
    Route("/api/admin/restart", admin_restart, methods=["POST"]),
    Route("/api/admin/datasets/{ds_id:int}", admin_dataset_delete, methods=["DELETE"]),
    Route("/api/admin/references/check", admin_reference_check, methods=["POST"]),
    Route("/api/admin/references", admin_reference_add, methods=["POST"]),
    Route("/api/admin/references/{ref_id}", admin_reference_delete, methods=["DELETE"]),
    Mount("/static", RevalidatedStaticFiles(directory=str(config.STATIC_DIR)), name="static"),
]



@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    # before the first request, so before any worker thread starts (see sandbox.start)
    sandbox.start()
    yield


app = Starlette(routes=routes, exception_handlers={HTTPException: http_exception}, lifespan=lifespan)
