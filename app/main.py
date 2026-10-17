"""HTTP API + static frontend (Starlette)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import secrets
import shutil
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

from . import config, export, reference, thorimage, yamlcheck
from .db import Database
from .glomeruli import load_glomeruli
from .ingest import ParsedData, ValidationError, load_file, load_parsed, save_parsed
from .metadata import (
    DEFAULT_CODING_UNITS,
    DEFAULT_INDICATORS,
    DEFAULT_RECORDING_SYSTEMS,
    DEFAULT_WAVELENGTH_NM,
    Metadata,
    dataset_title,
    generate_readme,
)

config.ensure_dirs()
db = Database(config.DB_PATH)

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


async def index(request: Request) -> Response:
    return FileResponse(config.STATIC_DIR / "index.html")


async def health(request: Request) -> Response:
    return JSONResponse({"ok": True, "time": dt.datetime.now().isoformat(timespec="seconds")})


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
        "odor_yamls": [{"name": y["name"], "n_odors": y["n_odors"]} for y in meta["odor_yamls"]],
        "experiment_xml": meta["experiment_xml"],
        "experiment_xml_info": meta["experiment_xml_info"],
    }


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
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    form = await request.form()
    uploads = [v for v in form.getlist("file") if isinstance(v, UploadFile)]
    if not uploads:
        return json_error(400, "no YAML file in the request (field name 'file')")
    ydir = d / "odor_yaml"
    ydir.mkdir(exist_ok=True)
    errors = []
    for up in uploads:
        name = safe_filename(up.filename, "odors.yaml")
        dest = ydir / name
        await _save_upload(up, dest, max_bytes=20 * 1024 * 1024)
        try:
            odors, _ = yamlcheck.parse_odor_yaml(dest.read_text(encoding="utf-8"))
        except (yamlcheck.YamlError, UnicodeDecodeError) as e:
            dest.unlink(missing_ok=True)
            errors.append(f"{name}: {e}")
            continue
        meta["odor_yamls"] = [y for y in meta["odor_yamls"] if y["name"] != name]
        meta["odor_yamls"].append({"name": name, "n_odors": len(odors)})
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
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    form = await request.form()
    up = form.get("file")
    if not isinstance(up, UploadFile):
        return json_error(400, "no XML file in the request (field name 'file')")
    name = safe_filename(up.filename, "Experiment.xml")
    dest = d / name
    await _save_upload(up, dest, max_bytes=50 * 1024 * 1024)
    try:
        info = thorimage.parse_experiment_xml(dest.read_text(encoding="utf-8", errors="replace"))
    except thorimage.ExperimentXmlError as e:
        dest.unlink(missing_ok=True)
        return JSONResponse({"ok": False, "errors": [str(e)]}, status_code=422)
    if meta.get("experiment_xml") and meta["experiment_xml"] != name:
        (d / meta["experiment_xml"]).unlink(missing_ok=True)
    meta["experiment_xml"] = name
    meta["experiment_xml_info"] = info["info"]
    _write_stage_meta(d, meta)
    return JSONResponse({"ok": True, "stage": _public_stage(meta), "info": info["info"], "suggestions": info["suggestions"]})


async def stage_remove_xml(request: Request) -> Response:
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    if meta.get("experiment_xml"):
        (d / meta["experiment_xml"]).unlink(missing_ok=True)
    meta["experiment_xml"] = None
    meta["experiment_xml_info"] = None
    _write_stage_meta(d, meta)
    return JSONResponse({"ok": True, "stage": _public_stage(meta)})


_FIELD_LABELS = {
    "uploader_name": "your name", "indicator": "calcium indicator", "driver": "Gal4 driver",
    "coding_unit": "coding unit", "age_min_days": "fly age (min)", "age_max_days": "fly age (max)",
    "power_mw": "power (mW)", "power_tolerance_mw": "power tolerance", "wavelength_nm": "wavelength",
    "panel_names": "panel name(s)", "recording_system": "recording system", "source_code_url": "analysis code link",
    "raw_data_path": "raw data path", "genotype": "genotype", "notes": "notes",
}


def _parse_metadata(body: Any) -> tuple[Optional[Metadata], list[str]]:
    try:
        return Metadata.model_validate(body or {}), []
    except PydanticValidationError as e:
        msgs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            label = _FIELD_LABELS.get(loc, loc or "metadata")
            if err["type"] in ("string_too_short", "string_type", "missing") and loc in ("uploader_name", "indicator", "driver", "coding_unit"):
                msgs.append(f"{label} is required")
            else:
                msgs.append(f"{label}: {err['msg'].replace('Value error, ', '')}")
        return None, msgs


async def stage_readme(request: Request) -> Response:
    """Generate a README.md from the metadata (not yet saved)."""
    d = _stage_dir(request.path_params["stage_id"])
    meta = _stage_meta(d)
    body = await request.json()
    m, errors = _parse_metadata(body.get("metadata"))
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


async def dataset_get(request: Request) -> Response:
    ds_id = int(request.path_params["ds_id"])
    d = await run_in_threadpool(db.get_dataset, ds_id)
    if d is None:
        return json_error(404, "dataset not found")
    d["readme_html"] = _readme_html(d["readme_md"])
    files = [d["data_file"], "README.md", "metadata.json"]
    files += [f"odor_yaml/{y}" for y in d["odor_yaml_files"]]
    if d.get("experiment_xml_file"):
        files.append(d["experiment_xml_file"])
    d["files"] = files
    return JSONResponse(d)


async def dataset_matrix(request: Request) -> Response:
    ds_id = int(request.path_params["ds_id"])
    m = await run_in_threadpool(db.get_matrix, ds_id)
    if m is None:
        return json_error(404, "dataset not found")
    return JSONResponse(m)


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
    for p in config.EXPORTS_DIR.glob("*.zip"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


async def download(request: Request) -> Response:
    """Zip of one or more datasets: ``?ids=1,2,3&format=files|wide|long`` (see export.py)."""
    try:
        ids = [int(x) for x in request.query_params.get("ids", "").split(",") if x.strip()]
    except ValueError:
        return json_error(400, "ids must be comma-separated dataset numbers")
    fmt = request.query_params.get("format", "files")
    if not ids or fmt not in export.FORMATS:
        return json_error(400, f"expected ?ids=1,2,...&format={'|'.join(export.FORMATS)}")
    cleanup_exports()
    dest = config.EXPORTS_DIR / f"{secrets.token_urlsafe(9)}.zip"
    try:
        name = await run_in_threadpool(export.write_zip, db, ids, fmt, dest, config.DATASETS_DIR)
    except export.ExportError as e:
        dest.unlink(missing_ok=True)
        return json_error(404, str(e))
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return FileResponse(
        dest, filename=name, media_type="application/zip", background=BackgroundTask(dest.unlink, missing_ok=True)
    )


async def reference_list(request: Request) -> Response:
    return JSONResponse({"datasets": reference.manifest()})


async def reference_get(request: Request) -> Response:
    data = await run_in_threadpool(reference.load, request.path_params["ref_id"])
    if data is None:
        return json_error(404, "reference dataset not found")
    return JSONResponse(data)


async def http_exception(request: Request, exc: HTTPException) -> Response:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


routes = [
    Route("/", index),
    Route("/api/health", health),
    Route("/api/options", options),
    Route("/api/stage", stage_create, methods=["POST"]),
    Route("/api/stage/{stage_id}", stage_get),
    Route("/api/stage/{stage_id}/odor_yaml", stage_add_yaml, methods=["POST"]),
    Route("/api/stage/{stage_id}/odor_yaml/{name}", stage_remove_yaml, methods=["DELETE"]),
    Route("/api/stage/{stage_id}/experiment_xml", stage_add_xml, methods=["POST"]),
    Route("/api/stage/{stage_id}/experiment_xml", stage_remove_xml, methods=["DELETE"]),
    Route("/api/stage/{stage_id}/readme", stage_readme, methods=["POST"]),
    Route("/api/stage/{stage_id}/commit", stage_commit, methods=["POST"]),
    Route("/api/datasets", datasets_list),
    Route("/api/datasets/{ds_id:int}", dataset_get),
    Route("/api/datasets/{ds_id:int}/matrix", dataset_matrix),
    Route("/api/datasets/{ds_id:int}/files/{path:path}", dataset_file),
    Route("/api/download", download),
    Route("/api/reference", reference_list),
    Route("/api/reference/{ref_id}", reference_get),
    Mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static"),
]

app = Starlette(routes=routes, exception_handlers={HTTPException: http_exception})
