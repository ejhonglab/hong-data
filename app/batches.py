"""Batches of predictions: one model's predictions of every fly it was not trained on, asked for at
once in the AL segmentation tab (held-out flies, flies without ROIs of their own, flies of
whoever is named in Settings), followed together; and the signals of all of their ROIs extracted
at once (`extract`), joined into one temporary dataset that whoever asked for it (or the Admin
tab) can delete. The signals of an extraction of one fly (not a batch's) become a temporary
dataset of their own too, as soon as they are done (`make_extraction_datasets`), so that every
extraction can be seen in Browse.

A batch is a row of ``prediction_batches``: its fit, who asked and when, its predictions (their
ids: each fly's, made now, or done before by the same model and kept), and once its signals are
asked for, ``extract`` ({ids: the extractions, by, at, rois, status, dataset_id, error}). The
odors of the model each fly lacks are each prediction's own (its result's ``missing_odors``, and
its folder's `predictions.MISSING`).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from . import config, extract, predictions

# a dataset's metadata key: {by, at, fit, extractions, and batch, or extraction and fly} for one made of a batch's signals,
# or of one extraction's (`extraction_dataset`)
TEMPORARY = "temporary"


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _row(r) -> dict:
    d = dict(r)
    d["ids"] = json.loads(d.pop("ids_json") or "[]")
    d["options"] = json.loads(d.pop("options_json") or "null") or predictions.options({})
    d["extract"] = json.loads(d.pop("extract_json")) if d.get("extract_json") else None
    return d


def get(db, batch_id: int) -> Optional[dict]:
    with db.connect() as con:
        r = con.execute("SELECT * FROM prediction_batches WHERE id = ?", (batch_id,)).fetchone()
    return _row(r) if r else None


def all_batches(db) -> list[dict]:
    with db.connect() as con:
        return [_row(r) for r in con.execute("SELECT * FROM prediction_batches ORDER BY id DESC")]


def create(db, fit_id: int, flies: list[dict], name: str, ip: Optional[str], opts: dict) -> dict:
    """A batch of predictions by fit `fit_id`'s model of `flies` (each as `predictions.locate` gives it):
    each fly's asked for (`predictions.create`), but one done before by that model is kept as it is, and
    one queued or running is followed as it is.

    Raises:
        ValueError: for no flies.
    """
    if not flies:
        raise ValueError("no flies to predict: every fly listed was trained on, or none is listed")
    ids = []
    for fly in flies:
        try:
            p, _ = predictions.create(db, fit_id, fly, name, ip, opts=opts)
        except predictions.Exists as e:
            p = e.prediction
            if p["status"] in ("failed", "interrupted"):
                p, _ = predictions.create(db, fit_id, fly, name, ip, overwrite=True, opts=opts)
        ids.append(p["id"])
    with db.connect() as con:
        batch_id = con.execute("INSERT INTO prediction_batches (fit_id, name, ip, created_at, ids_json, options_json) VALUES (?,?,?,?,?,?)",
                               (fit_id, name, ip, now(), json.dumps(list(dict.fromkeys(ids))), json.dumps(opts))).lastrowid
    return get(db, batch_id)


def delete(db, batch_id: int) -> bool:
    """Forgets a batch (its predictions, extractions and dataset stay)."""
    with db.connect() as con:
        return con.execute("DELETE FROM prediction_batches WHERE id = ?", (batch_id,)).rowcount > 0


def _set_extract(db, batch_id: int, **fields) -> None:
    with db.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT extract_json FROM prediction_batches WHERE id = ?", (batch_id,)).fetchone()
        if r is None:
            return
        con.execute("UPDATE prediction_batches SET extract_json = ? WHERE id = ?",
                    (json.dumps({**(json.loads(r["extract_json"]) if r["extract_json"] else {}), **fields}), batch_id))


def extract_all(db, batch_id: int, rois: str, opts: dict, name: str, ip: Optional[str], tensor_root: Path, bases: list[Path]) -> dict:
    """Asks for the signals of the ROIs of every prediction of a batch done (`rois`: "predicted", or
    "edited": each prediction's as edited where it was, else as predicted), with `opts`
    (`extract.options`); once they are all over, `finish` joins them into a temporary dataset.

    Raises:
        LookupError: no such batch.
        ValueError: none of its predictions is done, or its signals were asked for already and are not over.
    """
    b = get(db, batch_id)
    if b is None:
        raise LookupError("no such batch")
    if b["extract"] and b["extract"].get("status") == "running":
        raise ValueError("its signals are being extracted already")
    ids, skipped, why = [], [], []
    for pid in b["ids"]:
        p = predictions.get(db, pid)
        if p is None or p["status"] != "done":
            skipped.append(pid)
            continue
        kind = "edited" if rois == "edited" and predictions.edited_info(p) else "predicted"
        try:
            of = extract.locate(db, {"rois": kind, "prediction": pid}, tensor_root, bases)
        except ValueError as e:
            skipped.append(pid)
            why.append(f"{p['fly']}: {e}")
            continue
        ids.append(extract.create(db, of, opts, name, ip)["id"])
    if not ids:
        raise ValueError("none of its predictions is done" if not why else f"none of its flies' signals can be extracted ({'; '.join(why[:5])})")
    _set_extract(db, batch_id, ids=ids, skipped=skipped, by=name, ip=ip, at=now(), rois=rois, status="running", dataset_id=None, error=None)
    return get(db, batch_id)


def finish(db) -> list[int]:
    """Of each batch whose signals are being extracted and are all over, joins those done into a
    temporary dataset (`make_dataset`). Returns the datasets made. (Run by the fit worker after each
    extraction.)"""
    made = []
    for b in all_batches(db):
        x = b["extract"]
        if not x or x.get("status") != "running":
            continue
        exts = [extract.get(db, i) for i in x["ids"]]
        if any(e is not None and e["status"] in extract.ACTIVE for e in exts):
            continue
        done = [e for e in exts if e is not None and e["status"] == "done"]
        try:
            ds_id = make_dataset(db, b, done)
            _set_extract(db, b["id"], status="done", dataset_id=ds_id, done=len(done))
            made.append(ds_id)
        except Exception as e:  # noqa: BLE001 (said with the batch)
            _set_extract(db, b["id"], status="failed", error=str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}")
    return made


def joined_table(paths: list[Path]):
    """Extractions' ``responses_glomeruli.csv`` (trials × (date, fly_num, roi)) joined into one table, rows
    of trials of the same panel, odors and repeat as one (each fly's odors padded with solvent to the most
    any has)."""
    import pandas as pd
    from hong2p import olf

    parts = []
    for p in paths:
        names = pd.read_csv(p, header=None, nrows=4).iloc[3].tolist()
        k = sum(1 for v in names if isinstance(v, str) and v)
        parts.append(pd.read_csv(p, header=[0, 1, 2], index_col=list(range(k))))
    n = max(sum(1 for x in t.index.names if re.fullmatch(r"odor\d+", str(x))) for t in parts)
    padded = []
    for t in parts:
        names = list(t.index.names)
        odor_at = [i for i, x in enumerate(names) if re.fullmatch(r"odor\d+", str(x))]
        rows = [(r[names.index("panel")], r[names.index("is_pair")], *(r[i] for i in odor_at),
                 *([olf.solvent_str] * (n - len(odor_at))), r[names.index("repeat")]) for r in t.index]
        t = t.copy()
        t.index = pd.MultiIndex.from_tuples(rows, names=["panel", "is_pair", *(f"odor{i}" for i in range(1, n + 1)), "repeat"])
        padded.append(t)
    table = pd.concat(padded, axis=1, sort=False)
    table.columns.names = ["date", "fly_num", "roi"]
    return table


def make_dataset(db, b: dict, exts: list[dict]) -> int:
    """A temporary dataset of the signals of a batch's extractions done (their ROIs named as glomeruli):
    their ``responses_glomeruli.csv`` joined (`joined_table`), as an upload of it would be, its metadata
    `TEMPORARY` saying whose it is; the indicator and driver of the datasets the model was fit to, where
    all say the same ("unknown" otherwise)."""
    paths = [extract.out_dir(e["id"]) / "responses_glomeruli.csv" for e in exts]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise ValueError("none of its extractions has ROIs named as glomeruli")
    return _temporary(db, paths, f"batch_{b['id']}_responses.csv", b["extract"]["by"], b["fit_id"],
                      {"batch": b["id"], "extractions": [e["id"] for e in exts]},
                      f"Temporary: the signals of batch {b['id']}'s predictions (fit #{b['fit_id']}'s model) of {len(paths)} flies, "
                      f"extracted in the AL segmentation tab, joined. Whoever asked for it ({b['extract']['by']}), or the Admin tab, can delete it.")


def in_batch(db, ext_id: int) -> bool:
    """Whether an extraction is one of a batch's (whose signals are joined into a dataset of their own: `finish`)."""
    return any(ext_id in ((b["extract"] or {}).get("ids") or []) for b in all_batches(db))


def extraction_dataset(db, e: dict) -> int:
    """A temporary dataset of the signals of one extraction done (not a batch's), as a batch's is made
    (`make_dataset`): its ROIs named as glomeruli, as an upload of them would be.

    Raises:
        ValueError: if none of its ROIs is named as a glomerulus, or its fly has no date and number.
    """
    path = extract.out_dir(e["id"]) / "responses_glomeruli.csv"
    if not path.is_file():
        raise ValueError("none of its ROIs is named as a glomerulus (a dataset takes only those)")
    kind = (e.get("rois") or {}).get("kind")
    fit = (e.get("rois") or {}).get("fit")
    what = {"own": "its own ROIs", "predicted": f"fit #{fit}'s predicted ROIs", "edited": f"fit #{fit}'s predicted ROIs, as edited"}.get(kind, "its ROIs")
    return _temporary(db, [path], f"extraction_{e['id']}_responses.csv", e["name"], fit,
                      {"extraction": e["id"], "fly": e["fly"], "extractions": [e["id"]]},
                      f"Temporary: the signals of {e['fly']}'s {what}, extracted in the AL segmentation tab (extraction #{e['id']}). "
                      f"Whoever asked for it ({e['name']}), or the Admin tab, can delete it.")


def make_extraction_datasets(db, ids: Optional[list[int]] = None) -> list[int]:
    """Makes the temporary dataset of each extraction done (of `ids`; else every one) that is not a batch's and has
    none yet, nor was said to have none (its result's ``dataset``, or ``dataset_error``: why not, kept). Returns the
    datasets made. (Run by the fit worker after each extraction, and as it starts, for those made before.)"""
    made = []
    for e in (extract.get(db, i) for i in ids) if ids is not None else extract.all_extractions(db):
        if e is None or e["status"] != "done" or not e["result"] or "dataset" in e["result"] or "dataset_error" in e["result"] or in_batch(db, e["id"]):
            continue
        try:
            ds_id = extraction_dataset(db, e)
            extract.update(db, e["id"], result={**e["result"], "dataset": ds_id})
            made.append(ds_id)
        except Exception as err:  # noqa: BLE001 (said with the extraction)
            extract.update(db, e["id"], result={**e["result"], "dataset_error": str(err) if isinstance(err, ValueError) else f"{type(err).__name__}: {err}"})
    return made


def _temporary(db, paths: list[Path], file_name: str, by: str, fit_id: Optional[int], about: dict, notes: str) -> int:
    """A temporary dataset of extractions' ``responses_glomeruli.csv`` (`paths`), joined (`joined_table`), as an
    upload of it would be, named `file_name`, its metadata `TEMPORARY` saying whose it is (`by`) and what of
    (`about`); the indicator and driver of the datasets fit `fit_id`'s model was fit to, where all say the same
    ("unknown" otherwise, and without a fit)."""
    from .ingest import load_file
    from .metadata import Metadata, dataset_title, generate_readme

    table = joined_table(paths)
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=config.TMP_DIR))
    try:
        data = tmp / file_name
        table.to_csv(data)
        parsed = load_file(data)
        from . import fits

        # (the indicator and driver of the datasets the model was fit to, where they all say the same)
        fit = (fits.get(db, fit_id) if fit_id is not None else None) or {}
        ds = set(((fit.get("flies") or {}).get("datasets")) or [])
        metas = [d for d in db.list_datasets() if d["id"] in ds]
        one = lambda k: (lambda vs: vs.pop() if len(vs) == 1 else "unknown")({m[k] for m in metas if m.get(k)})  # noqa: E731
        m = Metadata(uploader_name=by, indicator=one("indicator"), driver=one("driver"), coding_unit="glomeruli", notes=notes)
        summary = parsed.summary()
        meta = {**m.model_dump(), TEMPORARY: {"by": by, "at": now(), "fit": fit_id, **about}}
        import hashlib

        sha = hashlib.sha256(data.read_bytes()).hexdigest()
        readme = generate_readme(m, summary, {"data_files": [{"name": data.name, "sha256": sha}], "odor_yamls": [], "experiment_xml": None, "sources": {}})

        ds_id = db.insert_dataset(created_at=dt.datetime.now().isoformat(timespec="seconds"), title=f"[temporary] {dataset_title(m, summary)}",
                                  metadata=meta, readme_md=readme, readme_edited=False, data_file=data.name, data_format=parsed.source_format,
                                  data_sha256=sha, odor_yaml_files=[], experiment_xml_file=None,
                                  experiment_xml_info=None, summary=summary, units=parsed.units, trials=parsed.trials, values=parsed.values)
        out = config.DATASETS_DIR / str(ds_id)
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(data), out / data.name)
        (out / "README.md").write_text(readme, encoding="utf-8")
        (out / "metadata.json").write_text(json.dumps({"dataset_id": ds_id, "metadata": meta, "summary": summary}, indent=1), encoding="utf-8")
        return ds_id
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def temporary_of(d: dict) -> Optional[dict]:
    """A dataset's `TEMPORARY` (whose it is), if it is one made of a batch's signals, or of one extraction's."""
    return (d.get("metadata") or {}).get(TEMPORARY) or (json.loads(d["temporary"]) if isinstance(d.get("temporary"), str) else d.get("temporary"))


def listing(db) -> list[dict]:
    """Every batch, newest first, with its predictions' statuses counted (``counts``) and its extraction's
    (``extract``, with ``counts`` of its extractions' statuses, while it has some)."""
    out = []
    for b in all_batches(db):
        ps = [predictions.get(db, i) for i in b["ids"]]
        counts: dict = {}
        for p in ps:
            s = p["status"] if p else "deleted"
            counts[s] = counts.get(s, 0) + 1
        x = b["extract"]
        if x:
            ec: dict = {}
            for i in x.get("ids", []):
                e = extract.get(db, i)
                s = e["status"] if e else "deleted"
                ec[s] = ec.get(s, 0) + 1
            x = {**x, "counts": ec}
        out.append({**b, "counts": counts, "extract": x})
    return out

