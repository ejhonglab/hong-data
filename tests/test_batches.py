"""Tests of batches of predictions (batches.py): the model's odors a fly lacks, a batch asked for and followed, its signals
extracted and joined into a temporary dataset, and that dataset deleted by whoever asked for it."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app import batches, config, extract, fits, predictions, roi_training
from tests.test_fits import GLOMS, NAMES, ODORS, call, data, tensor  # noqa: F401 (fixtures)
from tests.test_tensor import main, root  # noqa: F401 (fixtures)

FLIES = [("ann/analysis/2024-01-01/1", "ann/raw/2024-01-01/1"), ("ann/analysis/2024-01-01/2", "ann/raw/2024-01-01/2")]
NAMED = [f"{NAMES[a]} @ {c}" for a, c in ODORS]


def model_fit(db, odors):
    f = fits.create(db, "Ann", None, fits.options({"held_out": 1}))
    fits.update(db, f["id"], status="done", finished_at=fits.now(), flies={"train": ["ann_2024-01-01_3"], "held_out": ["ann_2024-01-01_1"],
                                                                          "datasets": [1], "odors": odors})
    (fits.fit_dir(f["id"]) / "model").mkdir(parents=True)
    (fits.fit_dir(f["id"]) / "model" / "model.json").write_text(json.dumps({"odors": odors, "aliases": {}, "options": {"image_odors": odors[:2]}}))
    return fits.get(db, f["id"])


def test_missing(main, tensor):  # noqa: F811
    from app import odormeta

    f = model_fit(main.db, [*NAMED, "geosmin @ -5"])
    index = odormeta.OdorIndex(main.db.list_odors())
    r = predictions.fly_missing(f["id"], tensor / FLIES[0][0], index)
    assert r == {"odors": 5, "missing": ["geosmin @ -5"], "missing_image_odors": [], "recordings": ["diagnostics1"]}
    status, r = call(main.predictions_missing, query={"fit": f["id"], "dir": FLIES[0][0]})
    assert status == 200 and r["missing"] == ["geosmin @ -5"]
    assert call(main.predictions_missing, query={"fit": f["id"], "dir": "ann/nowhere"})[0] == 400
    assert call(main.predictions_missing, query={"fit": 999, "dir": FLIES[0][0]})[0] == 400
    # its file in a prediction's folder
    predictions.write_missing(tensor, [*NAMED, "geosmin @ -5"], NAMED, NAMED[:2])
    rows = pd.read_csv(tensor / predictions.MISSING)
    assert list(rows.columns) == ["odor", "fly_has_it", "in_cellpose_images"] and rows.fly_has_it.tolist() == ["yes"] * 4 + ["no"]


def test_batch(main, tensor):  # noqa: F811
    f = model_fit(main.db, NAMED)
    one = predictions.create(main.db, f["id"], predictions.locate(*FLIES[0], config.TENSOR_ROOT, []), "Ann", None)[0]
    predictions.update(main.db, one["id"], status="done")
    ask = lambda **body: call(main.predictions_batch, method="POST", body={"name": "Bob", "fit": f["id"], **body})  # noqa: E731
    assert ask(flies="x")[0] == 400 and ask(flies=[])[0] == 400
    status, b = ask(flies=[{"dir": d, "raw_dir": r} for d, r in FLIES] + [{"dir": "ann/nowhere", "raw_dir": "x"}], clean=["margin"])
    assert status == 201 and len(b["ids"]) == 2 and b["ids"][0] == one["id"] and b["left_out"][0]["dir"] == "ann/nowhere"
    # (the one done before kept as it was; the other queued, cleaned as asked)
    assert predictions.get(main.db, one["id"])["status"] == "done" and predictions.get(main.db, b["ids"][1])["options"]["clean"] == ["margin"]
    listed = call(main.predictions_list)[1]["batches"]
    assert listed[0]["id"] == b["id"] and listed[0]["counts"] == {"done": 1, "queued": 1}
    # one that failed is asked for again
    predictions.update(main.db, b["ids"][1], status="failed")
    b2 = ask(flies=[{"dir": d, "raw_dir": r} for d, r in FLIES])[1]
    assert predictions.get(main.db, b2["ids"][1])["status"] == "queued"
    assert call(main.predictions_batch_delete, method="DELETE", path_params={"id": b2["id"]})[1] == {"deleted": True}
    assert call(main.predictions_batch_delete, method="DELETE", path_params={"id": b2["id"]})[0] == 404


def responses_table(fly_num: int, gloms: list[str], two_odors: bool = False) -> pd.DataFrame:
    """A responses_glomeruli.csv as an extraction writes it."""
    odors = [("diag", False, o, *(["solvent"] if two_odors else []), r) for o in NAMED for r in (0, 1)]
    rows = pd.MultiIndex.from_tuples(odors, names=["panel", "is_pair", "odor1", *(["odor2"] if two_odors else []), "repeat"])
    cols = pd.MultiIndex.from_tuples([("2024-01-01", fly_num, g) for g in gloms], names=["date", "fly_num", "roi"])
    return pd.DataFrame(np.arange(len(rows) * len(cols), dtype=float).reshape(len(rows), len(cols)) / 10 + fly_num, index=rows, columns=cols)


def test_joined_dataset(main, tensor, monkeypatch):  # noqa: F811
    db = main.db
    f = model_fit(db, NAMED)
    b = batches.create(db, f["id"], [predictions.locate(d, r, config.TENSOR_ROOT, []) for d, r in FLIES], "Bob", None, predictions.options({}))
    with pytest.raises(ValueError, match="none of its predictions is done"):
        batches.extract_all(db, b["id"], "predicted", extract.options({}), "Bob", None, config.TENSOR_ROOT, [])
    import tifffile

    for d, _ in FLIES:  # (a movie in each recording, as extracting needs)
        tifffile.imwrite(tensor / d / "diagnostics1" / "mocorr.tif", np.zeros((4, 3, 24, 30), np.uint16), imagej=True, metadata={"axes": "TZYX"})
    for i in b["ids"]:
        p = predictions.get(db, i)
        predictions.update(db, i, status="done")
        predictions.pred_dir(i).mkdir(parents=True, exist_ok=True)
        predictions.roiset(p).write_bytes(b"")
    status, out = call(main.extractions_batch, method="POST", body={"name": "Bob", "batch": b["id"], "rois": "edited"})
    assert status == 201 and len(out["extract"]["ids"]) == 2 and out["extract"]["status"] == "running" and out["extract"]["rois"] == "edited"
    assert call(main.extractions_batch, method="POST", body={"name": "Bob", "batch": b["id"]})[0] == 400  # (once at a time)
    assert call(main.extractions_batch, method="POST", body={"name": "Bob", "batch": 999})[0] == 404
    xs = [extract.get(db, i) for i in out["extract"]["ids"]]
    assert all(x["rois"]["kind"] == "predicted" for x in xs)  # (none was edited)
    assert batches.finish(db) == []  # (not over yet)
    # each fly's table (one with an odor more level), as the worker leaves them
    for x, (n, gl, two) in zip(xs, [(1, GLOMS[:3], False), (2, GLOMS[1:], True)]):
        extract.out_dir(x["id"]).mkdir(parents=True, exist_ok=True)
        responses_table(n, gl, two).to_csv(extract.out_dir(x["id"]) / "responses_glomeruli.csv")
        extract.update(db, x["id"], status="done")
    table = batches.joined_table([extract.out_dir(x["id"]) / "responses_glomeruli.csv" for x in xs])
    assert table.shape == (8, 6) and table.index.names == ["panel", "is_pair", "odor1", "odor2", "repeat"]
    made = batches.finish(db)
    assert len(made) == 1, json.dumps(batches.get(db, b["id"])["extract"])
    assert batches.finish(db) == []
    d = db.get_dataset(made[0])
    t = batches.temporary_of(d)
    assert t["by"] == "Bob" and t["batch"] == b["id"] and d["title"].startswith("[temporary]") and d["n_flies"] == 2 and d["n_units"] == 6
    assert d["uploader_name"] == "Bob" and (config.DATASETS_DIR / str(made[0]) / f"batch_{b['id']}_responses.csv").is_file()
    listed = {x["id"]: x for x in call(main.datasets_list)[1]["datasets"]}
    assert listed[made[0]]["temporary"]["by"] == "Bob" and all(x.get("temporary") is None for i, x in listed.items() if i != made[0])
    assert batches.get(db, b["id"])["extract"]["dataset_id"] == made[0]
    # not a dataset whose flies the AL segmentation tab lists
    assert all(f["datasets"][0]["id"] != made[0] for f in roi_training.inventory(db, config.TENSOR_ROOT, [])["flies"])
    # deleted only by whoever asked for it (or the Admin tab), and only a temporary one
    delete = lambda i, name: call(main.dataset_temporary_delete, method="DELETE", path_params={"ds_id": i}, query={"name": name})  # noqa: E731
    assert delete(made[0], "Cy")[0] == 403 and delete(made[0], "")[0] == 403 and delete(1, "Ann")[0] == 403 and delete(999, "Bob")[0] == 404
    assert delete(made[0], "bob")[0] == 200 and db.get_dataset(made[0]) is None
