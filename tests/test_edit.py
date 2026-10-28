"""Tests of editing a dataset's metadata as entered at upload (Browse: Edit). Run with:
OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import json

import pytest

from app import config
from app.ingest import load_file
from tests.test_ingest import COLS, ROWS, make_wide, write_csv
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)

MD = {"uploader_name": "me", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli", "panel_names": ["p1"],
      "raw_data_path": "tom/raw-gone", "notes": "first"}


@pytest.fixture
def stored(main, tmp_path, monkeypatch):  # noqa: F811
    """A function storing a dataset as an upload does (its files too): its id."""
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")

    def add(cols=COLS, readme_edited=False, **md):
        parsed = load_file(write_csv(make_wide(ROWS, cols), tmp_path, f"x{len(main.db.list_datasets())}.csv"))
        ds_id = main.db.insert_dataset(
            created_at="2026-01-02T03:04:05", title="t", metadata={**MD, **md}, readme_md="# by hand\n", readme_edited=readme_edited,
            data_file="x.csv", data_format=parsed.source_format, data_sha256=str(len(main.db.list_datasets())) * 64,
            odor_yaml_files=[], experiment_xml_file=None, experiment_xml_info=None, summary=parsed.summary(), units=parsed.units,
            trials=parsed.trials, values=parsed.values)
        folder = config.DATASETS_DIR / str(ds_id)
        folder.mkdir(parents=True)
        (folder / "README.md").write_text("# by hand\n")
        (folder / "metadata.json").write_text(json.dumps({"dataset_id": ds_id, "metadata": {**MD, **md}, "tensor_sources": {}}))
        return ds_id
    return add


def save(main, ds_id, metadata, base=None):  # noqa: F811
    base = main.db.get_dataset(ds_id)["metadata"] if base is None else base
    return call(main.dataset_metadata_save, body={"metadata": metadata, "base": base}, path_params={"ds_id": ds_id})


def test_edit(main, stored):  # noqa: F811
    ds_id = stored()
    edit = {**MD, "driver": "pebbled", "panel_names": "p1, p2", "manipulations": [{"key": "drug", "value": "CNO"}],
            "age_min_days": 2, "age_max_days": 4, "power_mw": 10, "power_measured_same_day": True, "notes": "second"}
    status, r = save(main, ds_id, edit)
    assert status == 200 and r["readme"] == "generated"
    assert set(r["changed"]) == {"driver", "panel_names", "manipulations", "age_min_days", "age_max_days", "power_mw",
                                 "power_measured_same_day", "notes"}
    # its columns, title, panels and README follow; its metadata.json says what changed
    d = main.db.get_dataset(ds_id)
    assert (d["driver"], d["panel_names"], d["power_measured_same_day"], d["notes"]) == ("pebbled", "p1, p2", 1, "second")
    assert d["title"] == "p1, p2: pebbled > GCaMP6f (glomeruli) [drug=CNO]" and d["metadata"]["age_max_days"] == 4
    assert [(p["panel"], p["source"]) for p in d["panels"]] == [("p1", "form"), ("p2", "form")]
    assert d["readme_md"].startswith(f"# {d['title']}") and not d["readme_edited"]
    folder = config.DATASETS_DIR / str(ds_id)
    assert (folder / "README.md").read_text() == d["readme_md"]
    saved = json.loads((folder / "metadata.json").read_text())
    assert saved["title"] == d["title"] and saved["metadata"]["driver"] == "pebbled"
    assert saved["edits"][0]["changed"]["driver"] == ["GH146", "pebbled"]
    assert [x["title"] for x in main.db.list_datasets()] == [d["title"]]
    # nothing changed: nothing saved
    status, r = save(main, ds_id, {**edit, "panel_names": ["p1", "p2"]})
    assert status == 200 and r["changed"] == [] and len(json.loads((folder / "metadata.json").read_text())["edits"]) == 1


def test_edit_refused(main, stored):  # noqa: F811
    ds_id = stored()
    # changed on the server since; for no dataset; not as expected
    assert save(main, ds_id, {**MD, "notes": "x"}, base={**MD, "notes": "older"})[0] == 409
    assert save(main, ds_id + 1, MD, base=MD)[0] == 404
    assert call(main.dataset_metadata_save, body={"metadata": MD}, path_params={"ds_id": ds_id})[0] == 400
    # checked as an upload's
    status, r = save(main, ds_id, {**MD, "driver": " "})
    assert status == 422 and r["errors"] == ["driver is required"]
    assert "max must be >= min" in save(main, ds_id, {**MD, "age_min_days": 4, "age_max_days": 2})[1]["errors"][0]
    # data paths: those changed checked on tensor (a path not found there, entered before, kept)
    status, r = save(main, ds_id, {**MD, "mocorr_data_path": "tom/nowhere"})
    assert status == 422 and not r["checks"]["mocorr_data_path"]["ok"]
    status, r = save(main, ds_id, {**MD, "mocorr_data_path": " tom//mocorr/ ", "notes": "x"})
    assert status == 200 and main.db.get_dataset(ds_id)["mocorr_data_path"] == "tom/mocorr"
    assert main.db.get_dataset(ds_id)["raw_data_path"] == "tom/raw-gone"
    # one person's datasets of a coding unit cannot have the same flies, but for other manipulations
    other = stored(manipulations=[{"key": "drug", "value": "CNO"}])
    status, r = save(main, ds_id, {**MD, "manipulations": [{"key": "drug", "value": "CNO"}]})
    assert status == 422 and f"exactly those of dataset #{other}" in r["errors"][0]
    assert save(main, ds_id, {**MD, "uploader_name": "you", "manipulations": [{"key": "drug", "value": "CNO"}]})[0] == 200
    # KCs numbered with gaps, as at upload
    kcs = stored(cols=[("2024-01-01", 1, "1"), ("2024-01-01", 1, "3")])
    status, r = save(main, kcs, {**MD, "coding_unit": "KCs"})
    assert status == 422 and "2" in r["errors"][0]


def test_edit_readme_by_hand(main, stored):  # noqa: F811
    ds_id = stored(readme_edited=True)
    status, r = save(main, ds_id, {**MD, "genotype": "w;+;+"})
    assert status == 200 and r == {"ok": True, "changed": ["genotype"], "readme": "kept"}
    assert main.db.get_dataset(ds_id)["readme_md"] == "# by hand\n"
    assert (config.DATASETS_DIR / str(ds_id) / "README.md").read_text() == "# by hand\n"
