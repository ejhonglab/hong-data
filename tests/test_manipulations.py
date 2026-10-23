"""Tests of manipulations entered at upload (what sets a dataset of the same flies apart).
Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from app import config, odormeta
from app.db import Database
from app.export import dataset_stem, datasets_frame
from app.ingest import load_file
from app.metadata import Metadata, dataset_title, generate_readme, manipulations_text, same_manipulations
from tests.test_ingest import COLS, ROWS, make_wide, write_csv
from tests.test_tensor import call, main, root, upload  # noqa: F401 (fixtures)

MD = {"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli"}
CNO = [{"key": "drug", "value": "CNO"}, {"key": "starved", "value": None}]


def test_metadata():
    m = Metadata.model_validate({**MD, "manipulations": [{"key": "  drug  ", "value": " CNO "}, {"key": "starved", "value": ""},
                                                         {"key": "", "value": None}]})
    assert m.manipulations == CNO and manipulations_text(m.manipulations) == "drug=CNO; starved"
    assert dataset_title(m, {"panel_names": ["kiwi"]}) == "kiwi: GH146 > GCaMP6f (glomeruli) [drug=CNO; starved]"
    assert dataset_title(Metadata.model_validate(MD), {"panel_names": ["kiwi"]}) == "kiwi: GH146 > GCaMP6f (glomeruli)"
    assert "| Manipulations | drug=CNO; starved |" in generate_readme(m, {"panel_names": ["kiwi"]}, {"data": "x.csv"})
    for bad, why in [([{"key": "", "value": "CNO"}], "'CNO' needs a name before it"), ([{"key": "a=b"}], "a name cannot have"),
                     ([{"key": "drug", "value": "CNO; 1 mM"}], "a value cannot have"), ([{"key": "Drug"}, {"key": "drug "}], "'drug' is given twice"),
                     ("drug", "expected a list of {key, value}")]:
        with pytest.raises(PydanticValidationError, match=why):
            Metadata.model_validate({**MD, "manipulations": bad})
    # the same in any order, case and spacing
    assert same_manipulations(CNO, [{"key": "Starved", "value": None}, {"key": "drug", "value": " cno"}])
    assert not same_manipulations(CNO, CNO[:1]) and not same_manipulations([{"key": "drug", "value": None}], CNO[:1])
    assert same_manipulations(None, [])


def test_stored(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    parsed = load_file(write_csv(make_wide(ROWS, COLS), tmp_path))
    insert = lambda md: db.insert_dataset(
        created_at="2026-10-23T10:00:00", title="t", metadata=md, readme_md="#", readme_edited=False, data_file="x.csv",
        data_format=parsed.source_format, data_sha256="0" * 64, odor_yaml_files=[], experiment_xml_file=None,
        experiment_xml_info=None, summary=parsed.summary(), units=parsed.units, trials=parsed.trials, values=parsed.values)
    a, b = insert({**MD, "manipulations": CNO}), insert(MD)
    assert [d["manipulations"] for d in sorted(db.list_datasets(), key=lambda d: d["id"])] == [CNO, []]
    assert db.get_dataset(a)["manipulations"] == CNO and db.get_dataset(b)["manipulations"] == []
    insert({**MD, "manipulations": [{"key": "Drug", "value": "cno"}, {"key": "temperature", "value": "30 C"}]})
    o = db.options()
    assert o["manipulation_keys"] == ["drug", "starved", "temperature"] and o["manipulation_values"] == ["30 C", "CNO"]
    # in downloads: datasets.csv, and the names of a dataset's files
    frame = datasets_frame([db.get_dataset(a), db.get_dataset(b)])
    assert frame["manipulations"].tolist()[0] == "drug=CNO; starved" and frame["manipulations"].isna().tolist()[1]
    trials = db.get_matrix_arrays(a)["trials"]
    assert dataset_stem(db.get_dataset(a), trials) == "p1_ann_glomeruli_drug-CNO_starved_2024-01-02"
    assert dataset_stem(db.get_dataset(b), trials) == "p1_ann_glomeruli_2024-01-02"


def test_same_flies(main, tmp_path, monkeypatch):  # noqa: F811
    # one person's datasets of the same flies and coding unit: kept apart if their manipulations differ
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "ODOR_SHEET_URL", "")
    rows, _, _ = odormeta.parse_sheet("name,abbrev\nvaleric acid,va\nethyl acetate,ea\n")
    main.db.apply_odor_sync(added=odormeta.diff(rows, [])["added"], changed=[], deleted=[], source="test",
                            fingerprint="0" * 64, details={}, when="2026-10-19T10:00:00")
    values = np.arange(len(ROWS) * len(COLS), dtype=float).reshape(len(ROWS), len(COLS))

    def stage(name, v, cols=COLS):
        status, r = upload(main, write_csv(make_wide(ROWS, cols, v), tmp_path, name))
        assert status == 200, r
        return {"stage_id": r["stage"]["stage_id"]}

    readme = lambda st, ms: call(main.stage_readme, body={"metadata": {**MD, "manipulations": ms}}, path_params=st)[1]
    commit = lambda st, ms: call(main.stage_commit, body={"metadata": {**MD, "manipulations": ms}, "readme": "#"}, path_params=st)[1]
    first = commit(stage("before.csv", values), [])
    assert first["ok"] and first["title"] == "p1: GH146 > GCaMP6f (glomeruli)"
    after = stage("after.csv", values + 1)
    r = readme(after, [])
    assert not r["ok"] and "are exactly those of dataset #1" in r["errors"][0] and "but for other manipulations" in r["errors"][0]
    r = readme(after, CNO)
    assert r["ok"] and r["warnings"] == [] and r["title"] == "p1: GH146 > GCaMP6f (glomeruli) [drug=CNO; starved]"
    second = commit(after, CNO)
    assert second["ok"] and main.db.get_dataset(second["dataset_id"])["manipulations"] == CNO
    # the same manipulations (in any case and order): refused as for none; some of its flies: a warning
    again = stage("again.csv", values + 2)
    r = readme(again, [{"key": "Starved"}, {"key": "DRUG", "value": "cno"}])
    assert not r["ok"] and f"are exactly those of dataset #{second['dataset_id']}" in r["errors"][0]
    assert readme(again, [{"key": "drug", "value": "CNO"}])["ok"]
    fewer = stage("fewer.csv", values[:, :3] + 3, COLS[:3])
    w = readme(fewer, CNO)["warnings"]
    assert len(w) == 1 and w[0].startswith("possible duplicate: 2 of its 2 flies") and f"dataset #{second['dataset_id']}" in w[0]
    # the same data is refused whatever its manipulations
    status, r = upload(main, write_csv(make_wide(ROWS, COLS[::-1], values[:, ::-1]), tmp_path, "reordered.csv"))
    assert status == 422 and r["errors"][0].startswith("this is the data of dataset #1")
    # Browse's list has them, and the upload form knows them
    assert [d["manipulations"] for d in call(main.datasets_list)[1]["datasets"]] == [CNO, []]
    o = call(main.options)[1]
    assert o["manipulation_keys"] == ["drug", "starved"] and o["manipulation_values"] == ["CNO"]
