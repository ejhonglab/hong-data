"""Tests of the response calculation entered at upload (the label of its colorbar), which, for
now, does not set a dataset of the same flies apart. Run with:
OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import numpy as np

from app import compare, config, odormeta
from app.db import Database
from app.export import datasets_frame
from app.ingest import load_file
from app.metadata import Metadata, generate_readme, same_response_calc
from tests.test_edit import save, stored  # noqa: F401 (fixtures)
from tests.test_ingest import COLS, ROWS, make_wide, write_csv
from tests.test_tensor import call, main, root, upload  # noqa: F401 (fixtures)

MD = {"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli"}
DFF = r"$\Delta F/F$"


def test_metadata():
    assert Metadata.model_validate({**MD, "response_calc": f"  {DFF} "}).response_calc == DFF
    assert Metadata.model_validate({**MD, "response_calc": " "}).response_calc is None
    readme = generate_readme(Metadata.model_validate({**MD, "response_calc": r"$|\Delta F|/F$"}), {}, {"data": "x.csv"})
    assert r"| Response calculation | $\|\Delta F\|/F$ |" in readme
    assert "Response calculation" not in generate_readme(Metadata.model_validate(MD), {}, {"data": "x.csv"})
    # the same whatever their spacing, but not their case (TeX's)
    assert same_response_calc(r"$\Delta F / F$", r" $\Delta  F / F$") and same_response_calc(None, " ")
    assert not same_response_calc(r"$\Delta F/F$", r"$\delta F/F$") and not same_response_calc(DFF, None)


def test_stored(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    parsed = load_file(write_csv(make_wide(ROWS, COLS), tmp_path))
    insert = lambda md: db.insert_dataset(
        created_at="2026-10-23T10:00:00", title="t", metadata=md, readme_md="#", readme_edited=False, data_file="x.csv",
        data_format=parsed.source_format, data_sha256="0" * 64, odor_yaml_files=[], experiment_xml_file=None,
        experiment_xml_info=None, summary=parsed.summary(), units=parsed.units, trials=parsed.trials, values=parsed.values)
    a, b = insert({**MD, "response_calc": DFF}), insert(MD)
    insert({**MD, "response_calc": "z-score"})
    insert({**MD, "response_calc": DFF})
    assert [d["response_calc"] for d in sorted(db.list_datasets(), key=lambda d: d["id"])] == [DFF, None, "z-score", DFF]
    assert db.get_dataset(a)["response_calc"] == DFF and db.get_dataset(b)["response_calc"] is None
    # for the upload form, the most used first; in downloads (datasets.csv) and Compare
    assert db.options()["response_calcs"] == [DFF, "z-score"]
    frame = datasets_frame([db.get_dataset(a), db.get_dataset(b)])
    assert frame["response_calc"].tolist()[0] == DFF and frame["response_calc"].isna().tolist()[1]
    assert compare.dataset(db, a, tmp_path)["response_calc"] == DFF and compare.dataset(db, b, tmp_path)["response_calc"] is None


def test_same_flies(main, tmp_path, monkeypatch):  # noqa: F811
    # one person's datasets of the same flies, coding unit and manipulations: refused, however
    # their responses were calculated, saying so if only that differs
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "ODOR_SHEET_URL", "")
    rows, _, _ = odormeta.parse_sheet("name,abbrev\nvaleric acid,va\nethyl acetate,ea\n")
    main.db.apply_odor_sync(added=odormeta.diff(rows, [])["added"], changed=[], deleted=[], source="test",
                            fingerprint="0" * 64, details={}, when="2026-10-19T10:00:00")
    values = np.arange(len(ROWS) * len(COLS), dtype=float).reshape(len(ROWS), len(COLS))

    def stage(name, v):
        status, r = upload(main, write_csv(make_wide(ROWS, COLS, v), tmp_path, name))
        assert status == 200, r
        return {"stage_id": r["stage"]["stage_id"]}

    md = lambda calc, **kw: {**MD, "response_calc": calc, **kw}
    readme = lambda st, calc, **kw: call(main.stage_readme, body={"metadata": md(calc, **kw)}, path_params=st)[1]
    commit = lambda st, calc, **kw: call(main.stage_commit, body={"metadata": md(calc, **kw), "readme": "#"}, path_params=st)[1]
    first = commit(stage("dff.csv", values), DFF)
    assert first["ok"] and main.db.get_dataset(first["dataset_id"])["response_calc"] == DFF
    z = stage("zscored.csv", values / 10)
    for r in (readme(z, "z-score"), commit(z, "z-score")):
        assert not r["ok"] and "are exactly those of dataset #1" in r["errors"][0]
        assert f"only their response calculations differ ('z-score' here, '{DFF}' there)" in r["errors"][0]
        assert "cannot be kept apart yet" in r["errors"][0] and "edit its details" not in r["errors"][0]
    # the same calculation (whatever its spacing): refused as before
    r = readme(z, r" $\Delta   F/F$ ")
    assert not r["ok"] and "but for other manipulations" in r["errors"][0] and "response calculations" not in r["errors"][0]
    # other manipulations set them apart, as before
    second = commit(z, "z-score", manipulations=[{"key": "drug", "value": "CNO"}])
    assert second["ok"]
    # a dataset stored without one, said to be editable
    r = readme(stage("again.csv", values + 1), DFF, manipulations=[{"key": "drug", "value": "CNO"}])
    assert f"('{DFF}' here, 'z-score' there)" in r["errors"][0]
    r = readme(stage("none.csv", values + 2), None, manipulations=[{"key": "drug", "value": "CNO"}])
    assert "but for other manipulations" not in r["errors"][0] and "(none given here, 'z-score' there)" in r["errors"][0]
    assert main.db.options()["response_calcs"] == [DFF, "z-score"]


def test_edit(main, stored):  # noqa: F811
    ds_id = stored(response_calc=DFF)
    status, r = save(main, ds_id, {**main.db.get_dataset(ds_id)["metadata"], "response_calc": " z-score "})
    assert status == 200 and r["changed"] == ["response_calc"]
    d = main.db.get_dataset(ds_id)
    assert d["response_calc"] == d["metadata"]["response_calc"] == "z-score" and "| Response calculation | z-score |" in d["readme_md"]
    # one of the same flies, which only that differs from: refused, as at upload
    other = stored(manipulations=[{"key": "drug", "value": "CNO"}])
    status, r = save(main, ds_id, {**d["metadata"], "manipulations": [{"key": "drug", "value": "CNO"}]})
    assert status == 422 and f"exactly those of dataset #{other}" in r["errors"][0]
    assert "('z-score' here, none given there)" in r["errors"][0] and f"(to give #{other} one, edit its details in Browse)" in r["errors"][0]
