"""Tests of the response calculation entered at upload (the label of its colorbar), which sets
the same recordings calculated otherwise apart: datasets of the same flies, each saying how
it was calculated. Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import numpy as np

from app import compare, config, odormeta
from app.db import Database
from app.export import dataset_stem, datasets_frame
from app.ingest import load_file
from app.metadata import Metadata, dataset_title, generate_readme, other_calculation, same_response_calc
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
    # another calculation of the same recordings: each says how, and not alike
    assert other_calculation(DFF, "z-score") and other_calculation(r"$\Delta F/F$", r"$\delta F/F$")
    assert not other_calculation(DFF, f" {DFF}") and not other_calculation(DFF, None) and not other_calculation(" ", None)
    # in its title (as written, spaces collapsed) and its downloads' names (ASCII)
    m = Metadata.model_validate({**MD, "response_calc": "baseline-noise   units", "manipulations": [{"key": "TNT"}]})
    assert dataset_title(m, {"panel_names": ["APL16"]}) == "APL16: GH146 > GCaMP6f (glomeruli) [TNT], baseline-noise units"
    assert dataset_title(Metadata.model_validate(MD), {"panel_names": ["APL16"]}) == "APL16: GH146 > GCaMP6f (glomeruli)"
    d = {"id": 1, "uploader_name": "Yang", "coding_unit": "KCs", "last_fly_date": "2024-08-14", "metadata": {},
         "manipulations": [{"key": "TNT", "value": None}]}
    trials = [{"panel": "APL16"}]
    assert dataset_stem({**d, "response_calc": "whole-recording z"}, trials) == "APL16_yang_KCs_TNT_whole-recording-z_2024-08-14"
    assert dataset_stem({**d, "response_calc": DFF}, trials) == "APL16_yang_KCs_TNT_Delta-FF_2024-08-14"
    assert dataset_stem({**d, "response_calc": None}, trials) == dataset_stem(d, trials) == "APL16_yang_KCs_TNT_2024-08-14"


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
    # one person's datasets of the same flies, coding unit and manipulations: refused, but for the
    # same recordings calculated otherwise, as many ways as they are, each saying how
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "ODOR_SHEET_URL", "")
    rows, _, _ = odormeta.parse_sheet("name,abbrev\nvaleric acid,va\nethyl acetate,ea\n")
    main.db.apply_odor_sync(added=odormeta.diff(rows, [])["added"], changed=[], deleted=[], source="test",
                            fingerprint="0" * 64, details={}, when="2026-10-19T10:00:00")
    values = np.arange(len(ROWS) * len(COLS), dtype=float).reshape(len(ROWS), len(COLS))
    cno = [{"key": "drug", "value": "CNO"}]

    def stage(name, v, cols=COLS):
        status, r = upload(main, write_csv(make_wide(ROWS, cols, v), tmp_path, name))
        assert status == 200, r
        return {"stage_id": r["stage"]["stage_id"]}

    md = lambda calc, **kw: {**MD, "response_calc": calc, **kw}
    readme = lambda st, calc, **kw: call(main.stage_readme, body={"metadata": md(calc, **kw)}, path_params=st)[1]
    commit = lambda st, calc, **kw: call(main.stage_commit, body={"metadata": md(calc, **kw), "readme": "#"}, path_params=st)[1]
    first = commit(stage("dff.csv", values), DFF)
    assert first["ok"] and first["title"] == f"p1: GH146 > GCaMP6f (glomeruli), {DFF}"
    z = stage("zscored.csv", values / 10)
    r = readme(z, "z-score")
    assert r["ok"] and r["warnings"] == [] and r["title"] == "p1: GH146 > GCaMP6f (glomeruli), z-score"
    second = commit(z, "z-score")
    assert second["ok"] and main.db.get_dataset(second["dataset_id"])["response_calc"] == "z-score"
    third = commit(stage("raw.csv", values * 7), "raw")
    assert third["ok"]
    ids = [first["dataset_id"], second["dataset_id"], third["dataset_id"]]
    assert main.db.calculations() == [ids]
    assert call(main.dataset_get, path_params={"ds_id": ids[0]})[1]["calculations"] == [
        {"id": ids[1], "title": "p1: GH146 > GCaMP6f (glomeruli), z-score", "response_calc": "z-score"},
        {"id": ids[2], "title": "p1: GH146 > GCaMP6f (glomeruli), raw", "response_calc": "raw"}]
    # the same calculation as one of them (whatever its spacing): refused
    r = readme(stage("again.csv", values + 1), " z-score ")
    assert not r["ok"] and len(r["errors"]) == 1 and f"are exactly those of dataset #{ids[1]}" in r["errors"][0]
    assert f"or as the same recordings calculated otherwise, each saying how (this one's response calculation is 'z-score', as #{ids[1]}'s)" in r["errors"][0]
    # none given beside them: refused, saying to enter one
    r = readme(stage("none.csv", values + 2), None)
    assert not r["ok"] and len(r["errors"]) == 3 and f"other than #{ids[0]}'s '{DFF}'" in r["errors"][0]
    # one stored without one: another of its flies calculated otherwise is refused until it has one
    plain = commit(stage("plain.csv", values + 3), None, manipulations=cno)
    assert plain["ok"] and plain["title"] == "p1: GH146 > GCaMP6f (glomeruli) [drug=CNO]"
    p = plain["dataset_id"]
    r = readme(stage("cno_z.csv", values + 4), "z-score", manipulations=cno)
    assert not r["ok"] and f"and #{p} does not (to give #{p} one, edit its details in Browse)" in r["errors"][0]
    r = readme(stage("cno_none.csv", values + 5), None, manipulations=cno)
    assert not r["ok"] and "but for other manipulations" in r["errors"][0] and f"(under Response calculation; to give #{p} one" in r["errors"][0]
    # some of their flies: warned of once for the recordings, naming their other calculations
    w = readme(stage("fewer.csv", values[:, :3] + 6, COLS[:3]), "dF")["warnings"]
    assert len(w) == 1 and w[0].startswith("possible duplicate: 2 of its 2 flies")
    assert f"also in dataset #{ids[0]} (p1: GH146 > GCaMP6f (glomeruli), {DFF}), uploaded by Ann on " in w[0]
    assert f"(and its other calculations, #{ids[1]}, #{ids[2]}), which has 1 more" in w[0]
    assert main.db.options()["response_calcs"] == [DFF, "raw", "z-score"]


def test_edit(main, stored):  # noqa: F811
    ds_id = stored(response_calc=DFF)
    status, r = save(main, ds_id, {**main.db.get_dataset(ds_id)["metadata"], "response_calc": " z-score "})
    assert status == 200 and r["changed"] == ["response_calc"]
    d = main.db.get_dataset(ds_id)
    assert d["response_calc"] == d["metadata"]["response_calc"] == "z-score" and "| Response calculation | z-score |" in d["readme_md"]
    assert d["title"] == "p1: GH146 > GCaMP6f (glomeruli), z-score" and d["readme_md"].startswith(f"# {d['title']}\n")
    # another calculation of its recordings: refused the same calculation, or none (for either)
    other = stored(response_calc=DFF)
    status, r = save(main, other, {**main.db.get_dataset(other)["metadata"], "response_calc": "z-score"})
    assert status == 422 and f"exactly those of dataset #{ds_id}" in r["errors"][0] and f"is 'z-score', as #{ds_id}'s" in r["errors"][0]
    status, r = save(main, ds_id, {**d["metadata"], "response_calc": None})
    assert status == 422 and f"exactly those of dataset #{other}" in r["errors"][0] and f"other than #{other}'s '{DFF}'" in r["errors"][0]
    assert save(main, other, {**main.db.get_dataset(other)["metadata"], "response_calc": "raw"})[0] == 200
    # one of the same flies without one, beside which it would be: refused until that has one
    cno = stored(manipulations=[{"key": "drug", "value": "CNO"}])
    status, r = save(main, ds_id, {**d["metadata"], "manipulations": [{"key": "drug", "value": "CNO"}]})
    assert status == 422 and f"exactly those of dataset #{cno}" in r["errors"][0]
    assert f"and #{cno} does not (to give #{cno} one, edit its details in Browse)" in r["errors"][0]
