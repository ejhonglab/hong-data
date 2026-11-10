"""Tests of model data for the Models tab (model_data.py) and its API: folders of model outputs
on tensor, their params.csv and provenance.json as a table of models, spike count files
matched to them, and a run read. Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/."""
from __future__ import annotations

import base64
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from app import model_data, odormeta
from tests.test_tensor import call, main, post, root  # noqa: F401 (fixtures)

ODORS = ["2h @ -3", "eb @ -3", "nope @ -2", "pa @ -3"]
PROVENANCE = {"generated": "2026-11-05T03:11:26+00:00", "script": "E1.py", "claw_thr": 70.0, "inhadd": [1, 2], "empty": {},
              "base": {"n_spikes_for_response": 2, "prat_claws": True},
              "runs": {"k0": {"seconds": 5}, "k0_x": {"seconds": 7, "host": "hal"}, "note": "x"}}  # (each model's, by its tag)


def counts(n_kcs: int, transposed: bool = False, types: bool = True) -> pd.DataFrame:
    """A run's spike counts as the models write them: KCs (kc_id, kc_type) × odors; the first KC
    spiking to none, the second once, the rest more."""
    v = np.arange(n_kcs * len(ODORS), dtype=float).reshape(n_kcs, len(ODORS)) % 5
    v[0] = 0
    v[1] = [1, 0, 0, 0]
    ix = (pd.MultiIndex.from_arrays([range(100, 100 + n_kcs), ["ab" if k % 2 else "g" for k in range(n_kcs)]], names=["kc_id", "kc_type"])
          if types else pd.Index(range(100, 100 + n_kcs), name="kc_id"))
    df = pd.DataFrame(v, index=ix, columns=pd.Index(ODORS, name="odor"))
    if transposed:
        df = df.T
        df.columns = df.columns.astype(str)  # (parquet's column names are text)
    return df


@pytest.fixture
def outputs(root):  # noqa: F811
    """Two experiments on tensor (tom/model_outputs), laid out as the KC-remapping experiments are."""
    top = root / "tom/model_outputs"
    e1, e2 = top / "E1_apl", top / "E2_gain"
    for d in (e1 / "sub", e1 / "probe", e2):
        d.mkdir(parents=True)
    pd.DataFrame([
        {"source_table": "s.csv", "tag": "k0", "variant": "k0", "base": "claw", "n_kcs": "6", "claw_thr": "", "rate_on": "0.05", "seed": "1"},
        {"source_table": "s.csv", "tag": "k0", "variant": "k0", "base": "claw-dynamics", "n_kcs": "6", "claw_thr": "40", "rate_on": "0.1", "seed": "2"},
        {"source_table": "s.csv", "tag": "k0_x", "variant": "k0_x", "base": "claw", "n_kcs": "6", "claw_thr": "", "rate_on": "nan", "seed": "3"},
        {"source_table": "s.csv", "tag": "gone", "variant": "gone", "base": "claw", "n_kcs": "6", "claw_thr": "", "rate_on": "", "seed": "True"},
    ]).to_csv(e1 / "params.csv", index=False)
    (e1 / "provenance.json").write_text(json.dumps(PROVENANCE))
    for name in ("claw_sp0.05_k0_on", "claw_sp0.05_k0_acute_off", "claw-dynamics_sp0.05_k0_on", "claw_sp0.05_k0_x_on", "sub/claw_sp0.05_k0_on"):
        counts(6).to_parquet(e1 / f"{name}_spike_counts.parquet")
    (e1 / "probe/provenance.json").write_text("{}")  # a run of its own: its files are of none of E1's models
    counts(6).to_parquet(e1 / "probe/k0_on_spike_counts.parquet")
    counts(6).to_parquet(e1 / "claw_sp0.05_k0_on_responses.parquet")  # (not spike counts)
    pd.DataFrame([{"tag": "f2", "variant": "f2", "base": "bouton", "wAPLKC": "0.25"}]).to_csv(e2 / "params.csv", index=False)
    (e2 / "provenance.json").write_text(json.dumps({"generated": "2026-11-04T00:00:00+00:00", "script": "E2.py",
                                                     "bases": {"bouton": {"n_spikes_for_response": 3}}}))
    counts(5, transposed=True, types=False).to_parquet(e2 / "f2_on_spike_counts.parquet")
    counts(5).to_parquet(top / "stray_on_spike_counts.parquet")  # of no experiment
    return top


@pytest.fixture
def odor_table(main):  # noqa: F811
    rows, _, _ = odormeta.parse_sheet("name,abbrev,cid\n2-heptanone,2h,8051\nethyl butyrate,eb,7762\npentyl acetate,pa,12348\n")
    main.db.apply_odor_sync(added=odormeta.diff(rows, [])["added"], changed=[], deleted=[], source="test",
                            fingerprint="0" * 64, details={}, when="2026-11-05T00:00:00")
    return main


def test_values():
    assert [model_data._value(t) for t in ("", "True", "False", "6", "-2", "0.05", "1e3", "nan", "inf", "k0")] == \
        [None, True, False, 6, -2, 0.05, 1000.0, "nan", "inf", "k0"]
    assert model_data.flatten(PROVENANCE) == {
        "generated": PROVENANCE["generated"], "script": "E1.py", "claw_thr": 70.0, "inhadd": "[1, 2]", "empty": "{}",
        "base->n_spikes_for_response": 2, "base->prat_claws": True, "runs->k0->seconds": 5, "runs->k0_x->seconds": 7,
        "runs->k0_x->host": "hal", "runs->note": "x"}
    assert model_data.parse_list(" a, b ,,a,c ") == ["a", "b", "c"]


def test_scan(outputs):
    s = model_data.scan(outputs)
    assert model_data.summary(s) == {"experiments": 2, "models": 5, "runs": 6, "unmatched": 2}
    assert [e["key"] for e in s["experiments"]] == ["E1_apl", "E2_gain"]
    runs = {(m["experiment"], m["params"]["tag"], m["params"]["base"]): [(r["label"], r["prefix"]) for r in m["runs"]] for m in s["models"]}
    assert runs == {
        # the longest name, and of models named alike, the one whose base begins the file's name; its folders' after its own
        ("E1_apl", "k0", "claw"): [("on", "claw_sp0.05"), ("acute_off", "claw_sp0.05"), ("sub/on", "claw_sp0.05")],
        ("E1_apl", "k0", "claw-dynamics"): [("on", "claw-dynamics_sp0.05")],
        ("E1_apl", "k0_x", "claw"): [("on", "claw_sp0.05")],
        ("E1_apl", "gone", "claw"): [],
        ("E2_gain", "f2", "bouton"): [("on", "")],
    }
    assert s["unmatched"] == ["E1_apl/probe/k0_on_spike_counts.parquet", "stray_on_spike_counts.parquet"]
    m = s["models"]
    assert m[0]["params"] == {"source_table": "s.csv", "tag": "k0", "variant": "k0", "base": "claw", "n_kcs": 6, "claw_thr": None,
                              "rate_on": 0.05, "seed": 1}
    assert m[2]["params"]["rate_on"] == "nan" and m[3]["params"]["seed"] is True
    # provenance.json's keys under a model's name: of it alone (named with * for it), not of all its experiment's models
    assert [x["own"] for x in m] == [{"runs->*->seconds": 5}, {"runs->*->seconds": 5}, {"runs->*->seconds": 7, "runs->*->host": "hal"}, {},
                                     {"bases->*->n_spikes_for_response": 3}]
    e1, e2 = s["experiments"]
    assert "runs->note" in e1["shared"] and not any(k.startswith("runs->k0") for k in e1["shared"]) and "runs->k0->seconds" in e1["flat"]
    assert e1["keys"][-2:] == ["runs->*->seconds", "runs->*->host"] and e2["shared"] == {"generated": "2026-11-04T00:00:00+00:00", "script": "E2.py"}
    assert [x["n_spikes"] for x in m] == [{"value": 2, "from": "base->n_spikes_for_response in provenance.json"}] * 4 + [
        {"value": 3, "from": "bases->bouton->n_spikes_for_response in provenance.json"}]
    # made again only when its files change
    assert model_data.scan(outputs) is s
    counts(6).to_parquet(outputs / "E1_apl/gone_on_spike_counts.parquet")
    s2 = model_data.scan(outputs)
    assert s2 is not s and [r["label"] for r in s2["models"][3]["runs"]] == ["on"]


def test_columns(outputs):
    keys = model_data.scan(outputs)["keys"]
    by = {k["name"]: k["files"] for k in keys}
    assert by["tag"] == 2 and by["generated"] == 2 and by["claw_thr"] == 2 and by["seed"] == 1  # (of 2 params.csv and 2 provenance.json)
    cols = model_data.columns(keys, {"n_kcs", "nope"}, ["base", "missing"], ["tag", "variant"])
    assert cols[:2] == ["generated", "base"] and cols[-2:] == ["tag", "variant"]
    assert "n_kcs" not in cols and "missing" not in cols
    middle = cols[2:-2]
    # those in more files first, ties in the order they first come (params.csv's columns, then provenance.json's keys)
    assert middle[:2] == ["claw_thr", "script"]  # (in 2 files each; base and tag placed already)
    assert middle.index("source_table") < middle.index("seed") < middle.index("inhadd") < middle.index("base->n_spikes_for_response")
    assert "runs->*->seconds" in middle and "bases->*->n_spikes_for_response" in middle and "runs->k0->seconds" not in middle


def test_read_run(outputs, odor_table):
    index = odormeta.OdorIndex(odor_table.db.list_odors())
    r = model_data.read_run(outputs / "E1_apl/claw_sp0.05_k0_on_spike_counts.parquet", index)
    assert r["kcs"] == {"labels": [str(k) for k in range(100, 106)], "groups": ["g", "ab"] * 3, "levels": ["kc_id", "kc_type"]}
    assert [o["odor"] for o in r["odors"]] == ODORS and not r["transposed"]
    assert [o["chem"]["name"] if o["chem"] else None for o in r["odors"]] == ["2-heptanone", "ethyl butyrate", None, "pentyl acetate"]
    assert [o["log10_conc"] for o in r["odors"]] == [-3, -3, -2, -3]
    v = np.frombuffer(base64.b64decode(r["values_b64"]), dtype=np.float32).reshape(6, 4)
    assert np.array_equal(v, counts(6).to_numpy(dtype=np.float32))
    t = model_data.read_run(outputs / "E2_gain/f2_on_spike_counts.parquet", index)  # (odors × KCs in its file)
    assert t["transposed"] and t["kcs"]["groups"] is None and len(t["kcs"]["labels"]) == 5 and len(t["odors"]) == 4


def test_api(main, outputs, odor_table):  # noqa: F811
    status, out = call(main.models_list)
    assert status == 200 and out["sources"] == [] and out["models"] == []
    for body, want in [({"path": "/abs/path"}, "absolute"), ({"path": "tom/nothing"}, "not found"),
                       ({"path": "tom/raw"}, "no params.csv"), ({}, "")]:
        status, out = call(main.admin_models_add, body=body)
        assert status == 400 and want in out["error"], (body, out)
    status, added = call(main.admin_models_add, body={"path": "tom//model_outputs/", "hidden": "n_kcs, seed", "first": "base"})
    assert status == 200 and added["path"] == "tom/model_outputs"
    assert added["scan"] == {"experiments": 2, "models": 5, "runs": 6, "unmatched": 2}
    assert (added["hidden"], added["first"], added["last"]) == ("n_kcs,seed", "base", model_data.DEFAULTS["last"])
    assert call(main.admin_models_add, body={"path": "tom/model_outputs"})[0] == 409
    status, out = call(main.models_list)
    assert status == 200 and [s["ok"] for s in out["sources"]] == [True]
    assert out["columns"][:2] == ["generated", "base"] and "seed" not in out["columns"] and "n_kcs" not in out["columns"]
    assert out["columns"][-4:] == ["tag", "variant", "rate_on", "source_table"]  # (the default last but rate_off, which none has)
    sid = added["id"]
    assert set(out["experiments"]) == {f"{sid}/E1_apl", f"{sid}/E2_gain"}
    e1 = out["experiments"][f"{sid}/E1_apl"]
    assert e1["flat"]["claw_thr"] == 70.0 and "claw_thr" in e1["params"]  # (a row's own value over its experiment's: in the browser)
    assert "runs->k0->seconds" not in e1["flat"] and out["models"][2]["own"] == {"runs->*->seconds": 7, "runs->*->host": "hal"}
    run_id = out["models"][0]["runs"][0]["id"]
    assert run_id == "tom/model_outputs/E1_apl/claw_sp0.05_k0_on_spike_counts.parquet"
    assert out["unmatched"] == ["tom/model_outputs/E1_apl/probe/k0_on_spike_counts.parquet", "tom/model_outputs/stray_on_spike_counts.parquet"]

    status, run = call(main.models_run, params={"id": run_id})
    assert status == 200 and run["file"] == "tom/model_outputs/E1_apl/claw_sp0.05_k0_on_spike_counts.parquet"
    assert run["provenance"] == PROVENANCE and run["odors"][0]["odor_id"] is not None
    for bad in ("", "tom/model_outputs/stray_on_spike_counts.parquet", "tom/model_outputs/E1_apl/probe/k0_on_spike_counts.parquet",
                "tom/model_outputs/../../etc/passwd", "tom/model_outputs/E1_apl/../E1_apl/claw_sp0.05_k0_on_spike_counts.parquet",
                "tom/other/E1_apl/claw_sp0.05_k0_on_spike_counts.parquet", "tom/model_outputs/E1_apl/claw_sp0.05_k0_on_responses.parquet"):
        assert call(main.models_run, params={"id": bad})[0] == 404, bad

    status, changed = call(main.admin_models_update, body={"hidden": "", "last": "tag"}, path_params={"id": sid})
    assert status == 200 and (changed["hidden"], changed["first"], changed["last"]) == ("", "base", "tag")
    assert call(main.admin_models_update, body={}, path_params={"id": sid + 1})[0] == 404
    cols = call(main.models_list)[1]["columns"]
    assert "n_kcs" in cols and cols[-1] == "tag"
    assert call(main.admin_models_delete, path_params={"id": sid})[1] == {"deleted": True}
    assert call(main.admin_models_delete, path_params={"id": sid})[0] == 404
    assert call(main.models_list)[1]["sources"] == [] and (outputs / "E1_apl/params.csv").is_file()


def test_baselines(main, outputs):  # noqa: F811
    """Models chosen as baselines: kept by their experiment's folder on tensor, name, base and row; listed alone, as /api/models lists them."""
    assert call(main.models_baselines) == (200, {"baselines": [], "experiments": {}, "models": [], "columns": []})
    sid = call(main.admin_models_add, body={"path": "tom/model_outputs"})[1]["id"]
    for body, want in [({}, 400), ({"experiment": f"{sid}/E1_apl", "row": True}, 400), ({"experiment": f"{sid + 1}/E1_apl", "row": 0}, 404),
                       ({"experiment": f"{sid}/E1_apl", "row": 9}, 404), ({"experiment": f"{sid}/nope", "row": 0}, 404)]:
        assert call(main.admin_models_baseline_add, body=body)[0] == want, body
    status, b = call(main.admin_models_baseline_add, body={"experiment": f"{sid}/E1_apl", "row": 1})  # (k0 of claw-dynamics)
    assert status == 200 and {k: b[k] for k in ("experiment", "name", "base", "row")} == {
        "experiment": "tom/model_outputs/E1_apl", "name": "k0", "base": "claw-dynamics", "row": 1}
    assert call(main.admin_models_baseline_add, body={"experiment": f"{sid}/E1_apl", "row": 1})[0] == 409
    assert call(main.admin_models_baseline_add, body={"experiment": f"{sid}/E2_gain", "row": 0})[0] == 200
    status, out = call(main.models_baselines)
    listed = call(main.models_list)[1]
    assert status == 200 and [x["ok"] for x in out["baselines"]] == [True, True] and [x["model"] for x in out["baselines"]] == [0, 1]
    assert [m["experiment"] for m in out["models"]] == [f"{sid}/E1_apl", f"{sid}/E2_gain"]
    assert {k: v for k, v in out["models"][0].items() if k != "baseline"} == listed["models"][1]  # (as /api/models has it)
    assert out["experiments"][f"{sid}/E1_apl"] == listed["experiments"][f"{sid}/E1_apl"]
    assert out["columns"][:2] == ["generated", "base"] and "wAPLKC" in out["columns"] and "n_kcs" not in out["columns"]
    # found again after rows are added before it (by its name and base), and after its folder is deleted and added again
    p = outputs / "E1_apl/params.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    pd.concat([df.iloc[[0]].assign(tag="new", variant="new"), df]).to_csv(p, index=False)
    assert call(main.models_baselines)[1]["models"][0]["params"]["base"] == "claw-dynamics"
    call(main.admin_models_delete, path_params={"id": sid})
    gone = call(main.models_baselines)[1]
    assert [x["ok"] for x in gone["baselines"]] == [False, False] and "not added" in gone["baselines"][0]["error"] and gone["models"] == []
    sid = call(main.admin_models_add, body={"path": "tom/model_outputs"})[1]["id"]
    out = call(main.models_baselines)[1]
    assert [m["experiment"] for m in out["models"]] == [f"{sid}/E1_apl", f"{sid}/E2_gain"]
    p.write_text("tag,base\nother,claw\n")
    out = call(main.models_baselines)[1]
    assert "no model k0 in its params.csv" in out["baselines"][0]["error"] and out["baselines"][1]["ok"]
    assert call(main.admin_models_baseline_delete, path_params={"id": b["id"]}) == (200, {"deleted": True})
    assert call(main.admin_models_baseline_delete, path_params={"id": b["id"]})[0] == 404
    assert len(call(main.models_baselines)[1]["baselines"]) == 1


def test_download(main, outputs, odor_table, tmp_path, monkeypatch):  # noqa: F811
    from app import config, matrix_download

    assert call(main.admin_models_add, body={"path": "tom/model_outputs"})[0] == 200
    run_id, index = "tom/model_outputs/E1_apl/claw_sp0.05_k0_on_spike_counts.parquet", odormeta.OdorIndex(main.db.list_odors())
    stem = "olfdb_model_E1_apl_claw_sp0.05_k0_on_spike_counts"

    def wide(run=run_id, **picked):
        dest = tmp_path / "wide.csv"
        assert matrix_download.model_download(main.db, run, index, "wide", dest, config.TENSOR_ROOT, **picked)[1] == "text/csv"
        return dest

    # its KCs' levels as in its file; its odors as there, with the odor table's name, abbreviation and CID, and their concentrations
    df = pd.read_csv(wide(), header=[0, 1, 2, 3, 4], index_col=[0, 1])
    assert df.index.names == ["kc_id", "kc_type"] and df.index.tolist()[:2] == [(100, "g"), (101, "ab")]
    assert df.columns.names == ["odor", "odor_table_name", "odor_table_abbrev", "cid", "log10_conc"]
    assert df.columns.tolist()[0] == ("2h @ -3", "2-heptanone", "2h", "8051", "-3")
    assert df.columns.get_level_values("odor").tolist() == ODORS and np.array_equal(df.to_numpy(), counts(6).to_numpy())
    # as shown: of the KCs and odors picked, in that order, transposed
    df = pd.read_csv(wide(rows=[3, 1], cols=[3, 0], transpose=True), header=[0, 1], index_col=[0, 1, 2, 3, 4])
    assert df.index.get_level_values("odor").tolist() == ["pa @ -3", "2h @ -3"] and df.columns.tolist() == [("103", "ab"), ("101", "ab")]
    assert np.array_equal(df.to_numpy(), counts(6).to_numpy()[[3, 1]][:, [3, 0]].T)
    assert len(pd.read_csv(wide("tom/model_outputs/E2_gain/f2_on_spike_counts.parquet"), header=[0, 1, 2, 3, 4], index_col=0)) == 5
    # its files: as on tensor, with its experiment's, and a README of its model
    dest = tmp_path / "files.zip"
    assert matrix_download.model_download(main.db, run_id, index, "files", dest, config.TENSOR_ROOT)[0] == f"{stem}_files.zip"
    with zipfile.ZipFile(dest) as zf:
        assert sorted(zf.namelist()) == [f"{stem}/{n}" for n in ("README.md", "claw_sp0.05_k0_on_spike_counts.parquet", "params.csv",
                                                                 "provenance.json")]
        assert zf.read(f"{stem}/params.csv") == (outputs / "E1_apl/params.csv").read_bytes()
        readme = zf.read(f"{stem}/README.md").decode()
    assert "row 1 of `tom/model_outputs/E1_apl/params.csv`" in readme and "- tag: k0\n" in readme and "- claw_thr: \n" in readme
    # through the API
    monkeypatch.setattr(config, "EXPORTS_DIR", tmp_path / "exports")
    config.EXPORTS_DIR.mkdir()
    r = post(main.models_run_download, {"id": run_id, "format": "files"})
    assert r.status_code == 200 and f'filename="{stem}_files.zip"' in r.headers["content-disposition"] and zipfile.is_zipfile(r.path)
    assert post(main.models_run_download, {"id": "tom/model_outputs/stray_on_spike_counts.parquet", "format": "wide"}).status_code == 404
    assert post(main.models_run_download, {"id": run_id, "format": "wide", "rows": [6]}).status_code == 422
