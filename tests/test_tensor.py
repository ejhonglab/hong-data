"""Tests of users' folders on tensor (path checks, finding a dataset's files). Run with: python -m pytest tests/"""
from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlencode

import pytest
from starlette.requests import Request

from app import tensor
from app.db import Database
from app.ingest import load_file, save_parsed
from tests.test_ingest import COLS, ROWS, make_wide, write_csv

XML = '<ThorImageExperiment><ExperimentNotes text="{notes}"/><MultiPhotonLaser position="920"/></ThorImageExperiment>'
YAML_AB = "odors:\n- {name: ethyl acetate, abbrev: ea, log10_conc: -4}\n- {name: valeric acid, abbrev: va, log10_conc: -3}\n"
YAML_C = "pins2odors:\n  3: {name: 2-heptanone, abbrev: 2h, log10_conc: -6}\n"


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "tensor"
    for p in ("tom/raw/2023-11-19/1/t1", "tom/raw/2023-11-19/1/t2", "tom/raw/20231120/fly02/t1", "tom/mocorr/2023-11-19/1",
              "tom/stim/old", "other"):
        (r / p).mkdir(parents=True)
    (r / "tom/raw/2023-11-19/1/t1/Experiment.xml").write_text(XML.format(notes=r"C:\stim\abc_stim.yaml then more"))
    (r / "tom/raw/2023-11-19/1/t2/Experiment.xml").write_text(XML.format(notes=""))
    (r / "tom/raw/20231120/fly02/t1/Experiment.xml").write_text(XML.format(notes="nothing"))
    (r / "tom/stim/abc_stim.yaml").write_text(YAML_AB)
    (r / "tom/stim/old/2023-11-20_c.yaml").write_text(YAML_C)
    (r / "tom/stim/2023-11-20_ab_too.yaml").write_text(YAML_AB)
    (r / "tom/stim/unrelated.yaml").write_text(YAML_C)
    (r / "tom/notes.txt").write_text("")
    return r


def test_normalize(root):
    assert tensor.normalize(" tom//raw/./ ", root) == "tom/raw"
    assert tensor.normalize("tom\\raw\\", root) == "tom/raw"
    for bad, hint in [(f"{root}/tom/raw", "'tom/raw'"), ("/Volumes/tensor/tom", "'tom'"), ("Z:\\tom\\raw", "'tom/raw'"),
                      ("/home/me/data", None), ("~/data", None)]:
        with pytest.raises(tensor.PathError, match="absolute paths are not accepted") as e:
            tensor.normalize(bad, root)
        assert (hint in str(e.value)) if hint else "did you mean" not in str(e.value)
    for bad in ("tom/../other", "..", ".", ""):
        with pytest.raises(tensor.PathError):
            tensor.normalize(bad, root)


def test_check_dir(root, tmp_path):
    assert tensor.check_dir("tom/raw/", root) == {"input": "tom/raw/", "path": "tom/raw", "ok": True, "error": None}
    assert "'tom' has no 'Raw' (did you mean 'tom/raw'?)" in tensor.check_dir("tom/Raw", root)["error"]
    assert "the top of tensor has no 'nobody'" in tensor.check_dir("nobody/raw", root)["error"]
    assert tensor.check_dir("tom/notes.txt", root)["error"] == "this is a file, not a folder"
    assert "absolute" in tensor.check_dir("/tom", root)["error"]
    (tmp_path / "outside").mkdir()
    os.symlink(tmp_path / "outside", root / "tom" / "escape")
    assert "outside tensor" in tensor.check_dir("tom/escape", root)["error"]
    os.symlink(root / "other", root / "tom" / "inside")
    assert tensor.check_dir("tom/inside", root)["ok"]
    assert "not available" in tensor.check_dir("tom", tmp_path / "unmounted")["error"]
    assert tensor.root_status(root)["ok"] and not tensor.root_status(tmp_path / "unmounted")["ok"]
    (tmp_path / "empty").mkdir()
    assert "unmounted" in tensor.root_status(tmp_path / "empty")["error"]


def test_file_path(root):
    rel, path = tensor.file_path("tom/stim/abc_stim.yaml", root, (".yaml", ".yml"))
    assert rel == "tom/stim/abc_stim.yaml" and path == root / rel
    with pytest.raises(tensor.PathError, match="not a .xml file"):
        tensor.file_path("tom/stim/abc_stim.yaml", root, (".xml",))
    with pytest.raises(tensor.PathError, match="not a file"):
        tensor.file_path("tom/raw", root, ("raw",))
    with pytest.raises(tensor.PathError, match="no 'x.yaml'"):
        tensor.file_path("tom/stim/x.yaml", root, (".yaml",))


def test_lookup(root):
    dirs = {"raw_dirs": ["tom/raw"], "mocorr_dirs": ["tom/mocorr"], "odor_dirs": ["tom/stim"]}
    flies = [{"date": "2023-11-19", "fly_num": 1}, {"date": "2023-11-20", "fly_num": 2}, {"date": "2023-11-21", "fly_num": 1}]
    odors = [("ea", -4.0), ("va", -3.0), ("2h", -6.0)]
    r = tensor.lookup(root, dirs, flies, odors)
    assert [f["raw"] for f in r["flies"]] == [["tom/raw/2023-11-19/1"], ["tom/raw/20231120/fly02"], []]
    assert [f["mocorr"] for f in r["flies"]] == [["tom/mocorr/2023-11-19/1"], [], []]
    assert r["raw_data_path"] == "tom/raw" and r["mocorr_data_path"] == "tom/mocorr/2023-11-19/1"
    assert r["experiment_xmls"] == ["tom/raw/2023-11-19/1/t1/Experiment.xml", "tom/raw/2023-11-19/1/t2/Experiment.xml",
                                    "tom/raw/20231120/fly02/t1/Experiment.xml"]
    y = r["odor_yamls"]
    # the YAML named in the notes is preferred over the one with the same odors found by date
    assert y["selected"] == ["tom/stim/abc_stim.yaml", "tom/stim/old/2023-11-20_c.yaml"]
    assert {c["path"]: (c["reason"], c["n_covered"]) for c in y["candidates"]} == {
        "tom/stim/abc_stim.yaml": (tensor.NAMED, 2), "tom/stim/old/2023-11-20_c.yaml": (tensor.DATED, 1),
        "tom/stim/2023-11-20_ab_too.yaml": (tensor.DATED, 2),
    }
    assert y["missing"] == []

    # nothing is selected unless every odor is described
    r = tensor.lookup(root, dirs, flies, odors + [("x", -2.0)])
    assert r["odor_yamls"]["selected"] == [] and r["odor_yamls"]["missing"] == ["x @ -2"]
    # a folder named by the date holds the flies directly
    r = tensor.lookup(root, {"raw_dirs": ["tom/raw/2023-11-19"]}, flies[:1], [])
    assert r["raw_data_path"] == "tom/raw/2023-11-19/1" and r["odor_yamls"]["candidates"] == []


def test_users(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    assert db.get_user("Tom") is None
    db.save_user("Tom", {"raw_dirs": ["tom/raw"]}, when="2026-10-17T10:00:00")
    u = db.save_user("tom", {"raw_dirs": ["tom/raw2"], "odor_dirs": ["tom/stim"]}, when="2026-10-17T11:00:00")
    assert u == {"name": "tom", "updated_at": "2026-10-17T11:00:00", "raw_dirs": ["tom/raw2"], "mocorr_dirs": [],
                 "odor_dirs": ["tom/stim"]}
    assert db.get_user("TOM") == u


@pytest.fixture
def main(tmp_path, root, monkeypatch):
    from app import config, main

    monkeypatch.setattr(config, "TENSOR_ROOT", root)
    monkeypatch.setattr(main, "db", Database(tmp_path / "db.sqlite3"))
    return main


def call(endpoint, *, body=None, params=None, path_params=None) -> tuple[int, dict]:
    """Status and JSON of an endpoint's response to a GET (with `params`) or a POST of `body` as JSON."""
    raw = json.dumps(body).encode() if body is not None else b""
    scope = {"type": "http", "method": "GET" if body is None else "POST", "path": "/", "headers": [(b"content-type", b"application/json")],
             "query_string": urlencode(params or {}).encode(), "path_params": path_params or {}}

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    response = asyncio.run(endpoint(Request(scope, receive)))
    return response.status_code, json.loads(response.body)


def test_user_api(main):
    save = lambda body: call(main.user_save, body=body)[1]
    get = lambda name: call(main.user_get, params={"name": name})[1]
    r = save({"name": "  Tom  O ", "raw_dirs": ["tom/raw", "/x"], "odor_dirs": ["tom/stim"]})
    assert not r["ok"] and not r["saved"] and [c["ok"] for c in r["dirs"]["raw_dirs"]] == [True, False]
    assert get("tom o") == {"user": None}
    r = save({"name": "  Tom  O ", "raw_dirs": ["tom/raw/", "tom/raw", ""]})
    assert r["ok"] and r["saved"] and r["user"]["raw_dirs"] == ["tom/raw"] and r["user"]["name"] == "Tom O"
    assert get("tom o")["user"]["raw_dirs"] == ["tom/raw"]
    # checked but not saved without a name
    r = save({"raw_dirs": ["tom/mocorr"]})
    assert r["ok"] and not r["saved"]
    assert call(main.user_save, body={"raw_dirs": "tom/raw"})[0] == 400


def test_metadata_check(main):
    check = lambda md: call(main.metadata_check, body={"metadata": md})[1]
    assert check({}) == {"ok": True, "errors": []}
    assert check({"driver": "GH146", "power_mw": 3, "wavelength_nm": 920, "panel_names": "a, b"})["ok"]
    r = check({"power_mw": 0, "age_min_days": -1})
    assert {e["field"] for e in r["errors"]} == {"power_mw", "age_min_days"}
    r = check({"age_min_days": 5, "age_max_days": 2})
    assert r["errors"] == [{"field": "", "message": "metadata: fly age range: max must be >= min"}]
    assert check({"power_tolerance_mw": 1})["errors"][0]["message"].endswith("need a power value")


def test_stage_from_tensor(main, root, tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    # flies of 2024-01-01 (fly 1) and 2024-01-02 (flies 1, 2); odors va @ -3 and ea @ -2
    for p in ("ann/raw/2024-01-01/1/t1", "ann/raw/2024-01-02/2/t1", "ann/yaml"):
        (root / p).mkdir(parents=True)
    (root / "ann/raw/2024-01-01/1/t1/Experiment.xml").write_text(XML.format(notes="ann_odors.yml"))
    (root / "ann/raw/2024-01-02/2/t1/Experiment.xml").write_text(XML.format(notes=""))
    (root / "ann/yaml/ann_odors.yml").write_text(YAML_AB.replace("-4", "-2"))
    main.db.save_user("Ann", {"raw_dirs": ["ann/raw"], "odor_dirs": ["ann/yaml"]}, when="2026-10-17T10:00:00")

    d = config.STAGING_DIR / "stage1234"
    d.mkdir(parents=True)
    save_parsed(load_file(write_csv(make_wide(ROWS, COLS), tmp_path)), d / "parsed")
    (d / "stage.json").write_text(json.dumps({"stage_id": "stage1234", "data_file": "x.csv", "size_bytes": 1, "sha256": "0" * 64,
                                              "source_format": "wide-csv", "odor_yamls": [], "experiment_xml": None,
                                              "experiment_xml_info": None}))
    stage = {"stage_id": "stage1234"}

    assert call(main.stage_lookup, body={"name": "nobody"}, path_params=stage)[1] == {"ok": True, "user": None}
    r = call(main.stage_lookup, body={"name": "ann"}, path_params=stage)[1]
    assert [f["raw"] for f in r["flies"]] == [["ann/raw/2024-01-01/1"], [], ["ann/raw/2024-01-02/2"]]
    assert r["raw_data_path"] == "ann/raw" and r["odor_yamls"]["selected"] == ["ann/yaml/ann_odors.yml"]

    status, r = call(main.stage_add_xml, body={"tensor_path": r["experiment_xmls"][0]}, path_params=stage)
    assert status == 200 and r["suggestions"] == {"wavelength_nm": 920.0}
    assert r["stage"]["experiment_xml_tensor_path"] == "ann/raw/2024-01-01/1/t1/Experiment.xml"
    r = call(main.stage_add_yaml, body={"tensor_paths": ["ann/yaml/ann_odors.yml", "/ann/x.yml", "ann/raw"]}, path_params=stage)[1]
    assert r["check"]["ok"] and r["stage"]["odor_yamls"] == [{"name": "ann_odors.yml", "n_odors": 2, "tensor_path": "ann/yaml/ann_odors.yml"}]
    assert len(r["errors"]) == 2 and "absolute" in r["errors"][0]
    assert call(main.stage_add_xml, body={"tensor_path": "ann/yaml/ann_odors.yml"}, path_params=stage)[0] == 422

    md = {"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli", "raw_data_path": "ann/raw"}
    readme = call(main.stage_readme, body={"metadata": md}, path_params=stage)[1]["readme"]
    assert "- odor metadata: `ann_odors.yml` (from tensor: `ann/yaml/ann_odors.yml`)" in readme
    assert "- ThorImage: `Experiment.xml` (from tensor: `ann/raw/2024-01-01/1/t1/Experiment.xml`)" in readme
