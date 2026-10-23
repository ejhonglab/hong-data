"""Tests of users' folders on tensor (path checks, finding a dataset's files). Run with: python -m pytest tests/"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
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
    # each fly's recordings oldest first (not by name), so the last is the most recent
    os.utime(root / "tom/raw/2023-11-19/1/t1/Experiment.xml", (2e9, 2e9))
    r = tensor.lookup(root, dirs, flies, odors)
    assert r["experiment_xmls"][:2] == ["tom/raw/2023-11-19/1/t2/Experiment.xml", "tom/raw/2023-11-19/1/t1/Experiment.xml"]
    # not anatomical recordings ("anat", "anatomical", or either then "_"), whenever made
    for name in ("anat", "anatomical", "anat_1024px", "anatomical_z"):
        (root / "tom/raw/2023-11-19/1" / name).mkdir()
        (root / "tom/raw/2023-11-19/1" / name / "Experiment.xml").write_text(XML.format(notes=""))
    for name in ("anatomy", "Anat", "t3_anat"):
        (root / "tom/raw/2023-11-19/1" / name).mkdir()
        (root / "tom/raw/2023-11-19/1" / name / "Experiment.xml").write_text(XML.format(notes=""))
        os.utime(root / "tom/raw/2023-11-19/1" / name / "Experiment.xml", (1e9, 1e9))
    r = tensor.lookup(root, dirs, flies[:1], [])
    assert [x.split("/")[-2] for x in r["experiment_xmls"]] == ["Anat", "anatomy", "t3_anat", "t2", "t1"]
    for name in ("anatomy", "Anat", "t3_anat"):
        (root / "tom/raw/2023-11-19/1" / name / "Experiment.xml").unlink()

    # and the folder of one of the dataset's flies, given as its raw data folder
    r = tensor.lookup(root, {"raw_dirs": ["tom/raw/2023-11-19/1"]}, flies, [])
    assert [f["raw"] for f in r["flies"]] == [["tom/raw/2023-11-19/1"], [], []] and len(r["experiment_xmls"]) == 2
    # but never the recordings of a fly not in the dataset, whatever folder is given
    for other in ("tom/raw/2023-11-19/1", "tom/raw/2023-11-19", "tom/raw/20231120/fly02", "tom/raw/2023-11-19/1/t1"):
        r = tensor.lookup(root, {"raw_dirs": [other]}, [{"date": "2023-11-20", "fly_num": 1}], [])
        assert r["flies"][0]["raw"] == [] and r["experiment_xmls"] == [], other
    r = tensor.lookup(root, {"raw_dirs": ["tom/raw/20231120/fly02"]}, [{"date": "2023-11-19", "fly_num": 2}], [])
    assert r["experiment_xmls"] == []  # the right number, on another date


DIAG = [("ethyl acetate", "ea", -8), ("valeric acid", "va", -4)]
MEGA = [("2-heptanone", "2h", -3), ("hexanal", "6al", -3), ("linalool", "Lin", -3)]


def generated(odors) -> str:  # as olf writes them, one per recording
    return "settings: {}\npins2odors:\n" + "".join(f"  {37 + i}: {{name: {n}, abbrev: {a}, log10_conc: {c}}}\n"
                                                    for i, (n, a, c) in enumerate(odors))


def olf_config(odors) -> str:  # olf's input: a panel's config
    return ("generator: basic\nodors:\n" + "".join(f"- {{name: {n}, abbrev: {a}, log10_conc: {c}}}\n" for n, a, c in odors)
            + "- {air_mix: [A, B]}\n")


def test_lookup_panels(tmp_path):
    root = tmp_path / "tensor"
    fly = root / "tom/raw/2024-01-05/1"

    def recording(name, notes, t):
        (fly / name).mkdir(parents=True)
        (fly / name / "Experiment.xml").write_text(XML.format(notes=notes))
        os.utime(fly / name / "Experiment.xml", (t, t))

    recording("glomeruli_diagnostics_1", r"C:\olf\20240105_110000_stimuli.yaml", 1e9)
    recording("megamat_1", r"20240105_120000_stimuli\20240105_120000_stimuli_0.yaml", 1e9 + 10)
    recording("megamat_2", r"20240105_120000_stimuli\20240105_120000_stimuli_1.yaml", 1e9 + 20)
    recording("megamat_2_redo", r"20240105_130000_stimuli\20240105_130000_stimuli_1.yaml", 1e9 + 30)  # replaces it
    olf = root / "tom/olf"
    for rel, text in [("20240105_110000_stimuli.yaml", generated(DIAG)),
                      ("20240105_120000_stimuli/20240105_120000_stimuli_0.yaml", generated(MEGA[:2])),
                      ("20240105_120000_stimuli/20240105_120000_stimuli_1.yaml", generated([("linalool", "Lin", -2)])),
                      ("20240105_130000_stimuli/20240105_130000_stimuli_1.yaml", generated(MEGA[2:])),
                      ("megamat.yaml", olf_config(MEGA)), ("old/glomeruli_diagnostics.yaml", olf_config(DIAG)),
                      ("venv/lib/20240105_odors.yaml", generated(DIAG + MEGA))]:  # not looked in
        (olf / rel).parent.mkdir(parents=True, exist_ok=True)
        (olf / rel).write_text(text)
    dirs = {"raw_dirs": ["tom/raw"], "odor_dirs": ["tom/olf"]}
    flies = [{"date": "2024-01-05", "fly_num": 1}]
    panel_odors = {"glomeruli_diagnostics": [("ea", -8.0), ("va", -4.0)], "megamat": [(a, float(c)) for _, a, c in MEGA]}
    gen = ["tom/olf/20240105_110000_stimuli.yaml", "tom/olf/20240105_120000_stimuli/20240105_120000_stimuli_0.yaml",
           "tom/olf/20240105_130000_stimuli/20240105_130000_stimuli_1.yaml"]

    # each panel by the YAMLs generated for its recordings (the one redone, not the one it replaced)
    r = tensor.lookup(root, dirs, flies, panel_odors)["odor_yamls"]
    assert [(p["panel"], p["selected"], p["complete"]) for p in r["panels"]] == [
        ("glomeruli_diagnostics", gen[:1], True), ("megamat", gen[1:], True)]
    assert r["panels"][1]["how"] == "named in the notes of its megamat recordings, of fly 1 of 2024-01-05"
    assert r["selected"] == gen and r["missing"] == [] and r["n_matrix_odors"] == 5
    assert not any("venv" in c["path"] or c["path"].endswith("120000_stimuli_1.yaml") for c in r["candidates"])
    # the same with no panel in the data: the names its recordings are named for
    r = tensor.lookup(root, dirs, flies, [o for odors in panel_odors.values() for o in odors])["odor_yamls"]
    assert r["selected"] == gen and [p["panel"] for p in r["panels"]] == [None]

    # without the YAMLs generated for its recordings, a panel's config, in whichever folder
    for rel in gen[1:]:
        (root / rel).unlink()
    r = tensor.lookup(root, dirs, flies, panel_odors)["odor_yamls"]
    assert [(p["selected"], p["how"]) for p in r["panels"]] == [
        (gen[:1], "named in the notes of its glomeruli_diagnostics recordings, of fly 1 of 2024-01-05"),
        (["tom/olf/megamat.yaml"], "the megamat panel's config")]
    (root / gen[0]).unlink()
    r = tensor.lookup(root, dirs, flies, [o for odors in panel_odors.values() for o in odors])["odor_yamls"]
    assert r["selected"] == ["tom/olf/megamat.yaml", "tom/olf/old/glomeruli_diagnostics.yaml"]  # the one describing more first
    assert {c["path"]: c["reason"] for c in r["candidates"]} == {
        "tom/olf/megamat.yaml": "the megamat panel's config",
        "tom/olf/old/glomeruli_diagnostics.yaml": "the glomeruli_diagnostics panel's config"}

    # a panel described in part: what for, and what not
    r = tensor.lookup(root, dirs, flies, {**panel_odors, "megamat": panel_odors["megamat"] + [("x", -1.0)]})["odor_yamls"]
    assert r["selected"] == [] and r["missing"] == ["x @ -1"]
    assert [(p["panel"], p["complete"], p["selected"]) for p in r["panels"]] == [
        ("glomeruli_diagnostics", True, ["tom/olf/old/glomeruli_diagnostics.yaml"]), ("megamat", False, ["tom/olf/megamat.yaml"])]


def test_lookup_recording_numbers(tmp_path):
    root = tmp_path / "tensor"
    fly = root / "tom/raw/2024-01-06/1"
    olf = root / "tom/olf"

    def recording(name, notes, t):
        (fly / name).mkdir(parents=True)
        (fly / name / "Experiment.xml").write_text(XML.format(notes=notes))
        os.utime(fly / name / "Experiment.xml", (t, t))

    def yaml_file(rel, odors):
        (olf / rel).parent.mkdir(parents=True, exist_ok=True)
        (olf / rel).write_text(generated(odors))

    # the only recording of a panel is named for it alone, and redone as ThorImage numbers a name used already
    recording("glomeruli_diagnostics", "20240106_100000_stimuli.yaml", 1e9)
    recording("glomeruli_diagnostics_001", "20240106_101000_stimuli.yaml", 1e9 + 10)
    yaml_file("20240106_100000_stimuli.yaml", [("ethyl acetate", "ea", -7)])  # replaced
    yaml_file("20240106_101000_stimuli.yaml", DIAG)
    # recording n from olf's YAML n - 1: the one whose notes name none takes it from the others' run
    recording("megamat_1", r"20240106_110000_stimuli\20240106_110000_stimuli_0.yaml", 1e9 + 20)
    recording("megamat_2", "", 1e9 + 30)
    yaml_file("20240106_110000_stimuli/20240106_110000_stimuli_0.yaml", MEGA[:2])
    yaml_file("20240106_110000_stimuli/20240106_110000_stimuli_1.yaml", MEGA[2:])
    dirs = {"raw_dirs": ["tom/raw"], "odor_dirs": ["tom/olf"]}
    flies = [{"date": "2024-01-06", "fly_num": 1}]
    panel_odors = {"glomeruli_diagnostics": [("ea", -8.0), ("va", -4.0)], "megamat": [(a, float(c)) for _, a, c in MEGA]}
    r = tensor.lookup(root, dirs, flies, panel_odors)["odor_yamls"]
    assert [(p["panel"], p["selected"], p["complete"]) for p in r["panels"]] == [
        ("glomeruli_diagnostics", ["tom/olf/20240106_101000_stimuli.yaml"], True),
        ("megamat", ["tom/olf/20240106_110000_stimuli/20240106_110000_stimuli_0.yaml",
                     "tom/olf/20240106_110000_stimuli/20240106_110000_stimuli_1.yaml"], True)]
    assert r["panels"][0]["remarks"] == []
    assert r["panels"][1]["remarks"] == [
        "megamat_2 (fly 1 of 2024-01-06): no YAML in its notes, so 20240106_110000_stimuli_1.yaml, by its number"]
    # notes naming the YAML of another number are pointed out; numbers past 9 are numbers too
    (fly / "megamat_2" / "Experiment.xml").write_text(XML.format(notes=r"20240106_110000_stimuli\20240106_110000_stimuli_0.yaml"))
    os.utime(fly / "megamat_2" / "Experiment.xml", (1e9 + 30, 1e9 + 30))
    recording("megamat_10", "", 1e9 + 40)
    r = tensor.lookup(root, dirs, flies, panel_odors)["odor_yamls"]
    assert r["panels"][1]["remarks"] == [
        "megamat_2 (fly 1 of 2024-01-06): its notes name 20240106_110000_stimuli_0.yaml, though by its number it would be "
        "the run's _stimuli_1.yaml",
        "megamat_10 (fly 1 of 2024-01-06): no YAML in its notes, so 20240106_110000_stimuli_9.yaml, by its number"]
    # nothing names _stimuli_1 now: it is found by the date in its path
    assert r["panels"][1]["complete"] and r["panels"][1]["how"] == "named in the notes of its megamat recordings, a fly's date in its path"
    # with no panel in the data, the same names
    r = tensor.lookup(root, dirs, flies, [o for odors in panel_odors.values() for o in odors])["odor_yamls"]
    assert r["panels"][0]["panel"] is None and len(r["panels"][0]["remarks"]) == 2


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
    # the raw data path as entered, with or without a user
    r = call(main.stage_lookup, body={"name": "", "raw_data_path": " ann//raw/"}, path_params=stage)[1]
    assert r["user"] is None and r["raw_data_path_given"] == "ann/raw" and r["raw_data_path"] == "ann/raw"
    assert r["experiment_xmls"] == ["ann/raw/2024-01-01/1/t1/Experiment.xml", "ann/raw/2024-01-02/2/t1/Experiment.xml"]
    assert r["odor_yamls"]["candidates"] == []  # no odor YAML folders without a user
    r = call(main.stage_lookup, body={"name": "ann", "raw_data_path": "ann/raw/2024-01-02"}, path_params=stage)[1]
    # (the YAML is named only in the notes of the fly not in that folder)
    assert [f["raw"] for f in r["flies"]] == [[], [], ["ann/raw/2024-01-02/2"]] and r["odor_yamls"]["candidates"] == []
    r = call(main.stage_lookup, body={"name": "ann", "raw_data_path": "ann/nothing"}, path_params=stage)[1]
    assert r["raw_error"] == "not found on tensor: 'ann' has no 'nothing'" and "flies" not in r
    r = call(main.stage_lookup, body={"name": "ann"}, path_params=stage)[1]
    assert [f["raw"] for f in r["flies"]] == [["ann/raw/2024-01-01/1"], [], ["ann/raw/2024-01-02/2"]]
    assert r["raw_data_path"] == "ann/raw" and r["odor_yamls"]["selected"] == ["ann/yaml/ann_odors.yml"]

    status, r = call(main.stage_add_xml, body={"tensor_path": r["experiment_xmls"][0]}, path_params=stage)
    # the laser wavelength in Experiment.xml is not read (it does not follow the laser)
    assert status == 200 and "laser_position" not in r["info"] and "suggestions" not in r
    assert r["stage"]["experiment_xml_tensor_path"] == "ann/raw/2024-01-01/1/t1/Experiment.xml"
    r = call(main.stage_add_yaml, body={"tensor_paths": ["ann/yaml/ann_odors.yml", "/ann/x.yml", "ann/raw"]}, path_params=stage)[1]
    assert r["check"]["ok"] and r["stage"]["odor_yamls"] == [{"name": "ann_odors.yml", "n_odors": 2, "tensor_path": "ann/yaml/ann_odors.yml"}]
    assert len(r["errors"]) == 2 and "absolute" in r["errors"][0]
    assert r["added"] == [{"name": "ann_odors.yml", "n_odors": 2, "tensor_path": "ann/yaml/ann_odors.yml", "replaced": False}]
    assert [f["name"] for f in r["files"]] == ["ann_odors.yml"] and r["files"][0]["table_empty"]
    assert call(main.stage_add_xml, body={"tensor_path": "ann/yaml/ann_odors.yml"}, path_params=stage)[0] == 422
    # a file of the same name replaces it, if it can be read; else the one attached stays
    r = call(main.stage_add_yaml, body={"tensor_paths": ["ann/yaml/ann_odors.yml"]}, path_params=stage)[1]
    assert r["ok"] and r["added"][0]["replaced"]
    (root / "ann/yaml/ann_odors.yml").write_text("odors: [")
    r = call(main.stage_add_yaml, body={"tensor_paths": ["ann/yaml/ann_odors.yml"]}, path_params=stage)[1]
    assert not r["ok"] and r["added"] == [] and "(the ann_odors.yml attached before is still attached)" in r["errors"][0]
    assert r["check"]["ok"] and r["stage"]["odor_yamls"][0]["n_odors"] == 2
    assert not (d / "odor_yaml" / ".incoming").exists()
    r = call(main.stage_remove_yaml, body={}, path_params={**stage, "name": "ann_odors.yml"})[1]
    assert r["removed"] == "ann_odors.yml" and r["files"] == [] and r["check"] is None
    (root / "ann/yaml/ann_odors.yml").write_text(YAML_AB.replace("-4", "-2"))
    call(main.stage_add_yaml, body={"tensor_paths": ["ann/yaml/ann_odors.yml"]}, path_params=stage)

    md = {"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli", "raw_data_path": "ann/raw"}
    readme = call(main.stage_readme, body={"metadata": md}, path_params=stage)[1]["readme"]
    assert "- odor metadata: `ann_odors.yml` (from tensor: `ann/yaml/ann_odors.yml`)" in readme
    assert "- ThorImage: `Experiment.xml` (from tensor: `ann/raw/2024-01-01/1/t1/Experiment.xml`)" in readme
    # data paths are checked as folders in the Settings tab are, and kept normalized
    assert "| Raw data (tensor) | `ann/raw` |" in readme
    status, r = call(main.stage_readme, body={"metadata": {**md, "raw_data_path": "ann/nothing"}}, path_params=stage)
    assert status == 422 and r["errors"] == ["raw data path: not found on tensor: 'ann' has no 'nothing'"]
    r = call(main.stage_readme, body={"metadata": {**md, "mocorr_data_path": f"{root}/ann/raw"}}, path_params=stage)[1]
    assert r["errors"][0].startswith("motion-corrected data path: absolute paths are not accepted")
    readme = call(main.stage_readme, body={"metadata": {**md, "raw_data_path": " ann//raw/ "}}, path_params=stage)[1]["readme"]
    assert "| Raw data (tensor) | `ann/raw` |" in readme
    assert call(main.stage_commit, body={"metadata": {**md, "raw_data_path": "x"}, "readme": "#"}, path_params=stage)[0] == 422

    # saved only once each of its odors is one in the odor table: here, once the sheet has them
    from app import odormeta
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "ODOR_SHEET_URL", "")
    r = call(main.stage_odors, path_params=stage)[1]
    assert [m["name"] for m in r["missing"]] == ["valeric acid", "ethyl acetate"] and r["sheet"]["url"] is None
    status, r = call(main.stage_commit, body={"metadata": md, "readme": "#"}, path_params=stage)
    assert status == 422 and "2 of its odors are not one odor in the odor table (va @ -3, ea @ -2)" in r["errors"][0]
    rows, _, _ = odormeta.parse_sheet("name,abbrev\nvaleric acid,va\n")
    main.db.apply_odor_sync(added=odormeta.diff(rows, [])["added"], changed=[], deleted=[], source="test",
                            fingerprint="0" * 64, details={}, when="2026-10-19T10:00:00")
    assert [m["name"] for m in call(main.stage_odors, path_params=stage)[1]["missing"]] == ["ethyl acetate"]
    rows, _, _ = odormeta.parse_sheet("name,abbrev\nvaleric acid,va\nethyl acetate,ea\n")
    main.db.apply_odor_sync(added=odormeta.diff(rows, main.db.list_odors())["added"], changed=[], deleted=[], source="test",
                            fingerprint="1" * 64, details={}, when="2026-10-19T10:00:00")
    r = call(main.stage_odors, path_params=stage)[1]
    assert r["missing"] == [] and r["sheet"] is None and {x["match"] for x in r["rows"]} == {"yaml"}
    # an upload is continued as it was left
    r = call(main.stage_get, path_params=stage)[1]
    assert r["stage"]["odor_yamls"][0]["name"] == "ann_odors.yml" and r["check"]["ok"] and r["files"][0]["n_in_table"] == 2
    (d / "x.csv").write_text("as uploaded")
    status, r = call(main.stage_commit, body={"metadata": md, "readme": "#"}, path_params=stage)
    assert status == 200 and r["ok"] and (config.DATASETS_DIR / str(r["dataset_id"]) / "odor_yaml" / "ann_odors.yml").exists()


def test_lowercase_only(main, tmp_path, monkeypatch):
    from app import config, odormeta

    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "ODOR_SHEET_URL", "")
    rows = [("p1", False, o, "solvent", r) for o in ("IaA @ -3", "va @ -3", "Va @ -2") for r in range(2)]
    d = config.STAGING_DIR / "stage5678"
    d.mkdir(parents=True)
    save_parsed(load_file(write_csv(make_wide(rows, COLS), tmp_path)), d / "parsed")
    (d / "stage.json").write_text(json.dumps({"stage_id": "stage5678", "data_file": "x.csv", "size_bytes": 1, "sha256": "0" * 64,
                                              "source_format": "wide-csv", "odor_yamls": [], "experiment_xml": None,
                                              "experiment_xml_info": None}))
    (d / "x.csv").write_text("as uploaded")
    stage = {"stage_id": "stage5678"}
    sheet, _, _ = odormeta.parse_sheet("name,abbrev\nisoamyl acetate,IaA\nvaleric acid,va\n")
    main.db.apply_odor_sync(added=odormeta.diff(sheet, [])["added"], changed=[], deleted=[], source="test",
                            fingerprint="0" * 64, details={}, when="2026-10-22T10:00:00")
    md = {"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli"}
    # names differing only in case are said to be, either way
    assert call(main.stage_odors, path_params=stage)[1]["case"] == {"lowercase_only": False, "uppercase": [], "clashes": [["Va", "va"]]}

    status, r = call(main.admin_odor_case, body={"lowercase_only": True})
    assert status == 200 and r["lowercase_only"] and call(main.admin_get)[1]["lowercase_only"]
    assert call(main.admin_odor_case, body={"lowercase_only": "yes"})[0] == 400
    # IaA is in the odor table already, Va nowhere
    assert call(main.stage_odors, path_params=stage)[1]["case"]["uppercase"] == ["Va"]
    status, r = call(main.stage_commit, body={"metadata": md, "readme": "#"}, path_params=stage)
    assert status == 422 and r["errors"][0].startswith("1 of its odor names have capital letters (Va), and new odor names must be lowercase")
    call(main.admin_odor_case, body={"lowercase_only": False})
    status, r = call(main.stage_commit, body={"metadata": md, "readme": "#"}, path_params=stage)
    assert status == 200 and r["ok"]
    # once in a dataset, it is in the database too
    call(main.admin_odor_case, body={"lowercase_only": True})
    assert "Va" in main._known_odor_names() and main._case_check([{"components": ["Va @ -1", "New @ -1"]}])["uppercase"] == ["New"]


def test_kc_ids_by_coding_unit(main, tmp_path, monkeypatch):
    from app import config

    # numbered ROIs (not 'cells') are refused with gaps once the coding unit says they are KCs
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    d = config.STAGING_DIR / "stage9999"
    d.mkdir(parents=True)
    save_parsed(load_file(write_csv(make_wide(ROWS, [("2024-01-01", 1, 1), ("2024-01-01", 1, 3)]), tmp_path)), d / "parsed")
    (d / "stage.json").write_text(json.dumps({"stage_id": "stage9999", "data_file": "x.csv", "size_bytes": 1, "sha256": "0" * 64,
                                              "source_format": "wide-csv", "odor_yamls": [], "experiment_xml": None,
                                              "experiment_xml_info": None}))
    stage = {"stage_id": "stage9999"}
    md = {"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "MB247", "coding_unit": "Kenyon cells"}
    status, r = call(main.stage_readme, body={"metadata": md}, path_params=stage)
    assert status == 422 and r["errors"][0].startswith("KC data must have every KC of each fly") and "2024-01-01 fly 1 has no 2" in r["errors"][0]
    status, r = call(main.stage_commit, body={"metadata": md, "readme": "#"}, path_params=stage)
    assert status == 422 and r["errors"][0].startswith("KC data must have every KC")
    assert call(main.stage_readme, body={"metadata": {**md, "coding_unit": "PNs"}}, path_params=stage)[1]["ok"]


def upload(main, path: Path) -> tuple[int, dict]:
    """Status and JSON of uploading a file to the staging area, as the upload form does (multipart)."""
    boundary = "olfdbtestboundary"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            "Content-Type: text/csv\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    scope = {"type": "http", "method": "POST", "path": "/", "query_string": b"", "path_params": {},
             "headers": [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())]}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    response = asyncio.run(main.stage_create(Request(scope, receive)))
    return response.status_code, json.loads(response.body)


def test_duplicates(main, tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "ODOR_SHEET_URL", "")
    values = np.arange(len(ROWS) * len(COLS), dtype=float).reshape(len(ROWS), len(COLS))
    first = write_csv(make_wide(ROWS, COLS, values), tmp_path, "first.csv")
    parsed = load_file(first)
    main.db.insert_dataset(
        created_at="2026-10-21T10:00:00", title="p1: GH146 > GCaMP6f (glomeruli)", readme_md="#", readme_edited=False,
        data_file="first.csv", metadata={"uploader_name": "Betty", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli"},
        data_format=parsed.source_format, data_sha256=main._sha256(first), odor_yaml_files=[], experiment_xml_file=None,
        experiment_xml_info=None, summary=parsed.summary(), units=parsed.units, trials=parsed.trials, values=parsed.values)
    which = "dataset #1 (p1: GH146 > GCaMP6f (glomeruli)), uploaded by Betty on 2026-10-21"
    # whoever uploads it: the same file, under another name, refused when it is read
    renamed = tmp_path / "renamed.csv"
    renamed.write_bytes(first.read_bytes())
    status, r = upload(main, renamed)
    assert status == 422 and r["errors"] == [f"this file was uploaded already: it is {which}. If this replaces it, delete #1 first (Admin tab)."]
    # ... and the same data in another file (its ROIs in another order)
    status, r = upload(main, write_csv(make_wide(ROWS, COLS[::-1], values[:, ::-1]), tmp_path, "reordered.csv"))
    assert status == 422 and r["errors"][0].startswith(f"this is the data of {which}, already (the same ROIs, trials and values")

    # other data of the same flies: refused for the same uploader and coding unit (however written), once they are entered
    status, r = upload(main, write_csv(make_wide(ROWS, COLS, values + 1), tmp_path, "same_flies.csv"))
    assert status == 200
    stage = {"stage_id": r["stage"]["stage_id"]}
    md = {"uploader_name": " betty ", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "Glomeruli"}
    readme = lambda **kw: call(main.stage_readme, body={"metadata": {**md, **kw}}, path_params=stage)
    status, r = readme()
    assert status == 422 and r["errors"] == [
        f"its flies (2024-01-01 fly 1, 2024-01-02 fly 1, 2024-01-02 fly 2) are exactly those of {which}, also of glomeruli: "
        "one person's datasets of a coding unit cannot have the same flies. If this replaces it, delete #1 first (Admin tab)."]
    status, r = call(main.stage_commit, body={"metadata": md, "readme": "#"}, path_params=stage)
    assert status == 422 and any("are exactly those of dataset #1" in e for e in r["errors"])
    # another's flies (one's fly 1 of a day is not another's), or of another coding unit (one fly recorded in two places)
    assert readme(uploader_name="Ann")[1]["ok"] and readme(coding_unit="PNs")[1]["ok"]

    # some flies of one of theirs: a warning, saying which dataset may need to be deleted
    status, r = upload(main, write_csv(make_wide(ROWS, COLS + [("2024-01-03", 1, "DM2")]), tmp_path, "more.csv"))
    stage = {"stage_id": r["stage"]["stage_id"]}
    assert status == 200 and not any("duplicate" in w for w in r["data"]["warnings"])  # (whose, not yet known)
    r = readme()[1]
    assert r["ok"] and r["warnings"] == [
        f"possible duplicate: 3 of its 4 flies (2024-01-01 fly 1, 2024-01-02 fly 1, 2024-01-02 fly 2) are also in {which}, "
        "all of whose flies are here, with 1 more: if this replaces it, #1 may need to be deleted (Admin tab)"]
    assert readme(uploader_name="Ann")[1]["warnings"] == []
    status, r = upload(main, write_csv(make_wide(ROWS, COLS[:3]), tmp_path, "fewer.csv"))  # (2 of its 3 flies)
    stage = {"stage_id": r["stage"]["stage_id"]}
    w = readme()[1]["warnings"]
    assert len(w) == 1 and w[0].startswith("possible duplicate: 2 of its 2 flies") and "which has 1 more: if these are the same recordings, this repeats part of #1" in w[0]
    status, r = upload(main, write_csv(make_wide(ROWS, [COLS[0], ("2024-01-03", 1, "DM2")]), tmp_path, "some.csv"))
    stage = {"stage_id": r["stage"]["stage_id"]}
    w = readme()[1]["warnings"]
    assert len(w) == 1 and w[0].startswith("possible duplicate: 1 of its 2 flies (2024-01-01 fly 1) is also in") and "which has 2 others" in w[0]
    status, r = upload(main, write_csv(make_wide(ROWS, [("2024-02-01", 1, "DM2")]), tmp_path, "other.csv"))
    stage = {"stage_id": r["stage"]["stage_id"]}
    assert readme()[1]["warnings"] == []

    # and refused when saving, if a dataset of the uploader's with its flies was saved since
    status, r = upload(main, write_csv(make_wide(ROWS, [("2024-04-01", 1, "DM2")]), tmp_path, "later.csv"))
    stage = {"stage_id": r["stage"]["stage_id"]}
    p = load_file(tmp_path / "later.csv")
    main.db.insert_dataset(
        created_at="2026-10-22T10:00:00", title="t", readme_md="#", readme_edited=False, data_file="x.csv",
        metadata={"uploader_name": "Jon", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli"},
        data_format=p.source_format, data_sha256="1" * 64, odor_yaml_files=[], experiment_xml_file=None,
        experiment_xml_info=None, summary=p.summary(), units=p.units, trials=p.trials, values=p.values + 1)
    status, r = call(main.stage_commit, body={"metadata": {**md, "uploader_name": "Jon"}, "readme": "#"}, path_params=stage)
    assert status == 422 and any("are exactly those of dataset #2 (t), uploaded by Jon on 2026-10-22" in e for e in r["errors"])


def test_check_data_path(root, tmp_path):
    assert tensor.check_data_path("tom//raw/", root) == {"input": "tom//raw/", "path": "tom/raw", "ok": True, "error": None,
                                                          "warning": None}
    assert not tensor.check_data_path("tom/nothing", root)["ok"] and not tensor.check_data_path("/tom/raw", root)["ok"]
    # tensor not there: well-formed paths are taken as they are, with a warning
    r = tensor.check_data_path("tom/raw/", tmp_path / "unmounted")
    assert r["ok"] and r["path"] == "tom/raw" and r["warning"].startswith("not checked: tensor is not available")
    assert not tensor.check_data_path("/tom/raw", tmp_path / "unmounted")["ok"]


def test_find_data(root, tmp_path, monkeypatch):
    from app import sandbox

    monkeypatch.setattr(sandbox, "ACTIVE", None)
    find = lambda p, bases=(): tensor.find_data(p, root, list(bases))
    r = find("tom/raw/2023-11-19")
    assert r["status"] == "ok" and r["path"] == str(root / "tom/raw/2023-11-19") and r["tensor_path"] == "tom/raw/2023-11-19"
    assert r["is_dir"] and r["entries"] == [{"name": "1", "dir": True, "size": None}] and r["n_entries"] == 1
    assert find("tom/notes.txt")["is_dir"] is False and find("tom/notes.txt")["entries"] == []
    # absolute paths: on the server, or as on a computer that mounts tensor
    assert find(f"{root}/tom/raw/")["tensor_path"] == "tom/raw"
    assert find("Z:\\tom\\raw")["tensor_path"] == "tom/raw" and find("/Volumes/tensor/tom/raw")["tensor_path"] == "tom/raw"
    assert find(None)["status"] == "none" and find("  ")["status"] == "none"
    r = find("tom/nothing")
    assert r["status"] == "not_found" and "Admin" not in r["error"] and find("../tensor/tom")["status"] == "not_found"
    # from a folder set in the Admin tab, which need not be in tensor
    (tmp_path / "elsewhere/2024").mkdir(parents=True)
    r = find("2024", [tmp_path / "elsewhere"])
    assert r["status"] == "ok" and r["tensor_path"] is None and r["path"] == str(tmp_path / "elsewhere/2024")
    # only what the server may read
    sandbox.activate({"read_dirs": [str(root / "other")], "write_dirs": []})
    assert find("tom/raw")["status"] == "not_allowed" and "Admin tab" in find("tom/raw")["error"]


def test_dataset_data_paths(main, root, tmp_path):
    parsed = load_file(write_csv(make_wide(ROWS, COLS), tmp_path))
    md = {"uploader_name": "me", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli",
          "raw_data_path": "tom/raw", "mocorr_data_path": "tom/mocorr-typo"}
    ds_id = main.db.insert_dataset(
        created_at="2026-01-02T03:04:05", title="t", metadata=md, readme_md="#", readme_edited=False, data_file="x.csv",
        data_format=parsed.source_format, data_sha256="0" * 64, odor_yaml_files=[], experiment_xml_file=None,
        experiment_xml_info=None, summary=parsed.summary(), units=parsed.units, trials=parsed.trials, values=parsed.values)
    status, r = call(main.dataset_data_paths, path_params={"ds_id": ds_id})
    assert status == 200 and r["raw"]["status"] == "ok" and [e["name"] for e in r["raw"]["entries"]] == ["2023-11-19", "20231120"]
    assert r["mocorr"]["status"] == "not_found" and r["mocorr"]["input"] == "tom/mocorr-typo"
    assert not r["raw"]["compared"] and r["raw"]["unmatched"] == [] and r["recordings"] is None  # only one found
    assert call(main.dataset_data_paths, path_params={"ds_id": ds_id + 1})[0] == 404

    # entered from the dataset's page, checked as at upload
    assert call(main.tensor_check, body={"path": "tom/mocorr/"})[1]["path"] == "tom/mocorr"
    put = lambda body: call(main.dataset_data_paths_save, body=body, path_params={"ds_id": ds_id})
    status, r = put({"mocorr_data_path": "tom/mocorr-typo2"})
    assert status == 422 and not r["checks"]["mocorr_data_path"]["ok"] and main.db.get_dataset(ds_id)["mocorr_data_path"] == "tom/mocorr-typo"
    status, r = put({"mocorr_data_path": " tom//mocorr/ "})
    assert status == 200 and r["metadata"]["mocorr_data_path"] == "tom/mocorr" and r["metadata"]["raw_data_path"] == "tom/raw"
    d = main.db.get_dataset(ds_id)
    assert d["mocorr_data_path"] == "tom/mocorr" and d["metadata"]["mocorr_data_path"] == "tom/mocorr"
    assert call(main.dataset_data_paths, path_params={"ds_id": ds_id})[1]["mocorr"]["status"] == "ok"
    status, r = put({"raw_data_path": ""})
    assert status == 200 and main.db.get_dataset(ds_id)["raw_data_path"] is None and r["metadata"]["mocorr_data_path"] == "tom/mocorr"
    assert put({"title": "x"})[0] == 400 and put({"raw_data_path": 3})[0] == 400
