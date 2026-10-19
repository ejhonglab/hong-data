"""Tests of reference matrices (built in, and added in the Admin tab). Run with:
OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import base64
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from app import config, odormeta, reference, sandbox
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)

SHEET = "name,abbrev,aliases,cid,cas\nethyl acetate,ea,,8857,141-78-6\n2-heptanone,2h,,8051,\nhexanal,hex,caproaldehyde,,\nlinalool,lin,,6549,\n"
INCHI_LIN = "InChI=1S/C10H18O/c1-5-10(4,11)8-6-7-9(2)3/h5,7,11H,1,6,8H2,2-4H3"


@pytest.fixture
def env(main, tmp_path, monkeypatch):  # noqa: F811
    """main, with an odor table, a built-in reference matrix, and the app's folders in tmp_path."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "ADMIN_FILE", tmp_path / "admin.json")
    monkeypatch.setattr(config, "REFERENCE_CACHE_DIR", tmp_path / "data" / "references")
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "manifest.json").write_text(json.dumps({"datasets": [
        {"id": "fruits", "title": "Fruits", "file": "fruits.csv", "kind": "sequential"},
        {"id": "gone", "title": "Gone", "file": "gone.csv"}]}))
    (builtin / "fruits.csv").write_text("chemical,banana,apple\nethyl acetate,1,2\nhexanal,3,\n")
    monkeypatch.setattr(config, "REFERENCE_DIR", builtin)
    monkeypatch.setattr(sandbox, "ACTIVE", None)
    monkeypatch.setattr(sandbox, "STATUS", {"settings": None, "started_at": None, "kernel": None})
    monkeypatch.setattr(reference, "_memo", OrderedDict())
    rows, _, _ = odormeta.parse_sheet(SHEET)
    main.db.apply_odor_sync(added=rows, changed=[], deleted=[], applied_by="me", source="test", fingerprint="0",
                            details={}, when="2026-10-18T00:00:00")
    main.db.save_inchis({6549: INCHI_LIN}, when="2026-10-18T00:00:00")
    return main


def index(main):  # noqa: F811
    return odormeta.OdorIndex(main.db.list_odors())


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_parse(tmp_path):
    # identifier columns: the rows are chemicals; text columns and rows without numbers left out
    p = write(tmp_path / "a.csv", "chemical,pubchem_cid,InChI,group,notes,banana,apple\n"
                                   "Ethyl Acetate,8857,,esters,nice,1,2\nhexanal,,,aldehydes,,x,0.5\n"
                                   "units,,,,,ng/g,ng/g\nsomething,CID 12,InChI=1S/X,other,,-1,\n")
    record, values = reference.parse(p)
    assert [r["label"] for r in record["rows"]] == ["Ethyl Acetate", "hexanal", "something"]
    assert [r["group"] for r in record["rows"]] == ["esters", "aldehydes", "other"]
    assert [c["label"] for c in record["cols"]] == ["banana", "apple"]
    assert record["ids"] == {"rows": [{"cid": 8857}, {}, {"cid": 12, "inchi": "InChI=1S/X"}], "cols": None}
    assert np.array_equal(values, np.array([[1, 2], [np.nan, 0.5], [-1, np.nan]], dtype=np.float32), equal_nan=True)
    assert record["kind"] == "diverging" and record["shape"] == [3, 2]
    assert record["notes"] == ["left out, as it has no numbers: column 'notes'", "left out, as it has no numbers: row 'units'",
                               "1 cell is not a number: shown as missing"]
    # identifier rows: the columns are chemicals
    p = write(tmp_path / "b.tsv", "receptor\tlinalool\tmystery\ncid\t6549\tnope\nOr22a\t1\t0\nOr47b\t0\t1\n")
    record, values = reference.parse(p)
    assert record["ids"] == {"rows": None, "cols": [{"cid": 6549}, {}]} and record["kind"] == "binary"
    assert [r["label"] for r in record["rows"]] == ["Or22a", "Or47b"] and "1 CID value not understood" in record["notes"]
    # blank rows and columns with labels are all missing values; blank ones without are left out
    record, values = reference.parse(write(tmp_path / "f.csv", "chemical,banana,,kiwi\nea,1,,\nhex,,,\n"))
    assert [c["label"] for c in record["cols"]] == ["banana", "kiwi"] and record["shape"] == [2, 2] and record["notes"] == []
    assert np.isnan(values[1]).all()
    # parquet, with numbers as numbers and missing cells as pandas' NA
    df = pd.DataFrame({"chemical": ["linalool", "hexanal"], "cid": pd.array([6549, None], dtype="Int64"),
                       "banana": pd.array([1.5, None], dtype="Float64")})
    df.to_parquet(tmp_path / "e.parquet")
    record, values = reference.parse(tmp_path / "e.parquet")
    assert record["ids"]["rows"] == [{"cid": 6549}, {}] and [r["label"] for r in record["rows"]] == ["linalool", "hexanal"]
    assert values[0, 0] == 1.5 and np.isnan(values[1, 0]) and record["notes"] == []
    # unless told otherwise; and files with nothing to show
    record, _ = reference.parse(p, chemical_axis="none", kind="sequential")
    assert record["ids"] == {"rows": None, "cols": None} and record["kind"] == "sequential"
    with pytest.raises(reference.ReferenceError, match="no numbers"):
        reference.parse(write(tmp_path / "c.csv", "a,b\nx,y\n"))
    with pytest.raises(reference.ReferenceError, match="not a table"):
        reference.parse(write(tmp_path / "d.xlsx", ""))


def test_match_chemical(env):
    ix = index(env)
    by = lambda ids, label="": ix.match_chemical(ids, label)  # noqa: E731
    ea, lin, hexanal = (next(o["id"] for o in ix.by_id.values() if o["name"] == n) for n in ("ethyl acetate", "linalool", "hexanal"))
    assert by({"cid": 8857}, "whatever") == (ea, "cid")
    assert by({"inchi": INCHI_LIN}) == (lin, "inchi")
    assert by({"cas": "141-78-6"}) == (ea, "cas")
    assert by({}, "Ethyl acetate") == (ea, "name") and by({}, "EA") == (ea, "abbrev") and by({}, "caproaldehyde") == (hexanal, "name")
    # a CID in the file that the table does not have: other identifiers, if they do not contradict it
    assert by({"cid": 999}, "hexanal") == (hexanal, "name")  # the table has no CID for hexanal
    assert by({"cid": 999}, "ethyl acetate") == (None, "none")  # its CID is 8857
    assert by({"inchi": "InChI=1S/other"}, "linalool") == (None, "none")
    assert by({}, "nothing") == (None, "none")


def test_builtin_and_chemicals(env):
    status, r = call(env.reference_list)
    fruits, gone = r["datasets"]
    assert fruits["state"] == "ok" and fruits["origin"] == "builtin" and fruits["shape"] == [2, 2]
    # no identifiers: chemicals found by their labels (ethyl acetate has a CID; hexanal, in the table, has none)
    assert fruits["chemicals"] == {"axis": "rows", "n": 2, "matched": 2, "identified": 1}
    assert fruits["warning"] == "1 of 2 chemicals could not be identified: no CID or InChI, in the file or the Odors table"
    assert gone["state"] == "error" and gone["error"] == "file missing from reference_data/"
    status, r = call(env.reference_get, path_params={"ref_id": "fruits"})
    assert status == 200 and r["rows"][0]["chem"] == {"name": "ethyl acetate", "abbrev": "ea", "cid": 8857, "how": "name"}
    assert r["rows"][1]["chem"]["cid"] is None and r["meta"]["kind"] == "sequential"
    values = np.frombuffer(base64.b64decode(r["values_b64"]), "<f4").reshape(r["shape"])
    assert values[0].tolist() == [1, 2] and np.isnan(values[1, 1])
    assert call(env.reference_get, path_params={"ref_id": "gone"})[1]["state"] == "error"
    assert call(env.reference_get, path_params={"ref_id": "nothing"})[0] == 404


def test_add_readable(env, tmp_path):
    src = write(tmp_path / "refs" / "volatiles.csv", "chemical,cid,banana\nlinalool,6549,1\n2-heptanone,,2\nfoo,,3\n")
    sandbox.activate({"read_dirs": [str(tmp_path / "refs")], "write_dirs": []})
    check = lambda path, **kw: call(env.admin_reference_check, body={"path": path, **kw})[1]  # noqa: E731
    c = check(str(src))
    assert c["ok"] and c["readable"] and not c["restart"] and c["grant"] is None
    assert c["preview"]["chemicals"] == {"axis": "rows", "n": 3, "matched": 2, "identified": 2, "unidentified": ["foo"], "ambiguous": []}
    assert check(str(tmp_path / "refs"))["error"] == "not a table it can read: give a .csv, .tsv, .tab, .txt, .parquet or .pq file"
    assert "not found" in check(str(tmp_path / "refs/none.csv"))["error"] and "absolute" in check("refs/x.csv")["error"]
    add = lambda **kw: call(env.admin_reference_add, body=kw)  # noqa: E731
    status, r = add(path=str(src))
    assert status == 422 and r["errors"] == ["give it a title"]
    status, r = add(path=str(src), title="Fruit volatiles (2024)", chemical_axis="rows")
    assert status == 200 and r == {"ok": True, "id": "fruit-volatiles-2024", "state": "ok", "error": None, "grant": None,
                                   "restart": False, "steps": []}
    assert add(path=str(src), title="Fruit volatiles (2024)")[1]["id"] == "fruit-volatiles-2024-2"
    cache = config.REFERENCE_CACHE_DIR / "fruit-volatiles-2024"
    assert sorted(p.name for p in cache.iterdir()) == ["import.json", "values.f32"]
    status, r = call(env.reference_get, path_params={"ref_id": "fruit-volatiles-2024"})
    assert [row.get("chem", {}).get("name") for row in r["rows"]] == ["linalool", "2-heptanone", None]
    assert r["warning"].startswith("1 of 3 chemicals") and r["meta"]["kind"] == "sequential"
    # imported again when the file changes, and an error once it cannot be read as a table
    imported = json.loads((cache / "import.json").read_text())["imported_at"]
    write(src, "chemical,cid,banana,apple\nlinalool,6549,1,5\n")
    assert call(env.reference_get, path_params={"ref_id": "fruit-volatiles-2024"})[1]["shape"] == [1, 2]
    write(src, "nothing,here\n")
    item = next(d for d in call(env.reference_list)[1]["datasets"] if d["id"] == "fruit-volatiles-2024")
    assert item["state"] == "error" and "no rows" in item["error"]
    # deleting: its import goes, its file stays
    status, r = call(env.admin_reference_delete, path_params={"ref_id": "fruit-volatiles-2024"})
    assert status == 200 and r == {"ok": True, "revoked": None} and not cache.exists() and src.exists()
    assert call(env.reference_get, path_params={"ref_id": "fruit-volatiles-2024"})[0] == 404
    assert call(env.admin_reference_delete, path_params={"ref_id": "fruits"})[0] == 409
    assert call(env.admin_reference_delete, path_params={"ref_id": "nothing"})[0] == 404
    assert imported and src.read_text() == "nothing,here\n"


def test_add_outside(env, tmp_path):
    src = write(tmp_path / "elsewhere" / "presence.csv", "pathway,linalool,ea\ncid,6549,8857\nterpenes,1,0\n")
    other = write(tmp_path / "elsewhere" / "other.csv", "chemical,x\nlinalool,1\n")
    sandbox.activate(sandbox.load())  # tensor only
    c = call(env.admin_reference_check, body={"path": str(src)})[1]
    assert c["ok"] and not c["readable"] and c["restart"] and c["grant"] == str(src)
    assert c["steps"] == ["Restart the server by hand: it is not run by Docker or systemd, which would start it again (DEPLOY.md)."]
    status, r = call(env.admin_reference_add, body={"path": str(src), "title": "Pathways"})
    assert status == 200 and r["state"] == "pending" and r["grant"] == str(src) and r["restart"]
    assert sandbox.load()["read_dirs"] == [str(config.TENSOR_ROOT), str(src)]
    assert call(env.admin_reference_add, body={"path": str(other), "title": "Other"})[1]["grant"] == str(other)
    assert call(env.reference_get, path_params={"ref_id": "pathways"})[1]["state"] == "pending"
    # after a restart: read, the columns being its chemicals (an identifier row)
    sandbox.activate(sandbox.load())
    status, r = call(env.reference_get, path_params={"ref_id": "pathways"})
    assert r["state"] == "ok" and [c["chem"]["abbrev"] for c in r["cols"]] == ["lin", "ea"] and r["warning"] is None
    # a file taken off the folders it may read is an error after the next restart
    sandbox.save({"read_dirs": [str(config.TENSOR_ROOT), str(src)], "write_dirs": []})
    sandbox.activate(sandbox.load())
    item = next(d for d in call(env.reference_list)[1]["datasets"] if d["id"] == "other")
    assert item["state"] == "error" and item["error"].startswith("not in the folders the server may read")
    # deleting takes its file off the folders it may read again
    assert call(env.admin_reference_delete, path_params={"ref_id": "pathways"})[1] == {"ok": True, "revoked": str(src)}
    assert sandbox.load()["read_dirs"] == [str(config.TENSOR_ROOT)] and src.exists()


def test_access_steps(monkeypatch, tmp_path):
    f = write(tmp_path / "x.csv", "")
    assert sandbox.access_steps(str(f), "Docker") == ["Restart the server (below)."]
    assert "<code>- /data/refs:/data/refs:ro</code>" in sandbox.access_steps("/data/refs/x.csv", "Docker")[0]
    assert "ProtectHome=read-only" in sandbox.access_steps("/home/me/x.csv", "systemd")[0]
    assert "BindReadOnlyPaths=-/mnt/other" in sandbox.access_steps("/mnt/other/x.csv", "systemd")[0]
    monkeypatch.setattr(config, "TENSOR_ROOT", config.Path("/local/tensor"))
    assert "BindReadOnlyPaths=-/local/other" in sandbox.access_steps("/local/other/x.csv", "systemd")[0]
    assert sandbox.access_steps("/local/tensor/nothing/x.csv", "systemd")[0] == "It is not on the server now: put it there first."
    assert sandbox.access_steps("/srv/nothing/x.csv", "systemd") == ["It is not on the server now: put it there first.",
                                                                      "Restart the server (below)."]
