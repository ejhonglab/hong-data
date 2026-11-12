"""Tests of reference datasets (built in, and added in the Admin tab). Run with:
OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import asyncio
import base64
import gzip
import json
import zipfile
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from starlette.requests import Request

from app import config, odormeta, reference, sandbox
from tests.test_tensor import call, main, post, root  # noqa: F401 (fixtures)

SHEET = "name,abbrev,aliases,cid,cas\nethyl acetate,ea,,8857,141-78-6\n2-heptanone,2h,,8051,\nhexanal,hex,caproaldehyde,,\nlinalool,lin,,6549,\n"
INCHI_LIN = "InChI=1S/C10H18O/c1-5-10(4,11)8-6-7-9(2)3/h5,7,11H,1,6,8H2,2-4H3"


@pytest.fixture
def env(main, tmp_path, monkeypatch):  # noqa: F811
    """main, with an odor table, a built-in reference dataset, and the app's folders in tmp_path."""
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
    main.db.apply_odor_sync(added=rows, changed=[], deleted=[], source="test", fingerprint="0",
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


MENTHONE = "InChI=1S/C10H18O/c1-7(2)9-5-4-8(3)6-10(9)11/h7-9H,4-6H2,1-3H3"  # (RDKit's, of each)
L_MENTHONE, D_MENTHONE = MENTHONE + "/t8-,9+/m1/s1", MENTHONE + "/t8-,9+/m0/s1"  # (2S,5R), (2R,5S)
D_ISOMENTHONE = MENTHONE + "/t8-,9-/m1/s1"  # (2R,5R)
HEXENAL = "InChI=1S/C6H10O/c1-2-3-4-5-6-7/h4-6H,2-3H2,1H3"
S_LINALOOL = INCHI_LIN + "/t10-/m1/s1"


def test_stereo_compatible():
    ok = odormeta.stereo_compatible
    # stereocenters: any where either leaves them unspecified, else only alike
    assert ok(MENTHONE, L_MENTHONE) and ok(D_ISOMENTHONE, MENTHONE) and ok(L_MENTHONE, L_MENTHONE)
    assert not ok(L_MENTHONE, D_ISOMENTHONE) and not ok(L_MENTHONE, D_MENTHONE)
    # one of two given (RDKit's): what has it so (C5 as l-menthone's: d-isomenthone's too; C2 as l-menthone's: neither d- one's);
    # relative (trans, either enantiomer, by hand): menthone, not isomenthone
    c5, c2 = MENTHONE + "/t8-,9?/m1/s1", MENTHONE + "/t8?,9-/m0/s1"
    assert ok(c5, L_MENTHONE) and ok(c5, D_ISOMENTHONE) and not ok(c5, D_MENTHONE)
    assert ok(c2, L_MENTHONE) and not ok(c2, D_ISOMENTHONE) and not ok(c2, D_MENTHONE) and ok(c2, c5)
    trans = MENTHONE + "/t8-,9+/s2"
    assert ok(trans, L_MENTHONE) and ok(trans, D_MENTHONE) and not ok(trans, D_ISOMENTHONE)
    # cis/trans alike; another structure, or one of several components (only itself), never
    assert ok(HEXENAL, HEXENAL + "/b5-4+") and not ok(HEXENAL + "/b5-4-", HEXENAL + "/b5-4+")
    assert not ok(MENTHONE, INCHI_LIN) and not ok("InChI=1S/C2H4O2.Na/c1-2(3)4;/h1H3,(H,3,4);/q;+1/p-1", "InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)")
    assert odormeta.inchi_skeleton(L_MENTHONE) == MENTHONE == odormeta.inchi_skeleton(MENTHONE)


def test_match_stereo():
    menthone = {"id": 1, "name": "menthone", "abbrev": "men", "cid": 6986, "inchi": MENTHONE}
    t2h = {"id": 2, "name": "trans-2-hexenal", "abbrev": "t2h", "cid": 5281168, "inchi": HEXENAL + "/b5-4+"}
    ix = odormeta.OdorIndex([menthone, t2h])
    # as the same but for stereochemistry, once neither CID nor InChI is the odor's (whatever CID is given)
    assert ix.match_chemical({"cid": 26447, "inchi": L_MENTHONE}, "dl-menthone") == (1, "stereo")
    assert ix.match_chemical({"cid": 6986, "inchi": MENTHONE}, "isomenthone") == (1, "cid")
    assert ix.match_chemical({"inchi": HEXENAL}, "2-hexenal") == (2, "stereo")
    assert ix.match_chemical({"inchi": HEXENAL + "/b5-4-"}, "(Z)-2-hexenal") == (None, "none")
    # the odor of its own InChI first; of two it could be, neither
    ix = odormeta.OdorIndex([menthone, {**menthone, "id": 3, "name": "l-menthone", "abbrev": "lmen", "cid": 26447, "inchi": L_MENTHONE}])
    assert ix.match_chemical({"inchi": L_MENTHONE}, "") == (3, "inchi") and ix.match_chemical({"inchi": D_ISOMENTHONE}, "") == (1, "stereo")
    ix = odormeta.OdorIndex([{**menthone, "inchi": L_MENTHONE}, {**menthone, "id": 3, "name": "d-isomenthone", "inchi": D_ISOMENTHONE}])
    cands = []
    assert ix.match_chemical({"inchi": MENTHONE}, "", cands) == (None, "ambiguous") and cands == [1, 3]


def test_maybe():
    # by name, but for what tells isomers apart: of chemicals with no structure given (VCF's "(unkn.str.)")
    stems = odormeta.name_stems
    assert stems("methylphenol (unkn.str.) (=cresol (unkn.str.))") == {"methylphenol", "cresol"}
    assert stems("(E)-2-hexenal") == stems("trans-2-hexenal") == {"hexenal"} and stems("p-1-menthen-8-ol") == {"menthenol"}
    assert stems("alpha-terpineol") == stems("α-terpineol (=p-1-menthen-8-ol)") - {"menthenol"} == {"terpineol"}
    assert stems("(1S,5R)-(+)-sabinene") == {"sabinene"} and "butanone" not in stems("x (=4-(3-hydroxy-2-methylphenyl)-2-butanone)")
    ix = odormeta.OdorIndex([{"id": 1, "name": "1-pentanol", "abbrev": "1-5ol"}, {"id": 2, "name": "3-octanol", "abbrev": "3oct"},
                             {"id": 3, "name": "p-cresol", "abbrev": "pcre", "aliases": ["4-methylphenol"]}])
    assert ix.maybe("pentanol (unkn.str.)") == [1] and ix.maybe("octanol (unkn.str.)") == [2] and ix.maybe("hexanol (unkn.str.)") == []
    assert ix.maybe("methylphenol (unkn.str.) (=cresol (unkn.str.))") == [3]


def test_list_stereo(env, tmp_path):
    # linalool (in the odor table with no stereocenter given) twice: as itself and as one enantiomer
    p = write(tmp_path / "refs" / "vcf.csv", "name,inchi,product_name,reference,quantity\n"
              f'linalool,"{INCHI_LIN}",APPLE,A,1\n(S)-linalool,"{S_LINALOOL}",APPLE,A,2\n(S)-linalool,"{S_LINALOOL}",BEER,B,4\n'
              "ethyl acetate,,BEER,B,3\nmystery,,BEER,B,5\nheptanone (unkn.str.),,BEER,B,6\n")
    sandbox.activate({"read_dirs": [str(tmp_path / "refs")], "write_dirs": []})
    fields = {"layout": "list", "rows": {"label": "name", "keep": ["inchi"]}, "cols": {"label": "product_name", "keep": ["reference"]},
              "value": "quantity", "empty": "missing", "merge": "sum"}
    c = call(env.admin_reference_check, body={"path": str(p), "fields": fields})[1]
    assert c["fields"]["used"]["merge"] == "sum"
    full = ("1 of the 2 odors of the Odors table it has (each once, its rows of one merged) is matched only as the same but for "
            "stereochemistry that one of the two leaves unspecified: linalool, as (S)-linalool. "
            "Its rows so matched have 2 of the 4 values (not 0 or missing) of those 2 (50%), and of its 6 (33%).")
    assert c["preview"]["stereo_warning"] == full
    assert call(env.admin_reference_add, body={"path": str(p), "title": "VCF", "fields": fields})[0] == 200
    listed = next(r for r in call(env.reference_list)[1]["datasets"] if r["id"] == "vcf")
    assert listed["stereo"] == "1 of its 2 odors of the Odors table matched only as the same but for stereochemistry (50% of their values)"
    assert listed["chemicals"]["odor_ids"] == [4, 4, 1]
    r = call(env.reference_get, path_params={"ref_id": "vcf"})[1]
    assert r["stereo_warning"] == full and r["chemicals"]["merge"] == "sum"
    assert [x["chem"]["how"] for x in r["rows"]] == ["inchi", "stereo", "name", "none", "none"]
    # what is not resolved: linalool given with its (S)- enantiomer in one product (its total?), a heptanone of no structure
    assert r["unresolved_warning"] == (
        "2 odors of the Odors table it has, or may have, may be miscounted, unresolved: 2-heptanone: heptanone (unkn.str.), "
        "of no structure given, may be it (not counted as it). linalool: in 1 of its columns, linalool, (S)-linalool, which may "
        "be one another's total or the same, not distinct isomers (summed, so counted twice if so).")
    # none so matched: no warning; merged as averaged unless said
    fields = {**fields, "rows": {"label": "name", "keep": []}}
    fields.pop("merge")
    status, _ = call(env.admin_reference_update, path_params={"ref_id": "vcf"}, body={"fields": fields})
    r = call(env.reference_get, path_params={"ref_id": "vcf"})[1]
    assert status == 200 and r["stereo_warning"] is None and r["chemicals"]["merge"] == "mean"


def test_builtin_and_chemicals(env):
    status, r = call(env.reference_list)
    fruits, gone = r["datasets"]
    assert fruits["state"] == "ok" and fruits["origin"] == "builtin" and fruits["shape"] == [2, 2]
    # no identifiers: chemicals found by their labels (ethyl acetate has a CID; hexanal, in the table, has none)
    assert fruits["chemicals"] == {"axis": "rows", "n": 2, "matched": 2, "identified": 1, "odor_ids": [1, 3]}
    assert fruits["warning"] == "1 of 2 chemicals could not be identified: no CID or InChI, in the file or the Odors table"
    assert gone["state"] == "error" and gone["error"] == "file missing from reference_data/"
    status, r = call(env.reference_get, path_params={"ref_id": "fruits"})
    assert status == 200 and r["rows"][0]["chem"] == {"odor_id": 1, "name": "ethyl acetate", "abbrev": "ea", "cid": 8857,
                                                      "how": "name", "identified": True}
    assert "odor_check" not in r
    assert r["rows"][1]["chem"]["cid"] is None and r["meta"]["kind"] == "sequential" and r["meta"]["scale"] is None
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
    assert c["preview"]["chemicals"] == {"axis": "rows", "n": 3, "matched": 2, "identified": 2, "unidentified": ["foo"], "ambiguous": [],
                                         "odors": False}
    assert "odor_labels" not in c["preview"] and "odor_check" not in c
    assert check(str(tmp_path / "refs"))["error"] == ("not a table it can read: give a .csv, .tsv, .tab, .txt, .parquet, .pq, .nc, .nc4 "
                                                       "or .netcdf file, or a MetaCyc or LOTUS bundle (.sqlite, .sqlite3, .db, "
                                                       "made by app/metacyc_build.py or app/lotus_build.py)")
    assert "not found" in check(str(tmp_path / "refs/none.csv"))["error"]
    assert check("refs/x.csv")["path"] == f"{config.TENSOR_ROOT}/refs/x.csv"  # (from the top of tensor: see test_from_tensor)
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
    assert status == 200 and r == {"ok": True, "revoked": None, "moved": None} and not cache.exists() and src.exists()
    assert call(env.reference_get, path_params={"ref_id": "fruit-volatiles-2024"})[0] == 404
    assert call(env.admin_reference_delete, path_params={"ref_id": "fruits"})[0] == 409
    assert call(env.admin_reference_delete, path_params={"ref_id": "nothing"})[0] == 404
    assert imported and src.read_text() == "nothing,here\n"


def test_from_tensor(env):
    # a path not absolute is from the top of tensor, kept as the absolute path
    src = write(config.TENSOR_ROOT / "tom" / "refs" / "volatiles.csv", "chemical,banana\nlinalool,1\n")
    sandbox.activate({"read_dirs": [str(config.TENSOR_ROOT)], "write_dirs": []})
    c = call(env.admin_reference_check, body={"path": " tom/refs/volatiles.csv"})[1]
    assert c["ok"] and c["readable"] and c["path"] == str(src) and c["preview"]["shape"] == [1, 1]
    assert "'..' is not allowed" in call(env.admin_reference_check, body={"path": "tom/../x.csv"})[1]["error"]
    status, r = call(env.admin_reference_add, body={"path": "tom/refs/volatiles.csv", "title": "Volatiles"})
    assert status == 200 and r["state"] == "ok"
    item = next(d for d in call(env.reference_list)[1]["datasets"] if d["id"] == "volatiles")
    assert item["path"] == str(src)


def test_examples(env, tmp_path):
    # the built-in examples only while there is no other reference dataset, built in or added
    (config.REFERENCE_DIR / "example.csv").write_text("chemical,banana\nethyl acetate,1\n")
    examples = [{"id": "example", "title": "EXAMPLE", "file": "example.csv", "example": True}]
    manifest = lambda datasets: (config.REFERENCE_DIR / "manifest.json").write_text(json.dumps({"datasets": datasets}))  # noqa: E731
    ids = lambda: [d["id"] for d in call(env.reference_list)[1]["datasets"]]  # noqa: E731
    manifest(examples)
    assert ids() == ["example"] and call(env.reference_get, path_params={"ref_id": "example"})[0] == 200
    manifest(examples + [{"id": "fruits", "title": "Fruits", "file": "fruits.csv"}])
    assert ids() == ["fruits"] and call(env.reference_get, path_params={"ref_id": "example"})[0] == 404
    manifest(examples)
    src = write(tmp_path / "refs" / "volatiles.csv", "chemical,cid,banana\nlinalool,6549,1\n")
    sandbox.activate({"read_dirs": [str(tmp_path / "refs")], "write_dirs": []})
    assert call(env.admin_reference_add, body={"path": str(src), "title": "Volatiles"})[0] == 200
    assert ids() == ["volatiles"]


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
    assert call(env.admin_reference_delete, path_params={"ref_id": "pathways"})[1] == {"ok": True, "revoked": str(src), "moved": None}
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


def test_glomeruli(env, tmp_path):
    p = write(tmp_path / "g.csv", "glomerulus,kc a,kc b\nDM2,1,0\nva2,0,1\nDL5,2,2\ntotal,3,3\n")
    record, _ = reference.parse(p)
    # by labels, but for case; not every label need be one
    assert reference.glomeruli(record) == {"axis": "rows", "names": ["DM2", "VA2", "DL5", None], "n": 4, "matched": 3}
    chem = reference.chemicals(record, odormeta.OdorIndex(env.db.list_odors()))
    assert chem["unrecognized"] and reference.warning(chem, reference.glomeruli(record)) is None
    # on columns too; too few (or under half of the labels) are none
    record, _ = reference.parse(write(tmp_path / "h.csv", "x,DA1,DA2,DA3\na,1,2,3\n"))
    assert reference.glomeruli(record)["axis"] == "cols"
    for text in ("x,DA1,DA2,y\na,1,2,3\n", "x,DA1,DA2,DA3,p,q,r,s\nt,1,2,3,4,5,6,7\n"):
        assert reference.glomeruli(reference.parse(write(tmp_path / "i.csv", text))[0])["axis"] is None
    # listed with them, and served with each item's
    (config.REFERENCE_DIR / "glom.csv").write_text(p.read_text())
    manifest = json.loads((config.REFERENCE_DIR / "manifest.json").read_text())
    manifest["datasets"].append({"id": "glom", "title": "Glomeruli", "file": "glom.csv"})
    (config.REFERENCE_DIR / "manifest.json").write_text(json.dumps(manifest))
    item = next(d for d in call(env.reference_list)[1]["datasets"] if d["id"] == "glom")
    assert item["glomeruli"] == {"axis": "rows", "n": 4, "matched": 3, "names": ["DM2", "VA2", "DL5"]} and item["warning"] is None
    r = call(env.reference_get, path_params={"ref_id": "glom"})[1]
    assert [x.get("glomerulus") for x in r["rows"]] == ["DM2", "VA2", "DL5", None] and r["glomeruli"]["axis"] == "rows"


def test_odor_strings(env, tmp_path):
    # odors as datasets write them, by glomeruli: the columns its chemicals, matched by abbreviation
    text = "glomerulus,2h @ -3,ea @ -2,hex @ -1.5,lin @ -3,pa @ -3\nDM2,1,2,3,4,5\nVA2,0,1,0,1,0\nDL5,2,2,2,2,2\n"
    src = write(tmp_path / "refs" / "orn.csv", text)
    record, _ = reference.parse(src)
    assert reference.odor_axis(record) == "cols"
    chem, gl = reference.chemicals(record, index(env)), reference.glomeruli(record)
    assert chem["axis"] == "cols" and chem["odors"] and gl["axis"] == "rows"
    assert [(i["abbrev"], i["how"], i["conc"]) for i in chem["items"]] == [
        ("2h", "abbrev", "-3"), ("ea", "abbrev", "-2"), ("hex", "abbrev", "-1.5"), ("lin", "abbrev", "-3"), (None, "none", "-3")]
    assert chem["matched"] == 4 and chem["unidentified"] == ["hex @ -1.5", "pa @ -3"]
    assert reference.warning(chem, gl) == ("1 of its 5 odors is not one odor in the Odors table (by abbreviation, as a dataset's are): "
                                           "add it to the odor sheet and sync it")
    # too few odor strings, or under half of an axis's labels, are none; a name is not an abbreviation
    for t in ("x,ea @ -2,2h @ -3,y\na,1,2,3\n", "x,ea @ -2,2h @ -3,lin @ -3,p,q,r,s\nt,1,2,3,4,5,6,7\n"):
        assert reference.odor_axis(reference.parse(write(tmp_path / "o.csv", t))[0]) is None
    names = reference.parse(write(tmp_path / "n.csv", "glomerulus,ethyl acetate @ -2,2h @ -3,lin @ -3\nDM2,1,2,3\n"))[0]
    assert [i["how"] for i in reference.chemicals(names, index(env))["items"]] == ["name", "abbrev", "abbrev"]
    # the Admin tab's check: with what the odor sheet needs, as an upload's check says it
    sandbox.activate({"read_dirs": [str(tmp_path / "refs")], "write_dirs": []})
    c = call(env.admin_reference_check, body={"path": str(src)})[1]
    assert c["preview"]["chemicals"]["odors"] and "odor_labels" not in c["preview"]
    assert [m["abbrev"] for m in c["odor_check"]["missing"]] == ["pa"] and c["odor_check"]["sheet"]["paste"]
    assert [r["match"] for r in c["odor_check"]["rows"]] == ["abbrev"] * 4 + ["none"]
    # and the Reference data tab's, once added
    assert call(env.admin_reference_add, body={"path": str(src), "title": "ORN deltas"})[0] == 200
    r = call(env.reference_get, path_params={"ref_id": "orn-deltas"})[1]
    assert r["chemicals"]["odors"] and r["warning"].startswith("1 of its 5 odors")
    assert [m["abbrev"] for m in r["odor_check"]["missing"]] == ["pa"]
    assert [c["chem"]["conc"] for c in r["cols"]] == ["-3", "-2", "-1.5", "-3", "-3"]
    # once the odor is in the odor table: all of them
    rows, _, _ = odormeta.parse_sheet("name,abbrev,aliases,cid,cas\npentyl acetate,pa,,12348,\n")
    env.db.apply_odor_sync(added=rows, changed=[], deleted=[], source="test", fingerprint="1",
                           details={}, when="2026-10-19T00:00:00")
    r = call(env.reference_get, path_params={"ref_id": "orn-deltas"})[1]
    assert r["chemicals"]["matched"] == 5 and not r["odor_check"]["missing"] and r["warning"].startswith("1 of 5 chemicals")


def test_download(env, tmp_path, monkeypatch):
    # one value per line (as VCF's): a level for each field of its rows and of its columns
    src = write(tmp_path / "refs" / "vcf.csv", "compound,cid,product,reference,amount\nlinalool,6549,apple,A 2001,1\n"
                                               "linalool,6549,apple,B 2003,3\nethyl acetate,8857,apple,A 2001,2\nfoo,,pear,A 2001,5\n")
    sandbox.activate({"read_dirs": [str(tmp_path / "refs")], "write_dirs": []})
    fields = {"layout": "list", "rows": {"label": "compound", "keep": ["cid"]}, "cols": {"label": "product", "keep": ["reference"]},
              "value": "amount"}
    assert call(env.admin_reference_add, body={"path": str(src), "title": "VCF", "fields": fields})[0] == 200
    from app import matrix_download

    def wide(**picked):
        dest = tmp_path / "wide.csv"
        assert matrix_download.reference_download(env.db, "vcf", index(env), "wide", dest, **picked) == ("olfdb_reference-vcf.csv", "text/csv")
        return dest

    df = pd.read_csv(wide(), header=[0, 1], index_col=[0, 1, 2, 3])
    assert df.index.names == ["compound", "cid", "odor_table_name", "odor_table_abbrev"] and df.columns.names == ["product", "reference"]
    assert df.columns.tolist() == [("apple", "A 2001"), ("apple", "B 2003"), ("pear", "A 2001")]
    text = lambda ix, level: [x if isinstance(x, str) else None for x in ix.get_level_values(level)]  # noqa: E731
    assert text(df.index, "compound") == ["linalool", "ethyl acetate", "foo"] and text(df.index, "odor_table_abbrev") == ["lin", "ea", None]
    assert np.array_equal(df.to_numpy(), [[1, 3, np.nan], [2, np.nan, np.nan], [np.nan, np.nan, 5]], equal_nan=True)
    # as shown: of rows and columns picked, in that order, transposed
    df = pd.read_csv(wide(rows=[2, 0], cols=[1, 0], transpose=True), header=[0, 1, 2, 3], index_col=[0, 1])
    assert df.index.tolist() == [("apple", "B 2003"), ("apple", "A 2001")] and text(df.columns, "compound") == ["foo", "linalool"]
    assert np.array_equal(df.to_numpy(), [[np.nan, 3], [np.nan, 1]], equal_nan=True)
    with pytest.raises(matrix_download.DownloadError, match="numbered from 0 to 2"):
        wide(rows=[3])
    # a table laid out as its matrix: its first column's header names its rows' labels
    write(tmp_path / "refs" / "orn.csv", "glomerulus,ea @ -2,2h @ -3,lin @ -3\nDM2,1,2,3\nva2,4,5,6\nDL5,7,8,9\n")
    assert call(env.admin_reference_add, body={"path": str(tmp_path / "refs" / "orn.csv"), "title": "ORN"})[0] == 200
    dest = tmp_path / "orn.csv"
    matrix_download.reference_download(env.db, "orn", index(env), "wide", dest)
    df = pd.read_csv(dest, header=[0, 1, 2, 3, 4], index_col=[0, 1])
    # (and its glomeruli as the app names them, where not as labelled)
    assert df.index.names == ["glomerulus", "glomerulus_2"] and df.index.tolist() == [("DM2", "DM2"), ("va2", "VA2"), ("DL5", "DL5")]
    assert df.columns.names == ["odor", "odor_table_name", "odor_table_abbrev", "cid", "log10_conc"]
    assert df.columns.tolist()[0] == ("ea @ -2", "ethyl acetate", "ea", "8857", "-2")
    # its original files: the file as added, with a README
    zipped = tmp_path / "files.zip"
    assert matrix_download.reference_download(env.db, "vcf", index(env), "files", zipped)[0] == "olfdb_reference-vcf_files.zip"
    with zipfile.ZipFile(zipped) as zf:
        assert sorted(zf.namelist()) == ["olfdb_reference-vcf/README.md", "olfdb_reference-vcf/vcf.csv"]
        assert zf.read("olfdb_reference-vcf/vcf.csv").decode() == src.read_text()
        readme = zf.read("olfdb_reference-vcf/README.md").decode()
    assert readme.startswith("# VCF\n") and '"product"' in readme and "3 rows × 3 columns" in readme
    # through the API
    monkeypatch.setattr(config, "EXPORTS_DIR", tmp_path / "exports")
    config.EXPORTS_DIR.mkdir()
    r = post(env.reference_download, {"format": "wide", "rows": [0], "cols": [0, 2]}, {"ref_id": "vcf"})
    assert r.status_code == 200 and 'filename="olfdb_reference-vcf.csv"' in r.headers["content-disposition"]
    assert pd.read_csv(r.path, header=[0, 1], index_col=[0, 1, 2, 3]).shape == (1, 2)
    for body, ref_id, status in (({"format": "png"}, "vcf", 400), ({"format": "wide", "cols": 3}, "vcf", 400),
                                 ({"format": "wide", "cols": [9]}, "vcf", 422), ({"format": "files"}, "nothing", 404)):
        r = post(env.reference_download, body, {"ref_id": ref_id})
        assert r.status_code == status, body
    assert len(list(config.EXPORTS_DIR.iterdir())) == 1  # (nothing left of those refused)


def test_dataarray(env, tmp_path):
    import xarray as xr

    # as the VCF one: compounds (identified) by sources (one product's twice, told apart)
    v = np.array([[1.0, np.nan, 2.0, 0.5], [np.nan, 3.0, np.nan, np.nan], [4.0, 4.0, np.nan, 1.0]])
    da = xr.DataArray(v, dims=["compound", "source"], coords={
        "name": ("compound", ["ethyl acetate", "δ-terpineol", "linalool"]), "cid": ("compound", np.array([8857, 81722, 6549])),
        "vcf_url": ("compound", [f"https://vcf.example/{i}" for i in range(3)]),
        "reference": ("source", ["Ref A", "Ref B", "Ref C", "Ref D"]),
        "product_name": ("source", ["COGNAC", "COGNAC", "APPLE", "ROSEMARY"]),
        "product_category": ("source", ["BRANDY", "BRANDY", "APPLE", "HERB"]),
        "product_group": ("source", ["Alcoholic", "Alcoholic", "Plants", "Plants"])})
    p = tmp_path / "vcf.nc"
    da.to_netcdf(p)
    record, values = reference.parse(p)
    assert values.tolist() == [[1, 0, 2, 0.5], [0, 3, 0, 0], [4, 4, 0, 1]] and record["notes"] == ["5 missing values taken as 0 (absent)"]
    assert [r["label"] for r in record["rows"]] == ["ethyl acetate", "δ-terpineol", "linalool"]
    # a product's name, but where two sources share it, what differs between them
    assert [c["label"] for c in record["cols"]] == ["COGNAC — Ref A", "COGNAC — Ref B", "APPLE", "ROSEMARY"]
    # (coarsest first: the fewest values)
    assert record["cols"][0]["detail"] == "product group: Alcoholic · product category: BRANDY · reference: Ref A"
    assert record["rows"][0]["detail"] is None  # (its link left out)
    assert record["ids"] == {"rows": [{"name": "ethyl acetate", "cid": 8857}, {"name": "δ-terpineol", "cid": 81722},
                                      {"name": "linalool", "cid": 6549}], "cols": None}
    chem = reference.chemicals(record, odormeta.OdorIndex(env.db.list_odors()))
    assert chem["axis"] == "rows" and chem["matched"] == 2 and chem["identified"] == 3
    record, _ = reference.parse(p, fields={"rows": {"label": "name", "keep": []}, "cols": {"label": "product_name", "keep": ["product_group"]}})
    assert [c["label"] for c in record["cols"]] == ["COGNAC (1)", "COGNAC (2)", "APPLE", "ROSEMARY"]  # (no reference to tell them apart)
    assert record["cols"][0]["detail"] == "product group: Alcoholic" and record["ids"]["rows"][0] == {"name": "ethyl acetate"}
    d = reference.describe_fields(p)
    assert d["kind"] == "dataarray" and d["dims"] == ["compound", "source"]
    assert d["suggested"]["cols"] == {"label": "product_name", "keep": ["reference", "product_category", "product_group"]}
    with pytest.raises(reference.ReferenceError, match="'nope'"):
        reference.parse(p, fields={"rows": {"label": "nope"}})
    xr.DataArray(np.zeros((2, 2, 2))).to_netcdf(tmp_path / "three.nc")
    with pytest.raises(reference.ReferenceError, match="3 dimensions"):
        reference.parse(tmp_path / "three.nc")

    # added in the Admin tab, and served with each item's detail; large, gzipped if asked
    big = xr.DataArray(np.where(np.random.default_rng(0).random((600, 600)) < 0.1, 1.0, np.nan), dims=["compound", "source"],
                       coords={"cid": ("compound", np.arange(1, 601))})
    big.to_netcdf(tmp_path / "big.nc")
    for path, title in ((p, "VCF"), (tmp_path / "big.nc", "Big")):
        status, r = call(env.admin_reference_add, body={"path": str(path), "title": title})
        assert status == 200, r
    r = call(env.reference_get, path_params={"ref_id": "vcf"})[1]
    assert r["cols"][1] == {"label": "COGNAC — Ref B", "detail": "product group: Alcoholic · product category: BRANDY · reference: Ref B"}
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [(b"accept-encoding", b"gzip, br")], "query_string": b"",
             "path_params": {"ref_id": "big"}}
    res = asyncio.run(env.reference_get(Request(scope)))
    assert res.headers["content-encoding"] == "gzip" and json.loads(gzip.decompress(res.body))["shape"] == [600, 600]


def test_list(env, tmp_path, monkeypatch):
    # one value per line, as VCF's: compounds (an InChI without its "InChI="), products, references, and quantities
    p = write(tmp_path / "vcf.csv", "inchi,name,quantity,product_name,reference,vcf_url\n"
              '"1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3",ethyl acetate,0.5,APPLE,Ref A,https://x/1\n'
              ",ethyl acetate,1.5,APPLE,Ref B,https://x/2\n"
              ",linalool,q,APPLE,Ref A,https://x/3\n"
              ",linalool,2,BEER,Ref C,https://x/4\n"
              ",,7,BEER,Ref C,https://x/5\n")
    d = reference.describe_fields(p)
    assert d["kind"] == "table" and [f["name"] for f in d["fields"]][:3] == ["inchi", "name", "quantity"]
    quantity = next(f for f in d["fields"] if f["name"] == "quantity")
    assert quantity["distinct"] == 5 and quantity["numeric"] == 0.8 and quantity["samples"] == ["0.5", "1.5", "q"]
    assert d["suggested"] == {"layout": "list", "rows": {"label": "name", "keep": ["inchi"]}, "cols": {"label": "product_name", "keep": []},
                              "value": "quantity", "empty": "missing"}
    # each name a row (the line with none left out), each product a column; several lines of one cell, their mean
    spec = {"layout": "list", "rows": {"label": "name", "keep": []}, "cols": {"label": "product_name", "keep": []}, "value": "quantity"}
    record, values = reference.parse(p, fields=spec)
    assert [r["label"] for r in record["rows"]] == ["ethyl acetate", "linalool"] and [c["label"] for c in record["cols"]] == ["APPLE", "BEER"]
    assert values[0, 0] == 1.0 and np.isnan(values[0, 1]) and np.isnan(values[1, 0]) and values[1, 1] == 2
    assert record["notes"] == ["1 line with no 'name' or 'product_name' left out", "1 value is not a number (e.g. 'q'): missing",
                               "1 cell is on more than one line: the mean of their numbers"]
    assert record["ids"] == {"rows": [{"name": "ethyl acetate"}, {"name": "linalool"}], "cols": None}
    # or presence; cells no line lists 0; the columns of products told apart by reference where need be
    record, values = reference.parse(p, fields={**spec, "value": None, "cols": {"label": "product_name", "keep": ["reference"]}})
    assert record["kind"] == "binary" and values.tolist() == [[1, 1, 0], [1, 0, 1]]
    assert [c["label"] for c in record["cols"]] == ["APPLE — Ref A", "APPLE — Ref B", "BEER"] and record["cols"][2]["detail"] == "reference: Ref C"
    record, values = reference.parse(p, fields={**spec, "empty": "zero"})
    assert values[0, 1] == 0 and np.isnan(values[1, 0])  # (listed, but no number: still missing)
    # an InChI without "InChI=" identifies its chemical; the rows of each combination of their fields
    record, _ = reference.parse(p, fields={**spec, "rows": {"label": "name", "keep": ["inchi"]}})
    assert [r["label"] for r in record["rows"]] == ["ethyl acetate (1)", "ethyl acetate (2)", "linalool"]
    assert record["ids"]["rows"][0] == {"name": "ethyl acetate", "inchi": "InChI=1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3"}
    for bad, why in (({**spec, "cols": {"label": "name"}}, "cannot be of both"), ({**spec, "value": "nope"}, "no field of"),
                     ({**spec, "rows": {"label": None}}, "choose the fields"), ({**spec, "value": "reference"}, "has no numbers")):
        with pytest.raises(reference.ReferenceError, match=why):
            reference.parse(p, fields=bad)
    # a matrix as laid out is not taken for a list
    m = write(tmp_path / "m.csv", "chemical,cid,banana,apple\nlinalool,6549,1,2\nethyl acetate,8857,2,3\nfoo,,3,4\n")
    assert reference.describe_fields(m)["suggested"]["layout"] == "matrix"
    # too large a matrix: refused, saying how large
    monkeypatch.setattr(reference, "MAX_CELLS", 3)
    with pytest.raises(reference.ReferenceError, match="2 rows × 2 columns"):
        reference.parse(p, fields=spec)


def test_list_added(env, tmp_path):
    p = write(tmp_path / "refs" / "vcf.csv", "name,product_name,reference,quantity\nethyl acetate,APPLE,A,1\nlinalool,APPLE,B,q\n"
              "ethyl acetate,BEER,C,2\nmystery,BEER,C,3\n")
    sandbox.activate({"read_dirs": [str(tmp_path / "refs")], "write_dirs": []})
    # checked as suggested, then as chosen; added with its choices, which it is read by again
    c = call(env.admin_reference_check, body={"path": str(p)})[1]
    assert c["ok"] and c["fields"]["used"]["layout"] == "list" and c["preview"]["shape"] == [3, 2]
    fields = {"layout": "list", "rows": {"label": "name", "keep": []}, "cols": {"label": "product_name", "keep": ["reference"]}, "value": None}
    c = call(env.admin_reference_check, body={"path": str(p), "fields": fields})[1]
    assert c["preview"]["shape"] == [3, 3] and c["preview"]["kind"] == "binary"
    bad = call(env.admin_reference_check, body={"path": str(p), "fields": {**fields, "cols": {"label": "name"}}})[1]
    assert not bad["ok"] and "cannot be of both" in bad["error"] and bad["fields"]["fields"]  # (its fields, to choose again)
    status, r = call(env.admin_reference_add, body={"path": str(p), "title": "VCF", "fields": fields})
    assert status == 200 and r["state"] == "ok"
    assert json.loads(env.db.list_references()[0]["fields"]) == {**fields, "layout": "list"}
    r = call(env.reference_get, path_params={"ref_id": "vcf"})[1]
    assert r["shape"] == [3, 3] and [x["label"] for x in r["cols"]] == ["APPLE — A", "APPLE — B", "BEER"]
    # each chemical said to be identified or not (shown only if so, by default): linalool by the odor table's CID
    assert [x["chem"]["identified"] for x in r["rows"]] == [True, True, False] and r["chemicals"]["unidentified"] == ["mystery"]


def test_change_value_and_delete_all(env, tmp_path):
    # VCF's (with Henry's law constants): quantities as written (ranges, trace) and their midpoints
    p = write(tmp_path / "refs" / "vcf.csv", "name,cid,product_name,reference,quantity,mid_ppm,k_henry,group\n"
              "ethyl acetate,8857,APPLE,A,2-4,3,0.059,Fruits\nlinalool,6549,APPLE,B,1,1,0.02,Fruits\n"
              "ethyl acetate,8857,BEER,C,trace,0.01,0.059,Drinks\n")
    other = write(tmp_path / "refs" / "fruits.csv", "chemical,banana\nethyl acetate,1\n")
    sandbox.activate({"read_dirs": [str(tmp_path / "refs")], "write_dirs": []})
    fields = {"layout": "list", "rows": {"label": "name", "keep": []}, "cols": {"label": "product_name", "keep": ["reference"]},
              "value": "mid_ppm", "empty": "missing", "merge": "sum"}
    # the Headspace calculator would search it so, and not as a matrix
    c = call(env.admin_reference_check, body={"path": str(p), "fields": fields})[1]
    assert c["preview"]["headspace"] == {"used": True, "value": "mid_ppm", "merge": "sum"} and c["preview"]["value_warning"] is None
    assert next(f for f in c["fields"]["fields"] if f["name"] == "cid")["id"] == "cid"
    c = call(env.admin_reference_check, body={"path": str(p), "fields": {**fields, "layout": "matrix"}})[1]
    assert c["preview"]["headspace"]["used"] is False
    assert call(env.admin_reference_add, body={"path": str(p), "title": "VCF", "value_label": "ppm", "fields": fields})[0] == 200
    assert call(env.admin_reference_add, body={"path": str(other), "title": "Fruits 2"})[0] == 200
    # its fields listed, to change them; most of quantity's lines not numbers: warned of
    listed = {r["id"]: r for r in call(env.reference_list)[1]["datasets"]}
    assert listed["vcf"]["fields"] == fields and listed["vcf"]["value_label"] == "ppm"
    c = call(env.admin_reference_check, body={"path": str(p), "fields": {**fields, "value": "quantity"}})[1]
    assert c["preview"]["value_warning"].startswith("67% of its lines' 'quantity' are not numbers (2 of 3, e.g. ")
    # not changed to an identifier, a field it has not, or one with no numbers
    for value, why in (("cid", "'cid' identifies chemicals (cid)"), ("nope", "no field of"), ("group", "has no numbers")):
        status, r = call(env.admin_reference_update, path_params={"ref_id": "vcf"}, body={"fields": {**fields, "value": value}})
        assert status == 422 and why in r["errors"][0], value
    assert json.loads(next(r for r in env.db.list_references() if r["id"] == "vcf")["fields"])["value"] == "mid_ppm"
    # changed: read again by it (its label kept)
    status, r = call(env.admin_reference_update, path_params={"ref_id": "vcf"}, body={"fields": {**fields, "value": "quantity"}})
    r = call(env.reference_get, path_params={"ref_id": "vcf"})[1]
    assert status == 200 and "2 values are not a number" in " ".join(r["notes"]) and r["meta"]["value_label"] == "ppm"
    # all deleted at once, those asked for (never their files); one not there said so
    status, r = call(env.admin_reference_delete_all, body={"ids": ["vcf", "fruits-2", "nope"]})
    assert status == 200 and [d["id"] for d in r["deleted"]] == ["vcf", "fruits-2"] and r["errors"] == [{"id": "nope", "error": "reference dataset not found"}]
    assert env.db.list_references() == [] and p.exists() and other.exists()
    assert call(env.admin_reference_delete_all, body={"ids": []})[0] == 400
