"""Odor metadata tests (sheet parsing, sync diff, dataset matching). Run with: python -m pytest tests/"""
from __future__ import annotations

import contextlib
import csv
import io
import json

import pytest

from app import config, odormeta
from app.db import Database
from app.yamlcheck import parse_odor_yaml

HEADER = "name,abbrev,aliases,cid,cas,solvent,supplier,catalog_no,catalog_url,metacyc,lotus,inventory,notes\n"


def rows_of(text):
    rows, errors, warnings = odormeta.parse_sheet(text)
    assert not errors, errors
    return rows


def apply_all(db, rows, *, changes=(), deletions=()):
    d = odormeta.diff(rows, db.list_odors())
    db.apply_odor_sync(
        added=d["added"], changed=[(c["id"], c["row"]) for c in d["changed"] if c["id"] in changes],
        deleted=list(deletions), source="test", fingerprint=odormeta.fingerprint(rows),
        details={}, when="2026-10-17T00:00:00",
    )
    return d


def test_template_parses():
    rows, errors, warnings = odormeta.parse_sheet(config.ODOR_TEMPLATE.read_text())
    assert not errors
    assert len(rows) > 50
    va = next(r for r in rows if r["name"] == "valeric acid")
    assert va["abbrev"] == "va" and va["cid"] == 7991 and va["aliases"] == ["pentanoic acid"] and va["solvent"] == "water"
    # the two geosmins share 'geos'
    assert any("'geos'" in w for w in warnings)


def test_header_synonyms_and_problems():
    text = ("Chemical,Abbreviation,CAS #,Sigma SKU,Quantity\n"
            "valeric acid,va,109-52-4,240370,100mL\n"
            "linalool,Lin,78-70-5,L2602,\n"     # bad CAS check digit
            ",,,,\n"
            "Valeric Acid,va2,,,\n")            # same name, other case
    rows, errors, warnings = odormeta.parse_sheet(text)
    assert [r["name"] for r in rows] == ["valeric acid", "linalool", "Valeric Acid"]
    assert rows[0]["cas"] == "109-52-4" and rows[0]["catalog_no"] == "240370"
    assert any("78-70-5" in e for e in errors) and any("more than one row" in e and "2, 5" in e for e in errors)
    assert warnings == ["ignored columns: Quantity"]
    assert odormeta.parse_sheet("abbrev,cid\nva,1\n")[1][0].startswith("no 'name' column")
    _, errors, _ = odormeta.parse_sheet(HEADER + "x,x,,12a,,,,,ftp://a,,LT123,,\n")
    assert len(errors) == 3  # CID, URL, LOTUS


def test_diff_and_apply(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    first = rows_of(HEADER + "valeric acid,va,pentanoic acid,7991,,water,,,,,,,\n"
                             "linalool,Lin,,6549,,,Sigma-Aldrich,L2602,,,,,\n"
                             "hexanal,6al,,6184,,,,,,,,,\n")
    d = apply_all(db, first)
    assert len(d["added"]) == 3 and not d["changed"]
    odors = {o["name"]: o for o in db.list_odors()}
    assert odors["valeric acid"]["aliases"] == ["pentanoic acid"] and odors["linalool"]["cid"] == 6549

    # rename (matched by CID), a changed field, a new odor, one gone from the sheet
    second = rows_of(HEADER + "pentanoic acid,va,valeric acid,7991,,water,,,,,,,\n"
                              "linalool,Lin,,6549,,,Sigma-Aldrich,L2602-1L,,,,,\n"
                              "ethyl acetate,ea,,8857,,,,,,,,,\n")
    d = odormeta.diff(second, db.list_odors())
    assert [r["name"] for r in d["added"]] == ["ethyl acetate"]
    assert [o["name"] for o in d["missing"]] == ["hexanal"]
    by_name = {c["row"]["name"]: c for c in d["changed"]}
    assert by_name["pentanoic acid"]["rename"] and by_name["pentanoic acid"]["odor"]["name"] == "valeric acid"
    assert [f["field"] for f in by_name["linalool"]["fields"]] == ["catalog_no"]
    # only the confirmed change is applied; the unconfirmed rename and the deletion are not
    apply_all(db, second, changes={odors["linalool"]["id"]})
    after = {o["name"]: o for o in db.list_odors()}
    assert set(after) == {"valeric acid", "linalool", "hexanal", "ethyl acetate"}
    assert after["linalool"]["catalog_no"] == "L2602-1L"
    apply_all(db, second, changes={odors["valeric acid"]["id"]}, deletions={odors["hexanal"]["id"]})
    after = {o["name"]: o for o in db.list_odors()}
    assert set(after) == {"pentanoic acid", "linalool", "ethyl acetate"}
    assert after["pentanoic acid"]["id"] == odors["valeric acid"]["id"]
    assert odormeta.diff(second, db.list_odors())["unchanged"] == 3
    assert [s["n_added"] for s in db.odor_syncs()] == [0, 1, 3]


def test_links():
    lk = odormeta.links({"cid": 7991, "cas": "109-52-4", "supplier": "Sigma-Aldrich", "catalog_no": "240370",
                         "metacyc": "CPD-1", "lotus": "LTS0000001"})
    assert lk["pubchem"] == "https://pubchem.ncbi.nlm.nih.gov/compound/7991"
    assert "sigmaaldrich.com" in lk["catalog"] and "240370" in lk["catalog"]
    assert lk["metacyc"].endswith("id=CPD-1") and lk["lotus"].endswith("/LTS0000001")
    assert odormeta.links({"catalog_no": "X1", "supplier": "Fisher"})["catalog"] is None
    assert odormeta.links({"catalog_no": "X1", "catalog_url": "https://a/b"})["catalog"] == "https://a/b"


def test_export_url():
    base = "https://docs.google.com/spreadsheets/d/AbC-1_2/"
    assert odormeta.export_url(base + "edit?usp=sharing") == base + "export?format=csv&gid=0"
    assert odormeta.export_url(base + "edit#gid=123") == base + "export?format=csv&gid=123"
    with pytest.raises(odormeta.SheetError):
        odormeta.export_url("https://example.com/spreadsheets/d/x/edit")


def test_dataset_matching(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    apply_all(db, rows_of(HEADER + "geosmin,geos,(-)-geosmin,29746,,,,,,,,,\n"
                                   "perfume geosmin,geos,,1213,,pfo,,,,,,,\n"
                                   "valeric acid,va,,7991,,water,,,,,,,\n"
                                   "1-hexanol,1-6ol,6ol,8103,,,,,,,,,\n"))
    index = odormeta.OdorIndex(db.list_odors())
    trials = [{"panel": "diag", "components": ["geos @ -5", "solvent"]}, {"panel": "diag", "components": ["va @ -4", "solvent"]},
              {"panel": "diag", "components": ["6ol @ -6", "solvent"]}, {"panel": "p2", "components": ["geos @ -4", "va @ -3"]},
              {"panel": "p2", "components": ["zzz @ -2", "solvent"]}]
    rows = odormeta.dataset_odors({}, trials, index, [])
    got = {(r["panel"], r["odor"]): (r["match"], index.by_id[r["odor_id"]]["name"] if r["odor_id"] else None) for r in rows}
    assert got[("diag", "geos @ -5")][0] == "ambiguous"
    assert got[("diag", "va @ -4")] == ("abbrev", "valeric acid")
    assert got[("diag", "6ol @ -6")] == ("abbrev", "1-hexanol")  # via an alias
    assert got[("p2", "zzz @ -2")] == ("none", None)
    assert len(rows) == 6  # components of the mixture separately, solvent left out
    # an odor YAML tells the two geosmins apart
    yaml_odors, _ = parse_odor_yaml("odors:\n - {name: perfume geosmin, abbrev: geos, log10_conc: -5, glomerulus: DA2}\n")
    rows = odormeta.dataset_odors({}, trials, index, yaml_odors)
    r = next(r for r in rows if r["odor"] == "geos @ -5")
    assert r["match"] == "yaml" and index.by_id[r["odor_id"]]["name"] == "perfume geosmin" and r["yaml"]["glomerulus"] == "DA2"
    assert next(r for r in rows if r["odor"] == "geos @ -4")["match"] == "ambiguous"  # the YAML is only at -5
    table = odormeta.table_rows(rows, index)
    csv.DictWriter(io.StringIO(), fieldnames=list(table[0])).writerows(table)
    assert next(t for t in table if t["odor"] == "va @ -3")["pubchem_url"].endswith("/7991")


def test_missing_rows(tmp_path, monkeypatch):
    db = Database(tmp_path / "db.sqlite3")
    apply_all(db, rows_of(HEADER + "geosmin,geos,,29746,,,,,,,,,\nperfume geosmin,geos,,1213,,pfo,,,,,,,\n"
                                   "valeric acid,va,,7991,,water,,,,,,,\n"))
    index = odormeta.OdorIndex(db.list_odors())
    trials = [{"panel": "a", "components": ["va @ -4", "2h @ -5"]}, {"panel": "a", "components": ["2h @ -3", "zzz @ -2"]},
              {"panel": "b", "components": ["2h @ -5", "geos @ -4"]}, {"panel": "b", "components": ["ethyl acetate @ -3"]}]
    yaml_odors, _ = parse_odor_yaml("odors:\n - {name: 2-heptanone, abbrev: 2h, log10_conc: -5, cid: 8051, cas: 110-43-0, solvent: pfo}\n"
                                    " - {name: 2-heptanone, abbrev: 2h, log10_conc: -3, cas: 110-43-1}\n"
                                    " - {name: ethyl acetate, log10_conc: -3}\n")
    missing = odormeta.missing_rows(odormeta.dataset_odors({}, trials, index, yaml_odors))
    # one row per chemical, whatever its concentrations and panels; the ambiguous geos is not one
    assert [(m["name"], m["abbrev"], m["odors"]) for m in missing] == [
        ("2-heptanone", "2h", ["2h @ -5", "2h @ -3"]), (None, "zzz", ["zzz @ -2"]), ("ethyl acetate", None, ["ethyl acetate @ -3"])]
    assert missing[0]["cid"] == 8051 and missing[0]["cas"] == "110-43-0" and missing[0]["solvent"] == "pfo"
    assert missing[0]["supplier"] is None and missing[0]["aliases"] == []

    # pasted in the order of the sheet's columns (its own headers, others left blank), else the template's
    sheet = "Inventory,Chemical,CAS #,Abbreviation,Notes,PubChem CID\nO-1,valeric acid,109-52-4,va,,7991\n"
    columns = odormeta.sheet_columns(sheet)
    assert columns == ["inventory", "name", "cas", "abbrev", "notes", "cid"] and odormeta.sheet_columns("\n\n") is None
    assert odormeta.sheet_columns("Chemical,Quantity\n") == ["name", None]
    assert odormeta.sheet_paste(missing, columns) == "\t2-heptanone\t110-43-0\t2h\t\t8051\n\t\t\tzzz\t\t\n\tethyl acetate\t\t\t\t\n"
    pasted = odormeta.sheet_paste(missing[:1], None)
    assert pasted.rstrip("\n").split("\t")[:6] == ["2-heptanone", "2h", "", "8051", "110-43-0", "pfo"]
    rows, errors, _ = odormeta.parse_sheet(HEADER + pasted.replace("\t", ","))  # as the sheet's CSV export would be
    assert not errors and rows[0]["name"] == "2-heptanone" and rows[0]["cid"] == 8051

    # the endpoint: those rows, in the order of the sheet's columns as last read
    from app import main
    monkeypatch.setattr(main, "db", db)
    monkeypatch.setattr(config, "ODOR_SHEET_URL", "https://docs.google.com/spreadsheets/d/abc/edit")
    monkeypatch.setattr(odormeta, "fetch_sheet", lambda url, timeout=20: sheet)
    monkeypatch.setattr(main, "sheet_watch", odormeta.SheetWatch(600))
    monkeypatch.setattr(main.db, "get_dataset", lambda ds_id: {"id": ds_id, "odor_yaml_files": []})
    monkeypatch.setattr(main.db, "get_trials", lambda ds_id: trials)
    from tests.test_tensor import call
    status, r = call(main.dataset_odor_table, path_params={"ds_id": 1})
    assert status == 200 and [m["abbrev"] for m in r["missing"]] == ["2h", "zzz", "ethyl acetate"]
    assert r["sheet"]["columns_from"] == "sheet" and r["sheet"]["paste"].startswith("\t\t\t2h\t") and r["sheet"]["checked_at"]
    monkeypatch.setattr(main.db, "get_trials", lambda ds_id: [{"panel": "a", "components": ["va @ -4"]}])
    assert call(main.dataset_odor_table, path_params={"ds_id": 1})[1]["sheet"] is None  # nothing missing


def test_check_yaml_odors(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    index = odormeta.OdorIndex([])
    yaml_odors, _ = parse_odor_yaml("odors:\n - {name: valeric acid, abbrev: va, log10_conc: -3}\n")
    assert odormeta.check_yaml_odors(yaml_odors, index)["table_empty"]
    apply_all(db, rows_of(HEADER + "geosmin,geos,(-)-geosmin,29746,,,,,,,,,\n"
                                   "perfume geosmin,geos,,1213,,pfo,,,,,,,\n"
                                   "valeric acid,va,pentanoic acid,7991,,water,,,,,,,\n"
                                   "1-hexanol,1-6ol,6ol,8103,,,,,,,,,\n"
                                   "linalool,lin,,6549,,,,,,,,,\n"))
    index = odormeta.OdorIndex(db.list_odors())
    yaml_odors, _ = parse_odor_yaml("""odors:
 - {name: pentanoic acid, abbrev: va, log10_conc: -3, cid: 7991}
 - {name: valeric acid, abbrev: va, log10_conc: -4}
 - {name: 1-hexanol, abbrev: hex1, log10_conc: -3}
 - {name: hexan-1-ol, abbrev: 6ol, log10_conc: -3}
 - {name: linalool, abbrev: Lin, log10_conc: -3, cid: 443158}
 - {name: perfume geosmin, abbrev: geos, log10_conc: -5}
 - {name: geos-something, abbrev: geos, log10_conc: -5}
 - {name: zzz, log10_conc: -1}
""")
    r = odormeta.check_yaml_odors(yaml_odors, index)
    got = {(o["name"], o["abbrev"]): o["problem"] for o in r["odors"]}
    assert got == {
        ("pentanoic acid", "va"): None,  # an alias, at two concentrations: once
        ("valeric acid", "va"): None,
        ("1-hexanol", "hex1"): "abbreviated 1-6ol in the odor table",
        ("hexan-1-ol", "6ol"): "named 1-hexanol in the odor table",  # matched through its abbreviation (an alias)
        ("linalool", "Lin"): "CID 443158 here, 6549 in the odor table (linalool)",  # abbreviation in another case: fine
        ("perfume geosmin", "geos"): None,
        ("geos-something", "geos"): "could be any of geosmin, perfume geosmin in the odor table",
        ("zzz", None): "not in the odor table",
    }
    assert r["n_in_table"] == 6 and not r["table_empty"]


class FakePubChem:
    """Stands in for urlopen: answers InChI requests for the CIDs in `inchis`, and counts requests."""

    def __init__(self, inchis: dict[int, str], fail: bool = False):
        self.inchis, self.fail, self.urls = inchis, fail, []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        if self.fail:
            raise OSError("no network")
        cids = url.split("/cid/")[1].split("/")[0].split(",")
        props = [{"CID": int(c), **({"InChI": self.inchis[int(c)]} if int(c) in self.inchis else {})} for c in cids]
        return contextlib.nullcontext(io.BytesIO(json.dumps({"PropertyTable": {"Properties": props}}).encode()))


@pytest.fixture
def pubchem(monkeypatch):
    monkeypatch.setattr(odormeta, "_failed", {})
    monkeypatch.setattr(odormeta, "INCHI_BATCH", 2)
    fake = FakePubChem({7991: "InChI=1S/C5H10O2/c1-2-3-4-5(6)7/h2-4H2,1H3,(H,6,7)", 702: "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"})
    monkeypatch.setattr(odormeta.urllib.request, "urlopen", fake)
    return fake


def test_fetch_inchis(pubchem, monkeypatch):
    # in batches; CIDs PubChem has no InChI for are left out
    assert odormeta.fetch_inchis([7991, 702, 1]) == {7991: pubchem.inchis[7991], 702: pubchem.inchis[702]}
    assert [u.split("/cid/")[1].split("/")[0] for u in pubchem.urls] == ["7991,702", "1"]
    # a failed request is not repeated for a while
    monkeypatch.setattr(odormeta.urllib.request, "urlopen", failing := FakePubChem({}, fail=True))
    assert odormeta.fetch_inchis([5]) == {} and odormeta.fetch_inchis([5]) == {}
    assert len(failing.urls) == 1


def test_inchis_in_database(pubchem, tmp_path, monkeypatch):
    from app import main

    db = Database(tmp_path / "db.sqlite3")
    monkeypatch.setattr(main, "db", db)
    apply_all(db, rows_of(HEADER + "valeric acid,va,,7991,,water,,,,,,,\n" "ethanol,EtOH,,702,,,,,,,,,\n"
                                   "no such cid,x,,1,,,,,,,,,\n" "no cid,y,,,,,,,,,,,\n"))
    assert {o["name"]: o["inchi"] for o in db.list_odors()} == dict.fromkeys(["valeric acid", "ethanol", "no such cid", "no cid"])
    main._fetch_inchis()
    assert {o["name"]: o["inchi"] for o in db.list_odors()} == {
        "valeric acid": pubchem.inchis[7991], "ethanol": pubchem.inchis[702], "no such cid": None, "no cid": None}
    assert db.inchis() == {7991: pubchem.inchis[7991], 702: pubchem.inchis[702]}
    # only CIDs still without an InChI are asked for again
    pubchem.urls.clear()
    monkeypatch.setattr(odormeta, "_failed", {})
    main._fetch_inchis()
    assert [u.split("/cid/")[1].split("/")[0] for u in pubchem.urls] == ["1"]
    # the InChI belongs to the CID, so it stays with an odor through a sync that changes other fields
    apply_all(db, rows_of(HEADER + "valeric acid,va,pentanoic acid,7991,,water,,,,,,,\n"), changes={o["id"] for o in db.list_odors()})
    assert next(o for o in db.list_odors() if o["name"] == "valeric acid")["inchi"] == pubchem.inchis[7991]
