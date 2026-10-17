"""Odor metadata tests (sheet parsing, sync diff, dataset matching). Run with: python -m pytest tests/"""
from __future__ import annotations

import csv
import io

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
        deleted=list(deletions), applied_by="me", source="test", fingerprint=odormeta.fingerprint(rows),
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
