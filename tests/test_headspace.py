"""Tests of the Headspace calculator. Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import pytest

from app import chemprops, headspace, odormeta
from app.db import Database
from tests.test_odormeta import HEADER, FakePubChem, apply_all, rows_of
from tests.test_tensor import call

# ethyl acetate, about as PubChem, and henrys-law.org, have it
EA = {"vapor_pressure": 12.6, "est_vapor_pressure": None, "molar_mass": 88.11, "density": 0.902, "k_henry": 5.9e-2}
OIL_MOLES = 0.770 / 226.45  # of n-hexadecane, per mL


def test_raoult():
    (r,), total = (lambda x: (x["components"], x["total_pa"]))(headspace.calculate([{"log10_conc": -3, **EA}], "paraffin oil"))
    n = 1e-3 * 0.902 / 88.11
    x = n / (n + 0.999 * OIL_MOLES)
    assert r["law"] == "raoult" and r["mole_fraction"] == pytest.approx(x) and r["pa"] == pytest.approx(x * 12600) and total == r["pa"]
    assert r["notes"] == [] and r["vapor_pressure"] == 12.6
    assert 30 < r["pa"] < 45  # (Pa: about 370 ppm)

    # together: each lowers the others' mole fractions; pure, its vapor pressure
    other = {**EA, "molar_mass": 46.07, "density": 0.789, "vapor_pressure": 7.9}
    a, b = headspace.calculate([{"log10_conc": -1, **EA}, {"log10_conc": -1, **other}], "paraffin oil")["components"]
    na, nb = 0.1 * 0.902 / 88.11, 0.1 * 0.789 / 46.07
    assert a["mole_fraction"] == pytest.approx(na / (na + nb + 0.8 * OIL_MOLES)) and b["pa"] == pytest.approx(nb / (na + nb + 0.8 * OIL_MOLES) * 7900)
    (r,) = headspace.calculate([{"log10_conc": 0, **EA}], "paraffin oil")["components"]
    assert r["pa"] == pytest.approx(12600)
    with pytest.raises(ValueError, match="more than the whole solution"):
        headspace.calculate([{"log10_conc": 0, **EA}, {"log10_conc": -1, **other}], "paraffin oil")


def test_assumptions():
    # an estimated vapor pressure, and a density taken as 1 g/mL, said so
    (r,) = headspace.calculate([{"log10_conc": -3, **EA, "vapor_pressure": None, "est_vapor_pressure": 10.0, "density": None}],
                               "paraffin oil")["components"]
    n = 1e-3 / 88.11
    assert r["pa"] == pytest.approx(n / (n + 0.999 * OIL_MOLES) * 10000) and r["vapor_pressure"] == 10.0
    assert r["notes"] == ["no vapor pressure found, so estimated from its structure (SIMPOL.1)",
                          "no density found, so taken as 1 g/mL"]
    # with no vapor pressure or molar mass, none; the others' without it
    res = headspace.calculate([{"log10_conc": -3, **EA, "vapor_pressure": None}, {"log10_conc": -3, **EA, "molar_mass": None},
                               {"log10_conc": -3, **EA}], "paraffin oil")
    a, b, c = res["components"]
    assert a["pa"] is None and b["pa"] is None and a["mole_fraction"] is not None and b["mole_fraction"] is None
    assert a["notes"] == ["mole fraction without the odors with no molar mass", "no vapor pressure, so its headspace cannot be computed"]
    assert b["notes"] == ["no molar mass, so its headspace cannot be computed"]
    assert res["total_pa"] == c["pa"] and c["mole_fraction"] == pytest.approx(a["mole_fraction"])


def test_henry():
    (r,) = headspace.calculate([{"log10_conc": -2, **EA}], "water")["components"]
    c = 1e-2 * 0.902 / 88.11 * 1e6  # mol/m³
    assert r["law"] == "henry" and r["pa"] == pytest.approx(c / 5.9e-2) and r["notes"] == []
    # more than water dissolves: its vapor pressure
    (r,) = headspace.calculate([{"log10_conc": -0.5, **EA}], "water")["components"]
    assert r["pa"] == 12600 and r["notes"] == ["more than water dissolves by Henry's law, so at its vapor pressure"]
    # with no constant, as an ideal solution, said to be likely too low
    (r,) = headspace.calculate([{"log10_conc": -2, **EA, "k_henry": None}], "water")["components"]
    n = 1e-2 * 0.902 / 88.11
    assert r["law"] == "raoult" and r["pa"] == pytest.approx(n / (n + 0.99 * 0.997 / 18.015) * 12600)
    assert "no Henry's law constant found" in r["notes"][0]


@pytest.fixture
def main(tmp_path, monkeypatch):
    from app import main

    inchis = {8857: "InChI=1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3", 702: "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"}
    monkeypatch.setattr(odormeta, "_failed", {})
    monkeypatch.setattr(odormeta.urllib.request, "urlopen", FakePubChem(inchis))
    db = Database(tmp_path / "db.sqlite3")
    for name, value in (("db", db), ("_inchi_tried", {}), ("_props_failed", {})):
        monkeypatch.setattr(main, name, value)
    apply_all(db, rows_of(HEADER + "ethyl acetate,ea,,8857,,,,,,,,,\n" "ethanol,EtOH,,702,,,,,,,,,\n" "no cid,y,,,,,,,,,,,\n"))
    main.asked = []

    def look_up(odors, save, refresh=False):
        odors = list(odors)
        main.asked.append(odors)
        failed = {}
        for cid, inchi, props in odors:
            for p in props:
                if cid == 702 and p == "density":
                    failed[(cid, p)] = "ValueError: https://pubchem.ncbi.nlm.nih.gov/compound/702\n(1) multiple unit definitions"
                elif cid == 702 and p == "vapor_pressure":
                    save(cid, p, None)
                else:
                    value = {**EA, "est_vapor_pressure": 7.0}[p]
                    save(cid, p, {"value": value, "source": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"})
        return failed

    monkeypatch.setattr(chemprops, "look_up", look_up)
    return main


def test_endpoint(main):
    ids = {o["name"]: o["id"] for o in main.db.list_odors()}
    body = {"solvent": "paraffin oil", "components": [{"odor_id": ids["ethyl acetate"], "log10_conc": -3},
                                                      {"odor_id": ids["ethanol"], "log10_conc": -2}, {"odor_id": ids["no cid"], "log10_conc": -4}]}
    status, r = call(main.headspace_calc, body=body)
    assert status == 200 and r["solvent"] == "paraffin oil" and r["temperature_c"] == pytest.approx(25)
    # what it needs looked up, all at once, once the InChIs are fetched
    needs = ["molar_mass", "density", "vapor_pressure", "est_vapor_pressure"]
    assert main.asked == [[(702, "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", needs), (8857, "InChI=1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3", needs)]]
    ea, etoh, nocid = r["components"]
    assert ea["odor_id"] == ids["ethyl acetate"] and ea["log10_conc"] == -3 and ea["law"] == "raoult" and ea["pa"] > 0
    assert ea["properties"] == {k: EA[k] for k in needs[:3]}  # (no estimate, with a vapor pressure from PubChem)
    assert ea["links"]["vapor_pressure"] == "https://pubchem.ncbi.nlm.nih.gov/compound/8857#section=Vapor-Pressure"
    assert ea["failed"] == {} and ea["notes"] == ["mole fraction without the odors with no molar mass"]
    # an estimate where PubChem has none; why what could not be looked up could not (on one line)
    assert etoh["vapor_pressure"] == 7.0
    assert etoh["failed"] == {"density": "ValueError: https://pubchem.ncbi.nlm.nih.gov/compound/702 (1) multiple unit definitions"}
    assert etoh["notes"] == ["no vapor pressure found, so estimated from its structure (SIMPOL.1)",
                             "no density found, so taken as 1 g/mL", "mole fraction without the odors with no molar mass"]
    assert nocid["pa"] is None and nocid["notes"][0] == "no CID in the odor table, so nothing about it could be looked up"
    assert r["total_pa"] == pytest.approx(ea["pa"] + etoh["pa"])

    # kept: looked up again only what it needs that it has not (nor could, in the last hour)
    assert next(o for o in main.db.list_odors() if o["cid"] == 8857)["density_g_ml"] == 0.902
    body["solvent"] = "water"
    status, r = call(main.headspace_calc, body=body)
    assert status == 200 and main.asked[1:] == [[(702, "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", ["k_henry"]),
                                                 (8857, "InChI=1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3", ["k_henry"])]]
    assert [c["law"] for c in r["components"]] == ["henry", "henry", None]
    status, r = call(main.headspace_calc, body=body)
    assert len(main.asked) == 2


@pytest.mark.parametrize("body, error", [
    ({"solvent": "oil", "components": []}, "expected"),
    ({"solvent": "water", "components": []}, "from 1 to 50 odors"),
    ({"solvent": "water", "components": [{"odor_id": 999, "log10_conc": -3}]}, "not an odor of the odor table: 999"),
    ({"solvent": "water", "components": [{"odor_id": "x", "log10_conc": -3}]}, "not an odor of the odor table: 'x'"),
    ({"solvent": "water", "components": [{"odor_id": 1, "log10_conc": 1}]}, "a number from -50 to 0"),
    ({"solvent": "water", "components": [{"odor_id": 1, "log10_conc": True}]}, "a number from -50 to 0"),
    ({"solvent": "water", "components": [{"odor_id": 1, "log10_conc": -1}, {"odor_id": 1, "log10_conc": -2}]}, "more than once"),
    ({"solvent": "water", "components": [{"odor_id": 1, "log10_conc": 0}, {"odor_id": 2, "log10_conc": 0}]}, "more than the whole solution"),
])
def test_endpoint_errors(main, body, error):
    status, r = call(main.headspace_calc, body=body)
    assert status == 400 and error in r["error"]
