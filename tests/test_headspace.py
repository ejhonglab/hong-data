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
# its Henry's law constant in paraffin oil, as chemutils estimates it (with thermo's vapor pressure)
PFO = {"k_henry_pfo": 0.0977, "k_henry_pfo_measured": False,
       "k_henry_pfo_details": {"vapor_pressure_kpa": 12.61, "vapor_pressure_source": "thermo WAGNER_MCGARRY",
                               "vapor_pressure_measured": True, "notes": None}}


def test_raoult():
    (r,), total = (lambda x: (x["components"], x["total_pa"]))(headspace.calculate([{"log10_conc": -3, **EA}], "paraffin oil", "raoult"))
    n = 1e-3 * 0.902 / 88.11
    x = n / (n + 0.999 * OIL_MOLES)
    assert r["law"] == "raoult" and r["mole_fraction"] == pytest.approx(x) and r["pa"] == pytest.approx(x * 12600) and total == r["pa"]
    assert r["notes"] == [] and r["vapor_pressure"] == 12.6
    assert 30 < r["pa"] < 45  # (Pa: about 370 ppm)

    # together: each lowers the others' mole fractions; pure, its vapor pressure
    other = {**EA, "molar_mass": 46.07, "density": 0.789, "vapor_pressure": 7.9}
    a, b = headspace.calculate([{"log10_conc": -1, **EA}, {"log10_conc": -1, **other}], "paraffin oil", "raoult")["components"]
    na, nb = 0.1 * 0.902 / 88.11, 0.1 * 0.789 / 46.07
    assert a["mole_fraction"] == pytest.approx(na / (na + nb + 0.8 * OIL_MOLES)) and b["pa"] == pytest.approx(nb / (na + nb + 0.8 * OIL_MOLES) * 7900)
    (r,) = headspace.calculate([{"log10_conc": 0, **EA}], "paraffin oil", "raoult")["components"]
    assert r["pa"] == pytest.approx(12600)
    with pytest.raises(ValueError, match="more than the whole solution"):
        headspace.calculate([{"log10_conc": 0, **EA}, {"log10_conc": -1, **other}], "paraffin oil", "raoult")


def test_assumptions():
    # an estimated vapor pressure, and a density taken as 1 g/mL, said so
    (r,) = headspace.calculate([{"log10_conc": -3, **EA, "vapor_pressure": None, "est_vapor_pressure": 10.0, "density": None}],
                               "paraffin oil", "raoult")["components"]
    n = 1e-3 / 88.11
    assert r["pa"] == pytest.approx(n / (n + 0.999 * OIL_MOLES) * 10000) and r["vapor_pressure"] == 10.0
    assert r["notes"] == ["no density found, so taken as 1 g/mL",
                          "no vapor pressure found, so estimated from its structure (SIMPOL.1)"]
    # with no vapor pressure or molar mass, none; the others' without it
    res = headspace.calculate([{"log10_conc": -3, **EA, "vapor_pressure": None}, {"log10_conc": -3, **EA, "molar_mass": None},
                               {"log10_conc": -3, **EA}], "paraffin oil", "raoult")
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


def test_henry_oil():
    # by default, Henry's law with the constant chemutils estimated, and the vapor pressure it was estimated with
    (r,) = headspace.calculate([{"log10_conc": -3, **EA, **PFO}], "paraffin oil")["components"]
    c = 1e-3 * 0.902 / 88.11 * 1e6  # mol/m³
    assert r["law"] == "henry" and r["pa"] == pytest.approx(c / 0.0977) and (r["k_henry"], r["k_henry_measured"]) == (0.0977, False)
    assert (r["vapor_pressure"], r["vapor_pressure_measured"], r["vapor_pressure_source"]) == (12.61, True, "thermo WAGNER_MCGARRY")
    assert r["notes"] == []
    # (Raoult's law, times its activity coefficient when dilute: 2.8)
    raoult = headspace.calculate([{"log10_conc": -3, **EA, **PFO}], "paraffin oil", "raoult")["components"][0]
    assert 2.5 < r["pa"] / raoult["pa"] < 3 and raoult["k_henry"] is None and raoult["vapor_pressure"] == 12.6
    # with chemutils' notes, and at most the vapor pressure
    details = {**PFO["k_henry_pfo_details"], "notes": "OH group: self-association lowers solubility at higher concentrations"}
    (r,) = headspace.calculate([{"log10_conc": -0.5, **EA, **PFO, "k_henry_pfo_details": details}], "paraffin oil")["components"]
    assert r["pa"] == pytest.approx(12610)
    assert r["notes"] == [details["notes"], "more than paraffin oil dissolves by Henry's law, so at its vapor pressure"]
    # with none estimated, Raoult's law, saying why
    (r,) = headspace.calculate([{"log10_conc": -3, **EA, "k_henry_pfo": None, "k_henry_pfo_details": {"notes": "no UNIFAC groups"}}],
                               "paraffin oil")["components"]
    assert r["law"] == "raoult" and r["pa"] == pytest.approx(raoult["pa"])
    assert r["notes"] == ["no Henry's law constant could be estimated in oil (no UNIFAC groups), so as an ideal solution (Raoult's law)"]


def test_measured():
    # whether the vapor pressure used is from measurements: as chemutils says of PubChem's, never of estimates
    for measured in (True, False, None):
        (r,) = headspace.calculate([{"log10_conc": -3, **EA, "vapor_pressure_measured": measured}], "paraffin oil", "raoult")["components"]
        assert r["vapor_pressure_measured"] is measured
    (r,) = headspace.calculate([{"log10_conc": -3, **EA, "vapor_pressure": None, "est_vapor_pressure": 10.0,
                                 "vapor_pressure_measured": True}], "water")["components"]
    assert r["vapor_pressure_measured"] is False and r["k_henry_measured"] is None


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
                elif p == "k_henry_pfo":
                    save(cid, p, {"value": None, "source": None, "measured": None, "details": {"notes": "no UNIFAC groups"}} if cid == 702
                         else {"value": 0.0977, "source": "UNIFAC estimate", "measured": False, "details": PFO["k_henry_pfo_details"]})
                else:
                    value = {**EA, "est_vapor_pressure": 7.0}[p]
                    save(cid, p, {"value": value, "source": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                                  "measured": {"vapor_pressure": True, "est_vapor_pressure": False}.get(p)})
        return failed

    monkeypatch.setattr(chemprops, "look_up", look_up)
    return main


def test_endpoint(main):
    ids = {o["name"]: o["id"] for o in main.db.list_odors()}
    body = {"solvent": "paraffin oil", "components": [{"odor_id": ids["ethyl acetate"], "log10_conc": -3},
                                                      {"odor_id": ids["ethanol"], "log10_conc": -2}, {"odor_id": ids["no cid"], "log10_conc": -4}]}
    status, r = call(main.headspace_calc, body=body)
    assert status == 200 and (r["solvent"], r["model"]) == ("paraffin oil", "henry") and r["temperature_c"] == pytest.approx(25)
    # what it needs looked up, all at once, once the InChIs are fetched
    needs = ["molar_mass", "density", "vapor_pressure", "est_vapor_pressure", "k_henry_pfo"]
    assert main.asked == [[(702, "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", needs), (8857, "InChI=1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3", needs)]]
    ea, etoh, nocid = r["components"]
    assert ea["odor_id"] == ids["ethyl acetate"] and ea["log10_conc"] == -3 and ea["law"] == "henry" and ea["pa"] > 0
    # (no estimate, with a vapor pressure from PubChem); its constant in oil an estimate, with a measured vapor pressure
    assert ea["properties"] == {**{k: EA[k] for k in needs[:3]}, "k_henry_pfo": 0.0977}
    assert (ea["k_henry_measured"], ea["vapor_pressure"], ea["vapor_pressure_measured"]) == (False, 12.61, True)
    assert ea["sources"]["k_henry_pfo"] == "UNIFAC estimate"
    assert ea["links"]["vapor_pressure"] == "https://pubchem.ncbi.nlm.nih.gov/compound/8857#section=Vapor-Pressure"
    assert ea["failed"] == {} and ea["notes"] == ["mole fraction without the odors with no molar mass"]
    # with no constant estimated, Raoult's law, with an estimated vapor pressure where PubChem has none; why what
    # could not be looked up could not (on one line)
    assert (etoh["law"], etoh["vapor_pressure"], etoh["vapor_pressure_measured"]) == ("raoult", 7.0, False)
    assert etoh["failed"] == {"density": "ValueError: https://pubchem.ncbi.nlm.nih.gov/compound/702 (1) multiple unit definitions"}
    assert etoh["notes"] == ["no density found, so taken as 1 g/mL", "mole fraction without the odors with no molar mass",
                             "no Henry's law constant could be estimated in oil (no UNIFAC groups), so as an ideal solution (Raoult's law)",
                             "no vapor pressure found, so estimated from its structure (SIMPOL.1)"]
    assert nocid["pa"] is None and nocid["notes"][0] == "no CID in the odor table, so nothing about it could be looked up"
    assert r["total_pa"] == pytest.approx(ea["pa"] + etoh["pa"])
    # kept with the odor table, with whether measured and the details
    o = next(o for o in main.db.list_odors() if o["cid"] == 8857)
    assert (o["k_henry_pfo_measured"], o["vapor_pressure_measured"], o["k_henry_pfo_details"]) == (False, True, PFO["k_henry_pfo_details"])

    # Raoult's law only, if asked: nothing more to look up
    status, r = call(main.headspace_calc, body={**body, "model": "raoult"})
    assert status == 200 and r["model"] == "raoult" and [c["law"] for c in r["components"]] == ["raoult", "raoult", None]
    assert len(main.asked) == 1 and r["components"][0]["vapor_pressure"] == 12.6

    # kept: looked up again only what it needs that it has not (nor could, in the last hour)
    assert next(o for o in main.db.list_odors() if o["cid"] == 8857)["density_g_ml"] == 0.902
    body["solvent"] = "water"
    status, r = call(main.headspace_calc, body=body)
    assert status == 200 and r["model"] is None and main.asked[1:] == [[(702, "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", ["k_henry"]),
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
    ({"solvent": "paraffin oil", "model": "unifac", "components": [{"odor_id": 1, "log10_conc": -3}]}, "the model is one of"),
])
def test_endpoint_errors(main, body, error):
    status, r = call(main.headspace_calc, body=body)
    assert status == 400 and error in r["error"]


def test_panels(main, tmp_path):
    from app.ingest import load_file
    from tests.test_ingest import COLS, make_wide, write_csv

    apply_all(main.db, rows_of(HEADER + "paraffin oil,pfo,,,,,,,,,,,\n" "valeric acid,va,,7991,,,,,,,,,\n"))
    rows = [("kiwi", False, odor, "solvent", rep) for odor in ("pfo @ 0", "EtOH @ -2", "ea @ -4.2", "ea @ -5", "kmix @ 0", "y @ -3")
            for rep in (0, 1)] + [("control", False, "va @ -3", "solvent", rep) for rep in (0, 1)]
    parsed = load_file(write_csv(make_wide(rows, COLS), tmp_path))
    main.db.insert_dataset(
        created_at="2026-10-23T10:00:00", title="t", readme_md="#", readme_edited=False, data_file="x.csv",
        metadata={"uploader_name": "me", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli"},
        data_format=parsed.source_format, data_sha256="0" * 64, odor_yaml_files=[], experiment_xml_file=None,
        experiment_xml_info=None, summary=parsed.summary(), units=parsed.units, trials=parsed.trials, values=parsed.values)
    ids = {o["abbrev"]: o["id"] for o in main.db.list_odors()}
    status, r = call(main.headspace_panels)
    # each panel's odors in the odor table with a CID, but solvents, at the highest concentration presented
    assert status == 200 and [p["panel"] for p in r["panels"]] == ["control", "kiwi"]
    control, kiwi = r["panels"]
    assert control == {"panel": "control", "datasets": 1, "components": [{"odor_id": ids["va"], "log10_conc": -3, "concs": [-3]}],
                       "skipped": []}
    assert kiwi["components"] == [{"odor_id": ids["EtOH"], "log10_conc": -2, "concs": [-2]},
                                  {"odor_id": ids["ea"], "log10_conc": -4.2, "concs": [-4.2, -5]}]
    assert kiwi["skipped"] == [{"odor": "pfo", "why": "a solvent"}, {"odor": "kmix", "why": "not in the odor table"},
                               {"odor": "y", "why": "no CID in the odor table"}]


def test_download(main):
    from tests.test_export import get_file

    ids = {o["name"]: o["id"] for o in main.db.list_odors()}
    params = {"solvent": "paraffin oil", "odors": f"{ids['ethyl acetate']}:-3,{ids['ethanol']}:-2", "units": "ppm"}
    get = lambda fmt, **kw: get_file(main.headspace_file, {**params, **kw}, {"fmt": fmt})
    r = get("csv")
    assert r.status_code == 200 and r.headers["content-disposition"] == 'attachment; filename="headspace_paraffin-oil_henry.csv"'
    import io

    import pandas as pd
    df = pd.read_csv(io.StringIO(r.body.decode()))
    # a row per odor, then one of them all together: what was used of each, whether measured, its headspace in every unit
    assert list(df["name"]) == ["ethyl acetate", "ethanol", "All together"] and list(df["law"][:2]) == ["henry", "raoult"]
    ea, etoh, total = df.to_dict("records")
    assert (ea["k_henry_mol_m3_pa"], ea["k_henry_measured"], ea["vapor_pressure_kpa"], ea["vapor_pressure_measured"]) == (0.0977, False, 12.61, True)
    assert ea["vapor_pressure_source"] == "thermo WAGNER_MCGARRY" and ea["k_henry_source"] == "UNIFAC estimate"
    assert (etoh["vapor_pressure_kpa"], etoh["vapor_pressure_measured"]) == (7.0, False) and "SIMPOL" in etoh["notes"]
    assert "density could not be looked up" in etoh["notes"]
    assert total["headspace_pa"] == pytest.approx(ea["headspace_pa"] + etoh["headspace_pa"])
    assert ea["headspace_ppm"] == pytest.approx(ea["headspace_pa"] / 101325 * 1e6)
    assert total["headspace_ug_l"] == pytest.approx(ea["headspace_ug_l"] + etoh["headspace_ug_l"])

    # LaTeX: as the tab shows it, in the units chosen, linked, values marked measured or estimates, with a legend
    tex = get("tex", units="log10").body.decode()
    lines = tex.splitlines()
    assert any(l.startswith(r"\href{https://pubchem.ncbi.nlm.nih.gov/compound/8857}{ethyl acetate} & ea & $-3$ & ") for l in lines)
    ea_line = next(l for l in lines if "{ethyl acetate}" in l)
    assert r"12.6$^{m}$" in ea_line and r"0.0977$^{e}$" in ea_line and r"8857\#section=Molecular-Weight}{88.11}" in ea_line
    assert r"$K_H$ in oil\\(mol/m$^3$/Pa)" in tex and r"Headspace\\($\log_{10}$ v/v)" in tex
    assert any(l.startswith("All together &") for l in lines)
    assert r"$^{e}$ estimated, with nothing measured of the compound itself (in oil, every $K_H$ is" in tex
    assert get("tex", model="raoult").body.decode().count("$K_H$") == 0  # (none, by Raoult's law)
    # errors
    assert get("csv", odors="x").status_code == 400 and get("tex", units="ppt").status_code == 400
    assert get("xlsx").status_code == 400 and get("csv", odors="999:-3").status_code == 400
