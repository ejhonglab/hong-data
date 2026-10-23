"""Tests of properties of odors looked up through chemutils. Run with: python -m pytest tests/"""
from __future__ import annotations

import pytest

from app import chemprops, config, odormeta
from app.db import PROPERTY_COLUMNS, Database
from tests.test_odormeta import HEADER, FakePubChem, apply_all, rows_of

# stands in for chemutils, as the process `chemprops.look_up` starts imports it: like it, logging
# to a file beside itself, adding a handler printing to stdout to the root logger, and printing
FAKE_CHEMUTILS = """
import logging, os, sys
import pint
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), "chemutils.log"))
sh = logging.StreamHandler(sys.stdout)
logging.getLogger().addHandler(sh)
ureg = pint.UnitRegistry()
chemspipy_api_key = os.environ.get("CHEMSPIDER_API_KEY")
VALUES = {"InChI=1S/a": (11.27, "kPa"), "InChI=1S/b": (None, None), "InChI=1S/c": (2.0, "mmHg"), "InChI=1S/d": (3.0, "g/mol")}

def _looked_up(what):
    print("looking up", what)
    logging.getLogger().warning("a warning")
    with open(os.environ["CHEMUTILS_CACHE"], "a") as f:
        f.write(what + "\\n")
    if what == "InChI=1S/exit":
        os._exit(3)

def inchi2vapor_pressure(inchi):
    _looked_up(inchi)
    if inchi not in VALUES:
        raise ValueError("not understood")
    q, units = VALUES[inchi]
    return {"vapor_pressure": q, "vapor_pressure_units": units,
            "vapor_pressure_sources": "https://pubchem.ncbi.nlm.nih.gov/compound/1" if q is not None else None}

def inchi2est_vapor_pressure(inchi):
    return {"vapor_pressure": 0.5, "vapor_pressure_units": "kPa",
            "vapor_pressure_sources": "SIMPOL.1 estimate (https://doi.org/10.5194/acp-8-2773-2008)"}

def cid2molar_mass(cid):
    _looked_up(f"CID {cid}")
    busy = os.environ["CHEMUTILS_CACHE"] + f".busy{cid}"
    if cid in (8, 9) and not (os.path.exists(busy) and (cid == 8 or open(busy).read() == "xx")):  # (8 once, 9 always)
        with open(busy, "a") as f:
            f.write("x")
        raise RuntimeError("PubChem HTTP Error 429 Too Many Requests")
    return {"molar_mass": 88.11, "molar_mass_units": "g/mol", "molar_mass_sources": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"}

def inchi2density(inchi):
    if inchi == "InChI=1S/b":  # (none on PubChem: from ChemSpider, with a key)
        if chemspipy_api_key is not None:
            raise RuntimeError("Forbidden")
        return {"density": float("nan"), "density_units": None, "density_sources": None}
    return {"density": 902.0, "density_units": "kg / m^3", "density_sources": ["https://a", "https://b"]}

def inchi2k_henry(inchi):
    return {"k_henry": 5.9e-4, "k_henry_units": "mol / L / kPa", "k_henry_sources": "https://henrys-law.org/x"}
"""


@pytest.fixture
def fake_chemutils(tmp_path, monkeypatch):
    """A folder with `FAKE_CHEMUTILS` as chemutils, first on the path of the process `chemprops.look_up` starts."""
    d = tmp_path / "fake"
    d.mkdir()
    (d / "chemutils.py").write_text(FAKE_CHEMUTILS)
    monkeypatch.setenv("PYTHONPATH", str(d))
    monkeypatch.setattr(config, "CHEMUTILS_CACHE", tmp_path / "cache.txt")
    return d


def test_look_up(fake_chemutils, tmp_path, monkeypatch):
    saved = {}
    save = lambda cid, prop, found: saved.__setitem__((cid, prop), found)
    vp = ["vapor_pressure"]
    failed = chemprops.look_up([(1, "InChI=1S/a", vp + ["molar_mass", "est_vapor_pressure"]), (2, "InChI=1S/b", vp + ["density"]),
                                (3, "InChI=1S/c", vp + ["density", "k_henry"]), (4, "InChI=1S/x", vp), (5, "InChI=1S/d", vp),
                                (6, "InChI=1S/a", [])], save)
    # each as it comes, in the units kept (from others converted); none found is saved too, but
    # what could not be looked up is not
    assert saved.pop((3, "vapor_pressure"))["value"] == pytest.approx(0.266645, rel=1e-5)
    assert saved == {
        (1, "vapor_pressure"): {"value": 11.27, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/1"},
        (1, "molar_mass"): {"value": 88.11, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/1"},
        (1, "est_vapor_pressure"): {"value": 0.5, "source": "SIMPOL.1 estimate (https://doi.org/10.5194/acp-8-2773-2008)"},
        (2, "vapor_pressure"): None, (2, "density"): None,
        (3, "density"): {"value": pytest.approx(0.902), "source": "https://a; https://b"},
        (3, "k_henry"): {"value": pytest.approx(5.9e-4), "source": "https://henrys-law.org/x"},
    }
    assert failed == {(4, "vapor_pressure"): "ValueError: not understood",
                      (5, "vapor_pressure"): "DimensionalityError: Cannot convert from 'gram / mole' ([mass] / [substance]) to "
                                             "'kilopascal' ([mass] / [length] / [time] ** 2)"}
    # those a server was too busy for tried again after a while, as long as it is
    monkeypatch.setattr(chemprops, "RETRY_WAITS_S", (0.01,))
    failed = chemprops.look_up([(8, "InChI=1S/a", ["molar_mass"]), (9, "InChI=1S/a", ["molar_mass"])], save)
    assert saved[(8, "molar_mass")]["value"] == 88.11
    assert failed == {(9, "molar_mass"): "RuntimeError: PubChem HTTP Error 429 Too Many Requests"}
    # with its cache in the data folder, its printing and logging kept from what it answers, and no log file
    assert (tmp_path / "cache.txt").read_text().splitlines() == ["InChI=1S/a", "CID 1", "InChI=1S/b", "InChI=1S/c", "InChI=1S/x",
                                                                 "InChI=1S/d", "CID 8", "CID 8", "CID 9", "CID 9"]
    assert not (fake_chemutils / "chemutils.log").exists()
    assert chemprops.look_up([], save) == {} and chemprops.look_up([(1, "InChI=1S/a", [])], save) == {}

    # those after it stopped could not be looked up
    saved.clear()
    failed = chemprops.look_up([(1, "InChI=1S/a", vp), (7, "InChI=1S/exit", vp + ["molar_mass"]), (2, "InChI=1S/b", vp)], save)
    assert list(saved) == [(1, "vapor_pressure")]
    assert failed == dict.fromkeys([(7, "vapor_pressure"), (7, "molar_mass"), (2, "vapor_pressure")],
                                   "chemutils' process ended early (exit status 3)")
    # nor any, without chemutils
    (fake_chemutils / "chemutils.py").write_text("raise ImportError('no pubchemprops')")
    failed = chemprops.look_up([(1, "InChI=1S/a", vp), (2, "InChI=1S/b", vp)], save)
    assert failed == dict.fromkeys([(1, "vapor_pressure"), (2, "vapor_pressure")], "chemutils could not be imported: ImportError: no pubchemprops")


def test_columns():
    # each property kept in the database, and nothing else
    assert set(PROPERTY_COLUMNS) == set(chemprops.PROPERTIES)


def test_properties_in_database(tmp_path, monkeypatch):
    from app import main

    inchis = {7991: "InChI=1S/C5H10O2/c1-2-3-4-5(6)7/h2-4H2,1H3,(H,6,7)", 702: "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
              176: "InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)"}
    monkeypatch.setattr(odormeta, "_failed", {})
    monkeypatch.setattr(odormeta.urllib.request, "urlopen", FakePubChem(inchis))
    db = Database(tmp_path / "db.sqlite3")
    for name, value in (("db", db), ("_inchi_tried", {}), ("_props_failed", {})):
        monkeypatch.setattr(main, name, value)
    asked = []

    def look_up(odors, save):
        odors = list(odors)
        asked.append(odors)
        if len(asked) == 1:  # an odor added meanwhile
            apply_all(db, rows_of(HEADER + "ethanol,EtOH,,702,,,,,,,,,\n"))
        for cid, inchi, props in odors:
            if cid == 7991:
                save(cid, "vapor_pressure", {"value": 0.0262, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/7991"})
                save(cid, "molar_mass", {"value": 102.13, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/7991"})
            elif cid == 702:
                save(cid, "vapor_pressure", None)
                save(cid, "molar_mass", {"value": 46.07, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/702"})
        return {(176, "vapor_pressure"): "no network", (176, "molar_mass"): "no network"}

    monkeypatch.setattr(chemprops, "look_up", look_up)
    apply_all(db, rows_of(HEADER + "valeric acid,va,,7991,,water,,,,,,,\n" "acetic acid,AA,,176,,,,,,,,,\n" "no cid,y,,,,,,,,,,,\n"))
    # the InChIs fetched first, then what is shown with each odor with one looked up, then for those added meanwhile
    main._fetch_properties()
    shown = ["vapor_pressure", "molar_mass"]
    assert asked == [[(176, inchis[176], shown), (7991, inchis[7991], shown)], [(702, inchis[702], shown)]]
    odors = {o["name"]: o for o in db.list_odors()}
    assert {n: o["vapor_pressure_kpa"] for n, o in odors.items()} == {"valeric acid": 0.0262, "ethanol": None, "acetic acid": None,
                                                                    "no cid": None}
    assert {n: o["molar_mass_g_mol"] for n, o in odors.items()} == {"valeric acid": 102.13, "ethanol": 46.07, "acetic acid": None,
                                                                  "no cid": None}
    assert odors["ethanol"]["vapor_pressure_at"] and not odors["acetic acid"]["vapor_pressure_at"]
    # linked to the section of the PubChem page it is from
    links = odormeta.links(odors["valeric acid"])
    assert links["vapor_pressure"] == "https://pubchem.ncbi.nlm.nih.gov/compound/7991#section=Vapor-Pressure"
    assert links["molar_mass"] == "https://pubchem.ncbi.nlm.nih.gov/compound/7991#section=Molecular-Weight"
    assert odormeta.links(odors["ethanol"])["vapor_pressure"] is None and links["density"] is None

    # those not found are not looked up again, and those that could not be only after an hour
    main._fetch_properties()
    assert len(asked) == 2
    main._props_failed[(176, "molar_mass")] = (main._props_failed[(176, "molar_mass")][0] - 3601, "no network")
    main._fetch_properties()
    assert asked[2:] == [[(176, inchis[176], ["molar_mass"])]]
    # a sync changing other fields keeps them with the odor (they belong to the CID)
    apply_all(db, rows_of(HEADER + "valeric acid,va,pentanoic acid,7991,,water,,,,,,,\n"), changes={o["id"] for o in db.list_odors()})
    o = next(o for o in db.list_odors() if o["name"] == "valeric acid")
    assert (o["vapor_pressure_kpa"], o["molar_mass_g_mol"]) == (0.0262, 102.13)


def test_properties_in_tables(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    apply_all(db, rows_of(HEADER + "valeric acid,va,,7991,,water,,,,,,,\n"))
    db.save_inchis({7991: "InChI=1S/C5H10O2/c1-2-3-4-5(6)7/h2-4H2,1H3,(H,6,7)"}, when="2026-10-23T00:00:00")
    db.save_property(7991, "vapor_pressure", {"value": 0.0262, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/7991"},
                     when="2026-10-23T00:00:00")
    db.save_property(7991, "molar_mass", {"value": 102.13, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/7991"},
                     when="2026-10-23T00:00:00")
    db.save_property(7991, "est_vapor_pressure", {"value": 0.02, "source": "SIMPOL.1 estimate (https://doi.org/10.5194/acp-8-2773-2008)"},
                     when="2026-10-23T00:00:00")
    index = odormeta.OdorIndex(db.list_odors())
    rows = [{"panel": "p", "odor": "va @ -3", "label": "va", "log10_conc": -3, "yaml": None, "odor_id": next(iter(index.by_id)),
             "match": "abbrev"}]
    (row,) = odormeta.table_rows(rows, index)
    assert row["vapor_pressure_kpa"] == 0.0262 and row["vapor_pressure_url"].endswith("/7991#section=Vapor-Pressure")
    assert row["molar_mass_g_mol"] == 102.13
    # an estimate linked to the paper of its method
    assert odormeta.links(index.by_id[rows[0]["odor_id"]])["est_vapor_pressure"] == "https://doi.org/10.5194/acp-8-2773-2008"
