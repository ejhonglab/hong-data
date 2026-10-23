"""Tests of properties of odors looked up through chemutils. Run with: python -m pytest tests/"""
from __future__ import annotations

import time

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

def _looked_up(what, ignore_cache=False):
    what += " (not from the cache)" * ignore_cache
    print("looking up", what)
    logging.getLogger().warning("a warning")
    with open(os.environ["CHEMUTILS_CACHE"], "a") as f:
        f.write(what + "\\n")
    if what == "InChI=1S/exit":
        os._exit(3)

def inchi2vapor_pressure(inchi, **kwargs):
    _looked_up(inchi, **kwargs)
    if inchi not in VALUES:
        raise ValueError("not understood")
    q, units = VALUES[inchi]
    return {"vapor_pressure": q, "vapor_pressure_units": units,
            "vapor_pressure_sources": "https://pubchem.ncbi.nlm.nih.gov/compound/1" if q is not None else None}

def inchi2est_vapor_pressure(inchi):
    return {"vapor_pressure": 0.5, "vapor_pressure_units": "kPa",
            "vapor_pressure_sources": "SIMPOL.1 estimate (https://doi.org/10.5194/acp-8-2773-2008)"}

def cid2molar_mass(cid, **kwargs):
    _looked_up(f"CID {cid}", **kwargs)
    busy = os.environ["CHEMUTILS_CACHE"] + f".busy{cid}"
    if cid in (8, 9) and not (os.path.exists(busy) and (cid == 8 or open(busy).read() == "xx")):  # (8 once, 9 always)
        with open(busy, "a") as f:
            f.write("x")
        raise RuntimeError("PubChem HTTP Error 429 Too Many Requests")
    return {"molar_mass": 88.11, "molar_mass_units": "g/mol", "molar_mass_sources": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"}

def inchi2density(inchi, **kwargs):
    if inchi == "InChI=1S/b":  # (none on PubChem: from ChemSpider, with a key)
        if chemspipy_api_key is not None:
            raise RuntimeError("Forbidden")
        return {"density": float("nan"), "density_units": None, "density_sources": None}
    return {"density": 902.0, "density_units": "kg / m^3", "density_sources": ["https://a", "https://b"]}

def inchi2k_henry(inchi, **kwargs):
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
    # looked up again, not from chemutils' cache (but estimates, which it does not keep)
    (tmp_path / "cache.txt").write_text("")
    saved.clear()
    assert chemprops.look_up([(1, "InChI=1S/a", vp + ["molar_mass", "est_vapor_pressure"])], save, refresh=True) == {}
    assert (tmp_path / "cache.txt").read_text().splitlines() == ["InChI=1S/a (not from the cache)", "CID 1 (not from the cache)"]
    assert len(saved) == 3
    # nor any, without chemutils
    (fake_chemutils / "chemutils.py").write_text("raise ImportError('no pubchemprops')")
    failed = chemprops.look_up([(1, "InChI=1S/a", vp), (2, "InChI=1S/b", vp)], save)
    assert failed == dict.fromkeys([(1, "vapor_pressure"), (2, "vapor_pressure")], "chemutils could not be imported: ImportError: no pubchemprops")


def test_outdated(monkeypatch):
    # looked up before chemutils' commit was kept, or with one before a change to how a property is looked up
    vapor, density = chemprops.CHANGES
    assert chemprops.outdated("vapor_pressure", None) == vapor and chemprops.outdated("density", None) == density
    assert chemprops.outdated("vapor_pressure", "5bfa72506efec0decb91cb62b26f0daa01e169de") == vapor
    assert chemprops.outdated("density", "95a66c6566cdfc8897249f057c9f92031023e146") == density
    assert chemprops.outdated("vapor_pressure", "95a66c6566cdfc8897249f057c9f92031023e146") is None
    # unchanged, or with a newer one (or one not from git)
    assert chemprops.outdated("molar_mass", None) is None
    assert chemprops.outdated("density", "9196a0c9df65a4f12980fa5d42074846821d4a5d") is None
    assert chemprops.outdated("density", "0.1.0") is None and chemprops.outdated("density", "f" * 40) is None
    # every change one of the commits, in order
    assert [chemprops.COMMITS.index(c) for c, _, _ in chemprops.CHANGES] == sorted(chemprops.COMMITS.index(c) for c, _, _ in chemprops.CHANGES)
    assert all(set(props) <= set(chemprops.PROPERTIES) for _, props, _ in chemprops.CHANGES)


def test_installed():
    # the one pinned in requirements.txt, as installed for these tests
    pinned = next(line for line in (config.ROOT / "requirements.txt").read_text().splitlines() if line.startswith("chemutils"))
    assert chemprops.installed() and pinned.endswith("@" + chemprops.installed())
    assert chemprops.installed().startswith(chemprops.COMMITS[-1])


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

    def look_up(odors, save, refresh=False):
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


def test_relookup(tmp_path, monkeypatch):
    from app import main
    from tests.test_tensor import call

    db = Database(tmp_path / "db.sqlite3")
    for name, value in (("db", db), ("_inchi_tried", {}), ("_props_failed", {}), ("_relookup_job", {"running": False})):
        monkeypatch.setattr(main, name, value)
    apply_all(db, rows_of(HEADER + "valeric acid,va,,7991,,water,,,,,,,\n" "ethanol,EtOH,,702,,,,,,,,,\n" "no inchi,y,,5,,,,,,,,,\n"))
    db.save_inchis({7991: "InChI=1S/C5H10O2/c1-2-3-4-5(6)7/h2-4H2,1H3,(H,6,7)", 702: "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"},
                   when="2026-10-23T00:00:00")
    old = "5bfa72506efec0decb91cb62b26f0daa01e169de"
    for cid in (7991, 702):  # (one looked up before the chemutils was kept, one with an older one)
        db.save_property(cid, "vapor_pressure", {"value": 1.0, "source": "x"}, when="2026-10-22T00:00:00",
                         chemutils=None if cid == 702 else old)
        db.save_property(cid, "molar_mass", {"value": 50.0, "source": "x"}, when="2026-10-22T00:00:00", chemutils=old)
    db.save_property(7991, "density", None, when="2026-10-22T00:00:00", chemutils=old)
    asked = []

    def look_up(odors, save, refresh=False):
        odors = list(odors)
        asked.append((odors, refresh))
        for cid, inchi, props in odors:
            for p in props:
                if (cid, p) != (702, "density"):
                    save(cid, p, {"value": 2.0, "source": "y"})
        return {}

    monkeypatch.setattr(chemprops, "look_up", look_up)
    fetched = []
    monkeypatch.setattr(main, "_fetch_properties", lambda: fetched.append(True))
    status, s = call(main.admin_chemutils)
    by = {p["prop"]: p for p in s["properties"]}
    # those looked up with a chemutils older than the last change to how they are looked up (only of the odor table)
    assert s["installed"] == {"commit": chemprops.installed(), "url": f"{chemprops.REPO}/commit/{chemprops.installed()}"}
    assert {p: (by[p]["looked_up"], by[p]["found"], by[p]["outdated"]) for p in by} == {
        "vapor_pressure": (2, 2, 2), "est_vapor_pressure": (0, 0, 0), "molar_mass": (2, 2, 0), "density": (1, 0, 1), "k_henry": (0, 0, 0)}
    assert by["density"]["change"]["commit"] == "9196a0c9df65" and by["molar_mass"]["change"] is None

    # those looked up before, again, not from chemutils' cache; then what is still missing (as after a sync)
    status, s = call(main.admin_chemutils_relookup, body={"props": ["vapor_pressure", "density"]})
    assert status == 200 and s["job"]["props"] == ["vapor_pressure", "density"]  # (running, or done already)
    for _ in range(100):
        if fetched:
            break
        time.sleep(0.05)
    job = main._relookup_job
    assert (job["total"], job["done"], job["failed"], job["props"]) == (3, 3, 0, ["vapor_pressure", "density"])
    assert asked == [([(702, db.inchis()[702], ["vapor_pressure"]), (7991, db.inchis()[7991], ["vapor_pressure", "density"])], True)]
    assert fetched == [True] and not job["running"] and job["finished_at"]
    s = call(main.admin_chemutils)[1]
    by = {p["prop"]: p for p in s["properties"]}
    assert (by["vapor_pressure"]["outdated"], by["density"]["outdated"], by["density"]["found"]) == (0, 0, 1)
    assert next(o for o in db.list_odors() if o["cid"] == 7991)["density_g_ml"] == 2.0

    # one at a time; of the properties there are
    main._relookup_job["running"] = True
    assert call(main.admin_chemutils_relookup, body={"props": ["density"]})[0] == 409
    main._relookup_job["running"] = False
    for body in ({"props": []}, {"props": ["x"]}, {}, []):
        assert call(main.admin_chemutils_relookup, body=body)[0] == 400
