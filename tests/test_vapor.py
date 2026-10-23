"""Tests of vapor pressures looked up through chemutils. Run with: python -m pytest tests/"""
from __future__ import annotations

import pytest

from app import config, odormeta, vapor
from app.db import Database
from tests.test_odormeta import HEADER, FakePubChem, apply_all, rows_of

# stands in for chemutils, as the process `vapor.look_up` starts imports it: like it, logging
# to a file beside itself, adding a handler printing to stdout to the root logger, and printing
FAKE_CHEMUTILS = """
import logging, os, sys
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), "chemutils.log"))
sh = logging.StreamHandler(sys.stdout)
logging.getLogger().addHandler(sh)
VALUES = {"InChI=1S/a": (11.27, "kPa"), "InChI=1S/b": (None, None), "InChI=1S/c": (2.0, "mmHg")}

def inchi2vapor_pressure(inchi):
    print("looking up", inchi)
    logging.getLogger().warning("a warning")
    with open(os.environ["CHEMUTILS_CACHE"], "a") as f:
        f.write(inchi + "\\n")
    if inchi == "InChI=1S/exit":
        os._exit(3)
    if inchi not in VALUES:
        raise ValueError("not understood")
    q, units = VALUES[inchi]
    return {"vapor_pressure": q, "vapor_pressure_units": units,
            "vapor_pressure_sources": "https://pubchem.ncbi.nlm.nih.gov/compound/1" if q is not None else None}
"""


@pytest.fixture
def fake_chemutils(tmp_path, monkeypatch):
    """A folder with `FAKE_CHEMUTILS` as chemutils, first on the path of the process `vapor.look_up` starts."""
    d = tmp_path / "fake"
    d.mkdir()
    (d / "chemutils.py").write_text(FAKE_CHEMUTILS)
    monkeypatch.setenv("PYTHONPATH", str(d))
    monkeypatch.setattr(config, "CHEMUTILS_CACHE", tmp_path / "cache.txt")
    return d


def test_look_up(fake_chemutils, tmp_path):
    saved = {}
    failed = vapor.look_up([(1, "InChI=1S/a"), (2, "InChI=1S/b"), (3, "InChI=1S/c"), (4, "InChI=1S/x")], saved.__setitem__)
    # each as it comes; none found is saved too, but what could not be looked up is not
    assert saved == {1: {"kpa": 11.27, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/1"}, 2: None}
    assert failed == {3: "ValueError: in mmHg, not kPa", 4: "ValueError: not understood"}
    # with its cache in the data folder, its printing and logging kept from what it answers, and no log file
    assert (tmp_path / "cache.txt").read_text().split() == ["InChI=1S/a", "InChI=1S/b", "InChI=1S/c", "InChI=1S/x"]
    assert not (fake_chemutils / "chemutils.log").exists()
    assert vapor.look_up([], saved.__setitem__) == {}

    # those after it stopped could not be looked up
    saved.clear()
    failed = vapor.look_up([(1, "InChI=1S/a"), (5, "InChI=1S/exit"), (2, "InChI=1S/b")], saved.__setitem__)
    assert list(saved) == [1] and failed == dict.fromkeys([5, 2], "chemutils' process ended early (exit status 3)")
    # nor any, without chemutils
    (fake_chemutils / "chemutils.py").write_text("raise ImportError('no pubchemprops')")
    failed = vapor.look_up([(1, "InChI=1S/a"), (2, "InChI=1S/b")], saved.__setitem__)
    assert failed == dict.fromkeys([1, 2], "chemutils could not be imported: ImportError: no pubchemprops")


def test_vapor_pressures_in_database(tmp_path, monkeypatch):
    from app import main

    inchis = {7991: "InChI=1S/C5H10O2/c1-2-3-4-5(6)7/h2-4H2,1H3,(H,6,7)", 702: "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
              176: "InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)"}
    monkeypatch.setattr(odormeta, "_failed", {})
    monkeypatch.setattr(odormeta.urllib.request, "urlopen", FakePubChem(inchis))
    db = Database(tmp_path / "db.sqlite3")
    for name, value in (("db", db), ("_inchi_tried", {}), ("_vapor_tried", {})):
        monkeypatch.setattr(main, name, value)
    asked = []

    def look_up(odors, save):
        odors = list(odors)
        asked.append(odors)
        if len(asked) == 1:  # an odor added meanwhile
            apply_all(db, rows_of(HEADER + "ethanol,EtOH,,702,,,,,,,,,\n"))
        for cid, inchi in odors:
            if cid == 7991:
                save(cid, {"kpa": 0.0262, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/7991"})
            elif cid == 702:
                save(cid, None)
        return {176: "no network"}

    monkeypatch.setattr(vapor, "look_up", look_up)
    apply_all(db, rows_of(HEADER + "valeric acid,va,,7991,,water,,,,,,,\n" "acetic acid,AA,,176,,,,,,,,,\n" "no cid,y,,,,,,,,,,,\n"))
    # the InChIs fetched first, then each odor with one looked up, then those added meanwhile
    main._fetch_vapor_pressures()
    assert asked == [[(176, inchis[176]), (7991, inchis[7991])], [(702, inchis[702])]]
    odors = {o["name"]: o for o in db.list_odors()}
    assert {n: o["vapor_pressure_kpa"] for n, o in odors.items()} == {"valeric acid": 0.0262, "ethanol": None, "acetic acid": None,
                                                                    "no cid": None}
    assert odors["ethanol"]["vapor_pressure_at"] and not odors["acetic acid"]["vapor_pressure_at"]
    # linked to the section of the PubChem page it is from
    assert odormeta.links(odors["valeric acid"])["vapor_pressure"] == "https://pubchem.ncbi.nlm.nih.gov/compound/7991#section=Vapor-Pressure"
    assert odormeta.links(odors["ethanol"])["vapor_pressure"] is None

    # those not found are not looked up again, and those that could not be only after an hour
    main._fetch_vapor_pressures()
    assert len(asked) == 2
    main._vapor_tried[176] -= 3601
    main._fetch_vapor_pressures()
    assert asked[2:] == [[(176, inchis[176])]]
    # a sync changing other fields keeps it with the odor (it belongs to the CID)
    apply_all(db, rows_of(HEADER + "valeric acid,va,pentanoic acid,7991,,water,,,,,,,\n"), changes={o["id"] for o in db.list_odors()})
    assert next(o for o in db.list_odors() if o["name"] == "valeric acid")["vapor_pressure_kpa"] == 0.0262


def test_vapor_pressures_in_tables(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    apply_all(db, rows_of(HEADER + "valeric acid,va,,7991,,water,,,,,,,\n"))
    db.save_inchis({7991: "InChI=1S/C5H10O2/c1-2-3-4-5(6)7/h2-4H2,1H3,(H,6,7)"}, when="2026-10-23T00:00:00")
    db.save_vapor_pressure(7991, {"kpa": 0.0262, "source": "https://pubchem.ncbi.nlm.nih.gov/compound/7991"}, when="2026-10-23T00:00:00")
    index = odormeta.OdorIndex(db.list_odors())
    rows = [{"panel": "p", "odor": "va @ -3", "label": "va", "log10_conc": -3, "yaml": None, "odor_id": next(iter(index.by_id)),
             "match": "abbrev"}]
    (row,) = odormeta.table_rows(rows, index)
    assert row["vapor_pressure_kpa"] == 0.0262 and row["vapor_pressure_url"].endswith("/7991#section=Vapor-Pressure")

