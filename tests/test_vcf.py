"""Tests of the Headspace calculator's search of VCF's foods, and the paraffin oil to match
them. Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import json
import time
from collections import OrderedDict

import pytest

from app import chemprops, config, headspace, reference, vcf
from tests.test_headspace import EA, PFO, main  # noqa: F401 (fixture)
from tests.test_tensor import call

INCHI_EA = "1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3"
INCHI_ETOH = "1S/C2H6O/c1-2-3/h3H,2H2,1H3"
INCHI_LIN = "1S/C10H18O/c1-5-10(4,11)8-6-7-9(2)3/h5,7,11H,1,6,8H2,2-4H3"
INCHI_HEX = "1S/C6H12O/c1-2-3-4-5-6-7/h6H,2-5H2,1H3"
HEADER = ("inchi,name,quantity,product_name,reference,cid,k_henry,k_henry_sources,k_henry_units,vapor_pressure,"
          "vapor_pressure_sources,vapor_pressure_units,group,category,mid_ppm\n")
KH = "Sander (L) https://henrys-law.org/x,mol / m^3 / Pa"
VP = "https://pubchem.ncbi.nlm.nih.gov/compound/{},kPa"
# as VCF's CSV: ethyl acetate in two kiwis (one with three references, one quantified twice by one
# of them), ethanol (no Henry's law constant), linalool (its constant in other units), something
# with neither, nor an InChI, and hexanal (neither, in a banana)
CSV = HEADER + "".join(f"{line}\n" for line in [
    f'"{INCHI_EA}",ethyl acetate,1,KIWIFRUIT,Ref A,8857.0,0.059,{KH},11.3,{VP.format(8857)},Fruits,KIWIFRUIT,1',
    f'"{INCHI_EA}",ethyl acetate,2-4,KIWIFRUIT,Ref B,8857.0,0.059,{KH},11.3,{VP.format(8857)},Fruits,KIWIFRUIT,3',
    f'"{INCHI_EA}",ethyl acetate,4,KIWIFRUIT,Ref B,8857.0,0.059,{KH},11.3,{VP.format(8857)},Fruits,KIWIFRUIT,4',
    f'"{INCHI_EA}",ethyl acetate,10,KIWIFRUIT,Ref C,8857.0,0.059,{KH},11.3,{VP.format(8857)},Fruits,KIWIFRUIT,10',
    f'"{INCHI_EA}",ethyl acetate,40,KIWI WINE,Ref D,8857.0,0.059,{KH},11.3,{VP.format(8857)},Alcoholic Beverages,KIWI WINE,40',
    f'"{INCHI_ETOH}",ethanol,1000,KIWI WINE,Ref D,702.0,,,,7.9,{VP.format(702)},Alcoholic Beverages,KIWI WINE,1000',
    f'"{INCHI_LIN}",linalool,0.5,KIWIFRUIT,Ref A,6549.0,2.1,Sander,mol / kg / bar,0.02,{VP.format(6549)},Fruits,KIWIFRUIT,0.5',
    f'"{INCHI_LIN}",linalool,0.1,BANANA,Ref E,6549.0,2.1,Sander,mol / kg / bar,0.02,{VP.format(6549)},Fruits,BANANA,0.1',
    ',mystery,2,KIWIFRUIT,Ref A,,,,,,,,Fruits,KIWIFRUIT,2',
    f'"{INCHI_HEX}",hexanal,3,BANANA,Ref E,6184.0,,,,,,,Fruits,BANANA,3',
])
FIELDS = {"layout": "list", "rows": {"label": "name", "keep": ["inchi"]}, "cols": {"label": "product_name", "keep": []},
          "value": "mid_ppm"}


@pytest.fixture
def table(tmp_path, monkeypatch):
    monkeypatch.setattr(vcf, "_memo", OrderedDict())
    p = tmp_path / "vcf.csv"
    p.write_text(CSV)
    return vcf.load({"id": "vcf", "title": "VCF", "path": str(p), "fields": FIELDS})


def test_molar_mass():
    assert vcf.molar_mass(f"InChI={INCHI_EA}") == pytest.approx(88.106) and vcf.molar_mass(INCHI_ETOH) == pytest.approx(46.069)
    assert vcf.molar_mass("InChI=1S/2C2H4O2.Ca/c2*1-2(3)4;/h2*1H3,(H,3,4);/q;;+2/p-2") == pytest.approx(2 * 60.052 + 40.078)
    assert vcf.molar_mass("InChI=1S/CUu/c1") is None and vcf.molar_mass(None) is None and vcf.molar_mass("nope") is None


def test_sample():
    ea = {"molar_mass": 88.11, "k_henry": 0.059, "vapor_pressure": 12.6}
    henry, capped, raoult, none, nomass, too_much = headspace.sample([
        {"ppm": 10, **ea}, {"ppm": 5e5, **ea}, {"ppm": 10, **ea, "k_henry": None}, {"ppm": 10, **ea, "k_henry": None, "vapor_pressure": None},
        {"ppm": 10, **ea, "molar_mass": None}, {"ppm": 2e6, **ea}])
    # by Henry's law, its concentration in the sample (as water) over the constant
    assert henry["law"] == "henry" and henry["pa"] == pytest.approx(10e-3 / 88.11 * 997 / 0.059) and henry["notes"] == []
    assert capped["pa"] == 12600 and "at its vapor pressure" in capped["notes"][0]
    # else by Raoult's law, said to be likely too low
    n = 10e-3 / 88.11
    assert raoult["law"] == "raoult" and raoult["pa"] == pytest.approx(n / (n + (1 - 1e-5) * 1000 / 18.015) * 12600)
    assert "likely too low" in raoult["notes"][0]
    assert none["pa"] is None and none["law"] is None and "no Henry's law constant or vapor pressure" in none["notes"][0]
    assert nomass["pa"] is None and too_much["pa"] is None and "more than the whole sample" in too_much["notes"][0]


def test_invert():
    other = {**EA, "molar_mass": 46.07, "density": 0.789, "vapor_pressure": 7.9}
    comps = [{**EA, **PFO, "pa": 50.0}, {**other, "pa": 30.0}, {**EA, "density": None, "pa": 2.0}, {**EA, "pa": None}]
    for model in headspace.MODELS:
        res = headspace.invert(comps, "paraffin oil", model)
        a, b, c, d = res["components"]
        # back again: the headspace asked for, of the mixture of them all
        given = [{**x, "log10_conc": r["log10_conc"]} for x, r in zip(comps[:3], res["components"])]
        fwd = headspace.calculate(given, "paraffin oil", model)
        assert [r["pa"] for r in fwd["components"]] == pytest.approx([50, 30, 2])
        assert [r["law"] for r in (a, b, c)] == [r["law"] for r in fwd["components"]]
        assert [r["mole_fraction"] for r in (a, b, c)] == pytest.approx([r["mole_fraction"] for r in fwd["components"]])
        assert res["total"] == pytest.approx(sum(10 ** r["log10_conc"] for r in (a, b, c)))
        assert d["log10_conc"] is None and d["notes"] == [] and c["notes"][0] == "no density found, so taken as 1 g/mL"
    # by Henry's law, from its own headspace alone
    (a,) = headspace.invert([{**EA, **PFO, "pa": 50.0}])["components"]
    assert a["law"] == "henry" and 10 ** a["log10_conc"] == pytest.approx(50 * 0.0977 / 1e6 * 88.11 / 0.902)
    # what cannot be reached: at its vapor pressure, or with none of it; Raoult's law with no vapor pressure
    hi, zero, novp = headspace.invert([{**EA, **PFO, "pa": 12610.0}, {**EA, **PFO, "pa": 0.0},
                                       {**EA, "vapor_pressure": None, "pa": 1.0}])["components"]
    assert hi["log10_conc"] is None and "at least its vapor pressure" in hi["notes"][-1]
    assert zero["log10_conc"] is None and zero["notes"] == ["no headspace, so none of it"]
    assert novp["log10_conc"] is None and novp["notes"][-1] == "no vapor pressure, so its headspace cannot be computed"
    # more than the whole solution, by Raoult's law: none of those
    (x, y) = headspace.invert([{**EA, "pa": 9000.0}, {**other, "pa": 6000.0}], "paraffin oil", "raoult")["components"]
    assert x["log10_conc"] is None and "more than the whole solution" in y["notes"][-1]


def test_search(table):
    assert [c["name"] for c in table["chemicals"]] == ["ethyl acetate", "ethanol", "linalool", "mystery", "hexanal"]
    assert table["names"] == {"products": ["BANANA", "KIWI WINE", "KIWIFRUIT"], "categories": ["BANANA", "KIWI WINE", "KIWIFRUIT"],
                              "groups": ["Alcoholic Beverages", "Fruits"]}
    ea, etoh, lin, mystery, _ = table["chemicals"]
    assert (ea["inchi"], ea["cid"], ea["k_henry"], ea["vapor_pressure"]) == (f"InChI={INCHI_EA}", 8857, 0.059, 11.3)
    assert ea["k_henry_source"].startswith("Sander") and ea["vapor_pressure_source"].endswith("/8857")
    assert lin["k_henry"] == pytest.approx(2.1 * 0.00997) and etoh["k_henry"] is None and mystery["molar_mass"] is None

    # in a product's name, category or group, but for case
    r = vcf.search(table, " kiwi ")
    assert r["query"] == "kiwi" and [p["name"] for p in r["products"]] == ["KIWI WINE", "KIWIFRUIT"]
    assert r["products"][1] == {"name": "KIWIFRUIT", "category": "KIWIFRUIT", "group": "Fruits", "lines": 6, "references": 3,
                                "chemicals": 3}
    names = [c["name"] for c in r["chemicals"]]
    assert names == ["ethanol", "ethyl acetate", "mystery", "linalool"]  # (the highest quantity first)
    assert len(r["lines"]) == 8 and r["lines"][0] == [names.index("ethyl acetate"), 1, 0, "1", 1.0]
    assert r["references"] == ["Ref A", "Ref B", "Ref C", "Ref D"]
    # the median of each product's references (each reference's lines averaged), then of the products
    c = r["chemicals"][1]
    assert c["per_product"] == [[0, 40.0, 1], [1, 3.5, 3]] and c["ppm"] == pytest.approx((3.5 + 40) / 2)
    assert (c["lines"], c["references"], c["products"], c["min"], c["max"]) == (5, 4, 2, 1.0, 40.0)
    # its headspace over the samples, as water: of the quantity, and of each product's; by Henry's law, else Raoult's
    vcf.add_headspace(r["chemicals"])
    assert c["file"] == {k: table["chemicals"][0][k] for k in ("k_henry", "k_henry_source", "vapor_pressure", "vapor_pressure_source")}
    assert (c["k_henry_from"], c["vapor_pressure_from"], c["mismatches"], c["pending"]) == ("file", "file", [], [])
    (expect,) = headspace.sample([{"ppm": c["ppm"], "molar_mass": c["molar_mass"], "k_henry": 0.059, "vapor_pressure": 11.3}])
    assert c["law"] == "henry" and c["pa"] == pytest.approx(expect["pa"]) and c["per_product_pa"][1] == pytest.approx(expect["pa"] * 3.5 / 21.75)
    assert r["chemicals"][0]["law"] == "raoult" and "likely too low" in r["chemicals"][0]["headspace_notes"][0]
    assert r["chemicals"][2]["pa"] is None and "no molar mass" in r["chemicals"][2]["headspace_notes"][0]

    # or the mean, geometric mean, ...
    c = next(c for c in vcf.search(table, "KIWIFRUIT", "mean")["chemicals"] if c["name"] == "ethyl acetate")
    assert c["ppm"] == pytest.approx((1 + 3.5 + 10) / 3)
    c = next(c for c in vcf.search(table, "kiwi", "geomean", "max")["chemicals"] if c["name"] == "ethyl acetate")
    assert c["ppm"] == 40 and c["per_product"][1][1] == pytest.approx(3.5 ** (1 / 3) * 10 ** (1 / 3))
    assert [c["name"] for c in vcf.search(table, "beverages")["chemicals"]] == ["ethanol", "ethyl acetate"]
    assert vcf.search(table, "durian")["chemicals"] == []
    with pytest.raises(vcf.VcfError, match="type part of"):
        vcf.search(table, "  ")
    with pytest.raises(vcf.VcfError, match="one of median"):
        vcf.search(table, "kiwi", "mode")


def test_add_headspace(table):
    r = vcf.search(table, "kiwi")
    names = [c["name"] for c in r["chemicals"]]
    found = lambda value, source="chemutils", measured=None: {"value": value, "source": source, "measured": measured}
    # what chemutils looks up used rather than the file's, saying where they differ (by more than 10%);
    # the file's where it finds none; estimates where neither has one
    looked = {8857: {"k_henry": found(0.0592), "vapor_pressure": found(12.6, measured=True)},
              702: {"k_henry": found(0.19), "vapor_pressure": found(7.9)},
              6549: {"k_henry": found(None), "vapor_pressure": found(None), "est_k_henry": found(9.0)}}
    vcf.add_headspace(r["chemicals"], looked, {(6549, "est_vapor_pressure")})
    ea, etoh, _, lin = (r["chemicals"][names.index(n)] for n in ("ethyl acetate", "ethanol", "mystery", "linalool"))
    assert (ea["k_henry"], ea["k_henry_from"], ea["vapor_pressure"], ea["vapor_pressure_from"], ea["vapor_pressure_measured"]) == (
        0.0592, "chemutils", 12.6, "chemutils", True)
    assert ea["mismatches"] == [{"prop": "vapor_pressure", "file": 11.3, "looked_up": 12.6}] and ea["file"]["vapor_pressure"] == 11.3
    (expect,) = headspace.sample([{"ppm": ea["ppm"], "molar_mass": ea["molar_mass"], "k_henry": 0.0592, "vapor_pressure": 12.6}])
    assert ea["pa"] == pytest.approx(expect["pa"])
    # (by Henry's law, with a constant chemutils found and the file did not have)
    assert (etoh["law"], etoh["k_henry_from"], etoh["mismatches"]) == ("henry", "chemutils", [])
    assert (lin["k_henry"], lin["k_henry_from"], lin["vapor_pressure_from"]) == (pytest.approx(2.1 * 0.00997), "file", "file")
    assert lin["pending"] == ["est_vapor_pressure"]
    # an estimated constant, said so
    r = vcf.search(table, "banana")
    hexanal = next(c for c in r["chemicals"] if c["name"] == "hexanal")
    vcf.add_headspace(r["chemicals"], {6184: {"k_henry": found(None), "vapor_pressure": found(None),
                                              "est_k_henry": found(0.05, "VP/AS estimate", False), "est_vapor_pressure": found(1.0)}})
    assert (hexanal["law"], hexanal["k_henry"], hexanal["k_henry_from"], hexanal["k_henry_measured"]) == ("henry", 0.05, "estimate", False)
    assert hexanal["vapor_pressure_from"] == "estimate" and "VP/AS" in hexanal["headspace_notes"][0]


# what the fake chemutils finds of VCF's chemicals (None: nothing), by CID
DETAILS = PFO["k_henry_pfo_details"]
LOOKED = {
    8857: {"k_henry": 0.059, "vapor_pressure": 12.6, "density": 0.902, "k_henry_pfo": 0.0977},
    702: {"k_henry": 0.19, "vapor_pressure": 7.9, "k_henry_pfo": None, "est_vapor_pressure": 7.0},
    6549: {"k_henry": None, "vapor_pressure": None, "density": 0.86, "k_henry_pfo": 1.5, "est_vapor_pressure": 0.03},
    6184: {"k_henry": None, "vapor_pressure": None, "est_k_henry": 0.05, "est_vapor_pressure": 1.0},
}


@pytest.fixture
def env(main, tmp_path, monkeypatch):  # noqa: F811
    """main (as the Headspace calculator's tests have it), with VCF's CSV as a built-in reference
    dataset, and chemutils finding `LOOKED` of VCF's chemicals (``main.asked``: each lookup's)."""
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "vcf.csv").write_text(CSV)
    (builtin / "fruits.csv").write_text("chemical,banana\nethyl acetate,1\n")
    (builtin / "manifest.json").write_text(json.dumps({"datasets": [
        {"id": "fruits", "title": "Fruits", "file": "fruits.csv"},
        {"id": "vcf", "title": "VCF 16.10", "file": "vcf.csv", "fields": FIELDS}]}))
    monkeypatch.setattr(config, "REFERENCE_DIR", builtin)
    monkeypatch.setattr(reference, "_memo", OrderedDict())
    monkeypatch.setattr(vcf, "_memo", OrderedDict())
    for name, value in (("_vcf_queue", {}), ("_vcf_busy", set()), ("_vcf_worker", {"running": False, "asked": 0})):
        monkeypatch.setattr(main, name, value)

    def look_up(odors, save, refresh=False):
        odors = list(odors)
        main.asked.append(odors)
        failed = {}
        for cid, inchi, props in odors:
            for p in props:
                if (cid, p) == (702, "density"):
                    failed[(cid, p)] = "ValueError: multiple unit definitions"
                elif p == "k_henry_pfo":
                    v = LOOKED[cid][p]
                    save(cid, p, {"value": v, "source": "UNIFAC estimate", "measured": False,
                                  "details": DETAILS if v else {"notes": "no UNIFAC groups"}})
                else:
                    v = LOOKED[cid].get(p)
                    save(cid, p, None if v is None else {"value": v, "source": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                                                          "measured": p == "vapor_pressure"})
        return failed

    monkeypatch.setattr(chemprops, "look_up", look_up)
    return main


def wait(main):  # noqa: F811
    for _ in range(200):
        if not main._vcf_worker["running"]:
            return
        time.sleep(0.01)
    raise AssertionError("the lookups did not finish")


def test_endpoint(env, monkeypatch):
    status, r = call(env.headspace_vcf)
    assert status == 200 and [s["id"] for s in r["sources"]] == ["vcf"] and r["stats"]["median"] == "median"
    assert r["sources"][0]["products"] == ["BANANA", "KIWI WINE", "KIWIFRUIT"] and r["sources"][0]["lines"] == 10

    # each chemical's Henry's law constant and vapor pressure looked up in the background, one CID at a time
    # here: those the file has neither of first, then those it has no constant of, then those to check
    monkeypatch.setattr(env, "VCF_LOOKUP_BATCH", 1)
    body = {"query": "a"}
    status, r = call(env.headspace_vcf, body=body)
    assert status == 200 and r["source"] == {"id": "vcf", "title": "VCF 16.10"} and "oil" not in r
    assert r["lookups"] == {"pending": 4, "mismatches": 0, "tolerance": 0.1}
    by_name = {c["name"]: c for c in r["chemicals"]}
    assert by_name["hexanal"]["pending"] == ["k_henry", "vapor_pressure"] and by_name["hexanal"]["pa"] is None
    wait(env)
    assert [[(cid, props) for cid, _, props in batch] for batch in env.asked] == [
        [(6184, ["k_henry", "vapor_pressure"])], [(702, ["k_henry", "vapor_pressure"])],
        [(8857, ["k_henry", "vapor_pressure"])], [(6549, ["k_henry", "vapor_pressure"])]]
    # kept apart from the odor table's (8857 is one of its odors)
    assert env.db.vcf_properties([8857])[8857]["vapor_pressure"]["value"] == 12.6
    assert next(o for o in env.db.list_odors() if o["cid"] == 8857)["vapor_pressure_at"] is None
    # then their estimates, where neither the file nor chemutils has one
    del env.asked[:]
    r = call(env.headspace_vcf, body=body)[1]
    wait(env)
    assert [[(cid, props) for cid, _, props in batch] for batch in env.asked] == [
        [(6184, ["est_k_henry", "est_vapor_pressure"])]]
    r = call(env.headspace_vcf, body=body)[1]
    by_name = {c["name"]: c for c in r["chemicals"]}
    assert r["lookups"] == {"pending": 0, "mismatches": 1, "tolerance": 0.1} and by_name["hexanal"]["k_henry_from"] == "estimate"
    # what chemutils found used rather than the file's, saying where it differs
    ea, etoh, lin = by_name["ethyl acetate"], by_name["ethanol"], by_name["linalool"]
    assert ea["mismatches"] == [{"prop": "vapor_pressure", "file": 11.3, "looked_up": 12.6}]
    assert (etoh["law"], etoh["k_henry"], etoh["k_henry_from"]) == ("henry", 0.19, "chemutils")
    assert (lin["k_henry_from"], lin["vapor_pressure_from"]) == ("file", "file")
    # matched to the odor table (ethyl acetate and ethanol are in it)
    ids = {o["name"]: o["id"] for o in env.db.list_odors()}
    assert ea["odor_id"] == ids["ethyl acetate"] and lin["odor_id"] is None

    # the paraffin oil to match: what it needs looked up in the background (but the molar mass, from the InChI)
    del env.asked[:]
    body = {"query": "kiwi", "model": "henry"}
    status, r = call(env.headspace_vcf, body=body)
    assert status == 200 and r["oil"]["model"] == "henry" and r["oil"]["error"] is None and r["oil"]["pending"] == 3
    wait(env)
    assert sorted(p for batch in env.asked for _, _, props in batch for p in props) == sorted(["density", "k_henry_pfo"] * 3 + ["est_vapor_pressure"])
    status, r = call(env.headspace_vcf, body=body)
    oil = r["oil"]
    assert oil["pending"] == 0
    names = [c["name"] for c in r["chemicals"]]
    ea, etoh, mystery = (oil["chemicals"][names.index(n)] for n in ("ethyl acetate", "ethanol", "mystery"))
    assert ea["law"] == "henry" and etoh["law"] == "raoult" and ea["pending"] is False
    assert mystery is None  # (no headspace)
    # as headspace.invert gives it, of them all together
    looked = env.db.vcf_properties([c["cid"] for c in r["chemicals"] if c["cid"]])
    given = []
    for c in r["chemicals"]:
        if c["pa"] is None:
            given.append({"pa": None})
            continue
        v = lambda p: (looked[c["cid"]].get(p) or {}).get("value")
        given.append({"pa": c["pa"], "molar_mass": c["molar_mass"], "density": v("density"), "vapor_pressure": v("vapor_pressure"),
                      "est_vapor_pressure": v("est_vapor_pressure"), "k_henry_pfo": v("k_henry_pfo"),
                      "k_henry_pfo_details": looked[c["cid"]]["k_henry_pfo"]["details"]})
    expect = headspace.invert(given, "paraffin oil", "henry")
    assert ea["log10_conc"] == pytest.approx(expect["components"][names.index("ethyl acetate")]["log10_conc"])
    assert oil["total"] == pytest.approx(expect["total"])
    assert ea["links"]["density"] == "https://pubchem.ncbi.nlm.nih.gov/compound/8857#section=Density"
    assert etoh["failed"] == {"density": "ValueError: multiple unit definitions"}
    # Raoult's law only: nothing more to look up
    n = len(env.asked)
    r = call(env.headspace_vcf, body={**body, "model": "raoult"})[1]
    assert [c and c["law"] for c in r["oil"]["chemicals"]] == ["raoult", "raoult", None, "raoult"] and len(env.asked) == n

    # too many to calculate in oil
    monkeypatch.setattr(env, "VCF_MAX_OIL", 2)
    r = call(env.headspace_vcf, body={**body, "model": "henry"})[1]
    assert "3 chemicals with a headspace, and at most 2 are calculated in oil" in r["oil"]["error"]


@pytest.mark.parametrize("body, error", [
    ({"query": ""}, "type part of"),
    ({"query": "kiwi", "source": "fruits"}, "no such VCF reference dataset"),
    ({"query": "kiwi", "model": "unifac"}, "the model is one of"),
    ({"query": "kiwi", "across_products": "mode"}, "one of median"),
    (["kiwi"], "expected"),
])
def test_endpoint_errors(env, body, error):
    status, r = call(env.headspace_vcf, body=body)
    assert status == 400 and error in r["error"]
