"""Tests of the Headspace calculator's serial dilutions. Run with: OLFDB_DATA_DIR=<a scratch
folder> python -m pytest tests/"""
from __future__ import annotations

import math

import pytest

from app import dilution
from tests.test_headspace import main  # noqa: F401 (fixture)
from tests.test_tensor import call


def _made(o: dict, solvent_ul: float = 2000.0) -> list[float]:
    """The log10 v/v of each of an odor's vials, from its volumes as pipetted."""
    vials = [math.log10(o["neat_ul"] / (o["neat_ul"] + solvent_ul))]
    for _ in range(len(o["vials"]) - 1):
        vials.append(vials[-1] + math.log10(o["dilution_ul"] / (o["dilution_ul"] + solvent_ul)))
    return vials


def test_plan():
    r = dilution.plan([-3, -4, -5, -6])
    direct, *diluted = r["odors"]
    # within the limits of the neat chemical with none: 2 µL into 2 mL
    assert (direct["neat_ul"], direct["dilution_ul"], direct["ok"], direct["notes"]) == (2.0, None, True, [])
    assert direct["vials"] == pytest.approx([-3], abs=1e-3)
    # the others: one volume of dilutions, the least by default, with the fewest of them (1, 2, 3)
    ((volume),) = r["volumes"]
    assert volume["odors"] == [1, 2, 3] and volume["within"] and volume["ul"] == 100
    assert [len(o["vials"]) - 1 for o in diluted] == [1, 2, 3]
    for t, o in zip([-4, -5, -6], diluted):
        assert o["ok"] and o["dilution_ul"] == volume["ul"] and 2 <= o["neat_ul"] <= 200
        assert o["vials"] == pytest.approx(_made(o)) and o["vials"][-1] == pytest.approx(t, abs=2e-3)
    assert r["dilution_limits"] == [100.0, 2000.0]

    # the neat chemical itself; more than the most of it; just more than the least gives: a dilution
    neat, much, near = dilution.plan([0, -1, -3.02])["odors"]
    assert (neat["neat_ul"], neat["vials"], neat["notes"]) == (None, [], ["the neat chemical itself, undiluted"])
    assert much["neat_ul"] == 222.0 and not much["ok"] and much["notes"] == ["needs 222 µL of the neat chemical, more than the most (200 µL)"]
    assert near["ok"] and len(near["vials"]) == 2 and near["vials"][-1] == pytest.approx(-3.02, abs=2e-3)


def test_limits():
    # the dilutions within their limits, where every odor can still be made within the neat chemical's
    r = dilution.plan([-3, -4, -5, -6], dilution_ul=(20, 50))
    assert r["dilution_limits"] == [20, 50] and [v["ul"] for v in r["volumes"]] == [20.0] and r["volumes"][0]["within"]
    assert all(o["ok"] and o["vials"][-1] == pytest.approx(t, abs=2e-3) for t, o in zip([-3, -4, -5, -6], r["odors"]))
    # else outside them, for them all, said so
    r = dilution.plan([-5, -6.3, -4.4], neat_ul=(5, 8), dilution_ul=(10, 20))
    ((volume),) = r["volumes"]
    assert not volume["within"] and volume["odors"] == [0, 1, 2] and not 10 <= volume["ul"] <= 20
    assert all(o["ok"] and 5 <= o["neat_ul"] <= 8 for o in r["odors"])
    # with as much neat chemical for every odor, a volume of each's own (that just makes it), with none for both
    r = dilution.plan([-5, -6.3], neat_ul=(10, 10))
    assert [v["odors"] for v in r["volumes"]] == [[0], [1]] and r["volumes"][0]["ul"] != r["volumes"][1]["ul"]
    for t, o in zip([-5, -6.3], r["odors"]):
        assert o["ok"] and o["neat_ul"] == 10 and o["vials"][-1] == pytest.approx(t, abs=2e-3)
    # by default, at least 100 µL, or the most if less
    assert dilution.plan([-5], dilution_ul=(None, 50))["dilution_limits"] == [50, 50]
    # less than the least of the neat chemical, with no volume of dilutions to try: said so
    (o,) = dilution.plan([-5], solvent_ml=0.001, dilution_ul=(5, None))["odors"]
    assert not o["ok"] and o["notes"][0].startswith("needs 1e-05 µL of the neat chemical, less than the least (2 µL)")


def test_fewest():
    # fewer dilutions over larger volumes; then the least volume pipetted as large as can be
    (o,) = dilution.plan([-9])["odors"]
    assert (len(o["vials"]), o["dilution_ul"]) == (6, 100) and o["vials"][-1] == pytest.approx(-9, abs=2e-3)
    (o,) = dilution.plan([-9], dilution_ul=(2, None))["odors"]
    assert len(o["vials"]) == 3 and o["vials"][-1] == pytest.approx(-9, abs=2e-3)
    (o,) = dilution.plan([-4], dilution_ul=(2, None))["odors"]
    assert len(o["vials"]) == 2 and min(o["neat_ul"], o["dilution_ul"]) >= 20
    # then round: not over larger volumes (not 25 µL, with less of the neat chemical)
    (o,) = dilution.plan([-4], dilution_ul=(23.7, None))["odors"]
    assert (len(o["vials"]), o["dilution_ul"]) == (2, 23.7)
    # in another amount of solvent, e.g. 5 mL
    (o,) = dilution.plan([-5], solvent_ml=5)["odors"]
    assert o["ok"] and o["vials"] == pytest.approx(_made(o, 5000)) and o["vials"][-1] == pytest.approx(-5, abs=2e-3)
    # many odors, at many concentrations: one volume, quickly
    targets = [-50, -20, -7.3, -3.1, -4.9, -6.6, -2.2] * 7 + [-8]
    r = dilution.plan(targets)
    assert len(r["volumes"]) == 1 and all(o["ok"] and o["vials"][-1] == pytest.approx(t, abs=5e-3) for t, o in zip(targets, r["odors"]))


def test_endpoint(main):
    status, r = call(main.headspace_dilutions, body={"targets": [-3, -5], "solvent_ml": 2, "neat_ul": [2, 200], "dilution_ul": [20, None]})
    assert status == 200 and r["dilution_limits"] == [20, 2000] and [v["ul"] for v in r["volumes"]] == [20.0]
    assert r["odors"][1]["dilution_ul"] == 20 and r["odors"][1]["vials"][-1] == pytest.approx(-5, abs=2e-3)
    status, r = call(main.headspace_dilutions, body={"targets": [-4]})
    assert status == 200 and r["dilution_limits"] == [100, 2000]


@pytest.mark.parametrize("body, error", [
    ({"targets": "-3"}, "expected"),
    ({"targets": []}, "from 1 to 50 odors"),
    ({"targets": [-3, 1]}, "a number from -50 to 0"),
    ({"targets": [True]}, "a number from -50 to 0"),
    ({"targets": [-3], "solvent_ml": 0}, "the solvent in each vial"),
    ({"targets": [-3], "solvent_ml": 2000}, "at most 1,000"),
    ({"targets": [-3], "neat_ul": [200, 2]}, "the neat chemical's least and most"),
    ({"targets": [-3], "neat_ul": [0, 2]}, "the neat chemical's least and most"),
    ({"targets": [-3], "dilution_ul": [50, 20]}, "the dilutions' least and most"),
    ({"targets": [-3], "dilution_ul": [-1, None]}, "the dilutions' least and most"),
    ({"targets": [-3], "dilution_ul": [None]}, "the dilutions' least and most"),
])
def test_endpoint_errors(main, body, error):
    status, r = call(main.headspace_dilutions, body=body)
    assert status == 400 and error in r["error"]
