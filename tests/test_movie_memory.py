"""Movies held in memory for the player's plots (movie_memory.py)."""
from __future__ import annotations

import time

import pytest

from app import movie_memory, movies, roi_responses
from tests.test_roi_responses import DIAG, PANEL, fly
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)


@pytest.fixture(autouse=True)
def empty():
    movie_memory.limit(0)
    yield
    movie_memory.limit(0)


def ready(holder, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = next((h for h in movie_memory.status() if h["holder"] == holder[:8]), None)
        if s and s["state"] != "loading":
            return s
        time.sleep(0.02)
    raise AssertionError(f"{holder} still loading")


def paths(d):
    return roi_responses.movie_paths(d / movies.CONCAT, None, lambda p: p)


def test_hold(tmp_path, monkeypatch):
    d = fly(tmp_path)
    f, ps = paths(d)
    assert ps == [d / DIAG / "mocorr.tif", d / PANEL / "mocorr.tif"]
    plain = roi_responses.compute(d / movies.CONCAT, None, d / DIAG / movies.ROISET, 1, "DM2", 2, [], lambda p: p)
    s = movie_memory.hold("aaaaaaaa1", f, ps, 2)
    assert s["movies"] == 2 and ready("aaaaaaaa1")["state"] == "ready"
    assert movie_memory.array(ps[0]) is not None and movie_memory.array(ps[0]).shape == movies.open_movie(ps[0]).data.shape
    # the same numbers from memory, to the last bit
    held = roi_responses.compute(d / movies.CONCAT, None, d / DIAG / movies.ROISET, 1, "DM2", 2, [], lambda p: p)
    assert held["values"] == plain["values"]

    # asked again for the same fly: the same slot; a second player another; a third none, unless it plots (then the
    # least recently used is let go of: the first, used less recently than the second)
    assert movie_memory.hold("aaaaaaaa1", f, ps, 2)["state"] == "ready"
    movie_memory.hold("bbbbbbbb2", f, ps, 2)
    ready("bbbbbbbb2")
    assert movie_memory.hold("cccccccc3", f, ps, 2) is None
    assert movie_memory.hold("cccccccc3", f, ps, 2, evict=True) is not None
    assert {h["holder"] for h in movie_memory.status()} == {"bbbbbbbb", "cccccccc"}

    # a player closed lets go of them after a while, unless it holds them again first; a closed one goes first
    monkeypatch.setattr(movie_memory, "LINGER_S", 0.2)
    movie_memory.release("bbbbbbbb2")
    assert next(h for h in movie_memory.status() if h["holder"] == "bbbbbbbb")["frees_in_s"] is not None
    movie_memory.hold("bbbbbbbb2", f, ps, 2)
    time.sleep(0.3)
    movie_memory.sweep()
    assert len(movie_memory.status()) == 2
    movie_memory.release("cccccccc3")
    time.sleep(0.3)
    assert [h["holder"] for h in movie_memory.status()] == ["bbbbbbbb"]
    # fewer allowed: those used least recently let go of
    movie_memory.limit(0)
    assert movie_memory.status() == [] and movie_memory.array(ps[0]) is None
    assert movie_memory.hold("dddddddd4", f, ps, 0, evict=True) is None


def test_api(main, root):  # noqa: F811
    d = fly(root)
    assert call(main.admin_movie_memory)[1]["slots"] == movie_memory.DEFAULT_SLOTS
    status, got = call(main.movie_memory_hold, body={"holder": "eeeeeeee5", "path": str(d / movies.CONCAT)})
    assert status == 200 and got["slot"]["movies"] == 2
    ready("eeeeeeee5")
    # plotting holds them too (here: already held); the admin's number, set lower, lets go of them
    body = {"path": str(d / movies.CONCAT), "rois": str(d / DIAG / movies.ROISET), "index": 1, "name": "DM2", "window": 2, "holder": "eeeeeeee5"}
    assert call(main.movie_roi_responses, body=body)[1]["slot"]["state"] == "ready"
    assert call(main.admin_movie_memory_save, body={"slots": -1})[0] == 400
    status, got = call(main.admin_movie_memory_save, body={"slots": 0})
    assert status == 200 and got["slots"] == 0 and got["held"] == []
    assert call(main.movie_memory_hold, body={"holder": "eeeeeeee5", "path": str(d / movies.CONCAT)})[1]["slot"] is None
    assert call(main.movie_memory_release, body={"holder": "eeeeeeee5"})[0] == 200
