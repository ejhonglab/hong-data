"""Each trial's dF/F image of a movie, and each odor's mean (trial_dff.py, the movie player's), and the API that makes them."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
import tifffile

from app import config, trial_dff
from tests.test_fits import call
from tests.test_tensor import main, root  # noqa: F401 (fixtures)


@pytest.fixture
def rec(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    d = tmp_path / "rec"
    d.mkdir()
    rng = np.random.default_rng(0)
    movie = rng.integers(100, 200, (24, 2, 5, 6)).astype(np.uint16)
    trials = [{"start_frame": 6 * k, "first_odor_frame": 6 * k + 2, "end_frame": 6 * k + 5, "odors": f"o{k // 2} @ -3"} for k in range(4)]
    tifffile.imwrite(d / "mocorr.tif", movie, imagej=True, metadata={"axes": "TZYX"})
    (d / "trial_frames_and_odors.json").write_text(json.dumps(trials))
    return d, movie, trials


def test_compute(rec):
    d, movie, trials = rec
    f = trial_dff.files(d / "mocorr.tif")
    assert f["source"] == "computed" and not f["ready"]
    seen = []
    f = trial_dff.compute(d / "mocorr.tif", d / "trial_frames_and_odors.json", lambda a, b: seen.append((a, b)))
    assert f["ready"] and seen[-1] == (4, 4)
    got, means = tifffile.imread(f["trials"]), tifffile.imread(f["odors"])
    m = movie.astype(np.float64)
    want = np.stack([((m[t["first_odor_frame"]:t["first_odor_frame"] + 2] - m[t["start_frame"]:t["first_odor_frame"]].mean(0))
                      / m[t["start_frame"]:t["first_odor_frame"]].mean(0)).mean(0) for t in trials])
    assert got.shape == (4, 2, 5, 6) and got.dtype == np.float32 and np.allclose(got, want, atol=1e-6)
    assert means.shape == (2, 2, 5, 6) and np.allclose(means[1], got[2:].mean(0), atol=1e-6)
    assert [x["odor"] for x in trial_dff.labels(d / "trial_frames_and_odors.json", "odors")] == ["o0 @ -3", "o1 @ -3"]
    assert trial_dff.labels(d / "trial_frames_and_odors.json", "trials")[3] == {"start": 3, "onset": 3, "end": 3, "odor": "o1 @ -3"}
    # al_analysis' own, where they are as new as the movie
    for n in trial_dff.OWN[False]:
        tifffile.imwrite(d / n, np.zeros((1, 2, 5, 6), np.float32))
    assert trial_dff.files(d / "mocorr.tif")["source"] == "al_analysis"
    old = os.stat(d / "mocorr.tif").st_mtime - 100
    os.utime(d / "trial_dff.tif", (old, old))
    assert trial_dff.files(d / "mocorr.tif")["source"] == "computed"


def test_api(main, root, rec):  # noqa: F811
    d, _, _ = rec
    movie = root / "ann/rec"
    movie.mkdir(parents=True)
    for n in ("mocorr.tif", "trial_frames_and_odors.json"):
        (movie / n).write_bytes((d / n).read_bytes())
    item = {"path": str(movie / "mocorr.tif"), "trials": str(movie / "trial_frames_and_odors.json")}
    ask = lambda: call(main.movie_trial_dff, method="POST", body={"movies": [item], "first": item["path"]})[1]["movies"][0]  # noqa: E731
    s = ask()
    assert s["status"] in ("queued", "running", "ready") and s["path"] == item["path"]
    for _ in range(100):
        if (s := ask())["status"] == "ready":
            break
        time.sleep(0.05)
    assert s["status"] == "ready" and s["source"] == "computed"
    assert call(main.movie_info, query={"path": s["trials"]})[1]["t"] == 4
    labels = call(main.movie_trials, query={"path": item["trials"], "by": "odors"})[1]["trials"]
    assert [x["odor"] for x in labels] == ["o0 @ -3", "o1 @ -3"]
    assert call(main.movie_trial_dff, method="POST", body={"movies": "x"})[0] == 400
