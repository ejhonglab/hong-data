"""Tests of the signals extracted from a fly's movies for its ROIs (extract.py), as al_analysis
extracts them: masks, traces, responses and the plane of an ROI used; its API, and the fit worker's
running of one. Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/."""
from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import roifile
import tifffile

from app import config, extract, fit_worker, movies, predictions
from app.ingest import load_file
from tests.test_fits import NAMES, call, data, request, tensor  # noqa: F401 (fixtures)
from tests.test_predictions import FLY, RAW, done_fit
from tests.test_tensor import main, root  # noqa: F401 (fixtures)

# masks as hong2p's ijroi2mask makes them (its code run with matplotlib, on polygons of x, y): what `polygon_mask` must give
MASKS = {
    "square": ([(2, 2), (5, 2), (5, 5), (2, 5)], ["........", "........", "........", "...###..", "...###..", "........", "........", "........"]),
    "diamond": ([(4, 1), (7, 4), (4, 7), (1, 4)], ["........", "........", "....#...", "...###..", "..######", "...###..", "....#...", "........"]),
    "diamond reversed": ([(1, 4), (4, 7), (7, 4), (4, 1)], ["........", "....#...", "...###..", "..#####.", "..######", "..#####.", "...###..", "....#..."]),
    "triangle": ([(1.25, 1.5), (6.75, 2.25), (3.5, 6.5)], ["........", "........", "..###...", "..#####.", "...###..", "...##...", "........", "........"]),
    "quarters": ([(2.25, 1.75), (6.0, 2.0), (6.5, 5.5), (4.0, 7.0), (1.0, 5.0)],
                 ["........", "........", "...###..", "..#####.", "..#####.", "..#####.", "...###..", "........"]),
}


def full_mask(points, shape) -> np.ndarray:
    y0, x0, m = extract.polygon_mask(np.asarray(points, float), shape)
    out = np.zeros(shape, bool)
    out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    return out


def test_masks():
    for name, (points, rows) in MASKS.items():
        want = np.array([[c == "#" for c in row] for row in rows])
        assert np.array_equal(full_mask(points, (8, 8)), want), name
    # (so not ImageJ's own, of pixels' centres: its 3 x 3 square is 2 x 3 here, and an outline drawn the other way round differs)
    assert full_mask(MASKS["diamond"][0], (8, 8)).sum() == 14 and full_mask(MASKS["diamond reversed"][0], (8, 8)).sum() == 24
    # off the plane, or of no area: no pixels
    assert extract.polygon_mask(np.array([[20, 20], [25, 20], [25, 25.0]]), (8, 8))[2].size == 0
    assert extract.polygon_mask(np.array([[1, 1], [3, 3.0]]), (8, 8))[2].size == 0
    assert full_mask([(-3, -3), (20, -3), (20, 20), (-3, 20)], (8, 8)).all()


def test_masks_as_matplotlib():
    """`inside` is matplotlib's own rule (which hong2p uses), also for points on the outline."""
    path = pytest.importorskip("matplotlib.path")
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:24, 0:24]
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    for step in (1, 0.5, 0.25, 0):
        for _ in range(100):
            pts = rng.uniform(-2, 26, (rng.integers(3, 10), 2))
            pts = np.round(pts / step) * step if step else pts
            assert np.array_equal(extract.inside(pts, xx, yy), path.Path(pts).contains_points(grid).reshape(24, 24))


# ------------------------------------------------------- a recording, made up

T, Z, Y, X = 40, 3, 16, 20
ROIS = [("DM2", 0, (2, 2, 7, 6)), ("DM2", 1, (2, 2, 7, 6)), ("DL5", 1, (10, 3, 16, 9)), ("VA1d", 2, (3, 9, 9, 14)),
        ("AL", 0, (0, 0, 19, 15)), ("DC2+", 2, (11, 8, 17, 13)), ("7", 0, (12, 9, 17, 14)), ("DL5", 2, (10, 3, 16, 9))]
TRIALS = [(0, 4, 9, "valeric acid", "va", -3), (10, 14, 19, "ethyl acetate", "ea", -2), (20, 24, 29, "valeric acid", "va", -3),
          (30, 35, 39, "ethyl acetate", "ea", -2)]


def recording(d, seed=0, gain=1.0, dtype="<u2"):
    """A recording as al_analysis leaves one: its movie (noise, and each ROI brighter after each odor comes on, DM2 more
    on its second plane), its ROIs (boxes, on their planes) and its trials."""
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    movie = rng.normal(200, 5, (T, Z, Y, X))
    for i, (name, z, (x0, y0, x1, y1)) in enumerate(ROIS):
        for k, (start, onset, end, *_) in enumerate(TRIALS):
            if name != "AL":  # (a pixel around each box too: masks are not quite the boxes drawn, see test_masks)
                movie[onset:onset + 3, z, max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1] += gain * 20 * (1 + i % 3 + k % 2) * (3 if (name, z) == ("DM2", 1) else 1)
    tifffile.imwrite(d / "mocorr.tif", movie.astype(dtype), imagej=dtype != "<f8", metadata={"axes": "TZYX"}, photometric="minisblack")
    for i, (name, z, (x0, y0, x1, y1)) in enumerate(ROIS):
        box = np.array([[x0, y0], [x1, y0], [x1 + 0.25, y1], [x0, y1 + 0.25]], "f4")
        roifile.ImagejRoi.frompoints(box, name=name, z=z).tofile(d / "RoiSet.zip", name=f"{i}.roi")
    (d / movies.TRIALS).write_text(json.dumps([{"start_frame": s, "first_odor_frame": o, "end_frame": e, "odors": f"{n} @ {c}",
                                                "name": n, "abbrev": a, "log10_conc": c} for s, o, e, n, a, c in TRIALS]))
    return tifffile.imread(d / "mocorr.tif")


def reference(movie, rois, trials):
    """Traces and responses computed as al_analysis and hong2p write them, step by step, with pandas."""
    traces = np.empty((len(movie), len(rois)))  # (float64, whatever the movie's numbers: as extract_traces_bool_masks')
    for i, r in enumerate(rois):
        traces[:, i] = np.mean(movie[:, r["z"]][:, full_mask_of(r, movie.shape[2:])], axis=1)
    traces = pd.DataFrame(traces)
    stats = []
    for t in trials:
        base = traces[t["start"]:t["onset"]].mean(axis=0)
        dff = (traces[t["onset"]:t["end"] + 1] - base) / base
        x = dff[:extract.N_RESPONSE]
        idx = np.abs(x).idxmax(axis=0)
        stats.append([x.loc[at, col] for col, at in zip(idx.index, idx)])
    stats = pd.DataFrame(stats)
    q = stats.max().to_frame("q")
    q["name"] = [r["name"] for r in rois]
    best = sorted(q.groupby("name", sort=False)["q"].idxmax())
    return traces.to_numpy(), stats.to_numpy(), best


def full_mask_of(r, shape) -> np.ndarray:
    out = np.zeros(shape, bool)
    out[r["y0"]:r["y0"] + r["mask"].shape[0], r["x0"]:r["x0"] + r["mask"].shape[1]] = r["mask"]
    return out


@pytest.mark.parametrize("dtype", ["<u2", ">f4", "<f8"])
def test_signals(tmp_path, dtype):
    d = tmp_path / "megamat1"
    movie = recording(d, dtype=dtype)
    said = []
    m = movies.open_movie(d / "mocorr.tif")
    rois = extract.read_rois(d / "RoiSet.zip", (Z, Y, X), said.append)
    # those of the antennal lobe's outline and ending in + left out, saying so; each on its plane, with its pixels
    assert [(r["name"], r["z"]) for r in rois] == [("DM2", 0), ("DM2", 1), ("DL5", 1), ("VA1d", 2), ("7", 0), ("DL5", 2)]
    assert any("DC2+" in s for s in said) and any("outlines of the whole antennal lobe" in s for s in said)
    assert rois[0]["pixels"] == rois[0]["mask"].sum() > 0 and (rois[0]["y0"], rois[0]["x0"]) == (2, 2)
    trials = extract.read_trials(d / movies.TRIALS)
    assert [(t["start"], t["onset"], t["end"], t["odors"]) for t in trials][:2] == [(0, 4, 9, ["va @ -3"]), (10, 14, 19, ["ea @ -2"])]

    traces = extract.plane_traces(m, rois)
    want_traces, want_stats, want_best = reference(movie, rois, trials)
    assert np.array_equal(traces, want_traces)  # (to the last bit: the same means, of the same numbers, in the same order)
    assert np.array_equal(extract.responses(traces, trials), want_stats)
    assert np.allclose(extract.responses(traces, trials)[0, 1], want_stats[0, 1]) and want_stats[0, 1] > 0.1

    # al_analysis' way: each ROI its plane of largest response in the recording (DM2's second)
    rec = {"traces": traces, "trials": trials}
    extract.combine("best_plane", rois, [rec])
    assert [i for i, p in enumerate(rec["planes"]) if p["best"]] == want_best
    assert rec["names"] == [rois[i]["name"] for i in want_best] and np.array_equal(rec["roi_traces"], traces[:, want_best])
    assert rec["names"][0] == "DM2" and 1 in want_best and 0 not in want_best and not rec["relative"]
    assert [p["weight"] for p in rec["planes"]] == [float(i in want_best) for i in range(len(rois))]

    # every voxel of it: its planes' means, by their pixels
    rec = {"traces": traces, "trials": trials}
    extract.combine("all_planes", rois, [rec])
    assert rec["names"] == ["DM2", "DL5", "VA1d", "7"]
    n0, n1 = rois[0]["pixels"], rois[1]["pixels"]
    assert np.allclose(rec["roi_traces"][:, 0], (traces[:, 0] * n0 + traces[:, 1] * n1) / (n0 + n1))
    both = np.concatenate([movie[:, 0][:, full_mask_of(rois[0], (Y, X))], movie[:, 1][:, full_mask_of(rois[1], (Y, X))]], axis=1)
    assert np.allclose(rec["roi_traces"][:, 0], both.mean(axis=1))
    assert np.array_equal(rec["roi_traces"][:, 2], traces[:, 3])  # (an ROI of one plane: that plane)

    # its planes by how little noise they have: of relative F, weights 1 / the variance of dF/F before the odor
    rec = {"traces": traces, "trials": trials}
    extract.combine("weighted", rois, [rec])
    w = np.array([p["weight"] for p in rec["planes"]])
    noise = extract.noise(traces, trials)
    assert rec["relative"] and np.isclose(w[0] + w[1], 1) and np.isclose(w[0] / w[1], noise[1] / noise[0]) and w[3] == 1
    assert np.allclose(rec["roi_traces"][:, 0], traces[:, 0] / traces[:, 0].mean() * w[0] + traces[:, 1] / traces[:, 1].mean() * w[1])
    pre = np.concatenate([(traces[t["start"]:t["onset"], 0] - traces[t["start"]:t["onset"], 0].mean()) / traces[t["start"]:t["onset"], 0].mean()
                          for t in trials])
    assert np.isclose(noise[0], (pre ** 2).sum() / (len(pre) - len(trials)))

    # dF/F at every time point of the trials, each against its own baseline
    rows, dff = extract.dff_traces(traces, trials)
    assert rows.shape == (40, 3) and list(rows[3]) == [3, 0, -1] and list(rows[14]) == [14, 1, 0]
    base = traces[10:14].mean(axis=0)
    assert np.allclose(dff[14], (traces[14] - base) / base) and np.allclose(dff[:4].mean(axis=0), 0)


def test_best_plane_of_the_fly(tmp_path):
    """One plane of an ROI for all of a fly's recordings: that of its largest response in any of them."""
    names = ["a", "a", "b"]
    rois = [{"name": n, "z": z, "pixels": 4} for n, z in zip(names, (0, 1, 0))]
    trials = [{"start": 0, "onset": 2, "end": 3, "odors": ["x @ -2"]}]
    rec1 = {"trials": trials, "traces": np.array([[1, 1, 1], [1, 1, 1], [3, 2, 1], [1, 1, 1.0]])}  # a: its first plane the better
    rec2 = {"trials": trials, "traces": np.array([[1, 1, 1], [1, 1, 1], [1.5, 6, 1], [1, 1, 1.0]])}  # its second, by more
    extract.combine("best_plane", rois, [rec1, rec2])
    assert [p["best"] for p in rec1["planes"]] == [True, False, True] and [p["best"] for p in rec2["planes"]] == [False, True, True]
    assert rec1["roi_traces"][2, 0] == 3 and rec2["roi_traces"][2, 0] == 6
    extract.combine("best_plane_fly", rois, [rec1, rec2])
    assert [p["weight"] for p in rec1["planes"]] == [p["weight"] for p in rec2["planes"]] == [0, 1, 1]
    assert rec1["roi_traces"][2, 0] == 2 and rec2["roi_traces"][2, 0] == 6 and rec1["names"] == ["a", "b"]
    assert [p["best"] for p in rec1["planes"]] == [True, False, True]  # (what al_analysis would use there, still said)
    # planes alike: the first; a plane with no response (no time point before the odor): not the best, unless all are
    assert extract.best_planes(["a", "a", "b", "b", "c"], np.array([1.0, 1.0, np.nan, 2.0, np.nan])) == [0, 3, 4]
    assert extract.panel("megamat_2") == "megamat" and extract.panel("peb_ALr_diag_1_002") == extract.DIAGNOSTIC
    assert extract.panel("my_odors_3") == "my_odors" and extract.panel("kiwi_ea_eb_only") == "kiwi" and extract.panel("7") == "7"


# ----------------------------------------------------------- asked for, and run

def test_api_and_worker(main, tensor, monkeypatch):  # noqa: F811
    monkeypatch.setattr(extract, "CHUNK_BYTES", 10 * Z * Y * X * 2)  # (movies read in several pieces)
    fly = tensor / FLY
    body = {"name": "Bob", "rois": "own", "dir": FLY}
    # who asks, of a fly with recordings (a movie and its trials) and ROIs
    assert call(main.extractions_create, method="POST", body={**body, "name": ""})[0] == 422
    status, r = call(main.extractions_create, method="POST", body=body)
    assert status == 400 and "no recording" in r["error"]  # (its diagnostic recording has no movie)
    for bad, why in [({"rois": "all"}, "rois:"), ({"mode": "median"}, "mode:"), ({"dir": "ann/analysis/2030-01-01/1"}, "motion-corrected data folder"),
                     ({"rois": "predicted", "prediction": 99}, "prediction"), ({"dir": ""}, "none given")]:
        status, r = call(main.extractions_create, method="POST", body={**body, **bad})
        assert status == 400 and why in r["error"], r
    recording(fly / "megamat1")
    recording(fly / "megamat2", seed=1, gain=0.5)
    recording(fly / "megamat2_redo", seed=2)  # (in megamat2's place)
    recording(fly / "kiwi_1", seed=3)
    (fly / "kiwi_1" / "RoiSet.zip").unlink()  # (no ROIs of its own, nor the fly's: left out)
    (fly / "suite2p").mkdir()
    status, e = call(main.extractions_create, method="POST", body=body)
    assert status == 201 and (e["status"], e["fly"], e["name"], e["ip"], e["rois"], e["options"]) == (
        "queued", "ann_2024-01-01_1", "Bob", "10.1.2.3", {"kind": "own"}, {"mode": "best_plane"})
    size = (fly / "megamat1/mocorr.tif").stat().st_size
    assert e["bytes"] == 3 * size and e["fly_dir"] == str(fly)

    # waiting: when it should end is a guess, said to be one; and what a restart would stop counts it
    status, r = call(main.extractions_list)
    assert [x["id"] for x in r["extractions"]] == [e["id"]] and set(r["modes"]) == set(extract.MODES) and r["rate"]["from"] == 0
    eta = r["extractions"][0]["eta"]
    assert eta["from"] == 0 and np.isclose(eta["seconds"], extract.GUESS_S_PER_GB * 3 * size / 2**30) and r["extractions"][0]["tensor_dir"] == FLY
    assert predictions.worker_jobs(main.db) == {"jobs": [], "queued": 1}

    claimed = extract.claim(main.db)
    assert claimed["status"] == "running" and extract.claim(main.db) is None
    assert call(main.extractions_delete, method="DELETE", path_params={"id": e["id"]})[0] == 409
    assert predictions.worker_jobs(main.db)["jobs"][0]["what"] == "The signals of ann_2024-01-01_1's ROIs being extracted, asked for by Bob"
    fit_worker.run_extraction(main.db, claimed)
    e = extract.get(main.db, e["id"])
    assert e["status"] == "done" and e["error"] is None and e["progress"] is None, (extract.out_dir(e["id"]) / "log.txt").read_text()
    res = e["result"]
    assert [(x["name"], x["panel"], x["trials"], x["frames"], x["planes"]) for x in res["recordings"]] == [
        ("megamat1", "megamat", 4, T, Z), ("megamat2_redo", "megamat", 4, T, Z)]
    assert (res["rois"], res["roi_planes"], res["glomeruli"], res["mode"], res["bytes"]) == (4, 6, 3, "best_plane", 3 * size) and res["seconds"] > 0
    log = (extract.out_dir(e["id"]) / "log.txt").read_text()
    assert "kiwi_1: left out" in log and "DC2+" in log and "megamat2_redo (megamat): 40 time points x 3 planes, 4 trials, 6 ROI-planes" in log

    # its folder: each recording's traces, dF/F and responses, and every recording's responses as an upload takes them
    d = extract.out_dir(e["id"])
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    per = ["dff.csv", "plane_traces.csv", "planes.csv", "responses.csv", "traces.csv"]
    assert files == ["README.txt", "log.txt", "provenance.json", *(f"recordings/{r}/{f}" for r in ("megamat1", "megamat2_redo") for f in per),
                     "responses.csv", "responses_glomeruli.csv"]
    movie = tifffile.imread(fly / "megamat1/mocorr.tif")
    rois = extract.read_rois(fly / "megamat1/RoiSet.zip", (Z, Y, X), lambda s: None)
    trials = extract.read_trials(fly / "megamat1" / movies.TRIALS)
    want_traces, want_stats, want_best = reference(movie, rois, trials)
    traces = pd.read_csv(d / "recordings/megamat1/traces.csv", index_col=0)
    kept = [rois[i]["name"] for i in want_best]  # (each ROI once, in the order of the planes kept)
    assert list(traces.columns) == kept and sorted(kept) == ["7", "DL5", "DM2", "VA1d"] and traces.index.name == "frame"
    assert np.array_equal(traces.to_numpy(), want_traces[:, want_best])  # (written so as to be read back the same)
    planes = pd.read_csv(d / "recordings/megamat1/plane_traces.csv", index_col=0, header=[0, 1])
    assert list(planes.columns) == [(r["name"], str(r["z"])) for r in rois] and np.array_equal(planes.to_numpy(), want_traces)
    own = pd.read_csv(d / "recordings/megamat1/responses.csv", index_col=[0, 1, 2, 3], header=[0, 1, 2, 3])
    assert own.index.names == ["panel", "is_pair", "odor1", "repeat"] and own.columns.names == ["date", "fly_num", "thorimage_id", "roi"]
    assert list(own.index) == [("megamat", False, "va @ -3", 0), ("megamat", False, "ea @ -2", 0), ("megamat", False, "va @ -3", 1),
                               ("megamat", False, "ea @ -2", 1)]
    assert list(own.columns)[0] == ("2024-01-01", "1", "megamat1", "DM2") and np.array_equal(own.to_numpy(), want_stats[:, want_best])
    dff = pd.read_csv(d / "recordings/megamat1/dff.csv")
    assert list(dff.columns) == ["frame", "trial", "from_onset", "odor", "repeat", *kept] and len(dff) == T
    assert list(dff.iloc[24][["frame", "trial", "from_onset", "odor", "repeat"]]) == [24, 2, 0, "va @ -3", 1]
    pl = pd.read_csv(d / "recordings/megamat1/planes.csv")
    assert list(pl["roi"]) == [r["name"] for r in rois] and list(pl["best"]) == [i in want_best for i in range(len(rois))]
    # the fly's: both recordings' trials, repeats counted through them; those named as glomeruli read as an upload's
    table = pd.read_csv(d / "responses.csv", index_col=[0, 1, 2, 3], header=[0, 1, 2])
    assert list(table.index.get_level_values("repeat")) == [0, 0, 1, 1, 2, 2, 3, 3] and table.shape == (8, 4)
    assert np.array_equal(table.to_numpy()[:4], want_stats[:, want_best])
    gloms = pd.read_csv(d / "responses_glomeruli.csv", index_col=[0, 1, 2, 3], header=[0, 1, 2])
    assert [c[2] for c in gloms.columns] == [n for n in kept if n != "7"]
    parsed = load_file(d / "responses_glomeruli.csv")
    assert {(u["date"], u["fly_num"], u["roi"]) for u in parsed.units} == {("2024-01-01", 1, n) for n in ("DM2", "DL5", "VA1d")}
    assert len(parsed.trials) == 8 and {t["panel"] for t in parsed.trials} == {"megamat"}
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["rois"]["kind"] == "own" and len(prov["rois"]["files"]) == 2 and len(prov["rois"]["files"][0]["sha256"]) == 64
    assert prov["recordings"][0]["movie"]["shape"] == {"t": T, "z": Z, "c": 1, "y": Y, "x": X} and prov["computed"]["n_volumes_for_response"] == 2
    assert prov["asked_for"]["by"] == "Bob" and prov["fly"]["date"] == "2024-01-01" and "al_analysis" in (d / "README.txt").read_text()
    # downloaded whole; its log
    z = zipfile.ZipFile(io.BytesIO(b"".join(extract.stream_zip(e["id"], "top"))))
    assert sorted(z.namelist()) == [f"top/{f}" for f in files]
    assert call(main.extractions_download, path_params={"id": 99})[0] == 404

    # one done: what the next one's time is estimated from
    again = call(main.extractions_create, method="POST", body={**body, "mode": "weighted"})[1]
    r = call(main.extractions_list)[1]
    eta = next(x for x in r["extractions"] if x["id"] == again["id"])["eta"]
    assert r["rate"]["from"] == 1 and eta["from"] == 1 and np.isclose(eta["seconds"], res["seconds"])
    running = {**again, "status": "running", "started_at": (dt.datetime.now().astimezone() - dt.timedelta(seconds=10)).isoformat(),
               "progress": {"done": 1, "total": 5}}
    extract.estimate([running])
    assert 39 < running["eta"]["seconds"] < 41  # (from how far it has got)
    fit_worker.run_extraction(main.db, extract.claim(main.db))
    w = pd.read_csv(extract.out_dir(again["id"]) / "recordings/megamat1/traces.csv", index_col=0)
    assert 0.9 < w["DM2"].mean() < 1.1  # (relative F)
    assert "relative" in (extract.out_dir(again["id"]) / "README.txt").read_text()

    # with a prediction's ROIs: as predicted, or as edited in the viewer (once they are)
    f = done_fit(main.db)
    p, _ = predictions.create(main.db, f["id"], predictions.locate(FLY, RAW, config.TENSOR_ROOT, []), "Ann", None)
    predictions.pred_dir(p["id"]).mkdir(parents=True)
    roifile.ImagejRoi.frompoints(np.array([[2, 2], [8, 2], [8, 7], [2, 7]], "f4"), name="DA1", z=1).tofile(predictions.roiset(p))
    assert "done" in call(main.extractions_create, method="POST", body={"name": "Cy", "rois": "predicted", "prediction": p["id"]})[1]["error"]
    predictions.update(main.db, p["id"], status="done")
    assert "not edited" in call(main.extractions_create, method="POST", body={"name": "Cy", "rois": "edited", "prediction": p["id"]})[1]["error"]
    predictions.save_edited(main.db, p["id"], [{"name": "DA1", "z": 1, "points": [[3, 3], [8, 3], [8, 7]]}], 0, "Cy", None)
    count = lambda pts: int(full_mask(pts, (Y, X)).sum())  # noqa: E731
    for kind, pixels in (("predicted", count([[2, 2], [8, 2], [8, 7], [2, 7]])), ("edited", count([[3, 3], [8, 3], [8, 7]]))):
        x = call(main.extractions_create, method="POST", body={"name": "Cy", "rois": kind, "prediction": p["id"]})[1]
        assert x["rois"] == {"kind": kind, "prediction": p["id"], "fit": f["id"]} and x["fly"] == p["fly"]
        fit_worker.run_extraction(main.db, extract.claim(main.db))
        x = extract.get(main.db, x["id"])
        assert x["status"] == "done" and x["result"]["rois"] == 1 and len(x["result"]["recordings"]) == 3  # (kiwi_1 too: the ROIs are the fly's)
        assert list(pd.read_csv(extract.out_dir(x["id"]) / "recordings/kiwi_1/planes.csv")["pixels"]) == [pixels]
        assert json.loads((extract.out_dir(x["id"]) / "provenance.json").read_text())["rois"]["files"][0]["path"].endswith(
            "_pred-edited_RoiSet.zip" if kind == "edited" else "ann_2024-01-01_1_RoiSet.zip")

    # what cannot be extracted is said, in its record and its log; deleted with its files
    (fly / "megamat1" / movies.TRIALS).write_text(json.dumps([{"start_frame": 0, "first_odor_frame": 4, "end_frame": 99, "odors": "va @ -3"}]))
    bad = call(main.extractions_create, method="POST", body=body)[1]
    fit_worker.run_extraction(main.db, extract.claim(main.db))
    bad = extract.get(main.db, bad["id"])
    assert bad["status"] == "failed" and "beyond its movie's 40 time points" in bad["error"]
    assert "beyond its movie" in call_text(main.extractions_log, bad["id"])
    assert call(main.extractions_delete, method="DELETE", path_params={"id": bad["id"]}) == (200, {"deleted": True})
    assert not extract.out_dir(bad["id"]).exists() and call(main.extractions_delete, method="DELETE", path_params={"id": bad["id"]})[0] == 404
    # left running by a worker that stopped: interrupted
    last = call(main.extractions_create, method="POST", body=body)[1]
    extract.claim(main.db)
    assert extract.interrupt_running(main.db) == [last["id"]] and extract.get(main.db, last["id"])["status"] == "interrupted"


def test_loop(main, tensor, monkeypatch):  # noqa: F811
    """The worker's thread for extractions: each one asked for run in turn, whatever else the worker does."""
    import threading

    from tests.test_predictions import wait

    monkeypatch.setattr(fit_worker, "POLL_S", 0.01)
    fly = tensor / FLY
    recording(fly / "megamat1")
    state, stop = {"extraction": None}, threading.Event()
    th = threading.Thread(target=fit_worker.extract_loop, args=(main.db, state, stop), daemon=True)
    th.start()
    try:
        ids = [call(main.extractions_create, method="POST", body={"name": "Bob", "rois": "own", "dir": FLY, "mode": m})[1]["id"]
               for m in ("best_plane", "all_planes")]
        assert wait(lambda: all(extract.get(main.db, i)["status"] == "done" for i in ids), 30)
        assert wait(lambda: state["extraction"] is None)
        assert (extract.out_dir(ids[1]) / "responses.csv").is_file()
    finally:
        stop.set()
        th.join(5)
    assert not th.is_alive()


def call_text(endpoint, ext_id) -> str:
    import asyncio

    return asyncio.run(endpoint(request(path_params={"id": ext_id}))).body.decode()
