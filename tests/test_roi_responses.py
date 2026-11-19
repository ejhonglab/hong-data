"""An ROI's mean response to each odor, for the movie player's plot (roi_responses.py)."""
from __future__ import annotations

import numpy as np
import pytest
import roifile
from starlette.exceptions import HTTPException

from app import extract, movies, roi_responses
from tests.test_extract import recording
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)

DIAG, PANEL = "glomeruli_diagnostics1", "megamat1"


def fly(top, concat=True, dtype="<u2"):
    """A fly's folder of two recordings (a diagnostic one, then a megamat one), registered together if `concat`."""
    d = top / "ann/analysis_intermediates/2024-01-01/1"
    recording(d / DIAG, seed=0, dtype=dtype)
    recording(d / PANEL, seed=1, gain=0.5, dtype=dtype)
    if concat:
        (d / movies.CONCAT).write_bytes(b"")  # (only that there is one matters)
    return d


def expected(d, recs, index, window=2):
    """Each (panel, odor)'s mean response of ROI `index`, from its traces read whole (`extract.plane_traces`)."""
    r = roifile.ImagejRoi.fromfile(d / DIAG / movies.ROISET)[index]
    out = {}
    for name in recs:
        m = movies.open_movie(d / name / "mocorr.tif")
        s = m.shape
        mask = extract.roi_mask(r, (s["z"], s["y"], s["x"]))
        trials = extract.read_trials(d / name / movies.TRIALS)
        for t, v in zip(trials, extract.responses(extract.plane_traces(m, [mask]), trials, window)[:, 0]):
            out.setdefault((extract.panel(name), " + ".join(t["odors"])), []).append(v)
    return {k: np.mean(v) for k, v in out.items()}


@pytest.mark.parametrize("dtype", ["<u2", ">f4"])
def test_compute(tmp_path, dtype):
    d = fly(tmp_path, dtype=dtype)
    got = roi_responses.compute(d / movies.CONCAT, None, d / DIAG / movies.ROISET, 1, "DM2", 2, [], lambda p: p)
    # every recording of the fly, its diagnostic odors first (in hong2p's order of them: va before ea), then the panel's
    # (megamat's order has va; ea, not in it, after it); each odor's name with its concentration, as both panels have it
    keys = [(c["panel"], c["odor"]) for c in got["columns"]]
    assert keys == [("glomeruli_diagnostics", "va @ -3"), ("glomeruli_diagnostics", "ea @ -2"), ("megamat", "va @ -3"), ("megamat", "ea @ -2")]
    assert [c["label"] for c in got["columns"]] == ["va @ -3", "ea @ -2"] * 2
    assert [c["diagnostic"] for c in got["columns"]] == [True, True, False, False] and {c["trials"] for c in got["columns"]} == {2}
    # the same numbers as from the whole movies, to the last bit, from only the time points needed
    want = expected(d, [DIAG, PANEL], 1)
    assert got["values"] == [want[k] for k in keys] and got["values"][0] > 0.1
    assert [(r["name"], r["window"], r["of"]) for r in got["recordings"]] == [(DIAG, 2, 40), (PANEL, 2, 40)]
    assert got["recordings"][0]["read"] == 6 + 6 + 6 + 7 and got["roi"] == "DM2" and not got["warnings"]
    # a recording's own movie: its fly's recordings all the same, those being registered together
    again = roi_responses.compute(d / DIAG / "mocorr.tif", d / DIAG / movies.TRIALS, d / DIAG / movies.ROISET, 1, "DM2", 2, [], lambda p: p)
    assert again["values"] == got["values"]
    # a longer window, as asked for
    longer = roi_responses.compute(d / movies.CONCAT, None, d / DIAG / movies.ROISET, 1, "DM2", 3, [], lambda p: p)
    want = expected(d, [DIAG, PANEL], 1, window=3)
    assert longer["values"] == [want[k] for k in keys] and longer["values"] != got["values"]


def test_one_recording(tmp_path):
    # a fly's recordings not registered together: the movie's own alone
    d = fly(tmp_path, concat=False)
    got = roi_responses.compute(d / PANEL / "mocorr.tif", d / PANEL / movies.TRIALS, d / DIAG / movies.ROISET, 3, "VA1d", 2, [], lambda p: p)
    assert [r["name"] for r in got["recordings"]] == [PANEL]
    assert [c["label"] for c in got["columns"]] == ["va", "ea"]
    want = expected(d, [PANEL], 3)
    assert got["values"] == [want[("megamat", "va @ -3")], want[("megamat", "ea @ -2")]]
    # an ROI the file no longer has under that name, or at all
    with pytest.raises(roi_responses.RoiResponsesError, match="is not DL5"):
        roi_responses.compute(d / PANEL / "mocorr.tif", d / PANEL / movies.TRIALS, d / DIAG / movies.ROISET, 3, "DL5", 2, [], lambda p: p)
    with pytest.raises(roi_responses.RoiResponsesError, match="no ROI 21"):
        roi_responses.compute(d / PANEL / "mocorr.tif", d / PANEL / movies.TRIALS, d / DIAG / movies.ROISET, 20, None, 2, [], lambda p: p)


def test_order():
    # a panel hong2p has no order of: its stimulus YAMLs' order, else as presented; an odor's concentrations together, lowest first
    keys = [("x", "b @ -2"), ("x", "c @ -1"), ("x", "a @ -1"), ("glomeruli_diagnostics", "ACV @ -2"), ("x", "a @ -3"),
            ("glomeruli_diagnostics", "ea @ -8"), ("glomeruli_diagnostics", "va @ -4")]
    assert roi_responses.order(keys, ["x", "glomeruli_diagnostics"], ["a @ -1", "b @ -2"]) == [
        ("glomeruli_diagnostics", "va @ -4"), ("glomeruli_diagnostics", "ea @ -8"), ("glomeruli_diagnostics", "ACV @ -2"),
        ("x", "a @ -3"), ("x", "a @ -1"), ("x", "b @ -2"), ("x", "c @ -1")]
    assert roi_responses.labels([("x", "a @ -3"), ("x", "a @ -1"), ("x", "b @ -2")]) == ["a @ -3", "a @ -1", "b"]
    assert roi_responses.needed([{"start": 0, "onset": 3, "end": 9}, {"start": 10, "onset": 12, "end": 13}], 3, 13).tolist() == [
        0, 1, 2, 3, 4, 5, 10, 11, 12]


def test_api(main, root):  # noqa: F811
    d = fly(root)
    body = {"path": str(d / movies.CONCAT), "rois": str(d / DIAG / movies.ROISET), "index": 1, "name": "DM2", "window": 2}
    status, got = call(main.movie_roi_responses, body=body)
    assert status == 200 and got["roi"] == "DM2" and len(got["values"]) == 4
    assert call(main.movie_roi_responses, body={**body, "window": 0})[0] == 400
    assert call(main.movie_roi_responses, body={**body, "name": "DL5"})[0] == 422
    with pytest.raises(HTTPException) as e:
        call(main.movie_roi_responses, body={**body, "rois": "/etc/RoiSet.zip"})
    assert e.value.status_code in (403, 404)
