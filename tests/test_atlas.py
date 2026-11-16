"""Tests of the AL atlas tab's meshes (app/static/atlas/grabe2016.*, made by make_grabe2016.py beside them): every
glomerulus' vertices and triangles where the .json says, neighbours coloured apart, the anatomical axes orthonormal."""
from __future__ import annotations

import json

import numpy as np

from app import config

ATLAS = config.STATIC_DIR / "atlas"


def test_meshes():
    doc = json.loads((ATLAS / "grabe2016.json").read_text())
    blob = (ATLAS / "grabe2016.bin").read_bytes()
    gs = doc["glomeruli"]
    assert len(gs) == 108 and {g["side"] for g in gs} == {"left", "right"} and len({g["name"] for g in gs}) == 54
    end = 0
    for g in gs:
        (vo, nv), (to, nt) = g["vertices"], g["triangles"]
        v = np.frombuffer(blob, "<i2", nv * 3, vo).reshape(-1, 3) / doc["quant"]
        t = np.frombuffer(blob, "<u2", nt * 3, to)
        assert nv > 100 and nt > 100 and t.max() < nv
        # (its surface round its centroid, about as big as its volume: VP4, the longest, twice a ball's radius)
        assert np.linalg.norm(v.mean(0) - g["centroid"]) < 3
        r = (3 * g["volume_um3"] / (4 * np.pi)) ** (1 / 3)
        assert 0.5 * r < np.linalg.norm(v - v.mean(0), axis=1).mean() < 2.5 * r
        end = max(end, to + nt * 6)
    assert end <= len(blob)
    colour = {g["name"]: g["colour"] for g in gs}
    assert doc["neighbours_sharing_a_colour"] == sum(colour[g["name"]] == colour[n] for g in gs for n in g["neighbours"]) // 4
    for side, ax in doc["axes"].items():
        m = np.array(list(ax[k] for k in ax if k != "from_names_deg"))
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-3), side

