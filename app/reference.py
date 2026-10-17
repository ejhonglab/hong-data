"""Fixed reference matrices (chemicals x fruits, chemicals x pathways, ...).

``reference_data/manifest.json`` lists them::

    {"datasets": [{"id": "...", "title": "...", "description": "...", "file": "x.csv",
                   "kind": "sequential" | "diverging" | "binary",
                   "value_label": "...", "row_label": "...", "col_label": "...",
                   "source": "..."}]}

Each CSV has row labels in its first column, optionally a ``group`` column (row
grouping), and one column per matrix column. Blank cells are missing values.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import config


def manifest() -> list[dict]:
    p = config.REFERENCE_DIR / "manifest.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    out = []
    for d in data.get("datasets", []):
        if not d.get("id") or not d.get("file"):
            continue
        entry = dict(d)
        entry["available"] = (config.REFERENCE_DIR / d["file"]).exists()
        out.append(entry)
    return out


def load(ref_id: str) -> Optional[dict[str, Any]]:
    entry = next((d for d in manifest() if d["id"] == ref_id), None)
    if entry is None or not entry["available"]:
        return None
    path: Path = config.REFERENCE_DIR / entry["file"]
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0)
    groups = None
    if "group" in df.columns:
        groups = df["group"].astype(str).tolist()
        df = df.drop(columns=["group"])
    df = df.apply(pd.to_numeric, errors="coerce")
    values = np.ascontiguousarray(df.to_numpy(dtype=np.float32))
    rows = [{"label": str(r), "group": (groups[i] if groups else None)} for i, r in enumerate(df.index)]
    cols = [{"label": str(c)} for c in df.columns]
    finite = np.isfinite(values)
    return {
        "meta": entry,
        "rows": rows,
        "cols": cols,
        "shape": list(values.shape),
        "dtype": "float32",
        "values_b64": base64.b64encode(values.astype("<f4").tobytes()).decode("ascii"),
        "vmin": float(values[finite].min()) if finite.any() else None,
        "vmax": float(values[finite].max()) if finite.any() else None,
    }
