"""The set of antennal-lobe glomerulus names accepted as ROI labels.

The default list follows the hemibrain / Bates et al. (2020) nomenclature plus a few
older synonyms that still appear in analysis outputs. Put one name per line in
``config/glomeruli.txt`` (next to this package's parent directory, or wherever
``OLFDB_GLOMERULI_FILE`` points) to replace the list without editing code.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_GLOMERULI = [
    "D",
    "DA1", "DA2", "DA3", "DA4l", "DA4m",
    "DC1", "DC2", "DC3", "DC4",
    "DL1", "DL2d", "DL2v", "DL3", "DL4", "DL5",
    "DM1", "DM2", "DM3", "DM4", "DM5", "DM6",
    "DP1l", "DP1m",
    "V",
    "VA1d", "VA1v", "VA1lm", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7l", "VA7m",
    "VC1", "VC2", "VC3", "VC3l", "VC3m", "VC4", "VC5",
    "VL1", "VL2a", "VL2p",
    "VM1", "VM2", "VM3", "VM4", "VM5d", "VM5v", "VM6", "VM6l", "VM6m", "VM6v", "VM7d", "VM7v",
    "VP1d", "VP1l", "VP1m", "VP2", "VP3", "VP4", "VP5",
]


def config_path() -> Path:
    env = os.environ.get("OLFDB_GLOMERULI_FILE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "config" / "glomeruli.txt"


def load_glomeruli() -> list[str]:
    p = config_path()
    if p.exists():
        names = []
        for line in p.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
        if names:
            return names
    return list(DEFAULT_GLOMERULI)


GLOMERULUS_NAMES: tuple[str, ...] = tuple(load_glomeruli())  # in the order of the list
GLOMERULI: frozenset[str] = frozenset(GLOMERULUS_NAMES)
_FOLDED = {g.casefold(): g for g in GLOMERULUS_NAMES}


def is_glomerulus(roi: str) -> bool:
    return roi in GLOMERULI


def same_but_case(roi: str) -> Optional[str]:
    """The accepted glomerulus name `roi` is but for upper and lower case, if any."""
    return _FOLDED.get(roi.casefold())


def is_int_roi(roi: str) -> bool:
    """Non-negative integer ROI id, written without sign or decimal point."""
    return isinstance(roi, str) and roi.isdigit()


def roi_sort_key(roi: str):
    """Integers numerically, then glomeruli/other names alphabetically (case-insensitive)."""
    if is_int_roi(roi):
        return (0, int(roi), "")
    return (1, 0, roi.lower())
