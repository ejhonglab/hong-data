"""Matrices most of whose cells are 0 (as MetaCyc's compounds by the pathways they are in),
kept as their other cells only: as a reference dataset's view is made (`reference.parse`),
imported (``values.npz``) and sent to the browser (``values_sparse``)."""
from __future__ import annotations

import base64
from typing import NamedTuple

import numpy as np


class Sparse(NamedTuple):
    """A matrix as its cells other than 0: its ``shape`` (rows, columns), and of those
    cells, their ``index`` (flat, row-major, ascending: int64) and ``values`` (float32, NaN
    where missing)."""

    shape: tuple[int, int]
    index: np.ndarray
    values: np.ndarray

    @classmethod
    def of(cls, shape: tuple[int, int], rows, cols, values) -> "Sparse":
        """Of its cells other than 0, given as their rows, columns and values (in any order)."""
        index = np.asarray(rows, dtype=np.int64) * shape[1] + np.asarray(cols, dtype=np.int64)
        order = np.argsort(index, kind="stable")
        return cls((int(shape[0]), int(shape[1])), index[order], np.asarray(values, dtype=np.float32)[order])

    def toarray(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=np.float32)
        out.flat[self.index] = self.values
        return out

    def range(self) -> tuple[float, float] | None:
        """Its smallest and largest values but missing ones (0 among them, if any cell is 0);
        None if all are missing."""
        finite = self.values[np.isfinite(self.values)]
        if len(self.index) < self.shape[0] * self.shape[1]:
            finite = np.append(finite, np.float32(0))
        return (float(finite.min()), float(finite.max())) if len(finite) else None

    def payload(self) -> dict:
        """As the browser reads it (app.js's `matrixValues`): its cells' indices (uint32) and
        values (float32), little-endian, in base64."""
        if self.shape[0] * self.shape[1] >= 2**32:
            raise ValueError(f"a matrix of {self.shape[0]} × {self.shape[1]} cells is too large to send")
        return {"index_b64": base64.b64encode(self.index.astype("<u4").tobytes()).decode("ascii"),
                "values_b64": base64.b64encode(self.values.astype("<f4").tobytes()).decode("ascii")}
