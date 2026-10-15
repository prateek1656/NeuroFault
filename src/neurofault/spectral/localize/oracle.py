"""A0 oracle: returns the true fault cells with zero reads (upper bound; current repo behaviour)."""

from __future__ import annotations

import torch

from neurofault.spectral.localize.common import empty_estimate
from neurofault.spectral.types import FaultEstimate, ReadCost, TileReaderLike


class OracleLocalizer:
    name = "A0_oracle"

    def __init__(self, truth_cells: torch.Tensor | None = None) -> None:
        self.truth_cells = truth_cells

    def set_truth(self, cells: torch.Tensor) -> None:
        self.truth_cells = cells

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        return ReadCost()

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        if self.truth_cells is None:
            return empty_estimate(), ReadCost()
        c = self.truth_cells.reshape(-1, 2).long()
        return FaultEstimate(cells=c, rows=c[:, 0].unique(), cols=c[:, 1].unique()), ReadCost()
