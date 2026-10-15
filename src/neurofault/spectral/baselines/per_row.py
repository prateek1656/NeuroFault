"""B2 / per-output gain: one multiplicative factor per W row (output), from forward probes.

WHY: patterned drift is multiplicative, W1 ~= W0 o (1 + F). A row-gradient F is exactly a
per-row scale, so this is the cheap incumbent that structured-drift methods must beat.

Mapping to AIHWKit: its ``PerColumnDriftCompensation`` reads eye(n) -> (n, m) and averages
dim=0, giving a factor per OUTPUT = per row of our (m, n) W. ``PerRow(probes='identity',
estimator='meanabs')`` reproduces it; other probes use the per-row least-squares estimator
(``ls``), which is exact for a pure row scale when noise-free and needs only P << n reads.
"""

from __future__ import annotations

import torch

from neurofault.spectral.baselines.base import (
    Bindable,
    Correction,
    ReadCost,
    TileReaderLike,
    apply_gains,
    ls_gain,
    make_probes,
)
from neurofault.spectral.types import make_generator


class PerRow(Bindable):
    def __init__(
        self,
        probes: str = "rademacher",
        p: int = 4,
        estimator: str = "ls",
        eps: float = 1e-4,
        seed: int = 0,
    ):
        if estimator not in ("ls", "meanabs"):
            raise ValueError("estimator must be 'ls' or 'meanabs'")
        self.probes, self.p, self.estimator, self.eps = probes, p, estimator, eps
        self.gen = make_generator(seed, f"per_row_{probes}")
        self.name = f"B2_per_row_{probes}{p if probes != 'identity' else ''}_{estimator}"
        self.x: torch.Tensor | None = None
        self.y0: torch.Tensor | None = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        self.x = make_probes(self.probes, reader.shape[1], self.p, self.gen)
        self.y0 = reader.mvm(self.x)
        return reader.cost - start

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        assert self.x is not None, "calibrate() first"
        start = reader.cost.copy()
        yt = reader.mvm(self.x)
        if self.estimator == "meanabs":
            g = self.y0.abs().mean(1).clamp(min=self.eps) / yt.abs().mean(1).clamp(min=self.eps)
        else:
            g = ls_gain(self.y0, yt, dim=1)
        return Correction(apply_gains(self.now(), row=g), reader.cost - start, {"row_gain": g})


def aihwkit_per_column(eps: float = 1e-4) -> PerRow:
    """AIHWKit PerColumnDriftCompensation (identity readout, mean|.| per output)."""
    return PerRow(probes="identity", estimator="meanabs", eps=eps)
