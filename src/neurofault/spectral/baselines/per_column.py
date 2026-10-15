"""Per-input (W column) gain, from backward probes when the tile supports them.

WHY: the column analogue of PerRow. A column gain scales input j, so it is only observable
through reads that expose columns separately: backward MVMs W^T z (one factor per output of
the transpose) or, forward-only, identity probes (n reads). 'auto' picks backward when
reader.backward_supported else the n-read identity fallback (cost reported honestly).
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


class PerColumn(Bindable):
    def __init__(self, probes: str = "rademacher", p: int = 4, mode: str = "auto", seed: int = 0):
        if mode not in ("auto", "backward", "forward"):
            raise ValueError("mode must be auto|backward|forward")
        self.probes, self.p, self.mode = probes, p, mode
        self.gen = make_generator(seed, f"per_col_{probes}")
        self.name = f"B2c_per_col_{probes}{p}"
        self.z: torch.Tensor | None = None
        self.y0: torch.Tensor | None = None
        self.bwd = True

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        m, n = reader.shape
        self.bwd = self.mode == "backward" or (self.mode == "auto" and reader.backward_supported)
        if self.bwd:
            self.z = make_probes(self.probes, m, self.p, self.gen)
            self.y0 = reader.mvm_t(self.z)
        else:
            self.z = make_probes("identity", n, n)
            self.y0 = reader.mvm(self.z).T  # (n, m): row j = column j of W
        return reader.cost - start

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        assert self.z is not None, "calibrate() first"
        start = reader.cost.copy()
        yt = reader.mvm_t(self.z) if self.bwd else reader.mvm(self.z).T
        g = ls_gain(self.y0, yt, dim=1)
        return Correction(apply_gains(self.now(), col=g), reader.cost - start, {"col_gain": g})
