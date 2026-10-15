"""B7 / B8: the ceilings. Oracle returns the true W0 for free; FullReadoutRecal pays m*n reads.

WHY: B8 bounds accuracy from above (no method can beat it); B7 defines where "cheap method"
ends - re-reading every cell at honest read noise. The oracle is the only baseline allowed to
see the truth, via the bound W0 provider.
"""

from __future__ import annotations

from neurofault.spectral.baselines.base import Bindable, Correction, ReadCost, TileReaderLike


class OracleExact(Bindable):
    name = "B8_oracle"

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        return ReadCost()

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        return Correction(self.ref().clone(), ReadCost(), {})


class FullReadoutRecal(Bindable):
    """rewrite=False: W_eff = the noisy read matrix (plan spec; reading alone does NOT undo
    drift, so it only matches B0 plus read noise). rewrite=True: read every cell, then
    reprogram the whole tile back to the stored digital W0 (m*n writes counted, ideal
    programming) - the physical meaning of a full recalibration."""

    def __init__(self, rewrite: bool = False):
        self.rewrite = rewrite
        self.name = "B7_full_readout_rewrite" if rewrite else "B7_full_readout"

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        return ReadCost()

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        start = reader.cost.copy()
        w = reader.full_readout()
        cost = reader.cost - start
        if self.rewrite:
            cost.writes += w.numel()
            w = self.ref().clone()
        return Correction(w, cost, {})


class NoCompensation(Bindable):
    """B0: W_eff = W(t), zero cost."""

    name = "B0_none"

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        return ReadCost()

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        return Correction(self.now().clone(), ReadCost(), {})
