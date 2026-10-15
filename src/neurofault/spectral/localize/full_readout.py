"""A1 full readout: n unit-vector forward reads (or m backward) recover E = W_now - W_ref.

Cost ceiling of the localization study ("beat this or don't bother"). Cells with |E_hat| above a calibrated
fixed-false-alarm threshold z*sigma are declared faulty.
"""

from __future__ import annotations

import torch

from neurofault.spectral.localize.common import RefLocalizer, cells_to_estimate, z_two_sided
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike


class FullReadoutLocalizer(RefLocalizer):
    name = "A1_full_readout"

    def __init__(self, w_ref, *, direction: str = "fwd", **kw) -> None:
        super().__init__(w_ref, **kw)
        self.direction = direction  # fwd | auto (the shorter side, needs backward support)

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        sigma = self._need_sigma()
        before = reader.cost.copy()
        m, n = self.shape
        use_bwd = self.direction == "auto" and reader.backward_supported and m < n
        if use_bwd:
            E = self.res_bwd(reader, torch.eye(m, dtype=DTYPE)).T
        else:
            E = self.res_fwd(reader, torch.eye(n, dtype=DTYPE))
        mask = E.abs() > z_two_sided(m * n, self.fa) * sigma
        return cells_to_estimate(mask, E.abs(), bwd=use_bwd), reader.cost - before
