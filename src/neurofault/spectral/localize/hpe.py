"""A2 HPE-style baseline: syndrome locates the faulty output line, bisection finds the cell.

Reimplemented as a baseline from the published abstracts/claims of US 11,804,859 (analog ECC) and
US 11,532,356 / 10,984,860 (self-healing DPE), not from their full text:
- stage 1 (US 11,804,859): a dense probe, repeated ``n_dense`` times; a 2-row syndrome
  (s1 = sum r_i, s2 = sum alpha_i r_i, alpha_i = (i+1)/m) with two outside-threshold values
  locates ONE erroneous output line (alpha = s2/s1) and its value. ``decoder="argmax"`` replaces
  the ratio by the maximum-likelihood argmax over per-output residuals: strictly stronger than
  the literal syndrome decode when all outputs are observable, so it is the conservative
  baseline for the NO-GO test (default). The syndrome here is computed digitally from the
  measured residual (the reader exposes no physical parity columns), so its noise is amplified
  by sqrt(m) relative to a real parity column; ``meta['extra_cols']`` records the parity-column
  area (2 for syndrome, 0 for argmax).
- stage 2 (US 11,532,356): half-set test vectors + binary search over the input index, one
  fault at a time (the worked example assumes <= 1 erroneous entry). Each level compares the
  mean residual of the left half-set probe with half the stage-1 value.
Forward direction only; single fault. Multiple faults are out of its scope (meta flags it).
"""

from __future__ import annotations

import math

import torch

from neurofault.spectral.localize.common import RefLocalizer, empty_estimate, z_two_sided
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike


class HPELocalizer(RefLocalizer):
    name = "A2_hpe"

    def __init__(
        self, w_ref, *, n_dense: int = 4, n_bisect: int = 1, decoder: str = "argmax", **kw
    ) -> None:
        super().__init__(w_ref, **kw)
        assert decoder in ("argmax", "syndrome")
        self.n_dense, self.n_bisect, self.decoder = n_dense, n_bisect, decoder

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        sigma = self._need_sigma()
        before = reader.cost.copy()
        m, n = self.shape
        ones = torch.ones(n, self.n_dense, dtype=DTYPE)
        rbar = self.res_fwd(reader, ones).mean(1)
        sd = sigma * math.sqrt(n / self.n_dense)  # noise sd of each averaged output
        if self.decoder == "argmax":
            i = int(rbar.abs().argmax())
            detected = bool(rbar[i].abs() > z_two_sided(m, self.fa) * sd)
            d = float(rbar[i])
        else:
            alpha = (torch.arange(m, dtype=DTYPE) + 1) / m
            s1, s2 = float(rbar.sum()), float((alpha * rbar).sum())
            detected = abs(s1) > z_two_sided(1, self.fa) * sd * math.sqrt(m)
            i = min(max(round(s2 / s1 * m) - 1, 0), m - 1) if detected and s1 != 0 else 0
            d = s1
        meta = {"assumes_single": True, "extra_cols": 2 if self.decoder == "syndrome" else 0}
        if not detected:
            return empty_estimate(**meta), reader.cost - before
        lo, hi = 0, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            x = torch.zeros(n, self.n_bisect, dtype=DTYPE)
            x[lo:mid] = 1.0
            r = self.res_fwd(reader, x)[i].mean()
            if abs(float(r)) > abs(d) / 2:
                hi = mid
            else:
                lo = mid
        cells = torch.tensor([[i, lo]], dtype=torch.long)
        est = FaultEstimate(cells=cells, rows=cells[:, 0], cols=cells[:, 1], meta=meta)
        return est, reader.cost - before
