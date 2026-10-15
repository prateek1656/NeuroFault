"""Bi-scaling: W(t) ~= diag(a) W0 diag(b), fit by alternating least squares (ALS).

WHY: multiplicative patterned drift W1 = W0 o (1 + F) with a separable F is a rank-1
multiplicative field; per-row and per-column gains each capture only half of it. Identifiability:
b multiplies INSIDE W0 X, so dense probes cannot separate it. One-hot probes read single W0
columns (forward) or rows (backward), giving observed ratios R_ij = a_i b_j on those entries:
    fwd one-hot columns J  -> entries (all i, j in J)
    bwd one-hot rows    I  -> entries (i in I, all j)
Model per observed entry: Yt_ij = a_i b_j Y0_ij. ALS alternately solves the closed-form
weighted LS for a (given b) and b (given a). Without backward reads b is only seen on J; the
remaining b_j are set to 1 (flagged in meta['b_unobserved']). Scale gauge (a*c, b/c) leaves W_eff
unchanged. Correction multipliers returned are 1/a and 1/b.
"""

from __future__ import annotations

import torch

from neurofault.spectral.baselines.base import (
    Bindable,
    Correction,
    ReadCost,
    TileReaderLike,
    apply_gains,
    make_probes,
)
from neurofault.spectral.types import DTYPE, make_generator


def als_rank1(
    z: torch.Tensor, mref: torch.Tensor, mask: torch.Tensor, iters: int = 500
) -> tuple[torch.Tensor, torch.Tensor]:
    """argmin_{a,b} sum_{mask} (z - a_i b_j mref)^2 by alternating closed-form updates."""
    m, n = z.shape
    a = torch.ones(m, dtype=DTYPE)
    b = torch.ones(n, dtype=DTYPE)
    zm = z * mask
    mm = mref * mask
    for _ in range(iters):
        prev = torch.outer(a, b)
        mb = mm * b[None, :]
        a = (zm * mb).sum(1) / (mb * mb).sum(1).clamp(min=1e-300)
        a = torch.where((mb * mb).sum(1) > 0, a, torch.ones_like(a))
        ma = mm * a[:, None]
        b = (zm * ma).sum(0) / (ma * ma).sum(0).clamp(min=1e-300)
        b = torch.where((ma * ma).sum(0) > 0, b, torch.ones_like(b))
        if float((torch.outer(a, b) - prev).abs().max()) < 1e-13:
            break
    return a, b


class BiScale(Bindable):
    def __init__(self, n_fwd: int = 4, n_bwd: int = 4, iters: int = 500, seed: int = 0):
        self.n_fwd, self.n_bwd, self.iters = n_fwd, n_bwd, iters
        self.gen = make_generator(seed, "biscale")
        self.name = f"B2b_biscale_f{n_fwd}b{n_bwd}"
        self.xf: torch.Tensor | None = None
        self.zb: torch.Tensor | None = None
        self.y0f: torch.Tensor | None = None
        self.y0b: torch.Tensor | None = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        m, n = reader.shape
        self.xf = make_probes("onehot", n, self.n_fwd, self.gen)
        self.y0f = reader.mvm(self.xf)  # (m, Pf) = W0[:, J]
        self.zb = None
        if reader.backward_supported and self.n_bwd > 0:
            self.zb = make_probes("onehot", m, self.n_bwd, self.gen)
            self.y0b = reader.mvm_t(self.zb)  # (n, Pb) = W0[I, :]^T
        return reader.cost - start

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        assert self.xf is not None, "calibrate() first"
        start = reader.cost.copy()
        m, n = reader.shape
        z = torch.zeros(m, n, dtype=DTYPE)
        mref = torch.zeros(m, n, dtype=DTYPE)
        mask = torch.zeros(m, n, dtype=DTYPE)
        cols = self.xf.argmax(0)
        z[:, cols] = reader.mvm(self.xf)
        mref[:, cols] = self.y0f
        mask[:, cols] = 1.0
        if self.zb is not None:
            rows = self.zb.argmax(0)
            z[rows, :] = reader.mvm_t(self.zb).T
            mref[rows, :] = self.y0b.T
            mask[rows, :] = 1.0
        a, b = als_rank1(z, mref, mask, self.iters)
        unobs = int((mask.sum(0) == 0).sum())
        row, col = 1.0 / a, 1.0 / b
        meta = {"row_gain": row, "col_gain": col, "b_unobserved": unobs}
        return Correction(apply_gains(self.now(), row=row, col=col), reader.cost - start, meta)
