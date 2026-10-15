"""A4 two-stage bidirectional localizer .

Stage 1 as A3 (dense fwd+bwd probes -> row set R, column set C). Stage 2 restricts group tests
to the located lines: a disjunct 0/1 design over the |C| columns drives forward probes (decoded
per row of R by COMP/DD), and a disjunct design over the |R| rows drives backward probes (decoded
per column of C). A cell (i, j) in R x C is declared faulty only if BOTH directions agree: two
independent measurements of each fault (the "pairing consistency check"). Without backward
support only the forward decode is used. ``d2`` is the designed max faults per located line.
"""

from __future__ import annotations

import torch

from neurofault.spectral.localize.common import (
    comp_dd,
    empty_estimate,
    kautz_singleton,
    z_two_sided,
)
from neurofault.spectral.localize.residual import ResidualLocalizer
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike


class TwoStageLocalizer(ResidualLocalizer):
    name = "A4_two_stage"

    def __init__(self, w_ref, *, d2: int = 2, decoder: str = "dd", **kw) -> None:
        super().__init__(w_ref, **kw)
        self.d2, self.decoder = d2, decoder

    def _decode(self, reader, lines, support, *, transpose: bool) -> torch.Tensor:
        sigma = self._need_sigma()
        M = kautz_singleton(support.numel(), self.d2)  # (P, s)
        full = self.shape[0] if transpose else self.shape[1]
        X = torch.zeros(full, M.shape[0], dtype=DTYPE)
        X[support] = M.T.to(DTYPE)
        r = (self.res_bwd(reader, X) if transpose else self.res_fwd(reader, X))[lines]
        w = M.sum(1).to(DTYPE)
        z = z_two_sided(lines.numel() * M.shape[0], self.fa)
        pos = r.abs() > z * sigma * w.sqrt()[None]
        return comp_dd(pos, M, dd=self.decoder == "dd")  # (len(lines), s)

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        before = reader.cost.copy()
        R, C = self.lines(reader)
        meta = {"n_rows": int(R.numel()), "n_cols": int(C.numel())}
        if R.numel() == 0 or C.numel() == 0:
            return empty_estimate(**meta), reader.cost - before
        fwd = self._decode(reader, R, C, transpose=False)  # (|R|, |C|)
        if reader.backward_supported:
            bwd = self._decode(reader, C, R, transpose=True).T
            fwd = fwd & bwd
        mask = torch.zeros(self.shape, dtype=torch.bool)
        mask[R[:, None], C[None, :]] = fwd
        cells = mask.nonzero()
        est = FaultEstimate(
            cells=cells, rows=cells[:, 0].unique(), cols=cells[:, 1].unique(), meta=meta
        )
        return est, reader.cost - before
