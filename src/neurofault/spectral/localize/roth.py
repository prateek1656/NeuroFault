"""A2b Wei-Roth forward-only group-testing localizer (arXiv 2402.13503).

What is reimplemented (from the paper's description, not its full proofs):
the outlier model y = c + eps + e with sparse e, a sparse 0-1 DISJUNCT parity-check matrix H
(Kautz-Singleton here), the syndrome s = H r and a group-testing (COMP, optionally DD) decode of
the outlier positions, up to d outliers by design.
- Stage 1 (output lines, the paper's setting): ``stage1="roth_syndrome"`` forms group syndromes
  of the dense-probe residual over the m outputs and decodes the bad output lines by COMP.
  ``stage1="direct"`` thresholds the residual of each output (more information, since this
  reader exposes every output; the syndrome decode is dominated by it, its advantage being
  fewer ADCs/parity columns: ``meta['n_syndromes']``).
- Stage 2 (cells; our forward-only extension, the paper stops at lines): the SAME disjunct matrix
  construction is used as an INPUT-side probe design over the n columns. Each probe is a 0/1
  indicator of a group; for every bad line a test is positive when |r| > z sigma sqrt(group
  weight); COMP(+DD) gives the faulty columns of that line. Exact for <= d faults per line when
  noise-free. One read direction only, no transpose.
"""

from __future__ import annotations

import torch

from neurofault.spectral.localize.common import (
    RefLocalizer,
    chi2_threshold,
    comp_dd,
    empty_estimate,
    kautz_singleton,
    rademacher,
    z_two_sided,
)
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike


class RothLocalizer(RefLocalizer):
    name = "A2b_roth"

    def __init__(
        self,
        w_ref,
        *,
        d_lines: int = 3,
        d_cells: int = 3,
        n_dense: int = 2,
        stage1: str = "roth_syndrome",
        decoder: str = "dd",
        **kw,
    ) -> None:
        super().__init__(w_ref, **kw)
        assert stage1 in ("roth_syndrome", "direct") and decoder in ("comp", "dd")
        m, n = self.shape
        self.d_lines, self.d_cells, self.n_dense = d_lines, d_cells, n_dense
        self.stage1, self.decoder = stage1, decoder
        self.H = kautz_singleton(m, d_lines)  # (P1, m)
        self.M = kautz_singleton(n, d_cells)  # (P2, n)

    def _bad_lines(self, reader: TileReaderLike, sigma: float) -> tuple[torch.Tensor, int]:
        m, n = self.shape
        X = rademacher(n, self.n_dense, self.gen)
        r = self.res_fwd(reader, X)  # (m, n_dense)
        if self.stage1 == "direct":
            stat = (r**2).sum(1) / (sigma**2 * n)
            return stat > chi2_threshold(self.n_dense, m, self.fa), m
        Hf = self.H.to(DTYPE)
        s = Hf @ r  # (P1, n_dense)
        w = Hf.sum(1)
        stat = (s**2).sum(1) / (sigma**2 * n * w)
        pos = stat > chi2_threshold(self.n_dense, self.H.shape[0], self.fa)
        return comp_dd(pos[None], self.H, dd=self.decoder == "dd")[0], self.H.shape[0]

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        sigma = self._need_sigma()
        before = reader.cost.copy()
        lines, n_syn = self._bad_lines(reader, sigma)
        rows = lines.nonzero()[:, 0]
        meta = {"n_syndromes": n_syn, "bad_lines": rows}
        if rows.numel() == 0:
            return empty_estimate(**meta), reader.cost - before
        r = self.res_fwd(reader, self.M.T.to(DTYPE))[rows]  # (B, P2)
        w = self.M.sum(1).to(DTYPE)
        z = z_two_sided(rows.numel() * self.M.shape[0], self.fa)
        pos = r.abs() > z * sigma * w.sqrt()[None]
        hit = comp_dd(pos, self.M, dd=self.decoder == "dd")  # (B, n)
        sub = hit.nonzero()
        cells = torch.stack([rows[sub[:, 0]], sub[:, 1]], dim=1)
        est = FaultEstimate(
            cells=cells, rows=cells[:, 0].unique(), cols=cells[:, 1].unique(), meta=meta
        )
        return est, reader.cost - before
