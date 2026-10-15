"""A3 row/column residual intersection (the "obvious" bidirectional baseline, 00-grounding).

Stage 1: ``n_fwd`` dense +-1 forward probes and ``n_bwd`` dense +-1 backward probes. The
residual energy of every output (row) / input (column) is compared with a chi^2 threshold at
fixed false alarm, giving row set R and column set C. Stage 2 (whenever |R||C| > 1: the block is ambiguous
or contains a false-alarm line): ``n_prune`` extra probes supported on the smaller of C / R, decoded line by line
with OMP; when the support has <= n_prune entries unit vectors are used (exact). If the
candidate block is larger than ``cap_cells`` pruning is skipped and R x C is returned
(the dense-fault regime where this design degenerates). Without backward support the column set
is all columns and pruning does the work.
"""

from __future__ import annotations

import math

import torch

from neurofault.spectral.localize.common import (
    RefLocalizer,
    chi2_threshold,
    empty_estimate,
    omp,
    rademacher,
)
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike


def prune_lines(
    loc: RefLocalizer,
    reader: TileReaderLike,
    lines: torch.Tensor,
    support: torch.Tensor,
    n_probe: int,
    *,
    transpose: bool,
) -> torch.Tensor:
    """Sparse-decode, line by line, the entries of E on lines x support.

    transpose=False: forward probes supported on ``support`` (columns), outputs read on
    ``lines`` (rows). transpose=True: the mirror image with backward probes. Returns a bool
    matrix (len(lines), len(support))."""
    sigma = loc._need_sigma()
    m, n = loc.shape
    full = m if transpose else n  # length of the probe vector
    s = support.numel()
    if s <= n_probe:
        A = torch.eye(s, dtype=DTYPE)
    else:
        A = rademacher(s, n_probe, loc.gen).T  # (K, s)
    K = A.shape[0]
    X = torch.zeros(full, K, dtype=DTYPE)
    X[support] = A.T
    R = (loc.res_bwd(reader, X) if transpose else loc.res_fwd(reader, X))[lines]  # (L, K)
    noise_sd = sigma * math.sqrt(float((A[0] ** 2).sum()))
    out = torch.zeros(lines.numel(), s, dtype=torch.bool)
    if K >= s:  # identity: direct thresholding of each entry
        out = R.abs() > chi2_threshold(1, lines.numel() * s, loc.fa) ** 0.5 * noise_sd
        return out
    for li in range(lines.numel()):
        idx, _ = omp(A, R[li], noise_sd, fa=loc.fa / max(lines.numel(), 1))
        out[li, idx] = True
    return out


class ResidualLocalizer(RefLocalizer):
    name = "A3_residual"

    def __init__(
        self,
        w_ref,
        *,
        n_fwd: int = 4,
        n_bwd: int = 4,
        n_prune: int = 16,
        cap_cells: int = 1 << 14,
        **kw,
    ) -> None:
        super().__init__(w_ref, **kw)
        self.n_fwd, self.n_bwd, self.n_prune, self.cap_cells = n_fwd, n_bwd, n_prune, cap_cells

    def lines(self, reader: TileReaderLike) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self._need_sigma()
        m, n = self.shape
        rf = self.res_fwd(reader, rademacher(n, self.n_fwd, self.gen))
        rows = (rf**2).sum(1) / (sigma**2 * n) > chi2_threshold(self.n_fwd, m, self.fa)
        if reader.backward_supported and self.n_bwd > 0:
            rb = self.res_bwd(reader, rademacher(m, self.n_bwd, self.gen))
            cols = (rb**2).sum(1) / (sigma**2 * m) > chi2_threshold(self.n_bwd, n, self.fa)
        else:
            cols = torch.ones(n, dtype=torch.bool)
        return rows.nonzero()[:, 0], cols.nonzero()[:, 0]

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        before = reader.cost.copy()
        R, C = self.lines(reader)
        meta = {"n_rows": int(R.numel()), "n_cols": int(C.numel())}
        if R.numel() == 0 or (C.numel() == 0):
            return empty_estimate(**meta), reader.cost - before
        mask = torch.zeros(self.shape, dtype=torch.bool)
        if R.numel() * C.numel() == 1 or R.numel() * C.numel() > self.cap_cells:
            mask[R[:, None], C[None, :]] = True
            meta["pruned"] = False
        else:
            meta["pruned"] = True
            if C.numel() <= R.numel() or not reader.backward_supported:
                sub = prune_lines(self, reader, R, C, self.n_prune, transpose=False)
            else:
                sub = prune_lines(self, reader, C, R, self.n_prune, transpose=True).T
            mask[R[:, None], C[None, :]] = sub
        cells = mask.nonzero()
        est = FaultEstimate(
            cells=cells, rows=cells[:, 0].unique(), cols=cells[:, 1].unique(), meta=meta
        )
        return est, reader.cost - before
