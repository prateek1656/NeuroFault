"""Wei & Roth (arXiv 2402.13503) on their own terms: bounded noise, line-level, redundancy cost.

Paper model: read y = c + eps + e, |eps_j| <= delta (bounded, tolerable) and a sparse outlier
vector e. A (lambda-1)-disjunct 0/1 parity-check matrix H (r x n, row weight <= rho, column weight
<= w) gives syndrome s = H y; chi_rho(s)_i = [|s_i| > rho delta]. Theorem 19 decoder (Eq. 25):
decode j iff (chi H)_j > lambda - tau - sigma, valid for 2 tau + sigma <= 2 lambda - w. It locates
every outlier with |e_j| > 2 rho delta and, for <= tau outliers, never a non-outlier. The paper
stops at output LINES; the cost is r redundant columns inside ONE matrix-vector product.

This module adds (all new; nothing in roth.py is changed):
- ``BoundedNoiseReader``: wraps a noiseless TileReader and adds per-output noise bounded by
  ``bound_unit * ||x||`` (``sign``: +-delta_unit iid, ``uniform``: U(-sqrt3, sqrt3) delta_unit; both
  have rms delta_unit, i.e. rms-matched to the Gaussian read model of sigma = delta_unit).
- ``WeiRothPaperLocalizer``: ``cell_stage="none"`` is the paper's line-level decoder (one MVM,
  ``meta['r_cols']`` redundant columns); ``cell_stage="ks"`` is OUR cell-level extension (same
  Kautz-Singleton input-side probes as roth.py, deterministic bounded-noise thresholds).
  ``threshold="worst"`` is the paper's rho*delta; ``"rss"`` uses z*delta*sqrt(w_i), the
  random-noise (non-adversarial) tightening, reported separately.
Redundancy is emulated digitally from ``w_ref`` as for every localizer here (syndrome of the
residual); r is H.shape[0].
"""

from __future__ import annotations

import math

import torch

from neurofault.spectral.localize.common import (
    RefLocalizer,
    comp_dd,
    empty_estimate,
    kautz_singleton,
    rademacher,
    z_two_sided,
)
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike


class BoundedNoiseReader:
    """TileReaderLike adding bounded (not Gaussian) output noise to a noiseless inner reader."""

    def __init__(self, inner, delta_unit: float, gen: torch.Generator, dist: str = "sign") -> None:
        assert dist in ("sign", "uniform")
        self.inner, self.delta_unit, self.gen, self.dist = inner, delta_unit, gen, dist

    @property
    def cost(self) -> ReadCost:
        return self.inner.cost

    @property
    def backward_supported(self) -> bool:
        return self.inner.backward_supported

    @property
    def shape(self) -> tuple[int, int]:
        return self.inner.shape

    @property
    def bound_unit(self) -> float:
        return self.delta_unit * (1.0 if self.dist == "sign" else math.sqrt(3.0))

    def _eps(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.dist == "sign":
            s = (torch.randint(0, 2, y.shape, generator=self.gen) * 2 - 1).to(DTYPE)
        else:
            s = (torch.rand(*y.shape, generator=self.gen, dtype=DTYPE) * 2 - 1) * math.sqrt(3.0)
        return y + self.delta_unit * z.to(DTYPE).norm(dim=0) * s

    def mvm(self, x: torch.Tensor) -> torch.Tensor:
        return self._eps(self.inner.mvm(x), x)

    def mvm_t(self, y: torch.Tensor) -> torch.Tensor:
        return self._eps(self.inner.mvm_t(y), y)

    def full_readout(self) -> torch.Tensor:
        raise NotImplementedError


def line_f1(pred_rows, truth_cells) -> float:
    """F1 between predicted output lines and the lines holding a visible fault."""
    p = {int(r) for r in torch.as_tensor(pred_rows).reshape(-1).tolist()}
    t = {int(r) for r in torch.as_tensor(truth_cells, dtype=torch.long).reshape(-1, 2)[:, 0]}
    if not p and not t:
        return 1.0
    tp = len(p & t)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(p), tp / len(t)
    return 2 * pr * rc / (pr + rc)


class WeiRothPaperLocalizer(RefLocalizer):
    name = "A2c_roth_paper"

    def __init__(
        self,
        w_ref,
        *,
        bound_unit: float,
        d_lines: int = 1,
        d_cells: int = 1,
        cell_stage: str = "none",
        threshold: str = "worst",
        decoder: str = "thm19",
        **kw,
    ) -> None:
        super().__init__(w_ref, **kw)
        assert cell_stage in ("none", "ks") and threshold in ("worst", "rss")
        assert decoder in ("thm19", "comp")
        m, n = self.shape
        self.bound_unit, self.cell_stage, self.threshold, self.decoder = (
            bound_unit,
            cell_stage,
            threshold,
            decoder,
        )
        self.H = kautz_singleton(m, d_lines)
        self.M = kautz_singleton(n, d_cells)
        self.lam = d_lines + 1

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        return ReadCost()  # delta is a model parameter of the code, not estimated

    def _decode_lines(self, chi: torch.Tensor) -> tuple[torch.Tensor, bool]:
        H = self.H.to(DTYPE)
        cnt = chi.to(DTYPE) @ H  # (m,)
        w = int(self.H.sum(0).max())
        room = 2 * self.lam - w
        if self.decoder == "thm19" and room >= 0:
            tau = room // 2
            sig = room - 2 * tau
            return cnt > self.lam - tau - sig, True
        return cnt >= H.sum(0), False  # COMP: all tests of the item positive

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        before = reader.cost.copy()
        _m, n = self.shape
        H = self.H.to(DTYPE)
        wt = H.sum(1)
        x = rademacher(n, 1, self.gen)
        s = H @ self.res_fwd(reader, x)  # (r, 1)
        dx = self.bound_unit * math.sqrt(n)
        if self.threshold == "worst":
            thr = float(wt.max()) * dx
            chi = s[:, 0].abs() > thr
        else:
            chi = s[:, 0].abs() > z_two_sided(self.H.shape[0], self.fa) * dx * wt.sqrt()
        lines, guaranteed = self._decode_lines(chi)
        rows = lines.nonzero()[:, 0]
        meta = {
            "r_cols": int(self.H.shape[0]),
            "paper_guarantee_form": guaranteed,
            "rho": int(wt.max()),
            "bad_lines": rows,
        }
        if rows.numel() == 0:
            return empty_estimate(**meta), reader.cost - before
        if self.cell_stage == "none":
            z = torch.zeros(0, dtype=torch.long)
            est = FaultEstimate(cells=torch.zeros(0, 2, dtype=torch.long), rows=rows, cols=z)
            est.meta = meta
            return est, reader.cost - before
        r = self.res_fwd(reader, self.M.T.to(DTYPE))[rows]
        w = self.M.sum(1).to(DTYPE)
        if self.threshold == "worst":
            pos = r.abs() > self.bound_unit * w.sqrt()[None]
        else:
            z = z_two_sided(rows.numel() * self.M.shape[0], self.fa)
            pos = r.abs() > z * self.bound_unit * w.sqrt()[None]
        hit = comp_dd(pos, self.M, dd=True)
        sub = hit.nonzero()
        cells = torch.stack([rows[sub[:, 0]], sub[:, 1]], dim=1)
        est = FaultEstimate(
            cells=cells, rows=cells[:, 0].unique(), cols=cells[:, 1].unique(), meta=meta
        )
        return est, reader.cost - before
