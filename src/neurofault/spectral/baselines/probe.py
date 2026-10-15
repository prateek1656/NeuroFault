"""B4 / B4b / B5 / B9: probe-based detectors, localizers and the low-rank competitor.

WHY: these are the incumbents the spectral methods are compared to at equal read budget.
- ProbeResidual (B4): stored reference outputs of fixed +-1 probes; the residual is zero when
  nothing changed, so row scores localise a stuck cell by residual magnitude (forward) and the
  column by backward residual scores.
- ChecksumABFT (B4b): Huang-Abraham row/column checksum (probe = ones); reference taken at
  calibrate; cheap, but blind to faults that cancel in a sum.
- ProbeLowRank (B5): single-view (generalised Nystrom) sketch of E = W(t) - W0 using FIXED
  sketches Omega (n,p) and Psi (m,p), p = k+s: E ~ (E Omega)(Psi^T E Omega)^+ (E^T Psi)^T, then
  truncated to rank k. A range-finder with a data-dependent Q would need W0^T Q, which was not
  stored at calibration, hence the fixed two-sided sketch. Needs backward reads.
- Watchdog statistics (B9): |alpha_GDC - 1| and the all-ones output-sum shift.
"""

from __future__ import annotations

import torch

from neurofault.spectral.baselines.base import (
    Bindable,
    Correction,
    FaultEstimate,
    ReadCost,
    TileReaderLike,
    make_probes,
)
from neurofault.spectral.types import make_generator


def _robust_threshold(score: torch.Tensor, z: float) -> torch.Tensor:
    med = score.median()
    mad = (score - med).abs().median() * 1.4826
    return med + z * mad.clamp(min=1e-300)


class ProbeResidual:
    """Detector (stat = Frobenius or max |residual|) and Localizer (row/col scores)."""

    def __init__(
        self,
        p: int = 8,
        statistic: str = "fro",
        n_rows: int | None = 1,
        z: float = 6.0,
        seed: int = 0,
    ):
        if statistic not in ("fro", "max"):
            raise ValueError("statistic must be fro|max")
        self.p, self.stat, self.n_rows, self.z = p, statistic, n_rows, z
        self.gen = make_generator(seed, "probe_residual")
        self.name = f"B4_probe_residual_P{p}"
        self.x = self.zb = self.y0 = self.y0b = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        m, n = reader.shape
        self.x = make_probes("rademacher", n, self.p, self.gen)
        self.y0 = reader.mvm(self.x)
        self.zb = None
        if reader.backward_supported:
            self.zb = make_probes("rademacher", m, self.p, self.gen)
            self.y0b = reader.mvm_t(self.zb)
        return reader.cost - start

    def _residual(self, reader: TileReaderLike, backward: bool) -> torch.Tensor:
        if backward:
            return reader.mvm_t(self.zb) - self.y0b
        return reader.mvm(self.x) - self.y0

    def statistic(self, reader: TileReaderLike, t: float) -> tuple[float, ReadCost]:
        start = reader.cost.copy()
        r = self._residual(reader, False)
        s = float(r.norm() if self.stat == "fro" else r.abs().max())
        return s, reader.cost - start

    def localize(self, reader: TileReaderLike, t: float) -> tuple[FaultEstimate, ReadCost]:
        start = reader.cost.copy()
        rs = self._residual(reader, False).norm(dim=1)
        rows = self._pick(rs)
        cols = None
        if self.zb is not None:
            cs = self._residual(reader, True).norm(dim=1)
            cols = self._pick(cs)
        if cols is not None and rows.numel() and cols.numel():
            rr, cc = torch.meshgrid(rows, cols, indexing="ij")
            cells = torch.stack([rr.flatten(), cc.flatten()], 1)
            score = (rs[rr] * cs[cc]).flatten()
        else:
            cells = torch.zeros(0, 2, dtype=torch.long)
            score = None
        est = FaultEstimate(cells, rows, cols, score, {"row_score": rs})
        return est, reader.cost - start

    def _pick(self, score: torch.Tensor) -> torch.Tensor:
        if self.n_rows is not None:
            k = min(self.n_rows, score.numel())
            return score.topk(k).indices.sort().values
        return (score > _robust_threshold(score, self.z)).nonzero().squeeze(1)


class ChecksumABFT:
    """Row (W 1) and column (W^T 1) checksums vs stored references."""

    def __init__(self, z: float = 6.0):
        self.name = "B4b_checksum_abft"
        self.z = z
        self.r0 = self.c0 = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        m, n = reader.shape
        self.r0 = reader.mvm(torch.ones(n, 1, dtype=torch.float64))[:, 0]
        self.c0 = (
            reader.mvm_t(torch.ones(m, 1, dtype=torch.float64))[:, 0]
            if reader.backward_supported
            else None
        )
        return reader.cost - start

    def _deltas(self, reader: TileReaderLike):
        m, n = reader.shape
        dr = reader.mvm(torch.ones(n, 1, dtype=torch.float64))[:, 0] - self.r0
        dc = None
        if self.c0 is not None:
            dc = reader.mvm_t(torch.ones(m, 1, dtype=torch.float64))[:, 0] - self.c0
        return dr, dc

    def statistic(self, reader: TileReaderLike, t: float) -> tuple[float, ReadCost]:
        start = reader.cost.copy()
        dr, dc = self._deltas(reader)
        s = float(dr.abs().max())
        if dc is not None:
            s = max(s, float(dc.abs().max()))
        return s, reader.cost - start

    def localize(self, reader: TileReaderLike, t: float) -> tuple[FaultEstimate, ReadCost]:
        """Classic single-error ABFT location: argmax row checksum x argmax column checksum."""
        start = reader.cost.copy()
        dr, dc = self._deltas(reader)
        rows = dr.abs().argmax()[None]
        cols = dc.abs().argmax()[None] if dc is not None else None
        cells = (
            torch.stack([rows, cols], 1)
            if cols is not None
            else torch.zeros(0, 2, dtype=torch.long)
        )
        return FaultEstimate(cells, rows, cols, None, {}), reader.cost - start


class ProbeLowRank(Bindable):
    """B5: rank-k estimate of E = W(t) - W0 from a two-sided fixed sketch; W_eff = W(t) - E_k."""

    def __init__(self, k: int = 2, s: int = 4, seed: int = 0):
        self.k, self.s = k, s
        self.gen = make_generator(seed, f"probe_lowrank_{k}")
        self.name = f"B5_probe_lowrank_k{k}_s{s}"
        self.om = self.ps = self.y0 = self.z0 = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        if not reader.backward_supported:
            raise RuntimeError("ProbeLowRank needs backward reads (two-sided sketch)")
        start = reader.cost.copy()
        m, n = reader.shape
        p = self.k + self.s
        self.om = make_probes("rademacher", n, p, self.gen)
        self.ps = make_probes("rademacher", m, p, self.gen)
        self.y0 = reader.mvm(self.om)
        self.z0 = reader.mvm_t(self.ps)
        return reader.cost - start

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        assert self.om is not None, "calibrate() first"
        start = reader.cost.copy()
        y = reader.mvm(self.om) - self.y0  # E Omega  (m, p)
        z = reader.mvm_t(self.ps) - self.z0  # E^T Psi  (n, p)
        core = self.ps.T @ y  # Psi^T E Omega
        # core truncated to rank k: an untruncated pinv amplifies the tail of a full-rank E
        uc, sc, vhc = torch.linalg.svd(core)
        kc = min(self.k, sc.numel())
        cinv = (vhc[:kc].T / sc[:kc].clamp(min=1e-300)) @ uc[:, :kc].T
        a = y @ cinv
        qa, ra = torch.linalg.qr(a)
        qb, rb = torch.linalg.qr(z)
        u, sv, vh = torch.linalg.svd(ra @ rb.T)
        k = min(self.k, sv.numel())
        e_k = (qa @ u[:, :k]) * sv[:k] @ (qb @ vh[:k].T).T
        return Correction(self.now() - e_k, reader.cost - start, {"E_k": e_k, "sv": sv})


class GDCAlphaStat:
    """B9(i): |alpha_GDC - 1| from a single all-ones probe (free watchdog statistic)."""

    def __init__(self):
        self.name = "B9_gdc_alpha_dev"
        self.ref = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        self.ref = float(
            reader.mvm(torch.ones(reader.shape[1], 1, dtype=torch.float64)).abs().mean()
        )
        return reader.cost - start

    def statistic(self, reader: TileReaderLike, t: float) -> tuple[float, ReadCost]:
        start = reader.cost.copy()
        cur = float(reader.mvm(torch.ones(reader.shape[1], 1, dtype=torch.float64)).abs().mean())
        return abs(self.ref / max(cur, 1e-300) - 1.0), reader.cost - start


class CurrentSumStat:
    """B9(ii): shift of the summed output current for the all-ones input (power-monitor style)."""

    def __init__(self):
        self.name = "B9_current_sum"
        self.ref = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        self.ref = float(reader.mvm(torch.ones(reader.shape[1], 1, dtype=torch.float64)).sum())
        return reader.cost - start

    def statistic(self, reader: TileReaderLike, t: float) -> tuple[float, ReadCost]:
        start = reader.cost.copy()
        cur = float(reader.mvm(torch.ones(reader.shape[1], 1, dtype=torch.float64)).sum())
        return abs(cur - self.ref), reader.cost - start
