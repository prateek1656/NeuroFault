"""A5 non-adaptive compressed sensing: P_f forward + P_b backward random probes, joint OMP.

All probes are drawn before any read (non-adaptive, hardware-simple); only the decode adapts.
A probe has ``weight`` active entries (random support) with +-1 values (``kind="rademacher"``)
or +1 (``"bernoulli"``); ``weight=None`` is the dense probe. Reads are exactly p_fwd + p_bwd.

Decode (all residuals whitened by the calibrated noise sd sigma*||x||):
- one OMP over the full m x n cell grid (matched-filter correlation of every cell with the forward measurements of its row and, if p_bwd > 0, the backward measurements of its column: two independent measurements per fault, T4), in operator form (no dense mn-column
  matrix), Bonferroni max-correlation stopping (at most ``max_nnz`` cells; the experiments pass
  2k+4, the designed fault count) and a final coefficient z-prune. Forward-only is
  the same with no backward term. ``screen=True`` restricts the grid to energy-flagged lines
  (cheaper, but a weaker detector than the matched filter at low SNR).
The stuck-hypothesis prior enters only through the z-prune on the fitted deviation (a stuck cell
has a deviation of at least a visible fraction of the range); no tuned lambda exists.
"""

from __future__ import annotations

import torch

from neurofault.spectral.localize.common import (
    RefLocalizer,
    chi2_threshold,
    empty_estimate,
    rademacher,
    z_two_sided,
)
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike


def sparse_probes(
    n: int, p: int, weight: int | None, gen: torch.Generator, kind: str = "rademacher"
) -> torch.Tensor:
    """(n, p) probes with ``weight`` active entries each (all n when weight is None/>=n)."""
    if weight is None or weight >= n:
        return rademacher(n, p, gen) if kind == "rademacher" else torch.ones(n, p, dtype=DTYPE)
    idx = torch.rand(n, p, generator=gen).argsort(0)[:weight]
    vals = (
        rademacher(weight, p, gen) if kind == "rademacher" else torch.ones(weight, p, dtype=DTYPE)
    )
    return torch.zeros(n, p, dtype=DTYPE).scatter_(0, idx, vals)


def flag_lines(Z: torch.Tensor, fa: float) -> torch.Tensor:
    """Whitened residuals Z (lines, probes) -> bool per line: chi^2 energy OR max-z test."""
    L, P = Z.shape
    e = (Z**2).sum(1) > chi2_threshold(P, L, fa / 2)
    mz = Z.abs().amax(1) > z_two_sided(L * P, fa / 2)
    return e | mz


def omp_maxcorr(
    A: torch.Tensor, y: torch.Tensor, zthr: float, max_nnz: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """OMP for y = A e + N(0, I): add the column with the largest normalized correlation until it
    drops under ``zthr``; least-squares refit; drop coefficients whose z-score falls under zthr."""
    cn = A.norm(dim=0).clamp_min(1e-12)
    sel: list[int] = []
    r = y.clone()
    coef = torch.zeros(0, dtype=DTYPE)
    while len(sel) < max_nnz:
        c = (A.T @ r).abs() / cn
        if sel:
            c[sel] = -1.0
        j = int(c.argmax())
        if float(c[j]) < zthr:
            break
        sel.append(j)
        coef = torch.linalg.lstsq(A[:, sel], y[:, None]).solution[:, 0]
        r = y - A[:, sel] @ coef
    if not sel:
        return torch.zeros(0, dtype=torch.long), coef
    As = A[:, sel]
    sd = torch.linalg.pinv(As.T @ As).diagonal().clamp_min(0).sqrt()
    keep = coef.abs() > zthr * sd
    return torch.tensor(sel, dtype=torch.long)[keep], coef[keep]


class NonAdaptiveCSLocalizer(RefLocalizer):
    name = "A5_cs"

    def __init__(
        self,
        w_ref,
        *,
        p_fwd: int = 16,
        p_bwd: int = 0,
        weight: int | None = None,
        kind: str = "rademacher",
        screen: bool = False,
        max_nnz: int = 128,
        **kw,
    ) -> None:
        super().__init__(w_ref, **kw)
        self.p_fwd, self.p_bwd, self.weight, self.kind = p_fwd, p_bwd, weight, kind
        self.screen, self.max_nnz = screen, max_nnz

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        sigma = max(self._need_sigma(), 1e-12)
        before = reader.cost.copy()
        m, n = self.shape
        bidir = self.p_bwd > 0 and reader.backward_supported
        Xf = sparse_probes(n, self.p_fwd, self.weight, self.gen, self.kind)
        nf = Xf.pow(2).sum(0).sqrt()
        Zf = self.res_fwd(reader, Xf) / (sigma * nf)
        if bidir:
            wb = None if self.weight is None else max(1, round(self.weight * m / n))
            Yb = sparse_probes(m, self.p_bwd, wb, self.gen, self.kind)
            nb = Yb.pow(2).sum(0).sqrt()
            Zb = self.res_bwd(reader, Yb) / (sigma * nb)
            Yn = Yb / nb
        else:
            Zb, Yn = torch.zeros(n, 0, dtype=DTYPE), torch.zeros(m, 0, dtype=DTYPE)
        R, C = torch.arange(m), torch.arange(n)
        meta = {"bidir": bidir, "screen": self.screen}
        if self.screen:  # optional energy screening of lines (weaker than the matched filter)
            R = flag_lines(Zf, self.fa).nonzero()[:, 0]
            C = flag_lines(Zb, self.fa).nonzero()[:, 0] if bidir else C
            meta.update(n_rows=int(R.numel()), n_cols=int(C.numel()))
            if R.numel() == 0 or C.numel() == 0:
                return empty_estimate(**meta), reader.cost - before
        mask = torch.zeros(m, n, dtype=torch.bool)
        sel = self._joint(Xf / nf, Zf, Yn, Zb, R, C)
        if sel.numel():
            mask[R[sel[:, 0]], C[sel[:, 1]]] = True
        return self._finish(mask, meta, reader, before)

    def _finish(self, mask, meta, reader, before):
        cells = mask.nonzero()
        est = FaultEstimate(
            cells=cells, rows=cells[:, 0].unique(), cols=cells[:, 1].unique(), meta=meta
        )
        return est, reader.cost - before

    def _joint(self, Xn, Zf, Yn, Zb, R, C) -> torch.Tensor:
        """Operator-form joint OMP. Xn (n, pf) / Yn (m, pb) are probes already divided by their
        norm (so that the whitened model is Zf[i, p] = sum_j E_ij Xn[j, p])."""
        Xc, Yr = Xn[C], Yn[R]  # (|C|, pf), (|R|, pb)
        zf, zb = Zf[R].clone(), Zb[C].clone()
        nR, nC = R.numel(), C.numel()
        zthr = z_two_sided(nR * nC, self.fa / 2)
        cn = (Xc.pow(2).sum(1)[None, :] + Yr.pow(2).sum(1)[:, None]).sqrt()  # (|R|, |C|)
        Kf, Kb = Xc @ Xc.T, Yr @ Yr.T
        ri: list[int] = []
        cj: list[int] = []
        e = torch.zeros(0, dtype=DTYPE)
        max_nnz = min(self.max_nnz, nR * nC, nR * Xn.shape[1] + nC * Yn.shape[1] - 1)
        while len(ri) < max_nnz:
            corr = (zf @ Xc.T + Yr @ zb.T).abs() / cn
            if ri:
                corr[ri, cj] = -1.0
            flat = int(corr.argmax())
            i, j = divmod(flat, nC)
            if float(corr[i, j]) < zthr:
                break
            ri.append(i)
            cj.append(j)
            e = self._refit(ri, cj, Kf, Kb, Zf[R], Xc, Zb[C], Yr, zf, zb)
        if not ri:
            return torch.zeros(0, 2, dtype=torch.long)
        G = self._gram(ri, cj, Kf, Kb)
        sd = torch.linalg.pinv(G).diagonal().clamp_min(0).sqrt()
        keep = e.abs() > zthr * sd
        return torch.stack([torch.tensor(ri), torch.tensor(cj)], 1)[keep]

    @staticmethod
    def _gram(ri, cj, Kf, Kb):
        a, b = torch.tensor(ri), torch.tensor(cj)
        same_r = (a[:, None] == a[None, :]).to(DTYPE)
        same_c = (b[:, None] == b[None, :]).to(DTYPE)
        return same_r * Kf[b[:, None], b[None, :]] + same_c * Kb[a[:, None], a[None, :]]

    def _refit(self, ri, cj, Kf, Kb, Zf0, Xc, Zb0, Yr, zf, zb) -> torch.Tensor:
        a, b = torch.tensor(ri), torch.tensor(cj)
        G = self._gram(ri, cj, Kf, Kb) + 1e-10 * torch.eye(len(ri), dtype=DTYPE)
        rhs = (Zf0[a] * Xc[b]).sum(1) + (Zb0[b] * Yr[a]).sum(1)
        e = torch.linalg.solve(G, rhs)
        zf.copy_(Zf0)
        zb.copy_(Zb0)
        zf.index_add_(0, a, -e[:, None] * Xc[b])
        zb.index_add_(0, b, -e[:, None] * Yr[a])
        return e
