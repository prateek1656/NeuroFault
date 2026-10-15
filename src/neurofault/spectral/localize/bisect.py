"""A7 adaptive bisection with Wald SPRT stopping, forward-only and bidirectional.

Every decision is an SPRT of H0: block response = 0 vs H1: |response| >= delta (delta = delta_frac
* max|w_ref|, the design deviation), on repeated reads of the SAME 0/1 probe; the LLR of the
summed residual S over t reads of a probe of weight w is log cosh(delta S / (s^2 w)) - t delta^2 /
(2 s^2 w) (sign of the deviation unknown), stopped at +-A, A = log((1-err)/err).
Stage 0: ones-probe SPRT on every output line (rows by forward, columns by backward in the
bidirectional variant) -> the lines containing faults. Stage 1: a shared bisection tree over the
column support (forward-only: all n columns; bidirectional: the located columns C, or the
located rows R when that side is smaller - "picks the direction whose block is smaller"). At
each node the LEFT half is tested; the right half is tested only for lines whose left half
contained a fault (a line with no fault on the left has all of its faults on the right).
Probes of different nodes at the same level are merged into one read when no line occurs in two
of them, so single-fault lines cost ~log2 n reads in total, independent of k.
"""

from __future__ import annotations

import math

import torch

from neurofault.spectral.localize.common import RefLocalizer, empty_estimate
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike


def _logcosh(a: torch.Tensor) -> torch.Tensor:
    a = a.abs()
    return a + torch.log1p(torch.exp(-2 * a)) - math.log(2.0)


class AdaptiveBisectionLocalizer(RefLocalizer):
    name = "A7_adaptive_bisection"

    def __init__(
        self,
        w_ref,
        *,
        bidirectional: bool = False,
        err: float = 0.01,
        delta_frac: float = 0.25,
        max_rep: int = 64,
        **kw,
    ) -> None:
        super().__init__(w_ref, **kw)
        self.bidirectional, self.err = bidirectional, err
        self.delta = delta_frac * float(self.w_ref.abs().max())
        self.max_rep, self.A = max_rep, math.log((1 - err) / err)

    # one slot = one probe vector, repeated until every tested line is decided
    def _sprt(self, reader, x: torch.Tensor, lines: torch.Tensor, w: int, transpose: bool):
        s2 = max(self._need_sigma(), 1e-12) ** 2 * w
        S = torch.zeros(lines.numel(), dtype=DTYPE)
        for t in range(1, self.max_rep + 1):
            r = (self.res_bwd if transpose else self.res_fwd)(reader, x[:, None])[lines, 0]
            S += r
            llr = _logcosh(self.delta * S / s2) - t * self.delta**2 / (2 * s2)
            if bool(((llr >= self.A) | (llr <= -self.A)).all()):
                break
        return llr > 0

    def _root(self, reader, transpose: bool) -> torch.Tensor:
        m, n = self.shape
        full, n_lines = (m, n) if transpose else (n, m)
        x = torch.ones(full, dtype=DTYPE)
        return self._sprt(reader, x, torch.arange(n_lines), full, transpose).nonzero()[:, 0]

    def _test_blocks(self, reader, lines, support, pl, lo, hi, transpose) -> torch.Tensor:
        """bool per pair: does line pl hold a fault in support[lo:hi]? Packs nodes into probes."""
        full = self.shape[0] if transpose else self.shape[1]
        slots: list[tuple[set, dict]] = []  # (lines used, {block: None})
        slot_of = []
        for p, (ln, a, b) in enumerate(zip(pl.tolist(), lo.tolist(), hi.tolist(), strict=True)):
            for s, (used, blocks) in enumerate(slots):
                if ln not in used:
                    used.add(ln)
                    blocks[(a, b)] = None
                    slot_of.append(s)
                    break
            else:
                slots.append(({ln}, {(a, b): None}))
                slot_of.append(len(slots) - 1)
        slot_of_t = torch.tensor(slot_of)
        out = torch.zeros(pl.numel(), dtype=torch.bool)
        for s, (_, blocks) in enumerate(slots):
            x = torch.zeros(full, dtype=DTYPE)
            for a, b in blocks:
                x[support[a:b]] = 1.0
            sel = (slot_of_t == s).nonzero()[:, 0]
            out[sel] = self._sprt(reader, x, lines[pl[sel]], int(x.sum()), transpose)
        return out

    def _tree(self, reader, lines, support, transpose) -> torch.Tensor:
        pl = torch.arange(lines.numel())
        lo = torch.zeros_like(pl)
        hi = torch.full_like(pl, support.numel())
        done_l, done_s = [], []
        while pl.numel():
            leaf = (hi - lo) <= 1
            if leaf.any():
                done_l.append(pl[leaf])
                done_s.append(lo[leaf])
            pl, lo, hi = pl[~leaf], lo[~leaf], hi[~leaf]
            if not pl.numel():
                break
            mid = (lo + hi) // 2
            left = self._test_blocks(reader, lines, support, pl, lo, mid, transpose)
            right = ~left  # parent nonempty and left empty -> right holds the fault(s)
            if left.any():
                li = left.nonzero()[:, 0]
                right[li] = self._test_blocks(
                    reader, lines, support, pl[li], mid[li], hi[li], transpose
                )
            pl = torch.cat([pl[left], pl[right]])
            nlo = torch.cat([lo[left], mid[right]])
            nhi = torch.cat([mid[left], hi[right]])
            lo, hi = nlo, nhi
        if not done_l:
            return torch.zeros(0, 2, dtype=torch.long)
        return torch.stack([lines[torch.cat(done_l)], support[torch.cat(done_s)]], 1)

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        before = reader.cost.copy()
        n = self.shape[1]
        rows = self._root(reader, False)
        meta = {"n_rows": int(rows.numel()), "bidir": False}
        if rows.numel() == 0:
            return empty_estimate(**meta), reader.cost - before
        transpose = False
        lines, support = rows, torch.arange(n)
        if self.bidirectional and reader.backward_supported:
            cols = self._root(reader, True)
            meta.update(bidir=True, n_cols=int(cols.numel()))
            if cols.numel() == 0:
                return empty_estimate(**meta), reader.cost - before
            if cols.numel() > rows.numel():
                transpose, lines, support = True, cols, rows
            else:
                lines, support = rows, cols
        found = self._tree(reader, lines, support, transpose)  # (K, 2) = (line, support)
        cells = found.flip(1) if transpose else found
        est = FaultEstimate(
            cells=cells, rows=cells[:, 0].unique(), cols=cells[:, 1].unique(), meta=meta
        )
        return est, reader.cost - before
