"""X-ABFT localizer (Liu, Xia, Wang, Chakrabarty, ACM TODAES 25(3), 29, 2020), forward-only.

Implemented from the paper text (Secs. 3.1-3.5, 4, 5, 7.2; Eqs. 1-8, 20), not from code:
- The weight array is partitioned into test blocks of r_t INPUT lines x c_t OUTPUT lines (the paper's
  "rows" = our columns n = inputs, the paper's "columns" = our rows m = outputs). Each sub-array of
  c_t outputs carries two checksum lines: a plain one (sum of its outputs) and a weighted one
  (sum_j j * output_j, j = 1..c_t).
- M designed test vectors per input block: x_k(i) = f(i)^(k-1) on the block's inputs, 0 elsewhere
  (f(i) = i linear, or 2^(i-1) exponential; local index i = 1..r_t). All output sub-arrays are read
  by the same vectors. Test vectors per round: M * ceil(n / r_t).
- Signatures A(k) = sum_j O_t(k,j) - O_s(k,1), B(k) = sum_j j O_t(k,j) - O_s(k,2) (Eqs. 1-2).
  Here the checksum outputs O_s come from the digital reference ``w_ref`` (every localizer in this
  package works on residuals r = read - w_ref x), i.e. the checksum cells are noise-free and
  fault-free: this FAVOURS X-ABFT; a physical checksum column adds read noise and can be faulty.
  A(k), B(k) are therefore sums of the measured residuals.
- Decoding, at most 2 faults per block (Sec. 3.4, Eq. 5): single fault (x, y, d): A(k) = f(x)^(k-1) d,
  B(k) = y A(k), so y = B(1)/A(1) and f(x) = A(2)/A(1) (Eq. 20), accepted when A(k+1) = u A(k)
  within relative tolerance ``eps`` (paper's A ~ B test) plus a ``z_tol``-sigma noise allowance. Otherwise, with
  M >= 4, the two-fault closed-form solve (Prony form: A(3) = s A(2) - p A(1), A(4) = s A(3) -
  p A(2); u1, u2 = roots of z^2 - s z + p; then d_f and y_f from A(1..2), B(1..2)). Faults in the
  same input row of a block (u1 == u2), > 2 faults, or a failed snap are "undecodable" (no cells).
- Detection (A, B ~ 0) uses a noise-aware chi^2 test over the 2M signatures at ``fa`` expected
  false alarms per call: the paper's tolerance eps0 made explicit. The decode itself is the paper's
  exact algebra, which divides by noisy signatures; that is the point of the comparison.

Cost (``meta``): ``n_test_vectors`` = M ceil(n/r_t) counted forward reads; paper's native cost model
TR = (tau + n_test_vectors)/tau with tau = 10_000 computing cycles, HR = 1 + 5/c_t (2+3 cells per
checksum pair, Sec. 7.2). ``validate=True`` adds a chi^2 consistency check of the snapped model
(not in the paper; off by default).
"""

from __future__ import annotations

import math

import torch

from neurofault.spectral.localize.common import RefLocalizer, chi2_threshold, empty_estimate
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike

TAU_CYCLES = 10_000


class XABFTLocalizer(RefLocalizer):
    name = "A9_xabft"

    def __init__(
        self,
        w_ref,
        *,
        r_t: int = 4,
        c_t: int = 16,
        M: int = 4,
        f: str = "linear",
        eps: float = 0.02,
        z_tol: float = 3.0,
        validate: bool = False,
        **kw,
    ) -> None:
        super().__init__(w_ref, **kw)
        assert f in ("linear", "exp") and M >= 2 and r_t >= 1 and c_t >= 1
        self.r_t, self.c_t, self.M, self.f, self.eps, self.z_tol, self.validate = (
            r_t,
            c_t,
            M,
            f,
            eps,
            z_tol,
            validate,
        )
        m, n = self.shape
        self.n_ib, self.n_ob = math.ceil(n / r_t), math.ceil(m / c_t)
        loc = torch.arange(r_t, dtype=DTYPE)
        self.fvals = loc + 1 if f == "linear" else 2.0**loc  # f(i), local i = 1..r_t
        self.colw = torch.arange(1, c_t + 1, dtype=DTYPE)

    @property
    def n_test_vectors(self) -> int:
        return self.M * self.n_ib

    def _vectors(self) -> torch.Tensor:
        n = self.shape[1]
        X = torch.zeros(n, self.n_ib * self.M, dtype=DTYPE)
        for b in range(self.n_ib):
            lo, hi = b * self.r_t, min(n, (b + 1) * self.r_t)
            fv = self.fvals[: hi - lo]
            for k in range(self.M):
                X[lo:hi, b * self.M + k] = fv**k
        return X

    def _snap_row(self, u: float) -> int:
        return int((self.fvals - u).abs().argmin())

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        sigma = self._need_sigma()
        before = reader.cost.copy()
        m, n = self.shape
        X = self._vectors()
        R = self.res_fwd(reader, X)  # (m, n_ib*M)
        pad = self.n_ob * self.c_t - m
        if pad:
            R = torch.cat([R, torch.zeros(pad, R.shape[1], dtype=DTYPE)])
        R = R.reshape(self.n_ob, self.c_t, self.n_ib, self.M)
        A = R.sum(1)  # (ob, ib, M)
        B = (self.colw[None, :, None, None] * R).sum(1)
        xn = X.norm(dim=0).reshape(self.n_ib, self.M)  # ||x_(b,k)||
        sdA = sigma * xn * math.sqrt(self.c_t)
        sdB = sigma * xn * math.sqrt(float((self.colw**2).sum()))
        stat = ((A / sdA) ** 2 + (B / sdB) ** 2).sum(-1)  # (ob, ib)
        flagged = stat > chi2_threshold(2 * self.M, self.n_ob * self.n_ib, self.fa)
        meta = {
            "n_test_vectors": self.n_test_vectors,
            "time_redundancy": (TAU_CYCLES + self.n_test_vectors) / TAU_CYCLES,
            "hardware_redundancy": 1.0 + 5.0 / self.c_t,
            "n_flagged": int(flagged.sum()),
            "undecodable": 0,
        }
        cells = []
        for ob, ib in flagged.nonzero().tolist():
            dec = self._decode(A[ob, ib], B[ob, ib], sdA[ib], sdB[ib])
            if dec is None:
                meta["undecodable"] += 1
                continue
            for xi, y in dec:
                row, col = ob * self.c_t + y - 1, ib * self.r_t + xi
                if row < m and col < n:
                    cells.append((row, col))
        if not cells:
            return empty_estimate(**meta), reader.cost - before
        c = torch.tensor(sorted(set(cells)), dtype=torch.long)
        return FaultEstimate(cells=c, rows=c[:, 0].unique(), cols=c[:, 1].unique(), meta=meta), (
            reader.cost - before
        )

    def _decode(self, A, B, sdA, sdB):
        """[(local input index, output index 1..c_t)] or None. A, B: (M,) signatures."""
        a, b = A.tolist(), B.tolist()
        if abs(a[0]) < 1e-12:
            return None
        u = a[1] / a[0]
        # A(k+1) = u A(k) within a tolerance: relative eps (ADC/variation margin, paper) plus a
        # noise allowance z_tol * sd (favours X-ABFT: the paper's eps is fixed, ours tracks noise)
        single = all(
            abs(a[k + 1] - u * a[k])
            <= self.eps * abs(a[k + 1]) + self.z_tol * float(sdA[k + 1]) * math.sqrt(1 + u * u)
            for k in range(1, self.M - 1)
        )
        if single:
            pairs = [(u, b[0] / a[0], a[0])]
        elif self.M >= 4:
            det = a[1] * a[1] - a[2] * a[0]
            if abs(det) < 1e-18:
                return None
            # a2 = s a1 - p a0 ; a3 = s a2 - p a1  ->  solve for (s, p)
            s = (a[2] * a[1] - a[3] * a[0]) / det
            p = (a[2] * a[2] - a[3] * a[1]) / det
            disc = s * s - 4 * p
            if disc < 0:
                return None
            u1, u2 = (s + math.sqrt(disc)) / 2, (s - math.sqrt(disc)) / 2
            if abs(u1 - u2) < 1e-12:
                return None
            d1 = (a[1] - u2 * a[0]) / (u1 - u2)
            d2 = a[0] - d1
            yd1 = (b[1] - u2 * b[0]) / (u1 - u2)
            yd2 = b[0] - yd1
            if abs(d1) < 1e-12 or abs(d2) < 1e-12:
                return None
            pairs = [(u1, yd1 / d1, d1), (u2, yd2 / d2, d2)]
        else:
            return None
        out = []
        for uu, yy, _d in pairs:
            xi, y = self._snap_row(uu), round(yy)
            if not 1 <= y <= self.c_t:
                return None
            out.append((xi, y))
        if len({x for x, _ in out}) < len(out):
            return None
        if self.validate and not self._consistent(out, pairs, a, b, sdA, sdB):
            return None
        return out

    def _consistent(self, out, pairs, a, b, sdA, sdB) -> bool:
        stat = 0.0
        for k in range(self.M):
            pa = sum(float(self.fvals[xi]) ** k * d for (xi, _), (_, _, d) in zip(out, pairs))
            pb = sum(float(self.fvals[xi]) ** k * y * d for (xi, y), (_, _, d) in zip(out, pairs))
            stat += ((a[k] - pa) / float(sdA[k])) ** 2 + ((b[k] - pb) / float(sdB[k])) ** 2
        return stat <= chi2_threshold(2 * self.M, 1, self.fa)
