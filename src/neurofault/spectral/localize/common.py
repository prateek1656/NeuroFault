"""Shared pieces of the localizers: reference-residual probing, thresholds, decoders.

Every localizer here works on RESIDUALS. The digital copy of the programmed weights
``w_ref`` (known at deploy time, free to use in software) turns a counted read into
``r = reader.mvm(x) - w_ref @ x = E x + noise`` with E = W_now - W_ref the (sparse) fault
perturbation. Estimators never see W_now: only ``TileReaderLike.mvm / mvm_t`` are called, so
every probe is counted in ``reader.cost`` ("no cheating" rule).

Noise model assumed by the thresholds: white per-cell read noise, i.e. each output of a probe
x carries N(0, sigma^2 ||x||^2). ``sigma`` (weight units per unit probe norm) is estimated in
``calibrate`` from two repeat reads of the same dense probes: the difference cancels any fault
(a fault is the same in both reads) and leaves pure noise, so calibration is valid at any time.
ADC/DAC quantization and output-referred noise are NOT modelled by the thresholds (the
experiments switch them off); this is the "calibrated on a clean tile at fixed false alarm"
discipline of calibration/evaluation separation, never tuned on the evaluation faults.

``fa`` = expected number of false-alarm decisions per decision family per localize() call (default 0.1).
"""

from __future__ import annotations

import math

import torch
from scipy.stats import chi2, norm

from neurofault.spectral.types import (
    DTYPE,
    FaultEstimate,
    ReadCost,
    TileReaderLike,
    make_generator,
)

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)


def z_two_sided(n_tests: int, fa: float = 0.5) -> float:
    """Gaussian threshold giving ``fa`` expected two-sided false alarms over ``n_tests``."""
    p = min(max(fa / max(n_tests, 1), 1e-300), 0.999)
    return float(norm.isf(p / 2.0))


def chi2_threshold(df: int, n_tests: int, fa: float = 0.5) -> float:
    p = min(max(fa / max(n_tests, 1), 1e-300), 0.999)
    return float(chi2.isf(p, df))


def rademacher(n: int, p: int, gen: torch.Generator) -> torch.Tensor:
    return (torch.randint(0, 2, (n, p), generator=gen) * 2 - 1).to(DTYPE)


def cells_to_estimate(
    mask: torch.Tensor, score: torch.Tensor | None = None, **meta
) -> FaultEstimate:
    """(m, n) bool mask -> FaultEstimate with cells, rows, cols (and score if given)."""
    cells = mask.nonzero()
    sc = None if score is None else score[mask]
    return FaultEstimate(
        cells=cells, rows=cells[:, 0].unique(), cols=cells[:, 1].unique(), score=sc, meta=meta
    )


def empty_estimate(**meta) -> FaultEstimate:
    z = torch.zeros(0, dtype=torch.long)
    return FaultEstimate(cells=torch.zeros(0, 2, dtype=torch.long), rows=z, cols=z, meta=meta)


class RefLocalizer:
    """Base: stores w_ref, the noise level and a probe RNG; implements calibrate()."""

    name = "ref"

    def __init__(
        self,
        w_ref: torch.Tensor,
        *,
        sigma: float | None = None,
        seed: int = 0,
        fa: float = 0.1,
        n_calib: int = 4,
    ) -> None:
        self.w_ref = w_ref.to(DTYPE)
        self.sigma = sigma
        self.fa = fa
        self.n_calib = n_calib
        self.gen = make_generator(seed, f"localize_{self.name}")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.w_ref.shape)

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        """Estimate the white read-noise level; costs 2*n_calib forward reads (0 if preset)."""
        if self.sigma is not None:
            return ReadCost()
        before = reader.cost.copy()
        n = self.shape[1]
        x = rademacher(n, self.n_calib, self.gen)
        d = reader.mvm(x) - reader.mvm(x)
        self.sigma = float(d.pow(2).mean().sqrt() / math.sqrt(2.0) / math.sqrt(n))
        return reader.cost - before

    def _need_sigma(self) -> float:
        if self.sigma is None:
            raise RuntimeError(f"{self.name}: call calibrate() first or pass sigma=")
        return self.sigma

    # residual probes (counted through the reader)
    def res_fwd(self, reader: TileReaderLike, x: torch.Tensor) -> torch.Tensor:
        return reader.mvm(x) - self.w_ref @ x

    def res_bwd(self, reader: TileReaderLike, y: torch.Tensor) -> torch.Tensor:
        return reader.mvm_t(y) - self.w_ref.T @ y


# ------------------------------------------------------------------ group-testing designs


def kautz_singleton(n_items: int, d: int) -> torch.Tensor:
    """0/1 test matrix (tests x items), d-disjunct (Kautz-Singleton / Reed-Solomon code).

    Items are polynomials of degree < l over GF(q) (q prime); test (a, v) contains the items
    whose polynomial takes value v at point a. Column weight q, any two columns share at most
    l-1 tests, so the matrix is d-disjunct when q - 1 >= d (l - 1). This is the class of sparse
    0-1 disjunct parity-check matrices Wei & Roth (arXiv 2402.13503) use for multi-outlier
    location. Picks the (q, l) with the fewest tests; falls back to the identity (one test per
    item, exact for any d) when no design has fewer tests than items."""
    best = None
    for q in _PRIMES:
        for ell in range(2, 9):
            if q**ell >= n_items and q - 1 >= d * (ell - 1):
                if best is None or q * q < best[0] * best[0]:
                    best = (q, ell)
                break
    if best is None or best[0] ** 2 >= n_items:
        return torch.eye(n_items, dtype=torch.uint8)
    q, ell = best
    idx = torch.arange(n_items)
    coef = torch.stack([(idx // q**i) % q for i in range(ell)], dim=1)  # (items, ell)
    a = torch.arange(q)
    val = torch.zeros(n_items, q, dtype=torch.long)
    for i in range(ell):
        val = (val + coef[:, i : i + 1] * (a[None, :] ** i)) % q
    rows = a[None, :] * q + val  # (items, q) test index
    M = torch.zeros(q * q, n_items, dtype=torch.uint8)
    M[rows.T.reshape(-1), idx.repeat(q)] = 1
    return M[M.sum(1) > 0]


def comp_dd(pos: torch.Tensor, M: torch.Tensor, *, tol: int = 0, dd: bool = True) -> torch.Tensor:
    """Group-testing decode, batched over leading rows.

    pos: (B, P) bool test outcomes; M: (P, N) 0/1 design. COMP: item j is a candidate iff at
    most ``tol`` of the tests containing it are negative. DD (definite defectives, optional):
    among candidates keep those that are the only candidate of some positive test. Returns
    (B, N) bool."""
    Mf = M.to(DTYPE)
    neg = (~pos).to(DTYPE)
    cand = (neg @ Mf) <= tol
    if not dd:
        return cand
    cnt = cand.to(DTYPE) @ Mf.T  # candidates per test
    sole = (pos & (cnt == 1)).to(DTYPE) @ Mf > 0
    return cand & sole


# ------------------------------------------------------------------ sparse recovery


def omp(
    A: torch.Tensor,
    y: torch.Tensor,
    noise_sd: float,
    *,
    fa: float,
    max_nnz: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OMP for y = A e + N(0, noise_sd^2). Stops when ||residual||^2 passes a chi^2 test.

    A: (K, N), y: (K,). Returns (support indices, coefficients) by least squares."""
    K, N = A.shape
    thr = noise_sd**2 * chi2.isf(min(max(fa, 1e-300), 0.999), K)
    nrm = A.norm(dim=0).clamp_min(1e-12)
    max_nnz = min(max_nnz or K - 1, K, N)
    sel: list[int] = []
    r = y.clone()
    coef = torch.zeros(0, dtype=DTYPE)
    while float(r @ r) > thr and len(sel) < max_nnz:
        c = (A.T @ r).abs() / nrm
        if sel:
            c[sel] = -1.0
        j = int(c.argmax())
        sel.append(j)
        coef = torch.linalg.lstsq(A[:, sel], y[:, None]).solution[:, 0]
        r = y - A[:, sel] @ coef
    return torch.tensor(sel, dtype=torch.long), coef
