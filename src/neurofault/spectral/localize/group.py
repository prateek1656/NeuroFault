"""A6 group testing: sparse-weight 0/1 probes, 1-bit thresholded outputs, COMP / DD decode.

A random design with constant column weight puts every item in ``c = round(T w / N)`` of the T
tests (w = active inputs per probe, the quantity E1b sweeps). A test is positive when the
residual of its output exceeds a threshold:
- ``thresh="np"``: z * sd at fixed false alarm over all tests (strict), sd = sigma*sqrt(w_test)
  for white per-cell noise (``noise_model="cell"``) or the probe-independent output floor
  sigma*sqrt(n_items) (``"floor"``, the TIA/ADC model of C2);
- ``thresh="half"``: half the design deviation ``delta_frac * max|w_ref|`` (noise-tolerant).
COMP keeps an item when at most ``floor(tol_frac * deg)`` of its tests are negative (tol_frac = 0
is plain COMP); ``decoder="dd"`` additionally keeps only items that are the sole candidate of a
positive test. Bidirectional: the same is done with backward probes over rows; a cell is declared
only when both directions agree (``combine="and"``) or either does (``"or"``). Reads are exactly
t_fwd + t_bwd.
"""

from __future__ import annotations

import math

import torch

from neurofault.spectral.localize.common import RefLocalizer, z_two_sided
from neurofault.spectral.types import DTYPE, FaultEstimate, ReadCost, TileReaderLike


def random_design(n_tests: int, n_items: int, weight: int, gen: torch.Generator) -> torch.Tensor:
    """(T, N) uint8, every item in c = clip(round(T*weight/N), 1, T) distinct random tests."""
    c = int(min(max(round(n_tests * weight / n_items), 1), n_tests))
    idx = torch.rand(n_tests, n_items, generator=gen).argsort(0)[:c]
    return torch.zeros(n_tests, n_items, dtype=torch.uint8).scatter_(0, idx, 1)


def tolerant_decode(
    pos: torch.Tensor, M: torch.Tensor, *, tol_frac: float, dd: bool
) -> torch.Tensor:
    """pos (B, T) bool, M (T, N) 0/1 -> (B, N) bool candidates."""
    Mf = M.to(DTYPE)
    deg = Mf.sum(0)
    cand = ((~pos).to(DTYPE) @ Mf) <= torch.floor(tol_frac * deg)[None, :] + 1e-9
    if not dd:
        return cand
    cnt = cand.to(DTYPE) @ Mf.T
    return cand & (((pos & (cnt == 1)).to(DTYPE) @ Mf) > 0)


class GroupTestingLocalizer(RefLocalizer):
    name = "A6_group_testing"

    def __init__(
        self,
        w_ref,
        *,
        t_fwd: int = 32,
        t_bwd: int = 0,
        weight: int | None = None,
        thresh: str = "np",
        tol_frac: float = 0.0,
        decoder: str = "comp",
        delta_frac: float = 0.25,
        combine: str = "and",
        noise_model: str = "cell",
        **kw,
    ) -> None:
        super().__init__(w_ref, **kw)
        assert noise_model in ("cell", "floor")
        assert thresh in ("np", "half") and decoder in ("comp", "dd") and combine in ("and", "or")
        n = self.shape[1]
        self.t_fwd, self.t_bwd = t_fwd, t_bwd
        self.weight = weight if weight is not None else max(n // 2, 1)
        self.thresh, self.tol_frac, self.decoder = thresh, tol_frac, decoder
        self.delta = delta_frac * float(self.w_ref.abs().max())
        self.combine = combine
        self.noise_model = noise_model

    def _side(self, reader, n_items: int, n_lines: int, T: int, w: int, transpose: bool):
        """Run T tests over ``n_items`` probe entries; decode each of the ``n_lines`` outputs."""
        sigma = max(self._need_sigma(), 1e-12)
        M = random_design(T, n_items, w, self.gen)
        X = M.T.to(DTYPE)
        r = self.res_bwd(reader, X) if transpose else self.res_fwd(reader, X)  # (lines, T)
        wt = M.sum(1).to(DTYPE).clamp_min(1.0)
        if self.thresh == "np":
            z = z_two_sided(n_lines * T, self.fa)
            # cell: white per-cell noise, sd sigma*sqrt(w_test); floor: probe-independent output
            # noise, sd sigma*sqrt(#items) (calibrate() measures it on a dense probe)
            sd = sigma * wt.sqrt() if self.noise_model == "cell" else sigma * math.sqrt(n_items)
            tau = z * (sd if torch.is_tensor(sd) else torch.full_like(wt, sd))
        else:
            tau = torch.full_like(wt, self.delta / 2)
        pos = r.abs() > tau[None, :]
        return tolerant_decode(pos, M, tol_frac=self.tol_frac, dd=self.decoder == "dd")

    def localize(self, reader: TileReaderLike, t: float = 0.0) -> tuple[FaultEstimate, ReadCost]:
        before = reader.cost.copy()
        m, n = self.shape
        mask = self._side(reader, n, m, self.t_fwd, self.weight, False)  # (m, n)
        bidir = self.t_bwd > 0 and reader.backward_supported
        if bidir:
            wb = max(1, round(self.weight * m / n))
            bm = self._side(reader, m, n, self.t_bwd, wb, True).T  # (m, n)
            mask = (mask & bm) if self.combine == "and" else (mask | bm)
        cells = mask.nonzero()
        est = FaultEstimate(
            cells=cells,
            rows=cells[:, 0].unique(),
            cols=cells[:, 1].unique(),
            meta={"bidir": bidir, "weight": self.weight, "tests": self.t_fwd + self.t_bwd},
        )
        return est, reader.cost - before


def optimal_weight_dense_vs_sparse(n: int, k_line: float) -> float:
    """Group-testing rule of thumb for s defects per line: inclusion p ~ ln2 / s (w = p n)."""
    return n * math.log(2.0) / max(k_line, 1.0)
