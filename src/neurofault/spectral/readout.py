"""Counted, noisy access to one tile.

Detectors, localizers and compensators only ever see a TileReader, never the
matrix, so a method cannot read the tile for free. Every call adds to
``reader.cost`` with exact arithmetic:

    mvm (n,p)->(m,p):    mvm_fwd += p, adc += m*p, dac += n*p
    mvm_t (m,p)->(n,p):  mvm_bwd += p, adc += n*p, dac += m*p
    full_readout:        cell_reads += m*n

(The plan only fixes the mvm_fwd/ADC/DAC counts for the forward path; DAC for
the backward path counts the m*p driven entries of y by symmetry.)

Noise model per call (``ReadNoise``):
- per_cell: every cell gets fresh N(0, (sigma_cell_frac * max|w|)^2) on every MVM
  (weight units, max over the CURRENT decoded tile). For a white per-cell model
  this is distributionally identical to adding N(0, sigma^2 ||x_j||^2) to each
  output of column j, which is what is done (no (m,n,p) tensor). With
  persistence='frozen' one additive noise matrix is drawn on first use and
  re-used for every later call; with one_over_f=True a noisy matrix is drawn
  once per call and shared by the p columns of that call.
- output_referred: N(0, (sigma_out_frac * rms(y))^2) on the pre-ADC output, rms
  taken over the call's noiseless-by-this-term result.
- both: the two above.
- DAC/ADC: symmetric uniform quantization with 2^(bits-1)-1 positive levels;
  the range is the abs-max of the data of that call (inputs: per call; outputs:
  the probe output abs-max of that call, so the ADC never clips). None = off.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from neurofault.spectral.noise import read_noise_cell
from neurofault.spectral.types import DTYPE, ConductanceState, ReadCost, ReadNoise


def quantize(x: torch.Tensor, bits: int | None) -> torch.Tensor:
    """Symmetric uniform quantizer with range = max|x| of this call."""
    if bits is None:
        return x
    amax = float(x.abs().max())
    if amax == 0.0:
        return x
    step = amax / (2 ** (bits - 1) - 1)
    return torch.round(x / step) * step


class TileReader:
    """TileReaderLike over a (possibly time-varying) ConductanceState."""

    def __init__(
        self,
        state_provider: Callable[[], ConductanceState],
        noise: ReadNoise,
        gen: torch.Generator,
        cost: ReadCost | None = None,
        backward_supported: bool = True,
    ) -> None:
        self._provider = state_provider
        self.noise = noise
        self.gen = gen
        self.cost = ReadCost() if cost is None else cost
        self.backward_supported = backward_supported
        self._frozen: torch.Tensor | None = None  # additive cell noise when persistence=frozen

    @property
    def shape(self) -> tuple[int, int]:
        return self._provider().shape

    # ------------------------------------------------------------------ internals

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, dtype=DTYPE)

    def _matrix(self, state: ConductanceState) -> tuple[torch.Tensor, float]:
        """(W as read this call, residual white sigma to apply analytically)."""
        nz = self.noise
        w = state.weights()
        white = nz.mode in ("per_cell", "both") and nz.sigma_cell_frac > 0.0
        sigma = nz.sigma_cell_frac * float(w.abs().max()) if white else 0.0
        if nz.persistence == "frozen":
            if self._frozen is None or self._frozen.shape != w.shape:
                self._frozen = read_noise_cell(state, nz, state.t, self.gen) - w
            return w + self._frozen, 0.0
        if nz.one_over_f:
            return read_noise_cell(state, nz, state.t, self.gen), 0.0
        return w, sigma

    def _apply(self, w: torch.Tensor, sigma: float, z: torch.Tensor) -> torch.Tensor:
        """Shared forward/backward path: z is (k, p) -> w @ z (rows, p)."""
        zq = quantize(z, self.noise.dac_bits)
        y = w @ zq
        if sigma > 0.0:
            y = y + sigma * zq.norm(dim=0) * self._randn(*y.shape)
        if self.noise.mode in ("output_referred", "both") and self.noise.sigma_out_frac > 0.0:
            rms = float(y.pow(2).mean().sqrt())
            y = y + self.noise.sigma_out_frac * rms * self._randn(*y.shape)
        return quantize(y, self.noise.adc_bits)

    # ------------------------------------------------------------------ public

    def mvm(self, x: torch.Tensor) -> torch.Tensor:
        """y = W x for x (n, p); returns (m, p)."""
        state = self._provider()
        m, n = state.shape
        x = x.to(DTYPE)
        assert x.ndim == 2 and x.shape[0] == n, f"x must be ({n}, p), got {tuple(x.shape)}"
        p = x.shape[1]
        w, sigma = self._matrix(state)
        y = self._apply(w, sigma, x)
        self.cost.mvm_fwd += p
        self.cost.adc += m * p
        self.cost.dac += n * p
        return y

    def mvm_t(self, y: torch.Tensor) -> torch.Tensor:
        """W^T y for y (m, p); returns (n, p). Raises if backward is unsupported."""
        if not self.backward_supported:
            raise RuntimeError("this TileReader does not support backward (transpose) MVMs")
        state = self._provider()
        m, n = state.shape
        y = y.to(DTYPE)
        assert y.ndim == 2 and y.shape[0] == m, f"y must be ({m}, p), got {tuple(y.shape)}"
        p = y.shape[1]
        w, sigma = self._matrix(state)
        out = self._apply(w.T, sigma, y)
        self.cost.mvm_bwd += p
        self.cost.adc += n * p
        self.cost.dac += m * p
        return out

    def full_readout(self) -> torch.Tensor:
        """Noisy (m, n) readout of every cell: the ceiling cost, m*n cell reads.

        Cell noise as in mvm (fresh per call unless frozen); output_referred noise
        is applied per cell as sigma_out_frac * rms(W). No ADC/DAC quantization
        (cells are read individually, not through the MVM chain)."""
        state = self._provider()
        m, n = state.shape
        w, sigma = self._matrix(state)
        if sigma > 0.0:
            w = w + sigma * self._randn(m, n)
        if self.noise.mode in ("output_referred", "both") and self.noise.sigma_out_frac > 0.0:
            w = w + self.noise.sigma_out_frac * float(w.pow(2).mean().sqrt()) * self._randn(m, n)
        self.cost.cell_reads += m * n
        return w
