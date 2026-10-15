"""B1 global drift compensation: one scalar alpha for the whole tile.

WHY: GDC is the incumbent every drift paper beats (Joshi 2020) and it is exact for uniform
power-law decay (D1). Variants differ only in the readout that defines alpha:
- aihwkit (B1b): reads eye(n); alpha = clamp(mean|Y0|) / clamp(mean|Yt|) - the installed
  aihwkit 1.1.0 GlobalDriftCompensation (clamp 1e-4 in ITS normalised weight units; ours are
  checkpoint units, so ``eps`` is a parameter, default the source's 1e-4).
- ones (B1c): a single all-ones probe; mean|y| suffers sign cancellation with signed weights.
- ls: P Rademacher probes, Frobenius-optimal scalar (what a noise-aware designer would do).
- exact_ref (B1d): baseline from the ideal W0 (bound), not a noisy t0 read.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from neurofault.spectral.baselines.base import (
    Bindable,
    Correction,
    ReadCost,
    TileReaderLike,
    apply_gains,
    ls_gain,
    make_probes,
)
from neurofault.spectral.types import DTYPE, make_generator

_VARIANTS = ("aihwkit", "ones", "ls", "exact_ref")


class GDC(Bindable):
    def __init__(self, variant: str = "aihwkit", p: int = 4, eps: float = 1e-4, seed: int = 0):
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        self.variant, self.p, self.eps = variant, p, eps
        self.gen = make_generator(seed, f"gdc_probes_{variant}")
        self.name = f"B1_gdc_{variant}"
        self.x: torch.Tensor | None = None
        self.y0: torch.Tensor | None = None
        self.ref_val: float | None = None

    def _probes(self, n: int) -> torch.Tensor:
        kind = {"aihwkit": "identity", "exact_ref": "identity", "ones": "ones"}.get(
            self.variant, "rademacher"
        )
        return make_probes(kind, n, self.p, self.gen)

    def _readout(self, y: torch.Tensor) -> float:
        return float(y.abs().mean().clamp(min=self.eps))

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        _, n = reader.shape
        self.x = self._probes(n)
        if self.variant == "exact_ref":
            self.ref_val = float(self.ref().abs().mean().clamp(min=self.eps))
        else:
            self.y0 = reader.mvm(self.x)
            self.ref_val = self._readout(self.y0)
        return reader.cost - start

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        assert self.x is not None, "calibrate() first"
        start = reader.cost.copy()
        yt = reader.mvm(self.x)
        if self.variant == "ls":
            alpha = float(ls_gain(self.y0, yt))
        else:
            alpha = self.ref_val / self._readout(yt)
        return Correction(apply_gains(self.now(), alpha), reader.cost - start, {"alpha": alpha})


class GDCJoshi(Bindable):
    """B1a: L reference columns programmed at a known conductance, one read per update.

    alpha = I_ref(t0) / I_ref(t) at constant V_cal. The reference devices are not part of the
    TileReader, so the harness binds ``ref_current(t)`` (sum of reference-device currents, which
    drift under the SAME model). Counted as one forward MVM over L extra columns."""

    def __init__(self, n_ref: int = 8):
        self.n_ref = n_ref
        self.name = f"B1a_gdc_joshi_L{n_ref}"
        self._ref_current: Callable[[float], float] | None = None
        self._i0: float | None = None

    def bind_reference(self, ref_current: Callable[[float], float]):
        self._ref_current = ref_current
        return self

    def _read(self, reader: TileReaderLike, t: float) -> ReadCost:
        assert self._ref_current is not None, "bind_reference() first"
        reader.cost.mvm_fwd += 1
        reader.cost.adc += self.n_ref
        reader.cost.dac += 1
        return ReadCost(mvm_fwd=1, adc=self.n_ref, dac=1)

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        c = self._read(reader, 0.0)
        self._i0 = float(self._ref_current(0.0))
        return c

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        c = self._read(reader, t)
        alpha = self._i0 / float(self._ref_current(t))
        return Correction(apply_gains(self.now(), alpha), c, {"alpha": alpha})


_ = DTYPE
