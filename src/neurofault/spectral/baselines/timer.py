"""B6 open-loop timers: no reads (TimerSchedule) or periodic recalibration (FixedPeriodRecal).

WHY: the claim "sensing beats a timer" needs a timer that is as good as an open-loop designer
can make it: the nominal drift model predicts alpha(t); when the TRUE drift deviates (different
nu, temperature) the timer is blind and the error shows up. mismatch scales the true nu
relative to the nominal one only in the harness; here alpha uses the nominal model.
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
)
from neurofault.spectral.params import val
from neurofault.spectral.types import T0_S, ConductanceState, DriftModel, make_generator


class TimerSchedule(Bindable):
    def __init__(self, alpha_fn: Callable[[float], float], name: str = "B6_timer"):
        self.alpha_fn = alpha_fn
        self.name = name

    @classmethod
    def power_law(cls, nu: float | None = None, t0: float = T0_S) -> TimerSchedule:
        """Uniform D1 model: alpha(t) = ((t + t0)/t0)^nu undoes G ~ ((t+t0)/t0)^-nu."""
        nu = val("d1_nu") if nu is None else nu
        return cls(lambda t: ((t + t0) / t0) ** nu, f"B6_timer_nu{nu:g}")

    @classmethod
    def from_model(
        cls, model: DriftModel, state0: ConductanceState, w0: torch.Tensor, seed: int = 0
    ) -> TimerSchedule:
        """Nominal-model alpha(t) = <W0, W0> / <W0, W_nom(t)> from one Monte-Carlo draw."""
        real = model.sample(state0.clone(), make_generator(seed, "timer_nominal"))
        denom0 = float((w0 * w0).sum())

        def fn(t: float) -> float:
            return denom0 / float((w0 * real.at(t).weights()).sum())

        return cls(fn, f"B6_timer_{model.name}")

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        return ReadCost()

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        a = float(self.alpha_fn(t))
        return Correction(apply_gains(self.now(), a), ReadCost(), {"alpha": a})


class FixedPeriodRecal(Bindable):
    """Re-run ``inner`` every ``period_s`` seconds; in between hold its last gains.

    Gain-type inner results (meta has alpha/row_gain/col_gain) are re-applied to the CURRENT
    W(t); matrix-type ones (full readout) are held stale. Held steps cost nothing."""

    def __init__(self, inner, period_s: float):
        self.inner, self.period_s = inner, period_s
        self.name = f"B6_period{period_s:g}_{inner.name}"
        self._last_t = 0.0
        self._held: Correction | None = None

    def bind(self, w_now, w0=None):
        super().bind(w_now, w0)
        if hasattr(self.inner, "bind"):
            self.inner.bind(w_now, w0)
        return self

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        self._last_t, self._held = 0.0, None
        return self.inner.calibrate(reader)

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        if self._held is None and t - self._last_t < self.period_s:
            return Correction(self.now().clone(), ReadCost(), {"recal": 0})
        if self._held is None or t - self._last_t >= self.period_s:
            self._held = self.inner.update(reader, t)
            self._last_t = t
            return Correction(self._held.W_eff, self._held.cost, {**self._held.meta, "recal": 1})
        meta = self._held.meta
        if any(k in meta for k in ("alpha", "row_gain", "col_gain")):
            w = apply_gains(
                self.now(), meta.get("alpha", 1.0), meta.get("row_gain"), meta.get("col_gain")
            )
        else:
            w = self._held.W_eff
        return Correction(w, ReadCost(), {**meta, "recal": 0})
