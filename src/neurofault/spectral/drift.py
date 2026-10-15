"""Drift models D0-D7 as sample-once / evaluate-anywhere realizations.

WHY this shape: every compensation experiment needs the SAME physical tile
evaluated at many times (t = 60 s ... 10 yr). A device's drift exponent is a
property of the device, so all per-device randomness is drawn once in
`DriftModel.sample(state0, gen)` and `realization.at(t)` is a deterministic
function of that draw - never redrawn per time point (which would turn
persistent device-to-device variation into fake read noise and wreck any
estimator that learns the pattern at t1 and applies it at t2).

Conventions
- state0 is the freshly programmed tile (time 0, t0 = 20 s default). `at(t)` is
  seconds since programming completed; power laws use ((t + t0) / t0)^-nu so
  at(0) is the identity - AIHWKit's own convention (t = t_inference + t_0).
- Drift acts per DEVICE: both G+ and G- of a pair (they drift whenever g_min > 0
  or either is non-zero); offset_refcol drifts only g_pos (the reference is
  shared, fixed hardware); signed_ideal drifts the signed value multiplicatively.
- Stuck devices (state0.stuck) never drift. Physical encodings are clamped to
  [g_min, g_max]; signed_ideal has no rails and is not clamped.
- Read-path effects (D4 gain, D7 periphery gain) are folded into the effective
  conductances and are NOT rail-clamped (a read gain is not a stored state);
  D7 also acts on stuck devices (it sits after the array), D4 gain does not.
"""

from __future__ import annotations

import math

import torch

from neurofault.faults.injection import _select_fault_indices
from neurofault.faults.reliability import arrhenius_acceleration_factor
from neurofault.spectral.params import val
from neurofault.spectral.types import (
    DTYPE,
    T0_S,
    ConductanceState,
    DriftModel,
    DriftRealization,
)

# --------------------------------------------------------------------------- helpers


def _stack(state: ConductanceState) -> torch.Tensor:
    """(m, n, D) device conductances: D = 2 for diff_pair, else 1."""
    if state.g_neg is not None:
        return torch.stack([state.g_pos, state.g_neg], dim=-1)
    return state.g_pos.unsqueeze(-1)


def _unstack(state0: ConductanceState, G: torch.Tensor, t: float) -> ConductanceState:
    out = state0.clone(t=float(t))
    out.g_pos = G[..., 0].clone()
    if state0.g_neg is not None:
        out.g_neg = G[..., 1].clone()
    return out


def _randn(shape, gen: torch.Generator, like: torch.Tensor) -> torch.Tensor:
    return torch.randn(shape, generator=gen, device=gen.device, dtype=DTYPE).to(like.device)


def _rand(shape, gen: torch.Generator, like: torch.Tensor) -> torch.Tensor:
    return torch.rand(shape, generator=gen, device=gen.device, dtype=DTYPE).to(like.device)


def pcm_nu_stats(G: torch.Tensor, g_max_uS: float) -> tuple[torch.Tensor, torch.Tensor]:
    """AIHWKit PCMLikeNoiseModel (mu_nu, sigma_nu) as a function of conductance
    (pcm.py generate_drift_coefficients; natural log; g/g_max floored at 1e-7)."""
    gr = (G.abs() / g_max_uS).clamp(min=val("d2_g_rel_floor"))
    lo, hi = val("d2_mu_clip")
    mu = (val("d2_mu_slope") * torch.log(gr) + val("d2_mu_intercept")).clamp(lo, hi)
    lo, hi = val("d2_sigma_clip")
    sig = (val("d2_sigma_slope") * torch.log(gr) + val("d2_sigma_intercept")).clamp(lo, hi)
    return mu, sig


class _Real:
    """Base realization: subclasses implement `_evolve(G, t)` on the (m, n, D) stack."""

    clamp = True
    exclude_stuck = True

    def __init__(self, state0: ConductanceState):
        self.state0 = state0
        self.stuck = state0.stuck[..., : 2 if state0.g_neg is not None else 1].clone()
        self.lo = state0.enc.g_min_uS
        self.hi = state0.enc.g_max_uS
        self.physical = state0.enc.kind != "signed_ideal"

    def _evolve(self, G: torch.Tensor, t: float) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, G: torch.Tensor, t: float) -> torch.Tensor:
        Gn = self._evolve(G, t)
        if self.clamp and self.physical:
            Gn = Gn.clamp(self.lo, self.hi)
        if self.exclude_stuck:
            Gn = torch.where(self.stuck, G, Gn)
        return Gn

    def at(self, t: float) -> ConductanceState:
        return _unstack(self.state0, self.apply(_stack(self.state0), t), t)

    def sweep(self, ts: list[float]) -> list[ConductanceState]:
        return [self.at(t) for t in ts]


# --------------------------------------------------------------------------- D0


class LegacyRailShift:
    """D0: the repo's `injection._drift` (a one-shot shift of a random subset of
    cells by drift_magnitude * g_range toward a rail). Regression/parity only.

    A step at t > 0 (identity at t = 0). Selection and sign use the same torch
    calls in the same order as `_drift`, so with an identically seeded generator
    on a single-device encoding (offset_refcol) the result matches it. With two
    devices each plane gets an independent selection (plane 0 first)."""

    name = "D0_legacy_rail_shift"

    def __init__(
        self,
        density: float = 0.1,
        drift_magnitude: float | None = None,
        direction: str = "toward_hrs",
        distribution: str = "random",
    ):
        self.density = density
        self.mag = val("d0_drift_magnitude") if drift_magnitude is None else drift_magnitude
        self.direction = direction
        self.distribution = distribution

    def sample(self, state0: ConductanceState, gen: torch.Generator) -> DriftRealization:
        m, n = state0.shape
        D = 2 if state0.g_neg is not None else 1
        delta = torch.zeros(m, n, D, dtype=DTYPE, device=state0.g_pos.device)
        g_range = state0.enc.g_max_uS - state0.enc.g_min_uS
        count = int(m * n * self.density)
        for d in range(D if count > 0 else 0):
            flat = _select_fault_indices(m, n, count, self.distribution, gen.device, gen)
            if self.direction == "toward_hrs":
                sign = -torch.ones(len(flat), dtype=DTYPE, device=gen.device)
            elif self.direction == "toward_lrs":
                sign = torch.ones(len(flat), dtype=DTYPE, device=gen.device)
            else:  # random_per_cell: decided once per cell
                coin = torch.rand(len(flat), device=gen.device, generator=gen)
                sign = torch.where(coin < 0.5, -1.0, 1.0).to(DTYPE)
            r, c = (flat // n).to(delta.device), (flat % n).to(delta.device)
            delta[r, c, d] = sign.to(delta.device) * self.mag * g_range
        return _LegacyReal(state0, delta)


class _LegacyReal(_Real):
    def __init__(self, state0: ConductanceState, delta: torch.Tensor):
        super().__init__(state0)
        self.delta = delta

    def _evolve(self, G: torch.Tensor, t: float) -> torch.Tensor:
        return G + self.delta if t > 0 else G


# --------------------------------------------------------------------------- D1 / D2 / D3


class _PowerLawReal(_Real):
    def __init__(self, state0: ConductanceState, nu: torch.Tensor, t0: float):
        super().__init__(state0)
        self.nu = nu
        self.t0 = t0

    def _evolve(self, G: torch.Tensor, t: float) -> torch.Tensor:
        return G * (((t + self.t0) / self.t0) ** (-self.nu))


class PowerLawUniform:
    """D1: G(t) = G0 ((t + t0)/t0)^-nu with one nu for every device (Joshi 2020)."""

    name = "D1_powerlaw_uniform"

    def __init__(self, nu: float | None = None, t0: float = T0_S):
        self.nu = val("d1_nu") if nu is None else nu
        self.t0 = t0

    def sample(self, state0: ConductanceState, gen: torch.Generator) -> DriftRealization:
        nu = torch.full((*state0.shape, 2 if state0.g_neg is not None else 1), self.nu, dtype=DTYPE)
        return _PowerLawReal(state0, nu.to(state0.g_pos.device), self.t0)


class PCMPerDevice:
    """D2: IBM/AIHWKit per-device exponent nu = |mu(g) + sigma(g) z|, z ~ N(0,1) drawn
    ONCE per device from the programmed conductance (both G+ and G-)."""

    name = "D2_pcm_per_device"

    def __init__(self, t0: float = T0_S, drift_scale: float = 1.0):
        self.t0 = t0
        self.drift_scale = drift_scale

    def _nu(self, G0: torch.Tensor, g_max: float, z: torch.Tensor, extra: torch.Tensor | None):
        mu, sig = pcm_nu_stats(G0, g_max)
        shift = z if extra is None else z + extra
        return (mu + sig * shift).abs() * self.drift_scale

    def sample(self, state0: ConductanceState, gen: torch.Generator) -> DriftRealization:
        G0 = _stack(state0)
        z = _randn(G0.shape, gen, G0)
        nu = self._nu(G0, state0.enc.g_max_uS, z, None)
        return _PowerLawReal(state0, nu, self.t0)


def _standardize(f: torch.Tensor, center: bool = True) -> torch.Tensor:
    f = f - f.mean() if center else f
    s = f.pow(2).mean().sqrt() if not center else f.std(correction=0)
    return f / s if float(s) > 0 else torch.zeros_like(f)


def _ramp(k: int, sign: float) -> torch.Tensor:
    if k < 2:
        return torch.zeros(k, dtype=DTYPE)
    return sign * torch.linspace(-1.0, 1.0, k, dtype=DTYPE)


def structured_field(
    kind: str, m: int, n: int, gen: torch.Generator, rank_r: int = 2, width_frac: float = 0.2
) -> torch.Tensor:
    """Unit-variance spatial pattern f_ij (m, n) float64 for D3 / D4. kinds:
    row_gradient | col_gradient | rank_r | row_plus_col | hotspot.
    rank_r is a sum of r separable cosines (rank <= r exactly: scaled, not mean-centred)."""
    dev = gen.device
    sgn = lambda: 1.0 if float(torch.rand(1, generator=gen, device=dev)) < 0.5 else -1.0
    if kind == "row_gradient":
        f = _ramp(m, sgn())[:, None].expand(m, n)
        return _standardize(f.clone()).cpu()
    if kind == "col_gradient":
        f = _ramp(n, sgn())[None, :].expand(m, n)
        return _standardize(f.clone()).cpu()
    if kind == "row_plus_col":
        f = _ramp(m, sgn())[:, None] + _ramp(n, sgn())[None, :]
        return _standardize(f).cpu()
    if kind == "rank_r":
        i = (torch.arange(m, dtype=DTYPE) + 0.5) / m
        j = (torch.arange(n, dtype=DTYPE) + 0.5) / n
        f = torch.zeros(m, n, dtype=DTYPE)
        for _ in range(rank_r):
            fu, fv = (int(x) for x in torch.randint(1, 4, (2,), generator=gen, device=dev))
            pu, pv = (float(x) * 2 * math.pi for x in torch.rand(2, generator=gen, device=dev))
            a = float(torch.randn(1, generator=gen, device=dev))
            f += a * torch.outer(torch.cos(math.pi * fu * i + pu), torch.cos(math.pi * fv * j + pv))
        return _standardize(f, center=False)
    if kind == "hotspot":
        ci = float(torch.rand(1, generator=gen, device=dev)) * m
        cj = float(torch.rand(1, generator=gen, device=dev)) * n
        w = max(width_frac * max(m, n), 1e-6)
        ii = torch.arange(m, dtype=DTYPE)[:, None]
        jj = torch.arange(n, dtype=DTYPE)[None, :]
        return _standardize(torch.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2 * w * w)))
    raise ValueError(f"unknown field kind {kind!r}")


class StructuredNu(PCMPerDevice):
    """D3: D2 plus a spatially structured shift of the exponent,
    nu_ij = |mu(g) + sigma(g) (z + rho f_ij)|, f unit variance (so rho =
    sigma_struct / sigma_iid, the registered knob). z is drawn FIRST and identically to
    D2, so rho = 0 reproduces D2 exactly for the same seed. The field is shared by
    G+ and G- (it is a property of position on the die)."""

    def __init__(
        self,
        field: str = "rank_r",
        rho: float = 1.0,
        rank_r: int | None = None,
        width_frac: float | None = None,
        t0: float = T0_S,
        drift_scale: float = 1.0,
    ):
        super().__init__(t0, drift_scale)
        self.field = field
        self.rho = rho
        self.rank_r = val("d3_rank_r") if rank_r is None else rank_r
        self.width_frac = val("d3_hotspot_width_frac") if width_frac is None else width_frac
        self.name = f"D3_structured_{field}_rho{rho:g}"

    def sample(self, state0: ConductanceState, gen: torch.Generator) -> DriftRealization:
        G0 = _stack(state0)
        z = _randn(G0.shape, gen, G0)
        m, n = state0.shape
        f = structured_field(self.field, m, n, gen, self.rank_r, self.width_frac).to(G0.device)
        nu = self._nu(G0, state0.enc.g_max_uS, z, self.rho * f[..., None])
        real = _PowerLawReal(state0, nu, self.t0)
        real.field = f  # type: ignore[attr-defined]  # ground truth for localisation studies
        return real


# --------------------------------------------------------------------------- D4


class _ThermalReal(_Real):
    exclude_stuck = False  # stuck handled explicitly below; base stage handles its own

    def __init__(self, state0, base: _Real | None, af: float, gain: torch.Tensor):
        super().__init__(state0)
        self.base, self.af, self.gain = base, af, gain

    def apply(self, G: torch.Tensor, t: float) -> torch.Tensor:
        if self.base is not None:
            G = self.base.apply(G, self.af * t)
        return torch.where(self.stuck, G, G * self.gain[..., None])


class ThermalGain:
    """D4: (i) Arrhenius time acceleration t -> AF t of an optional base drift model
    (AF from faults.reliability.arrhenius_acceleration_factor), (ii) a read-time
    conductance gain 1 + beta_T dT_ij on non-stuck devices, dT a spatial temperature
    field. All parameters are ASSUMED (no verified source) - sweep them.
    dT_kind: uniform | row_gradient | hotspot | mixed (mean + row gradient + hotspot,
    each contributing amp/3 at its maximum)."""

    def __init__(
        self,
        base: DriftModel | None = None,
        T_K: float | None = None,
        T_ref_K: float | None = None,
        Ea_eV: float | None = None,
        beta_per_K: float | None = None,
        dT_amp_K: float | None = None,
        dT_kind: str = "mixed",
        width_frac: float | None = None,
    ):
        self.base = base
        self.T = val("d4_T_K") if T_K is None else T_K
        self.T_ref = val("d4_T_ref_K") if T_ref_K is None else T_ref_K
        self.Ea = val("d4_Ea_eV") if Ea_eV is None else Ea_eV
        self.beta = val("d4_beta_per_K") if beta_per_K is None else beta_per_K
        self.amp = val("d4_dT_amp_K") if dT_amp_K is None else dT_amp_K
        self.kind = dT_kind
        self.width_frac = val("d3_hotspot_width_frac") if width_frac is None else width_frac
        self.name = f"D4_thermal{'+' + base.name if base else ''}"

    def _dT(self, m: int, n: int, gen: torch.Generator) -> torch.Tensor:
        ones = torch.ones(m, n, dtype=DTYPE)
        ramp = (torch.arange(m, dtype=DTYPE) / max(m - 1, 1))[:, None] * ones
        bump = None
        if self.kind in ("hotspot", "mixed"):
            f = structured_field("hotspot", m, n, gen, width_frac=self.width_frac)
            # unit-peak bump (structured_field is standardised): recover 0..1 profile
            bump = (f - f.min()) / (f.max() - f.min()).clamp(min=1e-12)
        if self.kind == "uniform":
            return self.amp * ones
        if self.kind == "row_gradient":
            return self.amp * ramp
        if self.kind == "hotspot":
            return self.amp * bump
        if self.kind == "mixed":
            return self.amp * (ones + ramp + bump) / 3.0
        raise ValueError(f"unknown dT_kind {self.kind!r}")

    def sample(self, state0: ConductanceState, gen: torch.Generator) -> DriftRealization:
        base = self.base.sample(state0, gen) if self.base is not None else None
        m, n = state0.shape
        gain = (1.0 + self.beta * self._dT(m, n, gen)).to(state0.g_pos.device)
        af = arrhenius_acceleration_factor(self.T, self.T_ref, self.Ea)
        return _ThermalReal(state0, base, af, gain)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- D5


class _RelaxReal(_Real):
    def __init__(
        self, state0, mode: str, z: torch.Tensor | None, scale: float, c: float, t0: float
    ):
        super().__init__(state0)
        self.mode, self.z, self.scale, self.c, self.t0 = mode, z, scale, c, t0
        G0 = _stack(state0)
        self.gbar = G0.mean(dim=(0, 1), keepdim=True)

    def _evolve(self, G: torch.Tensor, t: float) -> torch.Tensor:
        if t <= 0:
            return G
        if self.mode == "cmo":
            # ReRamCMONoiseModel.apply_drift_noise_to_conductance, minus its read noise.
            # ln t floored at 0 (t < 1 s would flip the sign of the decay); the noise term
            # keeps its t = 1 s intercept for any t > 0, as in the source.
            lnt = math.log(max(t, 1.0))
            mean = val("d5a_decay_mean_uS")[0] * lnt * self.scale
            s1, s0 = val("d5a_decay_std")
            sigma = max(s1 * lnt + s0, 0.0) * self.scale
            return G + mean + self.z * sigma
        lam = min(max(self.c * math.log1p(t / self.t0), 0.0), 1.0)
        return G - lam * (G - self.gbar)


class RRAMRelaxation:
    """D5. mode='cmo': sourced AIHWKit ReRamCMONoiseModel relaxation (level-independent
    mean shift decay*ln t plus per-device Gaussian with sigma(t) = s1 ln t + s0, z drawn
    once so a device keeps its own relaxation direction across t; constants labelled
    IBM-internal fits 'in review' in the source docstring). mode='mean_revert': ASSUMED
    W <- W - lam(t)(W - w_bar), lam = c ln(1 + t/t0), applied per device toward the
    per-polarity tile mean (for pairs this is exactly weight-space reversion to w_bar)."""

    def __init__(
        self,
        mode: str = "cmo",
        drift_scale: float = 1.0,
        c: float | None = None,
        t0: float = T0_S,
    ):
        if mode not in ("cmo", "mean_revert"):
            raise ValueError(mode)
        self.mode, self.scale, self.t0 = mode, drift_scale, t0
        self.c = val("d5b_c") if c is None else c
        self.name = f"D5_rram_{mode}"

    def sample(self, state0: ConductanceState, gen: torch.Generator) -> DriftRealization:
        G0 = _stack(state0)
        z = _randn(G0.shape, gen, G0) if self.mode == "cmo" else None
        return _RelaxReal(state0, self.mode, z, self.scale, self.c, self.t0)


# --------------------------------------------------------------------------- D7


class _PeripheryReal(_Real):
    clamp = False
    exclude_stuck = False

    def __init__(self, state0, z: torch.Tensor, sigma_p: float, t0: float):
        super().__init__(state0)
        self.z, self.sigma_p, self.t0 = z, sigma_p, t0

    def _evolve(self, G: torch.Tensor, t: float) -> torch.Tensor:
        growth = math.sqrt(max(math.log((t + self.t0) / self.t0), 0.0))
        return G * (1.0 + self.sigma_p * self.z * growth)


class PeripheryGain:
    """D7: per-line gain 1 + sigma_p z sqrt(ln((t+t0)/t0)). axis='out' scales rows of W
    (per-output TIA gain, the crossbar column); axis='in' scales columns of W (per-input
    DAC gain). Folded into both devices of a pair so W scales exactly; ASSUMED."""

    def __init__(self, sigma_p: float | None = None, axis: str = "out", t0: float = T0_S):
        if axis not in ("out", "in"):
            raise ValueError(axis)
        self.sigma_p = val("d7_sigma_p") if sigma_p is None else sigma_p
        self.axis, self.t0 = axis, t0
        self.name = f"D7_periphery_{axis}"

    def sample(self, state0: ConductanceState, gen: torch.Generator) -> DriftRealization:
        m, n = state0.shape
        G0 = _stack(state0)
        if self.axis == "out":
            z = _randn((m, 1, 1), gen, G0)
        else:
            z = _randn((1, n, 1), gen, G0)
        return _PeripheryReal(state0, z, self.sigma_p, self.t0)


# --------------------------------------------------------------------------- composition


class _ComposedReal(_Real):
    exclude_stuck = False
    clamp = False

    def __init__(self, state0: ConductanceState, parts: list):
        super().__init__(state0)
        self.parts = parts

    def apply(self, G: torch.Tensor, t: float) -> torch.Tensor:
        for p in self.parts:
            G = p.apply(G, t)
        return G


class ComposedDrift:
    """Apply several drift models in sequence at the same t (e.g. D2 then D7). Every
    part samples from the same generator in list order, so the whole composition is
    one replayable draw."""

    def __init__(self, models: list, name: str | None = None):
        self.models = list(models)
        self.name = name or "+".join(m.name for m in self.models)

    def sample(self, state0: ConductanceState, gen: torch.Generator) -> DriftRealization:
        return _ComposedReal(state0, [m.sample(state0, gen) for m in self.models])
