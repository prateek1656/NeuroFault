"""Programming and read noise.

The 'aihwkit_pcm_poly' programming model and the 1/f read-noise law are ported
from the installed AIHWKit source, aihwkit/inference/noise/pcm.py,
class PCMLikeNoiseModel (constants cited at each use). The 2% white per-cell
read noise and relative-Gaussian programming noise are ASSUMED (swept elsewhere).
"""

from __future__ import annotations

import math

import torch

from neurofault.spectral.types import (
    DTYPE,
    T0_S,
    ConductanceState,
    ProgNoise,
    ReadNoise,
)

# aihwkit/inference/noise/pcm.py, PCMLikeNoiseModel.__init__: default prog_coeff and the
# reference g_max those coefficients (in uS) were fitted at.
PCM_PROG_COEFF = (0.26348, 1.9650, -1.1731)
PCM_PROG_G_MAX_REF_US = 25.0
# PCMLikeNoiseModel.__init__ default t_read (1/f fit parameter), seconds.
PCM_T_READ_S = 250.0e-9
# PCMLikeNoiseModel.apply_drift_noise_to_conductance: q_s = min(0.0088 /
# max((|g|/g_max)^0.65, 1e-3), 0.2); sigma = q_s * sqrt(ln((t + t_read) / (2 t_read))).
PCM_QS_NUM, PCM_QS_EXP, PCM_QS_FLOOR, PCM_QS_MAX = 0.0088, 0.65, 1e-3, 0.2


def _devices(state: ConductanceState) -> list[tuple[torch.Tensor, int]]:
    """(conductance tensor, stuck-plane index) for each physical device array."""
    return [(state.g_pos, 0)] + ([] if state.g_neg is None else [(state.g_neg, 1)])


def _pcm_prog_sigma(g: torch.Tensor, g_max: float) -> torch.Tensor:
    """sigma_prog (uS) = (c0 + c1 x + c2 x^2) * g_max/25, x = g/g_max.

    Port of PCMLikeNoiseModel.apply_programming_noise_to_conductance. The
    polynomial is evaluated on |g| so signed_ideal states stay symmetric."""
    x = g.abs() / g_max
    c0, c1, c2 = PCM_PROG_COEFF
    sig = c0 + c1 * x + c2 * x * x
    return sig * (g_max / PCM_PROG_G_MAX_REF_US)


def program(state: ConductanceState, noise: ProgNoise, gen: torch.Generator) -> ConductanceState:
    """Return a new state with one-time programming noise on every non-stuck device.

    gauss_rel: additive N(0, (rel_sigma * g_max)^2) per device (ProgNoise doc: a
    fraction of g_max). aihwkit_pcm_poly: additive Gaussian with the PCM
    polynomial sigma(g). Both clamp at 0 like AIHWKit (g_prog.clamp_(min=0));
    signed_ideal has no physical rails and is left unclamped. Stuck devices are
    pinned, so they are never perturbed (their RNG draws are still consumed so
    that streams do not shift when a fault pattern changes)."""
    out = state.clone()
    if noise.kind == "none":
        return out
    g_max = state.enc.g_max_uS
    for g, plane in _devices(out):
        z = torch.randn(g.shape, generator=gen, dtype=g.dtype, device=g.device)
        if noise.kind == "gauss_rel":
            sigma = torch.full_like(g, noise.rel_sigma * g_max)
        elif noise.kind == "aihwkit_pcm_poly":
            sigma = _pcm_prog_sigma(g, g_max)
        else:
            raise ValueError(f"unknown prog noise {noise.kind!r}")
        new = g + sigma * z
        if state.enc.kind != "signed_ideal":
            new = new.clamp(min=0.0)
        g.copy_(torch.where(state.stuck[..., plane], g, new))
    return out


def one_over_f_sigma(g: torch.Tensor, g_max: float, t: float) -> torch.Tensor:
    """Relative 1/f read-noise sigma per device (multiplies |g|).

    Port of the noise term in PCMLikeNoiseModel.apply_drift_noise_to_conductance,
    with AIHWKit's t_total = t_inference + t_0 (t_0 = T0_S = 20 s)."""
    t_tot = t + T0_S
    q_s = (PCM_QS_NUM / ((g.abs() / g_max) ** PCM_QS_EXP).clamp(min=PCM_QS_FLOOR)).clamp(
        max=PCM_QS_MAX
    )
    return q_s * math.sqrt(math.log((t_tot + PCM_T_READ_S) / (2 * PCM_T_READ_S)))


def read_noise_cell(
    state: ConductanceState, noise: ReadNoise, t: float, gen: torch.Generator
) -> torch.Tensor:
    """One noisy readout of the whole tile as effective signed weights (m, n).

    Cell-level noise only: 1/f (multiplicative on |g|, per device, if
    noise.one_over_f) and white per-cell noise N(0, (sigma_cell_frac * max|w|)^2)
    in weight units (if mode is per_cell or both). Output-referred noise and
    ADC/DAC quantization belong to the periphery and are applied by TileReader.
    A state with no active cell-level noise returns state.weights() unchanged."""
    g_max = state.enc.g_max_uS
    if noise.one_over_f:
        noisy = state.clone()
        for g, _plane in _devices(noisy):
            z = torch.randn(g.shape, generator=gen, dtype=g.dtype, device=g.device)
            g.copy_((g + g.abs() * one_over_f_sigma(g, g_max, t) * z).clamp(min=0.0))
        w = noisy.weights()
    else:
        w = state.weights()
    if noise.mode in ("per_cell", "both") and noise.sigma_cell_frac > 0.0:
        wmax = float(state.weights().abs().max())
        w = w + noise.sigma_cell_frac * wmax * torch.randn(
            w.shape, generator=gen, dtype=DTYPE, device=w.device
        )
    return w
