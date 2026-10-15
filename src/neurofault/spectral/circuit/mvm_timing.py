"""Settling time of an ordinary TIA matrix-vector multiply, on the SAME op-amp model
as the eigenvector loop (never let two amplifier models enter).

A TIA row i with feedback conductance G_f, input cells summing to S_i and a
single-pole op-amp (omega = 2 pi GBW, DC gain A0) obeys, from the same KCL as the
loop but with the feedback closed around -o instead of an inverter chain:

    (S_i + G_f) n_i = sum_j G_ij x_j + G_f o_i,   do_i/dtau = -n_i - o_i / A0
    =>  do_i/dtau = -(G_f/(S_i + G_f) + 1/A0) o_i - (sum_j G_ij x_j)/(S_i + G_f)

a first-order lag with rate r_i = G_f/(S_i + G_f) + 1/A0 (x omega). The slowest
row (largest S_i) sets the array settle time; settling a step to 2^-bits of its
size takes bits*ln2 / (omega r). Because r depends only on the node load S_i and
G_f this is exactly the quantity the loop's growth rate is compared against.
"""

from __future__ import annotations

import math

import numpy as np
import torch

Tensor = torch.Tensor


def _np(x: Tensor | None) -> np.ndarray | None:
    return None if x is None else x.detach().cpu().numpy().astype(np.float64)


def mvm_settle_detail(
    g_pos: Tensor,
    g_neg: Tensor | None,
    gbw_hz: float,
    a0: float,
    bits: int,
    *,
    g_f: float | None = None,
    headroom: float = 4.0,
    direction: str = "fwd",
    load_stat: str = "max",
) -> dict:
    """Settle time (s) plus the numbers behind it.

    ``g_f`` default: headroom * rms of y = G_eff x for iid +-1 inputs (a
    deterministic expression, sqrt(mean_i sum_j G_ij^2)), so a 1 V input swing
    gives an output rms of 1/headroom volts (the probe's convention).
    ``direction`` 'fwd' (loads = row sums), 'bwd' (column sums, transpose read) or
    'both' (worst of the two). Differential pairs load a node with G+ + G-.
    ``load_stat`` 'max' (default: the slowest row sets the array settle) or 'mean'
    (the mean-load convention of the scratch probe, used for C-X0)."""
    gp = _np(g_pos)
    gn = _np(g_neg)
    if gn is None:
        tot, eff = np.abs(gp), gp
    else:
        tot, eff = gp + gn, gp - gn
    if g_f is None:
        g_f = headroom * math.sqrt(float(np.mean(np.sum(eff**2, axis=1))))
    loads = {"fwd": tot.sum(1), "bwd": tot.sum(0)}
    stat = {"max": np.max, "mean": np.mean}[load_stat]
    s_max = max(
        float(stat(loads[d])) for d in (["fwd", "bwd"] if direction == "both" else [direction])
    )
    omega = 2.0 * math.pi * gbw_hz
    rate = omega * (g_f / (s_max + g_f) + 1.0 / a0)
    t = bits * math.log(2.0) / rate
    return {"t_s": t, "rate_per_s": rate, "g_f": g_f, "s_max": s_max, "omega": omega}


def mvm_settle_time(
    g_pos: Tensor, g_neg: Tensor | None, gbw_hz: float, a0: float, bits: int
) -> float:
    """Time for the slowest TIA row to settle to 2^-bits (seconds). Baseline C' must beat."""
    return mvm_settle_detail(g_pos, g_neg, gbw_hz, a0, bits)["t_s"]
