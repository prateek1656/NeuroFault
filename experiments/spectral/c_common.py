"""Shared plumbing for the Direction C' (analog eigen loop) experiments C-X0, C-X1, C-X5.

Conventions: conductances in units of g_max = 1 (so G_lambda, sigma and the
row sums S share one unit); omega = 1 (GBW = 1/(2 pi) Hz) so simulator seconds are tau = omega t
and every time here is in 1/omega unless suffixed _us (converted at a stated GBW). Time in 1/omega
is GBW-independent in the linear regime, so GBW sweeps are post-hoc rescalings; only the 1 ns
comparator delay of the "MVM + comparator" baseline breaks that scaling and is added explicitly.

One output-noise convention for every arm (the scratch probe's): iid Gaussian of std sigma volts
on every readout line of the LIVE read, the healthy reference vector stored exactly (digital
record). Loop outputs peak at v_target = 0.5 V of a 1 V rail; the +-1 probe drives +-1 V into a
TIA whose G_f = 4 rms(Ax) leaves an output rms of 0.25 V.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from f0_common import get_tile

from neurofault.spectral.circuit import (
    LoopConfig,
    build_loop,
    mvm_settle_detail,
    simulate,
)
from neurofault.spectral.encoding import encode
from neurofault.spectral.types import EncodingSpec, make_generator

GBW_UNIT = 1.0 / (2.0 * math.pi)  # omega = 1
GMIN_FRAC = 0.05
A0_DEFAULT = 1e4  # 80 dB, the scratch probe's value
SIGMA_REF = 1e-3  # V, the scratch probe's reference output noise for SNR columns
T_CMP_S = 1e-9  # comparator delay of the MVM+comparator baseline


def tau_to_us(tau: float, gbw_hz: float) -> float:
    return float(tau) / (2.0 * math.pi * gbw_hz) * 1e6


def planes(
    ckpt_seed: int, layer: str, r0: int, c0: int, n: int, gmin: float = GMIN_FRAC
) -> tuple[np.ndarray, np.ndarray]:
    """(G+, G-) of an n x n tile at (r0, c0) in units of g_max, diff-pair encoding."""
    tile = get_tile(ckpt_seed, layer, r0, c0, n, n, n)
    enc = EncodingSpec("diff_pair", g_min_frac=gmin)
    st = encode(tile, enc)
    g = enc.g_max_uS
    return st.g_pos.numpy() / g, st.g_neg.numpy() / g


def flat_pilot(m: int, n: int, gen: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random-sign flat unit vectors (the plan's loop-only pilot direction)."""
    u = torch.sign(torch.randn(m, generator=gen, dtype=torch.float64)).numpy()
    v = torch.sign(torch.randn(n, generator=gen, dtype=torch.float64)).numpy()
    u[u == 0] = 1.0
    v[v == 0] = 1.0
    return u / math.sqrt(m), v / math.sqrt(n)


@dataclass
class Healthy:
    """Everything fixed by (tile, beta, f): matrix, spectrum, delta, hardware modal state."""

    gp: np.ndarray
    gn: np.ndarray
    beta: float
    f: float
    u0: np.ndarray
    v0: np.ndarray
    s1_a: float  # sigma_1 of the stored tile A (pilot scale reference)
    sig: np.ndarray  # top singular values of A + P (3)
    delta: float
    lam_ref: float

    @property
    def gap(self) -> float:
        return float(self.sig[0] / self.sig[1])


def make_healthy(gp, gn, beta, f, u0, v0) -> Healthy:
    a = gp - gn
    s1_a = float(np.linalg.norm(a, 2))
    p = beta * s1_a * np.outer(u0, v0)
    sv = np.linalg.svd(a + p, compute_uv=False)
    delta = f * (1.0 - float(sv[1] / sv[0]))
    return Healthy(gp, gn, beta, f, u0, v0, s1_a, sv[:3], delta, float(sv[0]))


def loop_config(h: Healthy, amp: str, **kw) -> LoopConfig:
    """Scratch-probe defaults (A0 = 80 dB, v_sat 1 V, target 0.5 V, hard clip) + overrides.
    ``amp``: clip | agc_ideal | limiter_<bw> | servo_<kp>."""
    base = {
        "gbw_hz": GBW_UNIT,
        "a0": A0_DEFAULT,
        "v_sat": 1.0,
        "delta": h.delta,
        "topology": "pilot",
        "saturation": "hard",
        "v_target_frac": 0.5,
        "g_lambda_ref": h.lam_ref,
    }
    if amp in ("clip", "agc_ideal"):
        base["amplitude"] = amp
    elif amp.startswith("limiter_"):
        base["amplitude"] = "peak_limiter"
        base["agc_bw_hz"] = float(amp.split("_", 1)[1]) * GBW_UNIT
    elif amp.startswith("servo_"):
        base["amplitude"] = "gl_servo"
        base["servo_kp"] = float(amp.split("_", 1)[1])
    else:
        raise ValueError(amp)
    base.update(kw)
    return LoopConfig(**base)


def build(h: Healthy, cfg: LoopConfig, seed: int, gp=None, gn=None):
    """Loop model on (possibly faulty) planes with the healthy pilot and G_lambda reference.
    The generator is rebuilt from ``seed`` every call so mismatch / offset draws are identical
    for the healthy and every faulty instance of one hardware sample."""
    gen = make_generator(seed, "c_loop_build")
    return build_loop(
        h.gp if gp is None else gp,
        h.gn if gn is None else gn,
        cfg,
        gen,
        pilot_beta=h.beta,
        pilot_u0=h.u0,
        pilot_v0=h.v0,
        pilot_s1_ref=h.s1_a,
    )


def run(model, v0, t_end, dt, n_samples, seed=0):
    tr = simulate(
        model,
        t_end,
        v0=v0,
        dt=dt,
        n_samples=n_samples,
        gen=make_generator(seed, "c_loop_sim"),
    )
    return tr


# ----------------------------------------------------------------------------- faults


def stuck_lrs(gp, gn, i: int, j: int, dev: int):
    """Pin device ``dev`` (0 = G+, 1 = G-) of cell (i, j) at g_max (= 1). Returns planes and the
    measured weight change (the scratch probe's stuck-at-rail, with the differential-pair state
    dependence of faults.inject)."""
    gp2, gn2 = gp.copy(), gn.copy()
    old = gp[i, j] - gn[i, j]
    if dev == 0:
        gp2[i, j] = 1.0
    else:
        gn2[i, j] = 1.0
    return gp2, gn2, float(gp2[i, j] - gn2[i, j] - old)


def draw_faults(m: int, n: int, k: int, rng: np.random.Generator):
    return [(int(rng.integers(m)), int(rng.integers(n)), int(rng.integers(2))) for _ in range(k)]


def fault_delta(gp, gn, i, j, dev) -> float:
    return stuck_lrs(gp, gn, i, j, dev)[2]


# ----------------------------------------------------------------------------- B1 probe


@dataclass
class Probe:
    g_f: float
    t_mvm8: float  # tau, slowest row (max load), forward
    t_mvm8_mean: float  # tau, mean-load convention of the scratch probe
    snr_ref: float  # median |r_i| / SIGMA_REF
    sigma_star: float  # output noise at which the +-1 probe localizes the row 90 %
    loc_at_ref: float
    deltas: np.ndarray


def probe_baseline(gp, gn, faults, rng: np.random.Generator, bits: int = 8) -> Probe:
    """B1 (+-1 probe, stored reference output) on the same tile and fault list."""
    import torch as T

    det = mvm_settle_detail(
        T.from_numpy(gp), T.from_numpy(gn), GBW_UNIT, A0_DEFAULT, bits, direction="fwd"
    )
    det_mean = mvm_settle_detail(
        T.from_numpy(gp),
        T.from_numpy(gn),
        GBW_UNIT,
        A0_DEFAULT,
        bits,
        direction="fwd",
        load_stat="mean",
    )
    g_f = det["g_f"]
    deltas = np.array([fault_delta(gp, gn, *f) for f in faults])
    amp = np.abs(deltas) / g_f  # |r_i| volts (x_j = +-1)
    m = gp.shape[0]
    xi_i = rng.standard_normal(len(faults))
    mx = np.abs(rng.standard_normal((len(faults), m - 1))).max(axis=1)

    def loc(sig: float) -> float:
        return float(np.mean(np.abs(amp / sig + xi_i) > mx))

    lo, hi = 1e-6, 1.0
    for _ in range(40):  # loc decreases with sigma
        mid = math.sqrt(lo * hi)
        if loc(mid) >= 0.9:
            lo = mid
        else:
            hi = mid
    return Probe(
        g_f=g_f,
        t_mvm8=det["t_s"],
        t_mvm8_mean=det_mean["t_s"],
        snr_ref=float(np.median(amp) / SIGMA_REF),
        sigma_star=math.sqrt(lo * hi),
        loc_at_ref=loc(SIGMA_REF),
        deltas=deltas,
    )


# ----------------------------------------------------------------------------- loop analysis


def smooth3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x
    y = x.copy()
    y[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return y


def first_stay(t: np.ndarray, ok: np.ndarray) -> float:
    """First time after which ``ok`` holds to the end of the record; inf if never."""
    if not ok[-1]:
        return math.inf
    bad = np.nonzero(~ok)[0]
    i = 0 if len(bad) == 0 else int(bad[-1]) + 1
    return float(t[i])


def localization_curves(
    d_u: np.ndarray,
    d_v: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    draws: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """loc_row(t), loc_col(t): fraction of (fault, noise draw) with argmax|d + sigma xi| = truth.
    d_u (F, T, m), d_v (F, T, n)."""
    out = []
    for d, idx in ((d_u, rows), (d_v, cols)):
        f, t, k = d.shape
        noise = sigma * rng.standard_normal((f, t, draws, k))
        arg = np.argmax(np.abs(d[:, :, None, :] + noise), axis=-1)  # (F, T, draws)
        out.append((arg == idx[:, None, None]).mean(axis=(0, 2)))
    return out[0], out[1]


def top_vec_mismatched(model, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Dominant mode of D^-1 (M - diag(G_lambda (1 + eps xi))): the loop's hardware fixed
    point with per-TIA G_lambda mismatch eps (relative sigma), without the amplitude servo's
    own bias. Symmetric reduction (the dilation is symmetric)."""
    from scipy.linalg import eigh

    m = model.topo.m_eff
    gl = model.g_nominal * (1.0 + eps * rng.standard_normal(model.n))
    d = model.topo.load + model.cfg.node_load_factor * gl
    sd = 1.0 / np.sqrt(d)
    z = sd[:, None] * (m - np.diag(gl)) * sd[None, :]
    n = z.shape[0]
    _, q = eigh(0.5 * (z + z.T), subset_by_index=[n - 1, n - 1])
    v = sd * q[:, 0]
    return v / np.linalg.norm(v)


def required_gl_precision(model, rng, target: float = 0.98, draws: int = 2) -> float:
    """Largest relative G_lambda mismatch sigma (grid) that keeps the settled state's purity
    cos^2 against the un-mismatched hardware state >= ``target``; 0.0005 if even that fails."""
    grid = [0.001, 0.003, 0.01, 0.03]
    base = top_vec_mismatched(model, 0.0, rng)
    best = 0.0005
    for eps in grid:
        pur = np.median(
            [float(top_vec_mismatched(model, eps, rng) @ base) ** 2 for _ in range(draws)]
        )
        if pur >= target:
            best = eps
        else:
            break
    return best


def energy_ratio(n: int, arrays_loop: int, t_loop_tau: float, t_mvm_tau: float, gbw_hz: float):
    """E(loop) / E(one MVM + comparator), registered constants: static array power
    arrays * 2 N^2 G_mean V^2 with G_mean = 25 uS, V = 0.2 V; 100 uW per op-amp (2N loop TIAs +
    2N inverters = 4N amps vs N TIAs); 0.5 pJ per comparator."""
    p_arr = 2 * n * n * 25e-6 * 0.2**2
    p_amp = 100e-6
    t_l = t_loop_tau / (2 * math.pi * gbw_hz)
    t_m = t_mvm_tau / (2 * math.pi * gbw_hz) + T_CMP_S
    e_loop = (arrays_loop * p_arr + 4 * n * p_amp) * t_l
    e_mvm = (p_arr + n * p_amp) * t_m + n * 0.5e-12
    return e_loop / e_mvm
