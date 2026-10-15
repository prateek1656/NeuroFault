"""Nodal ODE model of the Sun & Ielmini analog eigenvector loop.

Why a simulator and not a formula: every circuit number in the earlier analytical
estimates is a first-order rederivation; the dilated / signed-split topologies add stages that
Sun's Eq. 4 does not contain. This module derives the dynamics from the node
equations once, checks them against the closed form in unit tests, and then
runs the non-ideal cases (finite A0, second pole, saturation, offsets, noise,
G_lambda mismatch, amplitude control) numerically.

Derivation of the growth rate (resolves the disagreement between the F0 form and the earlier estimate)
---------------------------------------------------------------------------
Amp i is a single-pole op-amp, non-inverting input at (V_os,i + e_n,i), output
o_i, inverting node n_i, with a feedback conductance G_lambda,i from o_i to n_i
and the array cells G_ij from the (inverted) outputs u_j = -o_j to n_i.

    KCL at n_i (op-amp input draws no current):
        sum_j G_ij (u_j - n_i) + G_lambda,i (o_i - n_i) = 0
        =>  (S_i + G_lambda,i) n_i = sum_j G_ij u_j + G_lambda,i o_i,
        S_i = sum_j G_ij  (row sum of ALL conductances on the node).
    Single-pole op-amp, tau_a = A0 / (2 pi GBW) = A0 / omega:
        tau_a dy_i/dt = -y_i + A0 (V_os + e_n - n_i)
        =>  dy_i/dtau = -(n_i - V_os - e_n) - y_i / A0,   tau = omega t.
    With u = -o and the signed matrix M = G+ - G- (G- driven by +o):
        n_i = (-(M o)_i + G_lambda,i o_i) / D_i,   D_i = S_i + G_lambda,i
        =>  do/dtau = D^-1 (M - diag(G_lambda)) o  -  (1/A0) o.

Hence, for A0 finite the dynamics are exactly the ideal dynamics shifted by the
uniform amount -1/A0, and the fixed point (G_lambda = lambda) satisfies
M o = G_lambda o: the exact eigenvector. Modal growth rates (in units of omega) are
the eigenvalues of D^-1 (M - G_lambda I); for equal row sums S_i = S (which
makes D a scalar) the mode k grows at

        r_k = (lambda_k - G_lambda) / (S + G_lambda) - 1/A0     [x omega]

This AGREES WITH the registered form (lambda - G_lambda)/(S + G_lambda), including the
omega = 2 pi GBW factor, and differs from the earlier quoted
GBW (lambda_k - lambda_G)/(3 lambda_G + S) only in the coefficient c of G_lambda
in the denominator. Sun's circuit is not available here, but the earlier
form is reproduced exactly by ``LoopConfig.node_load_factor = 3`` (= an extra
2 G_lambda of conductance to ground hanging on the summing node, e.g. from the
inverter/feedback network): then D_i = S_i + c G_lambda,i with unchanged
numerator and fixed point. So the two sources describe the same model with
c = 1 (bare TIA node, what the cells and one feedback conductance give) versus
c = 3 (an additional feedback network). The simulator defaults to c = 1 and
exposes c; the difference is negligible for S >> G_lambda (S/sigma_1 ~ 8-10 for
differential pairs) and matters only when S/sigma_1 < 1. Which c Sun's netlist
has cannot be decided without the netlist (open issue).

Inverter lag is NOT a small correction: the slow growth rate is O(delta/(S/lambda))
while the inverter delay enters multiplied by the loop gain, so a unity-gain inverter
pole at ``ri`` (x GBW) divides the rate by about 1 + lambda/(ri D) (a factor ~2 for
ri = 0.5, S ~ lambda). It is modeled as two lag states (G+ array driven by lag(-o),
G- array by lag(-lag(-o))).

Amplitude control (the loop is linear until something stops it):
``clip``          hard/tanh output saturation at +-v_sat (Sun: the largest output
                  saturating stops growth; with many lines railed this latches the
                  sign pattern instead of the eigenvector - ``latched`` flag).
``peak_limiter``  common-mode AGC: extra decay kappa*max(0, peak/v_target - 1)
                  on every output; instantaneous peak detector, finite gain.
``agc_ideal``     instantaneous peak AGC: after every step the output vector is
                  rescaled to peak <= v_target (the idealisation of the scratch probe; infinite AGC bandwidth, so an upper bound on what any real limiter can do).
``gl_servo``      a common G_lambda is driven by a PI servo on ln(peak/v_target);
                  at equilibrium G_lambda = lambda, i.e. it also *measures* the
                  eigenvalue. Peak detection is ideal (no detector bandwidth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import torch

from ..types import make_generator
from .topologies import (
    Topology,
    differential,
    dilated_two_array,
    pilot,
    square_signed,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class LoopConfig:
    gbw_hz: float = 16e6  # AD823-class [assumed, unverified]
    a0: float = 1e5
    pole2_hz: float | None = None  # second pole of the amp (output stage), None = absent
    v_sat: float = 1.0
    v_os_rms: float = 0.0  # static offset per amp, drawn once at build_loop
    en_nv_rthz: float = 0.0  # input-referred white noise density
    delta: float = 0.02  # G_lambda = (1 - delta) * lambda_ref
    g_lambda_mismatch_sigma: float = 0.0  # relative sigma per TIA, drawn once
    inverter: Literal["ideal", "pole"] = "ideal"
    inverter_pole_hz: float | None = None  # default gbw/2 (unity-gain inverter closed-loop BW)
    topology: Literal["square_signed", "dilated_two_array", "pilot", "differential"] = (
        "square_signed"
    )
    saturation: Literal["hard", "tanh"] = "hard"
    amplitude: Literal["clip", "peak_limiter", "gl_servo", "agc_ideal"] = "clip"
    v_target_frac: float = 0.5  # limiter/servo target peak, fraction of v_sat
    agc_bw_hz: float | None = None  # peak_limiter gain; default gbw_hz/10 (kappa = 0.1 x omega)
    servo_kp: float = 0.5  # proportional gain of the G_lambda servo (x lambda_ref)
    node_load_factor: float = 1.0  # c in D = S + c G_lambda; 3 reproduces the earlier estimate
    g_lambda_ref: float | None = None  # override lambda_ref (reference-limited tests)
    pilot_loads_node: bool = False

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.gbw_hz


def eigen_reference(
    g_eff: Tensor | np.ndarray, kind: Literal["eig", "svd_dilation"]
) -> tuple[float, Tensor]:
    """Expected fixed point of the loop for effective signed matrix ``g_eff``.

    ``eig``: eigenvalue of largest real part and its (real) right eigenvector of a
    square matrix; raises if that eigenvalue is complex (the loop would ring, not
    settle). ``svd_dilation``: sigma_1 and [u1; v1]/sqrt(2), the top eigenpair of
    [[0, A], [A^T, 0]]. Vectors are unit norm; sign is arbitrary."""
    a = g_eff.detach().cpu().numpy() if isinstance(g_eff, torch.Tensor) else np.asarray(g_eff)
    a = a.astype(np.float64)
    if kind == "svd_dilation":
        u, s, vh = np.linalg.svd(a, full_matrices=False)
        vec = np.concatenate([u[:, 0], vh[0]]) / math.sqrt(2.0)
        return float(s[0]), torch.from_numpy(vec)
    w, v = np.linalg.eig(a)
    k = int(np.argmax(w.real))
    if abs(w[k].imag) > 1e-8 * max(abs(w[k]), 1e-30):
        raise ValueError(f"dominant eigenvalue is complex ({w[k]}); loop would oscillate")
    vec = v[:, k].real
    vec = vec / np.linalg.norm(vec)
    return float(w[k].real), torch.from_numpy(vec)


def analytic_rate(s: float, lam: float, g_lambda: float, a0: float, c: float = 1.0) -> float:
    """Closed-form modal growth rate in units of omega for equal row sums S."""
    return (lam - g_lambda) / (s + c * g_lambda) - 1.0 / a0


@dataclass
class LoopModel:
    cfg: LoopConfig
    topo: Topology
    g_lambda: np.ndarray  # (N,) per-amp feedback conductance incl. mismatch
    g_nominal: float  # (1 - delta) * lambda_ref before mismatch
    v_os: np.ndarray  # (N,)
    ref_value: float  # lambda_ref (or sigma_1)
    ref_vec: np.ndarray  # (N,) unit
    mismatch: np.ndarray  # (N,) relative factors (1 + sigma xi)

    @property
    def n(self) -> int:
        return self.topo.n_amps

    @property
    def omega(self) -> float:
        return self.cfg.omega

    def d_vector(self) -> np.ndarray:
        return self.topo.load + self.cfg.node_load_factor * self.g_lambda

    def modal_rates(self, k: int | None = None, vectors: bool = False):
        """Linearized modal growth rates in units of omega (eigenvalues of
        D^-1 (M - diag G_lambda) minus 1/A0), descending; optionally the matching
        output-space eigenvectors. The rates at the nominal G_lambda are the
        closed forms the unit tests compare simulations against."""
        m = self.topo.m_eff
        d = self.d_vector()
        a = m - np.diag(self.g_lambda)
        if np.allclose(m, m.T):
            sd = 1.0 / np.sqrt(d)
            z = sd[:, None] * a * sd[None, :]
            w, q = np.linalg.eigh(0.5 * (z + z.T))
            order = np.argsort(w)[::-1]
            w, q = w[order], q[:, order]
            vecs = sd[:, None] * q
        else:
            w, q = np.linalg.eig(a / d[:, None])
            order = np.argsort(w.real)[::-1]
            w, vecs = w[order].real, q[:, order].real
        rates = w - 1.0 / self.cfg.a0
        if k is not None:
            rates, vecs = rates[:k], vecs[:, :k]
        return (rates, vecs) if vectors else rates

    def default_dt_tau(self) -> float:
        """Step in omega-normalized time. D^-1(M-G) has spectral radius <= 1
        (Gershgorin), so RK4 is stable at dt <= ~2.8 (2 used); poles tighten it."""
        cfg = self.cfg
        dt = 2.0
        if cfg.pole2_hz is not None:
            dt = min(dt, 0.25 * cfg.gbw_hz / cfg.pole2_hz)
        if cfg.inverter == "pole":
            dt = min(dt, 0.25 * cfg.gbw_hz / self._inv_hz())
        if cfg.amplitude == "peak_limiter":
            dt = min(dt, 0.5 / self._kappa())
        return dt

    def _inv_hz(self) -> float:
        return self.cfg.inverter_pole_hz or 0.5 * self.cfg.gbw_hz

    def _kappa(self) -> float:
        return (self.cfg.agc_bw_hz or 0.1 * self.cfg.gbw_hz) / self.cfg.gbw_hz


def build_loop(
    g_pos: Tensor,
    g_neg: Tensor | None,
    cfg: LoopConfig,
    gen: torch.Generator | None = None,
    *,
    pilot_beta: float = 0.0,
    g_ref=None,
    pilot_u0=None,
    pilot_v0=None,
    pilot_s1_ref: float | None = None,
) -> LoopModel:
    """Assemble the loop for conductance planes ``g_pos``/``g_neg`` (any consistent
    unit; only ratios matter). ``g_ref`` is the reference array for the
    differential topology: a signed matrix or a (G+, G-) tuple. Mismatch and
    offsets are drawn once here, from ``gen``, so a model is a fixed instance."""
    gen = gen if gen is not None else make_generator(0, "loop_build")
    kind = cfg.topology
    if kind == "square_signed":
        topo = square_signed(g_pos, g_neg)
    elif kind == "dilated_two_array":
        topo = dilated_two_array(g_pos, g_neg)
    elif kind == "pilot":
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        topo = pilot(
            g_pos,
            g_neg,
            pilot_beta,
            pilot_u0,
            pilot_v0,
            rng=rng,
            loads_node=cfg.pilot_loads_node,
            s1_ref=pilot_s1_ref,
        )
    elif kind == "differential":
        if g_ref is None:
            raise ValueError("differential topology needs g_ref")
        topo = differential(g_pos, g_neg, g_ref)
    else:
        raise ValueError(f"unknown topology {kind}")
    lam, vec = eigen_reference(topo.base_eff, topo.ref_kind)
    lam_ref = cfg.g_lambda_ref if cfg.g_lambda_ref is not None else lam
    g_nom = (1.0 - cfg.delta) * lam_ref
    n = topo.n_amps
    xi = torch.randn(n, generator=gen, dtype=torch.float64).numpy()
    mism = 1.0 + cfg.g_lambda_mismatch_sigma * xi
    vo = torch.randn(n, generator=gen, dtype=torch.float64).numpy() * cfg.v_os_rms
    return LoopModel(
        cfg=cfg,
        topo=topo,
        g_lambda=g_nom * mism,
        g_nominal=g_nom,
        v_os=vo,
        ref_value=lam,
        ref_vec=vec.numpy(),
        mismatch=mism,
    )


@dataclass
class LoopTrace:
    """Result of ``simulate``. Times in seconds; ``v`` is the subsampled output
    vector history (T, N); ``peak``/``purity_t`` are on the same grid."""

    t: Tensor
    v: Tensor
    purity_t: Tensor  # cos^2(v(t), reference eigenvector)
    peak: Tensor  # max |v| at each sample
    frac_sat: Tensor  # fraction of lines within 5% of the rail
    alive: bool
    saturated: bool  # some line reached the rail at any time
    latched: bool  # >= half the lines railed at the end (sign-latch)
    v_sat: float
    g_servo: Tensor | None = None  # servo G_lambda history (T,) if gl_servo
    meta: dict = field(default_factory=dict)

    def purity(self, t: float | Tensor | None = None) -> Tensor:
        if t is None:
            return self.purity_t
        tt = torch.as_tensor(t, dtype=torch.float64)
        out = np.interp(tt.numpy(), self.t.numpy(), self.purity_t.numpy())
        return torch.from_numpy(np.asarray(out))

    def t_settle(self, tol: float = 1e-2, min_peak_frac: float = 0.0) -> float:
        """First time after which purity >= 1 - tol holds to the end of the trace
        (and peak >= min_peak_frac * v_sat); nan if it never does."""
        ok = (self.purity_t.numpy() >= 1.0 - tol) & (
            self.peak.numpy() >= min_peak_frac * self.v_sat
        )
        if not ok[-1]:
            return float("nan")
        bad = np.nonzero(~ok)[0]
        i = 0 if len(bad) == 0 else int(bad[-1]) + 1
        return float(self.t[i])

    def t_amp(self, frac: float = 0.5) -> float:
        """First time the peak output reaches frac * v_sat (log-linear interpolation
        between samples): Sun's 'computing time' to amplitude."""
        pk = self.peak.numpy()
        target = frac * self.v_sat
        idx = np.nonzero(pk >= target)[0]
        if len(idx) == 0:
            return float("nan")
        i = int(idx[0])
        if i == 0:
            return float(self.t[0])
        a, b = max(pk[i - 1], 1e-300), pk[i]
        w = (math.log(target) - math.log(a)) / max(math.log(b) - math.log(a), 1e-300)
        return float(self.t[i - 1] + w * (self.t[i] - self.t[i - 1]))


def simulate(
    model: LoopModel,
    t_end_s: float,
    *,
    x0_rms: float = 1e-5,
    dt: float | None = None,
    gen: torch.Generator | None = None,
    v0: Tensor | np.ndarray | None = None,
    n_samples: int = 2000,
    stop_when_settled: float | None = None,
) -> LoopTrace:
    """Integrate the loop ODE with fixed-step RK4 (noise held piecewise-constant
    per step: Euler-Maruyama-style forcing, variance en^2 / (2 dt)).

    ``dt`` in seconds (default: omega-normalized step 2, tightened by poles).
    ``v0`` warm-starts the outputs; otherwise iid noise of rms ``x0_rms`` (the
    seed that Sun's loop grows from). ``stop_when_settled`` = tol ends the run
    once purity >= 1 - tol with peak >= 0.1 v_sat, so t_end can be generous."""
    cfg = model.cfg
    topo = model.topo
    gen = gen if gen is not None else make_generator(0, "loop_sim")
    n = model.n
    w = model.omega
    dt_tau = (dt * w) if dt is not None else model.default_dt_tau()
    n_steps = max(1, math.ceil(t_end_s * w / dt_tau))
    dt_tau = t_end_s * w / n_steps if dt is None else dt_tau
    dt_s = dt_tau / w

    vs = cfg.v_sat
    hard = cfg.saturation == "hard"
    ymax = vs if hard else 8.0 * vs
    a0 = cfg.a0
    ideal = cfg.inverter == "ideal"
    has_p = cfg.pole2_hz is not None
    r2 = cfg.pole2_hz / cfg.gbw_hz if has_p else 0.0
    ri = model._inv_hz() / cfg.gbw_hz
    servo = cfg.amplitude == "gl_servo"
    limiter = cfg.amplitude == "peak_limiter"
    ideal_agc = cfg.amplitude == "agc_ideal"
    vt = cfg.v_target_frac * vs
    kappa = model._kappa()
    load = topo.load
    c = cfg.node_load_factor
    mism = model.mismatch
    m_eff = topo.m_eff
    mp, mn = topo.mp, topo.mn
    vos = model.v_os
    ref = model.ref_vec
    lam_ref = model.ref_value
    d_mean = float(load.mean() + c * model.g_nominal)
    kp = cfg.servo_kp
    ki = (kp * lam_ref) ** 2 / (4.0 * d_mean)
    sigma_e = cfg.en_nv_rthz * 1e-9 / math.sqrt(2.0 * dt_s) if cfg.en_nv_rthz > 0 else 0.0

    # state layout: y | p (pole2) | x1 x2 (inverter poles) | g (servo)
    off = {"y": 0}
    k = n
    if has_p:
        off["p"] = k
        k += n
    if not ideal:
        off["x1"] = k
        k += n
        off["x2"] = k
        k += n
    if servo:
        off["g"] = k
        k += 1

    def sat(x):
        return np.clip(x, -vs, vs) if hard else vs * np.tanh(x / vs)

    def rhs(z, e):
        y = z[:n]
        p = z[off["p"] : off["p"] + n] if has_p else y
        o = sat(p)
        peak = float(np.max(np.abs(o)))
        if servo:
            err = math.log(max(peak, 1e-300) / vt)
            errc = max(-1.0, min(1.0, err))
            g = z[off["g"]]
            g_eff = max(g + kp * lam_ref * max(-2.0, min(2.0, err)), 0.0)
            gl = g_eff * mism
        else:
            gl = model.g_lambda
        if ideal:
            num = -(m_eff @ o)
        else:
            num = mp @ z[off["x1"] : off["x1"] + n] + mn @ z[off["x2"] : off["x2"] + n]
        nn = (num + gl * o) / (load + c * gl)
        dz = np.empty_like(z)
        dy = -(nn - e) - y / a0
        if limiter:
            dy = dy - kappa * max(0.0, peak / vt - 1.0) * y
        dz[:n] = dy
        if has_p:
            dz[off["p"] : off["p"] + n] = r2 * (y - p)
        if not ideal:
            x1 = z[off["x1"] : off["x1"] + n]
            x2 = z[off["x2"] : off["x2"] + n]
            dz[off["x1"] : off["x1"] + n] = ri * (-o - x1)
            dz[off["x2"] : off["x2"] + n] = ri * (-x1 - x2)
        if servo:
            dz[off["g"]] = ki * errc
        return dz

    z = np.zeros(k)
    if v0 is not None:
        y0 = (v0.detach().cpu().numpy() if isinstance(v0, torch.Tensor) else np.asarray(v0)).astype(
            np.float64
        )
    else:
        y0 = x0_rms * torch.randn(n, generator=gen, dtype=torch.float64).numpy()
    z[:n] = y0
    o0 = sat(y0)
    if has_p:
        z[off["p"] : off["p"] + n] = y0
    if not ideal:
        z[off["x1"] : off["x1"] + n] = -o0
        z[off["x2"] : off["x2"] + n] = o0
    if servo:
        z[off["g"]] = model.g_nominal

    rec_every = max(1, n_steps // n_samples)
    n_rec = n_steps // rec_every + 2
    ts = np.zeros(n_rec)
    vs_hist = np.zeros((n_rec, n))
    pur = np.zeros(n_rec)
    peaks = np.zeros(n_rec)
    fsat = np.zeros(n_rec)
    gh = np.zeros(n_rec) if servo else None

    def out_vec(zz):
        return sat(zz[off["p"] : off["p"] + n] if has_p else zz[:n])

    def record(j, step, zz):
        o = out_vec(zz)
        nrm = float(np.linalg.norm(o))
        ts[j] = step * dt_s
        vs_hist[j] = o
        pur[j] = (float(o @ ref) ** 2) / (nrm * nrm) if nrm > 0 else 0.0
        peaks[j] = float(np.max(np.abs(o)))
        fsat[j] = float(np.mean(np.abs(o) >= 0.95 * vs))
        if servo:
            gh[j] = zz[off["g"]]

    record(0, 0, z)
    j = 0
    chunk = 1024
    noise = None
    stopped_at = n_steps
    last_rec = 0
    for step in range(1, n_steps + 1):
        if sigma_e > 0.0:
            ci = (step - 1) % chunk
            if ci == 0:
                noise = sigma_e * torch.randn(chunk, n, generator=gen, dtype=torch.float64).numpy()
            e = vos + noise[ci]
        else:
            e = vos
        k1 = rhs(z, e)
        k2 = rhs(z + 0.5 * dt_tau * k1, e)
        k3 = rhs(z + 0.5 * dt_tau * k2, e)
        k4 = rhs(z + dt_tau * k3, e)
        z = z + (dt_tau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        np.clip(z[:n], -ymax, ymax, out=z[:n])
        if ideal_agc:
            pk = float(np.max(np.abs(sat(z[off["p"] : off["p"] + n] if has_p else z[:n]))))
            if pk > vt:
                z[: off["g"] if servo else k] *= vt / pk
        if step % rec_every == 0:
            j += 1
            record(j, step, z)
            last_rec = step
            if (
                stop_when_settled is not None
                and pur[j] >= 1.0 - stop_when_settled
                and peaks[j] >= 0.1 * vs
            ):
                stopped_at = step
                break
    if last_rec != stopped_at:
        j += 1
        record(j, stopped_at, z)
    sl = slice(0, j + 1)
    alive = bool(peaks[j] >= 10.0 * max(peaks[0], 1e-300))
    return LoopTrace(
        t=torch.from_numpy(ts[sl].copy()),
        v=torch.from_numpy(vs_hist[sl].copy()),
        purity_t=torch.from_numpy(pur[sl].copy()),
        peak=torch.from_numpy(peaks[sl].copy()),
        frac_sat=torch.from_numpy(fsat[sl].copy()),
        alive=alive,
        saturated=bool(np.any(peaks[sl] >= 0.98 * vs)),
        latched=bool(fsat[j] >= 0.5),
        v_sat=vs,
        g_servo=None if gh is None else torch.from_numpy(gh[sl].copy()),
        meta={"n_steps": stopped_at, "dt_s": dt_s, "dt_tau": dt_tau},
    )


def with_config(model: LoopModel, **changes) -> LoopModel:
    """Same instance (cells, mismatch, offsets) with some config fields changed."""
    return replace(model, cfg=replace(model.cfg, **changes))
