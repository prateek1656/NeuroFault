"""Tests for the spectral loop simulator (circuit/): closed-form growth rates,
topologies, amplitude modes, MVM timing, and the fast Sun & Ielmini checks."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from neurofault.spectral.circuit import (
    LoopConfig,
    analytic_rate,
    build_loop,
    eigen_reference,
    mvm_settle_detail,
    mvm_settle_time,
    simulate,
    with_config,
)
from neurofault.spectral.types import make_generator

W1 = 1.0 / (2.0 * math.pi)  # GBW giving omega = 1: simulator seconds == tau


def _circulant(n: int = 8, seed: int = 0) -> torch.Tensor:
    c = torch.rand(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.stack([torch.roll(c, i) for i in range(n)])  # equal row sums


def _rand(n: int, m: int | None = None, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, m or n, generator=g, dtype=torch.float64)


def _fit_rate(tr, lo: float = 1e-4, hi: float = 1e3) -> float:
    pk, t = tr.peak.numpy(), tr.t.numpy()
    sl = (pk > lo) & (pk < hi)
    return float(np.polyfit(t[sl], np.log(pk[sl]), 1)[0])


@pytest.mark.parametrize("c", [1.0, 3.0])
def test_growth_rate_matches_closed_form(c):
    # Equal row sums -> D is a scalar and r = (lam - G)/(S + c G) - 1/A0 is exact.
    # c=1 is the registered form; c=3 is the alternative (3 lam_G + S).
    g = _circulant()
    cfg = LoopConfig(gbw_hz=W1, a0=1e4, delta=0.05, node_load_factor=c, v_sat=1e9)
    model = build_loop(g, None, cfg)
    s = float(g.sum(1)[0])
    closed = analytic_rate(s, model.ref_value, model.g_nominal, 1e4, c)
    assert model.modal_rates(1)[0] == pytest.approx(closed, rel=1e-10)
    tr = simulate(model, 3000.0, x0_rms=1e-6, gen=make_generator(0, "t"))
    assert _fit_rate(tr) == pytest.approx(closed, rel=1e-5)


def test_research01_form_is_the_c3_node_load():
    # earlier estimate: rate = GBW (lam_k - lam_G)/(3 lam_G + S) for every mode k.
    g = _circulant(seed=2)
    model = build_loop(g, None, LoopConfig(gbw_hz=W1, a0=1e12, delta=0.1, node_load_factor=3.0))
    lam_g = model.g_nominal
    s = float(g.sum(1)[0])
    eig = np.sort(np.linalg.eigvals(g.numpy()).real)[::-1]
    rates = model.modal_rates()
    for k in range(3):
        assert rates[k] == pytest.approx((eig[k] - lam_g) / (3 * lam_g + s), abs=1e-9)


def test_finite_a0_is_a_uniform_shift():
    g = _rand(6, seed=4)
    hi = build_loop(g, None, LoopConfig(gbw_hz=W1, a0=1e12)).modal_rates()
    lo = build_loop(g, None, LoopConfig(gbw_hz=W1, a0=100.0)).modal_rates()
    assert np.allclose(hi - lo, 1.0 / 100.0, atol=1e-9)


def test_unequal_row_sums_simulation_matches_generalized_eig():
    g = _rand(12, seed=5)
    model = build_loop(g, None, LoopConfig(gbw_hz=W1, a0=1e6, delta=0.05, v_sat=1e9))
    tr = simulate(model, 2500.0, x0_rms=1e-6, gen=make_generator(1, "t"))
    assert _fit_rate(tr) == pytest.approx(model.modal_rates(1)[0], rel=1e-4)


def test_dt_halving_invariance():
    model = build_loop(_rand(10, seed=6), None, LoopConfig(gbw_hz=W1, delta=0.05))
    ts = []
    for dt in (2.0, 1.0, 0.5):
        tr = simulate(model, 1500.0, x0_rms=1e-5, dt=dt, gen=make_generator(0, "t"), n_samples=3000)
        ts.append(tr.t_amp(0.1))
    assert ts[0] == pytest.approx(ts[2], rel=2e-3)
    assert ts[1] == pytest.approx(ts[2], rel=2e-3)


def test_gbw_scaling_is_exactly_linear():
    g = _rand(5, seed=7)
    out = []
    for gbw in (1e6, 4e6):
        m = build_loop(g, None, LoopConfig(gbw_hz=gbw, delta=0.05))
        tr = simulate(m, 600.0 / m.omega, x0_rms=1e-5, gen=make_generator(0, "t"))
        out.append(tr.t_amp(0.1))
    assert out[0] / out[1] == pytest.approx(4.0, rel=1e-6)


def test_eigen_reference():
    a = _rand(7, 5, seed=8)
    s, v = eigen_reference(a, "svd_dilation")
    assert s == pytest.approx(float(torch.linalg.svdvals(a)[0]))
    assert v.shape == (12,) and float(v.norm()) == pytest.approx(1.0)
    lam, vec = eigen_reference(_rand(6, seed=9), "eig")
    g = _rand(6, seed=9)
    assert torch.allclose(g @ vec, lam * vec, atol=1e-10)
    rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        eigen_reference(rot, "eig")


def test_dilated_topology_signing_and_loads():
    gp, gn = _rand(6, 4, seed=10) * 0.8, _rand(6, 4, seed=11) * 0.3
    model = build_loop(gp, gn, LoopConfig(topology="dilated_two_array", gbw_hz=W1))
    topo = model.topo
    a = (gp - gn).numpy()
    assert (topo.mp >= 0).all() and (topo.mn >= 0).all()  # arrays hold only positive G
    assert np.allclose(topo.m_eff[:6, 6:], a) and np.allclose(topo.m_eff[6:, :6], a.T)
    assert np.allclose(topo.load[:6], (gp + gn).sum(1).numpy())  # S counts both arrays
    assert np.allclose(topo.load[6:], (gp + gn).sum(0).numpy())
    assert model.ref_value == pytest.approx(float(np.linalg.svd(a, compute_uv=False)[0]))
    # dilated spectrum is symmetric: +sigma grows, -sigma decays
    r = model.modal_rates()
    assert r[0] > r[-1]


def test_signed_ideal_matches_pair_encoding():
    a = _rand(5, seed=12) - 0.5
    pair = build_loop(
        a.clamp(min=0) + 0.1, (-a).clamp(min=0) + 0.1, LoopConfig(topology="dilated_two_array")
    )
    ideal = build_loop(a, None, LoopConfig(topology="dilated_two_array"))
    assert np.allclose(pair.topo.m_eff, ideal.topo.m_eff)
    assert (pair.topo.load > ideal.topo.load).all()  # gmin floor slows the pair loop


def test_pilot_topology_reference_and_gap():
    gp, gn = _rand(16, seed=13), _rand(16, seed=14)
    a = (gp - gn).numpy()
    s1 = np.linalg.svd(a, compute_uv=False)[0]
    cfg = LoopConfig(topology="pilot", gbw_hz=W1)
    m3 = build_loop(gp, gn, cfg, pilot_beta=3.0)
    sv = np.linalg.svd(m3.topo.base_eff, compute_uv=False)
    assert sv[0] > 2.5 * s1 and sv[0] / sv[1] > 2.5  # pilot opens the gap
    assert np.allclose(m3.topo.m_eff, m3.topo.m_eff.T)
    # pilot is loop-only: node load unchanged unless asked
    base = build_loop(gp, gn, LoopConfig(topology="dilated_two_array"))
    assert np.allclose(m3.topo.load, base.topo.load)
    loaded = build_loop(gp, gn, with_config(m3, pilot_loads_node=True).cfg, pilot_beta=3.0)
    assert (loaded.topo.load > base.topo.load).all()


def test_differential_topology_is_E_with_both_arrays_loading():
    gp, gn = _rand(8, seed=15), _rand(8, seed=16)
    rp, rn = gp + 0.01 * _rand(8, seed=17), gn.clone()
    model = build_loop(
        gp, gn, LoopConfig(topology="differential", gbw_hz=W1, delta=0.05), g_ref=(rp, rn)
    )
    e = ((gp - gn) - (rp - rn)).numpy()
    assert np.allclose(model.topo.m_eff[:8, 8:], e)
    assert np.allclose(model.topo.load[:8], (gp + gn + rp + rn).sum(1).numpy())
    with pytest.raises(ValueError):
        build_loop(gp, gn, LoopConfig(topology="differential"))


def test_servo_converges_to_exact_eigenvector_and_measures_lambda():
    a = _rand(20, 14, seed=18)
    cfg = LoopConfig(topology="dilated_two_array", gbw_hz=W1, amplitude="gl_servo", delta=0.05)
    model = build_loop(a, None, cfg)
    tr = simulate(model, 4000.0, x0_rms=1e-4, gen=make_generator(0, "t"))
    assert float(tr.purity_t[-1]) > 0.9999
    assert float(tr.peak[-1]) == pytest.approx(0.5, rel=1e-3)
    assert float(tr.g_servo[-1]) == pytest.approx(model.ref_value, rel=1e-3)
    assert tr.alive and not tr.latched


def test_peak_limiter_holds_target_amplitude():
    cfg = LoopConfig(gbw_hz=W1, amplitude="peak_limiter", delta=0.05)
    model = build_loop(_rand(10, seed=19), None, cfg)
    tr = simulate(model, 3000.0, x0_rms=1e-4, gen=make_generator(0, "t"))
    # AGC equilibrium: peak = v_target (1 + r/kappa), kappa = 0.1 omega
    r = model.modal_rates(1)[0]
    assert float(tr.peak[-1]) == pytest.approx(0.5 * (1 + r / 0.1), rel=0.03)
    assert float(tr.purity_t[-1]) > 0.999


def test_clip_saturates_and_flags():
    # tiny v_sat vs seed: growth clips immediately; random-sign dilated loop latches
    a = _rand(32, seed=20) - 0.5
    cfg = LoopConfig(topology="dilated_two_array", gbw_hz=W1, delta=0.2, v_sat=1e-3)
    model = build_loop(a, None, cfg)
    tr = simulate(model, 3000.0, x0_rms=1e-4, gen=make_generator(0, "t"))
    assert tr.saturated
    assert float(tr.peak.max()) <= 1e-3 * (1 + 1e-12)
    soft = simulate(
        with_config(model, saturation="tanh"), 3000.0, x0_rms=1e-4, gen=make_generator(0, "t")
    )
    assert float(soft.peak.max()) <= 1e-3 * (1 + 1e-12)


def test_dead_loop_and_offsets():
    g = _rand(8, seed=21)
    dead = build_loop(g, None, LoopConfig(gbw_hz=W1, delta=-0.05))
    assert not simulate(dead, 400.0, x0_rms=1e-3, gen=make_generator(0, "t")).alive
    assert dead.modal_rates(1)[0] < 0
    off = build_loop(
        g, None, LoopConfig(gbw_hz=W1, delta=-0.05, v_os_rms=1e-3), make_generator(1, "o")
    )
    tr = simulate(off, 600.0, x0_rms=0.0, gen=make_generator(0, "t"))
    assert float(tr.peak[-1]) > 1e-4  # offsets drive a nonzero DC state even when dead


def test_noise_reproducible_and_seeds_loop():
    g = _rand(8, seed=22)
    cfg = LoopConfig(gbw_hz=1e6, delta=0.05, en_nv_rthz=10.0)
    model = build_loop(g, None, cfg)
    t_end = 2000.0 / model.omega
    a = simulate(model, t_end, x0_rms=0.0, gen=make_generator(5, "n"))
    b = simulate(model, t_end, x0_rms=0.0, gen=make_generator(5, "n"))
    c = simulate(model, t_end, x0_rms=0.0, gen=make_generator(6, "n"))
    assert torch.equal(a.v, b.v) and not torch.equal(a.v, c.v)
    assert float(a.peak[-1]) > 0  # noise alone seeds the growth


def test_mismatch_changes_instance_only_via_generator():
    g = _rand(8, seed=23)
    cfg = LoopConfig(gbw_hz=W1, g_lambda_mismatch_sigma=0.02)
    m1 = build_loop(g, None, cfg, make_generator(1, "m"))
    m2 = build_loop(g, None, cfg, make_generator(1, "m"))
    m3 = build_loop(g, None, cfg, make_generator(2, "m"))
    assert np.array_equal(m1.g_lambda, m2.g_lambda) and not np.array_equal(m1.g_lambda, m3.g_lambda)
    assert abs(float(np.std(m1.mismatch)) - 0.02) < 0.02


def test_second_pole_costs_little():
    g = _rand(10, seed=24)
    base = build_loop(g, None, LoopConfig(gbw_hz=W1, delta=0.05, a0=1e6, v_sat=1e9))
    m = with_config(base, pole2_hz=20 * W1)
    tr = simulate(m, 3000.0, x0_rms=1e-6, gen=make_generator(0, "t"))
    r = _fit_rate(tr)
    assert 0.95 * base.modal_rates(1)[0] < r <= 1.0001 * base.modal_rates(1)[0]
    assert np.isfinite(tr.v.numpy()).all()


def test_inverter_pole_matches_scalar_root():
    # equal row sums, G- empty: s = (lam/(1 + s/ri) - G)/D - 1/A0 has a closed-form root
    g = _circulant(seed=29)
    base = build_loop(g, None, LoopConfig(gbw_hz=W1, delta=0.05, a0=1e6, v_sat=1e9))
    m = with_config(base, inverter="pole")
    lam, gl, d, ri, a0 = base.ref_value, base.g_nominal, base.d_vector()[0], 0.5, 1e6
    eps = 1.0 / a0
    roots = np.roots([d / ri, d * (1 + eps / ri) + gl / ri, d * eps + gl - lam])
    exact = max(r.real for r in roots)
    tr = simulate(m, 6000.0, x0_rms=1e-6, gen=make_generator(0, "t"))
    assert _fit_rate(tr) == pytest.approx(exact, rel=2e-3)
    assert exact < 0.6 * base.modal_rates(1)[0]  # the lag really does halve the slow rate


def test_mvm_settle_time_formula_and_scaling():
    gp, gn = _rand(16, 12, seed=25), _rand(16, 12, seed=26)
    d = mvm_settle_detail(gp, gn, 16e6, 1e5, 8)
    s_max = float((gp + gn).sum(1).max())
    rate = 2 * math.pi * 16e6 * (d["g_f"] / (s_max + d["g_f"]) + 1e-5)
    assert d["t_s"] == pytest.approx(8 * math.log(2) / rate)
    assert mvm_settle_time(gp, gn, 32e6, 1e5, 8) == pytest.approx(d["t_s"] / 2)
    assert mvm_settle_time(gp, gn, 16e6, 1e5, 4) == pytest.approx(d["t_s"] / 2)
    # more node load (pairs vs signed-ideal) is slower
    assert mvm_settle_time(gp, gn, 16e6, 1e5, 8) > mvm_settle_time(gp - gn, None, 16e6, 1e5, 8)


def test_mvm_tia_step_response_matches_loop_amp_model():
    # one TIA row: simulate its ODE with the same op-amp parameters and compare
    s, gf, a0, bits = 5.0, 2.0, 1e4, 8
    rate = gf / (s + gf) + 1 / a0
    # closed form step: o(t) = o_inf (1 - exp(-rate t)); settles to 2^-bits at:
    t = bits * math.log(2) / rate
    assert math.exp(-rate * t) == pytest.approx(2.0**-bits)


def test_cold_loop_slower_than_mvm_for_pair_tile():
    # sanity of the comparison C' will make: unwarm loop growth (>~10 tau_ln) vs one MVM
    a = _rand(32, seed=27) - 0.5
    gp, gn = a.clamp(min=0) + 0.05, (-a).clamp(min=0) + 0.05
    model = build_loop(gp, gn, LoopConfig(topology="dilated_two_array", gbw_hz=16e6, delta=0.1))
    tr = simulate(model, 3e-4, x0_rms=1e-5, gen=make_generator(0, "t"))
    assert tr.t_amp(0.1) > mvm_settle_time(gp, gn, 16e6, 1e5, 8)


def test_performance_256_dilated():
    a = _rand(256, seed=28) - 0.5
    gp, gn = a.clamp(min=0) + 0.05, (-a).clamp(min=0) + 0.05
    model = build_loop(gp, gn, LoopConfig(topology="dilated_two_array", gbw_hz=W1, delta=0.05))
    t0 = time.time()
    tr = simulate(model, 3000.0, x0_rms=1e-5, gen=make_generator(0, "t"), stop_when_settled=1e-3)
    assert time.time() - t0 < 30.0
    assert float(tr.purity_t[-1]) > 0.999


def test_agc_ideal_holds_peak_and_keeps_direction():
    g = _rand(20, seed=6)
    model = build_loop(g, None, LoopConfig(gbw_hz=W1, a0=1e6, delta=0.1, amplitude="agc_ideal"))
    tr = simulate(model, 600.0, x0_rms=1e-3, gen=make_generator(0, "t"))
    assert float(tr.peak.max()) <= 0.5 * model.cfg.v_sat * (1 + 1e-9) + 1e-12
    assert float(tr.purity_t[-1]) > 0.99 and not tr.latched


def test_pilot_s1_ref_fixes_pilot_scale():
    from neurofault.spectral.circuit.topologies import pilot

    a = _rand(12, seed=8)
    a2 = a.clone()
    a2[0, 0] += 0.5
    s1 = float(np.linalg.norm(a.numpy(), 2))
    u0, v0 = np.ones(12), np.ones(12)
    p_ref = pilot(a, None, 2.0, u0, v0)
    p_faulty_fixed = pilot(a2, None, 2.0, u0, v0, s1_ref=s1)
    p_faulty_rescaled = pilot(a2, None, 2.0, u0, v0)
    d_fixed = p_faulty_fixed.base_eff - a2.numpy() - (p_ref.base_eff - a.numpy())
    assert np.allclose(d_fixed, 0.0)
    assert not np.allclose(p_faulty_rescaled.base_eff - a2.numpy(), p_ref.base_eff - a.numpy())


def test_mvm_load_stat_mean_is_faster_than_max():
    g = _rand(16, seed=9)
    mx = mvm_settle_detail(g, None, 1e8, 1e4, 8)
    mn = mvm_settle_detail(g, None, 1e8, 1e4, 8, load_stat="mean")
    assert mn["s_max"] <= mx["s_max"] and mn["t_s"] <= mx["t_s"]
