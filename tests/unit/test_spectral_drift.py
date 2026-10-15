"""spectral.drift: persistence, determinism, formula checks, D0 parity."""

from __future__ import annotations

import math

import pytest
import torch

from neurofault.config import FaultConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.faults.injection import inject_faults
from neurofault.faults.reliability import arrhenius_acceleration_factor
from neurofault.spectral.drift import (
    ComposedDrift,
    LegacyRailShift,
    PCMPerDevice,
    PeripheryGain,
    PowerLawUniform,
    RRAMRelaxation,
    StructuredNu,
    ThermalGain,
    structured_field,
)
from neurofault.spectral.faults import inject
from neurofault.spectral.params import PARAMS, dump_params
from neurofault.spectral.types import (
    T0_S,
    ConductanceState,
    DriftModel,
    EncodingSpec,
    FaultSpec,
    make_generator,
)
from tests.unit.fakes import make_handle

TS = [60.0, 3.6e3, 8.64e4, 2.6e6, 3.15e7, 3.15e8]


def make_state(kind="diff_pair", m=24, n=20, g_min_frac=0.0, seed=0, g_const=None):
    enc = EncodingSpec(kind=kind, g_min_frac=g_min_frac)
    w = torch.randn(m, n, dtype=torch.float64, generator=make_generator(seed, "w"))
    absmax = float(w.abs().max())
    if kind == "diff_pair":
        s = (enc.g_max_uS - enc.g_min_uS) / absmax
        gp = enc.g_min_uS + w.clamp(min=0) * s
        gn = enc.g_min_uS + (-w).clamp(min=0) * s
        if g_const is not None:
            gp, gn = torch.full_like(gp, g_const), torch.full_like(gn, g_const)
        return ConductanceState(gp, gn, s, enc)
    if kind == "offset_refcol":
        s = 0.5 * (enc.g_max_uS - enc.g_min_uS) / absmax
        return ConductanceState(0.5 * (enc.g_min_uS + enc.g_max_uS) + w * s, None, s, enc)
    s = enc.g_max_uS / absmax
    return ConductanceState(w * s, None, s, enc)


def gen(seed=0, stream="drift"):
    return make_generator(seed, stream)


def nu_of(state0, st_t, t):
    """Per-device effective exponent recovered from G(t)/G0 (pos device)."""
    return -torch.log(st_t.g_pos / state0.g_pos) / math.log((t + T0_S) / T0_S)


# ------------------------------------------------------------------ persistence / determinism


@pytest.mark.parametrize(
    "model", [PCMPerDevice(), StructuredNu("rank_r", 2.0), StructuredNu("hotspot", 1.0)]
)
def test_nu_persists_across_t(model):
    st0 = make_state(g_min_frac=0.05)
    real = model.sample(st0, gen())
    ref = None
    # devices that never touch the g_min clamp, where nu is directly recoverable
    mask = real.at(TS[3]).g_pos > st0.enc.g_min_uS * 1.001
    for t in TS[:4]:
        nu = nu_of(st0, real.at(t), t)[mask]
        if ref is None:
            ref = nu
        assert torch.allclose(nu, ref, atol=1e-9)


def test_determinism_by_generator():
    st0 = make_state(g_min_frac=0.05)
    a = PCMPerDevice().sample(st0, gen(3)).at(1e5)
    b = PCMPerDevice().sample(st0, gen(3)).at(1e5)
    c = PCMPerDevice().sample(st0, gen(4)).at(1e5)
    assert torch.equal(a.g_pos, b.g_pos) and torch.equal(a.g_neg, b.g_neg)
    assert not torch.equal(a.g_pos, c.g_pos)
    real = PCMPerDevice().sample(st0, gen(3))
    assert torch.equal(real.at(1e4).g_pos, real.at(1e4).g_pos)  # at() has no hidden RNG


def test_at_zero_is_identity_and_sweep_matches_at():
    st0 = make_state(g_min_frac=0.05)
    for model in (
        PowerLawUniform(),
        PCMPerDevice(),
        StructuredNu("row_gradient", 1.0),
        RRAMRelaxation("mean_revert"),
        PeripheryGain(),
    ):
        real = model.sample(st0, gen())
        s0 = real.at(0.0)
        assert torch.allclose(s0.g_pos, st0.g_pos) and s0.t == 0.0
        sw = real.sweep(TS)
        for t, s in zip(TS, sw):
            assert s.t == t and torch.equal(s.g_pos, real.at(t).g_pos)
    assert isinstance(PCMPerDevice(), DriftModel)


def test_state0_not_modified():
    st0 = make_state(g_min_frac=0.05)
    g, gn = st0.g_pos.clone(), st0.g_neg.clone()
    PCMPerDevice().sample(st0, gen()).at(1e6)
    assert torch.equal(st0.g_pos, g) and torch.equal(st0.g_neg, gn)


# ------------------------------------------------------------------ stuck, clamp, both devices


def test_stuck_devices_excluded_from_drift():
    st0 = make_state(g_min_frac=0.05)
    st1, _ = inject(st0, FaultSpec("stuck_random", k=40, rail="lrs"), gen(1, "faults"))
    for model in (
        PowerLawUniform(),
        PCMPerDevice(),
        RRAMRelaxation("cmo"),
        RRAMRelaxation("mean_revert"),
        LegacyRailShift(0.5),
        ThermalGain(PCMPerDevice(), dT_kind="uniform"),
    ):
        out = model.sample(st1, gen()).at(1e7)
        pos = st1.stuck[..., 0]
        neg = st1.stuck[..., 1]
        assert torch.equal(out.g_pos[pos], st1.g_pos[pos]), model.name
        assert torch.equal(out.g_neg[neg], st1.g_neg[neg]), model.name
        assert out.stuck.equal(st1.stuck)
    # and non-stuck devices did drift
    out = PCMPerDevice().sample(st1, gen()).at(1e7)
    assert (out.g_pos[~st1.stuck[..., 0]] < st1.g_pos[~st1.stuck[..., 0]]).any()


def test_clamped_to_rails_and_both_devices_drift():
    st0 = make_state(g_min_frac=0.05)
    out = PCMPerDevice().sample(st0, gen()).at(3.15e8)
    for g in (out.g_pos, out.g_neg):
        assert g.min() >= st0.enc.g_min_uS - 1e-12 and g.max() <= st0.enc.g_max_uS + 1e-12
    # idle devices at g_min=1.25 uS drift too (then clamp back to g_min): the larger
    # devices certainly moved on both polarities
    assert (out.g_neg < st0.g_neg).any() and (out.g_pos < st0.g_pos).any()


def test_offset_and_signed_encodings():
    for kind in ("offset_refcol", "signed_ideal"):
        st0 = make_state(kind, g_min_frac=0.05 if kind == "offset_refcol" else 0.0)
        out = PCMPerDevice().sample(st0, gen()).at(1e6)
        assert out.g_neg is None
        if kind == "signed_ideal":  # multiplicative, sign preserved, not clamped
            assert torch.all(torch.sign(out.g_pos) == torch.sign(st0.g_pos))
            assert torch.all(out.g_pos.abs() <= st0.g_pos.abs() + 1e-12)
        else:
            assert out.g_pos.min() >= st0.enc.g_min_uS - 1e-12


# ------------------------------------------------------------------ D1 / D2


def test_d1_uniform_scalar_compensation_exact():
    st0 = make_state(g_min_frac=0.0)
    w0 = st0.weights()
    for t in TS:
        st = PowerLawUniform().sample(st0, gen()).at(t)
        f = ((t + T0_S) / T0_S) ** -0.06
        assert torch.allclose(st.weights(), f * w0, atol=1e-14)
        resid = (st.weights() / f - w0).abs().max() / w0.abs().max()
        assert resid < 1e-3
    # scalar fit by least squares also leaves < 1e-3 residual
    st = PowerLawUniform().sample(st0, gen()).at(3.15e7)
    alpha = (st.weights() * w0).sum() / (w0 * w0).sum()
    assert (st.weights() / alpha - w0).abs().max() / w0.abs().max() < 1e-3


@pytest.mark.parametrize(
    "g,mu,sig", [(12.5, 0.049, 0.008), (0.25, 0.0957, 0.045), (25.0, 0.049, 0.008)]
)
def test_d2_distribution_matches_formula(g, mu, sig):
    enc = EncodingSpec(g_min_frac=0.0)
    n = 300
    gp = torch.full((n, n), g, dtype=torch.float64)
    st0 = ConductanceState(gp, torch.zeros_like(gp), 1.0, enc)
    t = 1e6
    out = PCMPerDevice().sample(st0, gen(11)).at(t)
    nu = nu_of(st0, out, t)
    # independent evaluation of the registered formulas (natural log, clips)
    lg = math.log(max(g / 25.0, 1e-7))
    mu_f = min(max(-0.0155 * lg + 0.0244, 0.049), 0.1)
    sg_f = min(max(-0.0125 * lg - 0.0059, 0.008), 0.045)
    assert mu_f == pytest.approx(mu, abs=2e-4) and sg_f == pytest.approx(sig, abs=2e-4)
    assert float(nu.mean()) == pytest.approx(
        mu_f, abs=2e-3
    )  # |.| fold-over biases the mean slightly
    assert float(nu.std()) == pytest.approx(sg_f, rel=0.05)


def test_d2_off_device_at_zero_stays_zero():
    st0 = make_state(g_min_frac=0.0)
    out = PCMPerDevice().sample(st0, gen()).at(1e6)
    zero = st0.g_neg == 0
    assert torch.all(out.g_neg[zero] == 0)


def test_d2_drift_scale():
    st0 = make_state(g_min_frac=0.0)
    a = PCMPerDevice(drift_scale=0.0).sample(st0, gen()).at(1e6)
    assert torch.allclose(a.g_pos, st0.g_pos)


# ------------------------------------------------------------------ D3


@pytest.mark.parametrize(
    "field", ["row_gradient", "col_gradient", "rank_r", "row_plus_col", "hotspot"]
)
def test_d3_rho0_equals_d2_same_seed(field):
    st0 = make_state(g_min_frac=0.05)
    d2 = PCMPerDevice().sample(st0, gen(5)).at(1e6)
    d3 = StructuredNu(field, 0.0).sample(st0, gen(5)).at(1e6)
    assert torch.equal(d2.g_pos, d3.g_pos) and torch.equal(d2.g_neg, d3.g_neg)


def test_d3_rho_changes_result_and_field_properties():
    st0 = make_state(g_min_frac=0.05)
    d2 = PCMPerDevice().sample(st0, gen(5)).at(1e6)
    d3 = StructuredNu("row_plus_col", 4.0).sample(st0, gen(5)).at(1e6)
    assert not torch.allclose(d2.g_pos, d3.g_pos)
    g = gen(2)
    rg = structured_field("row_gradient", 16, 12, g)
    assert torch.allclose(rg, rg[:, :1].expand_as(rg))  # constant along columns
    assert abs(float(rg.mean())) < 1e-12 and float(rg.std(correction=0)) == pytest.approx(1.0)
    cg = structured_field("col_gradient", 16, 12, g)
    assert torch.allclose(cg, cg[:1].expand_as(cg))
    for r in (1, 2, 3):
        f = structured_field("rank_r", 30, 28, gen(r), rank_r=r)
        assert torch.linalg.matrix_rank(f, atol=1e-9).item() <= r
        assert float(f.pow(2).mean()) == pytest.approx(1.0)
    rpc = structured_field("row_plus_col", 16, 12, g)
    assert torch.linalg.matrix_rank(rpc, atol=1e-9).item() <= 2
    hs = structured_field("hotspot", 30, 30, g)
    assert float(hs.max()) > 2.0  # localised bump
    with pytest.raises(ValueError):
        structured_field("nope", 4, 4, g)


def test_d3_structured_nu_error_is_lower_rank_than_iid():
    """The point of D3: structured exponent => low-rank residual after scalar GDC."""
    st0 = make_state(m=64, n=64, g_min_frac=0.0, g_const=None)
    t = 1e7

    def resid_energy_top1(model):
        out = model.sample(st0, gen(2)).at(t)
        w0, w = st0.weights(), out.weights()
        alpha = (w * w0).sum() / (w0 * w0).sum()
        e = w / alpha - w0
        s = torch.linalg.svdvals(e)
        return float(s[0] ** 2 / (s**2).sum())

    assert resid_energy_top1(StructuredNu("rank_r", 4.0, rank_r=1)) > resid_energy_top1(
        PCMPerDevice()
    )


# ------------------------------------------------------------------ D0 parity


@pytest.mark.parametrize("direction", ["toward_hrs", "toward_lrs", "random_per_cell"])
@pytest.mark.parametrize("distribution", ["random", "clustered", "gradient"])
def test_d0_parity_with_legacy_drift(direction, distribution):
    shape = (14, 11)
    handle = make_handle(shape)  # r_on=100, r_off=10000 -> g in [1e-4, 1e-2] S
    mon = HealthMonitor(handle)
    base = mon.g_min + torch.rand(shape, generator=gen(9, "m"), dtype=torch.float32) * mon.g_range
    handle.accessor.write(base.clone())
    cfg = FaultConfig(
        density=0.3,
        failure_model="drift",
        distribution=distribution,
        drift_direction=direction,
        drift_magnitude=0.2,
    )
    inject_faults(handle, mon, cfg, generator=gen(5, "legacy"))
    legacy = handle.accessor.read().double() * 1e6  # S -> uS

    enc = EncodingSpec("offset_refcol", g_max_uS=1e6 * mon.g_max, g_min_frac=mon.g_min / mon.g_max)
    st0 = ConductanceState(base.double() * 1e6, None, 1.0, enc)
    out = LegacyRailShift(0.3, 0.2, direction, distribution).sample(st0, gen(5, "legacy")).at(1.0)
    assert torch.allclose(out.g_pos, legacy, rtol=1e-5, atol=1e-2)
    assert (out.g_pos != st0.g_pos).sum() > 0
    assert (((out.g_pos - st0.g_pos).abs() > 1e-2) == ((legacy - st0.g_pos).abs() > 1e-2)).all()
    # t = 0 is the identity
    assert torch.equal(
        LegacyRailShift(0.3, 0.2, direction, distribution).sample(st0, gen(5)).at(0.0).g_pos,
        st0.g_pos,
    )


# ------------------------------------------------------------------ D4 / D5 / D7 / composition


def test_d4_time_acceleration_and_gain():
    st0 = make_state(g_min_frac=0.05)
    af = arrhenius_acceleration_factor(330.0, 300.0, 0.5)
    assert af > 1
    d2 = PCMPerDevice()
    plain = d2.sample(st0, gen(3)).at(af * 1e4)
    hot = ThermalGain(d2, T_K=330.0, T_ref_K=300.0, Ea_eV=0.5, beta_per_K=0.0).sample(st0, gen(3))
    assert torch.allclose(hot.at(1e4).g_pos, plain.g_pos)
    # no acceleration at T == T_ref
    cold = ThermalGain(d2, T_K=300.0, T_ref_K=300.0, beta_per_K=0.0).sample(st0, gen(3))
    assert torch.allclose(cold.at(1e4).g_pos, d2.sample(st0, gen(3)).at(1e4).g_pos)
    # uniform gain scales effective conductances (no base)
    g = ThermalGain(None, dT_amp_K=20.0, beta_per_K=1e-3, dT_kind="uniform").sample(st0, gen())
    assert torch.allclose(g.at(0.0).g_pos, st0.g_pos * 1.02)
    # all dT shapes build, gain positive and non-uniform for gradients
    for kind in ("uniform", "row_gradient", "hotspot", "mixed"):
        out = ThermalGain(None, dT_kind=kind).sample(st0, gen()).at(0.0)
        assert torch.all(out.g_pos >= 0)
    with pytest.raises(ValueError):
        ThermalGain(None, dT_kind="x").sample(st0, gen())


def test_d5_cmo_matches_source_formula_and_persists():
    st0 = make_state(g_min_frac=0.0, g_const=12.5)
    real = RRAMRelaxation("cmo").sample(st0, gen(4))
    ts = [1e2, 1e4, 1e6]
    z = None
    for t in ts:
        out = real.at(t)
        lnt = math.log(t)
        mean = -0.08900206 * lnt
        sigma = 0.04201137 * lnt + 0.41183342
        d = (out.g_pos - 12.5 - mean) / sigma
        if z is None:
            z = d
            assert abs(float(d.mean())) < 0.15 and float(d.std()) == pytest.approx(1.0, rel=0.1)
        assert torch.allclose(d, z, atol=1e-9)  # same z at every t
    # level independence: shift does not depend on G0 (away from the rails)
    st1 = make_state(g_min_frac=0.0, g_const=6.0)
    out = RRAMRelaxation("cmo", drift_scale=1.0).sample(st1, gen(4)).at(1e4)
    assert float((out.g_pos - 6.0).mean()) == pytest.approx(-0.08900206 * math.log(1e4), abs=0.05)


def test_d5_mean_revert_shrinks_toward_mean():
    st0 = make_state(g_min_frac=0.0)
    out = RRAMRelaxation("mean_revert", c=0.05).sample(st0, gen()).at(1e6)
    lam = 0.05 * math.log1p(1e6 / T0_S)
    w0, w = st0.weights(), out.weights()
    wbar = w0.mean()
    assert torch.allclose(w - wbar, (1 - lam) * (w0 - wbar), atol=1e-12)
    with pytest.raises(ValueError):
        RRAMRelaxation("bad")


def test_d7_periphery_gain_per_line_and_exact():
    st0 = make_state(g_min_frac=0.05)
    real = PeripheryGain(sigma_p=0.05, axis="out").sample(st0, gen(2))
    out = real.at(1e6)
    ratio = (out.weights() / st0.weights()).nan_to_num(1.0)
    w0 = st0.weights()
    ok = w0.abs() > 1e-3
    # same gain along each row (output line); gains differ across rows
    for i in range(w0.shape[0]):
        r = (out.weights()[i] / w0[i])[ok[i]]
        assert torch.allclose(r, r[0].expand_as(r), atol=1e-9)
    assert ratio.std() > 0
    g_in = PeripheryGain(sigma_p=0.05, axis="in").sample(st0, gen(2)).at(1e6).weights() / w0
    col = g_in[:, 0][ok[:, 0]]
    assert torch.allclose(col, col[0].expand_as(col), atol=1e-9)
    # acts on stuck devices too (post-array)
    st1, _ = inject(st0, FaultSpec("stuck_random", k=30), gen(1, "f"))
    o1 = PeripheryGain(sigma_p=0.1).sample(st1, gen(2)).at(1e6)
    assert not torch.equal(o1.g_pos[st1.stuck[..., 0]], st1.g_pos[st1.stuck[..., 0]])
    with pytest.raises(ValueError):
        PeripheryGain(axis="z")


def test_composed_equals_sequential_and_deterministic():
    st0 = make_state(g_min_frac=0.05)
    comp = ComposedDrift([PCMPerDevice(), PeripheryGain(sigma_p=0.05)])
    a = comp.sample(st0, gen(6)).at(1e6)
    b = comp.sample(st0, gen(6)).at(1e6)
    assert torch.equal(a.g_pos, b.g_pos)
    # manual: same generator, same order
    g = gen(6)
    d2 = PCMPerDevice().sample(st0, g)
    d7 = PeripheryGain(sigma_p=0.05).sample(st0, g)
    manual = d7.apply(d2.apply(torch.stack([st0.g_pos, st0.g_neg], -1), 1e6), 1e6)
    assert torch.allclose(a.g_pos, manual[..., 0]) and torch.allclose(a.g_neg, manual[..., 1])
    assert comp.name == "D2_pcm_per_device+D7_periphery_out"
    assert torch.equal(comp.sample(st0, gen(6)).at(0.0).g_pos, st0.g_pos)


# ------------------------------------------------------------------ params


def test_params_registry():
    d = dump_params()
    assert set(d) == set(PARAMS)
    for k, r in PARAMS.items():
        assert r.status in ("sourced", "assumed")
        assert r.source
        if r.status == "assumed":
            assert r.sweep and not r.verified, k
    assert PARAMS["d2_mu_slope"].value == -0.0155 and PARAMS["d2_mu_slope"].verified
    assert PARAMS["t0_s"].value == T0_S
    import json

    json.dumps(d)
