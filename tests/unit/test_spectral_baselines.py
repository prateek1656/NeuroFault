"""spectral.baselines: each compensator recovers its own model class, exact cost accounting."""

from __future__ import annotations

import pytest
import torch

from neurofault.spectral.baselines import (
    GDC,
    AdaBS,
    BiScale,
    ChecksumABFT,
    FixedPeriodRecal,
    FullReadoutRecal,
    GDCJoshi,
    NoCompensation,
    OracleExact,
    PerColumn,
    PerRow,
    ProbeLowRank,
    ProbeResidual,
    TimerSchedule,
    aihwkit_per_column,
    bind_all,
)
from neurofault.spectral.drift import PowerLawUniform
from neurofault.spectral.encoding import encode
from neurofault.spectral.faults import inject
from neurofault.spectral.readout import TileReader
from neurofault.spectral.tiles import synthetic_tile
from neurofault.spectral.types import (
    DTYPE,
    ConductanceState,
    EncodingSpec,
    FaultSpec,
    ReadNoise,
    make_generator,
)

M, N = 24, 20
QUIET = ReadNoise(mode="per_cell", sigma_cell_frac=0.0, adc_bits=None, dac_bits=None)


def tile_w(seed=0):
    return synthetic_tile(M, N, spectrum="gaussian", gen=make_generator(seed, "t"), std=0.05).w


def rel(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


class Rig:
    """A tile whose current weights come from a mutable matrix; reader is noise-free."""

    def __init__(self, w0, backward=True, noise=QUIET):
        self.w0, self.w = w0.clone(), w0.clone()
        enc = EncodingSpec(kind="signed_ideal")
        self.state0 = encode(type("T", (), {"w": w0})(), enc)
        self.reader = TileReader(
            lambda: ConductanceState(self.w * self.state0.scale, None, self.state0.scale, enc),
            noise,
            make_generator(1, "read"),
            backward_supported=backward,
        )

    def methods(self, *ms):
        bind_all(ms, lambda: self.w, lambda: self.w0)
        return ms


def run(method, rig, t=1.0):
    cal = method.calibrate(rig.reader)
    before = rig.reader.cost.copy()
    corr = method.update(rig.reader, t)
    assert rig.reader.cost - before == corr.cost  # reported cost == counted reads
    return cal, corr


def test_gdc_exact_on_uniform_power_law():
    w0 = tile_w()
    enc = EncodingSpec(kind="diff_pair", g_min_frac=0.0)
    st = encode(type("T", (), {"w": w0})(), enc)
    real = PowerLawUniform().sample(st, make_generator(0, "d"))
    cur = {"s": st}
    reader = TileReader(lambda: cur["s"], QUIET, make_generator(0, "r"))
    for v in ("aihwkit", "ls", "exact_ref"):
        g = GDC(v)
        bind_all([g], lambda: cur["s"].weights(), lambda: w0)
        cur["s"] = st
        g.calibrate(reader)
        cur["s"] = real.at(8.64e4)
        corr = g.update(reader, 8.64e4)
        assert rel(corr.W_eff, w0) < 1e-9, v
    assert rel(cur["s"].weights(), w0) > 0.1  # drift was real


def test_gdc_costs_exact():
    rig = Rig(tile_w())
    (g,) = rig.methods(GDC("aihwkit"))
    cal, corr = run(g, rig)
    assert cal.mvm_fwd == N and cal.adc == M * N and cal.dac == N * N
    assert corr.cost.mvm_fwd == N
    (g1,) = rig.methods(GDC("ones"))
    cal, corr = run(g1, rig)
    assert (cal.mvm_fwd, corr.cost.mvm_fwd, corr.cost.adc) == (1, 1, M)


def test_gdc_aihwkit_parity():
    torch.manual_seed(0)
    comp = pytest.importorskip("aihwkit.inference.compensation.drift")
    w0 = tile_w() * 10.0  # aihwkit clamps at 1e-4 in normalised units
    rig = Rig(w0)
    rig.w = w0 * (0.7 + 0.2 * torch.rand(M, N, dtype=DTYPE))
    (g,) = rig.methods(GDC("aihwkit"))
    rig.w, keep = w0, rig.w
    g.calibrate(rig.reader)
    rig.w = keep
    alpha = g.update(rig.reader, 1.0).meta["alpha"]
    ref = comp.GlobalDriftCompensation()
    ref_val = ref.readout(w0.T.float())  # eye(n) @ W^T
    a_ref = float(ref.apply(keep.T.float(), ref_val))
    assert alpha == pytest.approx(a_ref, rel=1e-5)
    pc = comp.PerColumnDriftCompensation()
    pr = aihwkit_per_column()
    rig.w = w0
    pr.bind(lambda: keep, None)
    pr.calibrate(rig.reader)
    rig.w = keep
    g_row = pr.update(rig.reader, 1.0).meta["row_gain"]
    a_pc = pc.apply(keep.T.float(), pc.readout(w0.T.float()))
    assert torch.allclose(g_row.float(), a_pc, rtol=1e-5)


def test_joshi_counts_and_exactness():
    rig = Rig(tile_w())
    g = GDCJoshi(8)
    rig.methods(g)
    decay = {"v": 1.0}
    g.bind_reference(lambda t: 3.0 * decay["v"])
    cal = g.calibrate(rig.reader)
    decay["v"] = 0.5
    rig.w = rig.w0 * 0.5
    corr = g.update(rig.reader, 1.0)
    assert rel(corr.W_eff, rig.w0) < 1e-12 and cal.mvm_fwd == 1 and corr.cost.adc == 8


def test_per_row_pure_row_gradient_and_cost():
    w0 = tile_w()
    rig = Rig(w0)
    s = torch.linspace(0.6, 1.0, M, dtype=DTYPE)
    (pr,) = rig.methods(PerRow(probes="rademacher", p=3))
    pr.calibrate(rig.reader)
    rig.w = w0 * s[:, None]
    before = rig.reader.cost.copy()
    corr = pr.update(rig.reader, 1.0)
    assert rel(corr.W_eff, w0) < 1e-10
    assert corr.cost.mvm_fwd == 3 and (rig.reader.cost - before).mvm_fwd == 3
    # a global scalar cannot fix a row gradient
    g = rig.methods(GDC("ls"))[0]
    rig.w = w0
    g.calibrate(rig.reader)
    rig.w = w0 * s[:, None]
    assert rel(g.update(rig.reader, 1.0).W_eff, w0) > 0.05


def test_per_column_backward_and_forward_fallback():
    w0 = tile_w()
    c = torch.linspace(0.5, 1.2, N, dtype=DTYPE)
    for backward in (True, False):
        rig = Rig(w0, backward=backward)
        (pc,) = rig.methods(PerColumn(p=3))
        cal = pc.calibrate(rig.reader)
        rig.w = w0 * c[None, :]
        corr = pc.update(rig.reader, 1.0)
        assert rel(corr.W_eff, w0) < 1e-10
        if backward:
            assert cal.mvm_bwd == 3 and cal.mvm_fwd == 0
        else:
            assert cal.mvm_fwd == N and cal.mvm_bwd == 0


def test_biscale_rank1_multiplicative():
    w0 = tile_w()
    a = 0.6 + 0.4 * torch.rand(M, dtype=DTYPE)
    b = 0.7 + 0.3 * torch.rand(N, dtype=DTYPE)
    rig = Rig(w0)
    (bs,) = rig.methods(BiScale(n_fwd=2, n_bwd=2))
    cal = bs.calibrate(rig.reader)
    rig.w = a[:, None] * w0 * b[None, :]
    corr = bs.update(rig.reader, 1.0)
    assert rel(corr.W_eff, w0) < 1e-9
    assert cal.mvm_fwd == 2 and cal.mvm_bwd == 2
    assert corr.cost.mvm_fwd == 2 and corr.cost.mvm_bwd == 2
    # per-row alone cannot
    (pr,) = rig.methods(PerRow(p=3))
    rig.w = w0
    pr.calibrate(rig.reader)
    rig.w = a[:, None] * w0 * b[None, :]
    assert rel(pr.update(rig.reader, 1.0).W_eff, w0) > 1e-2


def test_probe_residual_localizes_stuck_row_and_costs():
    enc = EncodingSpec(kind="signed_ideal")
    hits = 0
    for seed in range(20):
        w0 = tile_w(seed)
        st = encode(type("T", (), {"w": w0})(), enc)
        bad, truth = inject(
            st, FaultSpec("stuck_random", k=1, rail="lrs"), make_generator(seed, "f")
        )
        cur = {"s": st}
        reader = TileReader(lambda cur=cur: cur["s"], QUIET, make_generator(0, "r"))
        det = ProbeResidual(p=6)
        cal = det.calibrate(reader)
        assert (cal.mvm_fwd, cal.mvm_bwd) == (6, 6)
        s0, _ = det.statistic(reader, 0.0)
        assert s0 == 0.0
        cur["s"] = bad
        s1, c = det.statistic(reader, 1.0)
        assert s1 > 0 and c.mvm_fwd == 6 and c.mvm_bwd == 0
        est, c2 = det.localize(reader, 1.0)
        assert c2.mvm_fwd == 6 and c2.mvm_bwd == 6
        hits += int(est.rows[0] == truth.rows[0]) * int(est.cols[0] == truth.cols[0])
    assert hits == 20


def test_checksum_abft_detects_and_locates():
    enc = EncodingSpec(kind="signed_ideal")
    w0 = tile_w(3)
    st = encode(type("T", (), {"w": w0})(), enc)
    bad, truth = inject(st, FaultSpec("stuck_random", k=1, rail="lrs"), make_generator(3, "f"))
    cur = {"s": st}
    reader = TileReader(lambda: cur["s"], QUIET, make_generator(0, "r"))
    ab = ChecksumABFT()
    cal = ab.calibrate(reader)
    assert (cal.mvm_fwd, cal.mvm_bwd) == (1, 1)
    assert ab.statistic(reader, 0.0)[0] == 0.0
    cur["s"] = bad
    s, c = ab.statistic(reader, 1.0)
    assert s > 0 and (c.mvm_fwd, c.mvm_bwd) == (1, 1)
    est, _ = ab.localize(reader, 1.0)
    assert tuple(est.cells[0].tolist()) == tuple(truth.cells[0].tolist())


def test_probe_lowrank_recovers_low_rank_error():
    w0 = tile_w()
    u = torch.randn(M, 2, generator=make_generator(5, "u"), dtype=DTYPE)
    v = torch.randn(N, 2, generator=make_generator(6, "v"), dtype=DTYPE)
    rig = Rig(w0)
    (pl,) = rig.methods(ProbeLowRank(k=2, s=4))
    cal = pl.calibrate(rig.reader)
    assert cal.mvm_fwd == 6 and cal.mvm_bwd == 6
    rig.w = w0 + 0.05 * u @ v.T
    corr = pl.update(rig.reader, 1.0)
    assert rel(corr.W_eff, w0) < 1e-6
    assert corr.cost.mvm_fwd == 6 and corr.cost.mvm_bwd == 6
    with pytest.raises(RuntimeError):
        ProbeLowRank().calibrate(Rig(w0, backward=False).reader)


def test_timer_oracle_full_readout_nocomp():
    w0 = tile_w()
    rig = Rig(w0)
    nu, t0, t = 0.06, 20.0, 8.64e4
    rig.w = w0 * ((t + t0) / t0) ** (-nu)
    tm, orc, fr, nc = rig.methods(
        TimerSchedule.power_law(nu), OracleExact(), FullReadoutRecal(), NoCompensation()
    )
    c = tm.update(rig.reader, t)
    assert rel(c.W_eff, w0) < 1e-12 and c.cost == type(c.cost)()
    wrong = TimerSchedule.power_law(0.12)
    rig.methods(wrong)
    assert rel(wrong.update(rig.reader, t).W_eff, w0) > 0.05
    assert rel(orc.update(rig.reader, t).W_eff, w0) == 0.0
    cf = fr.update(rig.reader, t)
    assert cf.cost.cell_reads == M * N and rel(cf.W_eff, rig.w) < 1e-12
    assert rel(nc.update(rig.reader, t).W_eff, rig.w) == 0.0


def test_fixed_period_recal_holds_then_refreshes():
    w0 = tile_w()
    rig = Rig(w0)
    fp = FixedPeriodRecal(GDC("ls"), period_s=100.0)
    rig.methods(fp)
    fp.calibrate(rig.reader)
    rig.w = w0 * 0.8
    c1 = fp.update(rig.reader, 10.0)  # before first period: uncorrected, free
    assert c1.cost.mvm_fwd == 0 and rel(c1.W_eff, w0) > 0.1
    c2 = fp.update(rig.reader, 120.0)  # recal
    assert c2.cost.mvm_fwd > 0 and rel(c2.W_eff, w0) < 1e-10
    rig.w = w0 * 0.5
    c3 = fp.update(rig.reader, 150.0)  # held alpha applied to current W: stale
    assert c3.cost.mvm_fwd == 0 and rel(c3.W_eff, w0) > 0.1


def test_adabs_cost():
    a = AdaBS(n_images=512, batch_size=256)
    assert a.planned_cost().images == 512 and a.mvm_equivalents(10) == 5120


def test_unbound_raises():
    rig = Rig(tile_w())
    g = GDC("ls")
    g.calibrate(rig.reader)
    with pytest.raises(RuntimeError):
        g.update(rig.reader, 1.0)
