"""Direction-B localizers: exactness, permutation invariance, exact read counts, A2b regime."""

from __future__ import annotations

import pytest
import torch

from neurofault.spectral.encoding import encode
from neurofault.spectral.faults import inject
from neurofault.spectral.localize import (
    AdaptiveBisectionLocalizer,
    FullReadoutLocalizer,
    GroupTestingLocalizer,
    HPELocalizer,
    NonAdaptiveCSLocalizer,
    OracleLocalizer,
    ResidualLocalizer,
    RothLocalizer,
    TwoStageLocalizer,
)
from neurofault.spectral.localize.common import comp_dd, kautz_singleton
from neurofault.spectral.readout import TileReader
from neurofault.spectral.tiles import synthetic_tile
from neurofault.spectral.types import (
    DTYPE,
    EncodingSpec,
    FaultSpec,
    Localizer,
    ReadNoise,
    make_generator,
)

M = N = 64
ENC = EncodingSpec(kind="offset_refcol", g_min_frac=0.01)
QUIET = ReadNoise(mode="per_cell", sigma_cell_frac=0.0, adc_bits=None, dac_bits=None)


def _setup(spec, seed=0, noise=QUIET, w=None):
    tile = synthetic_tile(M, N, spectrum="gaussian", gen=make_generator(seed, "tiles"))
    if w is not None:
        tile.w = w
    st = encode(tile, ENC)
    bad, truth = inject(st, spec, make_generator(seed, "faults"))
    reader = TileReader(lambda: bad, noise, make_generator(seed, "read"))
    return tile.w, reader, truth


def _keys(cells):
    return {tuple(c) for c in cells.tolist()}


def _all(w):
    return [
        FullReadoutLocalizer(w, sigma=1e-9),
        HPELocalizer(w, sigma=1e-9),
        HPELocalizer(w, sigma=1e-9, decoder="syndrome"),
        ResidualLocalizer(w, sigma=1e-9),
        RothLocalizer(w, sigma=1e-9),
        TwoStageLocalizer(w, sigma=1e-9),
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_k1_noise_free_exact(seed):
    w, reader, truth = _setup(FaultSpec("stuck_random", k=1, rail="mixed", device="pos"), seed)
    assert abs(float(truth.delta_w[0])) > 1e-6
    for loc in _all(w):
        if loc.name == "A2_hpe" and loc.decoder == "syndrome":
            continue  # literal ratio decode is only exact at its own noise-free alpha grid
        est, _ = loc.localize(reader)
        assert _keys(est.cells) == _keys(truth.cells), loc.name
    assert isinstance(_all(w)[0], Localizer)


def test_syndrome_decoder_noise_free():
    w, reader, truth = _setup(FaultSpec("stuck_random", k=1, device="pos"), 5)
    est, _ = HPELocalizer(w, sigma=1e-9, decoder="syndrome").localize(reader)
    assert _keys(est.cells) == _keys(truth.cells)


def test_oracle_zero_reads():
    _, reader, truth = _setup(FaultSpec("stuck_random", k=3, device="pos"))
    est, cost = OracleLocalizer(truth.cells).localize(reader)
    assert _keys(est.cells) == _keys(truth.cells) and cost.mvm_fwd == cost.mvm_bwd == 0
    assert reader.cost.mvm_fwd == 0


def test_permutation_invariance():
    """Permuting rows/cols of W (and the fault) permutes the estimate (noise-free)."""
    tile = synthetic_tile(M, N, spectrum="gaussian", gen=make_generator(3, "tiles"))
    pr, pc = torch.randperm(M), torch.randperm(N)
    spec = FaultSpec("stuck_random", k=2, device="pos")
    outs = []
    for perm in (False, True):
        w = tile.w[pr][:, pc] if perm else tile.w
        t2 = synthetic_tile(M, N, spectrum="gaussian", gen=make_generator(3, "tiles"))
        t2.w = w
        st = encode(t2, ENC)
        bad, truth = inject(st, spec, make_generator(3, "faults"))
        if perm:  # re-place the same physical faults on the permuted matrix
            bad0, tr0 = inject(encode(tile, ENC), spec, make_generator(3, "faults"))
            inv_r, inv_c = pr.argsort(), pc.argsort()
            cells = torch.stack([inv_r[tr0.cells[:, 0]], inv_c[tr0.cells[:, 1]]], 1)
            bad = bad0.clone(g_pos=bad0.g_pos[pr][:, pc])
            truth.cells = cells
        reader = TileReader(lambda b=bad: b, QUIET, make_generator(0, "read"))
        est, _ = TwoStageLocalizer(w, sigma=1e-9).localize(reader)
        outs.append((_keys(est.cells), _keys(truth.cells)))
    assert outs[0][0] == outs[0][1] and outs[1][0] == outs[1][1]


def test_read_counts_exact():
    w, reader, _ = _setup(FaultSpec("stuck_random", k=1, device="pos"))
    _, c = FullReadoutLocalizer(w, sigma=1e-9).localize(reader)
    assert (c.mvm_fwd, c.mvm_bwd) == (N, 0)
    _, c = HPELocalizer(w, sigma=1e-9, n_dense=3, n_bisect=2).localize(reader)
    assert c.mvm_fwd == 3 + 2 * 6 and c.mvm_bwd == 0  # log2(64) = 6 levels
    loc = ResidualLocalizer(w, sigma=1e-9, n_fwd=5, n_bwd=3)
    _, c = loc.localize(reader)
    assert (c.mvm_fwd, c.mvm_bwd) == (5, 3)  # k=1: single line each, no pruning
    loc = RothLocalizer(w, sigma=1e-9, n_dense=2)
    _, c = loc.localize(reader)
    assert c.mvm_fwd == 2 + loc.M.shape[0] and c.mvm_bwd == 0


def test_calibrate_estimates_sigma_and_cost():
    noise = ReadNoise(mode="per_cell", sigma_cell_frac=0.02, adc_bits=None, dac_bits=None)
    w, reader, _ = _setup(FaultSpec("stuck_random", k=2, device="pos"), noise=noise)
    loc = FullReadoutLocalizer(w)
    c = loc.calibrate(reader)
    assert c.mvm_fwd == 8
    assert loc.sigma == pytest.approx(0.02 * float(w.abs().max()), rel=0.15)


def test_kautz_singleton_is_disjunct():
    Mt = kautz_singleton(256, 3).to(DTYPE)
    assert Mt.shape[0] < 256
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        idx = torch.randperm(256, generator=g)[:4]
        defect, other = idx[:3], idx[3]
        covered = Mt[:, defect].sum(1) > 0
        assert not bool(covered[Mt[:, other] > 0].all())  # d=3: no item hidden by 3 others
    pos = torch.zeros(1, Mt.shape[0], dtype=torch.bool)
    pos[0] = Mt[:, [5, 77, 200]].sum(1) > 0
    assert set(comp_dd(pos, Mt.to(torch.uint8), dd=False)[0].nonzero()[:, 0].tolist()) == {
        5,
        77,
        200,
    }


def test_a2b_multiple_bad_columns():
    """Design regime: several bad output lines, <= d faults each, noise free: exact recovery
    of lines (syndrome stage) and cells."""
    w = torch.randn(M, N, dtype=DTYPE) * 0.1
    st = encode(synthetic_tile(M, N, spectrum="gaussian", gen=make_generator(0, "t")), ENC)
    st.g_pos = encode(type("T", (), {"w": w})(), ENC).g_pos
    bad = st.clone()
    cells = torch.tensor([[3, 10], [3, 40], [20, 5], [41, 41], [41, 7], [41, 60], [60, 0]])
    bad.g_pos[cells[:, 0], cells[:, 1]] += 0.5 * (ENC.g_max_uS - ENC.g_min_uS) * 0.8
    reader = TileReader(lambda: bad, QUIET, make_generator(0, "read"))
    loc = RothLocalizer(st.weights(), sigma=1e-9, d_lines=4, d_cells=3)
    loc.w_ref = bad.weights() - (bad.g_pos - st.g_pos) / bad.scale
    est, cost = loc.localize(reader)
    assert set(est.meta["bad_lines"].tolist()) == {3, 20, 41, 60}
    assert _keys(est.cells) == _keys(cells)
    assert cost.mvm_bwd == 0


def test_noisy_k1_a3_and_a4_mostly_succeed():
    noise = ReadNoise(mode="per_cell", sigma_cell_frac=0.01, adc_bits=None, dac_bits=None)
    ok = {"a3": 0, "a4": 0}
    for s in range(20):
        w, reader, truth = _setup(FaultSpec("stuck_random", k=1, device="pos"), s, noise)
        for key, cls in (("a3", ResidualLocalizer), ("a4", TwoStageLocalizer)):
            loc = cls(w, seed=s)
            loc.calibrate(reader)
            est, _ = loc.localize(reader)
            ok[key] += _keys(est.cells) == _keys(truth.cells)
    assert ok["a3"] >= 17 and ok["a4"] >= 17  # fa=0.1 per family: ~10% FP lines


def test_multifault_a3_a4_noise_free():
    w, reader, truth = _setup(FaultSpec("stuck_random", k=5, device="pos"), 7)
    for loc in (ResidualLocalizer(w, sigma=1e-9, n_prune=64), TwoStageLocalizer(w, sigma=1e-9)):
        est, _ = loc.localize(reader)
        assert _keys(est.cells) == _keys(truth.cells), loc.name


# ------------------------------------------------------------------ A5 / A6 / A7


def _visible(w, truth, frac=0.1):
    return truth.cells[truth.delta_w.abs() >= frac * float(w.abs().max())]


def _wave3():
    return [
        ("A5f", lambda w: NonAdaptiveCSLocalizer(w, sigma=1e-9, p_fwd=40)),
        ("A5b", lambda w: NonAdaptiveCSLocalizer(w, sigma=1e-9, p_fwd=20, p_bwd=20)),
        ("A5s", lambda w: NonAdaptiveCSLocalizer(w, sigma=1e-9, p_fwd=48, weight=16)),
        ("A6f", lambda w: GroupTestingLocalizer(w, sigma=1e-9, t_fwd=40, weight=24)),
        ("A6b", lambda w: GroupTestingLocalizer(w, sigma=1e-9, t_fwd=30, t_bwd=30, weight=24)),
        ("A7f", lambda w: AdaptiveBisectionLocalizer(w, sigma=1e-9)),
        ("A7b", lambda w: AdaptiveBisectionLocalizer(w, sigma=1e-9, bidirectional=True)),
    ]


@pytest.mark.parametrize("k", [1, 3])
def test_wave3_noise_free_exact(k):
    for seed in range(3):
        w, reader, truth = _setup(FaultSpec("stuck_random", k=k, rail="mixed", device="pos"), seed)
        vis = _visible(w, truth)
        for name, make in _wave3():
            est, _ = make(w).localize(reader)
            assert _keys(vis) <= _keys(est.cells) <= _keys(truth.cells), (name, k, seed)


def test_wave3_no_fault_empty():
    w, reader, _ = _setup(FaultSpec("stuck_random", k=0, device="pos"), 1)
    for name, make in _wave3():
        est, _ = make(w).localize(reader)
        assert len(est.cells) == 0, name


def test_wave3_read_counts_exact():
    w, reader, _ = _setup(FaultSpec("stuck_random", k=2, device="pos"), 3)
    _, c = NonAdaptiveCSLocalizer(w, sigma=1e-9, p_fwd=7, p_bwd=5).localize(reader)
    assert (c.mvm_fwd, c.mvm_bwd) == (7, 5)
    _, c = NonAdaptiveCSLocalizer(w, sigma=1e-9, p_fwd=9).localize(reader)
    assert (c.mvm_fwd, c.mvm_bwd) == (9, 0)
    _, c = GroupTestingLocalizer(w, sigma=1e-9, t_fwd=11, t_bwd=6).localize(reader)
    assert (c.mvm_fwd, c.mvm_bwd) == (11, 6)
    _, c = GroupTestingLocalizer(w, sigma=1e-9, t_fwd=11).localize(reader)
    assert (c.mvm_fwd, c.mvm_bwd) == (11, 0)


def test_a7_k1_noise_free_reads():
    """k=1, noise free: forward-only = 1 root read + 2 reads per level at worst (left, and right
    only when left is positive); bidirectional = one root read per direction, nothing else."""
    for seed in range(3):
        w, reader, truth = _setup(FaultSpec("stuck_random", k=1, device="pos"), seed)
        if abs(float(truth.delta_w[0])) < 0.1 * float(w.abs().max()):
            continue
        _, cf = AdaptiveBisectionLocalizer(w, sigma=1e-9).localize(reader)
        assert cf.mvm_bwd == 0 and 1 + 6 <= cf.mvm_fwd <= 1 + 12
        _, cb = AdaptiveBisectionLocalizer(w, sigma=1e-9, bidirectional=True).localize(reader)
        assert (cb.mvm_fwd, cb.mvm_bwd) == (1, 1)


def test_a7_forward_only_never_calls_backward():
    w, reader, _ = _setup(FaultSpec("stuck_random", k=3, device="pos"), 4)
    reader.backward_supported = False
    for loc in (
        AdaptiveBisectionLocalizer(w, sigma=1e-9, bidirectional=True),
        NonAdaptiveCSLocalizer(w, sigma=1e-9, p_fwd=20, p_bwd=20),
        GroupTestingLocalizer(w, sigma=1e-9, t_fwd=20, t_bwd=20),
    ):
        _, c = loc.localize(reader)
        assert c.mvm_bwd == 0


def test_wave3_noisy_k2_mostly_succeed():
    noise = ReadNoise(mode="per_cell", sigma_cell_frac=0.01, adc_bits=None, dac_bits=None)
    good = {"a5": 0, "a7": 0}
    for s in range(12):
        w, reader, truth = _setup(FaultSpec("stuck_random", k=2, device="pos"), s, noise)
        vis = _keys(_visible(w, truth))
        for key, loc in (
            ("a5", NonAdaptiveCSLocalizer(w, p_fwd=24, p_bwd=24, seed=s)),
            ("a7", AdaptiveBisectionLocalizer(w, bidirectional=True, seed=s)),
        ):
            loc.calibrate(reader)
            est, _ = loc.localize(reader)
            good[key] += vis <= _keys(est.cells) and len(_keys(est.cells) - _keys(truth.cells)) <= 1
    assert good["a5"] >= 8 and good["a7"] >= 8
