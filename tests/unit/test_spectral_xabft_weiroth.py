"""E4 X-ABFT and E5 Wei-Roth-on-their-terms localizers: exactness, cost accounting, noise model."""

from __future__ import annotations

import math

import pytest
import torch

from neurofault.spectral.localize import BoundedNoiseReader, WeiRothPaperLocalizer, XABFTLocalizer
from neurofault.spectral.localize.weiroth import line_f1
from neurofault.spectral.types import DTYPE, ReadCost, make_generator

M = N = 64


class FakeReader:
    """Noiseless counted reader over an explicit matrix (forward only)."""

    backward_supported = False

    def __init__(self, w):
        self.w, self.cost = w, ReadCost()

    @property
    def shape(self):
        return tuple(self.w.shape)

    def mvm(self, x):
        self.cost.mvm_fwd += x.shape[1]
        return self.w @ x.to(DTYPE)


def _tile(seed=0):
    return torch.randn(M, N, generator=make_generator(seed, "xw"), dtype=DTYPE)


def _faulty(w, cells, delta=0.7):
    bad = w.clone()
    for i, j in cells:
        bad[i, j] += delta
    return bad


@pytest.mark.parametrize("f", ["linear", "exp"])
@pytest.mark.parametrize("cell", [(0, 0), (5, 13), (63, 63), (17, 62)])
def test_xabft_single_fault_exact(f, cell):
    w = _tile()
    loc = XABFTLocalizer(w, sigma=1e-9, f=f)
    est, cost = loc.localize(FakeReader(_faulty(w, [cell])))
    assert est.cells.tolist() == [list(cell)]
    assert cost.mvm_fwd == 4 * math.ceil(N / 4) == loc.n_test_vectors


def test_xabft_two_faults_in_one_block_exact():
    w = _tile(1)
    cells = [(2, 5), (9, 6)]  # same 4x16 block (inputs 4-7, outputs 0-15), different input rows
    loc = XABFTLocalizer(w, sigma=1e-9)
    est, _ = loc.localize(FakeReader(_faulty(w, cells, 0.4)))
    assert {tuple(c) for c in est.cells.tolist()} == set(cells)


def test_xabft_faults_in_different_blocks_and_no_fault():
    w = _tile(2)
    loc = XABFTLocalizer(w, sigma=1e-9, r_t=8, c_t=8, M=2)
    est, cost = loc.localize(FakeReader(_faulty(w, [(1, 1), (60, 40), (33, 9)])))
    assert {tuple(c) for c in est.cells.tolist()} == {(1, 1), (60, 40), (33, 9)}
    assert cost.mvm_fwd == 2 * 8
    est0, _ = loc.localize(FakeReader(w))
    assert est0.cells.shape == (0, 2)


def test_xabft_undecodable_cases_do_not_crash():
    w = _tile(3)
    # M=2 only guarantees DETECTION of 2 faults (paper Thm 1): flagged, but not correctly decoded
    est, _ = XABFTLocalizer(w, sigma=1e-9, M=2).localize(FakeReader(_faulty(w, [(0, 0), (3, 1)])))
    assert est.meta["n_flagged"] == 1 and {(0, 0), (3, 1)} != {tuple(c) for c in est.cells.tolist()}
    est, _ = XABFTLocalizer(w, sigma=1e-9).localize(FakeReader(_faulty(w, [(0, 0), (3, 0)])))
    assert est.meta["n_flagged"] == 1


def test_xabft_native_cost_meta():
    est, _ = XABFTLocalizer(_tile(), sigma=1e-9).localize(FakeReader(_tile()))
    assert est.meta["n_test_vectors"] == 64
    assert est.meta["time_redundancy"] == pytest.approx(1 + 64 / 10_000)
    assert est.meta["hardware_redundancy"] == pytest.approx(1 + 5 / 16)


def test_bounded_noise_reader_bound_rms_and_cost():
    w = _tile()
    gen = make_generator(0, "bn")
    for dist, bound in (("sign", 1.0), ("uniform", math.sqrt(3.0))):
        rd = BoundedNoiseReader(FakeReader(w), 0.01, gen, dist)
        x = torch.randn(N, 50, dtype=DTYPE)
        e = rd.mvm(x) - w @ x
        scale = 0.01 * x.norm(dim=0)
        assert (e.abs() <= bound * scale + 1e-12).all()
        assert float((e / scale).pow(2).mean().sqrt()) == pytest.approx(0.01 / 0.01, rel=0.1)
        assert rd.cost.mvm_fwd == 50 and rd.bound_unit == pytest.approx(0.01 * bound)


def test_weiroth_paper_line_level_guarantee_and_cost():
    w = _tile(4)
    delta = 1e-4
    loc = WeiRothPaperLocalizer(w, bound_unit=delta, d_lines=1, seed=1)
    big = 2 * float(loc.H.sum(1).max()) * delta * math.sqrt(N) * 1.5 + 1.0  # >> 2 rho delta_x
    bad = _faulty(w, [(11, 20)], big)
    rd = BoundedNoiseReader(FakeReader(bad), delta, make_generator(0, "bn"), "sign")
    est, cost = loc.localize(rd)
    assert est.rows.tolist() == [11] and est.cells.shape == (0, 2)  # line level only
    assert cost.mvm_fwd == 1 and est.meta["r_cols"] == loc.H.shape[0]
    assert line_f1(est.rows, torch.tensor([[11, 20]])) == 1.0


def test_weiroth_cell_extension_localizes_cell_and_charges_reads():
    w = _tile(5)
    loc = WeiRothPaperLocalizer(w, bound_unit=1e-5, d_lines=1, cell_stage="ks", seed=2)
    est, cost = loc.localize(FakeReader(_faulty(w, [(30, 7)], 2.0)))
    assert est.cells.tolist() == [[30, 7]]
    assert cost.mvm_fwd == 1 + loc.M.shape[0]


def test_weiroth_misses_outlier_below_two_rho_delta():
    w = _tile(6)
    loc = WeiRothPaperLocalizer(w, bound_unit=0.05, d_lines=1, seed=3)
    est, _ = loc.localize(FakeReader(_faulty(w, [(3, 3)], 0.1)))  # far below 2 rho delta_x
    assert est.rows.numel() == 0


def test_line_f1_edges():
    assert line_f1(torch.zeros(0, dtype=torch.long), torch.zeros(0, 2, dtype=torch.long)) == 1.0
    assert line_f1(torch.tensor([1]), torch.zeros(0, 2, dtype=torch.long)) == 0.0
    assert line_f1(torch.tensor([1, 2]), torch.tensor([[1, 0], [1, 5]])) == pytest.approx(2 / 3)


def test_weiroth_rss_threshold_is_tighter_than_worst_and_comp_fallback():
    n = 256  # row weight rho = 52: worst threshold 52 dx, rss threshold ~22 dx
    w = torch.randn(n, n, generator=make_generator(7, "xw"), dtype=DTYPE)
    delta, d = 0.01, 4
    bad = _faulty(w, [(40, 9)], 45 * delta * math.sqrt(n))  # 45 dx: between the two thresholds
    seen = {}
    for thr in ("worst", "rss"):
        loc = WeiRothPaperLocalizer(w, bound_unit=delta, d_lines=1, threshold=thr, seed=4)
        rd = BoundedNoiseReader(FakeReader(bad), delta, make_generator(1, "bn"), "sign")
        seen[thr] = loc.localize(rd)[0].rows.tolist()
    assert seen["worst"] == [] and seen["rss"] == [40]
    loc = WeiRothPaperLocalizer(
        w, bound_unit=delta, d_lines=d, seed=4
    )  # high d: guarantee form lost
    est, _ = loc.localize(FakeReader(_faulty(w, [(3, 3)], 50.0)))
    assert est.meta["paper_guarantee_form"] is False and 3 in est.rows.tolist()
