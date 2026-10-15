"""Constructed-case tests for spectral metrics and cost."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurofault.spectral import metrics as M
from neurofault.spectral.cost import CostModel
from neurofault.spectral.types import FaultTruth, ReadCost


def test_auc_perfect_chance_and_ties():
    assert M.auc([2, 3], [0, 1]) == 1.0
    assert M.auc([0, 1], [2, 3]) == 0.0
    assert M.auc([1, 1], [1, 1]) == 0.5
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2000, generator=g), torch.randn(2000, generator=g)
    assert abs(M.auc(a, b) - 0.5) < 0.03


def test_detection_at_fpr_is_strict_and_conservative():
    neg = torch.arange(100.0)
    pos = torch.tensor([50.0, 98.5, 150.0, 200.0])
    # threshold = 6th largest negative (94): exactly 5 of 100 negatives exceed it
    assert M.detection_at_fpr(pos, neg, 0.05) == pytest.approx(0.75)
    # all-tied statistic must not be credited
    assert M.detection_at_fpr([1.0] * 5, [1.0] * 100, 0.01) == 0.0


def test_rank_energy_oracle_exact_for_low_rank():
    g = torch.Generator().manual_seed(1)
    E = torch.randn(20, 3, generator=g, dtype=torch.float64) @ torch.randn(
        3, 16, generator=g, dtype=torch.float64
    )
    re = M.rank_energy(E, [1, 3, 5])
    assert re[3] == pytest.approx(1.0) and re[5] == pytest.approx(1.0)
    assert 0.0 < re[1] < 1.0


def test_rank_energy_data_weighting_changes_the_answer():
    # E has equal energy in two input directions; C only excites the first one
    E = torch.diag(torch.tensor([1.0, 1.0], dtype=torch.float64))
    Cs = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    assert M.rank_energy(E, [1], C_sqrt=Cs, basis=basis)[1] == pytest.approx(1.0)
    assert M.rank_energy(E, [1], basis=basis)[1] == pytest.approx(0.5)
    assert M.rank_energy(E, [1], C_sqrt=Cs)[1] == pytest.approx(1.0)  # oracle weighted


def test_rank_energy_random_subspace_is_a_null():
    g = torch.Generator().manual_seed(2)
    E = torch.randn(64, 64, generator=g, dtype=torch.float64)
    B, _ = torch.linalg.qr(torch.randn(64, 8, generator=g, dtype=torch.float64))
    assert M.rank_energy(E, [8], basis=B)[8] == pytest.approx(8 / 64, abs=0.06)


def test_rel_output_error():
    W = torch.eye(3, dtype=torch.float64)
    X = torch.eye(3, dtype=torch.float64)
    assert M.rel_output_error(W, W, X) == 0.0
    assert M.rel_output_error(1.1 * W, W, X) == pytest.approx(0.1)


def test_subspace_angle():
    U = torch.tensor([[1.0, 0], [0, 1.0], [0, 0]], dtype=torch.float64)
    assert M.subspace_angle(U, U) == pytest.approx(0.0, abs=1e-7)
    V = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float64)
    assert M.subspace_angle(U, V) == pytest.approx(np.pi / 2)
    c, s = np.cos(0.3), np.sin(0.3)
    W = torch.tensor([[c], [0.0], [s]], dtype=torch.float64)
    assert M.subspace_angle(U[:, :1], W) == pytest.approx(0.3)


def test_localization_f1_and_row_accuracy():
    truth = torch.tensor([[1, 2], [3, 4]])
    assert M.localization_f1(truth, truth) == 1.0
    assert M.localization_f1(torch.tensor([[1, 2]]), truth) == pytest.approx(2 / 3)
    assert M.localization_f1(torch.empty(0, 2, dtype=torch.long), truth) == 0.0
    assert M.localization_f1(torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2)) == 1.0
    ft = FaultTruth(
        cells=truth,
        delta_w=torch.zeros(2),
        rows=torch.tensor([1, 3]),
        cols=torch.tensor([2, 4]),
    )
    assert M.row_accuracy(torch.tensor([1, 3]), ft) == 1.0
    assert M.row_accuracy(torch.tensor([1]), ft) == pytest.approx(0.5)
    assert M.col_accuracy(torch.tensor([9]), ft) == 0.0


def test_bootstrap_cis_are_seeded_and_paired():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a - 1.0
    lo, hi, pt = M.paired_bootstrap_ci(a, b, seed=3)
    assert lo == hi == pt == 1.0  # constant paired difference: no variance
    unpaired_lo, _, _ = M.bootstrap_ci(a, seed=3)
    assert unpaired_lo < 3.0
    assert M.bootstrap_ci(a, seed=3) == M.bootstrap_ci(a, seed=3)
    lo, hi, pt = M.paired_ratio_ci([2, 4, 6], [10, 10, 10], seed=0)
    assert pt == pytest.approx(0.4) and lo <= pt <= hi
    with pytest.raises(ValueError):
        M.paired_bootstrap_ci([1, 2], [1])


def test_cost_model_arithmetic_and_assumed_ledger():
    cm = CostModel()
    c = ReadCost(mvm_fwd=2, mvm_bwd=1, adc=10, cell_reads=512, writes=1, images=0)
    e = (3 * cm.e_mvm_pj + 10 * cm.e_adc_pj + 512 * cm.e_cell_read_pj + cm.e_write_pj) * 1e-12
    assert cm.energy(c) == pytest.approx(e)
    lat = 3 * cm.t_mvm_s + 512 / cm.read_parallelism * cm.t_cell_read_s + cm.t_write_s
    assert cm.latency(c) == pytest.approx(lat)
    assert cm.energy(c + c) == pytest.approx(2 * cm.energy(c))
    assert cm.edp(c) == pytest.approx(cm.energy(c) * cm.latency(c))
    recs = cm.param_records()
    assert recs and all(r.status == "assumed" and not r.verified for r in recs.values())
