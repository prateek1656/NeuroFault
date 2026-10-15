"""Closed loop X.1: the probe-read fault-map estimator on synthetic layers (no CrossSim)."""

from __future__ import annotations

import pytest
import torch

from neurofault.spectral.closed_loop import ProbeEstimator


def _layer(density: float, seed: int = 0, shape=(300, 600)):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g, dtype=torch.float64) * 0.05
    mag = float(w.abs().max())
    k = int(w.numel() * density)
    idx = torch.randperm(w.numel(), generator=g)[:k]
    bad = w.clone().view(-1)
    bad[idx] = torch.where(torch.rand(k, generator=g) < 0.5, -mag, mag).double()
    truth = torch.zeros(w.numel(), dtype=torch.bool)
    truth[idx] = True
    return w, bad.view_as(w), truth.view_as(w)


@pytest.mark.parametrize("kind", ["adaptive", "readout"])
@pytest.mark.parametrize("density", [0.0005, 0.0126])
def test_estimator_finds_stuck_cells(kind, density):
    w, bad, truth = _layer(density)
    est = ProbeEstimator(kind=kind, seed=3, shadow=False)("L", lambda: bad, w, 0)
    assert int((est.mask & truth).sum()) >= 0.95 * int(truth.sum())
    assert int((est.mask & ~truth).sum()) <= 3
    # signed deviation estimate close to the true one on found cells
    found = est.mask & truth
    assert (est.deviation - (bad - w))[found].abs().max() < 0.1 * float(w.abs().max()) * 2


def test_adaptive_is_cheap_when_sparse_and_falls_back_when_dense():
    w, bad, _ = _layer(0.0005)
    sparse = ProbeEstimator(kind="adaptive", seed=3)("L", lambda: bad, w, 0)
    assert sparse.fallback_tiles == 0
    assert sparse.reads * 4 <= sparse.shadow["reads"]  # shadow = full readout
    w, bad, _ = _layer(0.0126, shape=(512, 512))  # full tiles only (an edge tile can be sparse)
    dense = ProbeEstimator(kind="adaptive", seed=3, shadow=False)("L", lambda: bad, w, 0)
    assert dense.fallback_tiles == dense.tiles


def test_estimator_is_deterministic_and_only_reads_through_provider():
    w, bad, _ = _layer(0.001)
    calls = []

    def provider():
        calls.append(1)
        return bad

    a = ProbeEstimator(kind="adaptive", seed=5)("L", provider, w, 2)
    b = ProbeEstimator(kind="adaptive", seed=5)("L", lambda: bad, w, 2)
    assert len(calls) == 1
    assert torch.equal(a.mask, b.mask) and a.reads == b.reads


def test_no_faults_no_beliefs():
    w, _, _ = _layer(0.0)
    est = ProbeEstimator(kind="adaptive", seed=1, shadow=False)("L", lambda: w.clone(), w, 0)
    assert int(est.mask.sum()) <= 1
