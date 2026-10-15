"""Canary split (retraining/canary.py) and FisherTracker (retraining/saliency.py)."""

from __future__ import annotations

import pytest
import torch

from neurofault.config import ExperimentConfig, RetrainingConfig
from neurofault.retraining.canary import build_canary, split_indices
from neurofault.retraining.saliency import FisherTracker


def _config(seed=3, canary_size=16) -> ExperimentConfig:
    return ExperimentConfig(
        name="canary_test",
        seed=seed,
        train_samples=64,
        test_samples=8,
        batch_size=8,
        retraining=RetrainingConfig(canary_size=canary_size),
    )


def test_split_is_deterministic_disjoint_and_complete():
    a_canary, a_burst = split_indices(64, 16, seed=3)
    b_canary, b_burst = split_indices(64, 16, seed=3)
    assert (a_canary, a_burst) == (b_canary, b_burst)
    assert len(a_canary) == 16
    assert set(a_canary).isdisjoint(a_burst)
    assert sorted(a_canary + a_burst) == list(range(64))
    other, _ = split_indices(64, 16, seed=4)
    assert other != a_canary


def test_split_rejects_canary_that_leaves_no_training_data():
    with pytest.raises(ValueError, match="must be < the train pool size"):
        split_indices(16, 16, seed=0)


def test_build_canary_does_not_touch_global_rng():
    torch.manual_seed(1234)
    expected = torch.rand(5)
    torch.manual_seed(1234)
    split = build_canary(_config())
    assert torch.equal(torch.rand(5), expected)
    assert len(split.canary_indices) == 16 and len(split.burst_indices) == 48


def test_canary_loader_is_clean_deterministic_and_burst_loader_excludes_canary():
    split = build_canary(_config())
    first = [x.clone() for x, _ in split.canary_loader]
    second = [x.clone() for x, _ in split.canary_loader]
    assert all(torch.equal(a, b) for a, b in zip(first, second))  # no augmentation randomness
    assert sum(x.size(0) for x in first) == 16
    # Burst loader covers exactly the burst indices, none of the canary's.
    assert set(split.burst_loader.dataset.indices) == set(split.burst_indices)
    assert set(split.burst_loader.dataset.indices).isdisjoint(split.canary_indices)
    # Canary images == the same train images through the clean transform (labels match too).
    full = split.canary_loader.dataset.dataset
    assert [full.targets[i] for i in split.canary_indices] == torch.cat(
        [y for _, y in split.canary_loader]
    ).tolist()


def test_fisher_tracker_ema_and_state_roundtrip():
    lin = torch.nn.Linear(3, 2, bias=False)
    tracker = FisherTracker({"lin_crossbar_0": lin.weight}, decay=0.5)

    lin.weight.grad = torch.full((2, 3), 2.0)
    tracker.update()  # first update seeds F = g^2
    assert torch.allclose(tracker.fisher["lin_crossbar_0"], torch.full((2, 3), 4.0))
    lin.weight.grad = torch.zeros(2, 3)
    tracker.update()  # 0.5*4 + 0.5*0
    assert torch.allclose(tracker.fisher["lin_crossbar_0"], torch.full((2, 3), 2.0))
    assert tracker.num_updates == 2

    state = tracker.state_dict()
    restored = FisherTracker({"lin_crossbar_0": lin.weight}, decay=0.9)
    restored.load_state_dict(state)
    assert torch.equal(restored.fisher["lin_crossbar_0"], tracker.fisher["lin_crossbar_0"])
    assert restored.decay == 0.5 and restored.num_updates == 2

    mismatched = FisherTracker({"other": lin.weight})
    with pytest.raises(ValueError, match="does not match"):
        mismatched.load_state_dict(state)
