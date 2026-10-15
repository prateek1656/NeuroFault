"""Same seed, same config -> identical results; different seed -> different
results. Catches stray unseeded randomness (e.g. a future `random.random()`
or `numpy.random` call that bypasses the seeded torch generator) and an
accidentally no-op `torch.manual_seed()` - either would silently make "every
reported number traceable to a specific config + seed" (the project's own reproducibility standard) false.
"""

from __future__ import annotations

import pytest
import torch

from neurofault.config import ExperimentConfig
from neurofault.data import load_datasets

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from experiments.run import (
    run_experiment,
)


def _tiny_config(seed: int) -> ExperimentConfig:
    config = ExperimentConfig(
        name=f"repro_check_seed_{seed}",
        simulator="crosssim",
        seed=seed,
        epochs=1,
        num_batches=4,
        train_samples=32,
        test_samples=16,
        batch_size=8,
        eval_interval=1,
    )
    config.crossbar.patch_layer_types = ["Linear"]
    config.fault.density = 0.1
    return config


def test_same_seed_produces_bit_identical_training_results():
    metrics_a = run_experiment(_tiny_config(seed=123))
    metrics_b = run_experiment(_tiny_config(seed=123))

    assert metrics_a["loss"] == metrics_b["loss"]
    assert metrics_a["accuracy"] == metrics_b["accuracy"]
    assert metrics_a["faults"] == metrics_b["faults"]


def test_different_seeds_produce_different_training_results():
    """Sanity check the seed actually does something - catches an
    accidentally no-op torch.manual_seed() call (e.g. one made before an
    import that itself reseeds the global generator)."""
    metrics_a = run_experiment(_tiny_config(seed=1))
    metrics_b = run_experiment(_tiny_config(seed=2))

    assert metrics_a["loss"] != metrics_b["loss"]


def test_load_datasets_shuffle_order_is_reproducible_given_the_same_seed():
    config = ExperimentConfig(
        name="repro_check_dataset", train_samples=32, test_samples=16, batch_size=8
    )

    torch.manual_seed(7)
    trainloader_a, _ = load_datasets(config)
    batch_a = next(iter(trainloader_a))[1]  # labels - order-sensitive, small, easy to compare

    torch.manual_seed(7)
    trainloader_b, _ = load_datasets(config)
    batch_b = next(iter(trainloader_b))[1]

    assert torch.equal(batch_a, batch_b)
