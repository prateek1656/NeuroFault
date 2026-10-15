"""End-to-end test through the real AIHWKit pipeline.

Skips cleanly (not a failure) if AIHWKit isn't installed - install via
`uv sync --extra aihwkit`. Uses TorchInferenceRPUConfig (see
crossbar/backends/aihwkit_backend.py's module docstring for why: AIHWKit's
default tile is broken upstream, this one is the pure-PyTorch tile that
actually works, verified during development).
"""

import pytest

pytest.importorskip("aihwkit", reason="AIHWKit not installed (uv sync --extra aihwkit)")

import torch

from neurofault.config import ExperimentConfig
from neurofault.models.cnn import build_model
from neurofault.system import FaultTolerantNeuromorphic


def test_tiny_end_to_end_run_does_not_raise():
    config = ExperimentConfig(
        model_arch="tiny_cnn",
        name="integration_smoke_aihwkit",
        simulator="aihwkit",
        epochs=1,
        num_batches=2,
        train_samples=16,
        test_samples=8,
        batch_size=4,
    )
    config.crossbar.patch_layer_types = ["Linear"]  # keep it minimal for speed
    config.device.name = "pcm"  # DeviceConfig's default "vteam" is meaningless here

    system = FaultTolerantNeuromorphic(build_model(config), config)
    assert len(system.handles) > 0  # actually built real crossbar handles

    x = torch.randn(4, 3, 32, 32)
    output = system.forward(x)

    assert output.shape == (4, 10)

    injected = system.inject_faults()
    assert injected > 0  # real faults actually landed on a real analog tile

    system.check_health()  # must not raise


def test_fault_injection_actually_changes_model_output():
    """Fault injection must reach the real forward-pass compute path, not
    just bookkeeping - the whole point of this backend."""
    config = ExperimentConfig(
        model_arch="tiny_cnn", name="integration_smoke_aihwkit_fault_effect", simulator="aihwkit"
    )
    config.crossbar.patch_layer_types = ["Linear"]
    config.device.name = "pcm"
    config.fault.density = 0.5  # high density so the effect is unmissable

    system = FaultTolerantNeuromorphic(build_model(config), config)
    x = torch.randn(4, 3, 32, 32)

    output_before = system.forward(x).detach().clone()
    system.inject_faults()
    output_after = system.forward(x).detach().clone()

    assert not torch.allclose(output_before, output_after)
