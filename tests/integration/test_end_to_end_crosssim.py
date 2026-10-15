"""End-to-end test through the real CrossSim pipeline.

Skips cleanly (not a failure) if CrossSim isn't installed - install via
`uv sync --extra crosssim`. CrossSim is confirmed to install and run cleanly
on CPU with no GPU/CUDA required - pure Python/NumPy/SciPy, no C++ extension.
"""

import pytest

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

import torch

from neurofault.config import ExperimentConfig
from neurofault.models.cnn import build_model
from neurofault.system import FaultTolerantNeuromorphic


def test_tiny_end_to_end_run_does_not_raise():
    config = ExperimentConfig(
        model_arch="tiny_cnn",
        name="integration_smoke_crosssim",
        simulator="crosssim",
        epochs=1,
        num_batches=2,
        train_samples=16,
        test_samples=8,
        batch_size=4,
    )
    config.crossbar.patch_layer_types = ["Linear"]  # keep it minimal for speed

    system = FaultTolerantNeuromorphic(build_model(config), config)
    assert len(system.handles) > 0  # actually built real crossbar handles

    x = torch.randn(4, 3, 32, 32)
    output = system.forward(x)

    assert output.shape == (4, 10)

    injected = system.inject_faults()
    assert injected > 0  # real faults actually landed on a real analog matrix

    system.check_health()  # must not raise


def test_fault_injection_actually_changes_model_output():
    """Fault injection must reach the real forward-pass compute path, not
    just bookkeeping - the whole point of this backend."""
    config = ExperimentConfig(
        model_arch="tiny_cnn", name="integration_smoke_crosssim_fault_effect", simulator="crosssim"
    )
    config.crossbar.patch_layer_types = ["Linear"]
    config.fault.density = 0.5  # high density so the effect is unmissable

    system = FaultTolerantNeuromorphic(build_model(config), config)
    x = torch.randn(4, 3, 32, 32)

    output_before = system.forward(x).detach().clone()
    system.inject_faults()
    output_after = system.forward(x).detach().clone()

    assert not torch.allclose(output_before, output_after)


def test_training_reaches_the_analog_core_via_synchronize():
    """Regression test for the real bug found during development: without
    system.synchronize() after optimizer.step(), training silently never
    updates CrossSim's analog compute path. Exercises the exact pattern
    experiments/run.py uses."""
    config = ExperimentConfig(
        model_arch="tiny_cnn", name="integration_smoke_crosssim_training", simulator="crosssim"
    )
    config.crossbar.patch_layer_types = ["Linear"]

    system = FaultTolerantNeuromorphic(build_model(config), config)
    handle = next(iter(system.handles.values()))
    matrix_before = handle.accessor.read().clone()

    optimizer = torch.optim.Adam(system.model.parameters(), lr=0.1)
    x = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(system.forward(x), targets)
        loss.backward()
        optimizer.step()
        system.synchronize()

    assert not torch.allclose(matrix_before, handle.accessor.read())


def test_devices_per_synapse_end_to_end_construction_faults_and_training():
    """Multi-device-per-synapse/bit-slicing, exercised through the
    full system, not just the backend module directly."""
    config = ExperimentConfig(
        model_arch="tiny_cnn",
        name="integration_smoke_crosssim_devices_per_synapse",
        simulator="crosssim",
    )
    config.crossbar.patch_layer_types = ["Linear"]
    config.crossbar.devices_per_synapse = 2

    system = FaultTolerantNeuromorphic(build_model(config), config)
    assert len(system.handles) > 0

    x = torch.randn(4, 3, 32, 32)
    output_before = system.forward(x).detach().clone()

    injected = system.inject_faults()
    assert injected > 0
    output_after = system.forward(x).detach().clone()
    assert not torch.allclose(output_before, output_after)

    handle = next(iter(system.handles.values()))
    matrix_before_training = handle.accessor.read().clone()

    optimizer = torch.optim.Adam(system.model.parameters(), lr=0.1)
    targets = torch.randint(0, 10, (4,))
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(system.forward(x), targets)
        loss.backward()
        optimizer.step()
        system.synchronize()

    assert not torch.allclose(matrix_before_training, handle.accessor.read())


def test_fault_aware_retraining_pins_stuck_cells_while_others_keep_learning():
    """The three claims this whole feature rests on, verified together:
    (1) no periodic re-injection happens (stuck cell count stays fixed after
    the one-time injection), (2) those stuck cells' values never drift from
    what they were pinned to at injection time despite repeated
    synchronize() calls that push the full updated weight into the analog
    core, and (3) non-stuck cells still change - real learning happens
    around the fixed defects, not a frozen crossbar."""
    config = ExperimentConfig(
        model_arch="tiny_cnn",
        name="integration_smoke_crosssim_fault_aware_retraining",
        simulator="crosssim",
    )
    config.crossbar.patch_layer_types = ["Linear"]
    config.fault.density = 0.3
    config.retraining.enable_fault_aware = True

    system = FaultTolerantNeuromorphic(build_model(config), config)
    handle = next(iter(system.handles.values()))
    monitor = next(iter(system.monitors.values()))

    system.inject_faults()  # pins stuck cells internally now - see system.py
    stuck_count_after_injection = int(monitor.stuck_mask.sum().item())
    assert stuck_count_after_injection > 0
    fixed_values_at_injection = handle.accessor.read()[monitor.stuck_mask].clone()

    optimizer = torch.optim.Adam(system.model.parameters(), lr=0.1)
    x = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    criterion = torch.nn.CrossEntropyLoss()

    matrix_before_training = handle.accessor.read().clone()
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(system.forward(x), targets)
        loss.backward()
        optimizer.step()
        system.synchronize()  # reapplies the pin internally now - see system.py

    matrix_after_training = handle.accessor.read()

    assert int(monitor.stuck_mask.sum().item()) == stuck_count_after_injection
    # allclose, not equal: CrossSim's get_matrix()/set_matrix() round-trip
    # introduces ~1e-8 float32 rounding noise even for values written back
    # unchanged (verified directly) - unrelated to the (disabled) read_noise/
    # programming_error nonideality models, just float roundoff through the
    # core's internal representation.
    assert torch.allclose(
        matrix_after_training[monitor.stuck_mask], fixed_values_at_injection, atol=1e-6
    )
    non_stuck = ~monitor.stuck_mask
    assert not torch.allclose(matrix_before_training[non_stuck], matrix_after_training[non_stuck])
