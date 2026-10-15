"""Stuck-at fault persistence across synchronize(): a stuck cell's conductance
must survive being re-derived from .weight, in EVERY mode - not just
fault-aware retraining.

This used to be a fault-aware-retraining-only mechanism
(_capture_fixed_defects()/reapply_stuck_faults()), tested only for that one
mode. A real bug found and fixed during development: stuck-at faults silently
self-healed within one batch in every OTHER mode, because nothing reapplied
them there - verified directly (86/86 stuck cells reverted after one real
optimizer.step()+synchronize() outside the retraining path). The mechanism is
now general (system.inject_faults() captures pins, system.synchronize()
reapplies them) and runs for every mode uniformly - these tests exercise it
through that real public API, not a retraining-specific one, so the same
assertions cover every mode structurally rather than by remembering to add
more mode-specific tests later.

Uses FakeCrossbar/make_handle directly with HealthMonitor rather than the
full FaultTolerantNeuromorphic's real backends, since the mechanism only
touches HealthMonitor.stuck_mask and the accessor - no backend required.
"""

from __future__ import annotations

import pytest
import torch

from neurofault.config import CrossbarConfig, ExperimentConfig
from neurofault.models.cnn import build_model
from neurofault.system import FaultTolerantNeuromorphic
from tests.unit.fakes import make_handle


def _system_with_fake_backend(monkeypatch, config: ExperimentConfig):
    """FaultTolerantNeuromorphic always builds real backend handles via
    _backend_module() - for a pure system.py unit test we don't want a real
    simulator installed, so swap in a fake backend module exposing the same
    patch_layers/build_handles/synchronize surface as crosssim_backend.py."""
    import neurofault.system as system_module

    handle = make_handle(shape=(4, 6))

    class _FakeBackend:
        @staticmethod
        def patch_layers(model, crossbar_config, device_cls, device_params, calibrated_ranges=None):
            return model

        @staticmethod
        def build_handles(model, device_params, crossbar_config):
            return {handle.name: handle}

        @staticmethod
        def synchronize(model):
            pass

    monkeypatch.setattr(system_module, "_backend_module", lambda simulator: _FakeBackend)
    system = FaultTolerantNeuromorphic(build_model(config), config)
    return system, handle


def test_stuck_cells_survive_synchronize_after_injection(monkeypatch):
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_stuck_persistence")
    system, handle = _system_with_fake_backend(monkeypatch, config)
    monitor = next(iter(system.monitors.values()))

    monitor.stuck_mask[0, 0] = True
    monitor.stuck_mask[2, 3] = True
    system._update_stuck_pins(handle.name, handle)
    fixed_values = handle.accessor.read()[monitor.stuck_mask].clone()

    overwritten = torch.randn(handle.accessor.shape)
    handle.accessor.write(overwritten)
    system.synchronize()  # the real per-batch call, not a hand-rolled reapply

    result = handle.accessor.read()
    assert torch.equal(result[monitor.stuck_mask], fixed_values)
    non_stuck = ~monitor.stuck_mask
    assert torch.equal(result[non_stuck], overwritten[non_stuck])


def test_stuck_pins_accumulate_across_multiple_injections(monkeypatch):
    """Periodic reinjection (every non-retraining mode) must keep pinning
    newly-stuck cells on top of previously-stuck ones, not just the latest
    batch - this is what distinguishes the general mechanism from the old
    once-only _capture_fixed_defects()."""
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_stuck_accumulate")
    system, handle = _system_with_fake_backend(monkeypatch, config)
    monitor = next(iter(system.monitors.values()))

    monitor.stuck_mask[0, 0] = True
    system._update_stuck_pins(handle.name, handle)
    first_fixed = handle.accessor.read()[0, 0].item()

    # A real synchronize() cycle in between - [0,0]'s pin must survive an
    # arbitrary overwrite, exactly like it would across many real training
    # batches before the next reinjection cycle.
    handle.accessor.write(torch.randn(handle.accessor.shape))
    system.synchronize()
    assert handle.accessor.read()[0, 0].item() == first_fixed

    # A second, later injection cycle marks a different cell stuck too. By
    # this point [0,0]'s value in the matrix is already correctly pinned
    # (from the synchronize() above), so re-capturing it is idempotent -
    # exactly what happens in the real training loop between reinjections.
    monitor.stuck_mask[2, 3] = True
    matrix = handle.accessor.read()
    matrix[2, 3] = 999.0  # what the fault injector would have just written
    handle.accessor.write(matrix)
    system._update_stuck_pins(handle.name, handle)

    overwritten = torch.randn(handle.accessor.shape)
    handle.accessor.write(overwritten)
    system.synchronize()

    result = handle.accessor.read()
    assert result[0, 0].item() == first_fixed
    assert result[2, 3].item() == 999.0


def test_synchronize_is_a_safe_noop_with_no_stuck_cells(monkeypatch):
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_no_faults")
    system, handle = _system_with_fake_backend(monkeypatch, config)

    overwritten = torch.randn(handle.accessor.shape)
    handle.accessor.write(overwritten)
    system.synchronize()  # must not raise, must not change anything

    assert torch.equal(handle.accessor.read(), overwritten)


def test_synchronize_without_injection_leaves_stuck_mask_unpinned(monkeypatch):
    """If stuck_mask is set directly (bypassing inject_faults(), so
    _update_stuck_pins() never ran), synchronize() must not crash or invent
    values - pinning only applies to cells it was actually told about."""
    config = ExperimentConfig(model_arch="tiny_cnn", name="test_uncaptured")
    system, handle = _system_with_fake_backend(monkeypatch, config)
    monitor = next(iter(system.monitors.values()))
    monitor.stuck_mask[0, 0] = True

    overwritten = torch.randn(handle.accessor.shape)
    handle.accessor.write(overwritten)
    system.synchronize()

    assert torch.equal(handle.accessor.read(), overwritten)


def test_calibrate_dac_raises_on_non_crosssim_backend(monkeypatch):
    """See CrossbarConfig.calibrate_dac's docstring - the fix
    (CrossSim's native input-percentile-scaling) is CrossSim-only; a
    non-CrossSim backend must raise rather than silently no-op the flag,
    same discipline as the existing calibrate_adc guard."""
    config = ExperimentConfig(
        model_arch="tiny_cnn",
        name="test_calibrate_dac_wrong_backend",
        simulator="aihwkit",
        crossbar=CrossbarConfig(calibrate_dac=True),
    )

    with pytest.raises(ValueError, match="calibrate_dac=True"):
        _system_with_fake_backend(monkeypatch, config)
