import itertools

import pytest

from neurofault.config import MitigationConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.crossbar.self_healing import RedundancyPool
from neurofault.mitigation.dispatch import apply_runtime_mitigation
from tests.unit.fakes import make_handle


@pytest.mark.parametrize(
    "enable_soft,enable_remap,enable_reset",
    list(itertools.product([False, True], repeat=3)),
)
def test_dispatch_never_calls_a_mitigation_whose_flag_is_off(
    enable_soft, enable_remap, enable_reset
):
    """Direct regression test for bug #2: the original's final `else`
    branch called reset_critical_layer unconditionally regardless of
    enable_layer_reset. Here, for every combination of flags, the
    corresponding action must fire iff its flag is True - especially at
    critical (very low) health, where the original bug lived."""
    handle = make_handle(shape=(4, 4))
    monitor = HealthMonitor(handle)
    pool = RedundancyPool(handle, redundancy_factor=0.5)

    reset_called = []

    def reset_fn():
        reset_called.append(True)
        return True

    config = MitigationConfig(
        enable_soft_mitigation=enable_soft,
        enable_remapping=enable_remap,
        enable_layer_reset=enable_reset,
        health_threshold=75.0,
    )

    # Exercise the "critical health" branch specifically - this is exactly
    # where the original unconditional reset call lived.
    outcome = apply_runtime_mitigation(
        handle,
        pool,
        monitor,
        critical_devices=[(0, 0)],
        avg_health=10.0,  # well below health_threshold/2 = 37.5
        mitigation_config=config,
        reset_fn=reset_fn,
    )

    assert outcome.layer_reset == enable_reset
    assert bool(reset_called) == enable_reset
    # Soft/remap must never fire in the critical-health branch regardless of their flags.
    assert outcome.soft_mitigated == 0
    assert outcome.remapped == 0


def test_dispatch_no_mitigations_config_does_nothing_at_critical_health():
    """The exact 'No Mitigations' experiment-arm scenario that the original
    bug silently broke: everything off, critical health, must do nothing."""
    handle = make_handle(shape=(4, 4))
    monitor = HealthMonitor(handle)
    pool = RedundancyPool(handle, redundancy_factor=0.5)
    config = MitigationConfig(
        enable_soft_mitigation=False,
        enable_remapping=False,
        enable_layer_reset=False,
    )

    outcome = apply_runtime_mitigation(
        handle,
        pool,
        monitor,
        critical_devices=[(0, 0)],
        avg_health=5.0,
        mitigation_config=config,
        reset_fn=lambda: True,
    )

    assert outcome.actions_taken == []
    assert outcome.layer_reset is False


def test_dispatch_empty_critical_devices_is_a_noop():
    handle = make_handle(shape=(4, 4))
    monitor = HealthMonitor(handle)
    pool = RedundancyPool(handle, redundancy_factor=0.5)
    config = MitigationConfig(
        enable_soft_mitigation=True, enable_remapping=True, enable_layer_reset=True
    )

    outcome = apply_runtime_mitigation(
        handle,
        pool,
        monitor,
        critical_devices=[],
        avg_health=5.0,
        mitigation_config=config,
        reset_fn=lambda: True,
    )

    assert outcome.actions_taken == []
